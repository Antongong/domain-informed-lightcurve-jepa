#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
import multiprocessing as mp

import light_curve

# Install FATS package NOT via pip, but rather via cloning and installing
# https://github.com/nabeelre/FATS.git (Python 3 fork)
import FATS
from FATS.Feature import FeatureSpace
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------
# Feature definitions (same as your Starembed script)
# -------------------------
FATS_feature_names = [
    'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Autocor_length', 'Con', 'PairSlopeTrend',
    'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1',
    'Freq1_harmonics_amplitude_2', 'Freq1_harmonics_amplitude_3',
    'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1',
    'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3',
    'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1',
    'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3',
    'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1',
    'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3',
    'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1',
    'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_3',
    'Freq3_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_1',
    'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3',
    'CAR_sigma', 'CAR_tau', 'CAR_mean', 'Con', 'PairSlopeTrend',
]

LC_extractor = light_curve.Extractor(
    light_curve.Amplitude(),
    light_curve.AndersonDarlingNormal(),
    light_curve.BeyondNStd(nstd=1),
    light_curve.BeyondNStd(nstd=2),
    light_curve.BeyondNStd(nstd=3),
    light_curve.Cusum(),
    light_curve.Eta(),
    light_curve.EtaE(),
    light_curve.InterPercentileRange(0.25),
    light_curve.InterPercentileRange(0.1),
    light_curve.Kurtosis(),
    light_curve.LinearFit(),
    light_curve.LinearTrend(),
    light_curve.MagnitudePercentageRatio(),
    light_curve.MaximumSlope(),
    light_curve.Mean(),
    light_curve.Median(),
    light_curve.MedianAbsoluteDeviation(),
    light_curve.MedianBufferRangePercentage(),
    light_curve.OtsuSplit(),
    light_curve.PercentAmplitude(),
    light_curve.ReducedChi2(),
    light_curve.Skew(),
    light_curve.StandardDeviation(),
    light_curve.StetsonK(),
    light_curve.WeightedMean(),
)

# -------------------------
# Arrow loading (same idea as your loader)
# -------------------------
def load_arrow_split(split_dir: str | Path) -> pa.Table:
    split_dir = Path(split_dir)
    arrow_files = sorted(split_dir.glob("data-*.arrow"))
    tables: List[pa.Table] = []
    for f in arrow_files:
        with f.open("rb") as fh:
            reader = ipc.RecordBatchStreamReader(fh)
            tables.append(reader.read_all())
    if not tables:
        raise FileNotFoundError(f"No data-*.arrow files found under: {split_dir}")
    return pa.concat_tables(tables)

# -------------------------
# Per-lightcurve feature computation (aligned with Starembed)
# -------------------------
# NOTE: we create one FeatureSpace per process in _worker_init() for speed.
_FATS_FS = None
_BANDS: List[str] = []
_LC_NAMES: List[str] = []

def _worker_init(bands: List[str]) -> None:
    global _FATS_FS, _BANDS, _LC_NAMES
    _BANDS = bands
    _LC_NAMES = list(LC_extractor.names)
    _FATS_FS = FeatureSpace(
        Data=['magnitude', 'time', 'error'],
        featureList=FATS_feature_names,
    )

def _as_np(x: Any) -> np.ndarray:
    # robust conversion of lists / arrays / pyarrow scalars to numpy
    return np.asarray(x, dtype=np.float64)

def calc_FATS_features(lc: Optional[Dict[str, Any]]) -> np.ndarray:
    if lc is None:
        return np.full(len(FATS_feature_names), -1.0, dtype=np.float64)

    mag = _as_np(lc.get("target", []))
    mjd = _as_np(lc.get("mjd", []))
    err = _as_np(lc.get("past_feat_dynamic_real", []))

    # Minimal sanity: length must align; if not, fail -> caught by caller and returns -2
    if len(mag) == 0 or len(mjd) == 0 or len(err) == 0:
        return np.full(len(FATS_feature_names), -1.0, dtype=np.float64)

    lc_2darr = np.array([mag, mjd, err], dtype=np.float64)
    fs = _FATS_FS.calculateFeature(lc_2darr)
    return np.asarray(fs.result(), dtype=np.float64)

def calc_LC_features(lc: Optional[Dict[str, Any]]) -> np.ndarray:
    if lc is None:
        return np.full(len(_LC_NAMES), -1.0, dtype=np.float64)

    mag = _as_np(lc.get("target", []))
    if len(mag) <= 4:
        return np.full(len(_LC_NAMES), -1.0, dtype=np.float64)

    mjd = _as_np(lc.get("mjd", []))
    err = _as_np(lc.get("past_feat_dynamic_real", []))

    feats = LC_extractor(mjd, mag, err, sorted=True, check=False)
    return np.asarray(feats, dtype=np.float64)

def _get_star_bands_container(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your Arrow rows look like: row["bands_data"][0]["g"]["target"] ...
    HF rows looked like: row["bands_data"]["g"]["target"] ...
    This normalizes both to a dict: bands_container[band] -> lc_dict
    """
    bd = row.get("bands_data", None)
    if bd is None:
        return {}
    if isinstance(bd, list):
        if len(bd) == 0:
            return {}
        if isinstance(bd[0], dict):
            return bd[0]
        return {}
    if isinstance(bd, dict):
        return bd
    return {}

def _parse_label(row: Dict[str, Any]) -> Tuple[int, str]:
    cls = row.get("class_str", None)
    if isinstance(cls, list):
        cls = cls[0] if cls else ""
    if cls is None:
        cls = row.get("class", "")
        if isinstance(cls, list):
            cls = cls[0] if cls else ""
    cls_str = "" if cls is None else str(cls)
    try:
        y = int(cls_str)
    except Exception:
        y = -1
    return y, cls_str

def process_rows(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: float32 [N, D]
      y: int32   [N]
      cls_str: object/string [N]
    """
    # Column ordering matches Starembed:
    #   (band1_FATS ... bandK_FATS) then (band1_LC ... bandK_LC)
    D = len(_BANDS) * (len(FATS_feature_names) + len(_LC_NAMES))
    X = np.empty((len(rows), D), dtype=np.float32)
    y = np.empty((len(rows),), dtype=np.int32)
    cls_str = np.empty((len(rows),), dtype=object)

    for i, row in enumerate(rows):
        bands_container = _get_star_bands_container(row)

        fats_parts: List[np.ndarray] = []
        lc_parts: List[np.ndarray] = []

        for band in _BANDS:
            lc = bands_container.get(band, None)
            try:
                fats = calc_FATS_features(lc)
            except Exception:
                fats = np.full(len(FATS_feature_names), -2.0, dtype=np.float64)
            fats_parts.append(fats)

        for band in _BANDS:
            lc = bands_container.get(band, None)
            try:
                lcf = calc_LC_features(lc)
            except Exception:
                lcf = np.full(len(_LC_NAMES), -2.0, dtype=np.float64)
            lc_parts.append(lcf)

        feats = np.concatenate(fats_parts + lc_parts).astype(np.float32, copy=False)
        X[i, :] = feats

        yi, cs = _parse_label(row)
        y[i] = yi
        cls_str[i] = cs

    return X, y, cls_str

# -------------------------
# Batch iteration
# -------------------------
def iter_row_batches(table: pa.Table, batch_size: int) -> Sequence[List[Dict[str, Any]]]:
    # Use record batches to avoid per-row slice overhead
    for rb in table.to_batches(max_chunksize=batch_size):
        yield rb.to_pylist()

# -------------------------
# Saving
# -------------------------
def save_npz(out_file: Path, X: np.ndarray, y: np.ndarray, feature_names: List[str], cls_str: Optional[np.ndarray] = None) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x": X, "y": y, "feature_names": np.array(feature_names, dtype=object)}
    if cls_str is not None:
        payload["class_str"] = np.array(cls_str, dtype=object)
    np.savez_compressed(str(out_file), **payload)

def save_arrow_fixed_list(out_file: Path, X: np.ndarray, y: np.ndarray, feature_names: List[str], cls_str: Optional[np.ndarray] = None) -> None:
    """
    Stores:
      - x: FixedSizeList<float32>[D]
      - y: int32
      - class_str: string (optional)
    Plus schema metadata with feature_names.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    D = X.shape[1]

    # child values for fixed-size list
    values = pa.array(X.reshape(-1), type=pa.float32())
    x_col = pa.FixedSizeListArray.from_arrays(values, D)

    y_col = pa.array(y, type=pa.int32())

    arrays = [x_col, y_col]
    names = ["x", "y"]

    if cls_str is not None:
        arrays.append(pa.array(cls_str.astype(str), type=pa.string()))
        names.append("class_str")

    schema = pa.schema([
        ("x", pa.fixed_size_list(pa.float32(), D)),
        ("y", pa.int32()),
        *([("class_str", pa.string())] if cls_str is not None else []),
    ])

    md = dict(schema.metadata or {})
    md[b"feature_names"] = json.dumps(feature_names).encode("utf-8")
    schema = schema.with_metadata(md)

    with pa.OSFile(str(out_file), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            rb = pa.RecordBatch.from_arrays(arrays, names=names).cast(schema)
            writer.write_batch(rb)

# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Starembed handcrafted features from Arrow shards (data-*.arrow)")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing split subdirs: train/test/validation/anom")
    parser.add_argument("--splits", type=str, default="train,validation,test,anom", help='Comma-separated splits to process')
    parser.add_argument("--bands", type=str, default=None, help='Comma-separated band list (e.g. "g,r"). If omitted, infer from first row.')
    parser.add_argument("--batch_size", type=int, default=256, help="Rows per batch (conversion to python objects happens per batch)")
    parser.add_argument("--num_workers", type=int, default=4, help="Process workers for CPU parallelism (0/1 = single process)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_format", type=str, choices=["npz", "arrow"], default="npz", help="Output format per split")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        split_dir = base_dir / split
        print(f"\n=== Processing split: {split} ({split_dir}) ===")
        table = load_arrow_split(split_dir)

        # Infer bands if not provided
        if args.bands is not None:
            bands = [b.strip() for b in args.bands.split(",") if b.strip()]
        else:
            first = table.slice(0, 1).to_pylist()[0]
            bc = _get_star_bands_container(first)
            bands = sorted(list(bc.keys()))
            print(f"Inferred bands: {bands}")

        # Build feature name list (same ordering as X)
        lc_names = list(LC_extractor.names)
        feature_names: List[str] = []
        for band in bands:
            feature_names.extend([f"{band}_{n}" for n in FATS_feature_names])
        for band in bands:
            feature_names.extend([f"{band}_{n}" for n in lc_names])

        N = table.num_rows
        print(f"Rows: {N}, bands: {bands}, feature_dim: {len(feature_names)}")

        # Extract
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_cls: List[np.ndarray] = []

        use_mp = args.num_workers is not None and args.num_workers > 1
        if use_mp:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.num_workers, initializer=_worker_init, initargs=(bands,)) as pool:
                batches = list(iter_row_batches(table, args.batch_size))
                for Xb, yb, csb in tqdm(
                    pool.imap(process_rows, batches),
                    total=len(batches),
                    desc="Computing handcrafted features"
                ):
                    all_X.append(Xb)
                    all_y.append(yb)
                    all_cls.append(csb)
        else:
            _worker_init(bands)
            for rows in tqdm(iter_row_batches(table, args.batch_size), total=(N + args.batch_size - 1) // args.batch_size):
                Xb, yb, csb = process_rows(rows)
                all_X.append(Xb)
                all_y.append(yb)
                all_cls.append(csb)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        cls_str = np.concatenate(all_cls, axis=0)

        assert X.shape[0] == N and y.shape[0] == N, "Row count mismatch after batching"

        # Save
        if args.save_format == "npz":
            out_file = out_dir / f"handcrafted_features_{split}.npz"
            save_npz(out_file, X, y, feature_names, cls_str=cls_str)
            print(f"Saved: {out_file}")
        else:
            out_file = out_dir / f"handcrafted_features_{split}.arrow"
            save_arrow_fixed_list(out_file, X, y, feature_names, cls_str=cls_str)
            print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()