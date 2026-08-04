#!/usr/bin/env python3
"""
For each truncation level (100%, 50%, 25%), compute mean and std of:
  - number of datapoints per light curve, per band
  - observational baseline (days between first and last point), per band

Reads the raw Arrow light-curve data directly (not the extracted handcrafted
features), across all four splits (train/validation/test/anom), so the stats
reflect the whole dataset at each truncation level, not just one split.
"""
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

DATA_COMPLETE = "/Users/anton.gong/Documents/SJTU/PYTHON/Yicheng proj/data_complete"
DATA_TRUNCATED = "/Users/anton.gong/Documents/SJTU/PYTHON/Yicheng proj/data_truncated"

TRUNC_TO_DIR = {
    100: DATA_COMPLETE,
    50: os.path.join(DATA_TRUNCATED, "pct50"),
    25: os.path.join(DATA_TRUNCATED, "pct25"),
}

SPLITS = ["train", "validation", "test", "anom"]
BANDS = ["g", "i", "r"]


def load_arrow_split(split_dir):
    split_dir = Path(split_dir)
    files = sorted(split_dir.glob("data-*.arrow"))
    tables = []
    for f in files:
        with f.open("rb") as fh:
            tables.append(ipc.RecordBatchStreamReader(fh).read_all())
    return pa.concat_tables(tables)


def get_bands_container(row):
    bd = row.get("bands_data", None)
    if isinstance(bd, list):
        return bd[0] if bd and isinstance(bd[0], dict) else {}
    return bd or {}


def collect_stats(base_dir):
    """Gather n_points and baseline (days) for every star, every band,
    across all available splits for one truncation level."""
    n_points_by_band = {b: [] for b in BANDS}
    baseline_by_band = {b: [] for b in BANDS}

    for split in SPLITS:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        table = load_arrow_split(split_dir)
        rows = table.to_pylist()
        for row in rows:
            bc = get_bands_container(row)
            for band in BANDS:
                lc = bc.get(band)
                if lc is None:
                    continue
                mjd = np.asarray(lc.get("mjd", []), dtype=np.float64)
                n = len(mjd)
                n_points_by_band[band].append(n)
                if n >= 2:
                    baseline_by_band[band].append(mjd[-1] - mjd[0])

    return n_points_by_band, baseline_by_band


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return arr.mean(), arr.std(ddof=0), len(arr)


def main():
    rows = []
    for pct, base_dir in TRUNC_TO_DIR.items():
        print(f"\n=== {pct}% truncation ({base_dir}) ===")
        n_points_by_band, baseline_by_band = collect_stats(base_dir)

        for band in BANDS:
            n_mean, n_std, n_count = summarize(n_points_by_band[band])
            b_mean, b_std, b_count = summarize(baseline_by_band[band])
            print(
                f"  band {band}: n_points={n_mean:.1f}+/-{n_std:.1f} (n={n_count})  "
                f"baseline_days={b_mean:.1f}+/-{b_std:.1f} (n={b_count})"
            )
            rows.append({
                "truncation_pct": pct, "band": band,
                "n_points_mean": n_mean, "n_points_std": n_std, "n_points_count": n_count,
                "baseline_days_mean": b_mean, "baseline_days_std": b_std, "baseline_days_count": b_count,
            })

        all_n = [x for band_list in n_points_by_band.values() for x in band_list]
        all_b = [x for band_list in baseline_by_band.values() for x in band_list]
        n_mean, n_std, n_count = summarize(all_n)
        b_mean, b_std, b_count = summarize(all_b)
        print(
            f"  ALL BANDS: n_points={n_mean:.1f}+/-{n_std:.1f} (n={n_count})  "
            f"baseline_days={b_mean:.1f}+/-{b_std:.1f} (n={b_count})"
        )
        rows.append({
            "truncation_pct": pct, "band": "all",
            "n_points_mean": n_mean, "n_points_std": n_std, "n_points_count": n_count,
            "baseline_days_mean": b_mean, "baseline_days_std": b_std, "baseline_days_count": b_count,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trunc_time_stats.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
