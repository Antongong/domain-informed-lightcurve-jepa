from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

import pyrregular.accessor  # noqa: F401  # registers the xarray accessor
from pyrregular.io_utils import load_from_file


PYRREGULAR_BASE_URL = "https://huggingface.co/datasets/splandi/pyrregular/resolve/main/data_final/"
PYRREGULAR_TREE_API = "https://huggingface.co/api/datasets/splandi/pyrregular/tree/main/data_final"
PYRREGULAR_FALLBACK_DATASETS = [
    "Abf.h5",
    "Ais.h5",
    "AllGestureWiimoteX.h5",
    "AllGestureWiimoteY.h5",
    "AllGestureWiimoteZ.h5",
    "Animals.h5",
    "AsphaltObstaclesCoordinates.h5",
    "AsphaltPavementTypeCoordinates.h5",
    "AsphaltRegularityCoordinates.h5",
    "CharacterTrajectories.h5",
    "CombinedTrajectories.h5",
    "DodgerLoopDay.h5",
    "DodgerLoopGame.h5",
    "DodgerLoopWeekend.h5",
    "Garment.h5",
    "Geolife.h5",
    "GeolifeSupervised.h5",
    "GestureMidAirD1.h5",
    "GestureMidAirD2.h5",
    "GestureMidAirD3.h5",
    "GesturePebbleZ1.h5",
    "GesturePebbleZ2.h5",
    "InsectWingbeat.h5",
    "JapaneseVowels.h5",
    "Ldfpa.h5",
    "MelbournePedestrian.h5",
    "Mimic3.h5",
    "PLAID.h5",
    "Pamap2.h5",
    "Physionet2012.h5",
    "Physionet2019.h5",
    "PickupGestureWiimoteZ.h5",
    "Seabirds.h5",
    "ShakeGestureWiimoteZ.h5",
    "SpokenArabicDigits.h5",
    "TDrive.h5",
    "Taxi.h5",
    "Vehicles.h5",
]


def ensure_dataset_name(name: str) -> str:
    return name if name.endswith(".h5") else f"{name}.h5"


def resolve_requested_datasets(datasets: Sequence[str]) -> List[str]:
    if len(datasets) == 1 and str(datasets[0]).strip().lower() in {"all", "*"}:
        return fetch_pyrregular_dataset_names()
    return [ensure_dataset_name(name) for name in datasets]


def fetch_pyrregular_dataset_names(timeout: int = 60) -> List[str]:
    try:
        with urllib.request.urlopen(PYRREGULAR_TREE_API, timeout=timeout) as handle:
            payload = json.load(handle)
        names = [
            item["path"].split("/")[-1]
            for item in payload
            if str(item.get("path", "")).endswith(".h5")
        ]
        if names:
            return sorted(names)
    except Exception:
        pass
    return list(PYRREGULAR_FALLBACK_DATASETS)


def is_valid_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def download_with_curl(url: str, out_path: Path) -> None:
    cmd = [
        "curl",
        "-L",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "-o",
        str(out_path),
        url,
    ]
    last_exc: Optional[BaseException] = None
    for attempt in range(1, 6):
        try:
            subprocess.run(cmd, check=True)
            last_exc = None
            break
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt >= 5:
                break
            time.sleep(5.0)
    if last_exc is not None:
        raise last_exc


def download_chunk(
    url: str,
    chunk_path: Path,
    start: int,
    end: int,
    expected_size: int,
) -> None:
    if chunk_path.exists() and chunk_path.stat().st_size == expected_size:
        return

    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    last_exc: Optional[BaseException] = None
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(1, 6):
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range request returned status={response.status_code} for bytes={start}-{end}"
                    )
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Chunk size mismatch for {chunk_path.name}: "
                    f"expected={expected_size} got={tmp_path.stat().st_size}"
                )
            tmp_path.replace(chunk_path)
            return
        except Exception as exc:
            last_exc = exc
            tmp_path.unlink(missing_ok=True)
            if attempt >= 5:
                break
            time.sleep(min(30.0, 2.0**attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download chunk bytes={start}-{end}")


def download_parallel(url: str, out_path: Path, workers: int, chunk_size_mb: int) -> None:
    chunk_bytes = max(1, int(chunk_size_mb)) * 1024 * 1024
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        total_size = int(head.headers.get("content-length", "0"))
    except Exception:
        total_size = 0

    if total_size <= 0 or workers <= 1 or total_size <= chunk_bytes:
        download_with_curl(url, out_path)
        return

    probe = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        allow_redirects=True,
        timeout=(30, 120),
    )
    probe.close()
    if probe.status_code != 206:
        download_with_curl(url, out_path)
        return

    parts_dir = out_path.with_name(out_path.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    parts: List[Tuple[int, Path]] = []
    start = 0
    index = 0
    while start < total_size:
        end = min(total_size - 1, start + chunk_bytes - 1)
        part_path = parts_dir / f"{index:04d}.part"
        parts.append((end - start + 1, part_path))
        start = end + 1
        index += 1

    print(
        f"[Download] {out_path.name}: size={total_size} bytes chunks={len(parts)} workers={min(workers, len(parts))}",
        flush=True,
    )

    futures = []
    with ThreadPoolExecutor(max_workers=min(workers, len(parts))) as pool:
        start = 0
        for expected_size, part_path in parts:
            end = start + expected_size - 1
            futures.append(pool.submit(download_chunk, url, part_path, start, end, expected_size))
            start = end + 1
        for future in as_completed(futures):
            future.result()

    merged_path = out_path.with_suffix(out_path.suffix + ".tmp")
    merged_path.unlink(missing_ok=True)
    with open(merged_path, "wb") as out_handle:
        for expected_size, part_path in parts:
            if part_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Chunk size mismatch while merging {part_path}: "
                    f"expected={expected_size} got={part_path.stat().st_size}"
                )
            with open(part_path, "rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=8 * 1024 * 1024)
    if merged_path.stat().st_size != total_size:
        raise RuntimeError(
            f"Merged file size mismatch for {out_path.name}: "
            f"expected={total_size} got={merged_path.stat().st_size}"
        )
    merged_path.replace(out_path)
    for _, part_path in parts:
        part_path.unlink(missing_ok=True)
    parts_dir.rmdir()


def ensure_dataset_local(name: str, cache_dir: Path, download_workers: int, download_chunk_mb: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = ensure_dataset_name(name)
    out_path = cache_dir / dataset_name
    if out_path.exists() and out_path.stat().st_size > 0:
        if is_valid_hdf5(out_path):
            return out_path
        out_path.unlink()

    url = f"{PYRREGULAR_BASE_URL}{dataset_name}"
    download_parallel(url, out_path, workers=download_workers, chunk_size_mb=download_chunk_mb)
    if not out_path.exists() or out_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to download dataset: {dataset_name}")
    if not is_valid_hdf5(out_path):
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid HDF5 dataset: {dataset_name}")
    return out_path


def load_pyrregular_dense(path: Path) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    da = load_from_file(path)
    task_cfg = da.attrs.get("configs", {}).get("default", {})
    task_name = task_cfg.get("task", "classification")
    if task_name != "classification":
        raise RuntimeError(f"Unsupported task={task_name!r}; only classification is handled.")
    if not hasattr(da, "irr"):
        raise RuntimeError(
            f"Dataset object {type(da).__name__} does not expose the irregular-series accessor required for classification."
        )
    x, t = da.irr.to_dense(index_scale=1.0, absolute_time=False)
    y, split = da.irr.get_task_target_and_split()
    return (
        da,
        np.asarray(x, dtype=np.float32),
        np.asarray(t, dtype=np.float32),
        np.asarray(y),
        np.asarray(split).astype(str),
    )


def chronos2_scale(values: np.ndarray) -> np.ndarray:
    mu = float(values.mean())
    sigma = float(values.std())
    if not np.isfinite(sigma) or sigma < 1.0e-6:
        sigma = 1.0
    return (values - mu) / sigma


def resolve_sample_times(t: np.ndarray, sample_idx: int) -> np.ndarray:
    return t[sample_idx, 0] if t.ndim == 3 else t[sample_idx]


def prepare_single_channel_lc(
    values: np.ndarray,
    times: np.ndarray,
    *,
    value_scaling: str,
    time_strategy: str,
    err_value: float,
) -> Optional[np.ndarray]:
    mask = np.isfinite(values)
    if time_strategy != "rank":
        mask &= np.isfinite(times)
    if not bool(mask.any()):
        return None

    vals = values[mask].astype(np.float32, copy=False)
    if value_scaling == "chronos2":
        vals = chronos2_scale(vals).astype(np.float32, copy=False)

    if time_strategy == "rank":
        t_out = np.arange(vals.shape[0], dtype=np.float32)
    else:
        t_out = times[mask].astype(np.float32, copy=False)
        t_out = t_out - np.nanmin(t_out)
        if time_strategy == "normalized":
            span = float(np.nanmax(t_out)) if t_out.size else 0.0
            if span > 1.0e-6:
                t_out = t_out / span
        elif time_strategy == "relative":
            diffs = np.diff(np.sort(t_out.astype(np.float64, copy=False)))
            diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-12)]
            if diffs.size:
                scale = float(np.median(diffs))
                if np.isfinite(scale) and scale > 1.0e-12:
                    t_out = (t_out / scale).astype(np.float32, copy=False)

    err = np.full_like(vals, float(err_value), dtype=np.float32)
    valid = np.ones_like(vals, dtype=np.float32)
    return np.stack([t_out, vals, err, valid], axis=-1)


def maybe_cap_lc_points(lc: Optional[np.ndarray], max_points: int) -> Optional[np.ndarray]:
    if lc is None:
        return None
    limit = int(max_points)
    if limit <= 0 or int(lc.shape[0]) <= limit:
        return lc
    idx = np.linspace(0, int(lc.shape[0]) - 1, num=limit, dtype=np.int64)
    idx = np.unique(idx)
    if idx.shape[0] <= 0:
        return lc[:1]
    return lc[idx]


def collate_lc_batch(samples: Sequence[np.ndarray]) -> torch.Tensor:
    max_len = max(sample.shape[0] for sample in samples)
    batch = torch.zeros((len(samples), max_len, 4), dtype=torch.float32)
    for i, sample in enumerate(samples):
        n = sample.shape[0]
        batch[i, :n] = torch.from_numpy(sample)
    return batch


def non_test_mask(split: np.ndarray) -> np.ndarray:
    return np.asarray(split).astype(str) != "test"


class PyrregularChannelDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        split: np.ndarray,
        *,
        train_only: bool,
        value_scaling: str,
        time_strategy: str,
        err_value: float,
        min_valid_points: int = 1,
        max_points_per_series: int = 0,
        progress_label: Optional[str] = None,
        progress_every_samples: int = 128,
    ) -> None:
        super().__init__()
        self.x = x
        self.t = t
        self.value_scaling = value_scaling
        self.time_strategy = time_strategy
        self.err_value = float(err_value)
        self.min_valid_points = max(1, int(min_valid_points))
        self.max_points_per_series = max(0, int(max_points_per_series))

        sample_mask = non_test_mask(split) if train_only else np.ones(x.shape[0], dtype=bool)
        sample_indices = np.nonzero(sample_mask)[0]
        total_samples = int(sample_indices.shape[0])
        self.index: List[Tuple[int, int]] = []
        self.lengths: List[int] = []
        progress_interval = max(1, min(max(1, int(progress_every_samples)), max(1, total_samples // 8)))
        for sample_pos, sample_idx in enumerate(sample_indices, start=1):
            times = resolve_sample_times(t, int(sample_idx))
            for channel_idx in range(x.shape[1]):
                lc = prepare_single_channel_lc(
                    x[int(sample_idx), channel_idx],
                    times,
                    value_scaling=self.value_scaling,
                    time_strategy=self.time_strategy,
                    err_value=self.err_value,
                )
                lc = maybe_cap_lc_points(lc, self.max_points_per_series)
                if lc is not None and int(lc.shape[0]) >= self.min_valid_points:
                    self.index.append((int(sample_idx), int(channel_idx)))
                    self.lengths.append(int(lc.shape[0]))
            if progress_label and (
                sample_pos == 1
                or sample_pos == total_samples
                or sample_pos % progress_interval == 0
            ):
                print(
                    f"[{progress_label}] indexed samples {sample_pos}/{total_samples}; series={len(self.index)}",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> np.ndarray:
        sample_idx, channel_idx = self.index[idx]
        times = resolve_sample_times(self.t, sample_idx)
        lc = prepare_single_channel_lc(
            self.x[sample_idx, channel_idx],
            times,
            value_scaling=self.value_scaling,
            time_strategy=self.time_strategy,
            err_value=self.err_value,
        )
        lc = maybe_cap_lc_points(lc, self.max_points_per_series)
        if lc is None or int(lc.shape[0]) < self.min_valid_points:
            raise RuntimeError(f"Unexpected invalid channel at dataset index={idx}")
        return lc


def pyrregular_channel_collate(batch: Sequence[np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {"X": {"raw_lc": collate_lc_batch(batch)}}


def compute_channel_tuning_stats(
    x: np.ndarray,
    t: np.ndarray,
    split: np.ndarray,
    *,
    value_scaling: str,
    time_strategy: str,
    err_value: float,
    theta_of_light_curve: float = 1000.0,
    min_valid_points: int = 1,
    seed: int = 42,
    value_sample_limit: int = 200_000,
    progress_label: Optional[str] = None,
    progress_every_samples: int = 128,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    sample_mask = non_test_mask(split)
    sample_indices = np.nonzero(sample_mask)[0]
    total_samples = int(sample_indices.shape[0])

    value_sample = np.empty((max(1, int(value_sample_limit)),), dtype=np.float32)
    seen_values = 0
    keep_values = 0

    valid_series = 0
    valid_samples = int(sample_mask.sum())
    max_len = 0
    total_len = 0
    span_values: List[float] = []
    min_positive_dt = math.inf

    progress_interval = max(1, min(max(1, int(progress_every_samples)), max(1, total_samples // 8)))
    for sample_pos, sample_idx in enumerate(sample_indices, start=1):
        times = resolve_sample_times(t, int(sample_idx))
        for channel_idx in range(x.shape[1]):
            lc = prepare_single_channel_lc(
                x[int(sample_idx), channel_idx],
                times,
                value_scaling=value_scaling,
                time_strategy=time_strategy,
                err_value=err_value,
            )
            if lc is None or int(lc.shape[0]) < max(1, int(min_valid_points)):
                continue

            valid_series += 1
            positions = lc[:, 0].astype(np.float64, copy=False)
            values = lc[:, 1].astype(np.float32, copy=False)
            values_centered = values - float(values.mean())
            n = int(values_centered.shape[0])
            max_len = max(max_len, n)
            total_len += n

            if positions.size > 1:
                span = float(positions[-1] - positions[0])
                if np.isfinite(span) and span > 0.0:
                    span_values.append(span)
                diffs = np.diff(positions)
                diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-12)]
                if diffs.size:
                    min_positive_dt = min(min_positive_dt, float(diffs.min()))

            if keep_values < value_sample.shape[0]:
                take = min(n, value_sample.shape[0] - keep_values)
                value_sample[keep_values : keep_values + take] = values_centered[:take]
                keep_values += take
                seen_values += n
            else:
                for value in values_centered:
                    seen_values += 1
                    j = rng.randint(0, seen_values)
                    if j < value_sample.shape[0]:
                        value_sample[j] = value

        if progress_label and (
            sample_pos == 1
            or sample_pos == total_samples
            or sample_pos % progress_interval == 0
        ):
            print(
                f"[{progress_label}] processed samples {sample_pos}/{total_samples}; valid_series={valid_series}",
                flush=True,
            )

    if valid_series <= 0:
        raise RuntimeError("No valid training channels found in dataset.")

    sampled_values = value_sample[:keep_values]
    if sampled_values.size <= 0:
        sampled_values = np.asarray([0.0], dtype=np.float32)

    q01, q99 = np.quantile(sampled_values, [0.01, 0.99]).tolist()
    if not np.isfinite(q01):
        q01 = float(np.nanmin(sampled_values))
    if not np.isfinite(q99):
        q99 = float(np.nanmax(sampled_values))
    if not np.isfinite(q01):
        q01 = -1.0
    if not np.isfinite(q99):
        q99 = 1.0
    if q99 <= q01:
        center = float(np.nanmean(sampled_values)) if np.isfinite(np.nanmean(sampled_values)) else 0.0
        q01 = center - 1.0
        q99 = center + 1.0

    if span_values:
        span_arr = np.asarray(span_values, dtype=np.float64)
        span_p95 = float(np.quantile(span_arr, 0.95))
        span_p99 = float(np.quantile(span_arr, 0.99))
        span_max = float(span_arr.max())
    else:
        span_p95 = float(max_len)
        span_p99 = float(max_len)
        span_max = float(max_len)

    if not math.isfinite(min_positive_dt):
        min_positive_dt = 1.0 if time_strategy == "rank" else max(span_p95 / max(max_len, 2), 1.0e-3)

    inference_min_period = max(2.0 * float(min_positive_dt), 1.0e-4)
    longest_delta_t_train = span_max if span_max > 0.0 else float(max_len)
    inference_max_period = max(inference_min_period * 4.0, longest_delta_t_train)
    raw_rope_max_period = max(1.0e-3, (float(theta_of_light_curve) / 1000.0) * longest_delta_t_train)
    per_rope_max_period = max(4.0, math.log10(inference_max_period) - math.log10(inference_min_period))

    max_k_periods = min(8192, max(1024, int(4 * max_len)))
    k_periods = max(512, int(max_k_periods))
    select_k = max(32, min(256, k_periods // 8))

    return {
        "train_samples": valid_samples,
        "train_series": valid_series,
        "n_channels": int(x.shape[1]),
        "max_seq_len": int(max_len),
        "mean_seq_len": float(total_len / max(valid_series, 1)),
        "span_p95": float(span_p95),
        "span_p99": float(span_p99),
        "span_max": float(span_max),
        "longest_delta_t_train": float(longest_delta_t_train),
        "min_positive_dt": float(min_positive_dt),
        "raw_vmin": float(q01),
        "raw_vmax": float(q99),
        "theta_of_light_curve": float(theta_of_light_curve),
        "raw_rope_max_period": float(raw_rope_max_period),
        "per_rope_max_period": float(per_rope_max_period),
        "pf_rope_max_period": 10.0,
        "inference_min_period": float(inference_min_period),
        "inference_max_period": float(inference_max_period),
        "inference_k_periods": int(k_periods),
        "inference_k_top": int(select_k),
        "inference_k_rand": int(select_k),
        "value_scaling": str(value_scaling),
        "time_strategy": str(time_strategy),
    }
