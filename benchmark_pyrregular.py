#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import requests
import torch
import yaml
from torch.nn import DataParallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pyrregular.accessor  # noqa: F401  # registers the xarray accessor
from pyrregular.io_utils import load_from_file

from extract_starembed_embeddings import build_model_from_config, load_model_from_ckpt
from pyrregular_utils import maybe_cap_lc_points, resolve_requested_datasets

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PYRREGULAR_BASE_URL = "https://huggingface.co/datasets/splandi/pyrregular/resolve/main/data_final/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the trained raw encoder on PYRREGULAR classification datasets."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="PYRREGULAR datasets to evaluate, or pass 'all' for the full current release.",
    )
    parser.add_argument("--cache_dir", type=str, default=str(Path.home() / ".cache" / "pyrregular"))
    parser.add_argument("--out_dir", type=str, default="runs/pyrregular")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--download_chunk_mb", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_gpus", type=int, default=8 if torch.cuda.is_available() else 0)
    parser.add_argument(
        "--value_scaling",
        type=str,
        default="chronos2",
        choices=["chronos2", "none"],
        help="chronos2 applies per-series z-score scaling before encoding.",
    )
    parser.add_argument(
        "--time_strategy",
        type=str,
        default="rank",
        choices=["rank", "original", "normalized", "relative"],
        help="rank matches the Chronos-2 habit of ignoring irregular timestamps and using sample order.",
    )
    parser.add_argument(
        "--channel_pool",
        type=str,
        default="mean",
        choices=["mean", "concat"],
        help="How to combine multivariate channels after encoding each channel separately.",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        default="logistic",
        help="Comma-separated subset of logistic,mlp,knn,rf.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--err_value", type=float, default=0.1)
    parser.add_argument("--max_points_per_series", type=int, default=0)
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--no_strict",
        dest="strict",
        action="store_false",
        help="Allow partial checkpoint loads. Default behavior is strict loading.",
    )
    parser.set_defaults(skip_existing=True, fail_fast=False)
    parser.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Recompute datasets even if a report already exists under out_dir.",
    )
    parser.add_argument(
        "--fail_fast",
        dest="fail_fast",
        action="store_true",
        help="Abort on the first dataset failure instead of recording the error and continuing.",
    )
    return parser.parse_args()


def ensure_dataset_name(name: str) -> str:
    return name if name.endswith(".h5") else f"{name}.h5"


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


def maybe_wrap_dataparallel(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_gpus: int,
    batch_size: int,
    tag: str,
) -> torch.nn.Module:
    if device.type != "cuda":
        return model
    available = torch.cuda.device_count()
    requested = available if int(num_gpus) <= 0 else min(int(num_gpus), available)
    if requested <= 1:
        print(f"[Init] {tag}: using a single GPU", flush=True)
        return model
    wrapped = DataParallel(model, device_ids=list(range(requested)))
    print(
        f"[Init] {tag}: using DataParallel on {requested} GPUs; global_batch_size={batch_size}",
        flush=True,
    )
    return wrapped


def build_benchmark_model(
    cfg: Dict[str, Any],
    ckpt_path: str,
    *,
    device: torch.device,
    strict: bool,
    num_gpus: int,
    batch_size: int,
) -> torch.nn.Module:
    model = build_model_from_config(cfg)
    load_model_from_ckpt(model, ckpt_path, strict=strict)
    model.enable[model.VIEW_RAW] = True
    model.enable[model.VIEW_PERIODOGRAM] = False
    model.enable[model.VIEW_PHASE_FOLDED] = False
    model.enable[model.VIEW_GROUP] = False
    model.phase_use_normalized_phase = False
    model.to(device)
    model = maybe_wrap_dataparallel(
        model,
        device=device,
        num_gpus=num_gpus,
        batch_size=batch_size,
        tag="inference",
    )
    model.eval()
    return model


def chronos2_scale(values: np.ndarray) -> np.ndarray:
    mu = float(values.mean())
    sigma = float(values.std())
    if not np.isfinite(sigma) or sigma < 1.0e-6:
        sigma = 1.0
    return (values - mu) / sigma


def prepare_single_channel_lc(
    values: np.ndarray,
    times: np.ndarray,
    *,
    value_scaling: str,
    time_strategy: str,
    err_value: float,
    max_points_per_series: int,
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
        t = np.arange(vals.shape[0], dtype=np.float32)
    else:
        t = times[mask].astype(np.float32, copy=False)
        t = t - np.nanmin(t)
        if time_strategy == "normalized":
            span = float(np.nanmax(t)) if t.size else 0.0
            if span > 1.0e-6:
                t = t / span
        elif time_strategy == "relative":
            diffs = np.diff(np.sort(t.astype(np.float64, copy=False)))
            diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-12)]
            if diffs.size:
                scale = float(np.median(diffs))
                if np.isfinite(scale) and scale > 1.0e-12:
                    t = (t / scale).astype(np.float32, copy=False)

    err = np.full_like(vals, float(err_value), dtype=np.float32)
    valid = np.ones_like(vals, dtype=np.float32)
    lc = np.stack([t, vals, err, valid], axis=-1)
    return maybe_cap_lc_points(lc, int(max_points_per_series))


def collate_lc_batch(samples: Sequence[np.ndarray]) -> torch.Tensor:
    max_len = max(sample.shape[0] for sample in samples)
    batch = torch.zeros((len(samples), max_len, 4), dtype=torch.float32)
    for i, sample in enumerate(samples):
        n = sample.shape[0]
        batch[i, :n] = torch.from_numpy(sample)
    return batch


@torch.inference_mode()
def encode_lc_batch(model: torch.nn.Module, batch_lc: torch.Tensor, device: torch.device) -> np.ndarray:
    out = model({"X": {"lc": batch_lc.to(device, non_blocking=True)}})
    emb = out["embeddings"]["raw"].detach().cpu().numpy().astype(np.float32, copy=False)
    return emb


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def extract_dataset_embeddings(
    model: torch.nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    value_scaling: str,
    time_strategy: str,
    channel_pool: str,
    err_value: float,
    max_points_per_series: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_channels, _ = x.shape
    channel_embeddings: Optional[np.ndarray] = None
    valid_channel = np.zeros((n_samples, n_channels), dtype=bool)
    base_batch_size = max(1, int(batch_size))

    progress = tqdm(total=n_channels, desc="Encoding channels", unit="ch") if tqdm is not None else None
    try:
        for channel_idx in range(n_channels):
            series_indices: List[int] = []
            series_payload: List[np.ndarray] = []
            for sample_idx in range(n_samples):
                times = t[sample_idx, 0] if t.ndim == 3 else t[sample_idx]
                lc = prepare_single_channel_lc(
                    x[sample_idx, channel_idx],
                    times,
                    value_scaling=value_scaling,
                    time_strategy=time_strategy,
                    err_value=err_value,
                    max_points_per_series=max_points_per_series,
                )
                if lc is None:
                    continue
                series_indices.append(sample_idx)
                series_payload.append(lc)

            if not series_payload:
                if progress is not None:
                    progress.update(1)
                continue

            current_batch_size = base_batch_size
            start = 0
            while start < len(series_payload):
                stop = min(start + current_batch_size, len(series_payload))
                batch_idx = series_indices[start:stop]
                batch_payload = series_payload[start:stop]
                batch_lc = collate_lc_batch(batch_payload)
                try:
                    emb = encode_lc_batch(model, batch_lc, device=device)
                except RuntimeError as exc:
                    if device.type == "cuda" and is_cuda_oom(exc):
                        torch.cuda.empty_cache()
                        if current_batch_size <= 1:
                            raise
                        current_batch_size = max(1, current_batch_size // 2)
                        continue
                    raise
                if channel_embeddings is None:
                    channel_embeddings = np.full(
                        (n_samples, n_channels, emb.shape[1]),
                        np.nan,
                        dtype=np.float32,
                    )
                channel_embeddings[np.asarray(batch_idx), channel_idx] = emb
                valid_channel[np.asarray(batch_idx), channel_idx] = True
                start = stop

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if channel_embeddings is None:
        raise RuntimeError("No valid channels could be encoded for this dataset.")

    if channel_pool == "mean":
        features = np.nanmean(channel_embeddings, axis=1)
    elif channel_pool == "concat":
        filled = np.nan_to_num(channel_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        features = filled.reshape(n_samples, -1)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported channel_pool={channel_pool!r}")

    valid_sample = valid_channel.any(axis=1) & np.all(np.isfinite(features), axis=1)
    return features, valid_sample


def evaluate_classifier(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)

    if name == "logistic":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "mlp":
        hidden_dim = max(64, min(512, xtr.shape[1] // 2 if xtr.shape[1] > 1 else 64))
        batch_size = max(8, min(256, xtr.shape[0]))
        class_counts = np.bincount(y_train)
        use_early_stopping = bool(xtr.shape[0] >= 32 and class_counts.size > 1 and class_counts.min() >= 2)
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            solver="adam",
            alpha=1.0e-4,
            learning_rate_init=1.0e-3,
            batch_size=batch_size,
            max_iter=300,
            early_stopping=use_early_stopping,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=seed,
        )
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=max(1, min(5, int(xtr.shape[0]))))
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported classifier={name!r}")

    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
    }


def compute_aggregate_metrics(summary: Dict[str, Any], classifiers: Sequence[str]) -> Dict[str, Dict[str, float]]:
    aggregate: Dict[str, Dict[str, float]] = {}
    datasets = summary.get("datasets", [])
    for classifier_name in classifiers:
        metrics = [entry.get("metrics", {}).get(classifier_name) for entry in datasets]
        metrics = [metric for metric in metrics if isinstance(metric, dict)]
        if not metrics:
            continue
        aggregate[classifier_name] = {
            "mean_accuracy": float(np.mean([metric["accuracy"] for metric in metrics])),
            "mean_macro_f1": float(np.mean([metric["macro_f1"] for metric in metrics])),
            "mean_macro_precision": float(np.mean([metric["macro_precision"] for metric in metrics])),
            "mean_macro_recall": float(np.mean([metric["macro_recall"] for metric in metrics])),
            "n_datasets": int(len(metrics)),
        }
    return aggregate


def split_train_test(x: np.ndarray, y: np.ndarray, split: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_mask = split == "test"
    train_mask = ~test_mask
    if not bool(test_mask.any()):
        raise RuntimeError("Dataset does not expose a 'test' split.")
    if int(train_mask.sum()) <= 1:
        raise RuntimeError("Dataset does not expose enough non-test samples for training.")
    return x[train_mask], y[train_mask], x[test_mask], y[test_mask]


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    classifiers = [name.strip() for name in args.classifiers.split(",") if name.strip()]
    dataset_names = resolve_requested_datasets(args.datasets)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg_max_points = int((cfg.get("pyrregular_adaptation", {}) or {}).get("max_points_per_series", 0))
    effective_max_points = int(args.max_points_per_series) if int(args.max_points_per_series) > 0 else cfg_max_points

    model = build_benchmark_model(
        cfg,
        args.ckpt,
        device=device,
        strict=bool(args.strict),
        num_gpus=int(args.num_gpus),
        batch_size=int(args.batch_size),
    )

    summary: Dict[str, Any] = {
        "config": args.config,
        "ckpt": args.ckpt,
        "requested_datasets": list(args.datasets),
        "resolved_datasets": dataset_names,
        "datasets": [],
        "failures": [],
        "value_scaling": args.value_scaling,
        "time_strategy": args.time_strategy,
        "channel_pool": args.channel_pool,
        "classifiers": classifiers,
        "seed": int(args.seed),
        "device": str(device),
        "num_gpus": int(args.num_gpus),
        "max_points_per_series": int(effective_max_points),
    }
    summary_path = out_dir / "summary.json"

    for dataset_name in dataset_names:
        dataset_canonical_name = ensure_dataset_name(dataset_name)
        dataset_out_dir = out_dir / Path(dataset_canonical_name).stem
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        report_path = dataset_out_dir / "report.json"

        if bool(args.skip_existing) and report_path.exists():
            dataset_result = json.loads(report_path.read_text(encoding="utf-8"))
            summary["datasets"].append(dataset_result)
            print(f"[Skip] {dataset_canonical_name} -> existing report at {report_path}", flush=True)
            write_summary(summary_path, summary)
            continue

        try:
            dataset_file = ensure_dataset_local(
                dataset_name,
                cache_dir,
                download_workers=int(args.download_workers),
                download_chunk_mb=int(args.download_chunk_mb),
            )
            da, x, t, y_raw, split = load_pyrregular_dense(dataset_file)
            task_cfg = da.attrs.get("configs", {}).get("default", {})
            if task_cfg.get("task", "classification") != "classification":
                raise RuntimeError(f"Unsupported task={task_cfg.get('task')!r}; only classification is handled.")

            features, valid_sample = extract_dataset_embeddings(
                model,
                x,
                t,
                device=device,
                batch_size=int(args.batch_size),
                value_scaling=args.value_scaling,
                time_strategy=args.time_strategy,
                channel_pool=args.channel_pool,
                err_value=float(args.err_value),
                max_points_per_series=int(effective_max_points),
            )

            y_kept = y_raw[valid_sample]
            split_kept = split[valid_sample]
            x_kept = features[valid_sample]

            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y_kept)
            x_train, y_train, x_test, y_test = split_train_test(x_kept, y_enc, split_kept)

            dataset_result = {
                "dataset": dataset_canonical_name,
                "title": str(da.attrs.get("title", dataset_canonical_name)),
                "source": str(da.attrs.get("source", "")),
                "shape_before_filter": {
                    "n_samples": int(x.shape[0]),
                    "n_channels": int(x.shape[1]),
                    "seq_len": int(x.shape[2]),
                },
                "shape_after_filter": {
                    "n_samples": int(x_kept.shape[0]),
                    "feature_dim": int(x_kept.shape[1]),
                },
                "max_points_per_series": int(effective_max_points),
                "dropped_samples": int((~valid_sample).sum()),
                "train_samples": int(x_train.shape[0]),
                "test_samples": int(x_test.shape[0]),
                "n_classes": int(len(label_encoder.classes_)),
                "label_classes": label_encoder.classes_.tolist(),
                "metrics": {},
            }

            for classifier_name in classifiers:
                metrics = evaluate_classifier(
                    classifier_name,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed=int(args.seed),
                )
                dataset_result["metrics"][classifier_name] = metrics

            report_path.write_text(json.dumps(dataset_result, indent=2), encoding="utf-8")
            summary["datasets"].append(dataset_result)
            summary["aggregate"] = compute_aggregate_metrics(summary, classifiers)
            print(json.dumps(dataset_result, indent=2), flush=True)
            write_summary(summary_path, summary)
        except Exception as exc:
            failure = {
                "dataset": dataset_canonical_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["failures"].append(failure)
            summary["aggregate"] = compute_aggregate_metrics(summary, classifiers)
            write_summary(summary_path, summary)
            print(f"[Fail] {dataset_canonical_name}: {type(exc).__name__}: {exc}", flush=True)
            if bool(args.fail_fast):
                raise

    summary["aggregate"] = compute_aggregate_metrics(summary, classifiers)
    write_summary(summary_path, summary)
    print(f"[OK] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
