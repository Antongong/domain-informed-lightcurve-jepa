#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a detached head for sin_injection_recovery_single_freq using a frozen
pretrained numeric encoder.

The script does two stages:
  1. Convert every light curve to a detached feature vector and save one npz per
     split for later reuse.
  2. Train a linear or MLP regression head on the cached vectors and record
     train/val/test inference CSVs.

Targets are:
  log_P, A, phi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_ddp_numeric import aggregate_views, build_model


DEFAULT_INPUT_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/sin_injection_recovery_single_freq"
)
DEFAULT_PRETRAIN_RUN = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/runs/EXP8_no_group_branch"
)
TARGET_NAMES = ["log_P", "A", "phi"]
TWO_PI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train detached log_P/A/phi recovery head from pretrained vectors."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pretrain-run", type=Path, default=DEFAULT_PRETRAIN_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PRETRAIN_RUN.parent
        / "EXP8_no_group_branch_sin_logP_A_phi_recovery",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--sigma-recovery",
        type=Path,
        default=None,
        help="Deprecated compatibility option; sigma recovery is no longer used.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Debug cap for vectorization/training. 0 means use all rows.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--views-order",
        type=str,
        default=None,
        help="Comma-separated pretrained views to concatenate. Defaults to checkpoint loss.views_order.",
    )
    parser.add_argument("--repr-mode", choices=["concat", "mean"], default="concat")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--periodogram-size",
        type=int,
        default=512,
        help="Fixed period-grid size used while vectorizing.",
    )
    parser.add_argument(
        "--period-window",
        type=float,
        default=4.0,
        help="Deprecated compatibility option; ignored because true P is not used for vectorization.",
    )
    parser.add_argument("--period-min", type=float, default=2.5)
    parser.add_argument("--period-max", type=float, default=100.0)
    parser.add_argument("--periodogram-chunk-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint_model(pretrain_run: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_path = pretrain_run / "ckpt_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise RuntimeError(f"Checkpoint has no cfg entry: {ckpt_path}")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_for_id(sample_id: str) -> str:
    index = int(sample_id)
    if index < 14_000:
        return "train"
    if index < 16_000:
        return "val"
    return "test"


def read_targets(input_dir: Path, metadata_path: Path, eps: float) -> Dict[str, dict]:
    meta_rows = {row["id"]: row for row in load_csv_rows(metadata_path)}

    out: Dict[str, dict] = {}
    for sample_id, meta in meta_rows.items():
        period = max(float(meta["P"]), eps)
        amplitude = max(float(meta["A"]), eps)
        split = meta.get("split") or split_for_id(sample_id)
        out[sample_id] = {
            "id": sample_id,
            "split": split,
            "P": period,
            "A": amplitude,
            "log_P": math.log(period),
            "phi": float(meta["phi"]) % TWO_PI,
            "path": input_dir / "injection" / split / f"{sample_id}.csv",
        }
    return out


def load_light_curve(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    time = np.asarray(data["time"], dtype=np.float32)
    mag = np.asarray(data["mag"], dtype=np.float32)
    mag_err = np.asarray(data["mag_err"], dtype=np.float32)
    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0.0)
    if valid.sum() < 3:
        raise ValueError(f"Need at least 3 valid light-curve points in {path}")
    order = np.argsort(time[valid])
    lc = np.stack([time[valid][order], mag[valid][order], mag_err[valid][order]], axis=-1)
    return lc.astype(np.float32, copy=False)


def pad_light_curves(curves: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    batch = len(curves)
    max_len = max(int(c.shape[0]) for c in curves)
    out = torch.zeros((batch, max_len, 4), dtype=torch.float32, device=device)
    for i, curve in enumerate(curves):
        n = int(curve.shape[0])
        out[i, :n, :3] = torch.as_tensor(curve, dtype=torch.float32, device=device)
        out[i, :n, 3] = 1.0
    return out


def fixed_grid_periodogram(
    time: torch.Tensor,
    mag: torch.Tensor,
    mag_err: torch.Tensor,
    mask: torch.Tensor,
    *,
    k_periods: int,
    period_min: float,
    period_max: float,
    chunk_size: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k_periods <= 0:
        raise ValueError("--periodogram-size must be positive")
    if period_min <= 0.0 or period_max <= period_min:
        raise ValueError("--period-min must be > 0 and --period-max must be greater than --period-min")

    device = time.device
    dtype = torch.float32
    period_grid = torch.exp(
        torch.linspace(math.log(period_min), math.log(period_max), k_periods, device=device, dtype=dtype)
    )
    periods = period_grid.view(1, -1).expand(time.shape[0], -1)

    valid = mask.bool() & torch.isfinite(time) & torch.isfinite(mag) & torch.isfinite(mag_err) & (mag_err > 0)
    w = torch.zeros_like(mag, dtype=dtype)
    w[valid] = 1.0 / mag_err[valid].clamp_min(eps).square()
    W = w.sum(dim=1).clamp_min(eps)
    Y = (w * mag).sum(dim=1)
    YY = (w * mag * mag).sum(dim=1)
    var_y = (YY - (Y * Y) / W).clamp_min(eps)

    B, K = periods.shape
    power = torch.empty((B, K), dtype=dtype, device=device)
    t3 = time.unsqueeze(-1)
    y3 = mag.unsqueeze(-1)
    w3 = w.unsqueeze(-1)
    W1 = W.view(B, 1)
    Y1 = Y.view(B, 1)
    var_y1 = var_y.view(B, 1)

    for k0 in range(0, K, chunk_size):
        k1 = min(k0 + chunk_size, K)
        omega = (TWO_PI / periods[:, k0:k1].clamp_min(eps)).unsqueeze(1)
        phase = t3 * omega
        c = torch.cos(phase)
        s = torch.sin(phase)

        C = (w3 * c).sum(dim=1)
        S = (w3 * s).sum(dim=1)
        CC = (w3 * c * c).sum(dim=1)
        SS = (w3 * s * s).sum(dim=1)
        CS = (w3 * c * s).sum(dim=1)
        YC = (w3 * y3 * c).sum(dim=1)
        YS = (w3 * y3 * s).sum(dim=1)

        YC0 = YC - (Y1 * C) / W1
        YS0 = YS - (Y1 * S) / W1
        D = (CC * SS - CS * CS).clamp_min(eps)
        num = SS * YC0.square() + CC * YS0.square() - 2.0 * CS * YC0 * YS0
        power[:, k0:k1] = torch.nan_to_num(num / (D * var_y1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    return periods, power


def prepare_signal_inputs(
    raw_lc: torch.Tensor,
    *,
    periodogram_size: int,
    period_min: float,
    period_max: float,
    periodogram_chunk_size: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    time = raw_lc[..., 0]
    mag = raw_lc[..., 1]
    mag_err = raw_lc[..., 2].abs()
    mask = raw_lc[..., 3] > 0
    mask_f = mask.float()

    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_mag = (mag * mask_f).sum(dim=1, keepdim=True) / denom
    mag_centered = (mag - mean_mag) * mask_f
    time_masked = time * mask_f
    mag_err_masked = mag_err * mask_f

    periods, power = fixed_grid_periodogram(
        time_masked,
        mag_centered,
        mag_err.clamp_min(eps),
        mask,
        k_periods=periodogram_size,
        period_min=period_min,
        period_max=period_max,
        chunk_size=periodogram_chunk_size,
        eps=eps,
    )
    best_idx = torch.argmax(power, dim=1)
    best_period = periods.gather(1, best_idx.view(-1, 1)).squeeze(1)
    best_power = power.gather(1, best_idx.view(-1, 1)).squeeze(1)
    phase_period = best_period.view(-1, 1).clamp_min(eps)
    phase_time = (time_masked - phase_period * torch.floor(time_masked / phase_period)) * mask_f

    return {
        "lc": torch.stack([time_masked, mag_centered, mag_err_masked, mask_f], dim=-1),
        "periodogram": torch.stack([periods, torch.log10(power.clamp_min(eps))], dim=-1),
        "phase_folded_lc": torch.stack([phase_time, mag_centered, mag_err_masked, mask_f], dim=-1),
        "best_period": best_period,
        "best_power": best_power,
    }


def current_target_array(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([[row["log_P"], row["A"], row["phi"]] for row in rows], dtype=np.float32)


def cache_has_current_targets(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=False)
        if not {"target_names", "targets", "A", "P", "phi"}.issubset(data.files):
            return False
        names = [str(x) for x in data["target_names"]]
        return names == TARGET_NAMES and int(data["targets"].shape[1]) == len(TARGET_NAMES)
    except (OSError, ValueError, KeyError):
        return False


def rewrite_cached_targets(path: Path, rows: Sequence[dict], split: str) -> bool:
    try:
        data = np.load(path, allow_pickle=False)
        if "features" not in data.files or "ids" not in data.files:
            return False
        ids = data["ids"].astype(str)
        row_by_id = {str(row["id"]): row for row in rows}
        if len(ids) != len(rows) or any(sample_id not in row_by_id for sample_id in ids):
            return False

        ordered_rows = [row_by_id[sample_id] for sample_id in ids]
        np.savez_compressed(
            path,
            ids=ids,
            split=np.asarray([split] * len(ids)),
            features=data["features"].astype(np.float32),
            targets=current_target_array(ordered_rows),
            target_names=np.asarray(TARGET_NAMES),
            P=np.asarray([float(row["P"]) for row in ordered_rows], dtype=np.float32),
            A=np.asarray([float(row["A"]) for row in ordered_rows], dtype=np.float32),
            phi=np.asarray([float(row["phi"]) for row in ordered_rows], dtype=np.float32),
        )
        print(f"[vectorize:{split}] rewrote targets in existing {path}")
        return True
    except (OSError, ValueError, KeyError):
        return False


@torch.no_grad()
def vectorize_split(
    *,
    split: str,
    rows: Sequence[dict],
    model: nn.Module,
    device: torch.device,
    views_order: Sequence[str],
    repr_mode: str,
    output_path: Path,
    batch_size: int,
    periodogram_size: int,
    period_min: float,
    period_max: float,
    periodogram_chunk_size: int,
    eps: float,
) -> None:
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    extras: Dict[str, List[float]] = {
        "P": [],
        "A": [],
        "phi": [],
    }
    ids: List[str] = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        curves = [load_light_curve(Path(row["path"])) for row in chunk]
        raw_lc = pad_light_curves(curves, device=device)
        xdict = prepare_signal_inputs(
            raw_lc,
            periodogram_size=periodogram_size,
            period_min=period_min,
            period_max=period_max,
            periodogram_chunk_size=periodogram_chunk_size,
            eps=eps,
        )
        out = model._encode_prepared_inputs(xdict)
        feat = aggregate_views(out["embeddings"], list(views_order), mode=repr_mode, require_all=True)
        features.append(feat.detach().float().cpu().numpy())
        targets.append(current_target_array(chunk))
        ids.extend(str(row["id"]) for row in chunk)
        for row in chunk:
            extras["P"].append(float(row["P"]))
            extras["A"].append(float(row["A"]))
            extras["phi"].append(float(row["phi"]))
        print(f"[vectorize:{split}] {min(start + batch_size, len(rows))}/{len(rows)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray(ids),
        split=np.asarray([split] * len(ids)),
        features=np.concatenate(features, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        target_names=np.asarray(TARGET_NAMES),
        P=np.asarray(extras["P"], dtype=np.float32),
        A=np.asarray(extras["A"], dtype=np.float32),
        phi=np.asarray(extras["phi"], dtype=np.float32),
    )
    print(f"[vectorize:{split}] wrote {output_path}")


def build_vector_cache(args: argparse.Namespace, model: nn.Module, cfg: Dict[str, Any], device: torch.device) -> List[str]:
    input_dir = args.input_dir.expanduser().resolve()
    metadata_path = args.metadata or (input_dir / "injection_recovery_single_freq.csv")
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if args.sigma_recovery is not None:
        print("[config] ignoring deprecated --sigma-recovery; targets come from injected metadata log_P/A/phi")

    if args.views_order is None:
        views_order = list(cfg.get("loss", {}).get("views_order", ["raw", "periodogram", "group", "phase_folded"]))
    else:
        views_order = [v.strip() for v in args.views_order.split(",") if v.strip()]
    if not views_order:
        raise ValueError("At least one view is required")

    targets_by_id = read_targets(input_dir, metadata_path, eps=args.eps)
    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for sample_id in sorted(targets_by_id):
        row = targets_by_id[sample_id]
        rows_by_split[row["split"]].append(row)
    if args.max_samples_per_split > 0:
        rows_by_split = {
            split: rows[: args.max_samples_per_split] for split, rows in rows_by_split.items()
        }

    cache_dir = args.output_dir.expanduser().resolve() / "vectors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        out_path = cache_dir / f"{split}_vectors.npz"
        if out_path.exists() and not args.rebuild_cache and cache_has_current_targets(out_path):
            print(f"[vectorize:{split}] using existing {out_path}")
            continue
        vectorize_split(
            split=split,
            rows=rows,
            model=model,
            device=device,
            views_order=views_order,
            repr_mode=args.repr_mode,
            output_path=out_path,
            batch_size=args.feature_batch_size,
            periodogram_size=args.periodogram_size,
            period_min=args.period_min,
            period_max=args.period_max,
            periodogram_chunk_size=args.periodogram_chunk_size,
            eps=args.eps,
        )

    manifest = {
        "input_dir": str(input_dir),
        "metadata": str(metadata_path),
        "pretrain_run": str(args.pretrain_run.expanduser().resolve()),
        "views_order": views_order,
        "repr_mode": args.repr_mode,
        "target_names": TARGET_NAMES,
        "periodogram_size": int(args.periodogram_size),
        "period_min": float(args.period_min),
        "period_max": float(args.period_max),
        "max_samples_per_split": int(args.max_samples_per_split),
    }
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return views_order


class VectorDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.ids = data["ids"].astype(str)
        self.features = torch.from_numpy(data["features"].astype(np.float32))
        self.targets = torch.from_numpy(data["targets"].astype(np.float32))
        target_names = [str(x) for x in data["target_names"]]
        if target_names != TARGET_NAMES:
            raise RuntimeError(
                f"{path} has target_names={target_names}; expected {TARGET_NAMES}. "
                "Rebuild or rewrite the vector cache."
            )
        self.P = data["P"].astype(np.float32)
        self.A = data["A"].astype(np.float32)
        self.phi = data["phi"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[idx], self.targets[idx], int(idx)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, kind: str, hidden_dim: int, dropout: float):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_dim, len(TARGET_NAMES))
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(TARGET_NAMES)),
            )
        else:
            raise ValueError(f"Unsupported head kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        pred = torch.empty_like(raw)
        pred[:, 0] = raw[:, 0]
        pred[:, 1] = F.softplus(raw[:, 1]) + 1.0e-6
        pred[:, 2] = torch.remainder(raw[:, 2], TWO_PI)
        return pred


def angular_delta(pred_phi: torch.Tensor, true_phi: torch.Tensor) -> torch.Tensor:
    return torch.remainder(pred_phi - true_phi + math.pi, TWO_PI) - math.pi


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff[:, 2] = angular_delta(pred[:, 2], target[:, 2])
    return torch.mean(torch.square(diff))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    sx = float(np.std(x[ok]))
    sy = float(np.std(y[ok]))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


@torch.no_grad()
def evaluate_head(head: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    head.eval()
    losses: List[float] = []
    pred_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []
    for xb, yb, _ in loader:
        xb = xb.to(device=device, non_blocking=True)
        yb = yb.to(device=device, non_blocking=True)
        pred_batch = head(xb)
        losses.append(float(regression_loss(pred_batch, yb).detach().cpu().item()) * xb.shape[0])
        pred_all.append(pred_batch.detach().cpu().numpy())
        true_all.append(yb.detach().cpu().numpy())
    pred = np.concatenate(pred_all, axis=0)
    true = np.concatenate(true_all, axis=0)
    n = max(1, true.shape[0])
    metrics = {"mse": float(sum(losses) / n)}
    for i, name in enumerate(TARGET_NAMES):
        if name == "phi":
            diff = ((pred[:, i] - true[:, i] + np.pi) % (2.0 * np.pi)) - np.pi
        else:
            diff = pred[:, i] - true[:, i]
        metrics[f"{name}/mae"] = float(np.mean(np.abs(diff)))
        metrics[f"{name}/rmse"] = float(np.sqrt(np.mean(np.square(diff))))
        metrics[f"{name}/pearson"] = pearson(true[:, i], pred[:, i])
    return metrics


@torch.no_grad()
def write_inference_csv(head: nn.Module, dataset: VectorDataset, path: Path, device: torch.device, batch_size: int) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    head.eval()
    rows: List[dict] = []
    for xb, yb, idx in loader:
        xb = xb.to(device=device, non_blocking=True)
        yb_device = yb.to(device=device, non_blocking=True)
        pred_tensor = head(xb)
        err_tensor = pred_tensor - yb_device
        err_tensor[:, 2] = angular_delta(pred_tensor[:, 2], yb_device[:, 2])
        per_target_sqerr = torch.square(err_tensor)
        per_sample_mse = per_target_sqerr.mean(dim=1)
        pred = pred_tensor.detach().cpu().numpy()
        true = yb.numpy()
        sqerr = per_target_sqerr.detach().cpu().numpy()
        mse = per_sample_mse.detach().cpu().numpy()
        for j, original_idx in enumerate(idx.numpy().tolist()):
            log_p_pred = float(pred[j, 0])
            a_pred = float(pred[j, 1])
            phi_pred = float(pred[j, 2])
            rows.append(
                {
                    "id": str(dataset.ids[original_idx]),
                    "P_true": f"{float(dataset.P[original_idx]):.12g}",
                    "A_true": f"{float(dataset.A[original_idx]):.12g}",
                    "phi_true": f"{float(dataset.phi[original_idx]):.12g}",
                    "log_P_true": f"{float(true[j, 0]):.12g}",
                    "log_P_pred": f"{log_p_pred:.12g}",
                    "P_pred": f"{float(math.exp(log_p_pred)):.12g}",
                    "A_pred": f"{a_pred:.12g}",
                    "phi_pred": f"{phi_pred:.12g}",
                    "sqerr_log_P": f"{float(sqerr[j, 0]):.12g}",
                    "sqerr_A": f"{float(sqerr[j, 1]):.12g}",
                    "sqerr_phi": f"{float(sqerr[j, 2]):.12g}",
                    "mse": f"{float(mse[j]):.12g}",
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[inference] wrote {path}")


def train_head(args: argparse.Namespace, device: torch.device) -> None:
    out_dir = args.output_dir.expanduser().resolve()
    cache_dir = out_dir / "vectors"
    train_ds = VectorDataset(cache_dir / "train_vectors.npz")
    val_ds = VectorDataset(cache_dir / "val_vectors.npz")
    test_ds = VectorDataset(cache_dir / "test_vectors.npz")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_loaders = {
        "train": DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=False),
        "val": DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=args.train_batch_size, shuffle=False),
    }

    head = RegressionHead(
        in_dim=int(train_ds.features.shape[1]),
        kind=args.head,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out_dir / "head_train_metrics.jsonl"
    ckpt_path = out_dir / "best_head.pt"
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        seen = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device=device, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = head(xb)
            loss = regression_loss(pred, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu().item()) * xb.shape[0]
            seen += xb.shape[0]

        record: Dict[str, Any] = {"epoch": epoch, "train_batch_mse": total / max(1, seen)}
        for split, loader in eval_loaders.items():
            metrics = evaluate_head(head, loader, device)
            record.update({f"{split}/{k}": v for k, v in metrics.items()})
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if epoch % args.log_every == 0:
            print(
                f"[epoch {epoch:04d}] train_mse={record['train/mse']:.6f} "
                f"val_mse={record['val/mse']:.6f} test_mse={record['test/mse']:.6f}"
            )
        if record["val/mse"] < best_val:
            best_val = float(record["val/mse"])
            torch.save(
                {
                    "head": head.state_dict(),
                    "args": vars(args),
                    "target_names": TARGET_NAMES,
                    "feature_dim": int(train_ds.features.shape[1]),
                    "best_epoch": epoch,
                    "best_val_mse": best_val,
                },
                ckpt_path,
            )

    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(best["head"])
    for split, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        write_inference_csv(
            head=head,
            dataset=ds,
            path=out_dir / f"inference_{split}.csv",
            device=device,
            batch_size=args.train_batch_size,
        )
    print(f"[done] best_epoch={best['best_epoch']} best_val_mse={best['best_val_mse']:.6f}")


def main() -> None:
    args = parse_args()
    if args.feature_batch_size <= 0 or args.train_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    model, cfg = load_checkpoint_model(args.pretrain_run.expanduser().resolve(), device)
    views_order = build_vector_cache(args, model, cfg, device)
    print(f"[config] views_order={views_order} repr_mode={args.repr_mode} device={device}")
    train_head(args, device)


if __name__ == "__main__":
    main()
