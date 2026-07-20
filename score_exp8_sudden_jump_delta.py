#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml

from extract_starembed_embeddings import build_model_from_config, load_model_from_ckpt
from visualize_exp8_starembed_inv_loss_anom import collect_split_losses


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed"
DEFAULT_INJECTED_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed_injected_anomalies"
DEFAULT_OUT_DIR = RUN_DIR / "starembed_sudden_jump_delta_lejepa_pred_term"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score EXP8 LeJEPA prediction term for original test and sudden-jump injections, then compute deltas."
    )
    parser.add_argument("--config", type=Path, default=RUN_DIR / "config_used.yaml")
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "ckpt_final.pt")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--injected-root", type=Path, default=DEFAULT_INJECTED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def rel_key(path: str | Path, split_dir: Path) -> str:
    return str(Path(path).resolve().relative_to(split_dir.resolve()))


def load_injection_info(path: Path) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            info[str(record["relative_path"])] = record
    return info


def score_dataset(
    *,
    model: torch.nn.Module,
    data_root: Path,
    split: str,
    views_order: Sequence[str],
    align_to: str | None,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    return collect_split_losses(
        model=model,
        data_root=data_root,
        split=split,
        views_order=views_order,
        align_to=align_to,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        limit=int(args.limit),
    )


def summarize(values: np.ndarray) -> Dict[str, float | int]:
    return {
        "n": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "frac_positive": float(np.mean(values > 0.0)),
    }


def write_delta_csv(
    path: Path,
    *,
    rows: Sequence[Dict[str, Any]],
) -> None:
    fieldnames = [
        "relative_path",
        "sourceid",
        "row_idx",
        "class_str",
        "y",
        "original_lejepa_pred_term",
        "sudden_jump_lejepa_pred_term",
        "delta_lejepa_pred_term",
        "original_lejepa_pred_term_g",
        "sudden_jump_lejepa_pred_term_g",
        "delta_lejepa_pred_term_g",
        "original_lejepa_pred_term_r",
        "sudden_jump_lejepa_pred_term_r",
        "delta_lejepa_pred_term_r",
        "jump_q",
        "jump_delta_mag_abs",
        "jump_delta_mag",
        "jump_sign",
        "g_start_rank",
        "r_start_rank",
        "g_n_modified",
        "r_n_modified",
        "original_path",
        "injected_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    original_split_dir = args.data_root / "test"
    sudden_root = args.injected_root / "sudden_jump"
    sudden_split_dir = sudden_root / "test"
    injection_info_path = sudden_root / "injection_info.jsonl"

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    views_order = list(cfg["loss"]["views_order"])
    align_to = cfg.get("loss", {}).get("lejepa", {}).get("align_to", None)

    device = torch.device(args.device)
    model = build_model_from_config(cfg)
    model.to(device)
    model.eval()
    load_model_from_ckpt(model, str(args.ckpt), strict=bool(args.strict))

    original = score_dataset(
        model=model,
        data_root=args.data_root,
        split="test",
        views_order=views_order,
        align_to=align_to,
        args=args,
        device=device,
    )
    sudden = score_dataset(
        model=model,
        data_root=sudden_root,
        split="test",
        views_order=views_order,
        align_to=align_to,
        args=args,
        device=device,
    )

    original_by_rel = {
        rel_key(path, original_split_dir): idx
        for idx, path in enumerate(original["path"])
    }
    sudden_by_rel = {
        rel_key(path, sudden_split_dir): idx
        for idx, path in enumerate(sudden["path"])
    }
    common_keys = [key for key in original_by_rel if key in sudden_by_rel]
    if len(common_keys) != len(original_by_rel) or len(common_keys) != len(sudden_by_rel):
        raise RuntimeError(
            f"Matched {len(common_keys)} samples, original={len(original_by_rel)}, sudden={len(sudden_by_rel)}"
        )

    injection_info = load_injection_info(injection_info_path)
    rows = []
    delta = np.empty((len(common_keys),), dtype=np.float32)
    delta_g = np.empty_like(delta)
    delta_r = np.empty_like(delta)
    original_loss = np.empty_like(delta)
    sudden_loss = np.empty_like(delta)

    for out_idx, key in enumerate(common_keys):
        oi = original_by_rel[key]
        si = sudden_by_rel[key]
        info = injection_info.get(key, {})
        inj = info.get("injection", {})
        bands = inj.get("bands", {})

        original_loss[out_idx] = float(original["loss"][oi])
        sudden_loss[out_idx] = float(sudden["loss"][si])
        delta[out_idx] = sudden_loss[out_idx] - original_loss[out_idx]
        delta_g[out_idx] = float(sudden["loss_g"][si]) - float(original["loss_g"][oi])
        delta_r[out_idx] = float(sudden["loss_r"][si]) - float(original["loss_r"][oi])

        rows.append(
            {
                "relative_path": key,
                "sourceid": info.get("sourceid", ""),
                "row_idx": info.get("row_idx", ""),
                "class_str": str(original["y_str"][oi]),
                "y": int(original["y"][oi]),
                "original_lejepa_pred_term": f"{float(original['loss'][oi]):.10g}",
                "sudden_jump_lejepa_pred_term": f"{float(sudden['loss'][si]):.10g}",
                "delta_lejepa_pred_term": f"{float(delta[out_idx]):.10g}",
                "original_lejepa_pred_term_g": f"{float(original['loss_g'][oi]):.10g}",
                "sudden_jump_lejepa_pred_term_g": f"{float(sudden['loss_g'][si]):.10g}",
                "delta_lejepa_pred_term_g": f"{float(delta_g[out_idx]):.10g}",
                "original_lejepa_pred_term_r": f"{float(original['loss_r'][oi]):.10g}",
                "sudden_jump_lejepa_pred_term_r": f"{float(sudden['loss_r'][si]):.10g}",
                "delta_lejepa_pred_term_r": f"{float(delta_r[out_idx]):.10g}",
                "jump_q": inj.get("q", ""),
                "jump_delta_mag_abs": inj.get("delta_mag_abs", ""),
                "jump_delta_mag": inj.get("delta_mag", ""),
                "jump_sign": inj.get("sign", ""),
                "g_start_rank": bands.get("g", {}).get("start_rank", ""),
                "r_start_rank": bands.get("r", {}).get("start_rank", ""),
                "g_n_modified": bands.get("g", {}).get("n_modified", ""),
                "r_n_modified": bands.get("r", {}).get("n_modified", ""),
                "original_path": str(original["path"][oi]),
                "injected_path": str(sudden["path"][si]),
            }
        )

    np.savez_compressed(
        args.out_dir / "exp8_sudden_jump_delta_lejepa_pred_term_scores.npz",
        relative_path=np.asarray(common_keys, dtype=object),
        original_lejepa_pred_term=original_loss,
        sudden_jump_lejepa_pred_term=sudden_loss,
        delta_lejepa_pred_term=delta,
        delta_lejepa_pred_term_g=delta_g,
        delta_lejepa_pred_term_r=delta_r,
    )
    write_delta_csv(args.out_dir / "exp8_sudden_jump_delta_lejepa_pred_term_scores.csv", rows=rows)

    jump_abs = np.asarray([float(row["jump_delta_mag_abs"]) for row in rows], dtype=np.float32)
    summary: Dict[str, Any] = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "data_root": str(args.data_root),
        "sudden_jump_root": str(sudden_root),
        "views_order": views_order,
        "align_to": align_to,
        "score": "LeJEPA prediction/invariance term from losses.LeJEPALoss inv_loss; delta = sudden_jump - original",
        "delta_stats": summarize(delta),
        "delta_g_stats": summarize(delta_g),
        "delta_r_stats": summarize(delta_r),
        "original_stats": summarize(original_loss),
        "sudden_jump_stats": summarize(sudden_loss),
        "jump_delta_mag_abs_stats": summarize(jump_abs),
        "corr_delta_vs_jump_delta_mag_abs": float(np.corrcoef(delta.astype(np.float64), jump_abs.astype(np.float64))[0, 1]),
    }
    (args.out_dir / "exp8_sudden_jump_delta_lejepa_pred_term_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
