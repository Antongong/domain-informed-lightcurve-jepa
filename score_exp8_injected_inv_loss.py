#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from extract_starembed_embeddings import build_model_from_config, load_model_from_ckpt
from visualize_exp8_starembed_inv_loss_anom import collect_split_losses


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed"
DEFAULT_INJECTED_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed_injected_anomalies"
DEFAULT_OUT_DIR = RUN_DIR / "starembed_injected_lejepa_pred_term"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score EXP8 LeJEPA prediction/invariance term on StarEmbed test and injected anomaly test sets."
    )
    parser.add_argument("--config", type=Path, default=RUN_DIR / "config_used.yaml")
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "ckpt_final.pt")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--injected-root", type=Path, default=DEFAULT_INJECTED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def write_scores_csv(path: Path, scores_by_name: Dict[str, Dict[str, np.ndarray]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "index", "lejepa_pred_term", "lejepa_pred_term_g", "lejepa_pred_term_r", "y", "y_str", "path"])
        for name, scores in scores_by_name.items():
            for idx in range(scores["loss"].shape[0]):
                writer.writerow(
                    [
                        name,
                        idx,
                        f"{float(scores['loss'][idx]):.10g}",
                        f"{float(scores['loss_g'][idx]):.10g}",
                        f"{float(scores['loss_r'][idx]):.10g}",
                        int(scores["y"][idx]),
                        str(scores["y_str"][idx]),
                        str(scores["path"][idx]),
                    ]
                )


def summary_stats(scores_by_name: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for name, scores in scores_by_name.items():
        loss = scores["loss"]
        out[name] = {
            "n": int(loss.shape[0]),
            "mean": float(np.mean(loss)),
            "std": float(np.std(loss)),
            "median": float(np.median(loss)),
            "p90": float(np.quantile(loss, 0.90)),
            "p95": float(np.quantile(loss, 0.95)),
            "p99": float(np.quantile(loss, 0.99)),
            "min": float(np.min(loss)),
            "max": float(np.max(loss)),
        }
    return out


def plot_overlay_hist(
    *,
    scores_by_name: Dict[str, Dict[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    bins: int,
) -> None:
    names = list(scores_by_name.keys())
    losses = [scores_by_name[name]["loss"] for name in names]
    all_loss = np.concatenate(losses, axis=0)
    edges = np.histogram_bin_edges(all_loss, bins=int(bins))

    labels = {
        "original_test": "Original test",
        "sudden_jump": "Injected sudden jump",
        "weather_anomaly": "Injected weather",
    }
    colors = {
        "original_test": "#4C78A8",
        "sudden_jump": "#F58518",
        "weather_anomaly": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    for name, loss in zip(names, losses):
        ax.hist(
            loss,
            bins=edges,
            label=f"{labels.get(name, name)} (n={loss.shape[0]})",
            color=colors.get(name, None),
            alpha=0.42,
            histtype="stepfilled",
            linewidth=1.2,
            edgecolor=colors.get(name, None),
        )

    ax.set_title("EXP8 StarEmbed LEJEPA Prediction Term")
    ax.set_xlabel("Per-target LeJEPA prediction term")
    ax.set_ylabel("Number of samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)

    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    views_order = list(cfg["loss"]["views_order"])
    align_to = cfg.get("loss", {}).get("lejepa", {}).get("align_to", None)
    device = torch.device(args.device)

    model = build_model_from_config(cfg)
    model.to(device)
    model.eval()
    load_model_from_ckpt(model, str(args.ckpt), strict=bool(args.strict))

    datasets = {
        "original_test": (args.data_root, "test"),
        "sudden_jump": (args.injected_root / "sudden_jump", "test"),
        "weather_anomaly": (args.injected_root / "weather_anomaly", "test"),
    }

    scores_by_name: Dict[str, Dict[str, np.ndarray]] = {}
    for name, (data_root, split) in datasets.items():
        scores_by_name[name] = score_dataset(
            model=model,
            data_root=data_root,
            split=split,
            views_order=views_order,
            align_to=align_to,
            args=args,
            device=device,
        )

    payload: Dict[str, np.ndarray] = {}
    for name, scores in scores_by_name.items():
        for key, value in scores.items():
            payload[f"{name}_{key}"] = value
    np.savez_compressed(args.out_dir / "exp8_injected_lejepa_pred_term_scores.npz", **payload)
    write_scores_csv(args.out_dir / "exp8_injected_lejepa_pred_term_scores.csv", scores_by_name)

    summary: Dict[str, Any] = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "data_root": str(args.data_root),
        "injected_root": str(args.injected_root),
        "views_order": views_order,
        "align_to": align_to,
        "score": (
            "mean of g-band and r-band per-sample LeJEPA prediction/invariance term "
            "from losses.LeJEPALoss inv_loss; not reciprocal loss"
        ),
        "datasets": {
            name: {"data_root": str(root), "split": split}
            for name, (root, split) in datasets.items()
        },
        "stats": summary_stats(scores_by_name),
    }
    (args.out_dir / "exp8_injected_lejepa_pred_term_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_overlay_hist(
        scores_by_name=scores_by_name,
        out_png=args.out_dir / "exp8_injected_lejepa_pred_term_hist.png",
        out_pdf=args.out_dir / "exp8_injected_lejepa_pred_term_hist.pdf",
        bins=int(args.bins),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
