#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from extract_starembed_embeddings import (
    PrecomputedStarEmbedPairDataset,
    build_model_from_config,
    load_model_from_ckpt,
    move_to_device,
    precomputed_starembed_pair_collate,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed"
DEFAULT_OUT_DIR = RUN_DIR / "starembed_inv_loss_anom_detection"


def per_sample_inv_loss(
    proj_dict: Dict[str, torch.Tensor],
    views_order: Sequence[str],
    *,
    align_to: str | None = None,
) -> torch.Tensor:
    missing = [view for view in views_order if view not in proj_dict]
    if missing:
        raise KeyError(f"missing views in model projections: {missing}")

    if align_to:
        if align_to not in proj_dict:
            raise KeyError(f"align_to={align_to!r} is not present in model projections")
        anchor = proj_dict[align_to].float()
        aligned = [proj_dict[view].float() for view in views_order if view != align_to]
        if not aligned:
            return torch.zeros(anchor.shape[0], device=anchor.device, dtype=torch.float32)
        return torch.stack([(anchor - z).square().mean(dim=-1) for z in aligned], dim=0).mean(dim=0)

    proj_seq = torch.stack([proj_dict[view].float() for view in views_order], dim=0)
    mean = proj_seq.mean(dim=0, keepdim=True)
    return (proj_seq - mean).square().mean(dim=(0, 2))


@torch.no_grad()
def collect_split_losses(
    *,
    model: torch.nn.Module,
    data_root: Path,
    split: str,
    views_order: Sequence[str],
    align_to: str | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    limit: int,
) -> Dict[str, np.ndarray]:
    dataset = PrecomputedStarEmbedPairDataset(data_root, split, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=precomputed_starembed_pair_collate,
        persistent_workers=(num_workers > 0),
    )

    losses: List[np.ndarray] = []
    losses_g: List[np.ndarray] = []
    losses_r: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    y_str_parts: List[np.ndarray] = []
    path_parts: List[str] = []

    progress = tqdm(total=len(dataset), desc=f"Scoring {split}", unit="target") if tqdm is not None else None
    try:
        for batch in loader:
            band_losses = []
            for band in ("g", "r"):
                out = model(move_to_device({"X": batch[band]["X"]}, device))
                band_loss = per_sample_inv_loss(
                    out["projections"],
                    views_order,
                    align_to=align_to,
                )
                band_losses.append(band_loss)

            loss_g, loss_r = band_losses
            loss = 0.5 * (loss_g + loss_r)

            losses.append(loss.detach().cpu().numpy().astype(np.float32))
            losses_g.append(loss_g.detach().cpu().numpy().astype(np.float32))
            losses_r.append(loss_r.detach().cpu().numpy().astype(np.float32))
            y_parts.append(batch["Y"]["star_class"].detach().cpu().numpy().astype(np.int64))
            y_str_parts.append(np.asarray(batch["Y"]["star_class_str"], dtype=object))
            path_parts.extend(batch["paths"])

            if progress is not None:
                progress.update(int(loss.shape[0]))
    finally:
        if progress is not None:
            progress.close()

    return {
        "loss": np.concatenate(losses, axis=0),
        "loss_g": np.concatenate(losses_g, axis=0),
        "loss_r": np.concatenate(losses_r, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "y_str": np.concatenate(y_str_parts, axis=0),
        "path": np.asarray(path_parts, dtype=object),
    }


def top_percent_purity(test_loss: np.ndarray, anom_loss: np.ndarray, percents: Sequence[float]) -> List[Dict[str, Any]]:
    losses = np.concatenate([test_loss, anom_loss], axis=0)
    is_anom = np.concatenate(
        [np.zeros(test_loss.shape[0], dtype=bool), np.ones(anom_loss.shape[0], dtype=bool)],
        axis=0,
    )
    order = np.argsort(losses)[::-1]

    rows: List[Dict[str, Any]] = []
    for pct in percents:
        k = max(1, int(np.ceil(losses.shape[0] * float(pct) / 100.0)))
        cutoff = float(losses[order[k - 1]])
        selected = losses >= cutoff
        selected_count = int(selected.sum())
        anom_count = int(is_anom[selected].sum())
        test_count = selected_count - anom_count
        rows.append(
            {
                "top_percent": float(pct),
                "threshold": cutoff,
                "selected_count": selected_count,
                "anom_count": anom_count,
                "test_count": test_count,
                "purity": float(anom_count / selected_count) if selected_count else 0.0,
            }
        )
    return rows


def write_scores_csv(path: Path, scores_by_split: Dict[str, Dict[str, np.ndarray]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "index", "inv_loss", "inv_loss_g", "inv_loss_r", "y", "y_str", "path"])
        for split, scores in scores_by_split.items():
            for idx in range(scores["loss"].shape[0]):
                writer.writerow(
                    [
                        split,
                        idx,
                        f"{float(scores['loss'][idx]):.10g}",
                        f"{float(scores['loss_g'][idx]):.10g}",
                        f"{float(scores['loss_r'][idx]):.10g}",
                        int(scores["y"][idx]),
                        str(scores["y_str"][idx]),
                        str(scores["path"][idx]),
                    ]
                )


def plot_histogram(
    *,
    test_loss: np.ndarray,
    anom_loss: np.ndarray,
    purity_rows: Sequence[Dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
    bins: int,
) -> None:
    all_loss = np.concatenate([test_loss, anom_loss], axis=0)
    edges = np.histogram_bin_edges(all_loss, bins=bins)

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    ax.hist(
        [test_loss, anom_loss],
        bins=edges,
        label=[f"Test (n={test_loss.shape[0]})", f"Anom (n={anom_loss.shape[0]})"],
        color=["#4C78A8", "#F58518"],
        alpha=0.78,
        histtype="bar",
    )

    line_styles = {1.0: "#B00020", 5.0: "#6A3D9A", 10.0: "#00897B"}
    y_max = ax.get_ylim()[1]
    for row in purity_rows:
        pct = float(row["top_percent"])
        color = line_styles.get(pct, "#333333")
        ax.axvline(row["threshold"], color=color, linestyle="--", linewidth=1.6)
        ax.text(
            row["threshold"],
            y_max * 0.98,
            f"Top {pct:g}%\nthr={row['threshold']:.3f}\npurity={row['purity']:.3f}",
            color=color,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )

    ax.set_title("EXP8 StarEmbed JEPA Inverse Loss: Test vs Anom")
    ax.set_xlabel("Per-target inverse loss")
    ax.set_ylabel("Number of samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)

    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EXP8 per-target JEPA inverse losses for StarEmbed test/anom and plot anomaly purity."
    )
    parser.add_argument("--config", type=Path, default=RUN_DIR / "config_used.yaml")
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "ckpt_final.pt")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


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

    scores_by_split: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("test", "anom"):
        scores_by_split[split] = collect_split_losses(
            model=model,
            data_root=args.data_root,
            split=split,
            views_order=views_order,
            align_to=align_to,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            limit=int(args.limit),
        )

    purity_rows = top_percent_purity(
        scores_by_split["test"]["loss"],
        scores_by_split["anom"]["loss"],
        percents=(1.0, 5.0, 10.0),
    )

    npz_payload: Dict[str, np.ndarray] = {}
    for split, scores in scores_by_split.items():
        for key, value in scores.items():
            npz_payload[f"{split}_{key}"] = value
    np.savez_compressed(args.out_dir / "exp8_starembed_inv_loss_scores.npz", **npz_payload)
    write_scores_csv(args.out_dir / "exp8_starembed_inv_loss_scores.csv", scores_by_split)

    summary = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "data_root": str(args.data_root),
        "views_order": views_order,
        "align_to": align_to,
        "score": "mean of g-band and r-band per-sample JEPA inverse loss",
        "splits": {
            split: {
                "n": int(scores["loss"].shape[0]),
                "mean": float(np.mean(scores["loss"])),
                "std": float(np.std(scores["loss"])),
                "median": float(np.median(scores["loss"])),
                "min": float(np.min(scores["loss"])),
                "max": float(np.max(scores["loss"])),
            }
            for split, scores in scores_by_split.items()
        },
        "top_percent_purity": purity_rows,
    }
    (args.out_dir / "exp8_starembed_inv_loss_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_histogram(
        test_loss=scores_by_split["test"]["loss"],
        anom_loss=scores_by_split["anom"]["loss"],
        purity_rows=purity_rows,
        out_png=args.out_dir / "exp8_starembed_inv_loss_hist.png",
        out_pdf=args.out_dir / "exp8_starembed_inv_loss_hist.pdf",
        bins=int(args.bins),
    )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
