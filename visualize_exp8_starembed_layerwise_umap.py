#!/usr/bin/env python3
"""Layer-wise UMAP for EXP8 StarEmbed test-set numeric encoder features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import umap


REPO_ROOT = Path("/home/rui/code/algorithm_base/timeseries")
SCRIPT_DIR = REPO_ROOT / "clip_experiments"
RUN_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch"
DEFAULT_DATA_ROOT = REPO_ROOT / "starembed_preprocessed"
DEFAULT_OUT_DIR = RUN_DIR / "starembed_test_layerwise_meanpool_umap"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_starembed_embeddings import (  # noqa: E402
    PrecomputedStarEmbedPairDataset,
    load_model_from_ckpt,
    move_to_device,
    precomputed_starembed_pair_collate,
)
from train_ddp_numeric import build_model  # noqa: E402


CLASS_NAMES = {
    1: "EW",
    2: "EA",
    4: "RRab",
    5: "RRc",
    6: "RRd",
    8: "RS CVn",
    13: "LPV",
}
CLASS_ORDER = [1, 2, 4, 5, 6, 8, 13]
CLASS_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
    6: "#9467bd",
    8: "#8c564b",
    13: "#e377c2",
}
VIEW_ORDER = ["raw", "periodogram", "phase_folded"]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask_f = (mask > 0).float().unsqueeze(-1) if mask.dtype != torch.bool else mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def encoder_layer_means(
    encoder: torch.nn.Module,
    token_embeddings: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Return mean-pooled token states after every transformer block."""
    x = encoder.emb_dropout(token_embeddings)
    cos, sin = encoder.rope(positions)

    feats = []
    for block in encoder.layers:
        x = block(x, mask=mask, cos=cos, sin=sin)
        feats.append(masked_mean(x, mask))
    return feats


def extract_view_layer_means(model: torch.nn.Module, xdict: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
    device = next(model.parameters()).device
    out: dict[str, list[torch.Tensor]] = {}

    if bool(model.enable.get(model.VIEW_RAW, True)):
        lc = xdict["lc"].to(device, non_blocking=True).float()
        t = lc[..., 0]
        v = lc[..., 1]
        sigma = lc[..., 2]
        mask = lc[..., 3]
        if model.raw_position_mode == "index":
            t = model._make_index_positions(v.shape[0], v.shape[1], device=device)
        token = model.raw_value_embed(v, sigma=sigma if model.raw_use_uncertainty else None)
        out["raw"] = encoder_layer_means(model.raw_encoder, token, t, mask)

    if bool(model.enable.get(model.VIEW_PERIODOGRAM, True)):
        pg = xdict["periodogram"].to(device, non_blocking=True).float()
        period = pg[..., 0].clamp_min(1.0e-12)
        value = pg[..., 1]
        if model.period_position_mode == "index":
            t = model._make_index_positions(value.shape[0], value.shape[1], device=device)
        else:
            t = torch.log10(period)
        token = model.period_value_embed(value, sigma=None)
        mask = torch.ones_like(value, dtype=torch.bool)
        out["periodogram"] = encoder_layer_means(model.periodogram_encoder, token, t, mask)

    if bool(model.enable.get(model.VIEW_PHASE_FOLDED, True)):
        pflc = xdict["phase_folded_lc"].to(device, non_blocking=True).float()
        best_period = xdict.get("best_period", None)
        phase_time = pflc[..., 0]
        value = pflc[..., 1]
        sigma = pflc[..., 2]
        mask = pflc[..., 3]
        if model.phase_use_normalized_phase and best_period is not None:
            bp = best_period.to(device, non_blocking=True).float().view(-1, 1).clamp_min(1.0e-12)
            phase = (phase_time / bp).clamp(0.0, 1.0)
        else:
            phase = phase_time
        if model.phase_position_mode == "index":
            t = model._make_index_positions(value.shape[0], value.shape[1], device=device)
        else:
            t = phase
        token = model.phase_value_embed(value, sigma=sigma if model.pf_use_uncertainty else None)
        out["phase_folded"] = encoder_layer_means(model.phase_encoder, token, t, mask)

    return out


def empty_feature_parts(n_layers: int) -> dict[str, list[list[np.ndarray]]]:
    return {view: [[] for _ in range(n_layers)] for view in VIEW_ORDER}


def extract_test_features(args: argparse.Namespace) -> tuple[dict[str, list[np.ndarray]], np.ndarray, list[str]]:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = build_model(cfg).to(args.device)
    load_model_from_ckpt(model, str(args.ckpt), strict=True)
    model.eval()

    dataset = PrecomputedStarEmbedPairDataset(args.data_root, "test", limit=int(args.limit))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(args.device.type == "cuda"),
        drop_last=False,
        collate_fn=precomputed_starembed_pair_collate,
        persistent_workers=(int(args.num_workers) > 0),
    )

    n_layers = len(model.raw_encoder.layers)
    parts = empty_feature_parts(n_layers)
    y_parts: list[np.ndarray] = []
    y_str_parts: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch_g = move_to_device(batch["g"]["X"], args.device)
            batch_r = move_to_device(batch["r"]["X"], args.device)
            feats_g = extract_view_layer_means(model, batch_g)
            feats_r = extract_view_layer_means(model, batch_r)

            for view in VIEW_ORDER:
                for layer_idx in range(n_layers):
                    feat = 0.5 * (feats_g[view][layer_idx] + feats_r[view][layer_idx])
                    parts[view][layer_idx].append(feat.detach().cpu().numpy().astype(np.float32))

            y_parts.append(batch["Y"]["star_class"].detach().cpu().numpy().astype(np.int64))
            y_str_parts.extend(str(x) for x in batch["Y"]["star_class_str"])

    features = {
        view: [np.concatenate(layer_parts, axis=0) for layer_parts in layer_lists]
        for view, layer_lists in parts.items()
    }
    labels = np.concatenate(y_parts, axis=0)
    return features, labels, y_str_parts


def run_umap(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    finite = np.all(np.isfinite(X), axis=1)
    if not finite.all():
        X = X[finite]
    if args.standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=args.metric,
        random_state=int(args.seed),
    )
    return reducer.fit_transform(X)


def select_plot_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(labels.shape[0])
    if fraction <= 0.0:
        raise ValueError("--plot_fraction must be positive")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in CLASS_ORDER:
        idx = np.flatnonzero(labels == class_id)
        if idx.size == 0:
            continue
        n_keep = max(1, int(np.ceil(idx.size * fraction)))
        selected.append(np.sort(rng.choice(idx, size=n_keep, replace=False)))
    if not selected:
        return np.arange(labels.shape[0])
    return np.sort(np.concatenate(selected))


def plot_view_grid(
    embeddings: list[np.ndarray],
    labels: np.ndarray,
    view: str,
    out_path: Path,
    *,
    point_size: float,
    alpha: float,
    plot_indices: np.ndarray,
    legend_marker_size: float,
    panel_box_aspect: float,
) -> None:
    n_layers = len(embeddings)
    ncols = 2
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 10.6), squeeze=False)
    plot_mask_base = np.zeros(labels.shape[0], dtype=bool)
    plot_mask_base[plot_indices] = True

    for layer_idx, emb in enumerate(embeddings):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        for class_id in CLASS_ORDER:
            mask = (labels == class_id) & plot_mask_base
            if not np.any(mask):
                continue
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=point_size,
                alpha=alpha,
                color=CLASS_COLORS[class_id],
                label=CLASS_NAMES.get(class_id, str(class_id)),
                linewidths=0,
            )
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(float(panel_box_aspect))

    for empty_idx in range(n_layers, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_COLORS[class_id],
            markeredgecolor="none",
            markersize=float(np.sqrt(legend_marker_size)),
            alpha=alpha,
        )
        for class_id in CLASS_ORDER
    ]
    legend_labels = [CLASS_NAMES[class_id] for class_id in CLASS_ORDER]
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(CLASS_ORDER), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_features(
    out_dir: Path,
    output_stem: str,
    features: dict[str, list[np.ndarray]],
    labels: np.ndarray,
    y_str: list[str],
) -> None:
    payload: dict[str, Any] = {"y": labels, "y_str": np.asarray(y_str, dtype=object)}
    for view, layer_features in features.items():
        for i, feat in enumerate(layer_features, start=1):
            payload[f"{view}_layer{i}"] = feat
    np.savez_compressed(out_dir / f"{output_stem}_starembed_test_layerwise_meanpool_features.npz", **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RUN_DIR / "config_used.yaml")
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "ckpt_final.pt")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--plot_fraction", type=float, default=0.65)
    parser.add_argument("--legend_marker_size", type=float, default=70.0)
    parser.add_argument(
        "--panel_box_aspect",
        type=float,
        default=0.72,
        help="Axes height/width ratio for each UMAP panel; values below 1 make rectangular panels.",
    )
    parser.add_argument("--output_stem", type=str, default="exp8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device)
    set_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, y_str = extract_test_features(args)
    save_features(args.out_dir, args.output_stem, features, labels, y_str)
    plot_indices = select_plot_indices(labels, float(args.plot_fraction), int(args.seed))

    manifest = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "data_root": str(args.data_root),
        "split": "test",
        "n_samples": int(labels.shape[0]),
        "views": VIEW_ORDER,
        "n_layers": len(next(iter(features.values()))),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "standardize": bool(args.standardize),
        "seed": int(args.seed),
        "plot_fraction": float(args.plot_fraction),
        "plot_n_samples": int(plot_indices.shape[0]),
        "feature": "g/r average of masked mean-pooled token states after each transformer block",
    }

    for view in VIEW_ORDER:
        print(f"[UMAP] {view}", flush=True)
        view_embeddings = [run_umap(layer_feat, args) for layer_feat in features[view]]
        np.savez_compressed(
            args.out_dir / f"{args.output_stem}_starembed_test_{view}_layerwise_umap.npz",
            **{f"layer{i + 1}": emb for i, emb in enumerate(view_embeddings)},
            y=labels,
            plot_indices=plot_indices,
        )
        out_path = args.out_dir / f"{args.output_stem}_starembed_test_{view}_layerwise_meanpool_umap.png"
        plot_view_grid(
            view_embeddings,
            labels,
            view,
            out_path,
            point_size=float(args.point_size),
            alpha=float(args.alpha),
            plot_indices=plot_indices,
            legend_marker_size=float(args.legend_marker_size),
            panel_box_aspect=float(args.panel_box_aspect),
        )
        print(f"Wrote {out_path}")
        print(f"Wrote {out_path.with_suffix('.pdf')}")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
