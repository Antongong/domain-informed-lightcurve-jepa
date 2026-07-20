#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import umap


DEFAULT_FEATURES_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/"
    "runs/EXP8_no_group_branch_starembed_features"
)
DEFAULT_RESULT_DIR = Path(
    "/home/rui/code/algorithm_base/timeseries/clip_experiments/"
    "runs/EXP8_no_group_branch_starembed_features/benchmark/x/clustering/"
    "EXP8_no_group_branch_starembed_features_all_concat_std0_p30.0_seed42"
)

NORMAL_CLASS_NAMES: Dict[int, str] = {
    1: "EW",
    2: "EA",
    4: "RRab",
    5: "RRc",
    6: "RRd",
    8: "RS_CVn",
    13: "LPV",
}
NORMAL_CLASS_ORDER = [1, 2, 4, 5, 6, 8, 13]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAP for EXP8 StarEmbed test union anomaly splits.")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--scenario", choices=["concat", "avg"], default="concat")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--standardize", action="store_true", help="Standardize on the union before UMAP.")
    return parser.parse_args()


def load_split(features_dir: Path, split: str, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    key = "x" if scenario == "concat" else "x_avg"
    path = features_dir / f"starembed_embeddings_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    with np.load(path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key {key!r} not found in {path}. Available keys: {list(data.keys())}")
        label_key = "y_str" if "y_str" in data else "y"
        X = np.asarray(data[key], dtype=np.float32)
        y = np.asarray([int(str(value)) for value in data[label_key].tolist()], dtype=np.int64)

    finite = np.all(np.isfinite(X), axis=1)
    return X[finite], y[finite]


def build_class_labels(y_test: np.ndarray, y_anom: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    anom_classes = sorted(set(int(v) for v in y_anom.tolist()))
    ordered = [label for label in NORMAL_CLASS_ORDER if label in set(y_test.tolist())]
    ordered.extend(label for label in anom_classes if label not in ordered)

    label_to_idx = {label: idx for idx, label in enumerate(ordered)}
    text_labels = [
        NORMAL_CLASS_NAMES[label] if label in NORMAL_CLASS_NAMES else f"Anom {label}"
        for label in ordered
    ]
    y_all_orig = np.concatenate([y_test, y_anom], axis=0)
    y_class = np.asarray([label_to_idx[int(label)] for label in y_all_orig.tolist()], dtype=np.int64)
    return y_class, text_labels


def plot_classwise(X2: np.ndarray, labels: np.ndarray, text_labels: List[str], output_base: Path) -> None:
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    for lbl in np.unique(labels):
        mask = labels == lbl
        label_idx = int(lbl)
        text = text_labels[label_idx] if 0 <= label_idx < len(text_labels) else str(label_idx)
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=14,
            label=text,
            alpha=0.68,
            color=cmap(label_idx % 20),
            edgecolors="none",
        )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_split(X2: np.ndarray, split_labels: np.ndarray, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
    styles = [
        ("test", 0, "#4C78A8", 12, 0.45),
        ("anom", 1, "#F58518", 18, 0.78),
    ]
    for text, label, color, size, alpha in styles:
        mask = split_labels == label
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=size,
            label=f"{text} (n={int(mask.sum())})",
            alpha=alpha,
            color=color,
            edgecolors="none",
        )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_split(args.features_dir, "test", args.scenario)
    X_anom, y_anom = load_split(args.features_dir, "anom", args.scenario)
    X = np.vstack([X_test, X_anom]).astype(np.float32)

    if args.standardize:
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X).astype(np.float32)

    y_class, text_labels = build_class_labels(y_test, y_anom)
    split_labels = np.concatenate(
        [np.zeros(X_test.shape[0], dtype=np.int64), np.ones(X_anom.shape[0], dtype=np.int64)],
        axis=0,
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=args.metric,
        random_state=int(args.random_state),
    )
    X_umap = reducer.fit_transform(X)

    stem = "umap_Test_Anom_Union"
    class_base = args.result_dir / f"{stem}_classwise"
    split_base = args.result_dir / f"{stem}_split"
    plot_classwise(X_umap, y_class, text_labels, class_base)
    plot_split(X_umap, split_labels, split_base)

    np.savez_compressed(
        args.result_dir / f"{stem}.npz",
        X_umap=X_umap.astype(np.float32),
        y_class=y_class,
        split=split_labels,
        text_labels=np.asarray(text_labels, dtype=object),
        y_test=y_test,
        y_anom=y_anom,
    )
    manifest = {
        "features_dir": str(args.features_dir),
        "scenario": args.scenario,
        "n_test": int(X_test.shape[0]),
        "n_anom": int(X_anom.shape[0]),
        "n_total": int(X.shape[0]),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "random_state": int(args.random_state),
        "standardize": bool(args.standardize),
        "class_labels": text_labels,
        "outputs": {
            "classwise_pdf": str(class_base.with_suffix(".pdf")),
            "classwise_png": str(class_base.with_suffix(".png")),
            "split_pdf": str(split_base.with_suffix(".pdf")),
            "split_png": str(split_base.with_suffix(".png")),
            "npz": str(args.result_dir / f"{stem}.npz"),
        },
    }
    (args.result_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
