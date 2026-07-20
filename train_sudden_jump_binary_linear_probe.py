#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_FEATURES_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch_starembed_features"
DEFAULT_INJECTED_FEATURES_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch_sudden_jump_binary_features"
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch_sudden_jump_binary_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe for original vs sudden-jump injected StarEmbed light curves."
    )
    parser.add_argument("--original-features-dir", type=Path, default=DEFAULT_ORIGINAL_FEATURES_DIR)
    parser.add_argument("--injected-features-dir", type=Path, default=DEFAULT_INJECTED_FEATURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-key", choices=["x", "x_avg"], default="x")
    parser.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_feature_split(features_dir: Path, split: str, feature_key: str) -> np.ndarray:
    path = features_dir / f"starembed_embeddings_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if feature_key not in data:
            raise KeyError(f"{feature_key!r} missing from {path}; keys={data.files}")
        return np.asarray(data[feature_key], dtype=np.float32)


def load_binary_split(
    original_features_dir: Path,
    injected_features_dir: Path,
    split: str,
    feature_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_original = load_feature_split(original_features_dir, split, feature_key)
    x_injected = load_feature_split(injected_features_dir, split, feature_key)
    if x_original.shape != x_injected.shape:
        raise ValueError(
            f"Shape mismatch for split={split}: original={x_original.shape}, injected={x_injected.shape}"
        )
    x = np.concatenate([x_original, x_injected], axis=0)
    y = np.concatenate(
        [
            np.zeros((x_original.shape[0],), dtype=np.int64),
            np.ones((x_injected.shape[0],), dtype=np.int64),
        ],
        axis=0,
    )
    return x, y


def parse_c_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--c-grid must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("All C values must be positive")
    return values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    clf = LogisticRegression(
        C=float(c_value),
        max_iter=int(max_iter),
        solver="lbfgs",
        class_weight=None,
        random_state=int(seed),
    )
    clf.fit(x_train, y_train)
    return clf


def write_metrics_csv(path: Path, metrics_by_split: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "accuracy", "precision", "recall", "f1"])
        for split, metrics in metrics_by_split.items():
            writer.writerow(
                [
                    split,
                    f"{metrics['accuracy']:.10g}",
                    f"{metrics['precision']:.10g}",
                    f"{metrics['recall']:.10g}",
                    f"{metrics['f1']:.10g}",
                ]
            )


def plot_confusion_matrix(cm: np.ndarray, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Original", "Injected"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("EXP8 Sudden-Jump Binary Linear Probe")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#222222"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=13)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_binary_split(args.original_features_dir, args.injected_features_dir, "train", args.feature_key)
    x_val, y_val = load_binary_split(args.original_features_dir, args.injected_features_dir, "validation", args.feature_key)
    x_test, y_test = load_binary_split(args.original_features_dir, args.injected_features_dir, "test", args.feature_key)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    c_grid = parse_c_grid(args.c_grid)
    sweep_rows = []
    best: tuple[float, float, LogisticRegression] | None = None
    for c_value in c_grid:
        clf = fit_logistic_regression(
            x_train_s,
            y_train,
            c_value=c_value,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
        )
        val_pred = clf.predict(x_val_s)
        val_metrics = compute_metrics(y_val, val_pred)
        sweep_rows.append({"C": float(c_value), **val_metrics})
        score = val_metrics["f1"]
        if best is None or score > best[0]:
            best = (score, float(c_value), clf)

    assert best is not None
    best_f1, best_c, best_clf = best

    predictions = {
        "train": best_clf.predict(x_train_s),
        "validation": best_clf.predict(x_val_s),
        "test": best_clf.predict(x_test_s),
    }
    targets = {"train": y_train, "validation": y_val, "test": y_test}
    metrics_by_split = {
        split: compute_metrics(targets[split], predictions[split])
        for split in ("train", "validation", "test")
    }
    cm = confusion_matrix(y_test, predictions["test"], labels=[0, 1])

    write_metrics_csv(args.out_dir / "binary_linear_probe_metrics.csv", metrics_by_split)
    with (args.out_dir / "binary_linear_probe_c_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["C", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    summary: Dict[str, Any] = {
        "original_features_dir": str(args.original_features_dir),
        "injected_features_dir": str(args.injected_features_dir),
        "feature_key": str(args.feature_key),
        "c_grid": c_grid,
        "best_c": float(best_c),
        "best_validation_f1": float(best_f1),
        "n_train": int(y_train.shape[0]),
        "n_validation": int(y_val.shape[0]),
        "n_test": int(y_test.shape[0]),
        "metrics": metrics_by_split,
        "test_confusion_matrix": cm.astype(int).tolist(),
        "label_order": ["original", "injected"],
    }
    (args.out_dir / "binary_linear_probe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_confusion_matrix(
        cm,
        out_png=args.out_dir / "binary_linear_probe_confusion_matrix.png",
        out_pdf=args.out_dir / "binary_linear_probe_confusion_matrix.pdf",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
