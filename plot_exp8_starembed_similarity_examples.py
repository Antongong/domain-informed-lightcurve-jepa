#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch_starembed_features"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "starembed_preprocessed"
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs/EXP8_no_group_branch_starembed_features/similarity_examples"

BAND_STYLES = {
    "g": {"color": "#2F6B9A", "marker": "o"},
    "r": {"color": "#B3433B", "marker": "s"},
}


def load_pt(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover_pt_files(split_dir: Path) -> List[Path]:
    manifest = split_dir / "manifest_all.txt"
    if manifest.exists():
        paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path for path in paths if path.exists()]
    return sorted(split_dir.rglob("*.pt"))


def valid_rows(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {arr.shape}")
    if arr.shape[1] >= 4:
        arr = arr[arr[:, 3] > 0.5]
    return arr


def best_period(periodogram: torch.Tensor) -> float:
    pg = periodogram.detach().cpu().float()
    idx = int(torch.argmax(pg[:, 1]).item())
    return float(pg[idx, 0].item())


def class_name(label_map: Dict[str, Any], class_id: int) -> str:
    names = label_map.get("class_name_by_id", {})
    label = str(names.get(str(class_id), class_id))
    return "RS CVn" if label == "RS_CVn" else label


def label_slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "class"


def source_id(item: Dict[str, Any], path: str | Path) -> str:
    meta = item.get("meta", {}) or {}
    return str(meta.get("sourceid", Path(path).stem))


def load_test_features(features_dir: Path, embedding_key: str) -> Dict[str, np.ndarray]:
    npz_path = features_dir / "starembed_embeddings_test.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing test embeddings: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if embedding_key not in data:
        raise KeyError(f"{npz_path} does not contain embedding key {embedding_key!r}; keys={list(data.files)}")
    return {key: data[key] for key in data.files}


def load_label_map(features_dir: Path) -> Dict[str, Any]:
    path = features_dir / "label_map_test.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        bad = np.where(norms[:, 0] <= 0)[0]
        raise ValueError(f"Found zero-norm embeddings at indices: {bad[:10].tolist()}")
    return x / norms


def select_targets(
    y: np.ndarray,
    *,
    seed: int,
    label_map: Dict[str, Any],
) -> List[int]:
    available = sorted(int(v) for v in np.unique(y))
    ordered = [int(v) for v in label_map.get("paper_class_order", []) if int(v) in available]
    ordered.extend(class_id for class_id in available if class_id not in ordered)
    if len(ordered) != 7:
        raise ValueError(f"Expected 7 test classes, found {len(ordered)}: {ordered}")

    rng = np.random.default_rng(int(seed))
    selected: List[int] = []
    for class_id in ordered:
        idx = np.where(y == class_id)[0]
        if idx.size == 0:
            raise ValueError(f"No test samples for class {class_id}")
        selected.append(int(rng.choice(idx)))
    return selected


def find_similarity_extremes(emb: np.ndarray, target_idx: int, metric: str) -> Dict[str, Any]:
    if metric == "l2":
        dists = np.linalg.norm(emb - emb[target_idx], axis=1)
        nearest = dists.copy()
        nearest[target_idx] = np.inf
        farthest = dists.copy()
        farthest[target_idx] = -np.inf

        similar_idx = int(np.argmin(nearest))
        dissimilar_idx = int(np.argmax(farthest))
        return {
            "similar_idx": similar_idx,
            "similar_l2": float(dists[similar_idx]),
            "dissimilar_idx": dissimilar_idx,
            "dissimilar_l2": float(dists[dissimilar_idx]),
        }

    sims = emb @ emb[target_idx]

    max_sims = sims.copy()
    max_sims[target_idx] = -np.inf
    min_sims = sims.copy()
    min_sims[target_idx] = np.inf

    similar_idx = int(np.argmax(max_sims))
    dissimilar_idx = int(np.argmin(min_sims))
    return {
        "similar_idx": similar_idx,
        "similar_cosine": float(sims[similar_idx]),
        "dissimilar_idx": dissimilar_idx,
        "dissimilar_cosine": float(sims[dissimilar_idx]),
    }


def build_selection(
    emb: np.ndarray,
    y: np.ndarray,
    paths: List[Path],
    *,
    seed: int,
    label_map: Dict[str, Any],
    metric: str,
) -> List[Dict[str, Any]]:
    selected = []
    for target_idx in select_targets(y, seed=seed, label_map=label_map):
        extremes = find_similarity_extremes(emb, target_idx, metric)
        row = {
            "target_idx": target_idx,
            "target_class": int(y[target_idx]),
            "target_path": str(paths[target_idx]),
            "similar_idx": extremes["similar_idx"],
            "similar_class": int(y[extremes["similar_idx"]]),
            "similar_path": str(paths[extremes["similar_idx"]]),
            "dissimilar_idx": extremes["dissimilar_idx"],
            "dissimilar_class": int(y[extremes["dissimilar_idx"]]),
            "dissimilar_path": str(paths[extremes["dissimilar_idx"]]),
        }
        if metric == "l2":
            row["similar_l2"] = extremes["similar_l2"]
            row["dissimilar_l2"] = extremes["dissimilar_l2"]
        else:
            row["similar_cosine"] = extremes["similar_cosine"]
            row["dissimilar_cosine"] = extremes["dissimilar_cosine"]
        selected.append(row)
    return selected


def plot_raw(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, style in BAND_STYLES.items():
        lc = valid_rows(item[band]["X"]["lc"])
        ax.scatter(
            lc[:, 0],
            lc[:, 1],
            s=6,
            alpha=0.68,
            color=style["color"],
            marker=style["marker"],
            label=band,
        )
    ax.invert_yaxis()
    ax.set_xlabel("time")
    ax.set_ylabel("centered mag")


def plot_gls(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, style in BAND_STYLES.items():
        pg = item[band]["X"]["periodogram"].detach().cpu().float().numpy()
        bp = best_period(item[band]["X"]["periodogram"])
        ax.plot(pg[:, 0], pg[:, 1], linewidth=1.0, alpha=0.82, color=style["color"], label=f"{band} P={bp:.4g}")
        ax.axvline(bp, color=style["color"], linestyle="--", linewidth=0.8, alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("log10 GLS power")


def plot_phase(ax: plt.Axes, item: Dict[str, Any]) -> None:
    for band, style in BAND_STYLES.items():
        pf = valid_rows(item[band]["X"]["phase_folded_lc"])
        bp = max(best_period(item[band]["X"]["periodogram"]), 1.0e-12)
        phase = np.mod(pf[:, 0] / bp, 1.0)
        ax.scatter(phase, pf[:, 1], s=6, alpha=0.68, color=style["color"], marker=style["marker"], label=band)
        ax.scatter(phase + 1.0, pf[:, 1], s=6, alpha=0.32, color=style["color"], marker=style["marker"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("phase")
    ax.set_ylabel("centered mag")


def metric_value(row: Dict[str, Any], relation: str, metric: str) -> float:
    if relation == "selected":
        return 0.0 if metric == "l2" else 1.0
    key = "similar" if relation == "similar" else "dissimilar"
    return float(row[f"{key}_{metric}"])


def metric_text(value: float, metric: str) -> str:
    return f"L2={value:.4f}" if metric == "l2" else f"cos={value:.4f}"


def metric_title(metric: str) -> str:
    return "L2 distance" if metric == "l2" else "cosine similarity"


def relation_rows(row: Dict[str, Any], paths: List[Path], metric: str) -> List[Dict[str, Any]]:
    if metric == "l2":
        selected_relation = "selected"
        similar_relation = "nearest"
        dissimilar_relation = "farthest"
    else:
        selected_relation = "selected"
        similar_relation = "most similar"
        dissimilar_relation = "most dissimilar"

    return [
        {
            "relation": selected_relation,
            "idx": int(row["target_idx"]),
            "class_id": int(row["target_class"]),
            "value": metric_value(row, "selected", metric),
            "path": paths[int(row["target_idx"])],
        },
        {
            "relation": similar_relation,
            "idx": int(row["similar_idx"]),
            "class_id": int(row["similar_class"]),
            "value": metric_value(row, "similar", metric),
            "path": paths[int(row["similar_idx"])],
        },
        {
            "relation": dissimilar_relation,
            "idx": int(row["dissimilar_idx"]),
            "class_id": int(row["dissimilar_class"]),
            "value": metric_value(row, "dissimilar", metric),
            "path": paths[int(row["dissimilar_idx"])],
        },
    ]


def plot_selection(
    selection: List[Dict[str, Any]],
    paths: List[Path],
    label_map: Dict[str, Any],
    *,
    seed: int,
    embedding_key: str,
    experiment_label: str,
    metric: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    nrows = len(selection) * 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15.5, 2.15 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    plotters = [plot_raw, plot_gls, plot_phase]
    col_titles = ["Raw light curve", "GLS periodogram", "Phase-folded"]
    row_idx = 0
    for selected in selection:
        target_class = int(selected["target_class"])
        target_label = class_name(label_map, target_class)
        for rel in relation_rows(selected, paths, metric):
            item = load_pt(rel["path"])
            sid = source_id(item, rel["path"])
            rel_label = rel["relation"].title()
            cls_label = class_name(label_map, rel["class_id"])

            for col, plotter in enumerate(plotters):
                ax = axes[row_idx, col]
                plotter(ax, item)
                ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.75)
                ax.legend(loc="best", frameon=False, fontsize=7)
                if row_idx == 0:
                    ax.set_title(col_titles[col], fontsize=11)

            axes[row_idx, 0].set_title(
                f"{target_label} target class | {rel_label}: idx={rel['idx']} class={cls_label} "
                f"{metric_text(float(rel['value']), metric)}\nsource={sid}",
                fontsize=9,
            )
            row_idx += 1

    fig.suptitle(
        f"{experiment_label} StarEmbed test-set {metric_title(metric)} examples "
        f"(embedding={embedding_key}, seed={seed})",
        fontsize=16,
    )
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_single_selection(
    selected: Dict[str, Any],
    paths: List[Path],
    label_map: Dict[str, Any],
    *,
    seed: int,
    embedding_key: str,
    experiment_label: str,
    single_label: str,
    metric: str,
    out_png: Path,
    out_pdf: Path,
    pair_only: bool = False,
    no_main_title: bool = False,
    simple_left_titles: bool = False,
) -> None:
    rows = relation_rows(selected, paths, metric)
    if pair_only:
        rows = rows[:2]

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(15.5, 2.25 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])
    plotters = [plot_raw, plot_gls, plot_phase]
    col_titles = ["Raw light curve", "GLS periodogram", "Phase-folded"]

    target_label = class_name(label_map, int(selected["target_class"]))
    for row_idx, rel in enumerate(rows):
        item = load_pt(rel["path"])
        sid = source_id(item, rel["path"])
        rel_label = rel["relation"].title()
        cls_label = class_name(label_map, rel["class_id"])

        for col, plotter in enumerate(plotters):
            ax = axes[row_idx, col]
            plotter(ax, item)
            ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.75)
            ax.legend(loc="best", frameon=False, fontsize=7)
            if row_idx == 0:
                ax.set_title(col_titles[col], fontsize=11)

        if simple_left_titles:
            row_title = "Query sample" if row_idx == 0 else rel_label
            if row_idx == 1:
                row_title = "Most similar sample" if metric == "cosine" else "Nearest sample"
        else:
            row_title = (
                f"{target_label} target class | {rel_label}: idx={rel['idx']} class={cls_label} "
                f"{metric_text(float(rel['value']), metric)}\nsource={sid}"
            )
        axes[row_idx, 0].set_title(row_title, fontsize=9)

    if not no_main_title:
        fig.suptitle(
            f"{experiment_label} StarEmbed {single_label} {metric_title(metric)} example "
            f"(embedding={embedding_key}, seed={seed})",
            fontsize=15,
        )
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def find_selection_row(
    selection: List[Dict[str, Any]],
    *,
    target_class: int | None,
    similar_class: int | None,
    dissimilar_class: int | None,
) -> Dict[str, Any]:
    matches = []
    for row in selection:
        if target_class is not None and int(row["target_class"]) != int(target_class):
            continue
        if similar_class is not None and int(row["similar_class"]) != int(similar_class):
            continue
        if dissimilar_class is not None and int(row["dissimilar_class"]) != int(dissimilar_class):
            continue
        matches.append(row)
    if not matches:
        raise ValueError(
            "No selected row matches "
            f"target_class={target_class}, similar_class={similar_class}, dissimilar_class={dissimilar_class}"
        )
    if len(matches) > 1:
        raise ValueError(f"Selection filter matched {len(matches)} rows; make it more specific")
    return matches[0]


def find_selection_row_from_all(
    emb: np.ndarray,
    y: np.ndarray,
    paths: List[Path],
    *,
    target_class: int | None,
    similar_class: int | None,
    dissimilar_class: int | None,
    metric: str,
) -> Dict[str, Any]:
    target_indices = np.arange(y.shape[0]) if target_class is None else np.flatnonzero(y == int(target_class))
    matches: List[Dict[str, Any]] = []
    for target_idx in target_indices:
        extremes = find_similarity_extremes(emb, int(target_idx), metric)
        row = {
            "target_idx": int(target_idx),
            "target_class": int(y[int(target_idx)]),
            "target_path": str(paths[int(target_idx)]),
            "similar_idx": extremes["similar_idx"],
            "similar_class": int(y[extremes["similar_idx"]]),
            "similar_path": str(paths[extremes["similar_idx"]]),
            "dissimilar_idx": extremes["dissimilar_idx"],
            "dissimilar_class": int(y[extremes["dissimilar_idx"]]),
            "dissimilar_path": str(paths[extremes["dissimilar_idx"]]),
        }
        if metric == "l2":
            row["similar_l2"] = extremes["similar_l2"]
            row["dissimilar_l2"] = extremes["dissimilar_l2"]
        else:
            row["similar_cosine"] = extremes["similar_cosine"]
            row["dissimilar_cosine"] = extremes["dissimilar_cosine"]

        if similar_class is not None and int(row["similar_class"]) != int(similar_class):
            continue
        if dissimilar_class is not None and int(row["dissimilar_class"]) != int(dissimilar_class):
            continue
        matches.append(row)

    if not matches:
        raise ValueError(
            "No full-test-set row matches "
            f"target_class={target_class}, similar_class={similar_class}, dissimilar_class={dissimilar_class}"
        )
    if metric == "l2":
        return min(matches, key=lambda row: float(row["similar_l2"]))
    return max(matches, key=lambda row: float(row["similar_cosine"]))


def class_order_from_labels(y: np.ndarray, label_map: Dict[str, Any]) -> List[int]:
    available = sorted(int(v) for v in np.unique(y))
    ordered = [int(v) for v in label_map.get("paper_class_order", []) if int(v) in available]
    ordered.extend(class_id for class_id in available if class_id not in ordered)
    return ordered


def compute_class_cosine_matrix(
    emb: np.ndarray,
    y: np.ndarray,
    class_order: List[int],
    *,
    device: str,
) -> np.ndarray:
    x = torch.as_tensor(np.asarray(emb, dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)

    means = []
    counts = []
    for class_id in class_order:
        mask = labels == int(class_id)
        count = int(mask.sum().item())
        if count == 0:
            raise ValueError(f"No embeddings for class {class_id}")
        means.append(x[mask].mean(dim=0))
        counts.append(count)

    mean_matrix = torch.stack(means, dim=0) @ torch.stack(means, dim=0).T
    matrix = mean_matrix.detach().cpu().numpy().astype(np.float64)

    # For same-class cells, remove identity self-pairs from the average.
    for idx, count in enumerate(counts):
        if count > 1:
            matrix[idx, idx] = (matrix[idx, idx] * count * count - count) / (count * (count - 1))
        else:
            matrix[idx, idx] = np.nan
    return matrix


def compute_class_l2_matrix(
    emb: np.ndarray,
    y: np.ndarray,
    class_order: List[int],
    *,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    x = torch.as_tensor(np.asarray(emb, dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)
    class_tensors = []
    for class_id in class_order:
        mask = labels == int(class_id)
        if int(mask.sum().item()) == 0:
            raise ValueError(f"No embeddings for class {class_id}")
        class_tensors.append(x[mask])

    matrix = np.empty((len(class_order), len(class_order)), dtype=np.float64)
    for row, a in enumerate(class_tensors):
        for col in range(row, len(class_tensors)):
            b = class_tensors[col]
            total = 0.0
            count = 0
            for start in range(0, a.shape[0], int(chunk_size)):
                distances = torch.cdist(a[start : start + int(chunk_size)], b, p=2)
                if row == col:
                    global_rows = torch.arange(
                        start,
                        min(start + int(chunk_size), a.shape[0]),
                        device=device,
                    )
                    distances[torch.arange(distances.shape[0], device=device), global_rows] = float("nan")
                    valid = torch.isfinite(distances)
                    total += float(distances[valid].sum().item())
                    count += int(valid.sum().item())
                else:
                    total += float(distances.sum().item())
                    count += int(distances.numel())
            value = total / count if count else np.nan
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def write_similarity_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    class_order: List[int],
    label_map: Dict[str, Any],
) -> None:
    labels = [class_name(label_map, class_id) for class_id in class_order]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.8f}" if np.isfinite(value) else "" for value in row])


def plot_similarity_matrix(
    matrix: np.ndarray,
    class_order: List[int],
    label_map: Dict[str, Any],
    *,
    embedding_key: str,
    split: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    labels = [class_name(label_map, class_id) for class_id in class_order]
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Similarity matrix contains no finite values")
    vmax = float(np.max(finite))
    vmin = float(np.min(finite))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=max(vmax, 1.0e-6)) if vmin < 0.0 else None

    fig, ax = plt.subplots(figsize=(8.8, 7.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.22, h_pad=0.10, wspace=0.04, hspace=0.04)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
   # ax.set_title(f"Average cosine similarity by class ({split}, embedding={embedding_key})")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                text = "NA"
            else:
                text = f"{value:.3f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average cosine similarity")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def plot_distance_matrix(
    matrix: np.ndarray,
    class_order: List[int],
    label_map: Dict[str, Any],
    *,
    embedding_key: str,
    split: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    labels = [class_name(label_map, class_id) for class_id in class_order]
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Distance matrix contains no finite values")
    vmax = float(np.max(finite))
    vmin = float(np.min(finite))

    fig, ax = plt.subplots(figsize=(8.8, 7.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.22, h_pad=0.10, wspace=0.04, hspace=0.04)
    im = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "NA" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average L2 distance")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def write_selection_csv(path: Path, selection: List[Dict[str, Any]], metric: str) -> None:
    value_fields = ["similar_l2", "dissimilar_l2"] if metric == "l2" else ["similar_cosine", "dissimilar_cosine"]
    fieldnames = [
        "target_idx",
        "target_class",
        "target_path",
        "similar_idx",
        "similar_class",
        value_fields[0],
        "similar_path",
        "dissimilar_idx",
        "dissimilar_class",
        value_fields[1],
        "dissimilar_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick one random EXP8 StarEmbed test light curve from each of the 7 classes, "
            "find its nearest/farthest test-set embeddings by the selected metric, "
            "and plot raw/GLS/phase-folded views."
        )
    )
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--embedding_key", choices=["x", "x_avg"], default="x")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class_l2_chunk_size", type=int, default=256)
    parser.add_argument("--experiment_label", type=str, default="EXP8")
    parser.add_argument("--output_prefix", type=str, default="exp8")
    parser.add_argument("--single_target_class", type=int, default=5, help="Target class for the separate example PDF.")
    parser.add_argument("--single_similar_class", type=int, default=5, help="Most-similar class for the separate example PDF.")
    parser.add_argument("--single_dissimilar_class", type=int, default=2, help="Most-dissimilar class for the separate example PDF.")
    parser.add_argument(
        "--single_pair_only",
        action="store_true",
        help="For the separate example PDF, plot only the query and most-similar rows.",
    )
    parser.add_argument(
        "--single_no_main_title",
        action="store_true",
        help="For the separate example PDF, omit the figure suptitle.",
    )
    parser.add_argument(
        "--single_simple_left_titles",
        action="store_true",
        help="For the separate example PDF, label left panels only by relation and hide ids/source text.",
    )
    parser.add_argument(
        "--single_search_all",
        action="store_true",
        help="If the seed-selected rows do not match the requested single classes, search all test samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = load_test_features(args.features_dir, args.embedding_key)
    label_map = load_label_map(args.features_dir)
    paths = discover_pt_files(args.data_root / args.split)
    y = np.asarray(features["y"], dtype=np.int64)
    x = np.asarray(features[args.embedding_key], dtype=np.float32)

    if len(paths) != x.shape[0]:
        raise ValueError(f"Path count ({len(paths)}) does not match embedding rows ({x.shape[0]})")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"Label count ({y.shape[0]}) does not match embedding rows ({x.shape[0]})")

    emb = normalized_embeddings(x) if args.metric == "cosine" else x
    selection = build_selection(emb, y, paths, seed=int(args.seed), label_map=label_map, metric=str(args.metric))

    stem = f"{args.output_prefix}_starembed_test_{args.metric}_{args.embedding_key}_seed{int(args.seed)}"
    write_selection_csv(args.out_dir / f"{stem}_selection.csv", selection, metric=str(args.metric))
    plot_selection(
        selection,
        paths,
        label_map,
        seed=int(args.seed),
        embedding_key=str(args.embedding_key),
        experiment_label=str(args.experiment_label),
        metric=str(args.metric),
        out_png=args.out_dir / f"{stem}_raw_gls_phase.png",
        out_pdf=args.out_dir / f"{stem}_raw_gls_phase.pdf",
    )

    single_stem = None
    try:
        single = find_selection_row(
            selection,
            target_class=args.single_target_class,
            similar_class=args.single_similar_class,
            dissimilar_class=args.single_dissimilar_class,
        )
    except ValueError as exc:
        if not args.single_search_all:
            print(f"[WARN] skipped separate example: {exc}")
            single = None
        else:
            print(f"[WARN] seed-selected row missing: {exc}")
            single = find_selection_row_from_all(
                emb,
                y,
                paths,
                target_class=args.single_target_class,
                similar_class=args.single_similar_class,
                dissimilar_class=args.single_dissimilar_class,
                metric=str(args.metric),
            )
            print(
                "[INFO] using full-test-set match: "
                f"target_idx={single['target_idx']} similar_idx={single['similar_idx']} "
                f"dissimilar_idx={single['dissimilar_idx']}"
            )
    else:
        pass

    if single is not None:
        single_target_label = class_name(label_map, int(single["target_class"]))
        single_similar_label = class_name(label_map, int(single["similar_class"]))
        single_dissimilar_label = class_name(label_map, int(single["dissimilar_class"]))
        single_label = f"{single_target_label}/{single_similar_label} vs {single_dissimilar_label}"
        single_stem = (
            f"{stem}_{label_slug(single_target_label)}_"
            f"{label_slug(single_similar_label)}_vs_{label_slug(single_dissimilar_label)}"
        )
        plot_single_selection(
            single,
            paths,
            label_map,
            seed=int(args.seed),
            embedding_key=str(args.embedding_key),
            experiment_label=str(args.experiment_label),
            single_label=single_label,
            metric=str(args.metric),
            out_png=args.out_dir / f"{single_stem}_raw_gls_phase.png",
            out_pdf=args.out_dir / f"{single_stem}_raw_gls_phase.pdf",
            pair_only=bool(args.single_pair_only),
            no_main_title=bool(args.single_no_main_title),
            simple_left_titles=bool(args.single_simple_left_titles),
        )

    class_order = class_order_from_labels(y, label_map)
    if args.metric == "l2":
        matrix = compute_class_l2_matrix(
            emb,
            y,
            class_order,
            device=str(args.device),
            chunk_size=int(args.class_l2_chunk_size),
        )
    else:
        matrix = compute_class_cosine_matrix(emb, y, class_order, device=str(args.device))
    matrix_stem = f"{args.output_prefix}_starembed_{args.split}_class_average_{args.metric}_{args.embedding_key}"
    write_similarity_matrix_csv(args.out_dir / f"{matrix_stem}.csv", matrix, class_order, label_map)
    if args.metric == "l2":
        plot_distance_matrix(
            matrix,
            class_order,
            label_map,
            embedding_key=str(args.embedding_key),
            split=str(args.split),
            out_png=args.out_dir / f"{matrix_stem}.png",
            out_pdf=args.out_dir / f"{matrix_stem}.pdf",
        )
    else:
        plot_similarity_matrix(
            matrix,
            class_order,
            label_map,
            embedding_key=str(args.embedding_key),
            split=str(args.split),
            out_png=args.out_dir / f"{matrix_stem}.png",
            out_pdf=args.out_dir / f"{matrix_stem}.pdf",
        )

    for row in selection:
        target_label = class_name(label_map, int(row["target_class"]))
        similar_label = class_name(label_map, int(row["similar_class"]))
        dissimilar_label = class_name(label_map, int(row["dissimilar_class"]))
        print(
            f"{target_label}: target idx={row['target_idx']} | "
            f"{'nearest' if args.metric == 'l2' else 'most similar'} idx={row['similar_idx']} "
            f"class={similar_label} {metric_text(float(row[f'similar_{args.metric}']), str(args.metric))} | "
            f"{'farthest' if args.metric == 'l2' else 'most dissimilar'} idx={row['dissimilar_idx']} "
            f"class={dissimilar_label} {metric_text(float(row[f'dissimilar_{args.metric}']), str(args.metric))}"
        )
    print(f"[OK] wrote {args.out_dir / f'{stem}_selection.csv'}")
    print(f"[OK] wrote {args.out_dir / f'{stem}_raw_gls_phase.png'}")
    print(f"[OK] wrote {args.out_dir / f'{stem}_raw_gls_phase.pdf'}")
    if single_stem is not None:
        print(f"[OK] wrote {args.out_dir / f'{single_stem}_raw_gls_phase.png'}")
        print(f"[OK] wrote {args.out_dir / f'{single_stem}_raw_gls_phase.pdf'}")
    print(f"[OK] wrote {args.out_dir / f'{matrix_stem}.csv'}")
    print(f"[OK] wrote {args.out_dir / f'{matrix_stem}.png'}")
    print(f"[OK] wrote {args.out_dir / f'{matrix_stem}.pdf'}")


if __name__ == "__main__":
    main()
