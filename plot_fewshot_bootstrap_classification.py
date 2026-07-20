#!/usr/bin/env python3
"""Plot few-shot bootstrap linear-classification metrics.

The input CSVs are produced by the bootstrap linear benchmark and contain
classification-report means/stds for each n-shot setting.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path("/home/rui/code/algorithm_base/timeseries")
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "clip_experiments/runs/EXP8_no_group_branch_starembed_features"
    / "benchmark/x/bootstrap_linear"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


MODEL_SPECS = [
    ModelSpec(
        "Hand-crafted feature",
        REPO_ROOT / "hand_crafted_feature/handcrafted_out/b_linear",
    ),
    ModelSpec("Chronos-2", REPO_ROOT / "Chronos-2/b_linear"),
    ModelSpec("Chronos-tiny-Bolt", REPO_ROOT / "Chronos-bolt/b_linear"),
    ModelSpec("Astromer-2", REPO_ROOT / "astromer-2/b_linear"),
    ModelSpec("Our model", DEFAULT_OUT_DIR),
    ModelSpec(
        "EXP9 (CLIP only)",
        REPO_ROOT
        / "clip_experiments/runs/EXP9_clip_only_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP10 (LEJEPA only)",
        REPO_ROOT
        / "clip_experiments/runs/EXP10_lejepa_only_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP11 (w/o CROPE)",
        REPO_ROOT
        / "clip_experiments/runs/EXP11_no_crope_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP12 (w/o error-aware numeric emb.)",
        REPO_ROOT
        / "clip_experiments/runs/EXP12_no_erroraware_numeric_embedding_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP13 (w/o GLS branch)",
        REPO_ROOT
        / "clip_experiments/runs/EXP13_no_gls_branch_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP14 (w/o periodogram numeric branch)",
        REPO_ROOT
        / "clip_experiments/runs/EXP14_no_periodogram_numeric_branch_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
    ModelSpec(
        "EXP15 (w/o raw numeric branch)",
        REPO_ROOT
        / "clip_experiments/runs/EXP15_no_raw_numeric_branch_no_group_starembed_features/benchmark/x/bootstrap_linear",
    ),
]

METRICS = {
    "Acc": ("accuracy", "f1-score_mean", "f1-score_std"),
    "Recall": ("macro avg", "recall_mean", "recall_std"),
    "Prec": ("macro avg", "precision_mean", "precision_std"),
    "F1": ("macro avg", "f1-score_mean", "f1-score_std"),
}


def find_metric_csv(model_path: Path, shot: int) -> Path:
    pattern = str(model_path / "**" / f"n{shot}" / "metrics_mean_std.csv")
    matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No metrics_mean_std.csv found for n{shot}: {model_path}")
    if len(matches) > 1:
        print(f"[warn] Multiple n{shot} CSVs under {model_path}; using {matches[0]}")
    return matches[0]


def extract_metric(csv_path: Path, row_label: str, mean_col: str, std_col: str) -> tuple[float, float]:
    df = pd.read_csv(csv_path, index_col=0)
    if row_label not in df.index:
        raise KeyError(f"Row {row_label!r} missing from {csv_path}")
    for col in (mean_col, std_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} missing from {csv_path}")

    mean = float(df.loc[row_label, mean_col]) * 100.0
    std = float(df.loc[row_label, std_col]) * 100.0
    return mean, std


def collect_results(model_specs: list[ModelSpec], shots: list[int]) -> pd.DataFrame:
    rows = []
    for model in model_specs:
        for shot in shots:
            csv_path = find_metric_csv(model.path, shot)
            for metric, (row_label, mean_col, std_col) in METRICS.items():
                mean, std = extract_metric(csv_path, row_label, mean_col, std_col)
                rows.append(
                    {
                        "model": model.name,
                        "shot": shot,
                        "metric": metric,
                        "mean_percent": mean,
                        "std_percent": std,
                        "source_csv": str(csv_path),
                    }
                )
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, out_dir: Path, out_stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True, sharey=False)
    axes = axes.ravel()
    shots = sorted(results["shot"].unique())
    palette = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P"]

    for ax, metric in zip(axes, METRICS.keys()):
        metric_df = results[results["metric"] == metric]
        for i, model in enumerate(results["model"].drop_duplicates()):
            model_df = metric_df[metric_df["model"] == model].sort_values("shot")
            ax.errorbar(
                model_df["shot"],
                model_df["mean_percent"],
                yerr=model_df["std_percent"],
                label=model,
                color=palette(i),
                marker=markers[i % len(markers)],
                linewidth=1.8,
                markersize=5.5,
                capsize=3,
            )

        ax.set_title(metric)
        ax.set_xscale("log")
        ax.set_xticks(shots)
        ax.set_xticklabels([str(s) for s in shots])
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("Score (%)")
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    #fig.suptitle("Few-shot Bootstrap Linear Classification", y=0.985, fontsize=15)
    fig.tight_layout(rect=(0, 0.065, 1, 0.965))

    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{out_stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_score_cell(mean: float, std: float, rank: int) -> str:
    cell = f"{mean:.2f} $\\pm$ {std:.2f}"
    if rank == 1:
        return f"\\textbf{{{cell}}}"
    if rank == 2:
        return f"\\underline{{{cell}}}"
    return cell


def write_f1_latex_table(results: pd.DataFrame, out_dir: Path, out_stem: str) -> Path:
    f1 = results[results["metric"] == "F1"].copy()
    shots = sorted(f1["shot"].unique())
    models = list(results["model"].drop_duplicates())

    ranks_by_shot: dict[int, dict[str, int]] = {}
    for shot in shots:
        shot_df = f1[f1["shot"] == shot].sort_values("mean_percent", ascending=False)
        ranks_by_shot[shot] = {
            row.model: rank
            for rank, row in enumerate(shot_df.itertuples(index=False), start=1)
        }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(shots) + "}",
        r"\hline",
        "Model & " + " & ".join(f"$n={shot}$" for shot in shots) + r" \\",
        r"\hline",
    ]

    for model in models:
        model_cells = []
        for shot in shots:
            row = f1[(f1["model"] == model) & (f1["shot"] == shot)]
            if row.empty:
                model_cells.append("--")
                continue
            record = row.iloc[0]
            rank = ranks_by_shot[shot][model]
            model_cells.append(
                format_score_cell(
                    float(record["mean_percent"]),
                    float(record["std_percent"]),
                    rank,
                )
            )
        lines.append(latex_escape(model) + " & " + " & ".join(model_cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Few-shot bootstrap linear-classification macro F1 "
                r"(\%, mean $\pm$ uncertainty) across sample counts.}"
            ),
            r"\label{tab:fewshot_bootstrap_classification_f1}",
            r"\end{table}",
            "",
        ]
    )

    tex_path = out_dir / f"{out_stem}_f1_table.tex"
    tex_path.write_text("\n".join(lines))
    return tex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where the plot and summary CSV are written.",
    )
    parser.add_argument(
        "--out_stem",
        default="fewshot_bootstrap_classification",
        help="Output filename stem for PDF/PNG/CSV.",
    )
    parser.add_argument(
        "--shots",
        default="1,5,20,100",
        help="Comma-separated few-shot sample counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    results = collect_results(MODEL_SPECS, shots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.out_dir / f"{args.out_stem}_summary.csv"
    results.to_csv(summary_path, index=False)
    tex_path = write_f1_latex_table(results, args.out_dir, args.out_stem)
    plot_results(results, args.out_dir, args.out_stem)

    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {args.out_dir / f'{args.out_stem}.pdf'}")
    print(f"Wrote {args.out_dir / f'{args.out_stem}.png'}")


if __name__ == "__main__":
    main()
