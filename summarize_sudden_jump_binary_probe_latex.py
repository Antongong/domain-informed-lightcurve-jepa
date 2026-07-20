#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "runs/sudden_jump_binary_probe_amplitude_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a LaTeX F1 summary table for sudden-jump binary probes.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--out-name", default="sudden_jump_binary_probe_f1_table.tex")
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_rows(result_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(result_root.glob("amp_*/binary_linear_probe_summary.json")):
        summary = load_summary(summary_path)
        amp_name = summary_path.parent.name.removeprefix("amp_").replace("p", ".")
        metrics = summary["metrics"]
        rows.append(
            {
                "amp_name": amp_name,
                "summary_path": summary_path,
                "best_c": float(summary["best_c"]),
                "train_f1": float(metrics["train"]["f1"]),
                "validation_f1": float(metrics["validation"]["f1"]),
                "test_f1": float(metrics["test"]["f1"]),
                "test_accuracy": float(metrics["test"]["accuracy"]),
                "test_precision": float(metrics["test"]["precision"]),
                "test_recall": float(metrics["test"]["recall"]),
            }
        )
    return rows


def amp_label(amp_name: str) -> str:
    parts = amp_name.split("_")
    if len(parts) == 2:
        return f"$U({parts[0]}, {parts[1]})$"
    return amp_name.replace("_", r"\_")


def write_latex(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Jump amplitude (mag) & Train F1 & Val F1 & Test F1 & Best $C$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{amp_label(row['amp_name'])} & "
            f"{row['train_f1']:.4f} & "
            f"{row['validation_f1']:.4f} & "
            f"{row['test_f1']:.4f} & "
            f"{row['best_c']:.4g} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Binary linear-probe F1 for detecting injected sudden jumps from EXP8 StarEmbed embeddings.}",
            r"\label{tab:sudden_jump_binary_probe_f1}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = find_rows(args.result_root)
    if not rows:
        raise FileNotFoundError(f"No binary_linear_probe_summary.json files found under {args.result_root}/amp_*")
    args.result_root.mkdir(parents=True, exist_ok=True)
    out_path = args.result_root / args.out_name
    write_latex(out_path, rows)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
