#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "runs/sudden_jump_binary_probe_model_sweep"
DEFAULT_MODEL_ORDER = ("EXP13", "EXP14", "EXP15")
DEFAULT_AMP_ORDER = ("amp_0p05_0p1", "amp_0p1_0p3", "amp_0p3_0p5", "amp_0p5_1p0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write LaTeX tables for sudden-jump binary-probe model sweep."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_ORDER))
    parser.add_argument("--amps", nargs="+", default=list(DEFAULT_AMP_ORDER))
    parser.add_argument("--f1-out-name", default="sudden_jump_model_sweep_test_f1_table.tex")
    parser.add_argument("--c-out-name", default="sudden_jump_model_sweep_best_c_table.tex")
    parser.add_argument("--csv-out-name", default="sudden_jump_model_sweep_summary.csv")
    return parser.parse_args()


def amp_label(amp_name: str) -> str:
    parts = amp_name.removeprefix("amp_").replace("p", ".").split("_")
    if len(parts) == 2:
        return f"$U({parts[0]}, {parts[1]})$"
    return amp_name.replace("_", r"\_")


def load_summary(result_root: Path, model: str, amp: str) -> Dict[str, Any]:
    path = result_root / model / amp / "binary_linear_probe_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(result_root: Path, models: List[str], amps: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in models:
        for amp in amps:
            summary = load_summary(result_root, model, amp)
            rows.append(
                {
                    "model": model,
                    "amp": amp,
                    "test_f1": float(summary["metrics"]["test"]["f1"]),
                    "best_c": float(summary["best_c"]),
                }
            )
    return rows


def value_by_model_amp(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for row in rows:
        values.setdefault(str(row["model"]), {})[str(row["amp"])] = float(row[key])
    return values


def write_metric_table(path: Path, *, caption: str, label: str, values: Dict[str, Dict[str, float]], models: List[str], amps: List[str], fmt: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(amps) + "}",
        r"\hline",
        "Model & " + " & ".join(amp_label(amp) for amp in amps) + r" \\",
        r"\hline",
    ]
    for model in models:
        entries = [format(values[model][amp], fmt) for amp in amps]
        lines.append(model + " & " + " & ".join(entries) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "amp", "test_f1", "best_c"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "amp": row["amp"],
                    "test_f1": f"{float(row['test_f1']):.10g}",
                    "best_c": f"{float(row['best_c']):.10g}",
                }
            )


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    models = list(args.models)
    amps = list(args.amps)
    rows = collect_rows(args.result_root, models, amps)
    f1_values = value_by_model_amp(rows, "test_f1")
    c_values = value_by_model_amp(rows, "best_c")

    write_metric_table(
        args.result_root / args.f1_out_name,
        caption="Test F1 for sudden-jump binary linear probes across model settings.",
        label="tab:sudden_jump_model_sweep_test_f1",
        values=f1_values,
        models=models,
        amps=amps,
        fmt=".4f",
    )
    write_metric_table(
        args.result_root / args.c_out_name,
        caption="Best validation-selected logistic-regression $C$ for sudden-jump binary probes.",
        label="tab:sudden_jump_model_sweep_best_c",
        values=c_values,
        models=models,
        amps=amps,
        fmt=".4g",
    )
    write_csv(args.result_root / args.csv_out_name, rows)
    print(f"[OK] wrote {args.result_root / args.f1_out_name}")
    print(f"[OK] wrote {args.result_root / args.c_out_name}")
    print(f"[OK] wrote {args.result_root / args.csv_out_name}")


if __name__ == "__main__":
    main()
