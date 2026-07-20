#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code generate the injection signal to the light curve for amplitude recovery
It would use the time of LEAVES sample times as the t
the error of mag is fixed to be 0.01 mag
y = A * sin(2 * pi * t / P + phi) + noise
P from 2.5 days to 100 days sampled log-uniformly
A from 0.05 mag to 1 mag sampled log-uniformly
phi from 0 to 2 * pi sampled uniformly

Sample 20000 light curves in total
the time should be more than 100 points
the total time span should be more than 100 days

record a csv file with the following columns:
id, P, A, phi
in output_dir/injection_recovery_single_freq.csv


The light curves are recorded in
in output_dir/injection/{train/val/test}/{id}.csv

it would be 7:1:2 split for train, val and test

"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


DEFAULT_LEAVES_DIR = Path("/home/rui/data/timeseries/aligned_leaves")
DEFAULT_OUTPUT_DIR = Path("sin_injection_recovery_single_freq")

N_LIGHT_CURVES = 20_000
N_TIME_TEMPLATES = 20_000
MIN_POINTS = 100
MIN_TIME_SPAN_DAYS = 100.0
MAG_ERR = 0.01
P_MIN_DAYS = 2.5
P_MAX_DAYS = 100.0
A_MIN_MAG = 0.05
A_MAX_MAG = 1.0


@dataclass(frozen=True)
class TimeTemplate:
    source_path: Path
    times: np.ndarray


@dataclass(frozen=True)
class InjectionSample:
    sample_id: str
    split: str
    period: float
    amplitude: float
    phi: float
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate single-frequency sinusoid injection-recovery light curves."
    )
    parser.add_argument(
        "--leaves-dir",
        type=Path,
        default=DEFAULT_LEAVES_DIR,
        help="Directory containing precomputed aligned LEAVES .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Defaults to runs/injection_recovery_single_freq.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_LIGHT_CURVES,
        help="Number of injected light curves to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for time-template selection, parameters, noise, and split order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output metadata file and injection directory before writing.",
    )
    parser.add_argument(
        "--max-source-samples",
        type=int,
        default=0,
        help="Maximum number of LEAVES .pt files to scan. 0 means scan all files.",
    )
    parser.add_argument(
        "--n-time-templates",
        type=int,
        default=N_TIME_TEMPLATES,
        help="Number of eligible LEAVES time templates to collect. 0 means collect all.",
    )
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=1000,
        help="Print source-scan progress every N files. 0 disables progress output.",
    )
    return parser.parse_args()


def iter_pt_paths(leaves_dir: Path) -> Iterable[Path]:
    manifest_all = leaves_dir / "manifest_all.txt"
    if manifest_all.exists():
        with manifest_all.open("r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if path:
                    yield Path(path)
        return

    rank_manifests = sorted(leaves_dir.glob("manifest_rank*.txt"))
    if rank_manifests:
        seen = set()
        for manifest in rank_manifests:
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    path = line.strip()
                    if path and path not in seen:
                        seen.add(path)
                        yield Path(path)
        return

    yield from sorted(leaves_dir.rglob("*.pt"))


def load_sample(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_valid_times(sample: dict) -> np.ndarray:
    lc = sample["X"]["lc"]
    if torch.is_tensor(lc):
        lc_np = lc.detach().cpu().numpy()
    else:
        lc_np = np.asarray(lc)

    times = lc_np[:, 0].astype(np.float64, copy=False)
    if lc_np.shape[1] >= 4:
        valid = lc_np[:, 3] > 0
    elif lc_np.shape[1] >= 3:
        valid = lc_np[:, 2] > 0
    else:
        valid = np.ones_like(times, dtype=bool)

    times = times[valid & np.isfinite(times)]
    if times.size == 0:
        return times

    return np.sort(times)


def discover_time_templates(
    leaves_dir: Path,
    n_time_templates: int = N_TIME_TEMPLATES,
    max_source_samples: int = 0,
    progress_every: int = 1000,
) -> List[TimeTemplate]:
    templates: List[TimeTemplate] = []
    for scanned, path in enumerate(iter_pt_paths(leaves_dir), start=1):
        if max_source_samples > 0 and scanned > max_source_samples:
            break

        try:
            times = extract_valid_times(load_sample(path))
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            print(f"Skipping unreadable sample {path}: {exc}")
            continue

        if times.size > MIN_POINTS and float(times[-1] - times[0]) > MIN_TIME_SPAN_DAYS:
            templates.append(TimeTemplate(source_path=path, times=times))
            if n_time_templates > 0 and len(templates) >= n_time_templates:
                print(f"Collected {len(templates)} eligible time templates")
                break

        if progress_every > 0 and scanned % progress_every == 0:
            print(f"Scanned {scanned} source samples; eligible templates: {len(templates)}")

    if not templates:
        raise RuntimeError(
            "No LEAVES samples passed the filters: "
            f"n_points > {MIN_POINTS}, time_span > {MIN_TIME_SPAN_DAYS} days."
        )
    return templates


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def split_for_index(index: int, n_samples: int) -> str:
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.1)
    if index < n_train:
        return "train"
    if index < n_train + n_val:
        return "val"
    return "test"


def make_light_curve(
    times: np.ndarray,
    period: float,
    amplitude: float,
    phi: float,
    np_rng: np.random.Generator,
) -> np.ndarray:
    signal = amplitude * np.sin((2.0 * math.pi * times / period) + phi)
    noise = np_rng.normal(loc=0.0, scale=MAG_ERR, size=times.shape)
    mag = signal + noise
    mag_err = np.full_like(times, MAG_ERR, dtype=np.float64)
    return np.column_stack((times, mag, mag_err))


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    metadata_path = output_dir / "injection_recovery_single_freq.csv"
    injection_dir = output_dir / "injection"

    if overwrite:
        if metadata_path.exists():
            metadata_path.unlink()
        if injection_dir.exists():
            shutil.rmtree(injection_dir)
    elif metadata_path.exists() or injection_dir.exists():
        raise FileExistsError(
            f"{output_dir} already contains injection output. "
            "Use --overwrite to replace it."
        )

    for split in ("train", "val", "test"):
        (injection_dir / split).mkdir(parents=True, exist_ok=True)


def write_light_curve(path: Path, light_curve: np.ndarray) -> None:
    np.savetxt(
        path,
        light_curve,
        delimiter=",",
        header="time,mag,mag_err",
        comments="",
        fmt=("%.10f", "%.10f", "%.10f"),
    )


def write_metadata(path: Path, samples: Sequence[InjectionSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "P", "A", "phi"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "id": sample.sample_id,
                    "P": f"{sample.period:.12g}",
                    "A": f"{sample.amplitude:.12g}",
                    "phi": f"{sample.phi:.12g}",
                }
            )


def generate_dataset(
    templates: Sequence[TimeTemplate],
    output_dir: Path,
    n_samples: int,
    seed: int,
) -> List[InjectionSample]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    samples: List[InjectionSample] = []

    for index in range(n_samples):
        template = rng.choice(templates)
        period = log_uniform(rng, P_MIN_DAYS, P_MAX_DAYS)
        amplitude = log_uniform(rng, A_MIN_MAG, A_MAX_MAG)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        split = split_for_index(index, n_samples)
        sample_id = f"{index:05d}"

        light_curve = make_light_curve(
            times=template.times,
            period=period,
            amplitude=amplitude,
            phi=phi,
            np_rng=np_rng,
        )
        write_light_curve(output_dir / "injection" / split / f"{sample_id}.csv", light_curve)
        samples.append(
            InjectionSample(
                sample_id=sample_id,
                split=split,
                period=period,
                amplitude=amplitude,
                phi=phi,
                source_path=template.source_path,
            )
        )

        if (index + 1) % 1000 == 0 or index + 1 == n_samples:
            print(f"Generated {index + 1}/{n_samples} light curves")

    return samples


def main() -> None:
    args = parse_args()
    leaves_dir = args.leaves_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if args.n_time_templates < 0:
        raise ValueError("--n-time-templates must be non-negative.")
    if args.max_source_samples < 0:
        raise ValueError("--max-source-samples must be non-negative.")
    if args.scan_progress_every < 0:
        raise ValueError("--scan-progress-every must be non-negative.")
    if not leaves_dir.exists():
        raise FileNotFoundError(f"LEAVES directory not found: {leaves_dir}")

    prepare_output(output_dir=output_dir, overwrite=args.overwrite)
    templates = discover_time_templates(
        leaves_dir,
        n_time_templates=args.n_time_templates,
        max_source_samples=args.max_source_samples,
        progress_every=args.scan_progress_every,
    )
    print(f"Found {len(templates)} eligible LEAVES time templates")

    samples = generate_dataset(
        templates=templates,
        output_dir=output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    write_metadata(output_dir / "injection_recovery_single_freq.csv", samples)
    print(f"Wrote metadata to {output_dir / 'injection_recovery_single_freq.csv'}")


if __name__ == "__main__":
    main()
