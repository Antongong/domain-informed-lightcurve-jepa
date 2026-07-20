#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recover sinusoid amplitude and phase with emcee for the single-frequency
injection-recovery dataset.

For each generated light curve, this script fixes P to the injected period and
fits only A and phi in

    mag = A * sin(2 * pi * time / P + phi)

using mag_err as the Gaussian likelihood sigma. It writes one row per target to
sigma_recovery.csv with posterior medians, standard deviations, and the
posterior correlation between A and phi.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import emcee
import numpy as np


DEFAULT_INPUT_DIR = Path("/home/rui/code/algorithm_base/timeseries/clip_experiments/sin_injection_recovery_single_freq")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "sigma_recovery.csv"
N_WALKERS = 32
N_STEPS = 5000
BURN_IN = 1000
THIN = 1
A_PRIOR_MAX = 2.0
TWO_PI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover A and phi for single-frequency sinusoid injections with emcee."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing injection_recovery_single_freq.csv and injection splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path. Defaults to input_dir/sigma_recovery.csv.",
    )
    parser.add_argument(
        "--n-walkers",
        type=int,
        default=N_WALKERS,
        help="Number of emcee walkers per target.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=N_STEPS,
        help="Number of emcee steps per target.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=BURN_IN,
        help="Number of initial steps to discard when summarizing chains.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=THIN,
        help="Thinning factor when flattening chains.",
    )
    parser.add_argument(
        "--a-prior-max",
        type=float,
        default=A_PRIOR_MAX,
        help="Uniform prior upper bound for amplitude A. Prior is 0 <= A <= this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4321,
        help="Random seed for walker initialization and MCMC sampling.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index in metadata, inclusive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of metadata rows to process. 0 means all rows after --start.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing entries and skip ids already present in the output CSV.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N targets. 0 disables progress output.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_for_id(sample_id: str) -> str:
    index = int(sample_id)
    if index < 14_000:
        return "train"
    if index < 16_000:
        return "val"
    return "test"


def light_curve_path(input_dir: Path, sample_id: str) -> Path:
    split = split_for_id(sample_id)
    return input_dir / "injection" / split / f"{sample_id}.csv"


def load_light_curve(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    time = np.asarray(data["time"], dtype=np.float64)
    mag = np.asarray(data["mag"], dtype=np.float64)
    sigma = np.asarray(data["mag_err"], dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(sigma) & (sigma > 0.0)
    if valid.sum() < 3:
        raise ValueError(f"Need at least 3 valid points, got {valid.sum()} in {path}")
    return time[valid], mag[valid], sigma[valid]


def linear_initial_guess(
    time: np.ndarray,
    mag: np.ndarray,
    sigma: np.ndarray,
    period: float,
) -> Tuple[float, float]:
    theta = TWO_PI * time / period
    design = np.column_stack((np.sin(theta), np.cos(theta)))
    weights = 1.0 / np.square(sigma)
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * mag)
    try:
        sin_coef, cos_coef = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        sin_coef, cos_coef = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            mag * np.sqrt(weights),
            rcond=None,
        )[0]

    amplitude = float(np.hypot(sin_coef, cos_coef))
    phi = float(np.mod(np.arctan2(cos_coef, sin_coef), TWO_PI))
    return amplitude, phi


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.mod(phi, TWO_PI)


def angular_delta(phi: np.ndarray, center: float) -> np.ndarray:
    return (phi - center + math.pi) % TWO_PI - math.pi


def initial_walkers(
    rng: np.random.Generator,
    n_walkers: int,
    a0: float,
    phi0: float,
    a_prior_max: float,
) -> np.ndarray:
    a_scale = max(1.0e-4, 0.03 * max(a0, 0.05))
    phi_scale = 0.03
    pos = np.empty((n_walkers, 2), dtype=np.float64)
    pos[:, 0] = rng.normal(loc=a0, scale=a_scale, size=n_walkers)
    pos[:, 1] = rng.normal(loc=phi0, scale=phi_scale, size=n_walkers)
    pos[:, 0] = np.clip(pos[:, 0], 1.0e-8, a_prior_max * (1.0 - 1.0e-8))
    pos[:, 1] = wrap_phi(pos[:, 1])
    return pos


def vectorized_log_prob(
    params: np.ndarray,
    phase_arg: np.ndarray,
    mag: np.ndarray,
    inv_sigma2: np.ndarray,
    log_norm: float,
    a_prior_max: float,
) -> np.ndarray:
    params = np.atleast_2d(params)
    amp = params[:, 0]
    phi = params[:, 1]
    ok = np.isfinite(amp) & np.isfinite(phi) & (amp >= 0.0) & (amp <= a_prior_max)
    out = np.full(params.shape[0], -np.inf, dtype=np.float64)
    if not np.any(ok):
        return out

    model = amp[ok, None] * np.sin(phase_arg[None, :] + phi[ok, None])
    resid = mag[None, :] - model
    out[ok] = -0.5 * np.sum(resid * resid * inv_sigma2[None, :], axis=1) + log_norm
    return out


def summarize_chain(flat_samples: np.ndarray) -> Tuple[float, float, float, float, float]:
    amp = flat_samples[:, 0]
    raw_phi = wrap_phi(flat_samples[:, 1])
    center = float(np.angle(np.mean(np.exp(1j * raw_phi))))
    center = float(np.mod(center, TWO_PI))
    phi_unwrapped = center + angular_delta(raw_phi, center)

    amp_fit = float(np.median(amp))
    phi_fit = float(np.mod(np.median(phi_unwrapped), TWO_PI))
    sigma_a = float(np.std(amp, ddof=1))
    sigma_phi = float(np.std(phi_unwrapped, ddof=1))

    if amp.size > 1 and sigma_a > 0.0 and sigma_phi > 0.0:
        corr = float(np.corrcoef(amp, phi_unwrapped)[0, 1])
    else:
        corr = float("nan")
    return amp_fit, phi_fit, sigma_a, sigma_phi, corr


def fit_one_target(
    row: dict,
    input_dir: Path,
    rng: np.random.Generator,
    n_walkers: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    a_prior_max: float,
) -> Dict[str, str]:
    sample_id = row["id"]
    period = float(row["P"])
    time, mag, sigma = load_light_curve(light_curve_path(input_dir, sample_id))
    phase_arg = TWO_PI * time / period
    inv_sigma2 = 1.0 / np.square(sigma)
    log_norm = -0.5 * float(np.sum(np.log(TWO_PI * np.square(sigma))))

    a0, phi0 = linear_initial_guess(time, mag, sigma, period)
    a0 = min(max(a0, 1.0e-6), a_prior_max * 0.95)
    pos = initial_walkers(rng, n_walkers, a0, phi0, a_prior_max)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        2,
        vectorized_log_prob,
        args=(phase_arg, mag, inv_sigma2, log_norm, a_prior_max),
        vectorize=True,
    )
    sampler.run_mcmc(pos, n_steps, progress=False)
    flat = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    if flat.shape[0] < 2:
        raise ValueError("No posterior samples left after burn-in/thinning")

    a_fit, phi_fit, sigma_a, sigma_phi, corr = summarize_chain(flat)
    return {
        "id": sample_id,
        "split": split_for_id(sample_id),
        "P": f"{period:.12g}",
        "A_true": f"{float(row['A']):.12g}",
        "phi_true": f"{float(row['phi']):.12g}",
        "A_fit": f"{a_fit:.12g}",
        "phi_fit": f"{phi_fit:.12g}",
        "sigma_A": f"{sigma_a:.12g}",
        "sigma_phi": f"{sigma_phi:.12g}",
        "corr_A_phi": f"{corr:.12g}",
    }


def existing_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def select_rows(rows: List[dict], start: int, limit: int) -> List[dict]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    selected = rows[start:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def output_mode(output_path: Path, resume: bool) -> str:
    return "a" if resume and output_path.exists() else "w"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = input_dir / "injection_recovery_single_freq.csv"

    if args.n_walkers < 4:
        raise ValueError("--n-walkers must be at least 4 for a 2D emcee fit")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.burn_in < 0 or args.burn_in >= args.n_steps:
        raise ValueError("--burn-in must satisfy 0 <= burn_in < n_steps")
    if args.thin <= 0:
        raise ValueError("--thin must be positive")
    if args.a_prior_max <= 0.0:
        raise ValueError("--a-prior-max must be positive")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    rows = select_rows(load_metadata(metadata_path), args.start, args.limit)
    done_ids = existing_ids(output_path) if args.resume else set()
    rows = [row for row in rows if row["id"] not in done_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "split",
        "P",
        "A_true",
        "phi_true",
        "A_fit",
        "phi_fit",
        "sigma_A",
        "sigma_phi",
        "corr_A_phi",
    ]
    rng = np.random.default_rng(args.seed)

    mode = output_mode(output_path, args.resume)
    write_header = mode == "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            result = fit_one_target(
                row=row,
                input_dir=input_dir,
                rng=rng,
                n_walkers=args.n_walkers,
                n_steps=args.n_steps,
                burn_in=args.burn_in,
                thin=args.thin,
                a_prior_max=args.a_prior_max,
            )
            writer.writerow(result)
            f.flush()
            if args.progress_every > 0 and (
                idx % args.progress_every == 0 or idx == len(rows)
            ):
                print(f"Recovered {idx}/{len(rows)} targets")

    print(f"Wrote recovery results to {output_path}")


if __name__ == "__main__":
    main()
