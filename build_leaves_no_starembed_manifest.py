#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GAIA_RE = re.compile(r"GaiaID(\d+)")
COORD_RE = re.compile(r"(?:^|[^0-9])(?:J)?(\d{6}(?:\.\d+)?)([+-])(\d{6}(?:\.\d+)?)")
SOURCEID_RE = re.compile(r"^[^_]+_(\d+)_\d+(?:__|$)")


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_starembed_paths(root: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = root / split
        manifest = split_dir / "manifest_all.txt"
        if manifest.exists():
            for line in read_manifest(manifest):
                path = Path(line)
                if path.exists():
                    yield path
        elif split_dir.exists():
            yield from sorted(split_dir.rglob("*.pt"))


def sourceid_from_starembed_path(path: Path) -> Optional[int]:
    match = SOURCEID_RE.match(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def hms_to_deg(hms: str) -> float:
    hh = int(hms[0:2])
    mm = int(hms[2:4])
    ss = float(hms[4:])
    return 15.0 * (hh + mm / 60.0 + ss / 3600.0)


def dms_to_deg(sign: str, dms: str) -> float:
    dd = int(dms[0:2])
    mm = int(dms[2:4])
    ss = float(dms[4:])
    value = dd + mm / 60.0 + ss / 3600.0
    return -value if sign == "-" else value


def coord_from_name(name: str) -> Optional[Tuple[float, float]]:
    match = COORD_RE.search(name)
    if not match:
        return None
    return hms_to_deg(match.group(1)), dms_to_deg(match.group(2), match.group(3))


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def path_target_name(path: str) -> str:
    return Path(path).stem.split("__", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a LEAVES manifest excluding only stars duplicated in the StarEmbed preprocessed set."
    )
    parser.add_argument("--leaves_manifest", type=Path, default=Path("/home/rui/data/timeseries/aligned_leaves/manifest_all.txt"))
    parser.add_argument("--starembed_root", type=Path, default=Path("/home/rui/code/algorithm_base/timeseries/starembed_preprocessed"))
    parser.add_argument("--starembed_catalog", type=Path, default=Path("/home/rui/data/timeseries/data_complete/starembedcmGaia.csv"))
    parser.add_argument("--out_manifest", type=Path, default=Path("/home/rui/data/timeseries/aligned_leaves/manifest_no_starembed_duplicates.txt"))
    parser.add_argument("--report", type=Path, default=Path("/home/rui/data/timeseries/aligned_leaves/manifest_no_starembed_duplicates_report.json"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test", "anom"])
    parser.add_argument("--match_arcsec", type=float, default=2.0)
    args = parser.parse_args()

    leaves_paths = read_manifest(args.leaves_manifest)
    sourceids = {
        sourceid
        for sourceid in (sourceid_from_starembed_path(path) for path in iter_starembed_paths(args.starembed_root, args.splits))
        if sourceid is not None
    }
    if not sourceids:
        raise RuntimeError(f"No StarEmbed source ids found under {args.starembed_root}")

    catalog = pd.read_csv(
        args.starembed_catalog,
        usecols=["panstarrs_source_id", "gaia_source_id", "ra", "dec"],
    )
    catalog = catalog[catalog["panstarrs_source_id"].isin(sourceids)].copy()

    gaia_ids = {
        str(int(value))
        for value in catalog["gaia_source_id"].dropna().tolist()
        if math.isfinite(float(value))
    }

    coord_catalog = catalog.dropna(subset=["ra", "dec"]).copy()
    coord_catalog = coord_catalog.drop_duplicates(subset=["ra", "dec"])
    tree = cKDTree(unit_vectors(coord_catalog["ra"].to_numpy(float), coord_catalog["dec"].to_numpy(float)))
    radius = 2.0 * math.sin(math.radians(float(args.match_arcsec) / 3600.0) / 2.0)

    kept: list[str] = []
    excluded_gaia = 0
    excluded_coord = 0
    coord_checked = 0
    coord_unmatched = 0

    for path in leaves_paths:
        name = path_target_name(path)

        gaia_match = GAIA_RE.search(name)
        if gaia_match and gaia_match.group(1) in gaia_ids:
            excluded_gaia += 1
            continue

        coord = coord_from_name(name)
        if coord is not None:
            coord_checked += 1
            vec = unit_vectors(np.array([coord[0]], dtype=float), np.array([coord[1]], dtype=float))[0]
            if tree.query_ball_point(vec, radius):
                excluded_coord += 1
                continue
            coord_unmatched += 1

        kept.append(path)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text("\n".join(kept) + "\n", encoding="utf-8")

    report = {
        "leaves_manifest": str(args.leaves_manifest),
        "starembed_root": str(args.starembed_root),
        "starembed_catalog": str(args.starembed_catalog),
        "splits": list(args.splits),
        "match_arcsec": float(args.match_arcsec),
        "starembed_sourceids": len(sourceids),
        "catalog_rows_for_sourceids": int(len(catalog)),
        "catalog_gaia_ids": len(gaia_ids),
        "catalog_coordinate_rows": int(len(coord_catalog)),
        "leaves_total": len(leaves_paths),
        "leaves_kept": len(kept),
        "leaves_excluded_total": len(leaves_paths) - len(kept),
        "leaves_excluded_gaia": excluded_gaia,
        "leaves_excluded_coordinate": excluded_coord,
        "leaves_coordinate_names_checked": coord_checked,
        "leaves_coordinate_names_unmatched": coord_unmatched,
        "out_manifest": str(args.out_manifest),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
