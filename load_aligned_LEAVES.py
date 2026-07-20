#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Independent dataset + dataloader for precomputed LEAVES outputs.

Assumptions about directory structure (as produced by your precompute script):
  out_dir/
    manifest_all.txt              (optional)
    manifest_rank0.txt            (optional)
    manifest_rank1.txt            (optional)
    ...
    <class_name>/
      *.pt                        (per-lightcurve dict)

This script does NOT require a manifest. It can:
  - use an existing manifest (merged or per-rank), OR
  - auto-discover all .pt under out_dir recursively, OR
  - auto-merge any manifest_rank*.txt into an in-memory path list.

It supports any number of ranks (DDP), via DistributedSampler.
"""

import os
import glob
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def _read_manifest(path: str) -> List[str]:
    paths: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    return paths


def _find_rank_manifests(out_dir: str) -> List[str]:
    # manifest_rank0.txt, manifest_rank1.txt, ...
    m = sorted(glob.glob(os.path.join(out_dir, "manifest_rank*.txt")))
    return m


def _discover_pt_files(out_dir: str) -> List[str]:
    # Recursively gather all .pt files under out_dir, excluding manifest*.txt
    pts = sorted(glob.glob(os.path.join(out_dir, "**", "*.pt"), recursive=True))
    return pts


def list_precomputed_paths(
    out_dir: str,
    manifest: Optional[str] = None,
    prefer_manifest_all: bool = True,
    allow_discovery_fallback: bool = True,
) -> List[str]:
    """
    Priority order:
      1) explicit `manifest` argument
      2) out_dir/manifest_all.txt (if prefer_manifest_all and exists)
      3) merge out_dir/manifest_rank*.txt (if any exist)
      4) recursive .pt discovery under out_dir (if allow_discovery_fallback)

    Raises if nothing is found.
    """
    # 1) explicit
    if manifest is not None:
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        paths = _read_manifest(manifest)
        if len(paths) == 0:
            raise RuntimeError(f"Manifest is empty: {manifest}")
        return paths

    # 2) manifest_all
    manifest_all = os.path.join(out_dir, "manifest_all.txt")
    if prefer_manifest_all and os.path.exists(manifest_all):
        paths = _read_manifest(manifest_all)
        if len(paths) > 0:
            return paths

    # 3) merge rank manifests
    rank_manifests = _find_rank_manifests(out_dir)
    if len(rank_manifests) > 0:
        merged: List[str] = []
        for mp in rank_manifests:
            merged.extend(_read_manifest(mp))
        merged = [p for p in merged if p]  # sanitize
        if len(merged) > 0:
            # dedup while preserving order
            seen = set()
            uniq: List[str] = []
            for p in merged:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return uniq

    # 4) fallback to discovery
    if allow_discovery_fallback:
        pts = _discover_pt_files(out_dir)
        if len(pts) > 0:
            return pts

    raise RuntimeError(
        "No precomputed samples found. Provide a manifest or ensure .pt files exist under out_dir."
    )


class PrecomputedLEAVESDataset(Dataset):
    """
    Returns per-sample dict loaded from .pt:
      {
        "X": {"lc": [2000,4], "periodogram": [2000,2], "phase_folded_lc": [2000,4]},
        "Y": {"seven_label": int, "ten_label": int},
        "meta": {...}
      }
    """

    def __init__(
        self,
        out_dir: str,
        manifest: Optional[str] = None,
        paths: Optional[Sequence[str]] = None,
        map_location: str = "cpu",
        validate: bool = True,
    ):
        self.out_dir = out_dir
        self.map_location = map_location

        if paths is not None:
            self.paths = list(paths)
        else:
            self.paths = list_precomputed_paths(out_dir=out_dir, manifest=manifest)

        if len(self.paths) == 0:
            raise RuntimeError("No sample paths found for PrecomputedLEAVESDataset.")

        if validate:
            # light validation: check first sample keys/shapes
            try:
                sample = torch.load(self.paths[0], map_location="cpu", weights_only=True)
            except TypeError:
                sample = torch.load(self.paths[0], map_location="cpu")
            self._validate_sample(sample)

    @staticmethod
    def _validate_sample(sample: Dict[str, Any]) -> None:
        if "X" not in sample or "Y" not in sample:
            raise ValueError("Sample missing required keys: 'X' and/or 'Y'.")
        x = sample["X"]
        if not all(k in x for k in ["lc", "periodogram", "phase_folded_lc"]):
            raise ValueError("Sample['X'] missing one of: lc, periodogram, phase_folded_lc.")
        # Only verify rank-2 shapes; allow different K/N if you ever change them.
        for k in ["lc", "phase_folded_lc"]:
            t = x[k]
            if not (torch.is_tensor(t) and t.ndim == 2 and t.shape[1] == 4):
                raise ValueError(f"Sample['X']['{k}'] must be a tensor [N,4]. Got {type(t)} {getattr(t,'shape',None)}")
        pg = x["periodogram"]
        if not (torch.is_tensor(pg) and pg.ndim == 2 and pg.shape[1] == 2):
            raise ValueError(f"Sample['X']['periodogram'] must be a tensor [K,2]. Got {type(pg)} {getattr(pg,'shape',None)}")
        y = sample["Y"]
        if "seven_label" not in y or "ten_label" not in y:
            raise ValueError("Sample['Y'] missing one of: seven_label, ten_label.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx]
        try:
            sample = torch.load(p, map_location=self.map_location, weights_only=True)
        except TypeError:
            sample = torch.load(p, map_location=self.map_location)
        # For safety, attach path even if not present
        if "meta" not in sample:
            sample["meta"] = {}
        sample["meta"].setdefault("saved_path", p)
        return sample


def precomputed_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates to:
      X:
        lc:              [B,N,4]
        periodogram:     [B,K,2]
        phase_folded_lc: [B,N,4]
        best_period:     [B]
      Y:
        seven_label: [B]
        ten_label:   [B]
      meta: list[dict]
    """
    lc = torch.stack([b["X"]["lc"] for b in batch], dim=0)
    pg = torch.stack([b["X"]["periodogram"] for b in batch], dim=0)
    pflc = torch.stack([b["X"]["phase_folded_lc"] for b in batch], dim=0)
    best_period = torch.tensor(
        [float(b.get("meta", {}).get("best_period", 1.0)) for b in batch],
        dtype=torch.float32,
    )

    y7 = torch.tensor([int(b["Y"]["seven_label"]) for b in batch], dtype=torch.long)
    y10 = torch.tensor([int(b["Y"]["ten_label"]) for b in batch], dtype=torch.long)

    return {
        "X": {"lc": lc, "periodogram": pg, "phase_folded_lc": pflc, "best_period": best_period},
        "Y": {"seven_label": y7, "ten_label": y10},
        "meta": [b.get("meta", {}) for b in batch],
    }


def build_precomputed_loader(
    out_dir: str,
    batch_size: int,
    manifest: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    distributed: bool = False,
    seed: int = 0,
) -> DataLoader:
    """
    If distributed=True, uses DistributedSampler and ignores shuffle at DataLoader level
    (shuffle is handled by the sampler).
    """
    ds = PrecomputedLEAVESDataset(
        out_dir=out_dir,
        manifest=manifest,
        paths=paths,
        map_location="cpu",
        validate=True,
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    sampler = None
    dl_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=drop_last)
        dl_shuffle = False  # sampler decides

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=dl_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=precomputed_collate,
        persistent_workers=persistent_workers,
    )
    return loader


# -------------------------
# Minimal example
# -------------------------
if __name__ == "__main__":
    # Example (single process):
    out_dir = "/home/rui/data/timeseries/aligned_leaves"
    loader = build_precomputed_loader(
        out_dir=out_dir,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        distributed=False,
    )
    batch = next(iter(loader))
    print("lc:", batch["X"]["lc"].shape)
    print("periodogram:", batch["X"]["periodogram"].shape)
    print("phase_folded_lc:", batch["X"]["phase_folded_lc"].shape)
    print("y7:", batch["Y"]["seven_label"].shape, "y10:", batch["Y"]["ten_label"].shape)
    print("meta[0] keys:", list(batch["meta"][0].keys())[:10])
    print("batch:\n" ,batch["X"])
    # Example (DDP training):
    #   torchrun --nproc_per_node=8 train.py
    # In your train.py, set distributed=True and call sampler.set_epoch(epoch) each epoch:
    #
    # loader = build_precomputed_loader(out_dir, batch_size, num_workers=..., distributed=True)
    # for epoch in range(E):
    #     if isinstance(loader.sampler, DistributedSampler):
    #         loader.sampler.set_epoch(epoch)
    #     for batch in loader:
    #         ...
