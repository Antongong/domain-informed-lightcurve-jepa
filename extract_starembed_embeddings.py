#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _discover_pt_files(split_dir: Path) -> List[Path]:
    manifest = split_dir / "manifest_all.txt"
    if manifest.exists():
        paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path for path in paths if path.exists()]
    return sorted(split_dir.rglob("*.pt"))


def _load_label_map(split_dir: Path) -> Dict[str, Any]:
    label_map_path = split_dir / "label_map.json"
    if label_map_path.exists():
        return json.loads(label_map_path.read_text(encoding="utf-8"))

    folders = [path.name for path in sorted(split_dir.iterdir()) if path.is_dir()]
    label_map = {"folders": folders, "folder_to_idx": {folder: idx for idx, folder in enumerate(folders)}}
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return label_map


def _parse_class_id(class_str: Any, folder: str) -> int:
    value = str(class_str).strip()
    if value:
        try:
            return int(value)
        except ValueError:
            pass

    folder_prefix = str(folder).split("__", 1)[0]
    try:
        return int(folder_prefix)
    except ValueError:
        return -1


class PrecomputedStarEmbedPairDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str, *, limit: int = 0):
        super().__init__()
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.paths = _discover_pt_files(self.split_dir)
        if int(limit) > 0:
            self.paths = self.paths[: int(limit)]
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found under {self.split_dir}")

        self.label_map = _load_label_map(self.split_dir)
        self.folder_to_idx = self.label_map.get("folder_to_idx", {})

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        item = _torch_load(path)

        meta = item.get("meta", {}) or {}
        folder = meta.get("class_folder", path.parent.name)
        class_str = meta.get("class_str", item.get("Y", {}).get("star_class_str", folder))
        y = _parse_class_id(class_str, folder)

        return {
            "g": item["g"]["X"],
            "r": item["r"]["X"],
            "y": y,
            "y_str": str(class_str),
            "meta": meta,
            "path": str(path),
        }


def precomputed_starembed_pair_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    def _stack_band(items: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        entries = list(items)
        periodogram = torch.stack([entry["periodogram"] for entry in entries], dim=0).float()
        log_power = periodogram[..., 1]
        best_idx = torch.argmax(log_power, dim=1)
        batch_idx = torch.arange(periodogram.shape[0], dtype=torch.long)
        return {
            "lc": torch.stack([entry["lc"] for entry in entries], dim=0).float(),
            "periodogram": periodogram,
            "phase_folded_lc": torch.stack([entry["phase_folded_lc"] for entry in entries], dim=0).float(),
            "best_period": periodogram[batch_idx, best_idx, 0].float(),
            "best_power": torch.pow(10.0, log_power[batch_idx, best_idx]).float(),
        }

    return {
        "g": {"X": _stack_band(sample["g"] for sample in batch)},
        "r": {"X": _stack_band(sample["r"] for sample in batch)},
        "Y": {
            "star_class": torch.tensor([sample["y"] for sample in batch], dtype=torch.long),
            "star_class_str": [sample["y_str"] for sample in batch],
        },
        "meta": [sample["meta"] for sample in batch],
        "paths": [sample["path"] for sample in batch],
    }


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def load_model_from_ckpt(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> None:
    ckpt = _torch_load(ckpt_path)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state = ckpt["model_state_dict"]
    else:
        state = {key: value for key, value in ckpt.items() if torch.is_tensor(value)}
    model.load_state_dict(state, strict=strict)


def build_model_from_config(cfg: Dict[str, Any]) -> torch.nn.Module:
    from train_ddp_numeric import build_model  # type: ignore

    return build_model(cfg)


def extract_repr(
    model_out: Dict[str, Any],
    out_info: str,
    views_order: Optional[Sequence[str]],
    repr_mode: str,
) -> torch.Tensor:
    if out_info not in model_out:
        raise KeyError(f"model output missing branch '{out_info}'")

    view_dict = model_out[out_info]
    if not isinstance(view_dict, dict) or not view_dict:
        raise ValueError(f"model_out['{out_info}'] must be a non-empty dict")

    views = list(views_order) if views_order is not None else sorted(view_dict.keys())
    missing = [view for view in views if view not in view_dict]
    if missing:
        raise KeyError(f"missing views in model_out['{out_info}']: {missing}")

    feats = [view_dict[view] for view in views]
    if repr_mode == "mean":
        return torch.stack(feats, dim=0).mean(dim=0)
    if repr_mode == "concat":
        return torch.cat(feats, dim=-1)
    raise ValueError(f"Unsupported repr_mode={repr_mode!r}")


class FeatureExtractorRunner(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        out_infos: Sequence[str],
        views_order: Optional[Sequence[str]],
        repr_mode: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.out_infos = list(out_infos)
        self.views_order = list(views_order) if views_order is not None else None
        self.repr_mode = repr_mode

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        out = self.model(batch)
        reps = [extract_repr(out, out_info, self.views_order, self.repr_mode) for out_info in self.out_infos]
        return torch.cat(reps, dim=-1) if len(reps) > 1 else reps[0]


def build_runner(
    model: torch.nn.Module,
    *,
    device: torch.device,
    out_infos: Sequence[str],
    views_order: Optional[Sequence[str]],
    repr_mode: str,
    global_batch_size: int,
    num_gpus: int,
) -> Tuple[torch.nn.Module, bool]:
    runner = FeatureExtractorRunner(
        model,
        out_infos=out_infos,
        views_order=views_order,
        repr_mode=repr_mode,
    ).to(device)

    if device.type != "cuda":
        return runner, False

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 1:
        print("[Init] using a single GPU", flush=True)
        return runner, False

    requested_gpus = available_gpus if int(num_gpus) <= 0 else min(int(num_gpus), available_gpus)
    if requested_gpus <= 1:
        print("[Init] using a single GPU", flush=True)
        return runner, False

    device_ids = list(range(requested_gpus))
    runner = torch.nn.DataParallel(runner, device_ids=device_ids)
    print(
        f"[Init] using DataParallel on {requested_gpus} GPUs; global_batch_size={global_batch_size}",
        flush=True,
    )
    return runner, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract StarEmbed benchmark features from precomputed numeric inputs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/rui/code/algorithm_base/timeseries/starembed_preprocessed",
        help="Root containing precomputed StarEmbed splits.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test", "anom"])
    parser.add_argument("--out_dir", type=str, default="runs/starembed")
    parser.add_argument("--out_info", type=str, default="embeddings", choices=["embeddings", "projections"])
    parser.add_argument("--out_infos", nargs="*", default=None)
    parser.add_argument("--repr_mode", type=str, default="concat", choices=["mean", "concat"])
    parser.add_argument("--views_order", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of CUDA GPUs to use. Default 0 uses all visible GPUs.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save_y_str", dest="save_y_str", action="store_true", default=True)
    parser.add_argument("--no_save_y_str", dest="save_y_str", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model = build_model_from_config(cfg)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    load_model_from_ckpt(model, args.ckpt, strict=bool(args.strict))

    out_infos = args.out_infos if args.out_infos else [args.out_info]
    views_order = args.views_order
    if views_order is None:
        views_order = cfg.get("loss", {}).get("views_order", None)

    runner, use_data_parallel = build_runner(
        model,
        device=device,
        out_infos=out_infos,
        views_order=views_order,
        repr_mode=args.repr_mode,
        global_batch_size=int(args.batch_size),
        num_gpus=int(args.num_gpus),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "config": args.config,
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "splits": list(args.splits),
        "out_infos": list(out_infos),
        "repr_mode": args.repr_mode,
        "views_order": list(views_order) if views_order is not None else None,
        "batch_size": int(args.batch_size),
        "limit": int(args.limit),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "num_gpus": int(args.num_gpus),
    }

    for split in args.splits:
        dataset = PrecomputedStarEmbedPairDataset(args.data_root, split, limit=int(args.limit))
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=precomputed_starembed_pair_collate,
            persistent_workers=(int(args.num_workers) > 0),
        )

        x_parts: List[np.ndarray] = []
        x_avg_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        y_str_parts: List[np.ndarray] = []

        progress = tqdm(total=len(dataset), desc=f"Extracting {split}", unit="lc") if tqdm is not None else None
        try:
            with torch.no_grad():
                for batch in loader:
                    band_g = {"X": batch["g"]["X"]}
                    band_r = {"X": batch["r"]["X"]}

                    if use_data_parallel:
                        rep_g = runner(band_g)
                        rep_r = runner(band_r)
                    else:
                        rep_g = runner(move_to_device(band_g, device))
                        rep_r = runner(move_to_device(band_r, device))

                    x = torch.cat([rep_g, rep_r], dim=1)
                    x_avg = 0.5 * (rep_g + rep_r)

                    x_parts.append(x.detach().cpu().numpy().astype(np.float32))
                    x_avg_parts.append(x_avg.detach().cpu().numpy().astype(np.float32))
                    y_parts.append(batch["Y"]["star_class"].detach().cpu().numpy().astype(np.int64))
                    if args.save_y_str:
                        y_str_parts.append(np.asarray(batch["Y"]["star_class_str"], dtype=object))

                    if progress is not None:
                        progress.update(int(batch["Y"]["star_class"].shape[0]))
        finally:
            if progress is not None:
                progress.close()

        x_all = np.concatenate(x_parts, axis=0) if x_parts else np.zeros((0, 0), dtype=np.float32)
        x_avg_all = np.concatenate(x_avg_parts, axis=0) if x_avg_parts else np.zeros((0, 0), dtype=np.float32)
        y_all = np.concatenate(y_parts, axis=0) if y_parts else np.zeros((0,), dtype=np.int64)

        out_path = out_dir / f"starembed_embeddings_{split}.npz"
        if args.save_y_str:
            y_str_all = np.concatenate(y_str_parts, axis=0) if y_str_parts else np.zeros((0,), dtype=object)
            np.savez_compressed(out_path, x=x_all, x_avg=x_avg_all, y=y_all, y_str=y_str_all)
        else:
            np.savez_compressed(out_path, x=x_all, x_avg=x_avg_all, y=y_all)

        label_map = dict(dataset.label_map)
        label_map["paper_class_order"] = [1, 2, 4, 5, 6, 8, 13]
        label_map["class_name_by_id"] = {
            "1": "EW",
            "2": "EA",
            "4": "RRab",
            "5": "RRc",
            "6": "RRd",
            "8": "RS_CVn",
            "13": "LPV",
        }
        (out_dir / f"label_map_{split}.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_path} : x={x_all.shape}, x_avg={x_avg_all.shape}, y={y_all.shape}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
