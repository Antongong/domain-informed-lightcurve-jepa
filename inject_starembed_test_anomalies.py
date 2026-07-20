#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_DATA_ROOT = Path("/home/rui/code/algorithm_base/timeseries/starembed_preprocessed")
DEFAULT_OUT_ROOT = Path("/home/rui/code/algorithm_base/timeseries/starembed_preprocessed_injected_anomalies")
EVENT_TYPES = ("sudden_jump", "weather_anomaly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject synthetic anomaly events into the precomputed StarEmbed test split. "
            "Each event type is written as a separate StarEmbed-compatible split root."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--splits", nargs="+", default=None, help="Optional list of splits to process, e.g. train validation test.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--event-types", nargs="+", choices=EVENT_TYPES, default=list(EVENT_TYPES))
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit. 0 means all samples.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sudden-jump-min-mag", type=float, default=0.50)
    parser.add_argument("--sudden-jump-max-mag", type=float, default=1.00)

    parser.add_argument(
        "--skip-recompute-views",
        action="store_true",
        help="Only update raw lc tensors and leave periodogram/phase_folded_lc copied from source.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of CUDA GPUs to use. 0 means all visible GPUs when --device starts with cuda.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA device ids to use, e.g. 0,1,2,3. Overrides --num-gpus.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--period-min", type=float, default=10.0 / (60.0 * 24.0))
    parser.add_argument("--period-max", type=float, default=2000.0)
    parser.add_argument("--k-periods", type=int, default=1_000_000)
    parser.add_argument("--k-out", type=int, default=1000)
    parser.add_argument("--k-top", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    return parser.parse_args()


def torch_load(path: Path) -> Any:
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


def deterministic_rng(seed: int, event_type: str, rel_path: str) -> np.random.Generator:
    key = f"{seed}|{event_type}|{rel_path}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    sub_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(sub_seed)


def valid_positions(lc: torch.Tensor) -> np.ndarray:
    arr = lc.detach().cpu().numpy()
    mask = arr[:, 3] > 0
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    return np.flatnonzero(mask & finite)


def inject_sudden_jump(
    item: Dict[str, Any],
    rng: np.random.Generator,
    *,
    min_mag: float,
    max_mag: float,
) -> Dict[str, Any]:
    q = float(rng.uniform(0.20, 0.80))
    delta_abs = float(rng.uniform(float(min_mag), float(max_mag)))
    sign = int(1 if rng.random() < 0.5 else -1)
    delta = float(sign * delta_abs)

    info: Dict[str, Any] = {
        "q": q,
        "delta_mag_abs": delta_abs,
        "sign": sign,
        "delta_mag": delta,
        "bands": {},
    }

    for band in ("g", "r"):
        lc = item[band]["X"]["lc"].clone()
        pos = valid_positions(lc)
        band_info: Dict[str, Any] = {"n_valid": int(pos.size), "n_modified": 0}
        if pos.size > 0:
            start_rank = int(np.floor(q * float(pos.size - 1)))
            changed = pos[start_rank:]
            lc[torch.as_tensor(changed, dtype=torch.long), 1] += delta
            band_info.update(
                {
                    "start_rank": int(start_rank),
                    "start_index": int(pos[start_rank]),
                    "start_time": float(lc[pos[start_rank], 0].item()),
                    "n_modified": int(changed.size),
                }
            )
        item[band]["X"]["lc"] = lc
        info["bands"][band] = band_info

    return info


def inject_weather_anomaly(
    item: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    p = float(rng.uniform(0.05, 0.10))
    delta = 5.0
    info: Dict[str, Any] = {
        "p": p,
        "delta_mag": delta,
        "bands": {},
    }

    for band in ("g", "r"):
        lc = item[band]["X"]["lc"].clone()
        pos = valid_positions(lc)
        n_pick = int(round(p * float(pos.size)))
        if pos.size > 0:
            n_pick = max(1, min(n_pick, int(pos.size)))
        changed = np.asarray([], dtype=np.int64)
        if n_pick > 0:
            changed = np.sort(rng.choice(pos, size=n_pick, replace=False).astype(np.int64, copy=False))
            lc[torch.as_tensor(changed, dtype=torch.long), 1] += delta
        item[band]["X"]["lc"] = lc
        info["bands"][band] = {
            "n_valid": int(pos.size),
            "n_modified": int(changed.size),
            "indices": changed.astype(int).tolist(),
            "times": [float(lc[int(idx), 0].item()) for idx in changed],
        }

    return info


@torch.no_grad()
def gls_batch(
    t: torch.Tensor,
    y: torch.Tensor,
    yerr: torch.Tensor,
    mask: torch.Tensor,
    periods: torch.Tensor,
    *,
    eps: float,
    chunk_size: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if t.ndim != 2 or y.ndim != 2 or yerr.ndim != 2 or mask.ndim != 2:
        raise ValueError("t, y, yerr, and mask must have shape [B, N].")

    device = t.device
    bsz = t.shape[0]
    t = t.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)
    yerr = yerr.to(device=device, dtype=dtype)
    periods = periods.to(device=device, dtype=dtype).clamp_min(eps)

    valid = mask & (yerr > 0)
    w = torch.zeros_like(y, dtype=dtype)
    w[valid] = 1.0 / (yerr[valid] ** 2)

    w_sum = torch.sum(w, dim=1).clamp_min(eps)
    wy_sum = torch.sum(w * y, dim=1)
    wyy_sum = torch.sum(w * y * y, dim=1)
    var_y = (wyy_sum - (wy_sum * wy_sum) / w_sum).clamp_min(eps)

    w_sum_1 = w_sum.view(bsz, 1)
    wy_sum_1 = wy_sum.view(bsz, 1)
    var_y_1 = var_y.view(bsz, 1)

    k_total = periods.numel()
    power_all = torch.empty((bsz, k_total), dtype=dtype, device=device)
    best_power = torch.full((bsz,), -float("inf"), dtype=dtype, device=device)
    best_period = torch.zeros((bsz,), dtype=dtype, device=device)

    t3 = t.unsqueeze(-1)
    y3 = y.unsqueeze(-1)
    w3 = w.unsqueeze(-1)

    for k0 in range(0, k_total, int(chunk_size)):
        k1 = min(k0 + int(chunk_size), k_total)
        omega = (2.0 * math.pi / periods[k0:k1]).view(1, 1, -1)
        phase = t3 * omega
        c = torch.cos(phase)
        s = torch.sin(phase)

        c_sum = torch.sum(w3 * c, dim=1)
        s_sum = torch.sum(w3 * s, dim=1)
        cc_sum = torch.sum(w3 * c * c, dim=1)
        ss_sum = torch.sum(w3 * s * s, dim=1)
        cs_sum = torch.sum(w3 * c * s, dim=1)
        yc_sum = torch.sum(w3 * y3 * c, dim=1)
        ys_sum = torch.sum(w3 * y3 * s, dim=1)

        yc0 = yc_sum - (wy_sum_1 * c_sum) / w_sum_1
        ys0 = ys_sum - (wy_sum_1 * s_sum) / w_sum_1
        denom = (cc_sum * ss_sum - cs_sum * cs_sum).clamp_min(eps)
        num = ss_sum * yc0 * yc0 + cc_sum * ys0 * ys0 - 2.0 * cs_sum * yc0 * ys0
        power = torch.nan_to_num(num / (denom * var_y_1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        power_all[:, k0:k1] = power
        chunk_max, chunk_arg = torch.max(power, dim=1)
        better = chunk_max > best_power
        if bool(better.any()):
            best_power[better] = chunk_max[better]
            global_idx = (chunk_arg + k0).to(torch.long)
            best_period[better] = periods[global_idx[better]]

    return power_all, best_power, best_period


@torch.no_grad()
def select_periodogram_points(
    power: torch.Tensor,
    periods: torch.Tensor,
    *,
    k_out: int,
    k_top: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, k_total = power.shape
    if k_out >= k_total:
        periods_out = periods.to(power.device).view(1, k_total).expand(bsz, k_total)
        return power, periods_out

    k_out = int(k_out)
    k_top = max(0, min(int(k_top), k_out))
    k_rand = k_out - k_top

    if k_top > 0:
        top_idx = torch.topk(power, k=k_top, dim=1, largest=True, sorted=False).indices
    else:
        top_idx = torch.empty((bsz, 0), dtype=torch.long, device=power.device)

    all_idx = torch.arange(k_total, dtype=torch.long, device=power.device).view(1, k_total).expand(bsz, k_total)
    if k_top > 0:
        remain_mask = torch.ones((bsz, k_total), dtype=torch.bool, device=power.device)
        remain_mask.scatter_(1, top_idx, False)
        remain_idx = all_idx[remain_mask].view(bsz, k_total - k_top)
    else:
        remain_idx = all_idx

    if k_rand > 0:
        gen = torch.Generator(device=power.device)
        gen.manual_seed(int(seed))
        draw = torch.rand((bsz, remain_idx.shape[1]), device=power.device, generator=gen)
        pick = torch.topk(draw, k=k_rand, dim=1, largest=True, sorted=False).indices
        rand_idx = torch.gather(remain_idx, 1, pick)
    else:
        rand_idx = torch.empty((bsz, 0), dtype=torch.long, device=power.device)

    keep_idx = torch.cat([top_idx, rand_idx], dim=1)
    keep_idx, _ = torch.sort(keep_idx, dim=1)
    selected_power = torch.gather(power, 1, keep_idx)
    selected_periods = periods.to(power.device)[keep_idx]
    return selected_power, selected_periods


def prepare_lc_batch(raw_lcs: Sequence[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.stack([lc.float() for lc in raw_lcs], dim=0).to(device)
    time = torch.nan_to_num(raw[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    mag = torch.nan_to_num(raw[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    mag_err = torch.nan_to_num(raw[..., 2].abs(), nan=0.1, posinf=0.1, neginf=0.1)
    mask = (raw[..., 3] > 0) & torch.isfinite(raw[..., 0]) & torch.isfinite(raw[..., 1]) & torch.isfinite(raw[..., 2])
    bad_err = (~torch.isfinite(mag_err)) | (mag_err <= 0.0) | (mag_err > 20.0)
    mag_err = torch.where(bad_err, torch.full_like(mag_err, 0.1), mag_err)

    sort_key = torch.where(mask, time, torch.full_like(time, float("inf")))
    sort_idx = torch.argsort(sort_key, dim=1)
    time = torch.gather(time, 1, sort_idx)
    mag = torch.gather(mag, 1, sort_idx)
    mag_err = torch.gather(mag_err, 1, sort_idx)
    mask = torch.gather(mask, 1, sort_idx)
    return time, mag, mag_err, mask


@torch.no_grad()
def recompute_band_views(
    raw_lcs: Sequence[torch.Tensor],
    periods: torch.Tensor,
    *,
    device: torch.device,
    args: argparse.Namespace,
    selection_seed: int,
) -> Dict[str, torch.Tensor]:
    time, mag, mag_err, mask = prepare_lc_batch(raw_lcs, device=device)
    mask_f = mask.to(torch.float32)

    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_mag = (mag * mask_f).sum(dim=1, keepdim=True) / denom
    mag_mean_sub = (mag - mean_mag) * mask_f
    time = time * mask_f
    mag_err_masked = mag_err * mask_f

    power, best_power, best_period = gls_batch(
        time,
        mag,
        mag_err.clamp_min(float(args.eps)),
        mask,
        periods,
        eps=float(args.eps),
        chunk_size=int(args.chunk_size),
    )
    selected_power, selected_periods = select_periodogram_points(
        power,
        periods,
        k_out=int(args.k_out),
        k_top=int(args.k_top),
        seed=int(selection_seed),
    )

    phase_period = best_period.view(-1, 1).clamp_min(float(args.period_min))
    phase_time = (time - phase_period * torch.floor(time / phase_period)) * mask_f
    lc = torch.stack([time, mag_mean_sub, mag_err_masked, mask_f], dim=-1).float().cpu()
    phase_folded_lc = torch.stack([phase_time, mag_mean_sub, mag_err_masked, mask_f], dim=-1).float().cpu()
    periodogram = torch.stack(
        [selected_periods, torch.log10(selected_power.clamp_min(float(args.eps)))],
        dim=-1,
    ).float().cpu()

    return {
        "lc": lc,
        "periodogram": periodogram,
        "phase_folded_lc": phase_folded_lc,
        "best_period": best_period.float().cpu(),
        "best_power": best_power.float().cpu(),
    }


def batched(seq: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def output_path_for(src_path: Path, src_split_dir: Path, dst_split_dir: Path) -> Path:
    rel = src_path.relative_to(src_split_dir)
    return dst_split_dir / rel


def write_label_map(src_split_dir: Path, dst_split_dir: Path) -> None:
    src_label_map = src_split_dir / "label_map.json"
    if src_label_map.exists():
        shutil.copy2(src_label_map, dst_split_dir / "label_map.json")
        return

    folders = [path.name for path in sorted(dst_split_dir.iterdir()) if path.is_dir()]
    label_map = {"folders": folders, "folder_to_idx": {folder: i for i, folder in enumerate(folders)}}
    (dst_split_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def resolve_devices(args: argparse.Namespace) -> List[str]:
    if not str(args.device).startswith("cuda"):
        return [str(args.device)]
    if not torch.cuda.is_available():
        raise RuntimeError("--device starts with cuda, but CUDA is not available")

    visible = torch.cuda.device_count()
    if str(args.gpu_ids).strip():
        gpu_ids = [int(part.strip()) for part in str(args.gpu_ids).split(",") if part.strip()]
    else:
        n_gpus = int(args.num_gpus)
        if n_gpus <= 0:
            n_gpus = visible
        gpu_ids = list(range(min(n_gpus, visible)))

    if not gpu_ids:
        raise ValueError("No CUDA devices selected")
    bad = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible]
    if bad:
        raise ValueError(f"Requested CUDA device ids outside visible range 0..{visible - 1}: {bad}")
    return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]


def write_event_config(event_type: str, paths: Sequence[Path], args: argparse.Namespace, devices: Sequence[str]) -> None:
    event_root = args.out_root / event_type
    config_path = event_root / "injection_config.json"
    config = {
        "event_type": event_type,
        "data_root": str(args.data_root),
        "split": str(args.split),
        "splits": list(getattr(args, "active_splits", [args.split])),
        "out_root": str(args.out_root),
        "seed": int(args.seed),
        "devices": list(devices),
        "batch_size_per_worker": int(args.batch_size),
        "sudden_jump_min_mag": float(args.sudden_jump_min_mag),
        "sudden_jump_max_mag": float(args.sudden_jump_max_mag),
        "skip_recompute_views": bool(args.skip_recompute_views),
        "period_min": float(args.period_min),
        "period_max": float(args.period_max),
        "k_periods": int(args.k_periods),
        "k_out": int(args.k_out),
        "k_top": int(args.k_top),
        "chunk_size": int(args.chunk_size),
        "eps": float(args.eps),
        "n_samples": int(len(paths)),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def merge_rank_outputs(
    event_type: str,
    paths: Sequence[Path],
    args: argparse.Namespace,
    world_size: int,
) -> None:
    src_split_dir = args.data_root / args.split
    event_root = args.out_root / event_type
    dst_split_dir = event_root / args.split
    manifest_path = dst_split_dir / "manifest_all.txt"
    info_path = event_root / "injection_info.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for src_path in paths:
            manifest_f.write(str(output_path_for(src_path, src_split_dir, dst_split_dir)) + "\n")

    with info_path.open("w", encoding="utf-8") as info_f:
        for rank in range(world_size):
            rank_info_path = event_root / f"injection_info_rank{rank}.jsonl"
            if rank_info_path.exists():
                with rank_info_path.open("r", encoding="utf-8") as rank_f:
                    shutil.copyfileobj(rank_f, info_f)

    write_label_map(src_split_dir, dst_split_dir)


def process_event_type_rank(
    rank: int,
    event_type: str,
    paths: Sequence[Path],
    args: argparse.Namespace,
    devices: Sequence[str],
) -> None:
    world_size = len(devices)
    shard_paths = list(paths)[rank::world_size]
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    process_event_type_shard(
        event_type=event_type,
        paths=shard_paths,
        args=args,
        rank=rank,
        world_size=world_size,
        device=device,
        show_progress=(rank == 0),
    )


def process_event_type_shard(
    *,
    event_type: str,
    paths: Sequence[Path],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    show_progress: bool,
) -> None:
    src_split_dir = args.data_root / args.split
    event_root = args.out_root / event_type
    dst_split_dir = event_root / args.split
    info_path = event_root / f"injection_info_rank{rank}.jsonl"
    manifest_path = dst_split_dir / f"manifest_rank{rank}.txt"

    dst_split_dir.mkdir(parents=True, exist_ok=True)
    periods = None
    if not args.skip_recompute_views:
        periods = torch.exp(
            torch.linspace(
                math.log(float(args.period_min)),
                math.log(float(args.period_max)),
                int(args.k_periods),
                device=device,
                dtype=torch.float32,
            )
        )

    progress = batched(paths, int(args.batch_size))
    if tqdm is not None and show_progress:
        n_batches = (len(paths) + int(args.batch_size) - 1) // int(args.batch_size)
        progress = tqdm(progress, total=n_batches, desc=f"inject {event_type} rank{rank}/{world_size}")

    written_paths: List[Path] = []
    with info_path.open("w", encoding="utf-8") as info_f:
        for batch_idx, batch_paths in enumerate(progress):
            loaded: List[Dict[str, Any]] = []
            records: List[Dict[str, Any]] = []

            for src_path in batch_paths:
                item = torch_load(src_path)
                rel_path = str(src_path.relative_to(src_split_dir))
                rng = deterministic_rng(int(args.seed), event_type, rel_path)

                if event_type == "sudden_jump":
                    injection_info = inject_sudden_jump(
                        item,
                        rng,
                        min_mag=float(args.sudden_jump_min_mag),
                        max_mag=float(args.sudden_jump_max_mag),
                    )
                elif event_type == "weather_anomaly":
                    injection_info = inject_weather_anomaly(item, rng)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported event type: {event_type}")

                dst_path = output_path_for(src_path, src_split_dir, dst_split_dir)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                meta = item.get("meta", {}) or {}
                item["meta"] = {
                    **meta,
                    "source_split": str(args.split),
                    "injected_anomaly_type": event_type,
                    "injected_from_path": str(src_path),
                }
                records.append(
                    {
                        "event_type": event_type,
                        "source_path": str(src_path),
                        "output_path": str(dst_path),
                        "relative_path": rel_path,
                        "sourceid": str(meta.get("sourceid", "")),
                        "row_idx": meta.get("row_idx", None),
                        "class_str": str(meta.get("class_str", item.get("Y", {}).get("star_class_str", ""))),
                        "class_folder": str(meta.get("class_folder", src_path.parent.name)),
                        "injection": injection_info,
                    }
                )
                loaded.append(item)

            if not args.skip_recompute_views:
                assert periods is not None
                global_batch_idx = int(batch_idx) * int(world_size) + int(rank)
                selection_seed = (
                    int(args.seed) + 1009 * global_batch_idx + (0 if event_type == "sudden_jump" else 1_000_003)
                )
                for band in ("g", "r"):
                    views = recompute_band_views(
                        [item[band]["X"]["lc"] for item in loaded],
                        periods,
                        device=device,
                        args=args,
                        selection_seed=selection_seed + (17 if band == "g" else 37),
                    )
                    for i, item in enumerate(loaded):
                        item[band]["X"]["lc"] = views["lc"][i].clone()
                        item[band]["X"]["periodogram"] = views["periodogram"][i].clone()
                        item[band]["X"]["phase_folded_lc"] = views["phase_folded_lc"][i].clone()
                        item.setdefault(band, {}).setdefault("meta", {})
                        item[band]["meta"]["best_period"] = float(views["best_period"][i].item())
                        item[band]["meta"]["best_power"] = float(views["best_power"][i].item())
                        records[i]["injection"]["bands"][band]["recomputed_best_period"] = float(
                            views["best_period"][i].item()
                        )
                        records[i]["injection"]["bands"][band]["recomputed_best_power"] = float(
                            views["best_power"][i].item()
                        )

            for item, record in zip(loaded, records):
                dst_path = Path(record["output_path"])
                tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
                torch.save(item, tmp_path)
                os.replace(tmp_path, dst_path)
                written_paths.append(dst_path)
                info_f.write(json.dumps(record, sort_keys=True) + "\n")

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for path in written_paths:
            manifest_f.write(str(path) + "\n")

    print(f"[OK] {event_type} rank {rank}/{world_size}: wrote {len(written_paths)} samples on {device}")


def process_event_type(
    event_type: str,
    paths: Sequence[Path],
    args: argparse.Namespace,
    devices: Sequence[str],
    *,
    reset_event_root: bool = True,
) -> None:
    event_root = args.out_root / event_type
    dst_split_dir = event_root / args.split

    if reset_event_root and event_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{event_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(event_root)

    dst_split_dir.mkdir(parents=True, exist_ok=True)
    write_event_config(event_type, paths, args, devices)

    if len(devices) == 1:
        process_event_type_rank(0, event_type, paths, args, devices)
    else:
        mp.spawn(
            process_event_type_rank,
            args=(event_type, list(paths), args, list(devices)),
            nprocs=len(devices),
            join=True,
        )

    merge_rank_outputs(event_type, paths, args, world_size=len(devices))
    print(f"[OK] {event_type}: wrote {len(paths)} samples to {dst_split_dir}")
    print(f"[OK] {event_type}: wrote merged injection info to {event_root / 'injection_info.jsonl'}")


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.k_periods) <= 0:
        raise ValueError("--k-periods must be positive")
    if int(args.k_out) <= 0:
        raise ValueError("--k-out must be positive")
    if int(args.k_top) < 0:
        raise ValueError("--k-top must be non-negative")
    if float(args.sudden_jump_min_mag) < 0.0 or float(args.sudden_jump_max_mag) <= float(args.sudden_jump_min_mag):
        raise ValueError("--sudden-jump-max-mag must be greater than --sudden-jump-min-mag, and both must be non-negative")

    splits = list(args.splits) if args.splits else [args.split]
    args.active_splits = splits
    args.out_root.mkdir(parents=True, exist_ok=True)
    devices = resolve_devices(args)

    for event_type in args.event_types:
        event_root = args.out_root / event_type
        if event_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{event_root} exists. Pass --overwrite to replace it.")
            shutil.rmtree(event_root)

        for split_idx, split in enumerate(splits):
            args.split = split
            src_split_dir = args.data_root / args.split
            if not src_split_dir.exists():
                raise FileNotFoundError(f"Input split not found: {src_split_dir}")
            paths = discover_pt_files(src_split_dir)
            if int(args.limit) > 0:
                paths = paths[: int(args.limit)]
            if not paths:
                raise FileNotFoundError(f"No .pt files found under {src_split_dir}")

            print(
                f"[Info] event={event_type} split={split} samples={len(paths)} devices={devices}",
                flush=True,
            )
            process_event_type(event_type, paths, args, devices, reset_event_root=False)


if __name__ == "__main__":
    main()
