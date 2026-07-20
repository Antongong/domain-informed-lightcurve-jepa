#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from extract_starembed_embeddings import load_model_from_ckpt
from pyrregular_utils import (
    PyrregularChannelDataset,
    collate_lc_batch,
    compute_channel_tuning_stats,
    ensure_dataset_local,
    load_pyrregular_dense,
    pyrregular_channel_collate,
)
from train_ddp_numeric import build_losses, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-specific PyRregular pretraining from the existing light-curve checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=str(Path.home() / ".cache" / "pyrregular"))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=0, help="Global SFT batch size. 0 enables auto-tuning.")
    parser.add_argument("--batch_size_candidates", type=str, default="512,384,256,192,128,96,64,48,32,24,16,8,4,2,1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_gpus", type=int, default=8 if torch.cuda.is_available() else 0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--download_chunk_mb", type=int, default=32)
    parser.add_argument("--value_scaling", type=str, default="none", choices=["none", "chronos2"])
    parser.add_argument("--time_strategy", type=str, default="relative", choices=["rank", "original", "normalized", "relative"])
    parser.add_argument("--err_value", type=float, default=0.1)
    parser.add_argument("--theta_of_light_curve", type=float, default=1000.0)
    parser.add_argument("--max_points_per_series", type=int, default=1000)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--periodogram_sample_series", type=int, default=128)
    parser.add_argument("--save_preprocessed", action="store_true")
    parser.add_argument("--preprocessed_chunk_size", type=int, default=2048)
    parser.add_argument("--save_intermediate_checkpoints", action="store_true")
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="full",
        choices=["full", "head_only", "last_k"],
        help="How much of the pretrained backbone to adapt on the target dataset.",
    )
    parser.add_argument(
        "--last_k_layers",
        type=int,
        default=2,
        help="Number of transformer blocks per encoder to unfreeze when --finetune_mode=last_k.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Dict[str, Any], extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_ref = model.module if isinstance(model, DataParallel) else model
    state = {
        "cfg": cfg,
        "model": {key: value.detach().cpu() for key, value in model_ref.state_dict().items()},
    }
    state.update(extra)
    torch.save(state, path)


def maybe_wrap_dataparallel(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_gpus: int,
    batch_size: int,
    tag: str,
) -> torch.nn.Module:
    if device.type != "cuda":
        return model
    available = torch.cuda.device_count()
    requested = available if int(num_gpus) <= 0 else min(int(num_gpus), available)
    if requested <= 1:
        print(f"[Init] {tag}: using a single GPU", flush=True)
        return model
    wrapped = DataParallel(model, device_ids=list(range(requested)))
    print(
        f"[Init] {tag}: using DataParallel on {requested} GPUs; global_batch_size={batch_size}",
        flush=True,
    )
    return wrapped


class StableSFTModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        out = self.model(batch)
        proj_dict = out["projections"]
        ref_tensor = next(iter(proj_dict.values()))
        aux_losses = dict(out.get("aux_losses", {}) or {})
        aux_metrics = dict(out.get("aux_metrics", {}) or {})

        aux_losses.setdefault("forecast_total", ref_tensor.new_zeros(()))
        aux_metrics.setdefault("forecast_q50_mae", ref_tensor.new_zeros(()))
        aux_metrics.setdefault("forecast_eligible_frac", ref_tensor.new_zeros(()))

        return {
            "projections": proj_dict,
            "aux_losses": aux_losses,
            "aux_metrics": aux_metrics,
        }


def parse_batch_size_candidates(spec: str) -> List[int]:
    vals = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(max(1, int(item)))
    return sorted(set(vals), reverse=True)


def save_preprocessed_dataset(
    dataset: PyrregularChannelDataset,
    out_dir: Path,
    *,
    chunk_size: int,
) -> None:
    pre_dir = out_dir / "preprocessed_train"
    pre_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = int(math.ceil(len(dataset) / max(1, int(chunk_size))))
    print(
        f"[Preprocess] saving {len(dataset)} series to {pre_dir} in {total_chunks} chunk(s)",
        flush=True,
    )

    metadata: Dict[str, Any] = {
        "num_series": int(len(dataset)),
        "chunk_size": int(chunk_size),
        "chunks": [],
    }

    for start in range(0, len(dataset), int(chunk_size)):
        stop = min(len(dataset), start + int(chunk_size))
        series = [torch.from_numpy(dataset[idx]) for idx in range(start, stop)]
        sample_index = [int(dataset.index[idx][0]) for idx in range(start, stop)]
        channel_index = [int(dataset.index[idx][1]) for idx in range(start, stop)]
        shard_path = pre_dir / f"train_shard_{start:08d}_{stop:08d}.pt"
        torch.save(
            {
                "series": series,
                "sample_index": sample_index,
                "channel_index": channel_index,
            },
            shard_path,
        )
        metadata["chunks"].append(
            {
                "path": str(shard_path),
                "start": int(start),
                "stop": int(stop),
                "num_series": int(stop - start),
            }
        )
        chunk_id = 1 + start // max(1, int(chunk_size))
        if chunk_id == 1 or chunk_id == total_chunks or chunk_id % 8 == 0:
            print(f"[Preprocess] wrote chunk {chunk_id}/{total_chunks} ({start}:{stop})", flush=True)

    (pre_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Preprocess] manifest: {pre_dir / 'manifest.json'}", flush=True)


def try_training_step(
    *,
    cfg: Dict[str, Any],
    init_ckpt: str,
    strict: bool,
    batch: torch.Tensor,
    device: torch.device,
    num_gpus: int,
) -> bool:
    base_model = None
    model = None
    loss_fn = None
    optimizer = None
    scaler = None
    try:
        base_model = build_model(cfg).to(device)
        load_model_from_ckpt(base_model, init_ckpt, strict=strict)
        model = StableSFTModelWrapper(base_model).to(device)
        model = maybe_wrap_dataparallel(
            model,
            device=device,
            num_gpus=num_gpus,
            batch_size=int(batch.shape[0]),
            tag="sft-batch-probe",
        )
        loss_fn = build_losses(cfg).to(device)
        optimizer = torch.optim.AdamW([p for p in base_model.parameters() if p.requires_grad], lr=1.0e-6)
        amp_enabled = bool(cfg.get("training", {}).get("amp", {}).get("enabled", False))
        use_amp = bool(amp_enabled and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)
        views_order = list(cfg["loss"]["views_order"])
        raw_lc = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=use_amp)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            out = model({"X": {"raw_lc": raw_lc}})
            proj_dict = {key: value.float() for key, value in out["projections"].items() if key in views_order}
            proj_seq = torch.stack([proj_dict[view] for view in views_order if view in proj_dict], dim=0)
            losses = loss_fn(proj_seq, proj_dict)
            aux_losses = {
                key: value.float().mean()
                for key, value in (out.get("aux_losses", {}) or {}).items()
            }
            total_loss = losses["total"]
            if aux_losses:
                total_loss = total_loss + torch.stack([value.float() for value in aux_losses.values()]).sum()

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return True
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return False
    finally:
        for obj in (model, base_model, loss_fn, optimizer, scaler):
            del obj
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def auto_tune_batch_size(
    dataset: PyrregularChannelDataset,
    *,
    cfg: Dict[str, Any],
    init_ckpt: str,
    strict: bool,
    device: torch.device,
    num_gpus: int,
    candidates: List[int],
) -> int:
    if not candidates:
        raise RuntimeError("No batch size candidates provided.")

    if getattr(dataset, "lengths", None):
        probe_order = np.argsort(np.asarray(dataset.lengths))[::-1].tolist()
    else:
        probe_order = list(range(len(dataset)))

    for candidate in candidates:
        actual = min(candidate, len(dataset))
        if actual <= 0:
            continue
        chosen = probe_order[:actual]
        samples = [dataset[idx] for idx in chosen]
        batch = collate_lc_batch(samples)
        ok = try_training_step(
            cfg=cfg,
            init_ckpt=init_ckpt,
            strict=strict,
            batch=batch,
            device=device,
            num_gpus=num_gpus,
        )
        if ok:
            print(f"[BatchSize] selected global_batch_size={actual}", flush=True)
            return int(actual)
        print(f"[BatchSize] OOM at global_batch_size={actual}; trying smaller", flush=True)

    raise RuntimeError("Failed to find a fitting batch size from the provided candidates.")


def apply_dataset_tuning(cfg: Dict[str, Any], stats: Dict[str, Any], out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    tuned = copy.deepcopy(cfg)
    tuned.setdefault("training", {})
    tuned["training"]["output_dir"] = str(out_dir)
    tuned["training"]["seed"] = int(args.seed)
    tuned.setdefault("probe", {})
    tuned["probe"]["enabled"] = False

    numc = tuned.setdefault("model", {}).setdefault("numeric", {})
    rawc = numc.setdefault("raw", {})
    perc = numc.setdefault("periodogram", {})
    pfc = numc.setdefault("phase_folded", {})
    infc = numc.setdefault("inference", {})

    rawc["vmin"] = float(stats["raw_vmin"])
    rawc["vmax"] = float(stats["raw_vmax"])
    rawc["rope_max_period"] = float(stats["raw_rope_max_period"])
    perc["rope_max_period"] = float(stats["per_rope_max_period"])
    pfc["rope_max_period"] = float(stats["pf_rope_max_period"])
    infc["min_period"] = float(stats["inference_min_period"])
    infc["max_period"] = float(stats["inference_max_period"])
    infc["k_periods"] = int(stats["inference_k_periods"])
    infc["k_top"] = int(stats["inference_k_top"])
    infc["k_rand"] = int(stats["inference_k_rand"])

    tuned["pyrregular_adaptation"] = {
        "dataset": str(args.dataset),
        "value_scaling": str(args.value_scaling),
        "time_strategy": str(args.time_strategy),
        "err_value": float(args.err_value),
        "theta_of_light_curve": float(args.theta_of_light_curve),
        "max_points_per_series": int(args.max_points_per_series),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_gpus": int(args.num_gpus),
        "save_preprocessed": bool(args.save_preprocessed),
        "save_intermediate_checkpoints": bool(args.save_intermediate_checkpoints),
        "finetune_mode": str(args.finetune_mode),
        "last_k_layers": int(args.last_k_layers),
        "tuning_stats": stats,
    }
    return tuned


@torch.inference_mode()
def estimate_periodogram_bounds(
    cfg: Dict[str, Any],
    dataset: PyrregularChannelDataset,
    *,
    device: torch.device,
    sample_series: int,
    seed: int,
    batch_size: int = 8,
) -> Dict[str, float]:
    if sample_series <= 0 or len(dataset) <= 0:
        return {"per_vmin": -6.0, "per_vmax": 0.0}

    rng = np.random.RandomState(int(seed))
    take = min(int(sample_series), len(dataset))
    chosen = rng.choice(len(dataset), size=take, replace=False)

    model = build_model(cfg).to(device)
    model.eval()
    values: List[np.ndarray] = []
    active_batch_size = max(1, int(batch_size))
    start = 0
    warned_fallback = False

    while start < take:
        chunk_stop = min(take, start + active_batch_size)
        chunk_idx = chosen[start:chunk_stop]
        try:
            samples = [dataset[int(i)] for i in chunk_idx]
            raw_batch = collate_lc_batch(samples).to(device, non_blocking=True)
            derived = model.preprocess_raw_light_curve(raw_batch)
            log_power = derived["periodogram"][..., 1].detach().cpu().numpy().astype(np.float32, copy=False)
            finite = log_power[np.isfinite(log_power)]
            if finite.size:
                values.append(finite)
            del raw_batch, derived, log_power, finite
            start = chunk_stop
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if active_batch_size <= 1:
                print(
                    "[Warn] periodogram bounds estimation hit OOM even at batch_size=1; using default range",
                    flush=True,
                )
                warned_fallback = True
                break
            next_batch_size = max(1, active_batch_size // 2)
            print(
                f"[Warn] periodogram bounds OOM at batch_size={active_batch_size}; retrying with batch_size={next_batch_size}",
                flush=True,
            )
            active_batch_size = next_batch_size

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if warned_fallback or not values:
        return {"per_vmin": -6.0, "per_vmax": 0.0}

    merged = np.concatenate(values, axis=0)
    q_low, q_high = np.quantile(merged, [0.01, 0.99]).tolist()
    if not np.isfinite(q_low):
        q_low = -6.0
    if not np.isfinite(q_high):
        q_high = 0.0
    q_high = min(0.0, float(q_high))
    if q_high <= q_low:
        return {"per_vmin": -6.0, "per_vmax": 0.0}
    margin = max(1.0e-3, 0.01 * (q_high - q_low))
    return {
        "per_vmin": float(q_low - margin),
        "per_vmax": float(min(0.0, q_high + margin)),
    }


def mean_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in items if key in item and math.isfinite(float(item[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def set_module_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def summarize_trainable_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_params += count
        if bool(param.requires_grad):
            trainable_params += count
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_fraction": float(trainable_params / max(1, total_params)),
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def configure_finetune_mode(
    model: torch.nn.Module,
    *,
    mode: str,
    last_k_layers: int,
) -> Dict[str, Any]:
    mode = str(mode).lower().strip()
    last_k_layers = max(1, int(last_k_layers))

    for param in model.parameters():
        param.requires_grad = False

    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif mode in {"head_only", "last_k"}:
        for encoder_name in ("raw_encoder", "periodogram_encoder", "phase_encoder"):
            encoder = getattr(model, encoder_name, None)
            if encoder is None:
                continue
            set_module_requires_grad(getattr(encoder, "norm", None), True)
            set_module_requires_grad(getattr(encoder, "pool", None), True)
            if mode == "last_k":
                layers = list(getattr(encoder, "layers", []) or [])
                start = max(0, len(layers) - last_k_layers)
                for layer in layers[start:]:
                    set_module_requires_grad(layer, True)

        set_module_requires_grad(getattr(model, "projectors", None), True)
        set_module_requires_grad(getattr(model, "group_fusion", None), True)
        set_module_requires_grad(getattr(model, "covariate_embedders", None), True)
        set_module_requires_grad(getattr(model, "forecast_head", None), True)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported finetune_mode={mode!r}")

    summary = summarize_trainable_parameters(model)
    summary["mode"] = mode
    summary["last_k_layers"] = int(last_k_layers)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    setup_t0 = time.time()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        raise RuntimeError(f"Empty config: {args.config}")

    set_seed(int(args.seed))

    print(f"[Setup] resolving dataset {args.dataset}", flush=True)
    dataset_path = ensure_dataset_local(
        args.dataset,
        Path(args.cache_dir),
        download_workers=int(args.download_workers),
        download_chunk_mb=int(args.download_chunk_mb),
    )
    print(f"[Setup] dataset ready at {dataset_path}", flush=True)

    print(f"[Setup] loading dense arrays for {args.dataset}", flush=True)
    da, x, t, y_raw, split = load_pyrregular_dense(dataset_path)
    print(
        f"[Setup] dense arrays loaded: samples={x.shape[0]} channels={x.shape[1]} seq_len={x.shape[2]} elapsed_s={time.time() - setup_t0:.1f}",
        flush=True,
    )

    min_valid_points = int(cfg.get("model", {}).get("numeric", {}).get("inference", {}).get("min_valid_points", 8))
    print(f"[Setup] computing tuning stats for {args.dataset}", flush=True)
    tuning_stats = compute_channel_tuning_stats(
        x,
        t,
        split,
        value_scaling=args.value_scaling,
        time_strategy=args.time_strategy,
        err_value=float(args.err_value),
        theta_of_light_curve=float(args.theta_of_light_curve),
        min_valid_points=min_valid_points,
        seed=int(args.seed),
        progress_label=f"Tuning:{Path(args.dataset).stem}",
    )
    print(
        f"[Setup] tuning stats ready: train_series={tuning_stats['train_series']} max_seq_len={tuning_stats['max_seq_len']} elapsed_s={time.time() - setup_t0:.1f}",
        flush=True,
    )

    tuned_cfg = apply_dataset_tuning(cfg, tuning_stats, out_dir, args)
    print(f"[Setup] building channel dataset for {args.dataset}", flush=True)
    train_dataset = PyrregularChannelDataset(
        x,
        t,
        split,
        train_only=True,
        value_scaling=args.value_scaling,
        time_strategy=args.time_strategy,
        err_value=float(args.err_value),
        min_valid_points=min_valid_points,
        max_points_per_series=int(args.max_points_per_series),
        progress_label=f"Dataset:{Path(args.dataset).stem}",
    )
    print(
        f"[Setup] channel dataset ready: series={len(train_dataset)} elapsed_s={time.time() - setup_t0:.1f}",
        flush=True,
    )
    print(f"[Setup] estimating periodogram bounds for {args.dataset}", flush=True)
    periodogram_bounds = estimate_periodogram_bounds(
        tuned_cfg,
        train_dataset,
        device=device,
        sample_series=int(args.periodogram_sample_series),
        seed=int(args.seed),
    )
    print(
        f"[Setup] periodogram bounds ready: vmin={periodogram_bounds['per_vmin']:.4f} vmax={periodogram_bounds['per_vmax']:.4f} elapsed_s={time.time() - setup_t0:.1f}",
        flush=True,
    )
    tuned_cfg["model"]["numeric"].setdefault("periodogram", {})
    tuned_cfg["model"]["numeric"]["periodogram"]["vmin"] = float(periodogram_bounds["per_vmin"])
    tuned_cfg["model"]["numeric"]["periodogram"]["vmax"] = float(periodogram_bounds["per_vmax"])
    tuned_cfg["pyrregular_adaptation"]["periodogram_value_range"] = periodogram_bounds
    tuned_cfg["pyrregular_adaptation"]["dataset_metadata"] = {
        "dataset": str(args.dataset),
        "title": str(da.attrs.get("title", str(args.dataset))),
        "source": str(da.attrs.get("source", "")),
        "shape": {
            "n_samples": int(x.shape[0]),
            "n_channels": int(x.shape[1]),
            "seq_len": int(x.shape[2]),
        },
        "train_split_samples": int((np.asarray(split).astype(str) != "test").sum()),
        "test_split_samples": int((np.asarray(split).astype(str) == "test").sum()),
        "n_labels": int(len(np.unique(y_raw))),
        "train_series": int(len(train_dataset)),
    }

    if bool(args.save_preprocessed):
        print(f"[Setup] saving preprocessed training shards for {args.dataset}", flush=True)
        save_preprocessed_dataset(
            train_dataset,
            out_dir,
            chunk_size=int(args.preprocessed_chunk_size),
        )
        print(f"[Setup] preprocessed shards ready for {args.dataset}", flush=True)

    effective_batch_size = int(args.batch_size)
    if effective_batch_size <= 0:
        print(f"[Setup] auto-tuning train batch size for {args.dataset}", flush=True)
        effective_batch_size = auto_tune_batch_size(
            train_dataset,
            cfg=tuned_cfg,
            init_ckpt=args.init_ckpt,
            strict=bool(args.strict),
            device=device,
            num_gpus=int(args.num_gpus),
            candidates=parse_batch_size_candidates(args.batch_size_candidates),
        )
    print(
        f"[Setup] starting training for {args.dataset} with global_batch_size={effective_batch_size} elapsed_s={time.time() - setup_t0:.1f}",
        flush=True,
    )
    tuned_cfg["pyrregular_adaptation"]["effective_batch_size"] = int(effective_batch_size)

    num_steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, int(effective_batch_size))))
    tuned_cfg["training"]["max_steps"] = int(num_steps_per_epoch * int(args.epochs))
    tuned_cfg["training"]["log_every"] = max(1, min(num_steps_per_epoch, int(cfg.get("training", {}).get("log_every", 10))))

    save_yaml(out_dir / "config_used.yaml", tuned_cfg)
    (out_dir / "tuning_stats.json").write_text(json.dumps(tuned_cfg["pyrregular_adaptation"], indent=2), encoding="utf-8")

    loader = DataLoader(
        train_dataset,
        batch_size=int(effective_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=pyrregular_channel_collate,
        drop_last=False,
    )

    base_model = build_model(tuned_cfg).to(device)
    load_model_from_ckpt(base_model, args.init_ckpt, strict=bool(args.strict))
    finetune_summary = configure_finetune_mode(
        base_model,
        mode=str(args.finetune_mode),
        last_k_layers=int(args.last_k_layers),
    )
    print(
        json.dumps(
            {
                "event": "finetune_mode",
                "dataset": str(args.dataset),
                "mode": finetune_summary["mode"],
                "last_k_layers": int(finetune_summary["last_k_layers"]),
                "trainable_params": int(finetune_summary["trainable_params"]),
                "total_params": int(finetune_summary["total_params"]),
                "trainable_fraction": float(finetune_summary["trainable_fraction"]),
            },
            indent=2,
        ),
        flush=True,
    )
    tuned_cfg["pyrregular_adaptation"]["finetune_summary"] = {
        key: value
        for key, value in finetune_summary.items()
        if key not in {"trainable_names", "frozen_names"}
    }
    model = StableSFTModelWrapper(base_model).to(device)
    model = maybe_wrap_dataparallel(
        model,
        device=device,
        num_gpus=int(args.num_gpus),
        batch_size=int(effective_batch_size),
        tag="sft",
    )
    loss_fn = build_losses(tuned_cfg).to(device)

    optim_cfg = tuned_cfg.get("training", {}).get("optimizer", {})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(optim_cfg.get("lr", 2.0e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1.0e-8)),
    )

    amp_enabled = bool(tuned_cfg.get("training", {}).get("amp", {}).get("enabled", False))
    use_amp = bool(amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)
    grad_clip = float(tuned_cfg.get("training", {}).get("grad_clip_norm", 0.0))
    views_order = list(tuned_cfg["loss"]["views_order"])
    save_yaml(out_dir / "config_used.yaml", tuned_cfg)
    (out_dir / "tuning_stats.json").write_text(json.dumps(tuned_cfg["pyrregular_adaptation"], indent=2), encoding="utf-8")
    (out_dir / "trainable_parameters.json").write_text(json.dumps(finetune_summary, indent=2), encoding="utf-8")

    log_path = out_dir / "train_metrics.jsonl"
    t0 = time.time()
    step = 0
    total_batches = int(len(loader))
    log_every = max(1, int(tuned_cfg.get("training", {}).get("log_every", 10)))

    print(
        json.dumps(
            {
                "event": "train_start",
                "dataset": str(args.dataset),
                "epochs": int(args.epochs),
                "effective_batch_size": int(effective_batch_size),
                "train_series": int(len(train_dataset)),
                "batches_per_epoch": int(total_batches),
                "log_every": int(log_every),
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(int(args.epochs)):
        model.train()
        epoch_metrics: List[Dict[str, float]] = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            raw_lc = batch["X"]["raw_lc"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                out = model({"X": {"raw_lc": raw_lc}})
                proj_dict = {key: value.float() for key, value in out["projections"].items() if key in views_order}
                proj_list = [proj_dict[view] for view in views_order if view in proj_dict]
                if len(proj_list) < 2:
                    raise RuntimeError("Need at least 2 active projection views for pretraining.")
                proj_seq = torch.stack(proj_list, dim=0)
                losses = loss_fn(proj_seq, proj_dict)
                aux_losses = {
                    key: value.float().mean()
                    for key, value in (out.get("aux_losses", {}) or {}).items()
                }
                aux_metrics = {
                    key: value.float().mean()
                    for key, value in (out.get("aux_metrics", {}) or {}).items()
                }
                total_loss = losses["total"]
                if aux_losses:
                    total_loss = total_loss + torch.stack([value.float() for value in aux_losses.values()]).sum()

            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            record = {f"loss/{key}": float(value.detach().cpu().item()) for key, value in losses.items()}
            for key, value in aux_losses.items():
                record[f"loss/{key}"] = float(value.detach().cpu().item())
            for key, value in aux_metrics.items():
                record[f"metric/{key}"] = float(value.detach().cpu().item())
            record["loss/total_with_aux"] = float(total_loss.detach().cpu().item())
            epoch_metrics.append(record)
            step += 1

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                progress_record = {
                    "event": "train_progress",
                    "epoch": int(epoch),
                    "batch": int(batch_idx),
                    "batches_per_epoch": int(total_batches),
                    "step": int(step),
                    "time_s": float(time.time() - t0),
                    "epoch_time_s": float(time.time() - epoch_start),
                    "loss/total_with_aux": float(record["loss/total_with_aux"]),
                }
                append_jsonl(log_path, progress_record)
                print(json.dumps(progress_record, indent=2), flush=True)

            del raw_lc, out, proj_dict, proj_list, proj_seq, losses, aux_losses, aux_metrics, total_loss

        epoch_mean = mean_dict(epoch_metrics)
        epoch_record = {
            "epoch": int(epoch),
            "step": int(step),
            "time_s": float(time.time() - t0),
        }
        epoch_record.update(epoch_mean)
        append_jsonl(log_path, epoch_record)
        print(json.dumps(epoch_record, indent=2), flush=True)

        if bool(args.save_intermediate_checkpoints):
            ckpt_epoch = out_dir / f"ckpt_epoch_{epoch + 1:02d}.pt"
            save_checkpoint(
                ckpt_epoch,
                base_model,
                tuned_cfg,
                {
                    "epoch": int(epoch + 1),
                    "step": int(step),
                    "dataset": str(args.dataset),
                    "tuning_stats": tuned_cfg["pyrregular_adaptation"],
                },
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(
        final_path,
        base_model,
        tuned_cfg,
        {
            "epoch": int(args.epochs),
            "step": int(step),
            "dataset": str(args.dataset),
            "tuning_stats": tuned_cfg["pyrregular_adaptation"],
        },
    )
    print(f"[Done] final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
