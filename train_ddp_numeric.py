from __future__ import annotations
import argparse
import contextlib
import copy
import gc
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Better allocator behavior on large long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from load_aligned_LEAVES import list_precomputed_paths, build_precomputed_loader
from losses import SIGReg, LeJEPALoss, MultiViewCLIPLoss, CombinedPretrainLoss
from model_numeric_only import MultiModalAstroModel
from probes import ProbeConfig, ProbeManager


# =========================================================
# Distributed helpers
# =========================================================
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_rank0() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[torch.device, int, int, int]:
    """
    Returns: (device, rank, local_rank, world_size)
    Works with torchrun and also single-process runs.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device("cuda", local_rank)
        return device, rank, local_rank, world_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 0, 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / float(get_world_size())
    return y


@torch.no_grad()
def reduce_dict_mean(d: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_distributed():
        return d
    keys = sorted(d.keys())
    if len(keys) == 0:
        return {}
    vals = torch.tensor([d[k] for k in keys], device=device, dtype=torch.float32)
    vals = reduce_mean(vals)
    return {k: float(v) for k, v in zip(keys, vals.detach().cpu().tolist())}


# =========================================================
# Numerical stability / NaN guards (DDP-safe)
# =========================================================
def _tensor_all_finite(x: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(x).all().item())
    except Exception:
        return False


def _sanitize_tensor(
    x: torch.Tensor,
    *,
    clamp_abs: Optional[float] = None,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if clamp_abs is not None:
        x = torch.clamp(x, min=-float(clamp_abs), max=float(clamp_abs))
    return x


def _sanitize_view_dict(view_dict: Dict[str, torch.Tensor], *, clamp_abs: Optional[float] = 1e4) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in view_dict.items():
        if torch.is_tensor(v):
            out[k] = _sanitize_tensor(v, clamp_abs=clamp_abs)
    return out


def _safe_sum_tensors(tensors: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Build a connected scalar from tensors but guaranteed finite.
    Used to build a 'dummy' loss whose backward triggers DDP hooks safely.
    """
    if len(tensors) == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    s = torch.zeros((), device=device, dtype=torch.float32)
    for t in tensors:
        s = s + _sanitize_tensor(t.float()).sum()
    return s


def _safe_float(t: torch.Tensor, default: float = 0.0) -> float:
    try:
        v = float(t.detach().float().cpu().item())
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        out[k] = fv if math.isfinite(fv) else 0.0
    return out


def _reset_module_parameters(m: torch.nn.Module) -> None:
    for mod in m.modules():
        if hasattr(mod, "reset_parameters") and callable(getattr(mod, "reset_parameters")):
            try:
                mod.reset_parameters()
            except Exception:
                pass


def _broadcast_module_from_rank0(m: torch.nn.Module) -> None:
    if not is_distributed():
        return
    for p in m.parameters():
        dist.broadcast(p.data, src=0)
    for b in m.buffers():
        dist.broadcast(b.data, src=0)


def _reset_probes_on_nan(probes: torch.nn.Module, probe_optimizer: Optional[torch.optim.Optimizer]) -> None:
    """
    Reset probe weights (rank0), broadcast, and clear optimizer state.
    This is optional and controlled by config.nan_guard.reset_probes_on_nonfinite.
    """
    pm = probes.module if isinstance(probes, DDP) else probes

    if is_rank0():
        _reset_module_parameters(pm)

    if is_distributed():
        barrier()
        _broadcast_module_from_rank0(pm)
        barrier()

    if probe_optimizer is not None:
        probe_optimizer.state.clear()
        probe_optimizer.zero_grad(set_to_none=True)


# =========================================================
# Differentiable all-gather (global-batch loss)
# =========================================================
class _AllGatherConcat(torch.autograd.Function):
    """
    Differentiable all-gather that concatenates along dim=0 (batch dim).
    Supports variable local batch sizes by padding to max then trimming.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not is_distributed():
            ctx.sizes = [x.shape[0]]
            ctx.rank = 0
            return x

        world = get_world_size()
        rank = get_rank()

        local_n = int(x.shape[0])
        n_t = torch.tensor([local_n], device=x.device, dtype=torch.long)
        n_list = [torch.zeros_like(n_t) for _ in range(world)]
        dist.all_gather(n_list, n_t)
        sizes = [int(t.item()) for t in n_list]
        max_n = max(sizes) if sizes else local_n

        if local_n < max_n:
            pad_shape = (max_n - local_n,) + tuple(x.shape[1:])
            pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=0)
        else:
            x_pad = x

        gathered = [torch.empty_like(x_pad) for _ in range(world)]
        dist.all_gather(gathered, x_pad)

        out = torch.cat([g[: sizes[i]] for i, g in enumerate(gathered)], dim=0)

        ctx.sizes = sizes
        ctx.rank = rank
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        if not is_distributed():
            return (grad_out,)

        sizes: List[int] = ctx.sizes
        rank: int = ctx.rank

        offset = int(sum(sizes[:rank]))
        local_n = int(sizes[rank])
        grad_local = grad_out[offset : offset + local_n].contiguous()
        return (grad_local,)


def all_gather_concat(x: torch.Tensor) -> torch.Tensor:
    return _AllGatherConcat.apply(x)


def gather_view_dict_with_grad(view_dict: Dict[str, torch.Tensor], views_order: List[str]) -> Dict[str, torch.Tensor]:
    if not is_distributed():
        return view_dict
    out: Dict[str, torch.Tensor] = {}
    for v in views_order:
        if v in view_dict:
            out[v] = all_gather_concat(view_dict[v])
    return out


# =========================================================
# IO utils
# =========================================================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise RuntimeError(f"Empty config: {path}")
    return cfg


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================================================
# Builders
# =========================================================
def build_model(cfg: Dict[str, Any]) -> MultiModalAstroModel:
    mc = cfg["model"]
    modalc = cfg.get("modalities", {}) or {}
    numc = mc.get("numeric", {}) or {}
    rawc = numc.get("raw", {}) or {}
    perc = numc.get("periodogram", {}) or {}
    pfc = numc.get("phase_folded", {}) or {}
    infc = numc.get("inference", {}) or {}
    poolc = numc.get("pooling", {}) or {}
    groupc = numc.get("group_fusion", {}) or {}
    forecastc = numc.get("forecast_aux", {}) or {}

    numeric_model_size = str(numc.get("model_size", "gpt2"))

    ablc = mc.get("ablations", {}) or {}
    pos_global = ablc.get("positional_embedding", None)

    raw_position_mode = pos_global or rawc.get("position_mode", rawc.get("positional_embedding", "time"))
    period_position_mode = pos_global or perc.get("position_mode", perc.get("positional_embedding", "time"))
    phase_position_mode = pos_global or pfc.get("position_mode", pfc.get("positional_embedding", "time"))

    unc_global = ablc.get("use_uncertainty", ablc.get("embedding_uncertainty", None))
    if unc_global is None:
        raw_use_uncertainty = bool(rawc.get("use_uncertainty", True))
        pf_use_uncertainty = bool(pfc.get("use_uncertainty", True))
    else:
        raw_use_uncertainty = bool(unc_global)
        pf_use_uncertainty = bool(unc_global)

    model = MultiModalAstroModel(
        common_dim=mc["common_dim"],
        projection_dim=mc["projection_dim"],
        projection_hidden_dim=mc.get("projection_hidden_dim", 0),
        projection_dropout=mc.get("projection_dropout", 0.0),
        use_projection=mc.get("use_projection", True),
        numeric_model_size=numeric_model_size,

        raw_embed_dim=rawc.get("embed_dim", None),
        raw_depth=rawc.get("depth", None),
        raw_heads=rawc.get("heads", None),
        raw_mlp_ratio=rawc.get("mlp_ratio", 4.0),
        raw_dropout=rawc.get("dropout", None),
        raw_rope_max_period=rawc.get("rope_max_period", 2000.0),

        per_embed_dim=perc.get("embed_dim", None),
        per_depth=perc.get("depth", None),
        per_heads=perc.get("heads", None),
        per_mlp_ratio=perc.get("mlp_ratio", 4.0),
        per_dropout=perc.get("dropout", None),
        per_rope_max_period=perc.get("rope_max_period", 2000.0),

        pf_embed_dim=pfc.get("embed_dim", None),
        pf_depth=pfc.get("depth", None),
        pf_heads=pfc.get("heads", None),
        pf_mlp_ratio=pfc.get("mlp_ratio", 4.0),
        pf_dropout=pfc.get("dropout", None),
        pf_rope_max_period=pfc.get("rope_max_period", 10.0),

        raw_position_mode=raw_position_mode,
        period_position_mode=period_position_mode,
        phase_position_mode=phase_position_mode,
        phase_use_normalized_phase=bool(pfc.get("use_normalized_phase", True)),

        raw_use_uncertainty=raw_use_uncertainty,
        pf_use_uncertainty=pf_use_uncertainty,

        raw_num_bins=int(rawc.get("num_bins", 2048)),
        raw_vmin=float(rawc.get("vmin", -5.0)),
        raw_vmax=float(rawc.get("vmax", 5.0)),
        per_num_bins=int(perc.get("num_bins", 2048)),
        per_vmin=float(perc.get("vmin", -10.0)),
        per_vmax=float(perc.get("vmax", 0.0)),

        pooling_mode=str(poolc.get("mode", "attn_stats")),
        pooling_dropout=float(poolc.get("dropout", 0.0)),

        group_enabled=bool(groupc.get("enabled", True)),
        group_num_layers=int(groupc.get("num_layers", 2)),
        group_num_heads=int(groupc.get("num_heads", 8)),
        group_dropout=float(groupc.get("dropout", 0.1)),
        group_use_best_period=bool(groupc.get("use_best_period", True)),
        group_use_best_power=bool(groupc.get("use_best_power", True)),
        group_use_time_span=bool(groupc.get("use_time_span", True)),
        group_use_mag_std=bool(groupc.get("use_mag_std", True)),
        group_use_valid_fraction=bool(groupc.get("use_valid_fraction", True)),

        forecast_enabled=bool(forecastc.get("enabled", False)),
        forecast_horizon=int(forecastc.get("horizon", 16)),
        forecast_min_context=int(forecastc.get("min_context", 32)),
        forecast_quantiles=list(forecastc.get("quantiles", [0.1, 0.5, 0.9])),
        forecast_weight=float(forecastc.get("weight", 0.2)),
        forecast_hidden_dim=int(forecastc.get("hidden_dim", 0)),
        forecast_dropout=float(forecastc.get("dropout", 0.1)),
        forecast_max_samples_per_batch=int(forecastc.get("max_samples_per_batch", 8)),

        inference_min_period=float(infc.get("min_period", 0.0069444444)),
        inference_max_period=float(infc.get("max_period", 2000.0)),
        inference_k_periods=int(infc.get("k_periods", 1_000_000)),
        inference_chunk_size=int(infc.get("chunk_size", 8192)),
        inference_k_top=int(infc.get("k_top", 500)),
        inference_k_rand=int(infc.get("k_rand", 500)),
        inference_min_valid_points=int(infc.get("min_valid_points", 8)),
        inference_eps=float(infc.get("eps", 1e-12)),
    )
    model.enable[model.VIEW_RAW] = bool(modalc.get("raw", True))
    model.enable[model.VIEW_PERIODOGRAM] = bool(modalc.get("periodogram", True))
    model.enable[model.VIEW_PHASE_FOLDED] = bool(modalc.get("phase_folded", True))
    if "group" in modalc:
        model.enable[model.VIEW_GROUP] = bool(modalc.get("group", True)) and bool(groupc.get("enabled", True))
    else:
        any_numeric_view = (
            bool(model.enable[model.VIEW_RAW])
            or bool(model.enable[model.VIEW_PERIODOGRAM])
            or bool(model.enable[model.VIEW_PHASE_FOLDED])
        )
        model.enable[model.VIEW_GROUP] = bool(groupc.get("enabled", True)) and any_numeric_view
    return model


def build_losses(cfg: Dict[str, Any]) -> CombinedPretrainLoss:
    lc = cfg["loss"]
    sig_cfg = lc["sigreg"]
    sigreg = SIGReg(
        knots=sig_cfg["knots"],
        max_t=sig_cfg["max_t"],
        proj_dim=sig_cfg["proj_dim"],
        refresh_every=sig_cfg.get("refresh_every", 0),
    )
    lejepa = LeJEPALoss(
        lamb=lc["lejepa"]["lamb"],
        sigreg=sigreg,
        align_to=lc["lejepa"].get("align_to", None),
    )

    clip_pairs = [tuple(p) for p in lc["clip"]["pairs"]]
    clip = MultiViewCLIPLoss(
        pairs=clip_pairs,
        temperature=lc["clip"]["temperature"],
        symmetric=lc["clip"].get("symmetric", True),
    )

    return CombinedPretrainLoss(
        use_lejepa=lc["lejepa"]["enabled"],
        lejepa_weight=lc["lejepa"]["weight"],
        lejepa=lejepa,
        use_clip=lc["clip"]["enabled"],
        clip_weight=lc["clip"]["weight"],
        clip=clip,
    )


def split_paths(paths: List[str], test_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    n = len(paths)
    n_test = int(round(n * float(test_ratio)))
    n_test = max(1, n_test)
    n_train = max(1, n - n_test)

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n).tolist()
    test_idx = set(perm[:n_test])
    train_paths = [p for i, p in enumerate(paths) if i not in test_idx]
    test_paths = [p for i, p in enumerate(paths) if i in test_idx]
    train_paths = train_paths[:n_train]
    return train_paths, test_paths


def build_loaders(cfg: Dict[str, Any], *, distributed: bool) -> Tuple[DataLoader, DataLoader]:
    dc = cfg["data"]
    dlc = cfg["dataloader"]

    out_dir = dc["precomputed"]["out_dir"]
    manifest = dc["precomputed"].get("manifest", None)
    prefer_manifest_all = bool(dc["precomputed"].get("prefer_manifest_all", True))
    allow_discovery_fallback = bool(dc["precomputed"].get("allow_discovery_fallback", True))

    all_paths = list_precomputed_paths(
        out_dir=out_dir,
        manifest=manifest,
        prefer_manifest_all=prefer_manifest_all,
        allow_discovery_fallback=allow_discovery_fallback,
    )

    sc = cfg.get("split", {}) or {}
    test_ratio = float(sc.get("test_ratio", 0.1))
    seed = int(sc.get("seed", cfg["training"]["seed"]))
    train_paths, test_paths = split_paths(all_paths, test_ratio=test_ratio, seed=seed)

    batch_size = int(dlc["batch_size"])
    batch_size_eval = int(dlc.get("batch_size_eval", batch_size))

    num_workers = int(dlc.get("num_workers", 4))
    pin_memory = bool(dlc.get("pin_memory", True))
    drop_last = bool(dlc.get("drop_last", True))
    persistent_workers = bool(dlc.get("persistent_workers", True)) if num_workers > 0 else False

    train_loader = build_precomputed_loader(
        out_dir=out_dir,
        batch_size=batch_size,
        manifest=None,
        paths=train_paths,
        num_workers=num_workers,
        shuffle=bool(dlc.get("shuffle", True)),
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        distributed=distributed,
        seed=int(cfg["training"]["seed"]),
    )

    test_loader = build_precomputed_loader(
        out_dir=out_dir,
        batch_size=batch_size_eval,
        manifest=None,
        paths=test_paths,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        distributed=distributed,
        seed=int(cfg["training"]["seed"]),
    )
    return train_loader, test_loader


# =========================================================
# Representation aggregation (mean / concat)
# =========================================================
def aggregate_views(
    view_dict: Dict[str, torch.Tensor],
    views_order: List[str],
    mode: str = "mean",
    *,
    require_all: bool = True,
) -> torch.Tensor:
    mode = str(mode).lower().strip()

    if require_all:
        missing = [v for v in views_order if v not in view_dict]
        if missing:
            raise RuntimeError(f"Missing views in view_dict required by views_order: {missing}")

    feats = [view_dict[v] for v in views_order if v in view_dict]
    if len(feats) == 0:
        raise RuntimeError("No features available for aggregation; check views_order / enabled views.")

    if mode == "mean":
        ref_shape = tuple(feats[0].shape)
        for i, f in enumerate(feats[1:], start=1):
            if tuple(f.shape) != ref_shape:
                raise ValueError(
                    "aggregate_views(mode='mean') requires identical shapes across views. "
                    f"View idx={i} has shape={tuple(f.shape)} but expected={ref_shape}."
                )
        return torch.stack(feats, dim=0).mean(dim=0)

    if mode == "concat":
        ref_prefix = tuple(feats[0].shape[:-1])
        for i, f in enumerate(feats[1:], start=1):
            if tuple(f.shape[:-1]) != ref_prefix:
                raise ValueError(
                    "aggregate_views(mode='concat') requires identical shapes on all dims except the last. "
                    f"View idx={i} has shape={tuple(f.shape)} but expected prefix={ref_prefix}."
                )
        return torch.cat(feats, dim=-1)

    raise ValueError(f"Unsupported repr mode: {mode} (supported: mean, concat)")


def repr_dim(base_dim: int, mode: str, n_views: int) -> int:
    m = str(mode).lower().strip()
    if m == "mean":
        return int(base_dim)
    if m == "concat":
        return int(base_dim) * int(n_views)
    raise ValueError(f"Unsupported repr mode for repr_dim: {mode} (supported: mean, concat)")


# =========================================================
# Probes eval
# =========================================================
@torch.no_grad()
def evaluate_probes(
    model: torch.nn.Module,
    probes: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    views_order: List[str],
    repr_mode: str,
    use_projection: bool,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    probes.eval()

    probe_m = probes.module if isinstance(probes, DDP) else probes

    totals: Dict[str, float] = {}
    count = 0

    for batch in loader:
        count += 1
        if max_batches > 0 and count > max_batches:
            break

        if "Y" not in batch:
            raise RuntimeError("Probe evaluation requires labels, but this dataloader was built without targets.")
        y7 = batch["Y"]["seven_label"].to(device, non_blocking=True)
        y10 = batch["Y"]["ten_label"].to(device, non_blocking=True)

        out = model(batch)

        emb = aggregate_views(out["embeddings"], views_order, mode=repr_mode, require_all=True).detach().float()
        emb = _sanitize_tensor(emb, clamp_abs=1e4)

        proj = None
        if use_projection and getattr(model, "module", model).use_projection:
            proj = aggregate_views(out["projections"], views_order, mode=repr_mode, require_all=True).detach().float()
            proj = _sanitize_tensor(proj, clamp_abs=1e4)

        m = probe_m.eval_on_batch(emb, proj, y7, y10)
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + float(v)

    if is_distributed():
        keys = sorted(totals.keys())
        vals = torch.tensor([totals[k] for k in keys] + [float(count)], device=device, dtype=torch.float32)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        count_all = float(vals[-1].item())
        if count_all <= 0:
            return {}
        return {k: float(vals[i].item()) / count_all for i, k in enumerate(keys)}

    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


# =========================================================
# Checkpoint I/O
# =========================================================
def _optimizer_state_to_cpu(optim: torch.optim.Optimizer) -> Dict[str, Any]:
    opt_state = optim.state_dict()
    cpu_state: Dict[str, Any] = {"param_groups": copy.deepcopy(opt_state.get("param_groups", [])), "state": {}}
    for pid, st in opt_state.get("state", {}).items():
        new_st: Dict[str, Any] = {}
        for k, v in st.items():
            if torch.is_tensor(v):
                new_st[k] = v.detach().cpu()
            else:
                new_st[k] = copy.deepcopy(v) if isinstance(v, (dict, list, tuple)) else v
        cpu_state["state"][pid] = new_st
    return cpu_state


def _move_optimizer_state_to_param_device(optim: torch.optim.Optimizer) -> None:
    for p, st in optim.state.items():
        if not torch.is_tensor(p):
            continue
        dev = p.device
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(dev, non_blocking=True)


def save_checkpoint(
    path: str,
    step: int,
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    probes: Optional[torch.nn.Module],
    probe_optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    if not is_rank0():
        return

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    model_m = model.module if isinstance(model, DDP) else model
    state: Dict[str, Any] = {
        "step": int(step),
        "cfg": cfg,
        "model": {k: v.detach().cpu() for k, v in model_m.state_dict().items()},
        "optimizer": _optimizer_state_to_cpu(optimizer),
    }

    if probes is not None:
        probes_m = probes.module if isinstance(probes, DDP) else probes
        if probes_m.has_any():
            state["probes"] = {k: v.detach().cpu() for k, v in probes_m.state_dict().items()}
    if probe_optimizer is not None:
        state["probe_optimizer"] = _optimizer_state_to_cpu(probe_optimizer)

    if scaler is not None:
        try:
            state["scaler"] = scaler.state_dict()
        except Exception:
            state["scaler"] = None

    torch.save(state, path)


def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    probes: Optional[torch.nn.Module],
    probe_optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_m = model.module if isinstance(model, DDP) else model
    model_m.load_state_dict(ckpt["model"], strict=True)

    optimizer.load_state_dict(ckpt["optimizer"])
    _move_optimizer_state_to_param_device(optimizer)

    if probes is not None and ("probes" in ckpt):
        probes_m = probes.module if isinstance(probes, DDP) else probes
        probes_m.load_state_dict(ckpt["probes"], strict=False)

    if probe_optimizer is not None and ("probe_optimizer" in ckpt):
        probe_optimizer.load_state_dict(ckpt["probe_optimizer"])
        _move_optimizer_state_to_param_device(probe_optimizer)

    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    return int(ckpt.get("step", 0))


# =========================================================
# Main
# =========================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None, help="Override training.output_dir")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt")
    args = ap.parse_args()

    device, rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    if is_rank0():
        print(f"[DDP] enabled={distributed} world_size={world_size}")

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.setdefault("training", {})
        cfg["training"]["output_dir"] = args.output_dir

    tc = cfg["training"]
    out_dir = tc["output_dir"]

    if is_rank0():
        os.makedirs(out_dir, exist_ok=True)
        save_yaml(os.path.join(out_dir, "config_used.yaml"), cfg)

    barrier()
    set_seed(int(tc["seed"]) + int(rank))

    train_loader, test_loader = build_loaders(cfg, distributed=distributed)

    model = build_model(cfg).to(device)
    loss_fn = build_losses(cfg).to(device)

    views_order: List[str] = list(cfg["loss"]["views_order"])
    repr_mode = str(cfg.get("probe", {}).get("repr_mode", "mean")).lower().strip()
    use_projection_for_probe = bool(cfg.get("probe", {}).get("use_projection_features", True))

    # ---- numerical stability options ----
    ng = tc.get("nan_guard", {}) or {}
    clamp_feature_abs = float(ng.get("clamp_feature_abs", 1e4)) if ng.get("clamp_feature_abs", None) is not None else 1e4
    reset_probes_on_nonfinite = bool(ng.get("reset_probes_on_nonfinite", False))

    # ---- probes config (correct dims if concat) ----
    pcfg_raw = cfg.get("probe", {}) or {}
    probe_cfg = ProbeConfig(
        enabled=bool(pcfg_raw.get("enabled", True)),
        lr=float(pcfg_raw.get("lr", 1e-3)),
        weight_decay=float(pcfg_raw.get("weight_decay", 0.0)),
        train_on_embeddings=bool(pcfg_raw.get("train_on_embeddings", True)),
        train_on_projections=str(pcfg_raw.get("train_on_projections", "auto")),
        supervised=False,
        supervised_weight=float(pcfg_raw.get("supervised_weight", 1.0)),
    )

    n_views_for_repr = len(views_order)
    emb_in_dim = repr_dim(int(cfg["model"]["common_dim"]), repr_mode, n_views_for_repr)
    proj_in_dim = repr_dim(int(cfg["model"]["projection_dim"]), repr_mode, n_views_for_repr)

    probes = ProbeManager(
        in_dim_emb=emb_in_dim,
        in_dim_proj=proj_in_dim,
        cfg=probe_cfg,
        model_use_projection=bool(cfg["model"].get("use_projection", True)),
    ).to(device)

    ddp_find_unused_parameters = bool(
        cfg.get("training", {}).get("ddp_find_unused_parameters", True)
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=ddp_find_unused_parameters,
        )
        if probes.has_any():
            probes = DDP(
                probes,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

    optim_cfg = tc["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1e-8)),
    )

    probe_optimizer = None
    probe_m = probes.module if isinstance(probes, DDP) else probes
    if probe_m.has_any():
        probe_optimizer = torch.optim.AdamW(
            [p for p in probes.parameters() if p.requires_grad],
            lr=float(probe_cfg.lr),
            weight_decay=float(probe_cfg.weight_decay),
        )

    amp_enabled = bool(tc.get("amp", {}).get("enabled", False))
    use_amp = bool(amp_enabled and device.type == "cuda")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if device.type == "cuda"
        else torch.amp.GradScaler("cpu", enabled=False)
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, device, model, optimizer, probes, probe_optimizer, scaler)
        if is_rank0():
            print(f"[Resume] Loaded {args.resume} at step={start_step}")

    max_steps = int(tc["max_steps"])
    log_every = int(tc.get("log_every", 50))
    save_every = int(tc.get("save_every", 10000))
    eval_every = int(tc.get("eval_every", 2000))
    grad_clip = float(tc.get("grad_clip_norm", 0.0))
    accum = max(1, int(tc.get("grad_accum_steps", 1)))
    probe_grad_clip = float(pcfg_raw.get("grad_clip_norm", tc.get("probe_grad_clip_norm", 0.0)))

    log_path = os.path.join(out_dir, "train_metrics.jsonl")
    test_log_path = os.path.join(out_dir, "test_probe_metrics.jsonl")

    model.train()
    probes.train()

    step = start_step
    t0 = time.time()
    epoch = 0

    while step < max_steps:
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= max_steps:
                break

            micro = step % accum
            do_sync = (micro == accum - 1)

            if micro == 0:
                optimizer.zero_grad(set_to_none=True)
                if probe_optimizer is not None:
                    probe_optimizer.zero_grad(set_to_none=True)
                accum_nonfinite_backbone = 0.0
                accum_nonfinite_probe = 0.0

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            model_sync_ctx = model.no_sync() if (distributed and not do_sync) else contextlib.nullcontext()

            # ------------------------
            # Backbone + GLOBAL-BATCH loss (NaN/Inf guarded)
            # ------------------------
            with model_sync_ctx:
                with autocast_ctx:
                    out = model(batch)

                # sanitize outputs
                out["embeddings"] = _sanitize_view_dict(out.get("embeddings", {}), clamp_abs=clamp_feature_abs)
                out["projections"] = _sanitize_view_dict(out.get("projections", {}), clamp_abs=clamp_feature_abs)

                proj_dict_local = out["projections"]
                proj_dict_g = gather_view_dict_with_grad(proj_dict_local, views_order) if distributed else proj_dict_local
                proj_dict = {k: v.float() for k, v in proj_dict_g.items()}

                proj_list = [proj_dict[v] for v in views_order if v in proj_dict]
                if len(proj_list) < 2:
                    raise RuntimeError("Need at least 2 active views in loss.views_order.")

                proj_seq = torch.stack(proj_list, dim=0)
                try:
                    losses = loss_fn(proj_seq, proj_dict)
                    aux_losses = out.get("aux_losses", {}) or {}
                    aux_metrics = out.get("aux_metrics", {}) or {}
                    for key, value in aux_losses.items():
                        losses[key] = value.float()
                    if aux_losses:
                        total_aux = torch.stack([v.float() for v in aux_losses.values()]).sum()
                        losses["total"] = losses["total"] + total_aux
                    for key, value in aux_metrics.items():
                        losses[key] = value.float()
                    total_loss = losses["total"] / float(accum)
                except Exception:
                    losses = {"total": torch.zeros((), device=device, dtype=torch.float32)}
                    total_loss = losses["total"]
                    dummy = _safe_sum_tensors(list(proj_dict_local.values()), device=device) * 0.0
                    scaler.scale(dummy).backward()
                    accum_nonfinite_backbone = 1.0
                else:
                    if not _tensor_all_finite(total_loss):
                        dummy = _safe_sum_tensors(list(proj_dict_local.values()), device=device) * 0.0
                        scaler.scale(dummy).backward()
                        accum_nonfinite_backbone = 1.0
                    else:
                        scaler.scale(total_loss).backward()

            # ------------------------
            # Detached probes (NaN/Inf guarded)
            # ------------------------
            probe_metrics: Dict[str, float] = {}
            if probe_optimizer is not None:
                if "Y" not in batch:
                    raise RuntimeError("Probe training requires labels, but this dataloader was built without targets.")
                y7 = batch["Y"]["seven_label"].to(device, non_blocking=True)
                y10 = batch["Y"]["ten_label"].to(device, non_blocking=True)

                probes_sync_ctx = probes.no_sync() if (distributed and not do_sync) else contextlib.nullcontext()
                with probes_sync_ctx:
                    emb_feat = aggregate_views(out["embeddings"], views_order, mode=repr_mode, require_all=True).detach().float()
                    emb_feat = _sanitize_tensor(emb_feat, clamp_abs=clamp_feature_abs)

                    proj_feat = None
                    if use_projection_for_probe and getattr(model, "module", model).use_projection:
                        proj_feat = aggregate_views(out["projections"], views_order, mode=repr_mode, require_all=True).detach().float()
                        proj_feat = _sanitize_tensor(proj_feat, clamp_abs=clamp_feature_abs)

                    probe_m = probes.module if isinstance(probes, DDP) else probes
                    try:
                        logits_loss, pm = probe_m.compute_loss_and_metrics(
                            emb_feat if probe_cfg.train_on_embeddings else None,
                            proj_feat,
                            y7,
                            y10,
                            prefix="probe",
                        )
                        logits_loss = logits_loss / float(accum)
                    except Exception:
                        pm = {}
                        dummy = _safe_sum_tensors([p for p in probe_m.parameters()], device=device) * 0.0
                        dummy.backward()
                        accum_nonfinite_probe = 1.0
                        probe_metrics = _sanitize_metrics(pm)
                    else:
                        if not _tensor_all_finite(logits_loss):
                            dummy = _safe_sum_tensors([p for p in probe_m.parameters()], device=device) * 0.0
                            dummy.backward()
                            accum_nonfinite_probe = 1.0
                        else:
                            logits_loss.backward()
                        probe_metrics = _sanitize_metrics(pm)

            # ------------------------
            # Optimizer step boundary
            # ------------------------
            if do_sync:
                # reduce nonfinite flags across ranks as MAX
                if is_distributed():
                    nf = torch.tensor([accum_nonfinite_backbone, accum_nonfinite_probe], device=device, dtype=torch.float32)
                    dist.all_reduce(nf, op=dist.ReduceOp.MAX)
                    nonfinite_backbone = float(nf[0].item())
                    nonfinite_probe = float(nf[1].item())
                else:
                    nonfinite_backbone = float(accum_nonfinite_backbone)
                    nonfinite_probe = float(accum_nonfinite_probe)

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                if probe_optimizer is not None:
                    if probe_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(probes.parameters(), probe_grad_clip)
                    probe_optimizer.step()

                # Optional: if probe loss ever became non-finite, hard reset probes to avoid “stuck NaN probes”
                if reset_probes_on_nonfinite and probe_optimizer is not None and nonfinite_probe > 0:
                    _reset_probes_on_nan(probes, probe_optimizer)

                if step % log_every == 0:
                    dt = time.time() - t0
                    losses_log = {k: _safe_float(v) for k, v in losses.items()}
                    losses_log = reduce_dict_mean({f"pretrain/{k}": v for k, v in losses_log.items()}, device=device)
                    probe_metrics_r = reduce_dict_mean(_sanitize_metrics(probe_metrics), device=device)

                    if is_rank0():
                        record: Dict[str, Any] = {"step": int(step), "time_s": float(dt)}
                        record.update(losses_log)
                        record.update(probe_metrics_r)
                        record["pretrain/nonfinite"] = nonfinite_backbone
                        record["probe/nonfinite"] = nonfinite_probe
                        append_jsonl(log_path, record)

                        msg = f"[Step {step:7d}] pretrain_total={losses_log.get('pretrain/total', 0.0):.6f} | dt={dt:.1f}s"
                        if nonfinite_backbone > 0:
                            msg += " | pretrain_nonfinite=1"
                        if "probe/total" in probe_metrics_r:
                            msg += f" | probe_total={probe_metrics_r['probe/total']:.6f}"
                        if nonfinite_probe > 0:
                            msg += " | probe_nonfinite=1"
                        p7 = probe_metrics_r.get("probe/emb_7_acc", None)
                        p10 = probe_metrics_r.get("probe/emb_10_acc", None)
                        if p7 is not None and p10 is not None:
                            msg += f" | probe_emb_acc7={p7:.3f} acc10={p10:.3f}"
                        print(msg)

                if eval_every > 0 and step > 0 and (step % eval_every == 0) and probe_m.has_any():
                    metrics = evaluate_probes(
                        model=model,
                        probes=probes,
                        loader=test_loader,
                        device=device,
                        views_order=views_order,
                        repr_mode=repr_mode,
                        use_projection=use_projection_for_probe,
                        max_batches=int(cfg.get("probe", {}).get("eval_max_batches", 0)),
                    )
                    if is_rank0():
                        metrics["step"] = int(step)
                        append_jsonl(test_log_path, metrics)
                        print(
                            f"[Eval][step={step}] "
                            + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if k != "step"])
                        )
                    model.train()
                    probes.train()

                if save_every > 0 and step > 0 and (step % save_every == 0):
                    ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
                    save_checkpoint(ckpt_path, step, cfg, model, optimizer, probes, probe_optimizer, scaler)
                    if is_rank0():
                        print(f"[Checkpoint] saved: {ckpt_path}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                del out, proj_dict_local, proj_dict_g, proj_dict, proj_list, proj_seq, losses, total_loss
                if device.type == "cuda":
                    torch.cuda.synchronize()

            step += 1

        epoch += 1

    final_path = os.path.join(out_dir, "ckpt_final.pt")
    save_checkpoint(final_path, step, cfg, model, optimizer, probes, probe_optimizer, scaler)
    if is_rank0():
        print(f"[Done] final checkpoint: {final_path}")

    barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
