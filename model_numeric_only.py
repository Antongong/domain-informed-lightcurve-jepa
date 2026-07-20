# model_numeric_only.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_GPT2_PRESETS = {
    # Canonical GPT-2 configs (n_layer, n_head, n_embd).
    # These are used as *defaults* when explicit depth/heads/embed_dim are not provided.
    "gpt2": {"depth": 12, "heads": 12, "embed_dim": 768, "dropout": 0.1},
    "gpt2-small": {"depth": 12, "heads": 12, "embed_dim": 768, "dropout": 0.1},
    "gpt2-medium": {"depth": 24, "heads": 16, "embed_dim": 1024, "dropout": 0.1},
    "gpt2-large": {"depth": 36, "heads": 20, "embed_dim": 1280, "dropout": 0.1},
    "gpt2-xl": {"depth": 48, "heads": 25, "embed_dim": 1600, "dropout": 0.1},
}


def resolve_gpt2_preset(name: str) -> Dict[str, int | float]:
    key = str(name).strip().lower()
    if key in ("small", "s"):
        key = "gpt2"
    if key not in _GPT2_PRESETS:
        raise ValueError(f"Unknown GPT-2 preset: {name!r}. Supported: {sorted(_GPT2_PRESETS.keys())}")
    return dict(_GPT2_PRESETS[key])

# =========================================================
# Numeric transformer primitives (copied from the original model)
# =========================================================
class ContinuousRotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings where the position is determined by a continuous
    vector (time / frequency) rather than integer indices.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)

        inv_freq = 1.0 / (self.max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        t: (B, S) continuous values (time or frequency or period)
        returns: cos, sin each (B, S, dim)
        """
        freqs = torch.outer(t.flatten(), self.inv_freq).view(t.shape[0], t.shape[1], -1)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: (B, S, H, D)
    cos,sin: (B, S, D)
    """
    cos = cos.unsqueeze(2)  # (B, S, 1, D)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ContinuousAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, S, C)
        mask: (B, S) float/bool with 1/True for valid
        rope_cos/sin: (B, S, head_dim)
        """
        B, S, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, S, H, D)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        q = q.permute(0, 2, 1, 3)  # (B, H, S, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Memory-efficient attention (avoids materializing (B,H,S,S))
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if mask is not None:
                mask_bool = (mask > 0) if mask.dtype != torch.bool else mask
                # In SDPA boolean masks, True means the element participates in attention.
                attn_mask = mask_bool.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )  # (B,H,S,D)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,S,S)
            if mask is not None:
                mask_bool = (mask > 0) if mask.dtype != torch.bool else mask
                key_mask = mask_bool.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
                attn = attn.masked_fill(~key_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v  # (B,H,S,D)

        out = out.transpose(1, 2).reshape(B, S, C)  # (B,S,C)
        out = self.proj(out)
        return out


class NumericTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ContinuousAttention(dim, num_heads=num_heads, dropout=float(dropout))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, rope_cos=cos, rope_sin=sin)
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedStatAttentionPool(nn.Module):
    """
    Pool token features with both learned attention and explicit morphology statistics.
    This preserves amplitude/shape cues better than plain mean pooling.
    """

    def __init__(self, dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        dim = int(dim)
        out_dim = int(out_dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
        hidden = max(dim, out_dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask_bool = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            mask_bool = (mask > 0) if mask.dtype != torch.bool else mask

        mask_f = mask_bool.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)

        scores = self.score(x).squeeze(-1)
        neg_large = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_bool, neg_large)
        attn = torch.softmax(scores, dim=1)
        attn = torch.where(mask_bool, attn, 0.0)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0e-9)
        attn_feat = torch.sum(x * attn.unsqueeze(-1), dim=1)

        mean = torch.sum(x * mask_f, dim=1) / denom
        centered = (x - mean.unsqueeze(1)) * mask_f
        var = torch.sum(centered.square(), dim=1) / denom
        std = torch.sqrt(var.clamp_min(1.0e-9))

        very_neg = torch.full_like(x, torch.finfo(x.dtype).min)
        max_feat = torch.max(torch.where(mask_bool.unsqueeze(-1), x, very_neg), dim=1).values
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, torch.zeros_like(max_feat))

        pooled = torch.cat([attn_feat, mean, std, max_feat], dim=-1)
        return self.fuse(pooled)


class NumericTransformer(nn.Module):
    """
    Numeric transformer that consumes *token embeddings* (already embedded), and uses continuous RoPE.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: Optional[int] = None,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        rope_max_period: float = 10000.0,
        apply_final_norm: bool = True,
        pooling_mode: str = "mean",
        pooling_dropout: float = 0.0,
    ):
        super().__init__()
        embed_dim = int(embed_dim)
        depth = int(depth)
        num_heads = int(num_heads)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.out_dim = int(out_dim) if out_dim is not None else embed_dim
        self.apply_final_norm = bool(apply_final_norm)
        self.pooling_mode = str(pooling_mode).lower().strip()

        self.emb_dropout = nn.Dropout(float(dropout))
        self.rope = ContinuousRotaryEmbedding(head_dim, max_period=float(rope_max_period))

        self.layers = nn.ModuleList(
            [
                NumericTransformerBlock(
                    embed_dim, num_heads=num_heads, mlp_ratio=float(mlp_ratio), dropout=float(dropout)
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if self.pooling_mode == "mean":
            self.pool = None
            self.out_proj = nn.Identity() if self.out_dim == embed_dim else nn.Linear(embed_dim, self.out_dim)
        elif self.pooling_mode == "attn_stats":
            self.pool = MaskedStatAttentionPool(embed_dim, self.out_dim, dropout=float(pooling_dropout))
            self.out_proj = None
        else:
            raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}. Expected 'mean' or 'attn_stats'.")

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,S,embed_dim) token embeddings
        t: (B,S) continuous positions
        mask: (B,S) 1/0 for valid
        returns: (B,out_dim)
        """
        x = self.emb_dropout(x)
        cos, sin = self.rope(t)  # (B,S,head_dim)

        for blk in self.layers:
            x = blk(x, mask=mask, cos=cos, sin=sin)

        if self.apply_final_norm:
            x = self.norm(x)

        if self.pooling_mode == "attn_stats":
            assert self.pool is not None
            return self.pool(x, mask=mask)

        if mask is not None:
            m = (mask > 0).float().unsqueeze(-1) if mask.dtype != torch.bool else mask.float().unsqueeze(-1)
            s = torch.sum(x * m, dim=1)
            denom = torch.clamp(m.sum(dim=1), min=1e-9)
            pooled = s / denom
        else:
            pooled = x.mean(dim=1)

        assert self.out_proj is not None
        return self.out_proj(pooled)

# =========================================================
# Quantized embedding (copied from the original model)
# =========================================================
class QuantizedGaussianEmbedding(nn.Module):
    """
    Quantize a scalar to one of N bins, then embed via nn.Embedding.

    For photometric magnitude with uncertainty:
      - the hard one-hot token is dispersed into a smooth discrete Gaussian over bins
        within ±(truncate_sigma * sigma) in VALUE space, implemented as a ±R token window
        where R = ceil(truncate_sigma * sigma * scale).
      - the discrete Gaussian is evaluated at bin-centers and normalized within the window.

    If sigma > sigma_missing_threshold, sigma is treated as "not provided", and dispersion is disabled
    (mass concentrated on a single token).
    """

    def __init__(
        self,
        num_bins: int,
        embed_dim: int,
        vmin: float,
        vmax: float,
        *,
        sigma_missing_threshold: float = 5.0,
        truncate_sigma: float = 3.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.embed_dim = int(embed_dim)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.sigma_missing_threshold = float(sigma_missing_threshold)
        self.truncate_sigma = float(truncate_sigma)
        self.eps = float(eps)

        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if not (self.vmax > self.vmin):
            raise ValueError("vmax must be > vmin")
        if self.truncate_sigma <= 0:
            raise ValueError("truncate_sigma must be > 0")

        self.embedding = nn.Embedding(self.num_bins, self.embed_dim)

        # scale converts value units to token units (same as original quantizer)
        scale = (self.num_bins - 1) / (self.vmax - self.vmin)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32), persistent=False)

        # Bin "centers" consistent with round-based quantization grid:
        # idx 0 -> vmin, idx (N-1) -> vmax (grid points). This is the natural reference
        # for evaluating a discrete Gaussian over token indices.
        centers = self.vmin + torch.arange(self.num_bins, dtype=torch.float32) / self.scale
        self.register_buffer("centers", centers, persistent=False)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: float tensor
        returns: long tensor of indices in [0, num_bins-1]
        """
        v = torch.nan_to_num(v, nan=0.0, posinf=self.vmax, neginf=self.vmin)
        v = torch.clamp(v, min=self.vmin, max=self.vmax)
        idx = torch.round((v - self.vmin) * self.scale).long()
        return torch.clamp(idx, 0, self.num_bins - 1)

    def forward(self, v: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        v: (B,S) float tensor
        sigma: (B,S) float tensor (optional). If provided, dispersion is applied.

        returns: (B,S,embed_dim)
        """
        v = torch.nan_to_num(v, nan=0.0, posinf=self.vmax, neginf=self.vmin).float()

        # Hard token embedding path (also used as fallback)
        idx0 = self.quantize(v)
        hard = self.embedding(idx0)

        if sigma is None:
            return hard

        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0).float()
        sigma = torch.clamp(sigma, min=0.0)

        # Decide where to apply Gaussian dispersion
        use_gauss = (
            torch.isfinite(v)
            & torch.isfinite(sigma)
            & (sigma > 0.0)
            & (sigma <= self.sigma_missing_threshold)
        )
        if not bool(use_gauss.any()):
            return hard

        # Token-radius corresponding to ±(truncate_sigma * sigma) in value space
        # radius_tokens = ceil(truncate_sigma * sigma * scale)
        radius = torch.ceil(self.truncate_sigma * sigma * self.scale).long()
        radius = torch.clamp(radius, min=0, max=self.num_bins - 1)
        radius = torch.where(use_gauss, radius, torch.zeros_like(radius))

        rmax = int(radius.max().item())
        if rmax == 0:
            return hard  # or fall through to blending

        offsets = torch.arange(-rmax, rmax + 1, device=v.device, dtype=torch.long)  # (K,)

        sigma_safe = torch.clamp(sigma, min=self.eps)
        v_exp = v.unsqueeze(-1)  # (B,S,1)

        # Accumulators
        num = torch.zeros_like(hard)                      # (B,S,D)
        den = torch.zeros(v.shape, device=v.device, dtype=hard.dtype)  # (B,S)

        chunk = 16  # tune: 16/32/64
        for off in offsets.split(chunk):
            # (B,S,kc)
            idx_chunk = idx0.unsqueeze(-1) + off.view(1, 1, -1)

            in_radius = off.abs().view(1, 1, -1) <= radius.unsqueeze(-1)
            in_bounds = (idx_chunk >= 0) & (idx_chunk < self.num_bins)
            valid = in_radius & in_bounds & use_gauss.unsqueeze(-1)

            idx_safe_chunk = idx_chunk.clamp(0, self.num_bins - 1)

            # centers gather only for this chunk
            c = self.centers.to(v.device)[idx_safe_chunk]              # (B,S,kc)
            diff = c - v_exp                                           # (B,S,kc)

            w = torch.exp(-0.5 * (diff / sigma_safe.unsqueeze(-1)) ** 2)
            w = w * valid.to(w.dtype)

            # accumulate
            den = den + w.sum(dim=-1).to(den.dtype)                    # (B,S)
            emb_chunk = self.embedding(idx_safe_chunk)                 # (B,S,kc,D) -- small kc
            num = num + (w.unsqueeze(-1).to(emb_chunk.dtype) * emb_chunk).sum(dim=-2)

        gauss = num / den.clamp_min(self.eps).unsqueeze(-1)

        out = torch.where(use_gauss.unsqueeze(-1), gauss, hard)
        return out

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)

        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarTokenEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        dim = int(dim)
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else dim
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, value: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if valid is None:
            valid = torch.isfinite(value)
        valid_f = valid.float()
        x = torch.stack([value, valid_f], dim=-1)
        return self.net(x)


class GroupFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        dim = int(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ContinuousAttention(dim, num_heads=int(num_heads), dropout=float(dropout))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GroupViewFusion(nn.Module):
    TOKEN_ORDER = [
        "cls",
        "raw",
        "periodogram",
        "phase_folded",
        "best_period",
        "best_power",
        "time_span",
        "mag_std",
        "valid_fraction",
    ]

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        dim = int(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_embed = nn.Embedding(len(self.TOKEN_ORDER), dim)
        self.type_to_idx = {name: idx for idx, name in enumerate(self.TOKEN_ORDER)}
        self.blocks = nn.ModuleList(
            [GroupFusionBlock(dim, num_heads=int(num_heads), mlp_ratio=2.0, dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, view_embeddings: Dict[str, torch.Tensor], covariate_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(view_embeddings.values())).device if view_embeddings else next(iter(covariate_tokens.values())).device
        batch_size = next(iter(view_embeddings.values())).shape[0] if view_embeddings else next(iter(covariate_tokens.values())).shape[0]

        tokens: List[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]
        type_ids: List[int] = [self.type_to_idx["cls"]]

        for name in ("raw", "periodogram", "phase_folded"):
            if name in view_embeddings:
                tokens.append(view_embeddings[name].unsqueeze(1))
                type_ids.append(self.type_to_idx[name])

        for name in ("best_period", "best_power", "time_span", "mag_std", "valid_fraction"):
            if name in covariate_tokens:
                tokens.append(covariate_tokens[name].unsqueeze(1))
                type_ids.append(self.type_to_idx[name])

        x = torch.cat(tokens, dim=1)
        type_tensor = torch.tensor(type_ids, device=device, dtype=torch.long)
        x = x + self.type_embed(type_tensor).unsqueeze(0)
        mask = torch.ones((batch_size, x.shape[1]), device=device, dtype=torch.bool)

        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        return x[:, 0]


class QuantileForecastHead(nn.Module):
    def __init__(self, in_dim: int, horizon: int, quantiles: List[float], hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        in_dim = int(in_dim)
        horizon = int(horizon)
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else in_dim
        self.horizon = horizon
        self.quantiles = tuple(float(q) for q in quantiles)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, horizon * len(self.quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        return out.view(x.shape[0], self.horizon, len(self.quantiles))

# =========================================================
# Numeric-only multi-view model
# =========================================================
class MultiModalAstroModel(nn.Module):
    """
    Revised numeric-only model (no text / no images).

    Views:
      - raw:           light curve (time, mag, mag_err, mask)
      - periodogram:   (period, log10_power)
      - phase_folded:  phase-folded LC (phase_time, mag, mag_err, mask)

    Key revision:
      - If period_position_mode == "time", positions = log10(period).
    """

    VIEW_RAW = "raw"
    VIEW_PERIODOGRAM = "periodogram"
    VIEW_PHASE_FOLDED = "phase_folded"
    VIEW_GROUP = "group"

    def __init__(
        self,
        *,
        common_dim: int = 256,
        use_projection: bool = True,
        projection_dim: int = 256,
        projection_hidden_dim: int = 0,
        projection_dropout: float = 0.0,

        # numeric presets / overrides (kept compatible with the old config style)
        numeric_model_size: str = "gpt2",

        # raw
        raw_embed_dim: Optional[int] = None,
        raw_depth: Optional[int] = None,
        raw_heads: Optional[int] = None,
        raw_dropout: Optional[float] = None,
        raw_mlp_ratio: float = 4.0,
        raw_rope_max_period: float = 10000.0,

        # periodogram
        per_embed_dim: Optional[int] = None,
        per_depth: Optional[int] = None,
        per_heads: Optional[int] = None,
        per_dropout: Optional[float] = None,
        per_mlp_ratio: float = 4.0,
        per_rope_max_period: float = 10000.0,

        # phase-folded
        pf_embed_dim: Optional[int] = None,
        pf_depth: Optional[int] = None,
        pf_heads: Optional[int] = None,
        pf_dropout: Optional[float] = None,
        pf_mlp_ratio: float = 4.0,
        pf_rope_max_period: float = 10000.0,

        # ablations
        raw_position_mode: str = "time",       # "time" | "index"
        period_position_mode: str = "time",    # "time" | "index"
        phase_position_mode: str = "time",     # "time" | "index"
        raw_use_uncertainty: bool = True,
        pf_use_uncertainty: bool = True,
        phase_use_normalized_phase: bool = True,

        # quantization ranges (defaults kept from old conventions)
        raw_num_bins: int = 512,
        raw_vmin: float = -2.0,
        raw_vmax: float = 2.0,
        per_num_bins: int = 512,
        per_vmin: float = -6.0,
        per_vmax: float = 2.0,

        # encoder pooling
        pooling_mode: str = "attn_stats",
        pooling_dropout: float = 0.0,

        # grouped multi-view fusion
        group_enabled: bool = True,
        group_num_layers: int = 2,
        group_num_heads: int = 8,
        group_dropout: float = 0.1,
        group_use_best_period: bool = True,
        group_use_best_power: bool = True,
        group_use_time_span: bool = True,
        group_use_mag_std: bool = True,
        group_use_valid_fraction: bool = True,

        # forecasting-style auxiliary objective
        forecast_enabled: bool = False,
        forecast_horizon: int = 16,
        forecast_min_context: int = 32,
        forecast_quantiles: Optional[List[float]] = None,
        forecast_weight: float = 0.2,
        forecast_hidden_dim: int = 0,
        forecast_dropout: float = 0.1,
        forecast_max_samples_per_batch: int = 8,

        # raw-input inference preprocessing
        inference_min_period: float = 0.0069444444,
        inference_max_period: float = 2000.0,
        inference_k_periods: int = 1_000_000,
        inference_chunk_size: int = 8192,
        inference_k_top: int = 500,
        inference_k_rand: int = 500,
        inference_min_valid_points: int = 8,
        inference_eps: float = 1e-12,
    ):
        super().__init__()

        self.common_dim = int(common_dim)
        self.use_projection = bool(use_projection)

        self.raw_position_mode = str(raw_position_mode).lower()
        self.period_position_mode = str(period_position_mode).lower()
        self.phase_position_mode = str(phase_position_mode).lower()

        self.raw_use_uncertainty = bool(raw_use_uncertainty)
        self.pf_use_uncertainty = bool(pf_use_uncertainty)
        self.phase_use_normalized_phase = bool(phase_use_normalized_phase)
        self.inference_min_period = float(inference_min_period)
        self.inference_max_period = float(inference_max_period)
        self.inference_k_periods = int(inference_k_periods)
        self.inference_chunk_size = int(inference_chunk_size)
        self.inference_k_top = int(inference_k_top)
        self.inference_k_rand = int(inference_k_rand)
        self.inference_min_valid_points = int(inference_min_valid_points)
        self.inference_eps = float(inference_eps)
        if self.inference_min_period <= 0.0:
            raise ValueError("inference_min_period must be > 0.")
        if self.inference_max_period <= self.inference_min_period:
            raise ValueError("inference_max_period must be > inference_min_period.")
        if self.inference_k_periods <= 1:
            raise ValueError("inference_k_periods must be > 1.")
        if self.inference_k_top < 0 or self.inference_k_rand < 0:
            raise ValueError("inference_k_top and inference_k_rand must be >= 0.")
        if self.inference_k_top + self.inference_k_rand <= 0:
            raise ValueError("inference_k_top + inference_k_rand must be > 0.")
        if self.inference_k_top + self.inference_k_rand > self.inference_k_periods:
            raise ValueError("inference_k_top + inference_k_rand cannot exceed inference_k_periods.")
        self._inference_period_grid: Optional[torch.Tensor] = None
        self.pooling_mode = str(pooling_mode).lower().strip()
        self.pooling_dropout = float(pooling_dropout)
        self.group_enabled = bool(group_enabled)
        self.group_use_best_period = bool(group_use_best_period)
        self.group_use_best_power = bool(group_use_best_power)
        self.group_use_time_span = bool(group_use_time_span)
        self.group_use_mag_std = bool(group_use_mag_std)
        self.group_use_valid_fraction = bool(group_use_valid_fraction)
        self.forecast_enabled = bool(forecast_enabled)
        self.forecast_horizon = int(forecast_horizon)
        self.forecast_min_context = int(forecast_min_context)
        self.forecast_quantiles = [float(q) for q in (forecast_quantiles or [0.1, 0.5, 0.9])]
        self.forecast_weight = float(forecast_weight)
        self.forecast_max_samples_per_batch = int(forecast_max_samples_per_batch)

        # preset
        preset = resolve_gpt2_preset(numeric_model_size)

        def _pick(default: Dict[str, int | float], embed_dim, depth, heads, dropout) -> Tuple[int, int, int, float]:
            ed = int(embed_dim) if embed_dim is not None else int(default["embed_dim"])
            dp = int(depth) if depth is not None else int(default["depth"])
            hd = int(heads) if heads is not None else int(default["heads"])
            dr = float(dropout) if dropout is not None else float(default["dropout"])
            return ed, dp, hd, dr

        raw_ed, raw_dp, raw_hd, raw_dr = _pick(preset, raw_embed_dim, raw_depth, raw_heads, raw_dropout)
        per_ed, per_dp, per_hd, per_dr = _pick(preset, per_embed_dim, per_depth, per_heads, per_dropout)
        pf_ed, pf_dp, pf_hd, pf_dr  = _pick(preset, pf_embed_dim, pf_depth, pf_heads, pf_dropout)

        # value embeddings
        self.raw_value_embed = QuantizedGaussianEmbedding(
            num_bins=int(raw_num_bins),
            embed_dim=raw_ed,
            vmin=float(raw_vmin),
            vmax=float(raw_vmax),
        )
        self.period_value_embed = QuantizedGaussianEmbedding(
            num_bins=int(per_num_bins),
            embed_dim=per_ed,
            vmin=float(per_vmin),
            vmax=float(per_vmax),
        )
        self.phase_value_embed = QuantizedGaussianEmbedding(
            num_bins=int(raw_num_bins),
            embed_dim=pf_ed,
            vmin=float(raw_vmin),
            vmax=float(raw_vmax),
        )

        # numeric encoders
        self.raw_encoder = NumericTransformer(
            embed_dim=raw_ed,
            out_dim=self.common_dim,
            depth=raw_dp,
            num_heads=raw_hd,
            mlp_ratio=float(raw_mlp_ratio),
            dropout=float(raw_dr),
            rope_max_period=float(raw_rope_max_period),
            apply_final_norm=True,
            pooling_mode=self.pooling_mode,
            pooling_dropout=self.pooling_dropout,
        )
        self.periodogram_encoder = NumericTransformer(
            embed_dim=per_ed,
            out_dim=self.common_dim,
            depth=per_dp,
            num_heads=per_hd,
            mlp_ratio=float(per_mlp_ratio),
            dropout=float(per_dr),
            rope_max_period=float(per_rope_max_period),
            apply_final_norm=True,
            pooling_mode=self.pooling_mode,
            pooling_dropout=self.pooling_dropout,
        )
        self.phase_encoder = NumericTransformer(
            embed_dim=pf_ed,
            out_dim=self.common_dim,
            depth=pf_dp,
            num_heads=pf_hd,
            mlp_ratio=float(pf_mlp_ratio),
            dropout=float(pf_dr),
            rope_max_period=float(pf_rope_max_period),
            apply_final_norm=True,
            pooling_mode=self.pooling_mode,
            pooling_dropout=self.pooling_dropout,
        )

        # projectors
        self.projectors = nn.ModuleDict()
        if self.use_projection:
            self.projectors[self.VIEW_RAW] = ProjectionHead(
                in_dim=self.common_dim,
                out_dim=int(projection_dim),
                hidden_dim=int(projection_hidden_dim),
                dropout=float(projection_dropout),
            )
            self.projectors[self.VIEW_PERIODOGRAM] = ProjectionHead(
                in_dim=self.common_dim,
                out_dim=int(projection_dim),
                hidden_dim=int(projection_hidden_dim),
                dropout=float(projection_dropout),
            )
            self.projectors[self.VIEW_PHASE_FOLDED] = ProjectionHead(
                in_dim=self.common_dim,
                out_dim=int(projection_dim),
                hidden_dim=int(projection_hidden_dim),
                dropout=float(projection_dropout),
            )
            if self.group_enabled:
                self.projectors[self.VIEW_GROUP] = ProjectionHead(
                    in_dim=self.common_dim,
                    out_dim=int(projection_dim),
                    hidden_dim=int(projection_hidden_dim),
                    dropout=float(projection_dropout),
                )

        # enable all 3 numeric views by default (config can still set these keys)
        self.enable: Dict[str, bool] = {
            self.VIEW_RAW: True,
            self.VIEW_PERIODOGRAM: True,
            self.VIEW_PHASE_FOLDED: True,
            self.VIEW_GROUP: self.group_enabled,
        }

        if self.group_enabled:
            self.group_fusion = GroupViewFusion(
                dim=self.common_dim,
                num_layers=int(group_num_layers),
                num_heads=int(group_num_heads),
                dropout=float(group_dropout),
            )
            self.covariate_embedders = nn.ModuleDict(
                {
                    "best_period": ScalarTokenEmbedding(self.common_dim, hidden_dim=self.common_dim, dropout=float(group_dropout)),
                    "best_power": ScalarTokenEmbedding(self.common_dim, hidden_dim=self.common_dim, dropout=float(group_dropout)),
                    "time_span": ScalarTokenEmbedding(self.common_dim, hidden_dim=self.common_dim, dropout=float(group_dropout)),
                    "mag_std": ScalarTokenEmbedding(self.common_dim, hidden_dim=self.common_dim, dropout=float(group_dropout)),
                    "valid_fraction": ScalarTokenEmbedding(self.common_dim, hidden_dim=self.common_dim, dropout=float(group_dropout)),
                }
            )
        else:
            self.group_fusion = None
            self.covariate_embedders = nn.ModuleDict()

        if self.forecast_enabled:
            self.forecast_head = QuantileForecastHead(
                in_dim=self.common_dim,
                horizon=self.forecast_horizon,
                quantiles=self.forecast_quantiles,
                hidden_dim=int(forecast_hidden_dim) if forecast_hidden_dim else self.common_dim,
                dropout=float(forecast_dropout),
            )
        else:
            self.forecast_head = None

    @staticmethod
    def _make_index_positions(B: int, S: int, device: torch.device) -> torch.Tensor:
        return torch.arange(S, device=device, dtype=torch.float32).view(1, S).expand(B, S)

    @staticmethod
    def _masked_span(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = (mask > 0) if mask.dtype != torch.bool else mask
        pos_inf = torch.full_like(values, float("inf"))
        neg_inf = torch.full_like(values, float("-inf"))
        vmin = torch.where(mask_bool, values, pos_inf).min(dim=1).values
        vmax = torch.where(mask_bool, values, neg_inf).max(dim=1).values
        span = vmax - vmin
        valid = mask_bool.any(dim=1)
        span = torch.where(valid, span, torch.zeros_like(span))
        return torch.nan_to_num(span, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_covariate_tokens(
        self,
        *,
        lc: Optional[torch.Tensor] = None,
        best_period: Optional[torch.Tensor] = None,
        best_power: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens: Dict[str, torch.Tensor] = {}
        if not self.group_enabled:
            return tokens

        if lc is not None:
            lc = lc.float()
            time = lc[..., 0]
            mag = lc[..., 1]
            mask = lc[..., 3] > 0
            mask_f = mask.float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean = (mag * mask_f).sum(dim=1) / denom
            centered = (mag - mean.unsqueeze(1)) * mask_f
            std = torch.sqrt((centered.square().sum(dim=1) / denom).clamp_min(1.0e-9))
            frac = mask_f.mean(dim=1)
            span = self._masked_span(time, mask)

            if self.group_use_time_span:
                tokens["time_span"] = self.covariate_embedders["time_span"](torch.log10(span.clamp_min(1.0e-6)))
            if self.group_use_mag_std:
                tokens["mag_std"] = self.covariate_embedders["mag_std"](std)
            if self.group_use_valid_fraction:
                tokens["valid_fraction"] = self.covariate_embedders["valid_fraction"](frac)

        if best_period is not None and self.group_use_best_period:
            bp = torch.nan_to_num(best_period.float(), nan=0.0, posinf=0.0, neginf=0.0)
            valid = torch.isfinite(best_period) & (best_period > 0)
            tokens["best_period"] = self.covariate_embedders["best_period"](torch.log10(bp.clamp_min(1.0e-6)), valid=valid)

        if best_power is not None and self.group_use_best_power:
            pw = torch.nan_to_num(best_power.float(), nan=0.0, posinf=0.0, neginf=0.0)
            valid = torch.isfinite(best_power)
            tokens["best_power"] = self.covariate_embedders["best_power"](torch.log1p(pw.clamp_min(0.0)), valid=valid)

        return tokens

    def _pinball_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = pred.new_tensor(self.forecast_quantiles).view(1, 1, -1)
        err = target.unsqueeze(-1) - pred
        loss = torch.maximum(q * err, (q - 1.0) * err)
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
        return (loss * mask_f).sum() / denom

    def _encode_raw_view(self, lc: torch.Tensor, device: torch.device) -> torch.Tensor:
        t = lc[..., 0]
        v = lc[..., 1]
        s = lc[..., 2]
        mask = lc[..., 3]

        if self.raw_position_mode == "index":
            t = self._make_index_positions(v.shape[0], v.shape[1], device=device)
        elif self.raw_position_mode != "time":
            raise ValueError(f"Unsupported raw_position_mode={self.raw_position_mode!r}")

        x = self.raw_value_embed(v, sigma=s if self.raw_use_uncertainty else None)
        return self.raw_encoder(x, t, mask=mask)

    def _encode_periodogram_view(self, pg: torch.Tensor, device: torch.device) -> torch.Tensor:
        per = pg[..., 0].clamp_min(1e-12)
        v = pg[..., 1]
        mask = torch.ones_like(v, dtype=torch.bool)

        if self.period_position_mode == "index":
            t = self._make_index_positions(v.shape[0], v.shape[1], device=device)
        elif self.period_position_mode == "time":
            t = torch.log10(per)
        else:
            raise ValueError(f"Unsupported period_position_mode={self.period_position_mode!r}")

        x = self.period_value_embed(v, sigma=None)
        return self.periodogram_encoder(x, t, mask=mask)

    def _encode_phase_view(self, pflc: torch.Tensor, best_period: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        phase_time = pflc[..., 0]
        v = pflc[..., 1]
        s = pflc[..., 2]
        mask = pflc[..., 3]

        if self.phase_use_normalized_phase and best_period is not None:
            bp = best_period.to(device, non_blocking=True).view(-1, 1).clamp_min(1e-12)
            phase = (phase_time / bp).clamp(0.0, 1.0)
        else:
            phase = phase_time

        if self.phase_position_mode == "index":
            t = self._make_index_positions(v.shape[0], v.shape[1], device=device)
        elif self.phase_position_mode == "time":
            t = phase
        else:
            raise ValueError(f"Unsupported phase_position_mode={self.phase_position_mode!r}")

        x = self.phase_value_embed(v, sigma=s if self.pf_use_uncertainty else None)
        return self.phase_encoder(x, t, mask=mask)

    def _resolve_device(self, *candidates: Any) -> torch.device:
        param = next(self.parameters(), None)
        if param is not None:
            return param.device

        buf = next(self.buffers(), None)
        if buf is not None:
            return buf.device

        stack: List[Any] = list(candidates)
        while stack:
            value = stack.pop()
            if torch.is_tensor(value):
                return value.device
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)

        raise RuntimeError("Could not determine model device from parameters, buffers, or inputs.")

    def _compute_forecast_aux(self, xdict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.forecast_enabled or self.forecast_head is None or "lc" not in xdict:
            return None

        device = self._resolve_device(xdict)
        lc = xdict["lc"].to(device, non_blocking=True).float()
        mask = lc[..., 3] > 0
        lengths = mask.sum(dim=1).long()
        eligible = lengths >= (self.forecast_min_context + self.forecast_horizon)
        if not bool(eligible.any()):
            return None

        eligible_idx = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        if self.forecast_max_samples_per_batch > 0 and eligible_idx.numel() > self.forecast_max_samples_per_batch:
            pick = torch.randperm(eligible_idx.numel(), device=device)[: self.forecast_max_samples_per_batch]
            used_idx = eligible_idx[pick]
        else:
            used_idx = eligible_idx

        prefix_lc = lc[used_idx].clone()
        used_mask = mask[used_idx]
        target = lc.new_zeros((used_idx.numel(), self.forecast_horizon))
        target_mask = torch.zeros((used_idx.numel(), self.forecast_horizon), device=device, dtype=torch.bool)

        for b in range(used_idx.numel()):
            valid_idx = torch.nonzero(used_mask[b], as_tuple=False).squeeze(1)
            future_idx = valid_idx[-self.forecast_horizon :]
            prefix_lc[b, future_idx, 3] = 0.0
            target[b] = lc[used_idx[b], future_idx, 1]
            target_mask[b] = True

        prefix_raw = self._encode_raw_view(prefix_lc, device=device)
        if self.group_enabled and self.group_fusion is not None:
            prefix_cov = self._build_covariate_tokens(lc=prefix_lc)
            context = self.group_fusion({self.VIEW_RAW: prefix_raw}, prefix_cov)
        else:
            context = prefix_raw

        pred = self.forecast_head(context)
        forecast_loss = self._pinball_loss(pred, target, target_mask) * self.forecast_weight
        median_idx = min(range(len(self.forecast_quantiles)), key=lambda i: abs(self.forecast_quantiles[i] - 0.5))
        median_mae = (pred[..., median_idx] - target).abs()
        median_mae = (median_mae * target_mask.float()).sum() / target_mask.float().sum().clamp_min(1.0)

        return {
            "forecast_total": forecast_loss,
            "forecast_q50_mae": median_mae.detach(),
            "forecast_eligible_frac": eligible.float().mean().detach(),
        }

    def _get_inference_period_grid(self, device: torch.device) -> torch.Tensor:
        grid = self._inference_period_grid
        if grid is None or grid.device != device:
            grid = torch.exp(
                torch.linspace(
                    math.log(self.inference_min_period),
                    math.log(self.inference_max_period),
                    self.inference_k_periods,
                    device=device,
                    dtype=torch.float32,
                )
            )
            self._inference_period_grid = grid
        return grid

    @staticmethod
    def _to_batched_tensor(
        value: Any,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        name: str,
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [N] or [B,N]. Got {tuple(tensor.shape)}.")
        return tensor

    def _extract_raw_light_curve(self, xdict: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
        if "raw_lc" in xdict:
            raw = xdict["raw_lc"]
        elif all(k in xdict for k in ("time", "mag")):
            time = self._to_batched_tensor(xdict["time"], device=device, dtype=torch.float32, name="time")
            mag = self._to_batched_tensor(xdict["mag"], device=device, dtype=torch.float32, name="mag")
            if time.shape != mag.shape:
                raise ValueError(f"time and mag must have the same shape. Got {tuple(time.shape)} vs {tuple(mag.shape)}.")

            mag_err_value = xdict.get("mag_err", None)
            if mag_err_value is None:
                mag_err = torch.full_like(mag, 0.1)
            else:
                mag_err = self._to_batched_tensor(mag_err_value, device=device, dtype=torch.float32, name="mag_err")
                if mag_err.shape != time.shape:
                    raise ValueError(
                        f"mag_err must have the same shape as time. Got {tuple(mag_err.shape)} vs {tuple(time.shape)}."
                    )

            mask_value = xdict.get("mask", None)
            if mask_value is None:
                mask = torch.ones_like(time, dtype=torch.bool)
            else:
                if torch.is_tensor(mask_value):
                    mask = mask_value.to(device=device, non_blocking=True)
                else:
                    mask = torch.as_tensor(mask_value, device=device)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                if mask.shape != time.shape:
                    raise ValueError(
                        f"mask must have the same shape as time. Got {tuple(mask.shape)} vs {tuple(time.shape)}."
                    )
                mask = mask > 0

            raw = torch.stack([time, mag, mag_err, mask.to(time.dtype)], dim=-1)
        elif "lc" in xdict and ("periodogram" not in xdict or "phase_folded_lc" not in xdict):
            raw = xdict["lc"]
        else:
            return None

        if torch.is_tensor(raw):
            raw_lc = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            raw_lc = torch.as_tensor(raw, device=device, dtype=torch.float32)
        if raw_lc.ndim == 2:
            raw_lc = raw_lc.unsqueeze(0)
        if raw_lc.ndim != 3 or raw_lc.shape[-1] not in (3, 4):
            raise ValueError(
                "raw light-curve input must have shape [N,3], [N,4], [B,N,3], or [B,N,4]. "
                f"Got {tuple(raw_lc.shape)}."
            )
        if raw_lc.shape[-1] == 3:
            ones = torch.ones(raw_lc.shape[:-1] + (1,), device=device, dtype=raw_lc.dtype)
            raw_lc = torch.cat([raw_lc, ones], dim=-1)
        return raw_lc

    @staticmethod
    def _gls_batch_gpu(
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
            raise ValueError("t, y, yerr, mask must all have shape [B,N].")
        if t.shape != y.shape or t.shape != yerr.shape or t.shape != mask.shape:
            raise ValueError("t, y, yerr, mask must all share shape [B,N].")
        if periods.ndim != 1:
            raise ValueError("periods must have shape [K].")

        device = t.device
        B, _ = t.shape

        t = t.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        yerr = yerr.to(device=device, dtype=dtype)
        periods = periods.to(device=device, dtype=dtype).clamp_min(eps)
        mask = mask.to(device=device, dtype=torch.bool)

        w = torch.zeros_like(t, dtype=dtype)
        valid = mask & (yerr > 0)
        w[valid] = 1.0 / (yerr[valid] ** 2)

        W = torch.sum(w, dim=1).clamp_min(eps)
        Y = torch.sum(w * y, dim=1)
        YY = torch.sum(w * y * y, dim=1)
        var_y = (YY - (Y * Y) / W).clamp_min(eps)

        W1 = W.view(B, 1)
        Y1 = Y.view(B, 1)
        var_y1 = var_y.view(B, 1)

        K = periods.numel()
        power_all = torch.empty((B, K), dtype=dtype, device=device)
        best_power = torch.full((B,), -float("inf"), dtype=dtype, device=device)
        best_period = torch.zeros((B,), dtype=dtype, device=device)

        t3 = t.unsqueeze(-1)
        y3 = y.unsqueeze(-1)
        w3 = w.unsqueeze(-1)
        two_pi = 2.0 * math.pi

        for k0 in range(0, K, chunk_size):
            k1 = min(k0 + chunk_size, K)
            per_chunk = periods[k0:k1]
            omega = (two_pi / per_chunk).view(1, 1, -1)

            phase = t3 * omega
            c = torch.cos(phase)
            s = torch.sin(phase)

            C = torch.sum(w3 * c, dim=1)
            S = torch.sum(w3 * s, dim=1)
            CC = torch.sum(w3 * c * c, dim=1)
            SS = torch.sum(w3 * s * s, dim=1)
            CS = torch.sum(w3 * c * s, dim=1)
            YC = torch.sum(w3 * y3 * c, dim=1)
            YS = torch.sum(w3 * y3 * s, dim=1)

            YC0 = YC - (Y1 * C) / W1
            YS0 = YS - (Y1 * S) / W1
            D = (CC * SS - CS * CS).clamp_min(eps)

            num = (SS * (YC0 * YC0) + CC * (YS0 * YS0) - 2.0 * CS * YC0 * YS0)
            power_chunk = torch.nan_to_num(num / (D * var_y1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

            power_all[:, k0:k1] = power_chunk

            chunk_max, chunk_arg = torch.max(power_chunk, dim=1)
            better = chunk_max > best_power
            if better.any():
                best_power[better] = chunk_max[better]
                global_idx = (chunk_arg + k0).to(torch.long)
                best_period[better] = periods[global_idx[better]]

        return power_all, best_power, best_period

    @staticmethod
    def _select_periodogram_points_exact(
        power_all: torch.Tensor,
        periods: torch.Tensor,
        *,
        k_top: int,
        k_rand: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if power_all.ndim != 2:
            raise ValueError("power_all must have shape [B,K].")
        if periods.ndim != 1:
            raise ValueError("periods must have shape [K].")
        if power_all.shape[1] != periods.numel():
            raise ValueError("power_all.shape[1] must equal periods.numel().")

        B, total_k = power_all.shape
        if k_top + k_rand > total_k:
            raise ValueError(f"Requested {k_top + k_rand} periodogram points, but only {total_k} are available.")

        device = power_all.device
        selected_power = torch.empty((B, k_top + k_rand), device=device, dtype=power_all.dtype)
        selected_periods = torch.empty((B, k_top + k_rand), device=device, dtype=periods.dtype)

        for b in range(B):
            this_power = power_all[b]
            if k_top > 0:
                _, top_idx = torch.topk(this_power, k=k_top, largest=True, sorted=False)
            else:
                top_idx = torch.empty((0,), device=device, dtype=torch.long)

            if k_rand > 0:
                mask_pick = torch.ones(total_k, dtype=torch.bool, device=device)
                mask_pick[top_idx] = False
                rest_idx = torch.nonzero(mask_pick, as_tuple=False).squeeze(1)
                rand_idx = rest_idx[torch.randperm(rest_idx.numel(), device=device)[:k_rand]]
            else:
                rand_idx = torch.empty((0,), device=device, dtype=torch.long)

            keep_idx = torch.cat([top_idx, rand_idx], dim=0)
            keep_idx, _ = torch.sort(keep_idx, descending=False)
            selected_power[b] = this_power[keep_idx]
            selected_periods[b] = periods[keep_idx]

        return selected_power, selected_periods

    def preprocess_raw_light_curve(
        self,
        raw_light_curve: Any,
        *,
        include_periodogram: bool = True,
        include_phase_folded: bool = True,
        include_best_period: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = self._resolve_device(raw_light_curve)
        raw_lc = self._extract_raw_light_curve({"raw_lc": raw_light_curve}, device=device)
        assert raw_lc is not None

        time = raw_lc[..., 0]
        mag = raw_lc[..., 1]
        mag_err = raw_lc[..., 2].abs()
        mask = raw_lc[..., 3] > 0

        finite = torch.isfinite(time) & torch.isfinite(mag) & torch.isfinite(mag_err)
        mask = mask & finite

        time = torch.nan_to_num(time, nan=0.0, posinf=0.0, neginf=0.0)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        mag_err = torch.nan_to_num(mag_err, nan=0.1, posinf=0.1, neginf=0.1)
        bad_err = (~torch.isfinite(mag_err)) | (mag_err <= 0.0) | (mag_err > 20.0)
        mag_err = torch.where(bad_err, torch.full_like(mag_err, 0.1), mag_err)

        sort_key = torch.where(mask, time, torch.full_like(time, float("inf")))
        sort_idx = torch.argsort(sort_key, dim=1)

        time = torch.gather(time, 1, sort_idx)
        mag = torch.gather(mag, 1, sort_idx)
        mag_err = torch.gather(mag_err, 1, sort_idx)
        mask = torch.gather(mask, 1, sort_idx)

        valid_points = mask.sum(dim=1)
        too_short = valid_points < self.inference_min_valid_points
        if bool(too_short.any()):
            bad_ids = torch.nonzero(too_short, as_tuple=False).squeeze(1).detach().cpu().tolist()
            raise ValueError(
                "Raw inference input has too few valid points after sanitization. "
                f"Need at least {self.inference_min_valid_points}; failed samples={bad_ids}."
            )

        mask_f = mask.to(torch.float32)
        time = time * mask_f
        mag = mag * mask_f
        mag_err = mag_err * mask_f

        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_mag = (mag * mask_f).sum(dim=1, keepdim=True) / denom
        mag_mean_sub = (mag - mean_mag) * mask_f

        lc = torch.stack([time, mag_mean_sub, mag_err, mask_f], dim=-1)
        out = {"lc": lc}

        needs_gls = bool(include_periodogram or include_phase_folded or include_best_period)
        if not needs_gls:
            return out

        periods = self._get_inference_period_grid(device)
        power_all, best_power, best_period = self._gls_batch_gpu(
            time,
            mag,
            mag_err.clamp_min(self.inference_eps),
            mask,
            periods,
            eps=self.inference_eps,
            chunk_size=self.inference_chunk_size,
            dtype=torch.float32,
        )

        if include_periodogram:
            selected_power, selected_periods = self._select_periodogram_points_exact(
                power_all,
                periods,
                k_top=self.inference_k_top,
                k_rand=self.inference_k_rand,
            )
            out["periodogram"] = torch.stack(
                [selected_periods, torch.log10(selected_power.clamp_min(self.inference_eps))],
                dim=-1,
            )

        if include_phase_folded:
            phase_period = best_period.view(-1, 1).clamp_min(self.inference_min_period)
            phase_time = (time - phase_period * torch.floor(time / phase_period)) * mask_f
            out["phase_folded_lc"] = torch.stack([phase_time, mag_mean_sub, mag_err, mask_f], dim=-1)

        if include_best_period or include_phase_folded:
            out["best_period"] = best_period
        if include_periodogram:
            out["best_power"] = best_power
        return out

    def prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        xdict = batch.get("X", batch)
        device = self._resolve_device(xdict)
        prepared: Dict[str, Any] = dict(xdict)

        needs_raw = self.enable.get(self.VIEW_RAW, False) and "lc" not in prepared
        needs_periodogram = self.enable.get(self.VIEW_PERIODOGRAM, False) and "periodogram" not in prepared
        needs_phase = self.enable.get(self.VIEW_PHASE_FOLDED, False) and "phase_folded_lc" not in prepared
        needs_best_period = (
            self.enable.get(self.VIEW_PHASE_FOLDED, False)
            and self.phase_use_normalized_phase
            and "best_period" not in prepared
        )

        if needs_raw or needs_periodogram or needs_phase or needs_best_period:
            raw_lc = self._extract_raw_light_curve(prepared, device=device)
            if raw_lc is None:
                missing = []
                if needs_raw:
                    missing.append("lc")
                if needs_periodogram:
                    missing.append("periodogram")
                if needs_phase:
                    missing.append("phase_folded_lc")
                if needs_best_period:
                    missing.append("best_period")
                raise ValueError(
                    "Missing required inputs for model forward. "
                    f"Need precomputed {missing} or raw inputs via `raw_lc` or (`time`, `mag`, `mag_err`, `mask`)."
                )

            derived = self.preprocess_raw_light_curve(
                raw_lc,
                include_periodogram=needs_periodogram,
                include_phase_folded=needs_phase,
                include_best_period=needs_best_period,
            )
            for key, value in derived.items():
                prepared.setdefault(key, value)

        return prepared

    def _encode_prepared_inputs(self, xdict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        device = self._resolve_device(xdict)
        """
        Expected prepared inputs:
          batch["X"]["lc"]:              (B,N,4) [time, mag, mag_err, mask_float]
          batch["X"]["periodogram"]:     (B,K,2) [period, log10_power]
          batch["X"]["phase_folded_lc"]: (B,N,4) [phase_time, mag, mag_err, mask_float]
          batch["X"]["best_period"]:     (B,) float32 (optional but recommended)

        Returns:
          {
            "embeddings":  {view: (B, common_dim)},
            "projections": {view: (B, projection_dim or common_dim)},
          }
        """
        embeddings: Dict[str, torch.Tensor] = {}
        projections: Dict[str, torch.Tensor] = {}

        lc_tensor: Optional[torch.Tensor] = None
        pg_tensor: Optional[torch.Tensor] = None
        pflc_tensor: Optional[torch.Tensor] = None
        best_period_tensor: Optional[torch.Tensor] = None
        best_power_tensor: Optional[torch.Tensor] = None

        if "best_period" in xdict:
            best_period_tensor = xdict["best_period"].to(device, non_blocking=True)
        if "best_power" in xdict:
            best_power_tensor = xdict["best_power"].to(device, non_blocking=True)

        # -------- raw --------
        if self.enable.get(self.VIEW_RAW, False):
            lc_tensor = xdict["lc"].to(device, non_blocking=True)
            vec = self._encode_raw_view(lc_tensor, device=device)
            embeddings[self.VIEW_RAW] = vec
            projections[self.VIEW_RAW] = self.projectors[self.VIEW_RAW](vec) if self.use_projection else vec

        # -------- periodogram --------
        if self.enable.get(self.VIEW_PERIODOGRAM, False):
            pg_tensor = xdict["periodogram"].to(device, non_blocking=True)
            vec = self._encode_periodogram_view(pg_tensor, device=device)
            embeddings[self.VIEW_PERIODOGRAM] = vec
            projections[self.VIEW_PERIODOGRAM] = self.projectors[self.VIEW_PERIODOGRAM](vec) if self.use_projection else vec

        # -------- phase folded --------
        if self.enable.get(self.VIEW_PHASE_FOLDED, False):
            pflc_tensor = xdict["phase_folded_lc"].to(device, non_blocking=True)
            vec = self._encode_phase_view(pflc_tensor, best_period_tensor, device=device)
            embeddings[self.VIEW_PHASE_FOLDED] = vec
            projections[self.VIEW_PHASE_FOLDED] = self.projectors[self.VIEW_PHASE_FOLDED](vec) if self.use_projection else vec

        if self.enable.get(self.VIEW_GROUP, False) and self.group_enabled and self.group_fusion is not None:
            cov_tokens = self._build_covariate_tokens(
                lc=lc_tensor,
                best_period=best_period_tensor,
                best_power=best_power_tensor,
            )
            group_vec = self.group_fusion(embeddings, cov_tokens)
            embeddings[self.VIEW_GROUP] = group_vec
            projections[self.VIEW_GROUP] = self.projectors[self.VIEW_GROUP](group_vec) if self.use_projection else group_vec

        return {"embeddings": embeddings, "projections": projections}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Accepts either:
          - precomputed inputs under batch["X"] with keys `lc`, `periodogram`, `phase_folded_lc`
          - raw inputs via `raw_lc`
          - raw inputs via `time`, `mag`, and optional `mag_err`, `mask`
        """
        prepared = self.prepare_inputs(batch)
        out = self._encode_prepared_inputs(prepared)
        if self.training:
            aux_losses = self._compute_forecast_aux(prepared)
            if aux_losses is not None:
                out["aux_losses"] = {
                    "forecast_total": aux_losses["forecast_total"],
                }
                out["aux_metrics"] = {
                    "forecast_q50_mae": aux_losses["forecast_q50_mae"],
                    "forecast_eligible_frac": aux_losses["forecast_eligible_frac"],
                }
        return out

    def forward_inference(
        self,
        *,
        raw_lc: Optional[Any] = None,
        time: Optional[Any] = None,
        mag: Optional[Any] = None,
        mag_err: Optional[Any] = None,
        mask: Optional[Any] = None,
        return_inputs: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience inference entrypoint for raw light curves.

        Accepted raw formats:
          - `raw_lc`: [N,3], [N,4], [B,N,3], or [B,N,4]
            Columns are (time, mag, mag_err[, mask]).
          - or separate `time`, `mag`, optional `mag_err`, `mask` with shape [N] or [B,N].
        """
        if raw_lc is None:
            if time is None or mag is None:
                raise ValueError("Provide either `raw_lc` or both `time` and `mag`.")
            batch = {"X": {"time": time, "mag": mag}}
            if mag_err is not None:
                batch["X"]["mag_err"] = mag_err
            if mask is not None:
                batch["X"]["mask"] = mask
        else:
            batch = {"X": {"raw_lc": raw_lc}}

        prepared = self.prepare_inputs(batch)
        outputs: Dict[str, Any] = self._encode_prepared_inputs(prepared)
        if return_inputs:
            outputs["inputs"] = prepared
        return outputs
