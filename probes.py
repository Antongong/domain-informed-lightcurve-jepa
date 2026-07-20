# probes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """
    Simple linear classifier (a.k.a. linear probe).
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _masked_ce_and_acc(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes CE loss and accuracy, ignoring labels < 0.
    """
    assert logits.ndim == 2
    assert y.ndim == 1
    mask = y >= 0
    if mask.sum() == 0:
        loss = torch.tensor(0.0, device=logits.device)
        acc = torch.tensor(0.0, device=logits.device)
        return loss, acc

    logits_m = logits[mask]
    y_m = y[mask]
    loss = F.cross_entropy(logits_m, y_m)

    pred = logits_m.argmax(dim=-1)
    acc = (pred == y_m).float().mean()
    return loss, acc


@dataclass
class ProbeConfig:
    enabled: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    train_on_embeddings: bool = True
    train_on_projections: str = "auto"  # "auto" | "true" | "false"
    # Supervised backbone option: if True, probe loss can be used to train the backbone (no detach).
    supervised: bool = False
    supervised_weight: float = 1.0


class ProbeManager(nn.Module):
    """
    Holds (up to) 4 probes:
      - emb_7, emb_10
      - proj_7, proj_10
    """

    def __init__(self, in_dim_emb: int, in_dim_proj: Optional[int], cfg: ProbeConfig, model_use_projection: bool):
        super().__init__()
        self.cfg = cfg
        self.model_use_projection = bool(model_use_projection)

        self.probes = nn.ModuleDict()

        if cfg.enabled and cfg.train_on_embeddings:
            self.probes["emb_7"] = LinearProbe(in_dim_emb, 7)
            self.probes["emb_10"] = LinearProbe(in_dim_emb, 10)

        train_on_proj = False
        if cfg.enabled:
            if str(cfg.train_on_projections).lower() == "auto":
                train_on_proj = bool(model_use_projection) and (in_dim_proj is not None)
            else:
                train_on_proj = str(cfg.train_on_projections).lower() in ("true", "1", "yes", "y")
        if train_on_proj:
            if in_dim_proj is None:
                raise ValueError("Requested projection probes but in_dim_proj is None.")
            self.probes["proj_7"] = LinearProbe(in_dim_proj, 7)
            self.probes["proj_10"] = LinearProbe(in_dim_proj, 10)

    def build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))

    def has_any(self) -> bool:
        return len(self.probes) > 0

    def forward(self, features: torch.Tensor, head: str) -> torch.Tensor:
        return self.probes[head](features)

    def compute_loss_and_metrics(
        self,
        feat_emb: Optional[torch.Tensor],
        feat_proj: Optional[torch.Tensor],
        y7: torch.Tensor,
        y10: torch.Tensor,
        *,
        prefix: str = "probe",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes probe loss and scalar metrics WITHOUT doing backward() / optimizer.step().
        This can be used in two regimes:
          - linear probe training (features detached, optimize only probes)
          - supervised training (features NOT detached, optimize probes + backbone)
        """
        if not self.has_any():
            return torch.tensor(0.0, device=y7.device), {}

        metrics: Dict[str, float] = {}
        total = torch.tensor(0.0, device=y7.device)

        if feat_emb is not None and "emb_7" in self.probes:
            logits = self.probes["emb_7"](feat_emb)
            loss, acc = _masked_ce_and_acc(logits, y7)
            total = total + loss
            metrics[f"{prefix}/emb_7_loss"] = float(loss.detach().cpu().item())
            metrics[f"{prefix}/emb_7_acc"] = float(acc.detach().cpu().item())

        if feat_emb is not None and "emb_10" in self.probes:
            logits = self.probes["emb_10"](feat_emb)
            loss, acc = _masked_ce_and_acc(logits, y10)
            total = total + loss
            metrics[f"{prefix}/emb_10_loss"] = float(loss.detach().cpu().item())
            metrics[f"{prefix}/emb_10_acc"] = float(acc.detach().cpu().item())

        if feat_proj is not None and "proj_7" in self.probes:
            logits = self.probes["proj_7"](feat_proj)
            loss, acc = _masked_ce_and_acc(logits, y7)
            total = total + loss
            metrics[f"{prefix}/proj_7_loss"] = float(loss.detach().cpu().item())
            metrics[f"{prefix}/proj_7_acc"] = float(acc.detach().cpu().item())

        if feat_proj is not None and "proj_10" in self.probes:
            logits = self.probes["proj_10"](feat_proj)
            loss, acc = _masked_ce_and_acc(logits, y10)
            total = total + loss
            metrics[f"{prefix}/proj_10_loss"] = float(loss.detach().cpu().item())
            metrics[f"{prefix}/proj_10_acc"] = float(acc.detach().cpu().item())

        metrics[f"{prefix}/total"] = float(total.detach().cpu().item())
        return total, metrics

    def train_on_batch(
        self,
        feat_emb: Optional[torch.Tensor],
        feat_proj: Optional[torch.Tensor],
        y7: torch.Tensor,
        y10: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Standard linear-probe training: features MUST be detached from the backbone before calling.
        """
        if not self.has_any():
            return {}

        optimizer.zero_grad(set_to_none=True)

        total, metrics = self.compute_loss_and_metrics(feat_emb, feat_proj, y7, y10, prefix="probe")
        if total.requires_grad:
            total.backward()
            optimizer.step()
        return metrics

    @torch.no_grad()
    def eval_on_batch(
        self,
        feat_emb: Optional[torch.Tensor],
        feat_proj: Optional[torch.Tensor],
        y7: torch.Tensor,
        y10: torch.Tensor,
    ) -> Dict[str, float]:
        if not self.has_any():
            return {}
        _, metrics = self.compute_loss_and_metrics(feat_emb, feat_proj, y7, y10, prefix="probe")
        return metrics
