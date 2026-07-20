# losses.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Signature Regularization (SIGReg), stabilized:
      - caches random projection matrix A
      - refresh_every=0 => keep fixed
      - does math in float32 (recommended for cos/sin stability)
    """
    def __init__(
        self,
        knots: int = 17,
        max_t: float = 3.0,
        proj_dim: int = 256,
        refresh_every: int = 0,
    ):
        super().__init__()
        t = torch.linspace(0.0, max_t, knots, dtype=torch.float32)
        dt = max_t / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)

        self.knots = knots
        self.max_t = max_t
        self.proj_dim = proj_dim
        self.refresh_every = int(refresh_every)
        self._calls = 0

        self.register_buffer("t", t, persistent=False)                 # (T,)
        self.register_buffer("phi", window, persistent=False)          # (T,)
        self.register_buffer("weights", weights * window, persistent=False)  # (T,)

        self._A: Optional[torch.Tensor] = None  # (D, P)

    def _make_A(self, D: int, device: torch.device) -> torch.Tensor:
        A = torch.randn(D, self.proj_dim, device=device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12)
        return A

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: [V, B, D] (on one device)
        """
        self._calls += 1
        V, B, D = proj.shape
        device = proj.device

        proj_f = proj.float()

        need_refresh = (
            self._A is None
            or self._A.device != device
            or self._A.shape[0] != D
            or (self.refresh_every > 0 and (self._calls % self.refresh_every == 0))
        )
        if need_refresh:
            self._A = self._make_A(D, device)

        A = self._A  # (D,P), float32

        x = (proj_f @ A).unsqueeze(-1) * self.t  # (V,B,P,T)
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()  # (V,B,T)
        statistic = (err @ self.weights) * B  # (V,B)
        return statistic.mean()


class LeJEPALoss(nn.Module):
    """
    inv_loss defaults to squared deviation from the view mean.
    If align_to is set, all non-anchor views are aligned to that view.
    total = lamb*sigreg + (1-lamb)*inv
    """
    def __init__(self, lamb: float = 0.02, sigreg: Optional[SIGReg] = None, align_to: Optional[str] = None):
        super().__init__()
        assert 0.0 <= lamb <= 1.0
        self.lamb = float(lamb)
        self.sigreg = sigreg if sigreg is not None else SIGReg()
        self.align_to = str(align_to).strip() if align_to else None

    def forward(self, proj_seq: torch.Tensor, proj_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        proj_f = proj_seq.float()
        if self.align_to and proj_dict is not None and self.align_to in proj_dict:
            anchor = proj_dict[self.align_to].float()
            aligned = [z.float() for name, z in proj_dict.items() if name != self.align_to]
            if aligned:
                inv_loss = torch.stack([(anchor - z).square().mean() for z in aligned]).mean()
            else:
                inv_loss = torch.tensor(0.0, device=anchor.device, dtype=torch.float32)
        else:
            mean = proj_f.mean(0, keepdim=True)  # (1,B,D)
            inv_loss = (mean - proj_f).square().mean()
        sigreg_loss = self.sigreg(proj_f)
        total = self.lamb * sigreg_loss + (1.0 - self.lamb) * inv_loss
        return {"lejepa_total": total, "inv_loss": inv_loss, "sigreg_loss": sigreg_loss}


class CLIPPairLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        assert temperature > 0
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.shape[0]
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        logits = (z_a @ z_b.t()) / self.temperature  # (B,B)
        targets = torch.arange(B, device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        if not self.symmetric:
            return loss_ab
        loss_ba = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_ab + loss_ba)


class MultiViewCLIPLoss(nn.Module):
    def __init__(self, pairs: Sequence[Tuple[str, str]], temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        self.pairs = list(pairs)
        self.pair_loss = CLIPPairLoss(temperature=temperature, symmetric=symmetric)

    def forward(self, proj_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: List[torch.Tensor] = []
        for a, b in self.pairs:
            if a in proj_dict and b in proj_dict:
                losses.append(self.pair_loss(proj_dict[a], proj_dict[b]))
        if len(losses) == 0:
            total = torch.tensor(0.0, device=next(iter(proj_dict.values())).device)
        else:
            total = torch.stack(losses).mean()
        return {"clip_total": total}


class CombinedPretrainLoss(nn.Module):
    def __init__(
        self,
        use_lejepa: bool = True,
        lejepa_weight: float = 1.0,
        lejepa: Optional[LeJEPALoss] = None,
        use_clip: bool = True,
        clip_weight: float = 1.0,
        clip: Optional[MultiViewCLIPLoss] = None,
    ):
        super().__init__()
        self.use_lejepa = bool(use_lejepa)
        self.use_clip = bool(use_clip)
        self.lejepa_weight = float(lejepa_weight)
        self.clip_weight = float(clip_weight)
        self.lejepa = lejepa if lejepa is not None else LeJEPALoss()
        self.clip = clip if clip is not None else MultiViewCLIPLoss(pairs=[("img_ts", "text")])

    def forward(self, proj_seq: torch.Tensor, proj_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=proj_seq.device, dtype=torch.float32)

        if self.use_lejepa:
            lej = self.lejepa(proj_seq, proj_dict)
            out.update(lej)
            total = total + self.lejepa_weight * lej["lejepa_total"]

        if self.use_clip:
            cl = self.clip(proj_dict)
            out.update(cl)
            total = total + self.clip_weight * cl["clip_total"]

        out["total"] = total
        return out
