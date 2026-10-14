"""Identity CE + multi-scale InfoNCE + attribute BCE."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """a,b: (B,D) L2-normalized; match diagonal pairs."""
    logits = (a @ b.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def tgfr_loss(
    out: dict,
    labels: torch.Tensor,
    attrs: torch.Tensor,
    *,
    lambda_id: float = 1.0,
    lambda_global: float = 0.5,
    lambda_local: float = 0.5,
    lambda_attr: float = 0.2,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict]:
    id_loss = F.cross_entropy(out["logits"], labels)
    g_loss = info_nce(out["global"], out["text"], temperature)
    l_loss = info_nce(out["local"], out["text"], temperature)
    attr_loss = F.binary_cross_entropy_with_logits(out["attr_logits"], attrs)
    total = (
        lambda_id * id_loss
        + lambda_global * g_loss
        + lambda_local * l_loss
        + lambda_attr * attr_loss
    )
    return total, {
        "total": float(total.detach()),
        "id": float(id_loss.detach()),
        "global": float(g_loss.detach()),
        "local": float(l_loss.detach()),
        "attr": float(attr_loss.detach()),
    }
