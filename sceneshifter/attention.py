"""Gather-style scene attention used by Wan-like adapters."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q/k/v`` are ``[B, S, H, D]``."""
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_, k_, v_)
    return out.permute(0, 2, 1, 3)


def gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = q.new_zeros(q.shape)
    if query_index.numel() == 0 or key_index.numel() == 0:
        return out
    q_sel = q.index_select(1, query_index)
    k_sel = k.index_select(1, key_index)
    v_sel = v.index_select(1, key_index)
    out.index_copy_(1, query_index, scaled_dot_product(q_sel, k_sel, v_sel))
    return out
