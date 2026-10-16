"""RoPE used only to rebuild MultiTalk ref-attn score queries.

Temporary match of the feature model's ``rope_apply``: a new-scene segment is
rotated with temporal indices starting at 0. Self-attention keeps the official
global RoPE.
"""

from __future__ import annotations

import torch


def rope_apply(x, grid_sizes, freqs, frame_index_scale=1):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    frame_index_scale = max(1, int(frame_index_scale))

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        f = int(f)
        h = int(h)
        w = int(w)
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        t_idx = torch.arange(f, dtype=torch.long) * frame_index_scale
        if t_idx.numel() > 0 and int(t_idx[-1]) >= int(freqs[0].shape[0]):
            raise ValueError(
                f"rope_apply temporal index out of bounds: last={int(t_idx[-1])}, "
                f"max={int(freqs[0].shape[0]) - 1}, frame_index_scale={frame_index_scale}, f={f}"
            )
        freqs_t = freqs[0].index_select(0, t_idx.to(freqs[0].device))
        freqs_i = torch.cat(
            [
                freqs_t.view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()
