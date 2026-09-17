from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sceneshifter.router import FramePromptRouter, create_router


def canonical_mask():
    return [
        [[1, "full"], [0, "full"]],
        [[1, "fg"], [1, "full"]],
    ]


def canonical_config(**overrides):
    cfg = {
        "enabled": True,
        "num_frames": 8,
        "frame_splits": [3],
        "segment_prompts": ["scene 0", "scene 1"],
        "segment_prompt_start_timestep": 0,
        "len_trans": 2,
        "len_trans_context": 1,
        "trans_timestep_range": [2, 5],
        "attention_control_schedule": [
            {
                "timestep_range": [0, 39],
                "dit_layer_indices": None,
                "self_attention_mask": canonical_mask(),
                "params": {
                    "mask_type": "scene0",
                    "auto_q_scale": True,
                    "q_scale_bg_only": False,
                    "apply_on_uncond_passes": False,
                    "q_scale": 0.5,
                },
            }
        ],
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def toy_config():
    return canonical_config()


@pytest.fixture
def toy_router(toy_config):
    return FramePromptRouter(toy_config, backend="multitalk", patch_height=2, patch_width=2)


@pytest.fixture
def bg_mask_path(tmp_path):
    # 2x2 latent: token 0 FG, others BG. File polarity 1=BG, 0=FG.
    mask = np.array([[0, 1], [1, 1]], dtype=np.float32)
    path = tmp_path / "subject0.npy"
    np.save(path, mask)
    return {"0": str(path)}


class TinySelfAttention(torch.nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.o = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, *args, **kwargs):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        return self.o((q * 0 + v).flatten(2))


class TinyDiT(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.self_attn = TinySelfAttention(dim)
        self.cross_attn = TinySelfAttention(dim)

    def forward(self, x, **kwargs):
        return self.self_attn(x, **kwargs)
