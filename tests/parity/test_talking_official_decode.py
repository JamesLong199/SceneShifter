"""Official MultiTalk/InfiniteTalk decode hooks used for SceneShifter parity."""

from __future__ import annotations

import types

import torch

from sceneshifter.parity.unified import (
    OFFICIAL_IT_FIRST_CLIP_PIN,
    disable_talking_color_correction,
    install_infinitetalk_first_clip_pin_disable,
)


def test_official_color_correction_is_disabled():
    def original(source_chunk, reference, strength):
        return source_chunk + 7

    utils = types.SimpleNamespace(match_and_blend_colors=original)
    talking = types.SimpleNamespace(match_and_blend_colors=original)
    assert disable_talking_color_correction(utils, talking) == 2
    assert disable_talking_color_correction(utils, talking) == 0
    source = torch.tensor([1.0, 2.0])
    assert torch.equal(utils.match_and_blend_colors(source, source, 1.0), source)
    assert torch.equal(talking.match_and_blend_colors(source, source, 1.0), source)


class _OfficialLikeInfiniteTalk:
    def generate_infinitetalk(self, latent, latent_motion_frames, cur_motion_frames_latent_num):
        for _ in range(2):
            latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
            latent = latent + 1
            latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
        return latent


def test_infinitetalk_first_clip_latent_pin_is_removed():
    import inspect

    raw = inspect.getsource(_OfficialLikeInfiniteTalk.generate_infinitetalk)
    assert raw.count(OFFICIAL_IT_FIRST_CLIP_PIN) == 2
    count = install_infinitetalk_first_clip_pin_disable(_OfficialLikeInfiniteTalk)
    assert count == 2
    assert install_infinitetalk_first_clip_pin_disable(_OfficialLikeInfiniteTalk) == 2

    latent = torch.zeros(2, 3, 2, 2)
    motion = torch.full((2, 1, 2, 2), 9.0)
    out = _OfficialLikeInfiniteTalk().generate_infinitetalk(latent, motion, 1)
    assert torch.equal(out, torch.full((2, 3, 2, 2), 2.0))
