"""Token layouts. Shared kernel sees noisy/video frames; adapters add extras."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoisyLayout:
    noisy_frames: int
    spatial_tokens: int

    @property
    def noisy_tokens(self) -> int:
        return self.noisy_frames * self.spatial_tokens

    @property
    def noisy_slice(self) -> slice:
        return slice(0, self.noisy_tokens)


@dataclass(frozen=True)
class S2VTokenLayout:
    """``[noisy | reference | FramePack]``. FramePack is never scene-assigned."""

    noisy_frames: int
    spatial_tokens: int
    reference_frames: int = 1
    framepack_tokens: int = 0

    def __post_init__(self) -> None:
        if self.noisy_frames < 1 or self.spatial_tokens < 1:
            raise ValueError("noisy_frames and spatial_tokens must be positive")
        if self.reference_frames != 1:
            raise ValueError("SceneShifter currently requires exactly one reference frame")
        if self.framepack_tokens < 0:
            raise ValueError("framepack_tokens must be >= 0")

    @property
    def noisy_tokens(self) -> int:
        return self.noisy_frames * self.spatial_tokens

    @property
    def noisy_slice(self) -> slice:
        return slice(0, self.noisy_tokens)

    @property
    def reference_tokens(self) -> int:
        return self.reference_frames * self.spatial_tokens

    @property
    def reference_slice(self) -> slice:
        return slice(self.noisy_tokens, self.noisy_tokens + self.reference_tokens)

    @property
    def framepack_slice(self) -> slice:
        start = self.noisy_tokens + self.reference_tokens
        return slice(start, start + self.framepack_tokens)

    @property
    def total_tokens(self) -> int:
        return self.framepack_slice.stop
