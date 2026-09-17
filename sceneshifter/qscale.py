"""F9: per-backbone Q-scale magnitude. Does not choose FG vs BG (that is F10)."""

from __future__ import annotations

from typing import Callable, Dict

Formula = Callable[[int], float]


def formula_multitalk(scene_length: int) -> float:
    return 0.8 + (float(scene_length) - 3.0) * 0.2 / 17.0


def formula_infinitetalk(scene_length: int) -> float:
    return formula_multitalk(scene_length)


def formula_fantasytalking(scene_length: int) -> float:
    return 0.8 + (float(scene_length) - 3.0) * 0.2 / 18.0


def formula_wan_s2v(scene_length: int) -> float:
    return 0.9 + (float(scene_length) - 7.0) * 0.1 / 14.0


def formula_wan_fun_camera(scene_length: int) -> float:
    return formula_wan_s2v(scene_length)


def formula_wan_animate(scene_length: int) -> float:
    return 0.92 + (float(scene_length) - 7.0) * 0.08 / 14.0


def formula_hunyuan_avatar(scene_length: int) -> float:
    return 0.9 + (float(scene_length) - 9.0) / 24.0 * 0.1


FORMULAS: Dict[str, Formula] = {
    "multitalk": formula_multitalk,
    "infinitetalk": formula_infinitetalk,
    "fantasytalking": formula_fantasytalking,
    "wan_s2v": formula_wan_s2v,
    "wan_fun_camera": formula_wan_fun_camera,
    "wan_animate": formula_wan_animate,
    "hunyuan_avatar": formula_hunyuan_avatar,
}


def qscale_value(
    scene_length: int,
    backend: str,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)
    try:
        return FORMULAS[backend](int(scene_length))
    except KeyError as exc:
        raise KeyError(f"unknown q-scale backend {backend!r}") from exc
