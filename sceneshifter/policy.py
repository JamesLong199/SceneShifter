"""F4: attention-control schedule matching."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


def normalize_schedule(blocks: Iterable[Dict[str, Any]], num_segments: int) -> List[Dict[str, Any]]:
    from .masks import identity_self_attention_mask, normalize_self_attention_mask

    out = []
    for block in blocks:
        normalized = dict(block)
        raw_mask = block.get("self_attention_mask")
        if raw_mask is None:
            raw_mask = identity_self_attention_mask(num_segments)
        normalized["_mask"] = normalize_self_attention_mask(raw_mask)
        layers = block.get("dit_layer_indices")
        normalized["_layers"] = None if layers is None else {int(v) for v in layers}
        params = dict(block.get("params") or {})
        if "apply_on_uncond" in params and "apply_on_uncond_passes" in params:
            if bool(params["apply_on_uncond"]) != bool(params["apply_on_uncond_passes"]):
                raise ValueError("apply_on_uncond and apply_on_uncond_passes must agree")
        if "apply_on_uncond_passes" not in params and "apply_on_uncond" in params:
            params["apply_on_uncond_passes"] = bool(params["apply_on_uncond"])
        normalized["params"] = params
        lo, hi = map(int, block["timestep_range"])
        normalized["timestep_range"] = [lo, hi]
        out.append(normalized)
    return out


def schedule_layer_overlaps(blocks: List[Dict[str, Any]]) -> List[str]:
    errors = []
    for i, a in enumerate(blocks):
        a_lo, a_hi = a["timestep_range"]
        a_layers: Optional[Set[int]] = a["_layers"]
        for j, b in enumerate(blocks):
            if j <= i:
                continue
            b_lo, b_hi = b["timestep_range"]
            if a_hi < b_lo or b_hi < a_lo:
                continue
            b_layers = b["_layers"]
            if a_layers is None or b_layers is None or (a_layers & b_layers):
                errors.append(
                    f"schedule blocks {i} and {j} overlap on timestep "
                    f"[{max(a_lo, b_lo)}, {min(a_hi, b_hi)}]"
                )
    return errors


def active_block(
    blocks: List[Dict[str, Any]],
    timestep_idx: int,
    block_idx: Optional[int],
    pass_type: str,
) -> Optional[Dict[str, Any]]:
    match = None
    for block in blocks:
        lo, hi = block["timestep_range"]
        if not (lo <= int(timestep_idx) <= hi):
            continue
        layers = block["_layers"]
        if layers is not None and block_idx is not None and int(block_idx) not in layers:
            continue
        apply_uncond = bool(block["params"].get("apply_on_uncond_passes", False))
        if pass_type == "uncond" and not apply_uncond:
            continue
        match = block
    return match


def attention_row(block: Dict[str, Any], segment: int, query_region: Optional[str]):
    row = block["_mask"][int(segment)]
    if isinstance(row, dict):
        if query_region is None:
            raise ValueError("split query row requires fg/bg query routing")
        return row[query_region]
    return row
