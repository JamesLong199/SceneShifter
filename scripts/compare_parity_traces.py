#!/usr/bin/env python3
"""Report the first unequal tensor fingerprint in two parity traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_trace(path: str) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def identity(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": row.get("event"),
        "step": row.get("step"),
        "layer": row.get("layer"),
        "metadata": row.get("metadata"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("unified")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()
    reference = load_trace(args.reference)
    unified = load_trace(args.unified)
    result: Dict[str, Any] = {
        "reference_events": len(reference),
        "unified_events": len(unified),
        "pass": False,
        "first_difference": None,
    }
    common = min(len(reference), len(unified))
    for index in range(common):
        ref, uni = reference[index], unified[index]
        if identity(ref) != identity(uni) or ref.get("sha256") != uni.get("sha256"):
            result["first_difference"] = {
                "index": index,
                "reference": ref,
                "unified": uni,
            }
            break
    else:
        if len(reference) == len(unified):
            result["pass"] = True
        else:
            result["first_difference"] = {
                "index": common,
                "reference": reference[common] if common < len(reference) else None,
                "unified": unified[common] if common < len(unified) else None,
            }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
