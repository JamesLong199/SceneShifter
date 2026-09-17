#!/usr/bin/env python3
"""Batch wrapper: one official backbone process per env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sceneshifter.data import is_sceneshifter_metadata, load_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--metadata_glob", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    paths = sorted(Path().glob(args.metadata_glob))
    print(f"found {len(paths)} metadata files for {args.backbone}")
    for path in paths:
        data = load_metadata(str(path))
        print(path, "sceneshifter" if is_sceneshifter_metadata(data) else "single_prompt")
        if args.dry_run:
            print(json.dumps({"id": path.stem, "keys": list(data)}, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
