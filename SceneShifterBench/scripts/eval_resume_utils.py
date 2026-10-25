#!/usr/bin/env python3
"""Shared helpers for --ids filtering and --resume merge across evaluation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_ids(value: str) -> Optional[Set[int]]:
    """Parse comma-separated ids/ranges like '1,003,10-12'. Empty -> None (no filter)."""
    value = str(value or "").strip()
    if not value:
        return None

    out: Set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid --ids range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def extract_video_id(value: Any) -> Optional[int]:
    """Extract a numeric video id from fields/paths/keys."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    m = re.search(r"(\d+)", Path(text).stem)
    if not m:
        m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def id_key(vid: int, width: int = 3) -> str:
    return f"{int(vid):0{width}d}"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def ids_from_list(
    items: Sequence[Any],
    *,
    id_fields: Sequence[str] = ("video_id", "id", "video", "video_path"),
) -> Set[int]:
    done: Set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            vid = extract_video_id(item)
            if vid is not None:
                done.add(vid)
            continue
        for key in id_fields:
            if key in item:
                vid = extract_video_id(item.get(key))
                if vid is not None:
                    done.add(vid)
                    break
    return done


def ids_from_dict_keys(mapping: Dict[Any, Any]) -> Set[int]:
    done: Set[int] = set()
    for key in mapping.keys():
        vid = extract_video_id(key)
        if vid is not None:
            done.add(vid)
    return done


def resolve_todo_ids(
    *,
    requested_ids: Optional[Set[int]],
    available_ids: Iterable[int],
    done_ids: Optional[Set[int]] = None,
    resume: bool = False,
) -> Tuple[Set[int], Set[int]]:
    """
    Return (todo_ids, skipped_done_ids).

    - requested_ids=None means all available_ids
    - resume=True skips ids already present in done_ids
    """
    available = set(int(x) for x in available_ids)
    wanted = available if requested_ids is None else (available & set(requested_ids))
    done = set(done_ids or set())
    if resume:
        skipped = wanted & done
        todo = wanted - done
        return todo, skipped
    return wanted, set()


def merge_list_by_video_id(
    existing: Optional[Sequence[Any]],
    new_items: Sequence[Any],
    *,
    id_fields: Sequence[str] = ("video_id", "id", "video", "video_path"),
) -> List[Any]:
    """Merge list entries keyed by video id; new items overwrite same id."""
    by_id: Dict[int, Any] = {}
    order: List[int] = []

    def _put(item: Any) -> None:
        if not isinstance(item, dict):
            return
        vid = None
        for key in id_fields:
            if key in item:
                vid = extract_video_id(item.get(key))
                if vid is not None:
                    break
        if vid is None:
            return
        if vid not in by_id:
            order.append(vid)
        by_id[vid] = item

    for item in existing or []:
        _put(item)
    for item in new_items:
        _put(item)

    return [by_id[vid] for vid in sorted(order)]


def merge_dict_by_video_key(
    existing: Optional[Dict[Any, Any]],
    new_items: Dict[Any, Any],
    *,
    key_width: int = 3,
) -> Dict[str, Any]:
    """Merge dicts keyed by video id string/int; normalize keys to zero-padded strings."""
    out: Dict[str, Any] = {}
    for src in (existing or {}, new_items):
        for key, value in src.items():
            vid = extract_video_id(key)
            if vid is None:
                out[str(key)] = value
                continue
            out[id_key(vid, width=key_width)] = value
    return dict(sorted(out.items(), key=lambda kv: (extract_video_id(kv[0]) is None, extract_video_id(kv[0]) or 0, kv[0])))


def filter_paths_by_ids(paths: Sequence[Path], ids: Optional[Set[int]]) -> List[Path]:
    if ids is None:
        return list(paths)
    out: List[Path] = []
    for p in paths:
        vid = extract_video_id(p)
        if vid is not None and vid in ids:
            out.append(p)
    return out


def add_ids_resume_args(parser, *, ids_default: str = "", resume_help: Optional[str] = None) -> None:
    parser.add_argument(
        "--ids",
        type=str,
        default=ids_default,
        help="Comma-separated video ids/ranges to evaluate, e.g. '1,003,10-12'. Empty = all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=resume_help
        or (
            "Skip video ids already present in the existing output JSON and merge new results into it."
        ),
    )


def last_frame_index_from_frames_dir(frames_dir: Path) -> Optional[int]:
    """Largest 0-based extracted frame index, e.g. 80 for 00.jpg…80.jpg."""
    if not frames_dir.is_dir():
        return None
    last: Optional[int] = None
    for p in frames_dir.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if not p.stem.isdigit():
            continue
        idx = int(p.stem)
        last = idx if last is None else max(last, idx)
    return last


def last_frame_index_from_video_path(video_path: Path) -> Optional[int]:
    """Last 0-based frame index from the container frame count."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        return None
    return n - 1


def filter_last_frame_cuts(
    cut_frames: Iterable[Any],
    last_frame_idx: Optional[int],
) -> List[int]:
    """Drop cuts on/after the last frame. A 1-frame tail is not a real scene change."""
    out: List[int] = []
    for raw in cut_frames:
        try:
            frame = int(raw)
        except (TypeError, ValueError):
            continue
        if frame < 0:
            continue
        if last_frame_idx is not None and frame >= last_frame_idx:
            continue
        out.append(frame)
    return out


def log_resume_plan(*, label: str, todo: Set[int], skipped: Set[int], requested: Optional[Set[int]]) -> None:
    req = "all" if requested is None else f"{len(requested)} requested"
    print(
        f"[{label}] ids={req}; resume_skip={len(skipped)}; will_evaluate={len(todo)}"
        + (f" ({sorted(todo)[:8]}{'...' if len(todo) > 8 else ''})" if todo else "")
    )
