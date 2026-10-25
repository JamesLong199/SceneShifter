#!/usr/bin/env python3
"""
Extract video frames at 25 fps and save them into individual folders.

Structure: sceneshifter_t10_0101/frames/001/00.jpg, 01.jpg, 02.jpg, etc.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import os

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    extract_video_id,
    filter_paths_by_ids,
    log_resume_plan,
    parse_ids,
)


def _natural_sort_key(p: Path) -> tuple:
    """Sort like 1, 2, 10 (instead of 1, 10, 2) when filenames have integers."""
    parts = []
    for chunk in re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _video_id_from_path(video_path: Path) -> Optional[str]:
    """Extract video ID from filenames like '001.mp4' -> '001'."""
    m = re.match(r"^(\d+)\.mp4$", video_path.name)
    if not m:
        return None
    return m.group(1).zfill(3)  # Ensure 3-digit format with leading zeros


def extract_frames(video_path: Path, output_base_dir: Path, target_fps: float = 25.0) -> bool:
    """
    Extract frames from video at target_fps and save to output directory.
    
    Args:
        video_path: Path to input video file
        output_base_dir: Base directory for output (e.g., sceneshifter_t10_0101)
        target_fps: Target frame rate for extraction
        
    Returns:
        True if successful, False otherwise
    """
    video_id = _video_id_from_path(video_path)
    if video_id is None:
        print(f"Warning: Could not extract video ID from {video_path.name}, skipping...")
        return False
    
    # Create output directory: sceneshifter_t10_0101/frames/001/
    frames_dir = output_base_dir / "frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if original_fps <= 0:
        print(f"Error: Invalid FPS for video {video_path}")
        cap.release()
        return False
    
    # Calculate frame interval (how many frames to skip)
    # If original_fps is 30 and target_fps is 25, we want every 30/25 = 1.2 frames
    # So we'll extract every frame at intervals of original_fps / target_fps
    frame_interval = original_fps / target_fps
    
    print(f"Processing {video_path.name}:")
    print(f"  Original FPS: {original_fps:.2f}, Target FPS: {target_fps:.2f}")
    print(f"  Total frames: {total_frames}, Frame interval: {frame_interval:.3f}")
    
    frame_count = 0
    saved_count = 0
    next_frame_to_save = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if we've reached the next frame to extract
        if frame_count >= next_frame_to_save:
            frame_filename = frames_dir / f"{saved_count:02d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
            next_frame_to_save += frame_interval
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames, saved {saved_count} frames", end='\r')
    
    cap.release()
    print(f"  Completed: saved {saved_count} frames to {frames_dir}")
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames at 25 fps.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Run directory containing videos/. The shell always passes this.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Base output directory (frames will be saved to <output-dir>/frames/<id>/). Defaults to --input-dir.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target frame rate for extraction (default: 25.0).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.mp4",
        help="Glob pattern for video files. Only {id}.mp4 names are accepted.",
    )
    add_ids_resume_args(
        parser,
        resume_help="Skip videos that already have a non-empty frames/<id>/ directory.",
    )

    args = parser.parse_args()
    if not str(args.input_dir).strip():
        raise SystemExit("--input-dir is required")

    input_dir = Path(os.path.join(args.input_dir, "videos")).expanduser().resolve()
    output_dir = Path(args.output_dir if str(args.output_dir).strip() else args.input_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    videos = sorted(input_dir.glob(args.glob), key=_natural_sort_key)
    if not videos:
        print(f"No videos found matching pattern '{args.glob}' in {input_dir}")
        return

    requested = parse_ids(args.ids)
    videos = filter_paths_by_ids(videos, requested)

    skipped = 0
    if args.resume:
        kept = []
        for video_path in videos:
            vid = extract_video_id(video_path)
            frames_dir = output_dir / "frames" / (f"{vid:03d}" if vid is not None else "")
            if vid is not None and frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
                skipped += 1
                continue
            kept.append(video_path)
        videos = kept

    log_resume_plan(
        label="extract_frames",
        todo={extract_video_id(p) for p in videos if extract_video_id(p) is not None},
        skipped=set(),
        requested=requested,
    )
    if args.resume:
        print(f"[extract_frames] resume skipped existing frame dirs: {skipped}")

    print(f"Found {len(videos)} video(s) to process")
    print(f"Output directory: {output_dir / 'frames'}")
    print(f"Target FPS: {args.fps}\n")

    success_count = 0
    for video_path in videos:
        if extract_frames(video_path, output_dir, args.fps):
            success_count += 1
        print()

    print(f"\nCompleted: {success_count}/{len(videos)} videos processed successfully")


if __name__ == "__main__":
    main()
