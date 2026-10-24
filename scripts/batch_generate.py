#!/usr/bin/env python3
"""Batch SceneShifter generation with one process per GPU and file-based resume.

The finished video for every backbone is ``<output_root>/videos/<id>.mp4``.
A rerun skips that file unless ``--overwrite`` is set. ``resume_state.json``
records the status of this invocation; the next run uses the video files.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import queue
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKBONES = (
    "multitalk",
    "infinitetalk",
    "fantasytalking",
    "wan_s2v",
    "hunyuan_avatar",
)


def setup_logging(log_file: str | None, force: bool = False) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=handlers,
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SceneShifter generation")
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--metadata_glob",
        required=True,
        help="Glob under dataset_root, for example ablations/metadata_jsons/*_metadata.json",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, for example 0,1,2")
    parser.add_argument("--ids", default=None, help="Comma-separated sample ids")
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_file", default=None)
    parser.add_argument(
        "--resume_state",
        default=None,
        help="Where to write this run's status json. Defaults to <output_root>/resume_state.json",
    )
    return parser.parse_args()


def sample_id_from_path(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_metadata"):
        stem = stem[: -len("_metadata")]
    return stem.split("_")[0]


def video_path(output_root: Path, sample_id: str) -> Path:
    return output_root / "videos" / f"{sample_id}.mp4"


def load_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_root = Path(args.dataset_root).resolve()
    files = sorted(glob.glob(str(dataset_root / args.metadata_glob)))
    if args.ids:
        wanted = {part.strip() for part in args.ids.split(",") if part.strip()}
        wanted |= {f"{int(part):03d}" for part in wanted if part.isdigit()}
        files = [path for path in files if sample_id_from_path(path) in wanted]
    if args.start_id is not None or args.end_id is not None:
        kept = []
        for path in files:
            sid = sample_id_from_path(path)
            if not sid.isdigit():
                continue
            number = int(sid)
            if args.start_id is not None and number < args.start_id:
                continue
            if args.end_id is not None and number > args.end_id:
                continue
            kept.append(path)
        files = kept
    tasks = []
    seen = set()
    for path in files:
        sid = sample_id_from_path(path)
        if sid in seen:
            raise RuntimeError(f"duplicate sample id {sid} from {path}")
        seen.add(sid)
        tasks.append(
            {
                "id": sid,
                "metadata_path": str(Path(path).resolve()),
            }
        )
    return tasks


def task_config(args: argparse.Namespace, task: Dict[str, Any]) -> Dict[str, Any]:
    from sceneshifter.common import default_unified_weights, load_backbone_config

    cfg = load_backbone_config(args.backbone)
    cfg["sample_id"] = task["id"]
    cfg["dataset_root"] = str(Path(args.dataset_root).resolve())
    cfg["metadata"] = task["metadata_path"]
    unified = dict(cfg.get("unified") or {})
    unified.update(default_unified_weights(args.backbone))
    cfg["unified"] = unified
    return cfg


def _existing_paths(dataset_root: Path, value: Any, label: str, missing: List[str]) -> None:
    if isinstance(value, str) and value:
        path = Path(value)
        resolved = path if path.is_absolute() else dataset_root / path
        if not resolved.is_file():
            missing.append(f"{label}: {resolved}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _existing_paths(dataset_root, item, f"{label}[{key}]", missing)


def dry_run(args: argparse.Namespace, tasks: List[Dict[str, Any]]) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    missing: List[str] = []
    for task in tasks:
        data = json.loads(Path(task["metadata_path"]).read_text())
        routing = data.get("frame_prompt_routing") or {}
        _existing_paths(dataset_root, data.get("cond_image"), f"{task['id']} cond_image", missing)
        _existing_paths(dataset_root, data.get("cond_audio"), f"{task['id']} cond_audio", missing)
        for key in ("background_mask_path", "head_mask_path"):
            _existing_paths(dataset_root, routing.get(key), f"{task['id']} {key}", missing)
    logging.info("Dry run scanned %d tasks for %s.", len(tasks), args.backbone)
    if missing:
        logging.warning("%d missing paths", len(missing))
        for item in missing:
            logging.warning("  %s", item)
        return 1
    logging.info("All referenced paths exist.")
    return 0


def _record_failure(output_root: Path, task_id: str, gpu_id: str, task: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    record = {
        "task_id": task_id,
        "gpu_id": gpu_id,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "metadata_path": task.get("metadata_path"),
    }
    failure_dir = output_root / "failure_logs"
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir / f"{task_id}.json").write_text(json.dumps(record, indent=2))
    return record


def _prepend_python_bin_to_path() -> None:
    """Official scripts call ffmpeg and ffprobe by name. A direct env python
    does not put that env's bin directory on PATH the way conda activate does.
    """
    env_bin = str(Path(sys.executable).resolve().parent)
    parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if env_bin not in parts:
        os.environ["PATH"] = os.pathsep.join([env_bin, *parts])


def worker_main(gpu_id: str, tasks_map, task_queue, status, in_progress, failure_records, stop_event, args) -> None:
    _prepend_python_bin_to_path()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_logging(args.log_file, force=True)
    # Import after the GPU pin. Package import loads torch.
    from sceneshifter.unified import run_unified

    output_root = Path(args.output_root).resolve()
    while not stop_event.is_set():
        try:
            task_id = task_queue.get(timeout=3)
        except queue.Empty:
            continue
        if task_id is None:
            break
        task = tasks_map[task_id]
        status[task_id] = "running"
        in_progress[task_id] = str(gpu_id)
        try:
            logging.info("[GPU %s] starting %s", gpu_id, task_id)
            run_unified(task_config(args, dict(task)), output_root)
            written = video_path(output_root, task_id)
            if not written.is_file():
                raise FileNotFoundError(f"expected video missing: {written}")
            status[task_id] = "done"
            failure_records.pop(task_id, None)
            logging.info("[GPU %s] finished %s -> %s", gpu_id, task_id, written)
        except Exception as exc:  # noqa: BLE001
            status[task_id] = "failed"
            failure_records[task_id] = _record_failure(output_root, task_id, str(gpu_id), dict(task), exc)
            logging.exception("[GPU %s] task %s failed: %s", gpu_id, task_id, exc)
        finally:
            in_progress.pop(task_id, None)
    logging.info("[GPU %s] worker exiting", gpu_id)


def launch_workers(args: argparse.Namespace, tasks: List[Dict[str, Any]]) -> tuple[Dict[str, str], Dict[str, Any]]:
    ctx = get_context("spawn")
    manager = ctx.Manager()
    tasks_map = manager.dict({task["id"]: task for task in tasks})
    status = manager.dict({task["id"]: "pending" for task in tasks})
    in_progress = manager.dict()
    failure_records = manager.dict()
    task_queue = manager.Queue()
    stop_event = manager.Event()
    output_root = Path(args.output_root).resolve()

    for task in tasks:
        destination = video_path(output_root, task["id"])
        if args.skip_existing and destination.is_file() and not args.overwrite:
            status[task["id"]] = "done"
            logging.info("Skipping %s (exists: %s)", task["id"], destination)
        else:
            task_queue.put(task["id"])

    gpu_list = [part.strip() for part in args.gpus.split(",") if part.strip()]
    workers = []
    for gpu_id in gpu_list:
        process = ctx.Process(
            target=worker_main,
            args=(gpu_id, tasks_map, task_queue, status, in_progress, failure_records, stop_event, args),
            daemon=False,
        )
        process.start()
        workers.append((gpu_id, process))

    try:
        while True:
            for gpu_id, process in list(workers):
                if process.exitcode in (None, 0):
                    continue
                logging.error("Worker on GPU %s exited with code %s", gpu_id, process.exitcode)
                process.join()
                for task_id, owner in list(in_progress.items()):
                    if owner == gpu_id:
                        status[task_id] = "pending"
                        task_queue.put(task_id)
                        in_progress.pop(task_id, None)
                workers = [(gid, proc) for gid, proc in workers if proc is not process]
            alive = any(process.is_alive() for _, process in workers)
            if not alive or (task_queue.empty() and len(in_progress) == 0):
                break
            time.sleep(2)
    finally:
        stop_event.set()
        for _ in workers:
            task_queue.put(None)
        for _, process in workers:
            process.join()

    return dict(status), dict(failure_records)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_args.json").write_text(
        json.dumps({key: getattr(args, key) for key in vars(args)}, indent=2, default=str)
    )
    tasks = load_tasks(args)
    if not tasks:
        logging.info("No tasks to run.")
        return 0
    if args.dry_run:
        return dry_run(args, tasks)

    logging.info("Loaded %d tasks for %s.", len(tasks), args.backbone)
    status, failures = launch_workers(args, tasks)
    done = sum(value == "done" for value in status.values())
    failed = sum(value == "failed" for value in status.values())
    pending = len(status) - done - failed
    logging.info("Batch finished. Done: %d, Failed: %d, Pending: %d", done, failed, pending)
    (output_root / "failure_reasons.json").write_text(json.dumps(failures, indent=2))
    resume_path = Path(args.resume_state) if args.resume_state else output_root / "resume_state.json"
    resume_path.write_text(json.dumps(status, indent=2))
    logging.info("Saved resume state to %s", resume_path)
    return 0 if failed == 0 and pending == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
