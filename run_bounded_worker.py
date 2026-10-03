"""Run one PRC worker with a memory watchdog; manage only the spawned child.

The monitor changes no scientific parameters and never touches another job.
Failed or interrupted workers are not accepted as successful artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import psutil


ALLOWED_WORKERS = {"movement_prepared_stream_verify.py", "movement_memmap_fit.py",
                   "v9b_stream_predict.py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=sorted(ALLOWED_WORKERS), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=1.5)
    parser.add_argument("--max-worker-rss-gib", type=float, default=2.0)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    worker = root / args.worker
    report_path = args.report.resolve()
    if (Path.cwd().resolve() != root or not worker.is_file()
            or not report_path.is_relative_to(root / "artifacts")
            or report_path.exists()
            or not math.isfinite(args.min_free_gib)
            or not math.isfinite(args.max_worker_rss_gib)
            or args.min_free_gib < 1.5 or args.max_worker_rss_gib > 2.0
            or args.max_worker_rss_gib <= 0):
        raise ValueError("Invalid worker, workspace, immutable report or memory limits")
    worker_args = args.worker_args
    if worker_args and worker_args[0] == "--":
        worker_args = worker_args[1:]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(worker)
    start = time.monotonic()
    peak_rss = 0.0
    minimum_free = psutil.virtual_memory().available / 2**30
    if minimum_free < args.min_free_gib:
        raise MemoryError("Insufficient free memory before worker launch")
    child = subprocess.Popen([sys.executable, str(worker), *worker_args])
    reason = None
    monitor_error = None
    try:
        try:
            process = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            process = None
        while child.poll() is None:
            available = psutil.virtual_memory().available / 2**30
            minimum_free = min(minimum_free, available)
            try:
                if process is None:
                    break
                rss = process.memory_info().rss / 2**30
            except psutil.NoSuchProcess:
                break
            peak_rss = max(peak_rss, rss)
            if available < args.min_free_gib or rss > args.max_worker_rss_gib:
                reason = "available_memory_floor" if available < args.min_free_gib else "worker_rss_ceiling"
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            time.sleep(0.5)
    except BaseException as exc:
        reason = "monitor_interrupted" if isinstance(exc, KeyboardInterrupt) else "monitor_error"
        monitor_error = type(exc).__name__
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    exit_code = child.wait()
    unchanged = sha256(worker) == before
    report = {"worker": args.worker, "worker_args": worker_args,
              "worker_source_sha256": before,
              "monitor_source_sha256": sha256(Path(__file__)),
              "child_pid": child.pid, "exit_code": exit_code,
              "abort_reason": reason, "peak_worker_rss_gib": peak_rss,
              "monitor_error_type": monitor_error,
              "minimum_available_gib": minimum_free,
              "minimum_available_limit_gib": args.min_free_gib,
              "maximum_worker_rss_limit_gib": args.max_worker_rss_gib,
              "memory_sample_interval_seconds": 0.5,
              "elapsed_seconds": time.monotonic() - start,
              "worker_source_unchanged": unchanged,
              "success": exit_code == 0 and reason is None and unchanged}
    with report_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        raise SystemExit(exit_code if exit_code else 1)


if __name__ == "__main__":
    main()
