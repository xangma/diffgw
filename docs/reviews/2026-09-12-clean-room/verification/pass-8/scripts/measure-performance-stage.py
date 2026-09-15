#!/usr/bin/env python3
"""Run one finite NR stage inside its resource-limited container.

This is operational instrumentation, not waveform-generation code. The caller
chooses the exact command and expected exit status; checkpoint exit 2 must be
requested explicitly. A separate systemd limit remains the outer watchdog.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time


def disk_bytes(path):
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file() and not p.is_symlink()
    )


def cgroup_value(name):
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--expect-exit", action="append", type=int, required=True)
    parser.add_argument("--max-output-gib", default=32.0, type=float)
    parser.add_argument("--min-free-gib", default=50.0, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.metrics.exists():
        parser.error("A command and a previously unused metrics path are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    before_bytes = disk_bytes(args.output_dir)
    record = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "cwd": str(Path.cwd()),
        "expected_exit_codes": args.expect_exit,
        "timeout_seconds": args.timeout_seconds,
        "output_bytes_before": before_bytes,
        "cpu_stat_before": cgroup_value("cpu.stat"),
    }
    print(json.dumps(record), flush=True)
    child = subprocess.Popen(command, start_new_session=True)
    record["container_child_pid"] = child.pid
    reason = None
    while child.poll() is None:
        elapsed = time.monotonic() - start
        if elapsed >= args.timeout_seconds:
            reason = "wall_time_limit"
        elif shutil.disk_usage(args.output_dir).free < args.min_free_gib * 2**30:
            reason = "minimum_free_space"
        elif disk_bytes(args.output_dir) > args.max_output_gib * 2**30:
            reason = "output_size_limit"
        if reason:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        try:
            child.wait(timeout=min(10, max(0.01, args.timeout_seconds - elapsed)))
        except subprocess.TimeoutExpired:
            pass
    record.update(
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - start,
        exit_code=child.returncode,
        watchdog_reason=reason,
        accepted_exit=(reason is None and child.returncode in args.expect_exit),
        child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        cgroup_memory_peak_bytes=cgroup_value("memory.peak"),
        cgroup_memory_events=cgroup_value("memory.events"),
        cpu_stat_after=cgroup_value("cpu.stat"),
        output_bytes_after=disk_bytes(args.output_dir),
        filesystem_free_bytes_after=shutil.disk_usage(args.output_dir).free,
    )
    args.metrics.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return 0 if record["accepted_exit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
