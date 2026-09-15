#!/usr/bin/env python3
"""Launch one recorded stage on len; never overwrite an existing stage."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess


ROOT = Path("/home/xangma/simulations/torchwave-nr/2026-09-12-pilot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--image", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-gib", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=6900)
    parser.add_argument("--expect-exit", type=int, default=0)
    parser.add_argument("--command-json", required=True)
    args = parser.parse_args()
    assert socket.gethostname() == "len"
    assert re.fullmatch(r"[a-z0-9-]+", args.stage)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
    assert args.workdir.startswith("/campaign/")
    assert args.output_dir.startswith("/campaign/")
    assert ".." not in Path(args.workdir).parts
    assert ".." not in Path(args.output_dir).parts
    command = json.loads(args.command_json)
    assert command and all(isinstance(arg, str) for arg in command)
    name = "torchwave-nr-" + args.stage
    log = ROOT / "logs" / (args.stage + ".log")
    script = ROOT / (args.stage + ".sh")
    job_file = ROOT / "provenance" / (args.stage + "-job.json")
    metrics_file = ROOT / "provenance" / (args.stage + "-metrics.json")
    assert not any(path.exists() for path in (log, script, job_file, metrics_file))
    subprocess.run(["docker", "image", "inspect", args.image], check=True,
                   stdout=subprocess.DEVNULL)
    (ROOT / args.workdir.removeprefix("/campaign/")).mkdir(parents=True, exist_ok=True)
    docker_command = [
        "/usr/bin/docker", "run", "--name", name,
        "--cpus", str(args.cpus),
        "--memory", f"{args.memory_gib}g",
        "--memory-swap", f"{args.memory_gib}g",
        "--pids-limit", "512", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--user", "1000:1000",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "--mount", f"type=bind,src={ROOT},dst=/campaign",
        "--env", "HOME=/tmp", "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--workdir", args.workdir, "--entrypoint", "python", args.image,
        "-u", "/campaign/scripts/measure-stage.py",
        "--metrics", f"/campaign/provenance/{args.stage}-metrics.json",
        "--output-dir", args.output_dir,
        "--timeout-seconds", str(args.timeout_seconds),
        "--expect-exit", str(args.expect_exit), "--", *command,
    ]
    script.write_text("#!/bin/bash\nset -euo pipefail\nexec "
                      + shlex.join(docker_command) + " > "
                      + shlex.quote(str(log)) + " 2>&1\n")
    unit_command = [
        "systemd-run", "--user", "--unit=" + name,
        "--property=RuntimeMaxSec=" + str(args.timeout_seconds + 60),
        "--property=ExecStop=/usr/bin/docker stop -t 30 " + name,
        "--working-directory=" + str(ROOT), "/bin/bash", str(script),
    ]
    record = dict(
        host="len", cwd=str(ROOT), container_cwd=args.workdir,
        requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        unit=name + ".service", container=name, image_id=args.image,
        cpus=args.cpus, memory_gib=args.memory_gib,
        command=command, docker_command=docker_command,
        systemd_command=unit_command, log_path=str(log),
        stop_command="systemctl --user stop " + name + ".service",
        expected_next_check="within one minute after launch",
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurement_sha256=hashlib.sha256(
            (ROOT / "scripts/measure-stage.py").read_bytes()).hexdigest(),
    )
    job_file.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(unit_command, check=True)
    record["initial_systemd_state"] = subprocess.check_output(
        ["systemctl", "--user", "show", name + ".service",
         "-p", "MainPID", "-p", "ActiveState", "-p", "SubState"], text=True)
    job_file.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
