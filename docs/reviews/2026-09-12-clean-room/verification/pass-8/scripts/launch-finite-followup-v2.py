#!/usr/bin/env python3
"""Record and launch the one-shot, bounded NR prerequisite continuation."""

import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path("/home/xangma/simulations/torchwave-nr/2026-09-12-pilot")
PROVENANCE = ROOT / "provenance"
NAME = "torchwave-nr-finite-followup-v2"
LOG = ROOT / "logs/finite-followup-v2.log"
JOB = PROVENANCE / "finite-followup-v2-job.json"


def main():
    assert socket.gethostname() == "len"
    assert not any(path.exists() for path in
                   (JOB, LOG, PROVENANCE / "finite-followup-v2-state.json"))
    plan = PROVENANCE / "finite-followup-v2-plan.json"
    assert plan.is_file()
    command = ["/usr/bin/python3", "-u", str(ROOT / "scripts/run-finite-followup-v2.py")]
    launch = ["systemd-run", "--user", "--unit=" + NAME,
              "--property=RuntimeMaxSec=48120", "--property=TimeoutStopSec=90",
              "--property=MemoryMax=256M", "--property=CPUQuota=10%",
              "--property=StandardOutput=append:" + str(LOG),
              "--property=StandardError=inherit", "--working-directory=" + str(ROOT),
              *command]
    record = {
        "host": "len", "cwd": str(ROOT), "command": command,
        "systemd_command": launch, "unit": NAME + ".service",
        "requested_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "log_path": str(LOG), "state_path": str(PROVENANCE / "finite-followup-v2-state.json"),
        "expected_next_check": "within one minute; driver polls every 30 seconds",
        "stop_command": "systemctl --user stop " + NAME + ".service",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Accepted controlled ID and prepared inputs, independent P11 comparison, and 1M binary checkpoint/restart; no longer evolution",
    }
    JOB.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(launch, check=True)
    record["initial_systemd_state"] = subprocess.check_output(
        ["systemctl", "--user", "show", NAME + ".service", "-p", "MainPID",
         "-p", "ActiveState", "-p", "SubState"], text=True)
    JOB.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
