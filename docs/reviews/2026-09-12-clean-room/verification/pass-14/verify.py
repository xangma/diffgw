"""Record this pass's ordinary tests and unchanged strict acceptance suites."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONPATH="src:tests", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
           LAL_DATA_PATH="/tmp/torchwave-oracle-data")


def hashes():
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"),
                    *ROOT.glob("scripts/**/*.py"), *ROOT.glob("tests/data/*.json"),
                    ROOT / "docs/model_manifest.json"])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


record = dict(host=platform.node(), cwd=str(ROOT), pid=os.getpid(), started_utc=utc(),
              source_sha256_before=hashes(), runs=[])
child = None


def save():
    (OUT / "verification-run.json").write_text(json.dumps(record, indent=2) + "\n")


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    record.update(stopped_utc=utc(), stopped_by_signal=signum)
    save()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
save()
print(f"verification: driver_pid={os.getpid()}, host={platform.node()}, cwd={ROOT}", flush=True)
for label, args in [
    ("pytest", ["-m", "pytest", "-q"]),
    ("catalog", ["tests/parity_harness.py", "--strict", "--scope=catalog",
                 "--json-out", str(OUT / "catalog.json")]),
    ("holdout", ["tests/parity_harness.py", "--strict", "--scope=holdout",
                 "--json-out", str(OUT / "holdout.json")]),
    ("components-after", [str(OUT / "verify-components.py"),
                    "--json-out", str(OUT / "components-after.json")]),
]:
    command = [sys.executable, *args]
    run = dict(label=label, command=command, log=str(OUT / f"{label}.log"),
               started_utc=utc())
    record["runs"].append(run)
    started = time.monotonic()
    with open(run["log"], "w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        run["pid"] = child.pid
        save()
        print(f"{label}: pid={child.pid}, log={run['log']}", flush=True)
        run["returncode"] = child.wait()
    run.update(seconds=time.monotonic() - started, ended_utc=utc())
    save()
    print(f"{label}: exit={run['returncode']}, seconds={run['seconds']:.2f}", flush=True)
record.update(ended_utc=utc(), source_sha256_after=hashes())
record["source_stable"] = record["source_sha256_before"] == record["source_sha256_after"]
save()
sys.exit(0 if record["source_stable"] and record["runs"][0]["returncode"] == 0 else 1)
