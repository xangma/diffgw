#!/usr/bin/env python3
"""Run parity gates against an immutable, hash-verified source snapshot."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PYTHON = "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_files():
    paths = []
    for base in (ROOT / "src",):
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                paths.append(p)
    paths += [ROOT / "tests" / "parity_harness.py",
              ROOT / "tests" / "kerr_public_oracle.py",
              ROOT / "docs" / "model_manifest.json"]
    init = ROOT / "tests" / "__init__.py"
    if init.exists():
        paths.append(init)
    return sorted(paths)


def hashes(paths, base):
    return {str(p.relative_to(base)): sha256(p) for p in paths}


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = input_files()
    before = hashes(files, ROOT)
    snap = Path(tempfile.mkdtemp(prefix="torchwave-parity-snapshot-", dir="/tmp"))
    copied = []
    for src in files:
        rel = src.relative_to(ROOT)
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    init = snap / "tests" / "__init__.py"
    generated_init = not init.exists()
    if generated_init:
        init.write_bytes(b"")
    snapshot_before = hashes(copied, snap)
    expected = {str(p.relative_to(ROOT)): before[str(p.relative_to(ROOT))] for p in files}
    mismatches = sorted(k for k, v in expected.items() if snapshot_before.get(k) != v)
    if mismatches:
        raise SystemExit("snapshot hash mismatch: " + ", ".join(mismatches))
    env = os.environ.copy()
    env.update({"PYTHONPATH": "src:tests", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "LAL_DATA_PATH": "/private/tmp/torchwave-oracle-data"})
    versions = {}
    for name, cmd in (("python", [PYTHON, "--version"]),
                      ("packages", [PYTHON, "-c", "import torch,numpy,lal,lalsimulation; print('torch='+torch.__version__); print('numpy='+numpy.__version__); print('lal='+getattr(lal,'__version__','unknown')); print('lalsimulation='+getattr(lalsimulation,'__version__','unknown'))"]),
                      ("platform", None)):
        if cmd:
            ver = subprocess.run(cmd, text=True, capture_output=True, check=False)
            versions[name] = ver.stdout.strip() or ver.stderr.strip()
        else:
            versions[name] = platform.platform()
    working_tree_after_copy = hashes(files, ROOT)
    meta = {"started_unix": started, "pid": os.getpid(), "host": socket.gethostname(),
            "cwd": str(ROOT), "snapshot_cwd": str(snap), "snapshot": str(snap),
            "versions": versions, "source_sha_before": before,
            "snapshot_sha_before": snapshot_before, "working_tree_sha_after_copy": working_tree_after_copy,
            "changed_inputs": mismatches, "working_tree_changes": {},
            "generated_tests_init": generated_init, "returncodes": {}, "scopes": {}}
    atomic_json(out / "source-record.json", meta)
    atomic_json(out / "startup.json", {"pid": os.getpid(), "status": "ready", "started_unix": started,
                                        "snapshot_cwd": str(snap)})
    overall = 0
    for scope in ("catalog", "holdout"):
        result_json = (out / f"{scope}.json").resolve()
        stdout_log = out / f"{scope}.stdout.log"
        stderr_log = out / f"{scope}.stderr.log"
        command = [PYTHON, "tests/parity_harness.py", "--strict", f"--scope={scope}",
                   "--json-out", str(result_json)]
        meta["scopes"][scope] = {"command": command, "cwd": str(snap),
                                  "stdout": str(stdout_log), "stderr": str(stderr_log)}
        atomic_json(out / "source-record.json", meta)
        with stdout_log.open("w") as so, stderr_log.open("w") as se:
            proc = subprocess.run(command, cwd=snap, env=env, stdout=so, stderr=se, text=True, check=False)
        meta["returncodes"][scope] = proc.returncode
        overall = overall or proc.returncode
        meta["scopes"][scope]["returncode"] = proc.returncode
        atomic_json(out / "source-record.json", meta)
    source_after = hashes(files, ROOT)
    snapshot_after = hashes(copied, snap)
    changed = sorted(k for k in before if snapshot_after.get(k) != snapshot_before.get(k))
    working_changes = sorted(k for k in before if source_after.get(k) != before[k])
    meta["source_sha_after"] = source_after
    meta["snapshot_sha_after"] = snapshot_after
    meta["changed_inputs"] = changed
    meta["working_tree_changes"] = working_changes
    meta["finished_unix"] = time.time()
    atomic_json(out / "source-record.json", meta)
    atomic_json(out / "startup.json", {"pid": os.getpid(), "status": "finished",
                                        "started_unix": started, "finished_unix": meta["finished_unix"],
                                        "returncodes": meta["returncodes"]})
    raise SystemExit(overall)


if __name__ == "__main__":
    main()
