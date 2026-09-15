#!/usr/bin/env python3
"""Print the completed container build's public provenance as JSON."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return dict(command=command, exit_code=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


parser = argparse.ArgumentParser()
parser.add_argument("--binaries", nargs="+", default=[
    "SolveXcts", "EvolveGhBinaryBlackHole", "EvolveGhSingleBlackHole", "CharacteristicExtract"])
args = parser.parse_args()
root = Path("/work/spectre")
commands = {
    "source_commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
    "source_status": ["git", "-C", str(root), "status", "--short"],
    "submodules": ["git", "-C", str(root), "submodule", "status", "--recursive"],
    "python_packages": [sys.executable, "-m", "pip", "freeze", "--all"],
    "system_packages": ["dpkg-query", "-W"],
    "compiler": ["clang++", "--version"],
    "fortran": ["gfortran", "--version"],
    "cmake": ["cmake", "--version"],
    "cli_version": ["spectre", "--version"],
    "binding_import": [sys.executable, "-c", "import spectre.IO.H5, spectre.Domain, spectre.SphericalHarmonics; print('binding imports passed')"],
}
report = {key: run(command) for key, command in commands.items()}
report["cmake_cache"] = (root / "build/CMakeCache.txt").read_text()
report["spectre_license"] = (root / "LICENSE.txt").read_text()
report["binaries"] = {}
for name in args.binaries:
    path = root / "build/bin" / name
    assert path.is_file(), path
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    report["binaries"][name] = dict(path=str(path), bytes=path.stat().st_size,
                                    sha256=digest.hexdigest(), linked_libraries=run(["ldd", str(path)]))
print(json.dumps(report, indent=2))
assert all(record["exit_code"] == 0 for key, record in report.items() if key in commands)
