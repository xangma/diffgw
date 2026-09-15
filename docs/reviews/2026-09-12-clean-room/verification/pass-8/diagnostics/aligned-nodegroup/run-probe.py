#!/usr/bin/env python3
"""Compile and check a Charm nodegroup allocation regression probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--expect-misalignment", action="store_true")
args = parser.parse_args()
source = Path("/campaign/diagnostics/aligned-nodegroup")
charm = Path("/work/charm_7_0_0/multicore-linux-x86_64-gcc")
for name in ("aligned-nodegroup.ci", "aligned-nodegroup.cpp"):
    assert not Path(name).exists()
    shutil.copyfile(source / name, name)
commands = [
    [str(charm / "bin/charmc"), "aligned-nodegroup.ci"],
    [str(charm / "bin/charmc"), "-O2", "-language", "charm++", "-o",
     "aligned-nodegroup", "aligned-nodegroup.cpp"],
    ["./aligned-nodegroup", "+p8"],
]
for command in commands[:2]:
    print(json.dumps(command), flush=True)
    subprocess.run(command, check=True, timeout=120)
result = subprocess.run(commands[2], text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=30)
Path("probe-output.txt").write_text(result.stdout)
print(result.stdout, flush=True)
rows = [line for line in result.stdout.splitlines()
        if line.startswith("AlignedNodeGroup this=")]
if args.expect_misalignment:
    accepted = result.returncode != 0 and "Under-aligned nodegroup allocation" in result.stdout
else:
    accepted = result.returncode == 0 and len(rows) == 16 and all("mod64=0 " in row for row in rows)
record = dict(commands=commands, expected_misalignment=args.expect_misalignment,
              exit_code=result.returncode, allocation_rows=rows, accepted=accepted,
              hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [Path("aligned-nodegroup.ci"), Path("aligned-nodegroup.cpp"),
                                   Path("aligned-nodegroup"), charm / "lib/libck.a"]})
Path("probe-result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(0 if accepted else 1)
