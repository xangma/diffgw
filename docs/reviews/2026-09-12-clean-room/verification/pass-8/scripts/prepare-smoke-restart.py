#!/usr/bin/env python3
"""Prepare one recorded restart overlay after a successful smoke checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path("/campaign")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute and under /campaign")
    if ".." in path.parts:
        raise argparse.ArgumentTypeError("path must not contain '..'")
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "path must be absolute and under /campaign") from error
    return path


parser = argparse.ArgumentParser()
parser.add_argument("--evolution-dir", type=campaign_path,
                    default=ROOT / "smoke/evolution")
parser.add_argument("--metrics", type=campaign_path,
                    default=ROOT / "provenance/evolution-smoke-metrics.json")
parser.add_argument("--record", type=campaign_path,
                    default=ROOT / "provenance/smoke-restart-preparation.json")
parser.add_argument("--num-procs", type=int, default=8)
parser.add_argument("--input-name", choices=["Inspiral.yaml", "KerrSchild.yaml"],
                    default="Inspiral.yaml")
args = parser.parse_args()
assert args.num_procs > 0

segment = args.evolution_dir / "Segment_0000"
metrics = json.loads(args.metrics.read_text())
assert metrics["exit_code"] == 2, "The first segment must exit through checkpointing"
assert metrics["accepted_exit"] and metrics["watchdog_reason"] is None
checkpoints = sorted((segment / "Checkpoints").glob("Checkpoint_[0-9][0-9][0-9][0-9]"))
assert len(checkpoints) == 1, checkpoints
checkpoint = checkpoints[0]
assert checkpoint.is_dir() and any(checkpoint.iterdir()), checkpoint
checkpoint_files = {str(p.relative_to(checkpoint)): {
    "bytes": p.stat().st_size, "sha256": sha256(p)}
    for p in sorted(checkpoint.rglob("*")) if p.is_file()}
assert {"MainChares.dat", "RO.dat"} <= checkpoint_files.keys()
assert all(value["bytes"] > 0 for value in checkpoint_files.values())
input_path = segment / args.input_name
metadata, options = yaml.safe_load_all(input_path.read_text())
executable = metadata["Executable"]
assert executable in ("EvolveGhBinaryBlackHole", "EvolveGhSingleBlackHole")
assert metrics["command"] == [executable, f"+p{args.num_procs}",
                               "--input-file", str(input_path)]
phases = options["PhaseChangeAndTriggers"]
changed = 0
for entry in phases:
    for phase in entry["PhaseChanges"]:
        if isinstance(phase, dict) and "CheckpointAndExitAfterWallclock" in phase:
            assert phase["CheckpointAndExitAfterWallclock"]["WallclockHours"] == 0.0
            phase["CheckpointAndExitAfterWallclock"]["WallclockHours"] = "None"
            changed += 1
assert changed == 1
overlay = input_path.with_name(input_path.stem + ".overlay_"
                               + checkpoint.name[-4:] + ".yaml")
assert not overlay.exists(), overlay
overlay.write_text(yaml.safe_dump({"PhaseChangeAndTriggers": phases},
                                 explicit_start=True, sort_keys=False))
next_segment = args.evolution_dir / "Segment_0001"
assert not next_segment.exists(), next_segment
next_segment.mkdir()
record = dict(
    checkpoint=str(checkpoint), overlay=str(overlay), cwd=str(next_segment),
    command=[executable, f"+p{args.num_procs}", "+restart", str(checkpoint)],
    original_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
    overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest(),
    checkpoint_files=checkpoint_files,
    restart_launched=False,
)
path = args.record
assert not path.exists(), path
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
