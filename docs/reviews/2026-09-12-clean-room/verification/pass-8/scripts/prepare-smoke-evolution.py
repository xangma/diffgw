#!/usr/bin/env python3
"""Render and validate a finite restart smoke input; do not launch evolution."""

import hashlib
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path("/campaign")
SEGMENT = ROOT / "smoke/evolution/Segment_0000"
assert not SEGMENT.exists(), "Never overwrite an existing evolution segment"
command = [
    "spectre", "bbh", "start-inspiral", "/campaign/smoke/id/InitialData.yaml",
    "--lev", "-2", "--segments-dir", "/campaign/smoke/evolution",
    "--no-schedule", "--num-procs", "8", "--param", "FinalTime=1.0", "--no-submit",
]
subprocess.run(command, check=True)
path = SEGMENT / "Inspiral.yaml"
original = path.read_text()
metadata, options = yaml.safe_load_all(original)
assert metadata["Executable"] == "EvolveGhBinaryBlackHole"
assert "Next" not in metadata, "No automatic continuation is allowed in this test"
assert options["Evolution"]["InitialSlabSize"] == 0.1
assert options["Evolution"]["InitialTime"] == 0.0
completions = [entry for entry in options["EventsAndTriggersAtSlabs"]
               if "Completion" in entry.get("Events", [])]
assert len(completions) == 1
assert completions[0]["Trigger"] == {
    "TimeCompares": {"Comparison": "GreaterThanOrEqualTo", "Value": 1.0}}
changed = 0
for entry in options["PhaseChangeAndTriggers"]:
    for phase in entry["PhaseChanges"]:
        if isinstance(phase, dict) and "CheckpointAndExitAfterWallclock" in phase:
            assert phase["CheckpointAndExitAfterWallclock"]["WallclockHours"] == 23.5
            phase["CheckpointAndExitAfterWallclock"]["WallclockHours"] = 0.0
            entry["Trigger"] = {
                "TimeCompares": {"Comparison": "GreaterThanOrEqualTo", "Value": 0.5}}
            changed += 1
assert changed == 1
archive = ROOT / "provenance/smoke-Inspiral-rendered-original.yaml"
assert not archive.exists()
archive.write_text(original)
path.write_text(yaml.safe_dump_all([metadata, options], explicit_start=True, sort_keys=False))
validate = ["EvolveGhBinaryBlackHole", "+p1", "--input-file", str(path), "--check-options"]
subprocess.run(validate, check=True, cwd=SEGMENT)
record = dict(render_command=command, validation_command=validate,
              checkpoint_trigger_time=0.5, final_time_trigger=1.0,
              input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              original_sha256=hashlib.sha256(original.encode()).hexdigest(),
              evolution_launched=False)
(ROOT / "provenance/smoke-evolution-render.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
