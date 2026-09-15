#!/usr/bin/env python3
"""Render a finite Lev0 checkpoint smoke input from controlled P9 initial data; never launch evolution."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path("/campaign")
DEFAULT_FINAL_QUALITY = ROOT / "controlled-id/final-quality.json"
DEFAULT_OUTPUT = ROOT / "controlled-p9-lev0-evolution"
DEFAULT_PROVENANCE = ROOT / "provenance"


def campaign_path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("path must be absolute and under /campaign")
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError("path must be under /campaign") from error
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-quality", type=campaign_path,
                        default=DEFAULT_FINAL_QUALITY)
    parser.add_argument("--output-dir", type=campaign_path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-dir", type=campaign_path,
                        default=DEFAULT_PROVENANCE)
    args = parser.parse_args()
    quality = json.loads(args.final_quality.read_text())
    assert quality.get("target_converged") is True, quality
    assert quality["target_tolerance"] == 5e-5
    run_dir = campaign_path(quality["run_dir"])
    id_input = run_dir / "InitialData.yaml"
    assert id_input.is_file(), id_input

    # Refuse stale or unverified controlled data before rendering an evolution.
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    id_quality = importlib.import_module("id_quality")
    inspected = id_quality.inspect_id(run_dir, quality["target_tolerance"])
    assert inspected["target_converged"] is True, inspected
    assert inspected["sha256"] == quality["sha256"], "Controlled ID changed"

    output = args.output_dir
    segment = output / "Segment_0000"
    template = output / "Inspiral.p9-lev0-template.yaml"
    manifest = args.provenance_dir / "controlled-p9-lev0-evolution-manifest.json"
    check_record = args.provenance_dir / "controlled-p9-lev0-evolution-check-options.json"
    for path in (output, segment, template, manifest, check_record):
        assert not path.exists(), f"Refusing to overwrite {path}"
    output.mkdir(parents=True, exist_ok=False)
    args.provenance_dir.mkdir(parents=True, exist_ok=True)

    inspiral = importlib.import_module("spectre.Pipelines.Bbh.Inspiral")
    original_template = Path(inspiral.INSPIRAL_INPUT_FILE_TEMPLATE)
    template.write_text(original_template.read_text())
    inspiral.start_inspiral(
        id_input_file_path=id_input,
        id_run_dir=run_dir,
        inspiral_input_file_template=template,
        input_file_name="Inspiral.yaml",
        segments_dir=output,
        lev=0,
        scheduler=None,
        no_schedule=True,
        submit=False,
        num_procs=16,
        continue_with_ringdown=False,
        eccentricity_control=False,
        extra_params={"FinalTime": 1.0},
    )

    input_path = segment / "Inspiral.yaml"
    assert input_path.is_file(), input_path
    metadata, options = yaml.safe_load_all(input_path.read_text())
    assert metadata["Executable"] == "EvolveGhBinaryBlackHole"
    assert "Next" not in metadata, "Automatic continuation is forbidden"
    assert metadata["Lev"] == 0
    assert options["Evolution"]["InitialTime"] == 0.0
    completion = [entry for entry in options["EventsAndTriggersAtSlabs"]
                  if "Completion" in entry.get("Events", [])]
    assert len(completion) == 1
    assert completion[0]["Trigger"] == {
        "TimeCompares": {"Comparison": "GreaterThanOrEqualTo", "Value": 1.0}}
    checkpoint = [entry for entry in options["PhaseChangeAndTriggers"]
                  if any(isinstance(p, dict) and "CheckpointAndExitAfterWallclock" in p
                         for p in entry["PhaseChanges"])]
    assert len(checkpoint) == 1
    checkpoint[0]["Trigger"] = {
        "TimeCompares": {"Comparison": "GreaterThanOrEqualTo", "Value": 0.5}}
    for phase in checkpoint[0]["PhaseChanges"]:
        if "CheckpointAndExitAfterWallclock" in phase:
            phase["CheckpointAndExitAfterWallclock"]["WallclockHours"] = 0.0
    input_path.write_text(yaml.safe_dump_all([metadata, options],
                                             explicit_start=True, sort_keys=False))

    command = ["EvolveGhBinaryBlackHole", "+p1", "--input-file", str(input_path),
               "--check-options"]
    checked = subprocess.run(command, cwd=segment, text=True, capture_output=True,
                             check=False, timeout=120)
    check_record.write_text(json.dumps({"command": command, "cwd": str(segment),
                                       "returncode": checked.returncode,
                                       "stdout": checked.stdout, "stderr": checked.stderr,
                                       "executed_evolution": False}, indent=2) + "\n")
    assert checked.returncode == 0, checked.stderr
    manifest.write_text(json.dumps({
        "pinned_api": "spectre.Pipelines.Bbh.Inspiral.start_inspiral",
        "source": {"final_quality": str(args.final_quality),
                   "final_quality_sha256": sha256(args.final_quality),
                   "initial_data": str(id_input), "initial_data_sha256": sha256(id_input),
                   "reinspection": inspected},
        "rendered_input": str(input_path),
        "rendered_input_sha256": sha256(input_path),
        "check_options_record": str(check_record),
        "configuration": {"lev": 0, "initial_data_polynomial_order": 9,
                           "evolution_polynomial_order": 7, "num_procs": 16, "final_time": 1.0,
                           "checkpoint_time": 0.5, "continue_with_ringdown": False,
                           "eccentricity_control": False, "no_schedule": True,
                           "submit": False},
        "evolution_launched": False
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "input": str(input_path),
                      "check_returncode": checked.returncode}, indent=2))


if __name__ == "__main__":
    main()
