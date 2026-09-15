#!/usr/bin/env python3
"""Prepare a finite Lev1 performance evolution input; do not launch it."""

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
DEFAULT_PERFORMANCE_DIR = ROOT / "performance/Lev1"
DEFAULT_PROVENANCE = ROOT / "provenance"
TARGET_TOLERANCE = 5e-5


def campaign_path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "path must be absolute and must not contain '..'")
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "path must be absolute and under /campaign") from error
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-quality", type=campaign_path,
                        default=DEFAULT_FINAL_QUALITY)
    parser.add_argument("--performance-dir", type=campaign_path,
                        default=DEFAULT_PERFORMANCE_DIR)
    parser.add_argument("--provenance-dir", type=campaign_path,
                        default=DEFAULT_PROVENANCE)
    parser.add_argument("--num-procs", type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.num_procs <= 64

    quality = json.loads(args.final_quality.read_text())
    assert quality.get("target_converged") is True, quality
    assert quality.get("target_tolerance") == TARGET_TOLERANCE, quality
    run_dir = campaign_path(quality["run_dir"])
    id_input = run_dir / "InitialData.yaml"
    assert id_input.is_file(), id_input

    # Re-inspect the actual controlled ID through the pinned helper before
    # allowing it to become the source of the performance evolution.
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    id_quality = importlib.import_module("id_quality")
    inspected = id_quality.inspect_id(run_dir, TARGET_TOLERANCE)
    assert inspected["target_converged"] is True, inspected
    assert inspected["sha256"] == quality["sha256"], "Controlled ID files changed"

    performance_dir = args.performance_dir
    segment = performance_dir / "Segment_0000"
    template_copy = performance_dir / "Inspiral.performance-template.yaml"
    manifest = args.provenance_dir / "performance-evolution-manifest.json"
    check_record = args.provenance_dir / "performance-evolution-check-options.json"
    for path in (segment, template_copy, manifest, check_record):
        assert not path.exists(), f"Refusing to overwrite {path}"
    performance_dir.mkdir(parents=True, exist_ok=False)
    args.provenance_dir.mkdir(parents=True, exist_ok=True)

    inspiral = importlib.import_module("spectre.Pipelines.Bbh.Inspiral")
    original_template = Path(inspiral.INSPIRAL_INPUT_FILE_TEMPLATE)
    original_text = original_template.read_text()
    needle = "Radius: [100, 150, 200]"
    assert original_text.count(needle) == 1
    custom_text = original_text.replace(needle, "Radius: [200, 300, 400]", 1)
    template_copy.write_text(custom_text)

    inspiral.start_inspiral(
        id_input_file_path=id_input,
        id_run_dir=run_dir,
        inspiral_input_file_template=template_copy,
        input_file_name="Inspiral.yaml",
        segments_dir=performance_dir,
        lev=1,
        scheduler=None,
        no_schedule=True,
        submit=False,
        num_procs=args.num_procs,
        continue_with_ringdown=False,
        eccentricity_control=False,
        extra_params={"FinalTime": 100.0, "OuterShellRadius": 1500.0},
    )

    input_path = segment / "Inspiral.yaml"
    assert input_path.is_file(), input_path
    metadata, options = yaml.safe_load_all(input_path.read_text())
    assert metadata["Executable"] == "EvolveGhBinaryBlackHole"
    assert "Next" not in metadata
    assert metadata["Lev"] == 1
    assert options["Evolution"]["InitialTime"] == 0.0
    interpolation = options["InterpolationTargets"]["BondiSachsInterpolation"]
    assert interpolation["Radius"] == [200, 300, 400]
    assert options["DomainCreator"]["BinaryCompactObject"]["OuterShell"]["Radius"] == 1500.0
    assert options["Evolution"]["InitialSlabSize"] > 0
    completions = [entry for entry in options["EventsAndTriggersAtSlabs"]
                   if "Completion" in entry.get("Events", [])]
    assert len(completions) == 1
    assert completions[0]["Trigger"] == {
        "TimeCompares": {"Comparison": "GreaterThanOrEqualTo", "Value": 100.0}}

    check_command = [
        "EvolveGhBinaryBlackHole", "+p1", "--input-file", str(input_path),
        "--check-options",
    ]
    checked = subprocess.run(check_command, cwd=segment, text=True,
                             capture_output=True, check=False, timeout=120)
    check_record.write_text(json.dumps({
        "command": check_command,
        "cwd": str(segment),
        "returncode": checked.returncode,
        "stdout": checked.stdout,
        "stderr": checked.stderr,
        "executed_evolution": False,
    }, indent=2) + "\n")
    assert checked.returncode == 0, checked.stderr

    manifest.write_text(json.dumps({
        "source": {
            "final_quality": str(args.final_quality),
            "final_quality_sha256": sha256(args.final_quality),
            "reinspection": inspected,
            "initial_data": str(id_input),
            "initial_data_sha256": sha256(id_input),
        },
        "pinned_api": "spectre.Pipelines.Bbh.Inspiral.start_inspiral",
        "original_template": str(original_template),
        "original_template_sha256": sha256(original_template),
        "custom_template": str(template_copy),
        "custom_template_sha256": sha256(template_copy),
        "segments_dir": str(performance_dir),
        "rendered_input": str(input_path),
        "rendered_input_sha256": sha256(input_path),
        "check_options_record": str(check_record),
        "configuration": {
            "lev": 1,
            "num_procs": args.num_procs,
            "final_time": 100.0,
            "outer_shell_radius": 1500.0,
            "bondi_sachs_radii": [200, 300, 400],
            "continue_with_ringdown": False,
            "eccentricity_control": False,
            "no_schedule": True,
            "submit": False,
        },
        "evolution_launched": False,
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest),
                      "input": str(input_path),
                      "check_returncode": checked.returncode}, indent=2))


if __name__ == "__main__":
    main()
