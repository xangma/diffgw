#!/usr/bin/env python3
"""Prepare an independent L1/P11 initial-data verification; do not solve it."""

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path("/campaign")
DEFAULT_FINAL_QUALITY = ROOT / "controlled-id/final-quality.json"
DEFAULT_OUTPUT = ROOT / "independent-p11"
DEFAULT_PROVENANCE = ROOT / "provenance"
TARGET_TOLERANCE = 5e-5
CHECK_TIMEOUT_SECONDS = 120


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


def rotation_without_orbital(rotation, orbital_angular_velocity):
    result = [float(value) for value in rotation]
    assert len(result) == 3
    result[2] -= orbital_angular_velocity
    return result


def assert_options_match(source, rendered, path=()):
    """Require identical physics, allowing only the P9-to-P11 grid uplift."""
    if path == ("DomainCreator", "BinaryCompactObject", "InitialGridPoints"):
        assert isinstance(source, dict) and isinstance(rendered, dict)
        assert source.keys() == rendered.keys()
        for key in source:
            assert len(source[key]) == len(rendered[key]), path + (key,)
            assert all(
                new == old + 2
                for old, new in zip(source[key], rendered[key])
            ), path + (key, source[key], rendered[key])
        return
    if isinstance(source, dict):
        assert isinstance(rendered, dict) and source.keys() == rendered.keys(), path
        for key in source:
            assert_options_match(source[key], rendered[key], path + (key,))
        return
    if isinstance(source, list):
        assert isinstance(rendered, list) and len(source) == len(rendered), path
        for index, (old, new) in enumerate(zip(source, rendered)):
            assert_options_match(old, new, path + (index,))
        return
    assert source == rendered, (path, source, rendered)


def text_stream(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-quality", type=campaign_path,
                        default=DEFAULT_FINAL_QUALITY)
    parser.add_argument("--output", type=campaign_path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-dir", type=campaign_path,
                        default=DEFAULT_PROVENANCE)
    parser.add_argument("--num-procs", type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.num_procs <= 16

    quality = json.loads(args.final_quality.read_text())
    assert quality.get("target_converged") is True, quality
    assert quality.get("target_tolerance") == TARGET_TOLERANCE, quality
    run_dir = Path(quality["run_dir"])
    assert run_dir.is_absolute() and ".." not in run_dir.parts
    run_dir.relative_to(ROOT)
    source_input = run_dir / "InitialData.yaml"
    assert source_input.is_file(), source_input
    metadata, source_options = yaml.safe_load_all(source_input.read_text())
    target_params = metadata["TargetParams"]

    binary = source_options["Background"]["Binary"]
    x_coords = binary["XCoords"]
    separation = abs(float(x_coords[1]) - float(x_coords[0]))
    orbital_angular_velocity = float(binary["AngularVelocity"])
    radial_expansion_velocity = float(binary["Expansion"])
    object_left = binary["ObjectLeft"]["KerrSchild"]
    object_right = binary["ObjectRight"]["KerrSchild"]
    domain = source_options["DomainCreator"]["BinaryCompactObject"]
    assert domain["InitialGridPoints"]["ObjectAShell"] == [10, 10, 14]
    assert domain["InitialRefinement"]["ObjectAShell"] == [1, 1, 1]
    center_offset = domain["CenterOfMassOffset"]
    assert len(center_offset) == 2
    for object_name in ("ObjectA", "ObjectB"):
        negative_expansion = domain[object_name]["Interior"][
            "ExciseWithBoundaryCondition"]["ApparentHorizon"]["NegativeExpansion"]
        assert isinstance(negative_expansion, dict)
    rotation_a = domain["ObjectA"]["Interior"]["ExciseWithBoundaryCondition"][
        "ApparentHorizon"]["Rotation"]
    rotation_b = domain["ObjectB"]["Interior"]["ExciseWithBoundaryCondition"][
        "ApparentHorizon"]["Rotation"]
    center_of_mass_offset = [
        float(x_coords[1]) - target_params["MassB"] / (
            target_params["MassA"] + target_params["MassB"]) * separation,
        float(center_offset[0]),
        float(center_offset[1]),
    ]
    linear_velocity = [float(value) for value in binary["LinearVelocity"]]

    output = args.output
    manifest = args.provenance_dir / "independent-p11-manifest.json"
    check_record = args.provenance_dir / "independent-p11-check-options.json"
    for path in (output, manifest, check_record):
        assert not path.exists(), f"Refusing to overwrite {path}"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    id_quality = importlib.import_module("id_quality")
    p9_reinspection = id_quality.inspect_id(run_dir, TARGET_TOLERANCE)
    assert p9_reinspection["target_converged"] is True, p9_reinspection
    assert quality["sha256"] == p9_reinspection["sha256"]
    for name, digest in quality["sha256"].items():
        assert sha256(run_dir / name) == digest

    output.mkdir(parents=True, exist_ok=False)
    args.provenance_dir.mkdir(parents=True, exist_ok=True)

    initial_data = importlib.import_module("spectre.Pipelines.Bbh.InitialData")
    template = Path(initial_data.ID_INPUT_FILE_TEMPLATE)
    initial_data.generate_id(
        target_params=target_params,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        conformal_mass_a=float(object_right["Mass"]),
        conformal_mass_b=float(object_left["Mass"]),
        horizon_rotation_a=rotation_without_orbital(
            rotation_a, orbital_angular_velocity),
        horizon_rotation_b=rotation_without_orbital(
            rotation_b, orbital_angular_velocity),
        center_of_mass_offset=center_of_mass_offset,
        linear_velocity=linear_velocity,
        refinement_level=1,
        polynomial_order=11,
        id_input_file_template=template,
        control=False,
        evolve=False,
        eccentricity_control=False,
        negative_expansion_bc=True,
        run_dir=output,
        scheduler=None,
        no_schedule=True,
        submit=False,
        num_procs=args.num_procs,
        executable="/work/spectre/build/bin/SolveXcts",
    )

    rendered = output / "InitialData.yaml"
    assert rendered.is_file(), rendered
    rendered_metadata, rendered_options = yaml.safe_load_all(rendered.read_text())
    assert rendered_metadata["Executable"] == "SolveXcts"
    next_step = rendered_metadata["Next"]
    assert next_step["Run"] == "spectre.Pipelines.Bbh.PostprocessId:postprocess_id"
    next_with = next_step["With"]
    assert next_with["control"] is False
    assert next_with["evolve"] is False
    assert next_with["eccentricity_control"] is False
    assert next_with["negative_expansion_bc"] is True
    assert next_with["pipeline_dir"] in (None, "None")
    assert_options_match(source_options, rendered_options)

    check_command = [
        "SolveXcts", "+p1", "--input-file", str(rendered), "--check-options",
    ]
    try:
        checked = subprocess.run(
            check_command, cwd=output, text=True, capture_output=True,
            check=False, timeout=CHECK_TIMEOUT_SECONDS)
        check_payload = {
            "command": check_command,
            "cwd": str(output),
            "returncode": checked.returncode,
            "stdout": text_stream(checked.stdout),
            "stderr": text_stream(checked.stderr),
            "timed_out": False,
            "executed_initial_data_solve": False,
        }
    except subprocess.TimeoutExpired as error:
        check_payload = {
            "command": check_command,
            "cwd": str(output),
            "returncode": None,
            "stdout": text_stream(error.stdout),
            "stderr": text_stream(error.stderr),
            "timed_out": True,
            "timeout_seconds": CHECK_TIMEOUT_SECONDS,
            "executed_initial_data_solve": False,
        }
    check_record.write_text(json.dumps(check_payload, indent=2) + "\n")
    assert check_payload["returncode"] == 0, check_payload

    manifest.write_text(json.dumps({
        "source": {
            "final_quality": str(args.final_quality),
            "final_quality_sha256": sha256(args.final_quality),
            "controlled_run_dir": str(run_dir),
            "controlled_initial_data": str(source_input),
            "controlled_initial_data_sha256": sha256(source_input),
            "p9_reinspection": p9_reinspection,
        },
        "pinned_api": "spectre.Pipelines.Bbh.InitialData.generate_id",
        "api_signature": str(inspect.signature(initial_data.generate_id)),
        "template": str(template),
        "template_sha256": sha256(template),
        "rendered_initial_data": str(rendered),
        "rendered_initial_data_sha256": sha256(rendered),
        "check_options_record": str(check_record),
        "configuration": {
            "refinement_level": 1,
            "polynomial_order": 11,
            "num_procs": args.num_procs,
            "control": False,
            "evolve": False,
            "eccentricity_control": False,
            "negative_expansion_bc": True,
            "separation": separation,
            "orbital_angular_velocity": orbital_angular_velocity,
            "radial_expansion_velocity": radial_expansion_velocity,
            "conformal_mass_a": float(object_right["Mass"]),
            "conformal_mass_b": float(object_left["Mass"]),
            "horizon_rotation_a": rotation_without_orbital(
                rotation_a, orbital_angular_velocity),
            "horizon_rotation_b": rotation_without_orbital(
                rotation_b, orbital_angular_velocity),
        },
        "solve_launched": False,
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest),
                      "rendered_input": str(rendered)}, indent=2))


if __name__ == "__main__":
    main()
