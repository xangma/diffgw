#!/usr/bin/env python3
"""Compare completed P9/P11 initial-data records; never launch a solve."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import yaml


ROOT = Path("/campaign")
DEFAULT_MANIFEST = ROOT / "provenance/independent-p11-manifest.json"
DEFAULT_OUTPUT = ROOT / "provenance/independent-p11-comparison.json"
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
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_finite(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, path + (index,))
    elif isinstance(value, float):
        assert math.isfinite(value), path


def load_preparation_helpers():
    path = Path(__file__).with_name("prepare-independent-p11.py")
    spec = importlib.util.spec_from_file_location("prepare_independent_p11", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=campaign_path,
                        default=DEFAULT_MANIFEST)
    parser.add_argument("--p11-run-dir", type=campaign_path)
    parser.add_argument("--output", type=campaign_path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert not args.output.exists(), f"Refusing to overwrite {args.output}"

    manifest = json.loads(args.manifest.read_text())
    source = manifest["source"]
    final_quality = Path(source["final_quality"])
    p9_run_dir = Path(source["controlled_run_dir"])
    p9_input = Path(source["controlled_initial_data"])
    rendered_input = Path(manifest["rendered_initial_data"])
    p11_run_dir = (args.p11_run_dir if args.p11_run_dir is not None
                   else rendered_input.parent)
    p11_input = p11_run_dir / "InitialData.yaml"
    for path in (final_quality, p9_run_dir, p9_input, rendered_input,
                 p11_run_dir, p11_input):
        campaign_path(str(path))
    assert (final_quality.is_file() and p9_input.is_file()
            and rendered_input.is_file() and p11_input.is_file())
    assert p9_run_dir.is_dir() and p11_run_dir.is_dir()

    assert sha256(final_quality) == source["final_quality_sha256"]
    assert sha256(p9_input) == source["controlled_initial_data_sha256"]
    assert sha256(rendered_input) == manifest["rendered_initial_data_sha256"]
    assert sha256(p11_input) == manifest["rendered_initial_data_sha256"]

    scripts = load_preparation_helpers()
    metadata9, options9 = yaml.safe_load_all(p9_input.read_text())
    metadata11, options11 = yaml.safe_load_all(rendered_input.read_text())
    assert metadata9["TargetParams"] == metadata11["TargetParams"]
    scripts.assert_options_match(options9, options11)

    quality = json.loads(final_quality.read_text())
    assert quality["sha256"] == source["p9_reinspection"]["sha256"]

    scripts_dir = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(scripts_dir))
    from id_quality import inspect_id

    p9 = inspect_id(p9_run_dir, TARGET_TOLERANCE)
    p11 = inspect_id(p11_run_dir, TARGET_TOLERANCE)
    assert p9["sha256"] == source["p9_reinspection"]["sha256"]
    require_finite(p9)
    require_finite(p11)

    per_hole = {}
    for name in ("A", "B"):
        p9_hole = p9["holes"][name]
        p11_hole = p11["holes"][name]
        mass_delta = p11_hole["mass"] - p9_hole["mass"]
        spin_delta = [new - old for old, new in zip(
            p9_hole["dimensionless_spin"], p11_hole["dimensionless_spin"])]
        per_hole[name] = {
            "p9_mass": p9_hole["mass"],
            "p11_mass": p11_hole["mass"],
            "mass_delta_p11_minus_p9": mass_delta,
            "p9_dimensionless_spin": p9_hole["dimensionless_spin"],
            "p11_dimensionless_spin": p11_hole["dimensionless_spin"],
            "dimensionless_spin_delta_p11_minus_p9": spin_delta,
        }

    result = {
        "manifest": str(args.manifest),
        "p9_run_dir": str(p9_run_dir),
        "p11_run_dir": str(p11_run_dir),
        "target_tolerance": TARGET_TOLERANCE,
        "p9_target_residuals": p9["target_residuals"],
        "p11_target_residuals": p11["target_residuals"],
        "p9_max_abs_target_residual": p9["max_abs_target_residual"],
        "p11_max_abs_target_residual": p11["max_abs_target_residual"],
        "per_hole": per_hole,
        "nonlinear_residuals": {
            "p9": p9["nonlinear_final"]["Residual"],
            "p11": p11["nonlinear_final"]["Residual"],
        },
        "p9_sha256": p9["sha256"],
        "p11_sha256": p11["sha256"],
        "rendered_input_sha256": sha256(rendered_input),
        "spectral_convergence_claim": False,
        "comparison_passed": (p9["target_converged"] is True
                              and p11["target_converged"] is True),
    }
    require_finite(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["comparison_passed"]:
        raise SystemExit("P9/P11 mass/spin target tolerance was not met")


if __name__ == "__main__":
    main()
