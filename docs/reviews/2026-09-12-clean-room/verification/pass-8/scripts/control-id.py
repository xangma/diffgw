#!/usr/bin/env python3
"""Run a finite mass/spin control pilot using unmodified pinned SpECTRE APIs."""

import argparse
import functools
import importlib
import json
import logging
from pathlib import Path
import subprocess

from id_quality import inspect_id


ROOT = Path("/campaign/controlled-id")
SEED = ROOT / "ControlParams_000"
TOLERANCE = 5e-5
MAX_CORRECTIONS = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-procs", type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.num_procs <= 64
    logging.basicConfig(level=logging.INFO)
    assert not SEED.exists(), "Use a new campaign path; never overwrite a seed"
    ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        "spectre", "bbh", "generate-id", "-q", "1",
        "--chi-A", "0", "0", "0", "--chi-B", "0", "0", "0",
        "-D", "15.366", "-w", "0.01533628349983242",
        "-a", "-4.827331550791904e-05", "-e", "0", "--no-control",
        "-L", "1", "-P", "9", "--run-dir", str(SEED),
        "--no-schedule", "--num-procs", str(args.num_procs), "--submit",
    ]
    (ROOT / "seed-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    first = inspect_id(SEED, TOLERANCE)
    (ROOT / "seed-quality.json").write_text(json.dumps(first, indent=2) + "\n")
    final_dir = SEED
    if not first["target_converged"]:
        control = importlib.import_module("spectre.Pipelines.Bbh.ControlId")
        # Upstream control_id omits resource kwargs on its inner generate_id
        # calls. This local adapter bounds the processor count for every solve.
        bounded_generate_id = functools.partial(
            control.generate_id, num_procs=args.num_procs,
            executable="/work/spectre/build/bin/SolveXcts",
            no_schedule=True, submit=True, eccentricity_control=False,
        )
        def checked_generate_id(*args, **kwargs):
            result = bounded_generate_id(*args, **kwargs)
            quality = inspect_id(kwargs["run_dir"], TOLERANCE)
            (Path(kwargs["run_dir"]) / "quality.json").write_text(
                json.dumps(quality, indent=2) + "\n")
            return result

        control.generate_id = checked_generate_id
        final_dir = Path(control.control_id(
            SEED / "InitialData.yaml", id_run_dir=SEED,
            control_params=["MassA", "MassB", "DimensionlessSpinA", "DimensionlessSpinB"],
            residual_tolerance=TOLERANCE, max_iterations=MAX_CORRECTIONS,
            control_delay=2, refinement_level=1, polynomial_order=9,
            negative_expansion_bc=True,
        ))
    final = inspect_id(final_dir, TOLERANCE)
    final.update(max_correction_solves=MAX_CORRECTIONS, num_procs_each_solve=args.num_procs)
    (ROOT / "final-quality.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)
    if not final["target_converged"]:
        raise SystemExit("Control budget exhausted without mass/spin convergence")


if __name__ == "__main__":
    main()
