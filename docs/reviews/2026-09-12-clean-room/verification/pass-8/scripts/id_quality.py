#!/usr/bin/env python3
"""Read achieved initial-data quantities; never infer convergence from exit 0."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml


def table(path, subfile):
    with h5py.File(path, "r") as file:
        dataset = file[subfile]
        legend = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in dataset.attrs["Legend"]]
        data = np.asarray(dataset, dtype=float)
    assert data.ndim == 2 and data.shape[0] > 0
    assert data.shape[1] == len(legend)
    return dict(zip(legend, data[-1])), data, legend


def inspect_id(run_dir, target_tolerance=None):
    run_dir = Path(run_dir)
    input_path = run_dir / "InitialData.yaml"
    metadata, options = yaml.safe_load_all(input_path.read_text())
    assert options["Amr"]["Iterations"] == 1, "Review multi-AMR residual sequences"
    residual, history, columns = table(run_dir / "BbhReductions.h5",
                                        "NewtonRaphsonResiduals.dat")
    for column in ("Walltime", "Iteration", "GlobalizationStep", "Residual"):
        assert np.isfinite(history[:, columns.index(column)]).all(), column
    history = history[np.argsort(history[:, columns.index("Walltime")], kind="stable")]
    starts = np.flatnonzero(
        (history[:, columns.index("Iteration")] == 0)
        & (history[:, columns.index("GlobalizationStep")] == 0))
    assert len(starts) == 1 and starts[0] == 0, "Expected one fresh nonlinear solve"
    residual = dict(zip(columns, history[-1]))
    assert residual["Residual"] >= 0
    assert residual["Residual"] < 1e-8, residual
    target = metadata["TargetParams"]
    holes = {}
    errors = []
    for name in ("A", "B"):
        row, _, _ = table(run_dir / "Horizons.h5", f"Ah{name}.dat")
        mass = row["ChristodoulouMass"]
        spin = np.array([row["DimensionlessSpinVector_" + axis] for axis in "xyz"])
        assert np.isfinite(mass) and np.isfinite(spin).all()
        assert mass > 0 and np.linalg.norm(spin) < 1
        errors.extend([mass - target["Mass" + name]])
        errors.extend(spin - np.asarray(target["DimensionlessSpin" + name]))
        holes[name] = dict(mass=mass, dimensionless_spin=spin.tolist(),
                           spin_norm=float(np.linalg.norm(spin)))
    errors = np.asarray(errors)
    maximum = float(np.max(np.abs(errors)))
    record = dict(
        run_dir=str(run_dir.resolve()),
        nonlinear_final=residual,
        nonlinear_history_rows=len(history),
        nonlinear_iteration_zero_rows=int(np.count_nonzero(
            history[:, columns.index("Iteration")] == 0)),
        holes=holes,
        mass_sum=holes["A"]["mass"] + holes["B"]["mass"],
        mass_ratio_a_over_b=holes["A"]["mass"] / holes["B"]["mass"],
        target_residuals=errors.tolist(),
        max_abs_target_residual=maximum,
        target_tolerance=target_tolerance,
        target_converged=(maximum < target_tolerance
                          if target_tolerance is not None else None),
        sha256={name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                for name in ("InitialData.yaml", "BbhReductions.h5", "Horizons.h5")},
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target-tolerance", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = inspect_id(args.run_dir, args.target_tolerance)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.target_tolerance is not None and not result["target_converged"]:
        raise SystemExit("Mass/spin targets have not converged")


if __name__ == "__main__":
    main()
