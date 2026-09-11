"""Regenerate the optional-development Kerr spectral surrogate without LAL/data files.

Install qnm==0.4.4 and scipy in a separate development environment, then run:
    python scripts/generate_kerr_qnm.py --output-dir /tmp/torchwave-qnm

Stein (2019), JOSS 4(42), 1683, https://doi.org/10.21105/joss.01683:
Leaver radial continued fractions and the Cook-Zalutskiy angular eigenproblem.
The n=0 branch is followed continuously from the Schwarzschild l=2 root.
Neither this solver nor its generated validation files are runtime dependencies.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebpts1, chebval
from qnm.nearby import NearbyRootFinder

T_MIN, T_MAX = np.sqrt(0.0001), np.sqrt(1.9999)


def from_x(x):
    return 1 - (T_MIN + (x + 1) * (T_MAX - T_MIN) / 2) ** 2


def to_x(a):
    return 2 * (np.sqrt(1 - a) - T_MIN) / (T_MAX - T_MIN) - 1


def solve(a, seed, angular_seed=4, strict=False):
    finder = NearbyRootFinder(
        a=abs(a), s=-2, m=2 if a >= 0 else -2, A_closest_to=angular_seed,
        l_max=40 if strict else 28, omega_guess=seed, tol=1e-12,
        cf_tol=1e-16 if strict else 2e-15, Nr_min=2000 if strict else 600,
        Nr_max=30000, n_inv=0,
    )
    w = finder.do_solve()
    if w is None:
        # MINPACK can report stagnation after already reaching a tiny residual.
        # Retry its last candidate without relaxing continued-fraction accuracy.
        finder.set_params(omega_guess=complex(*finder.opt_res.x), tol=1e-10)
        w = finder.do_solve()
    if w is None or w.real <= 0 or w.imag >= 0:
        raise RuntimeError(f"No physical root at a={a}: {finder.opt_res}")
    residual = float(np.linalg.norm(finder([w.real, w.imag])))
    if residual > 1e-9:
        raise RuntimeError(f"Radial residual {residual} at a={a}")
    cf_error, cf_iterations = finder.get_cf_err()
    return {
        "a": float(a), "wr": float(w.real), "damping": float(-w.imag),
        "angular_real": float(finder.A.real), "angular_imag": float(finder.A.imag),
        "radial_residual": residual, "cf_error": float(cf_error),
        "cf_iterations": int(cf_iterations),
    }


def sequence(spins):
    # A coarse root guess, not a calibration value; the equations refine it.
    zero = solve(0.0, 0.37 - 0.09j)
    output = {0.0: zero}
    # Intermediate continuation steps prevent switching to an overtone near a=1.
    support = np.unique(np.concatenate((spins, from_x(np.linspace(-1, 1, 401)))))
    for sign in (-1, 1):
        last = zero
        for a in sorted((a for a in support if a * sign > 0), key=abs):
            last = solve(a, last["wr"] - 1j * last["damping"],
                         last["angular_real"] + 1j * last["angular_imag"])
            output[float(a)] = last
    return [output[float(a)] for a in spins]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = chebpts1(128)
    training = sequence(from_x(x))
    y = np.array([[r["wr"], r["damping"] / np.sqrt(1 - r["a"])] for r in training])
    coeff = chebfit(x, y, 95)
    rng = np.random.default_rng(20260911)
    spins = np.unique(np.concatenate((
        from_x(rng.uniform(-1, 1, 1600)), np.linspace(-0.9999, 0.9999, 401),
        [-0.9999, -0.9995, -0.99, -1e-7, 0, 1e-7, 0.9, 0.99,
         0.9975, 0.9976, 0.9977, 0.9995, 0.9998, 0.9999],
    )))
    holdouts = sequence(spins)
    expected = np.array([[r["wr"], r["damping"]] for r in holdouts])
    actual = chebval(to_x(spins), coeff).T
    actual[:, 1] *= np.sqrt(1 - spins)
    errors = np.abs(actual - expected) / expected
    strict = []
    for a in [-0.9999, -0.5, 0, 0.5, 0.99, 0.9976, 0.9999]:
        initial = sequence([a])[0]
        strict.append(solve(a, initial["wr"] - 1j * initial["damping"],
                            initial["angular_real"] + 1j * initial["angular_imag"], True))
    result = {
        "source": "Independent Kerr eigenproblem solves; no waveform oracle or input tables",
        "versions": {p: importlib.metadata.version(p) for p in ("qnm", "numpy", "scipy")},
        "solver": {"l_max": 28, "cf_tol": 2e-15, "Nr_min": 600, "Nr_max": 30000},
        "strict_solver": {"l_max": 40, "cf_tol": 1e-16, "Nr_min": 2000, "Nr_max": 30000},
        "coordinate": "sqrt(1-a)", "t_min": T_MIN, "t_max": T_MAX,
        "degree": 95, "training_count": len(training), "holdout_count": len(holdouts),
        "max_relative_holdout": errors.max(axis=0).tolist(),
        "worst_a": spins[errors.argmax(axis=0)].tolist(), "coefficients": coeff.T.tolist(),
    }
    for name, data in (("training", training), ("holdouts", holdouts), ("strict", strict), ("result", result)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "coefficients"}, indent=2))


if __name__ == "__main__":
    main()
