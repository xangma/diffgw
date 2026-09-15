#!/usr/bin/env python3
"""Bounded, clean-room TEOB QNM rational calibration experiment.

Targets are generated from TorchWave's independent Kerr eigenproblem data;
the published Eq. (91) numbers are reported only as an after-fit validation.
No LAL code, output, or QNM table is read by this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch
import scipy

from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from kerr_leaver_solver import solve

ROOT = Path(__file__).resolve().parents[1]
SPIN_MAX = 0.95
# (y0,b1,b2,b3,c1,c2,c3), copied from rounded arXiv:1806.01772v2 Eq. (91)
PUBLISHED = {
    "omega220": (0.373672, -1.74085, 0.808214, -0.0598838, -2.07641, 1.31524, -0.235896),
    "alpha220": (0.0889623, -1.82261, 0.701584, 0.121126, -1.80020, 0.720117, 0.0811633),
    "alpha21": (0.184953, -1.41681, -0.0593166, 0.476420, -1.35955, -0.0763529, 0.438558),
}


def rational(spin, coeffs):
    """Evaluate y0*(1+b1*a+b2*a²+b3*a³)/(1+c1*a+c2*a²+c3*a³)."""
    a = np.asarray(spin, dtype=float)
    y0, b1, b2, b3, c1, c2, c3 = np.asarray(coeffs, dtype=float)
    return y0 * (1 + b1*a + b2*a*a + b3*a**3) / (1 + c1*a + c2*a*a + c3*a**3)


def _targets(spins):
    a = torch.as_tensor(spins, dtype=torch.float64)
    wr, al0 = dimensionless_qnm(2, 2, 0, a)
    _, al1 = dimensionless_qnm(2, 2, 1, a)
    return {"omega220": wr.numpy(), "alpha220": al0.numpy(), "alpha21": (al1-al0).numpy()}


def fit_rational(spins, values):
    """Fit the fixed cubic-over-cubic basis, with y0 fixed by the spin-zero datum."""
    x = np.asarray(spins, float); y = np.asarray(values, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 8 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("spins and targets must be finite one-dimensional arrays of equal length")
    if abs(x[0]) > 1e-14 or np.any(np.diff(x) <= 0) or np.any(x < 0) or np.any(x > SPIN_MAX) or np.any(y <= 0):
        raise ValueError("spins must be strictly increasing in [0,.95] and targets positive with first spin zero")
    y0 = float(y[0])
    def unpack(z): return np.r_[y0, z]
    def residual(z): return rational(x, unpack(z)) - y
    result = least_squares(residual, np.zeros(6), method="trf", max_nfev=20000,
                           xtol=1e-14, ftol=1e-14, gtol=1e-14)
    coeffs = unpack(result.x)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("non-finite rational coefficients")
    if not result.success:
        raise RuntimeError(f"rational optimizer failed: {result.message}")
    return coeffs, result


def denominator_poles(coeffs):
    """Return real denominator roots in the experiment's closed domain."""
    _, _, _, _, c1, c2, c3 = coeffs
    roots = np.roots([c3, c2, c1, 1.0]) if c3 != 0.0 else np.roots([c2, c1, 1.0])
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9)


def validate_coefficients(coeffs, domain=(0.0, SPIN_MAX)):
    """Reject poles, non-finite coefficients, or nonpositive damping fit."""
    c = np.asarray(coeffs, float)
    if c.shape != (7,) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be seven finite values")
    lo, hi = domain
    if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi > SPIN_MAX or lo >= hi:
        raise ValueError("domain must be finite and ordered within [0,.95]")
    def roots_and_extrema(p):
        roots = np.roots(p[::-1])
        derivative = np.array([i*p[i] for i in range(1, len(p))], float)
        critical = np.roots(derivative[::-1]) if np.any(derivative[1:]) else np.array([])
        return roots, critical
    den_poly = np.array([1., c[4], c[5], c[6]])
    num_poly = np.array([1., c[1], c[2], c[3]])
    droots, dcrit = roots_and_extrema(den_poly)
    nroots, ncrit = roots_and_extrema(num_poly)
    real_in = lambda z: any(abs(z0.imag) < 1e-9 and lo <= z0.real <= hi for z0 in z)
    if real_in(droots):
        raise ValueError("rational denominator has a pole or near-pole in domain")
    if real_in(nroots):
        raise ValueError("rational numerator changes sign in domain")
    probes = np.r_[lo, hi, [z.real for z in np.r_[dcrit, ncrit] if abs(z.imag) < 1e-9 and lo <= z.real <= hi]]
    den = np.polyval(den_poly[::-1], probes)
    num = np.polyval(num_poly[::-1], probes) * c[0]
    if np.min(abs(den)) < 1e-6 or np.any(den <= 0):
        raise ValueError("rational denominator has a pole or near-pole in domain")
    if np.any(num <= 0):
        raise ValueError("damping/frequency fit must remain positive in domain")
    return True


def run(points=49, holdout_spins=None):
    sources = [Path(__file__), ROOT/"scripts/kerr_leaver_solver.py", ROOT/"src/torchwave/approximants/kerr_qnm_generated.py"]
    source_hashes_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if points < 8:
        raise ValueError("points must be >= 8")
    # Explicit Chebyshev-like bounded grid, frozen before any validation.
    train = SPIN_MAX * (1 - np.cos(np.linspace(0, np.pi, points))) / 2
    holdout = np.asarray(holdout_spins if holdout_spins is not None else
                         [0.037, 0.173, 0.311, 0.527, 0.713, 0.887, 0.943], float)
    if holdout.ndim != 1 or len(holdout) == 0 or not np.all(np.isfinite(holdout)) or np.any((holdout < 0) | (holdout > SPIN_MAX)) or np.any(np.isclose(holdout[:, None], train[None, :], atol=1e-14)):
        raise ValueError("holdouts must lie in [0, .95]")
    data = _targets(train)
    fits = {}
    for name, values in data.items():
        coeffs, opt = fit_rational(train, values)
        validate_coefficients(coeffs)
        pred = rational(train, coeffs)
        hv = rational(holdout, coeffs)
        truth = _targets(holdout)[name]
        pub = rational(holdout, PUBLISHED[name])
        den = 1 + coeffs[4]*train + coeffs[5]*train**2 + coeffs[6]*train**3
        fits[name] = {"coefficients": coeffs.tolist(), "train_rmse": float(np.sqrt(np.mean((pred-values)**2))),
                      "train_max_abs": float(np.max(abs(pred-values))), "holdout_rmse": float(np.sqrt(np.mean((hv-truth)**2))),
                      "holdout_max_abs": float(np.max(abs(hv-truth))), "published_holdout_rmse": float(np.sqrt(np.mean((pub-truth)**2))),
                      "min_abs_denominator_train": float(np.min(abs(den))), "real_denominator_roots": denominator_poles(coeffs),
                      "optimizer_success": bool(opt.success)}
    # Independent direct Leaver checks on the same frozen holdout coordinates.
    direct = []
    for a in holdout:
        r0 = solve(float(a), 2, 2, 0, strict=True)
        r1 = solve(float(a), 2, 2, 1, strict=True)
        direct_row = {"spin": float(a), "omega220": [r0.omega.real, -r0.omega.imag],
                      "omega221": [r1.omega.real, -r1.omega.imag]}
        physical = {"omega220": r0.omega.real, "alpha220": -r0.omega.imag,
                    "alpha21": -r1.omega.imag + r0.omega.imag}
        direct_row["fit_minus_direct"] = {
            name: float(rational(a, fits[name]["coefficients"]) - value)
            for name, value in physical.items()}
        direct.append(direct_row)
        if not (r0.converged and r1.converged and r0.omega.real > 0 and r1.omega.real > 0 and r0.omega.imag < 0 and r1.omega.imag < 0 and -r1.omega.imag > -r0.omega.imag):
            raise RuntimeError(f"nonphysical or unordered direct roots at spin={a}")
    source_hashes_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("calibration inputs changed during run")
    dataset = {"train_spins": train.tolist(), "train_values": {k: v.tolist() for k,v in data.items()}, "holdout_spins": holdout.tolist(), "holdout_truth": {k: v.tolist() for k,v in _targets(holdout).items()}}
    dataset_sha = hashlib.sha256(json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"experiment": "independent bounded TEOB QNM refit", "domain": [0.0, SPIN_MAX],
            "negative_spin": "unsupported", "units": "dimensionless M_f omega and positive damping alpha",
            "train_spins": train.tolist(), "train_values": dataset["train_values"], "holdout_spins": holdout.tolist(), "holdout_truth": dataset["holdout_truth"], "dataset_sha256": dataset_sha,
            "objective": "unweighted least squares on each of omega220, alpha220, alpha21",
            "basis": "y0*(1+b1*a+b2*a^2+b3*a^3)/(1+c1*a+c2*a^2+c3*a^3), y0 fixed at a=0",
            "targets": "dimensionless_qnm(2,2,n), alpha21=alpha221-alpha220",
            "fits": fits, "direct_leaver_holdouts": direct,
            "source_sha256_before": source_hashes_before, "source_sha256_after": source_hashes_after,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
            "published_comparison": "rounded Eq. (91) coefficients evaluated after fitting; never used in optimization"}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--points", type=int, default=49); p.add_argument("--output", type=Path)
    args = p.parse_args(); report = run(args.points); text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output: args.output.write_text(text)
    print(text)


if __name__ == "__main__": main()
