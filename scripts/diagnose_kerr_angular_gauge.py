"""Clean-room comparison of the two Leaver angular phase gauges.

The public LALSimulation functions are used only as a black-box diagnostic.
"""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import lalsimulation as lalsim

from torchwave.approximants.kerr_angular import (
    leaver_frobenius_coefficients, leaver_frobenius_harmonic,
)

SPINS = [0.0, 0.3, 0.7, 0.9]
THETAS = [0.0, 0.4, 0.8, math.pi / 2, 2.3, math.pi]
# The fresh Frobenius helper is deliberately specialized to m=2 (see its
# docstring); compare both requested ell values within that validated family.
MODES = [(2, 2), (3, 2)]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_hash():
    path = Path("src/torchwave/approximants/kerr_angular.py")
    data = path.read_bytes()
    try:
        return {"sha256": sha256(data), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except OSError:
        return {"sha256": None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def public_value(spin, ell, m, theta):
    # LAL uses a=spin/2 (2M=1); its eigenvalue API returns (A, omega).
    A, omega = lalsim.SimBlackHoleRingdownModeEigenvaluesLeaver(spin / 2, ell, m, -2)
    c = (spin / 2) * omega
    value = lalsim.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(
        math.cos(theta), spin / 2, ell, m, -2, A, omega
    ) / math.sqrt(2 * math.pi)
    return complex(value), complex(A), complex(omega), complex(c)


def norm_public(spin, ell, m, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    vals = np.array([public_value(spin, ell, m, math.acos(float(x)))[0] for x in nodes])
    return float(2 * math.pi * np.sum(weights * abs(vals) ** 2))


def integrals_public(spin, ell, m, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    vals = np.array([public_value(spin, ell, m, math.acos(float(x)))[0] for x in nodes])
    return {
        "hermitian": [float(2 * math.pi * np.sum(weights * abs(vals) ** 2)), 0.0],
        "bilinear": [complex(2 * math.pi * np.sum(weights * vals ** 2)).real, complex(2 * math.pi * np.sum(weights * vals ** 2)).imag],
    }


def residual(ell, c, theta, n_terms):
    x = torch.tensor(float(math.cos(theta)), dtype=torch.float64, requires_grad=True)
    cc = torch.tensor(c, dtype=torch.complex128)
    y = leaver_frobenius_harmonic(ell, cc, torch.acos(x), n_terms=n_terms,
                                  phase_convention="a0_real_positive")
    yp = torch.complex(torch.autograd.grad(y.real, x, create_graph=True)[0], torch.autograd.grad(y.imag, x, create_graph=True)[0])
    ypp = torch.complex(torch.autograd.grad(yp.real, x, create_graph=True)[0], torch.autograd.grad(yp.imag, x)[0])
    A = leaver_frobenius_coefficients(ell, cc, n_terms=n_terms)[0]
    return float(abs(((1-x*x)*ypp - 2*x*yp + (cc*cc*x*x + 4*cc*x - 2 + A - (2-2*x)**2/(1-x*x))*y).detach()))


def public_ode_residual(spin, ell, m, theta, h):
    ref, A, omega, c = public_value(spin, ell, m, theta)
    x = math.cos(theta)
    def f(u):
        return lalsim.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(u, spin / 2, ell, m, -2, A, omega) / math.sqrt(2 * math.pi)
    yp = (f(x+h)-f(x-h))/(2*h)
    ypp = (f(x+h)-2*f(x)+f(x-h))/(h*h)
    return abs((1-x*x)*ypp - 2*x*yp + (c*c*x*x + 4*c*x - 2 + A - (m-2*x)**2/(1-x*x))*ref)


def main():
    before_run = source_hash()
    rows = []
    for spin in SPINS:
        for ell, m in MODES:
            for theta in THETAS:
                ref, A, omega, c = public_value(spin, ell, m, theta)
                cc = torch.tensor(c, dtype=torch.complex128)
                raw = complex(leaver_frobenius_harmonic(ell, cc, torch.tensor(theta, dtype=torch.float64), phase_convention="a0_real_positive").item())
                fixed = complex(leaver_frobenius_harmonic(ell, cc, torch.tensor(theta, dtype=torch.float64)).item())
                ratio = None if abs(ref) < 1e-14 else raw / ref
                fresh_A = complex(leaver_frobenius_coefficients(ell, cc, n_terms=64)[0].item())
                rows.append({"spin": spin, "ell": ell, "m": m, "theta": theta,
                    "A": [A.real, A.imag], "omega": [omega.real, omega.imag], "c": [c.real, c.imag],
                    "fresh_A": [fresh_A.real, fresh_A.imag], "fresh_minus_public_A": [fresh_A.real-A.real, fresh_A.imag-A.imag],
                    "public_unit": [ref.real, ref.imag], "raw_unit": [raw.real, raw.imag],
                    "fixed_unit": [fixed.real, fixed.imag], "raw_ratio": None if ratio is None else [ratio.real, ratio.imag],
                    "raw_abs_error": abs(raw-ref), "fixed_abs_error": abs(fixed-ref),
                    "ode_residual_a0_raw": None if theta in (0.0, math.pi) else residual(ell, c, theta, 64),
                    "public_ode_fd": None if theta in (0.0, math.pi) else {str(h): public_ode_residual(spin, ell, m, theta, h) for h in [.01,.005,.0025,.00125]}})
    convergence = []
    spin, ell, m = .7, 2, 2
    _, _, _, c = public_value(spin, ell, m, .8)
    for n_terms in [32, 48, 64, 80]:
        for order in [128, 256, 512]:
            cc = torch.tensor(c, dtype=torch.complex128)
            th = torch.tensor(.8, dtype=torch.float64)
            v = leaver_frobenius_harmonic(ell, cc, th, n_terms=n_terms, quadrature_order=order)
            nodes, weights = np.polynomial.legendre.leggauss(order)
            q = leaver_frobenius_harmonic(ell, cc, torch.acos(torch.from_numpy(nodes)), n_terms=n_terms, quadrature_order=order)
            convergence.append({"n_terms": n_terms, "quadrature_order": order, "value": [v.item().real, v.item().imag], "norm": float(2*math.pi*torch.from_numpy(weights) @ q.abs().square())})
    after_run = source_hash()
    out = {"status": "complete", "api": {"package": "lalsimulation", "version": getattr(lalsim, "__version__", None), "eigen_signature": lalsim.SimBlackHoleRingdownModeEigenvaluesLeaver.__doc__, "wave_signature": lalsim.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver.__doc__, "normalization": "public output divided by sqrt(2*pi); checked by Gauss-Legendre norm"}, "publication": "Leaver 1985, Eqs. (16)-(21), https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf", "hashes": {"before_run": before_run, "after_run": after_run, "original_startup": None, "note": "Original startup bytes were not recorded; helper was untracked."}, "cases": rows, "convergence": convergence, "public_norms": {f"{s}:{l}:{m}": {str(o): integrals_public(s,l,m,o) for o in [64,128,256]} for s in SPINS for l,m in MODES}}
    path = Path("docs/reviews/2026-09-13-parity-review/verification/kerr-angular")
    path.mkdir(parents=True, exist_ok=True)
    (path / "gauge.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"hashes": out["hashes"], "max_raw_error": max(x["raw_abs_error"] for x in rows), "max_fixed_error": max(x["fixed_abs_error"] for x in rows), "max_ode_residual": max(x["ode_residual_a0_raw"] or 0 for x in rows)}, indent=2))


if __name__ == "__main__":
    main()
