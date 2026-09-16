"""Independent spin -2 Kerr eigenproblem solver, in units M=1.

The radial recurrence and its asymptotic remainder are reconstructed from
Cook & Zalutskiy, arXiv:1410.7698v2, Eqs. (23)--(44). The angular matrix follows
their Eqs. (49)--(56). No external QNM implementation or numerical table is
read. NumPy and SciPy supply generic linear algebra and root finding. Optional
gmpy2 supplies higher precision arithmetic for near-extremal damped branches;
it is required when generating the full spin domain for those modes.

The convention is exp(-i omega t), Re(omega)>0, Im(omega)<0, and the returned
separation constant is A, with A(c=0)=ell*(ell+1)-2, not lambda.
"""

from dataclasses import dataclass
from functools import lru_cache
import cmath
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class KerrRoot:
    omega: complex
    separation_constant: complex
    radial_residual: complex
    cf_depth: int
    cf_error: float
    angular_error: float
    converged: bool
    metadata: dict


@lru_cache(maxsize=128)
def _angular_basis(m: int, ell_max: int):
    ell = np.arange(max(2, abs(m)), ell_max + 2, dtype=float)
    upper = ell[1:]
    off = np.sqrt((upper**2 - m * m) * (upper**2 - 4) / (upper**2 * (4 * upper**2 - 1)))
    x = np.diag(2 * m / (ell * (ell + 1))) + np.diag(off, 1) + np.diag(off, -1)
    return np.diag(ell[:-1] * (ell[:-1] + 1) - 2), x[:-1, :-1], (x @ x)[:-1, :-1]


def angular_eigenvalue(c: complex, ell: int, m: int, seed: Optional[complex] = None, ell_max: int = 20) -> complex:
    """Select A on the spherical branch, or nearest a continuation seed."""
    if ell < max(2, abs(m)) or ell_max < ell + 3:
        raise ValueError("Require ell>=max(2,abs(m)) and ell_max>=ell+3")
    c = complex(c)
    if not np.isfinite(c) or abs(c) > 2:
        raise ValueError("Angular solver requires finite |a*omega|<=2")
    if c == 0:
        return complex(ell * (ell + 1) - 2)
    a0, x, x2 = _angular_basis(m, ell_max)
    previous = complex(ell * (ell + 1) - 2) if seed is None else complex(seed)
    steps = range(1, 9) if seed is None else (8,)
    for step in steps:
        z = c * (step / 8)
        eigenvalues = np.linalg.eigvals(a0 - z * z * x2 - 4 * z * x)
        previous = complex(eigenvalues[np.argmin(abs(eigenvalues - previous))])
    return previous


def _angular_extended(a, omega, ell, m, seed, ell_max, precision):
    """Refine a complex-symmetric Rayleigh quotient in generic MP arithmetic."""
    import gmpy2

    c_double = a * omega
    anchor = angular_eigenvalue(c_double, ell, m, seed, ell_max)
    a0, x, x2 = _angular_basis(m, ell_max)
    values, vectors = np.linalg.eig(a0 - c_double * c_double * x2 - 4 * c_double * x)
    direction = vectors[:, np.argmin(abs(values - anchor))]
    with gmpy2.context(gmpy2.get_context(), precision=precision):
        c = gmpy2.mpfr(a) * gmpy2.mpc(omega)
        ls = list(range(max(2, abs(m)), ell_max + 2))
        v = [gmpy2.mpc(z) for z in direction] + [gmpy2.mpc(0)]
        diagonal = [gmpy2.mpfr(2 * m) / (l * (l + 1)) for l in ls]
        off = [gmpy2.sqrt(gmpy2.mpfr((l * l - m * m) * (l * l - 4)) / (l * l * (4 * l * l - 1))) for l in ls[1:]]
        xv = [
            diagonal[k] * v[k]
            + (off[k - 1] * v[k - 1] if k else 0)
            + (off[k] * v[k + 1] if k + 1 < len(v) else 0)
            for k in range(len(v))
        ]
        norm = sum(z * z for z in v)
        if abs(norm) < 1e-8:
            raise ValueError("Angular eigenvector is near a self-orthogonal degeneracy")
        diagonal_part = sum((l * (l + 1) - 2) * z * z for l, z in zip(ls, v))
        return (diagonal_part - c * c * sum(z * z for z in xv) - 4 * c * sum(z * w for z, w in zip(v, xv))) / norm


def _radial_parameters(a, m, omega, separation_constant, sqrt_real=math.sqrt, sqrt_complex=cmath.sqrt):
    s = -2
    b = sqrt_real((1 - a) * (1 + a))
    sigma_plus = (2 * omega * (1 + b) - m * a) / (2 * b)
    sigma_minus = (2 * omega * (1 - b) - m * a) / (2 * b)
    xi, eta = -s - 1j * sigma_plus, -s + 1j * sigma_minus
    p = 1j * b * omega
    alpha = 1 - 4j * omega
    gamma, delta = 1 + s + 2 * eta, 1 + s + 2 * xi
    sigma = (
        separation_constant
        + a * a * omega * omega
        - 8 * omega * omega
        + p * (2 * alpha + gamma - delta)
        + (1 + s - (gamma + delta) / 2) * (s + (gamma + delta) / 2)
    )
    d0, d1, d2 = delta, 4 * p - 2 * alpha + gamma - delta - 2, 2 * alpha - gamma + 2
    d3, d4 = alpha * (4 * p - delta) - sigma, alpha * (alpha - gamma + 1)
    u1 = sqrt_complex(-4 * p)
    if u1.real > 0:
        u1 = -u1
    if u1.real == 0:
        raise ValueError("The minimal asymptotic branch is degenerate")
    u2 = -(8 * p - 4 * alpha + 2 * gamma + 2 * delta + 3) / 4
    u3 = (
        32 * p * (2 * p - 4 * alpha + gamma + 3 * delta + 4)
        + 4 * (gamma + delta) * (gamma + delta - 2)
        + 16 * sigma
        + 3
    ) / (32 * u1)
    return d0, d1, d2, d3, d4, u1, u2, u3


def _fraction(parameters, n, depth, square_root=math.sqrt):
    d0, d1, d2, d3, d4, u1, u2, u3 = parameters
    beta_linear, gamma_linear, gamma_constant = d1 + 2, d2 - 3, d4 - d2 + 2

    def coefficients(k):
        square = k * k
        return (
            (k + 1) * (k + d0),
            -2 * square + beta_linear * k + d3,
            square + gamma_linear * k + gamma_constant,
        )

    remainder = 1 + u1 / square_root(depth) + u2 / depth + u3 / (depth * square_root(depth))
    for k in range(depth, n, -1):
        alpha, beta, gamma = coefficients(k)
        remainder = -gamma / (beta + alpha * remainder)
    left = 0j
    for k in range(1, n + 1):
        alpha_previous, beta_previous, _ = coefficients(k - 1)
        _, _, gamma = coefficients(k)
        left = alpha_previous * gamma / (beta_previous - left)
    alpha, beta, _ = coefficients(n)
    return beta - left + alpha * remainder


def radial_residual(a, m, n, omega, separation_constant, depth=512, precision=53):
    """Unscaled n-inverted radial continued fraction, Eqs. (39), (44)."""
    if not 0 <= n < depth:
        raise ValueError("Require 0<=n<depth")
    if precision > 53:
        import gmpy2

        with gmpy2.context(gmpy2.get_context(), precision=precision):
            parameters = _radial_parameters(
                gmpy2.mpfr(a),
                m,
                gmpy2.mpc(omega),
                gmpy2.mpc(separation_constant),
                gmpy2.sqrt,
                gmpy2.sqrt,
            )
            return complex(_fraction(parameters, n, depth, gmpy2.sqrt))
    parameters = _radial_parameters(a, m, complex(omega), complex(separation_constant))
    return _fraction(parameters, n, depth)


def _correct(a, ell, m, n, omega_seed, angular_seed, strict=False, fixed_a=None):
    ell_max = max(ell + 8, 24 if strict else 18)
    max_depth = 131072
    horizon_scale = math.sqrt((1 - a) * (1 + a))
    damped = a * m < 0 or (abs(m) < ell and -complex(omega_seed).imag > 3 * horizon_scale)
    initial_depth = 512 if strict else 128
    if damped:
        decay = -_radial_parameters(a, m, complex(omega_seed), 0.0)[5].real
        estimate = ((4.5 if strict else 3.5) / decay) ** 2
        initial_depth = max(initial_depth, 2 ** int(math.ceil(math.log2(estimate))))
    initial_depth = min(max_depth // 2, initial_depth)
    precision = (160 if strict else 113) if horizon_scale < 0.03 and damped else 53
    tolerance = 3e-13 if strict else 1e-12
    omega = complex(omega_seed)

    def angular(z, seed, basis=ell_max):
        if fixed_a is not None:
            return complex(fixed_a)
        if precision > 53:
            return _angular_extended(a, z, ell, m, seed, basis, precision)
        return angular_eigenvalue(a * z, ell, m, seed, basis)

    def radial(z, eigenvalue, depth):
        return radial_residual(a, m, n, z, eigenvalue, depth, precision)

    anchor = angular(omega, angular_seed)
    total_evaluations, success = 0, False
    frequency_change, cf_error = float("inf"), float("inf")
    for depth in (initial_depth * 2**k for k in range(int(math.log2(max_depth / initial_depth)) + 1)):

        def objective(parts):
            z = complex(*parts)
            try:
                eigenvalue = angular(z, anchor)
                value = radial(z, eigenvalue, depth)
            except (ValueError, ZeroDivisionError, OverflowError):
                return (1e100, 1e100)
            return (value.real, value.imag)

        solution = root(
            objective,
            [omega.real, omega.imag],
            method="hybr",
            options={"xtol": 3e-12, "maxfev": 100},
        )
        total_evaluations += solution.nfev
        refined = complex(*solution.x)
        frequency_change = abs(refined - omega)
        omega = refined
        anchor = angular(omega, anchor)
        residual = radial(omega, anchor, depth)
        deeper = radial(omega, anchor, 2 * depth)
        cf_error = abs(deeper - residual)
        derivative_step = 1e-6
        shifted_a = angular(omega + derivative_step, anchor)
        derivative = (radial(omega + derivative_step, shifted_a, depth) - residual) / derivative_step
        derivative_size = max(abs(derivative), 1e-12)
        frequency_error = (abs(residual) + cf_error) / derivative_size
        if depth > initial_depth and frequency_change < tolerance and frequency_error < tolerance:
            success = True
            break
    refined_angular = angular(omega, anchor, ell_max + 4)
    angular_error = float(abs(refined_angular - anchor))
    physical = omega.real > 0 and omega.imag < 0
    return KerrRoot(
        omega,
        complex(anchor),
        residual,
        depth,
        cf_error,
        angular_error,
        bool(success and physical and angular_error < 3e-11),
        {
            "angular_ell_max": ell_max,
            "frequency_depth_change": frequency_change,
            "estimated_frequency_error": frequency_error,
            "radial_derivative_magnitude": derivative_size,
            "root_evaluations": total_evaluations,
            "strict": strict,
            "arithmetic_bits": precision,
            "branch_origin": "provided seed; overtone identity requires continuation",
            "minimal_remainder_real_part": _radial_parameters(a, m, omega, anchor)[5].real,
        },
    )


def _wkb_seed(ell: int, n: int) -> complex:
    angular = ell * (ell + 1)
    radius = (3 * (angular + 3) + math.sqrt(9 * (angular + 3) ** 2 - 96 * angular)) / (2 * angular)
    potential = angular / radius**2 - (2 * angular + 6) / radius**3 + 12 / radius**4
    second = 6 * angular / radius**4 - 12 * (2 * angular + 6) / radius**5 + 240 / radius**6
    second *= (1 - 2 / radius) ** 2
    return cmath.sqrt(potential - 1j * (n + 0.5) * math.sqrt(-2 * second))


@lru_cache(maxsize=3)
def _schwarzschild_roots(n: int):
    seed = _wkb_seed(20, n)
    roots = {}
    for effective_ell in np.linspace(20.0, 2.0, 181):
        eigenvalue = effective_ell * (effective_ell + 1) - 2
        result = _correct(0.0, 2, 0, n, seed, eigenvalue, fixed_a=eigenvalue)
        if not result.converged:
            raise RuntimeError(f"Schwarzschild WKB continuation failed at ell={effective_ell}, n={n}")
        if abs(result.omega - seed) > 0.1:
            raise RuntimeError("Schwarzschild continuation made an unexpectedly large branch jump")
        seed = result.omega
        if abs(effective_ell - round(effective_ell)) < 1e-12:
            roots[round(effective_ell)] = result
    return roots


def solve(a: float, ell: int, m: int, n: int, omega_seed: Optional[complex] = None, angular_seed: Optional[complex] = None, strict: bool = False) -> KerrRoot:
    """Solve one low Kerr overtone with refinement diagnostics.

    Supported: ell=2..12, n=0..2, |m|<=ell, |a|<1 and |a*omega|<=2.
    """
    if any(int(x) != x for x in (ell, m, n)):
        raise ValueError("ell, m, and n must be integers")
    ell, m, n, a = int(ell), int(m), int(n), float(a)
    if not np.isfinite(a) or abs(a) >= 1 or not 2 <= ell <= 12 or abs(m) > ell or not 0 <= n <= 2:
        raise ValueError("Require ell=2..12, |m|<=ell, n=0..2 and finite |a|<1")
    if omega_seed is not None:
        if not np.isfinite(omega_seed):
            raise ValueError("Frequency seed must be finite")
        return _correct(a, ell, m, n, omega_seed, angular_seed, strict)
    initial = _schwarzschild_roots(n)[ell]
    if a == 0:
        result = _correct(a, ell, m, n, initial.omega, initial.separation_constant, strict)
    else:
        target_coordinate = -math.log1p(-abs(a))
        count = max(2, int(math.ceil(target_coordinate / 0.025)) + 1)
        spins = np.sign(a) * (-np.expm1(-np.linspace(0.0, target_coordinate, count)))
        spins[-1] = a
        previous = [(0.0, initial)]
        for spin in spins[1:]:
            coordinate = -math.log1p(-abs(float(spin)))
            seed, angular_seed = previous[-1][1].omega, previous[-1][1].separation_constant
            if len(previous) > 1:
                x0, r0 = previous[-2]
                x1, r1 = previous[-1]
                ratio = (coordinate - x1) / (x1 - x0)
                seed += ratio * (r1.omega - r0.omega)
                angular_seed += ratio * (r1.separation_constant - r0.separation_constant)
            result = _correct(
                float(spin),
                ell,
                m,
                n,
                seed,
                angular_seed,
                strict and spin == spins[-1],
            )
            if not result.converged or abs(result.omega - seed) > 0.1 * abs(previous[-1][1].omega.imag):
                raise RuntimeError(f"Spin continuation failed at a={spin}, mode={(ell,m,n)}")
            previous = (previous + [(coordinate, result)])[-2:]
    return KerrRoot(
        result.omega,
        result.separation_constant,
        result.radial_residual,
        result.cf_depth,
        result.cf_error,
        result.angular_error,
        result.converged,
        {**result.metadata, "branch_origin": "large-ell WKB then separation/spin continuation"},
    )


def angular_verification_report(output: str = "kerr_angular_convergence.json"):
    """Save clean-room Frobenius convergence evidence (no oracle data)."""
    import hashlib
    import json
    import platform
    import torch
    from torchwave.approximants.kerr_angular import (
        leaver_frobenius_coefficients,
        leaver_frobenius_harmonic,
        spin_weighted_spheroidal_harmonic,
    )

    c_values = [0.0, 0.3, 0.7, 0.95, 0.3 - 0.1j, 0.7 - 0.2j, 0.95 - 0.25j]
    theta = torch.linspace(0.04, math.pi - 0.04, 33, dtype=torch.float64)
    qnodes, qweights = np.polynomial.legendre.leggauss(512)
    qx = torch.from_numpy(qnodes)
    qw = torch.from_numpy(qweights)
    rows = []
    for cvalue in c_values:
        c = torch.tensor(cvalue, dtype=torch.complex128)
        a32, _ = leaver_frobenius_coefficients(2, c, n_terms=32)
        a48, coeff = leaver_frobenius_coefficients(2, c, n_terms=48)
        y32 = leaver_frobenius_harmonic(2, c, theta, n_terms=32)
        y48 = leaver_frobenius_harmonic(2, c, theta, n_terms=48)
        yref = spin_weighted_spheroidal_harmonic(2, 2, c, theta, ell_max=18)
        a56, _ = leaver_frobenius_coefficients(2, c, n_terms=56)
        ygrid = leaver_frobenius_harmonic(2, c, theta, n_terms=56, quadrature_order=512)
        yq = leaver_frobenius_harmonic(2, c, torch.acos(qx), n_terms=48, quadrature_order=512)
        norm_error = float(abs(2 * math.pi * (qw @ yq.abs().square()) - 1))
        norm_orders = {}
        for qo in (128, 256, 512):
            yqo = leaver_frobenius_harmonic(2, c, torch.acos(qx), n_terms=48, quadrature_order=qo)
            norm_orders[str(qo)] = float(abs(2 * math.pi * (qw @ yqo.abs().square()) - 1))
        xx = torch.linspace(-0.8, 0.8, 17, dtype=torch.float64, requires_grad=True)
        yy = leaver_frobenius_harmonic(2, c, torch.acos(xx), n_terms=56)
        yr = torch.autograd.grad(yy.real.sum(), xx, create_graph=True)[0]
        yi = torch.autograd.grad(yy.imag.sum(), xx, create_graph=True)[0]
        yp = yr + 1j * yi
        ypp = (
            torch.autograd.grad(yp.real.sum(), xx, create_graph=True)[0]
            + 1j * torch.autograd.grad(yp.imag.sum(), xx)[0]
        )
        ode = (1 - xx * xx) * ypp - 2 * xx * yp + (c * c * xx * xx + 4 * c * xx - 2 + a56 - (2 - 2 * xx) ** 2 / (1 - xx * xx)) * yy
        rows.append(
            {
                "c": [float(cvalue.real), float(cvalue.imag)],
                "A_n32": [float(a32.real), float(a32.imag)],
                "A_n48": [float(a48.real), float(a48.imag)],
                "delta": float(abs(a48 - a32)),
                "harmonic_delta_max": float((y48 - y32).abs().max()),
                "harmonic_vs_spectral_max": float((ygrid - yref).abs().max()),
                "normalization_512_error": norm_error,
                "normalization_order_errors": norm_orders,
                "ode_residual_max": float(ode.abs().max().detach()),
                "coefficient_tail_n48": float(torch.linalg.vector_norm(coeff[-4:])),
            }
        )
    source = Path(__file__).resolve().parents[1] / "approximants/kerr_angular.py"
    report = {
        "inputs": {"s": -2, "ell": 2, "m": 2, "c": [[float(v.real), float(v.imag)] for v in c_values], "truncations": [32, 48]},
        "rows": rows,
        "provenance": {
            "paper": "E. W. Leaver, Proc. R. Soc. A 402, 285 (1985)",
            "url": "https://blackholes.tecnico.ulisboa.pt/gritting/pdf/black_holes/Leaver_An-Analytic-representation-for-the-quasi-normal-modes-of-Kerr-black-holes.pdf",
            "equations": "angular equation (16), Frobenius series (18), recurrence (19), coefficients (20), continued fraction (21), specialized algebraically to s=-2,m=2",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
            "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
