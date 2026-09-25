#!/usr/bin/env python3
"""Dense JAX validation for the paper-evidenced XPNR angle components.

This is a component check, not a full-waveform or physical-accuracy claim.
It evaluates the inherited polynomial tables through the published XO4a
polynomial and parameter transforms, then checks sampled values against the
stdlib scalar reference.  It also compares JAX derivatives of the printed
alpha/beta merger-ringdown ansatzes with independent analytic derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diffgw_reference.xpnr_angles import (
    alpha_parameters,
    beta_parameters,
    evaluate_calibration_polynomial,
    freeze_fit_coordinates,
)
from jaxwave.approximants.pnr_coefficients import COEFF_ARRAYS
from jaxwave.approximants.precession_pnr import (
    _eval_coeff_poly,
    compute_pnr_alpha_coefficients,
    compute_pnr_beta_coefficients,
    eval_pnr_alpha_mr,
    eval_pnr_beta_mr,
)


NAMES = ("A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3", "B4", "B5", "Bf")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=262_144)
    parser.add_argument("--scalar-samples", type=int, default=128)
    parser.add_argument("--output")
    return parser.parse_args()


def _canonical_table_hash() -> str:
    digest = hashlib.sha256()
    for name in NAMES:
        values = np.asarray(COEFF_ARRAYS[name], dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _dense_components(q: jax.Array, chi: jax.Array, costheta: jax.Array) -> jax.Array:
    eta = q / (1.0 + q) ** 2
    polynomials = jnp.stack([_eval_coeff_poly(COEFF_ARRAYS[name], eta, chi, costheta) for name in NAMES])
    alpha = jnp.stack(compute_pnr_alpha_coefficients(eta, chi, costheta, q))
    theta_final = 0.2 + 0.6 * (costheta + 0.95) / 1.9
    beta = jnp.stack(compute_pnr_beta_coefficients(eta, chi, costheta, jnp.cos(theta_final), q))
    return jnp.concatenate((polynomials, alpha, beta), axis=0)


def _ansatz_derivative_errors(mf: jax.Array) -> tuple[jax.Array, jax.Array]:
    alpha_coefficients = (0.04, -0.02, 0.01, 0.12)
    beta_coefficients = (0.5, 0.03, -0.2, -0.08, 180.0, -0.03)
    jax_alpha = jax.vmap(jax.grad(lambda x: eval_pnr_alpha_mr(x, *alpha_coefficients)))(mf)
    jax_beta = jax.vmap(jax.grad(lambda x: eval_pnr_beta_mr(x, *beta_coefficients)))(mf)

    a1, a2, a3, a4 = alpha_coefficients
    alpha_denominator = a3 + (mf - a4) ** 2
    exact_alpha = a1 / mf**2 + 2.0 * a2 * jnp.sqrt(a3) * (mf - a4) / alpha_denominator**2

    _, b1, b2, b3, b4, b5 = beta_coefficients
    numerator = b1 + b2 * mf + b3 * mf**2
    denominator = 1.0 + b4 * (mf + b5) ** 2
    exact_beta = ((b2 + 2.0 * b3 * mf) * denominator - numerator * 2.0 * b4 * (mf + b5)) / denominator**2
    return jnp.max(jnp.abs(jax_alpha - exact_alpha)), jnp.max(jnp.abs(jax_beta - exact_beta))


def main() -> int:
    args = _arguments()
    if args.points < 2 or args.scalar_samples < 1:
        raise SystemExit("--points must be >= 2 and --scalar-samples must be >= 1")

    phase = jnp.linspace(0.0, 1.0, args.points, dtype=jnp.float64)
    q = 1.0 + 7.0 * phase
    chi = 0.01 + 0.79 * jnp.mod(phase * 8191.0, 1.0)
    costheta = -0.95 + 1.90 * jnp.mod(phase * 131071.0, 1.0)
    mf = jnp.linspace(0.02, 0.2, args.points, dtype=jnp.float64)

    started = time.perf_counter()
    dense = jax.jit(_dense_components)(q, chi, costheta)
    derivative_errors = jax.jit(_ansatz_derivative_errors)(mf)
    dense.block_until_ready()
    derivative_errors[0].block_until_ready()
    elapsed = time.perf_counter() - started

    dense_numpy = np.asarray(dense)
    sample_indices = np.linspace(0, args.points - 1, args.scalar_samples, dtype=np.int64)
    max_abs = 0.0
    max_rel = 0.0
    failures = 0
    for index in sample_indices:
        q_value = float(q[index])
        chi_value = float(chi[index])
        costheta_value = float(costheta[index])
        eta_value = q_value / (1.0 + q_value) ** 2
        fit = freeze_fit_coordinates(eta_value, chi_value, costheta_value, q_value)
        lambdas = {
            name: evaluate_calibration_polynomial(np.asarray(COEFF_ARRAYS[name]), eta_value, chi_value, costheta_value)
            for name in NAMES
        }
        fit_lambdas = {
            name: evaluate_calibration_polynomial(np.asarray(COEFF_ARRAYS[name]), fit.eta, fit.chi, fit.cos_theta_ls)
            for name in NAMES
        }
        theta_final = 0.2 + 0.6 * (costheta_value + 0.95) / 1.9
        expected = np.asarray(
            [
                *lambdas.values(),
                *alpha_parameters(fit.chi, fit.cos_theta_ls, *(fit_lambdas[name] for name in NAMES[:4])),
                *beta_parameters(
                    theta_final,
                    fit.chi,
                    fit.cos_theta_ls,
                    *(fit_lambdas[name] for name in NAMES[4:10]),
                ),
            ],
            dtype=np.float64,
        )
        observed = dense_numpy[:, index]
        absolute = np.abs(observed - expected)
        relative = absolute / np.maximum(np.abs(expected), 1.0)
        max_abs = max(max_abs, float(np.max(absolute)))
        max_rel = max(max_rel, float(np.max(relative)))
        failures += int(np.count_nonzero((absolute > 2.0e-10) & (relative > 2.0e-10)))

    result = {
        "scope": "XPNR printed polynomial/transform/ansatz component evidence only",
        "qualified_full_waveform": False,
        "host": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "points": args.points,
        "scalar_samples": args.scalar_samples,
        "canonical_coefficient_hash": _canonical_table_hash(),
        "all_dense_values_finite": bool(np.isfinite(dense_numpy).all()),
        "scalar_comparison_failures": failures,
        "max_scalar_absolute_error": max_abs,
        "max_scalar_scaled_error": max_rel,
        "max_alpha_derivative_absolute_error": float(derivative_errors[0]),
        "max_beta_derivative_absolute_error": float(derivative_errors[1]),
        "elapsed_seconds": elapsed,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    return int(
        not result["all_dense_values_finite"]
        or failures != 0
        or result["max_alpha_derivative_absolute_error"] > 2.0e-10
        or result["max_beta_derivative_absolute_error"] > 2.0e-10
    )


if __name__ == "__main__":
    raise SystemExit(main())
