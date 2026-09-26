"""Finite interstitial QNM derivative and pinned waveform-impact certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import h5py
import jax
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline
import torch

from . import qualify_generator as q

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from diffgw_core.kerr_qnm_220_data import (  # noqa: E402
    COEFFS_220, COEFFS_2M20, DEGREE, KNOTS_220, KNOTS_2M20,
)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220  # noqa: E402
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220  # noqa: E402


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-27/qnm_spline_component_plan.json"


def _author_fit(data: np.ndarray, magnitude: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Local polynomial value and signed-spin derivative from author rows."""
    x = -np.log1p(-data[:, 0])
    target = -np.log1p(-magnitude)
    selected = np.sort(np.argsort(np.abs(x - target))[:count])
    xs = x[selected]
    ys = np.stack((data[selected, 1], -data[selected, 2]), axis=-1)
    if len(xs) != count or not np.all(np.diff(xs) > 0):
        raise ValueError("invalid Cook author interpolation stencil")
    fits = [Polynomial.fit(xs, ys[:, component], deg=count - 1,
                           domain=[xs[0], xs[-1]]) for component in range(2)]
    value = np.asarray([fit(target) for fit in fits], dtype=np.float64)
    derivative_magnitude = np.asarray([fit.deriv()(target) for fit in fits]) / (1 - magnitude)
    return value, derivative_magnitude


def _spline_value_derivative(spin: float) -> tuple[np.ndarray, np.ndarray]:
    knots, controls = ((KNOTS_220, COEFFS_220) if spin >= 0
                       else (KNOTS_2M20, COEFFS_2M20))
    magnitude = abs(spin)
    x = -np.log1p(-magnitude)
    coefficients = np.asarray(controls, dtype=np.float64) * np.array([1.0, -1.0])
    splines = [BSpline(knots, coefficients[:, component], DEGREE, extrapolate=False)
               for component in range(2)]
    value = np.asarray([spline(x) for spline in splines])
    derivative = np.asarray([spline.derivative()(x) for spline in splines])
    derivative *= (1.0 if spin >= 0 else -1.0) / (1 - magnitude)
    return value, derivative


def _torch_value_derivative(spin: float) -> tuple[np.ndarray, np.ndarray]:
    argument = torch.tensor(spin, dtype=torch.float64, requires_grad=True)
    values = signed_qnm_220(argument)
    derivatives = [torch.autograd.grad(value, argument, retain_graph=True)[0]
                   for value in values]
    return (np.asarray([float(value.detach()) for value in values]),
            np.asarray([float(derivative.detach()) for derivative in derivatives]))


def _jax_value_derivative(spin: float) -> tuple[np.ndarray, np.ndarray]:
    def vector(argument):
        return jnp.stack(evaluate_kerr_qnm_220(argument))

    argument = jnp.asarray(spin, dtype=jnp.float64)
    return np.asarray(vector(argument)), np.asarray(jax.jacfwd(vector)(argument))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-spline-component-plan.v1"
            or plan["component_id"] != "V4-QNM-NATIVE-220-SPLINE"
            or len(plan["signed_spins"]) != 12
            or plan["no_continuous_bound_claim"] is not True
            or plan["historical_binary_parity_claim"] is not False
            or plan["release_qualified"] is not False
            or plan["production_change"] is not False):
        parser.error("frozen component plan differs")
    for name, digest in {**plan["source_sha256"],
                         **plan["prior_evidence_sha256"]}.items():
        if q.sha256(q.ROOT / name) != digest:
            parser.error(f"pinned source or prior evidence changed: {name}")
    if q.sha256(args.author_archive) != plan["author_archive_sha256"]:
        parser.error("Cook author archive SHA-256 changed")
    prior = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-25/"
                        "qnm_ringdown_waveform_current.json").read_text())
    witness = prior["rows"][plan["waveform_impact_row"]]
    if (prior["status"] != "diagnostic_complete"
            or witness["baseline_reconstruction_complex_relative_l2"] != 0.0
            or witness["whole_mode_complex_relative_l2"] >=
            plan["waveform_whole_mode_complex_relative_l2_max"]):
        parser.error("pinned waveform impact witness fails its frozen gate")
    with h5py.File(args.author_archive) as archive:
        author = {branch: np.asarray(archive[path][:, :3], dtype=np.float64)
                  for branch, path in plan["author_datasets"].items()}
    for branch, data in author.items():
        if (data.ndim != 2 or data.shape[1] != 3 or data[0, 0] != 0.0
                or not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0)
                or data[-1, 0] <= 0.9999):
            parser.error(f"unexpected Cook author data: {branch}")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.qnm-spline-component.v1", status="running",
        component_id=plan["component_id"], release_qualified=False,
        production_change=False, historical_binary_parity_claim=False,
        continuous_domain_error_bound_claim=False,
        scope=dict(q.SCOPE, description="Finite independent-author-data value and derivative check for the native QNM spline.",
                   known_limitations=[
                       "Local author-data polynomial derivatives are numerical estimates; stencil agreement is not a rigorous interpolation remainder bound.",
                       "These 12 interstitial points do not establish a continuous-domain error bound.",
                       "The pinned single-row QNM substitution is a fixed-boundary whole-mode witness, not a domain-wide waveform or historical-binary parity qualification.",
                   ]),
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        source_sha256=plan["source_sha256"],
        prior_evidence_sha256=plan["prior_evidence_sha256"],
        author_archive_sha256=plan["author_archive_sha256"],
        process=dict(host=socket.gethostname(), cwd=str(q.ROOT)),
        waveform_impact=dict(row=plan["waveform_impact_row"],
                             whole_mode_complex_relative_l2=witness["whole_mode_complex_relative_l2"],
                             fixed_boundary_reconstruction_complex_relative_l2=witness["baseline_reconstruction_complex_relative_l2"],
                             prior_evidence="dev/native_rom/evidence/2026-09-25/qnm_ringdown_waveform_current.json"),
        rows=[], errors=[], gates={})
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        for spin in plan["signed_spins"]:
            magnitude = abs(spin)
            data = author["positive" if spin > 0 else "negative"]
            author8, derivative8 = _author_fit(data, magnitude, 8)
            author10, derivative10 = _author_fit(data, magnitude, 10)
            if spin < 0:
                derivative8, derivative10 = -derivative8, -derivative10
            analytic, analytic_derivative = _spline_value_derivative(spin)
            torch_value, torch_derivative = _torch_value_derivative(spin)
            jax_value, jax_derivative = _jax_value_derivative(spin)
            row = dict(signed_spin=spin, cook_stencil_8=dict(value=author8.tolist(),
                            derivative=derivative8.tolist()),
                       cook_stencil_10=dict(value=author10.tolist(),
                            derivative=derivative10.tolist()),
                       native_analytic=dict(value=analytic.tolist(),
                            derivative=analytic_derivative.tolist()),
                       torch=dict(value=torch_value.tolist(), derivative=torch_derivative.tolist()),
                       jax=dict(value=jax_value.tolist(), derivative=jax_derivative.tolist()),
                       value_error_abs=np.abs(analytic-author10).tolist(),
                       derivative_error_scaled=(np.abs(analytic_derivative-derivative10)
                            / np.maximum(1.0, np.abs(derivative10))).tolist(),
                       author_stencil_derivative_difference_scaled=(np.abs(derivative8-derivative10)
                            / np.maximum(1.0, np.abs(derivative10))).tolist(),
                       torch_jax_derivative_error_abs=np.abs(torch_derivative-jax_derivative).tolist(),
                       analytic_backend_derivative_error_abs=np.maximum(
                            np.abs(analytic_derivative-torch_derivative),
                            np.abs(analytic_derivative-jax_derivative)).tolist())
            if not np.isfinite(np.asarray([*row["value_error_abs"],
                                          *row["derivative_error_scaled"],
                                          *row["author_stencil_derivative_difference_scaled"],
                                          *row["torch_jax_derivative_error_abs"],
                                          *row["analytic_backend_derivative_error_abs"]])).all():
                raise ValueError(f"nonfinite result at signed spin {spin}")
            report["rows"].append(row)
            q.write_evidence(args.json_out, report)
        fields = ("value_error_abs", "derivative_error_scaled",
                  "author_stencil_derivative_difference_scaled",
                  "torch_jax_derivative_error_abs", "analytic_backend_derivative_error_abs")
        maxima = {field: max(max(row[field]) for row in report["rows"]) for field in fields}
        report["summary"] = dict(row_count=len(report["rows"]), maxima=maxima,
                                 elapsed_seconds=time.monotonic()-started)
        report["gates"] = dict(
            author_value=maxima["value_error_abs"] < plan["value_absolute_tolerance"],
            author_derivative=maxima["derivative_error_scaled"] < plan["derivative_scaled_tolerance"],
            author_stencil_stability=maxima["author_stencil_derivative_difference_scaled"]
                < plan["author_stencil_scaled_tolerance"],
            torch_jax_derivative=maxima["torch_jax_derivative_error_abs"]
                < plan["torch_jax_derivative_absolute_tolerance"],
            analytic_backend_derivative=maxima["analytic_backend_derivative_error_abs"]
                < plan["analytic_backend_derivative_absolute_tolerance"],
            pinned_waveform_impact=witness["whole_mode_complex_relative_l2"]
                < plan["waveform_whole_mode_complex_relative_l2_max"])
        report["status"] = ("diagnostic_complete" if all(report["gates"].values())
                            else "diagnostic_failed_gates")
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], summary=report.get("summary"),
                          gates=report["gates"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
