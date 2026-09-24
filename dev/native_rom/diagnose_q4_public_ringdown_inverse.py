"""Invert the published q4 ringdown ansatz on a held-out public mode window.

This diagnostic fits only the public postpeak mode to its own analytic ansatz.
It does not fit the native generator or qualify either implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_peak, public_mode
from torchwave.approximants.eob.merger_ringdown import (
    seobnrv4_merger_shape, seobnrv4_peak_preserving_shape,
)
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters
from torchwave.constants import C_SI, MPC_SI


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_public_ringdown_inverse_plan.json"


def _phase_kernel(u, w, d1, d2):
    change = np.expm1(-d1*u)
    z = d2/(1+d2)*change
    ratio = np.ones_like(z)
    np.divide(np.log1p(z), z, out=ratio, where=np.abs(z) > 1e-10)
    return change*ratio/d1


def _fit(time_M, mode, shape, w, gamma, window, bounds, kind):
    mask = (time_M >= window[0]) & (time_M <= window[1])
    t = time_M[mask]
    measured = np.log(np.abs(mode[mask])) if kind == "amplitude" else -np.unwrap(np.angle(mode[mask]))
    c1, c2, d1, d2 = shape

    def values(t0):
        u = t-t0
        transition = np.tanh(c1*u)
        factor = 1 + gamma/c1*transition/(1+transition*math.tanh(c2))
        if kind == "amplitude":
            basis = np.log(factor)-gamma*u
            a = float(np.mean(measured-basis))
            residual = measured-(a+basis)
            return float(np.mean(residual**2)), dict(t0_M=float(t0), amplitude=float(np.exp(a)))
        kernel = _phase_kernel(u, w, d1, d2)
        basis = w*(u+kernel)
        matrix = np.column_stack((np.ones_like(u), -kernel))
        phi0, w0 = np.linalg.lstsq(matrix, measured-basis, rcond=None)[0]
        residual = measured-(basis+matrix@np.array([phi0, w0]))
        return float(np.mean(residual**2)), dict(t0_M=float(t0), phase=float(phi0), frequency_M=float(w0))

    result = minimize_scalar(lambda t0: values(t0)[0], bounds=bounds,
                             method="bounded", options={"xatol": 1e-14})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"{kind} inverse fit failed")
    loss, parameters = values(result.x)
    parameters["fit_rms"] = math.sqrt(loss)
    parameters["fit_samples"] = int(len(t))
    return parameters


def _predictions(time_M, amplitude_fit, phase_fit, shape, w, gamma):
    c1, c2, d1, d2 = shape
    u = time_M-amplitude_fit["t0_M"]
    transition = np.tanh(c1*u)
    amplitude = amplitude_fit["amplitude"]*(1+gamma/c1*transition/
                  (1+transition*math.tanh(c2)))*np.exp(-gamma*u)
    u = time_M-phase_fit["t0_M"]
    kernel = _phase_kernel(u, w, d1, d2)
    phase = phase_fit["phase"]+w*(u+kernel)-phase_fit["frequency_M"]*kernel
    return amplitude, phase


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    inputs = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.q4-public-ringdown-inverse-plan.v1"
            or plan["row_name"] != "q4_mixed_spin" or plan["output_step_M"] != 0.5
            or plan["fit_window_M"] != [0.5, 30]
            or plan["holdout_windows_M"] != [[30, 80], [80, 120]]
            or plan["time_origin_bounds_M"] != [-0.3, 0.0]
            or plan["fit_rms_limit"] != 1e-8
            or plan["holdout_rms_limit"] != 1e-8
            or plan["origin_agreement_limit_M"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in inputs.items()} != plan["input_sha256"]
            or {path: q.sha256(q.ROOT / path) for path in plan["source_sha256"]}
               != plan["source_sha256"]):
        parser.error("frozen inverse plan or inputs differ")
    campaign, attachment, parity = (json.loads(inputs[name].read_text()) for name in
                                    ("campaign", "attachment", "qnm_parity"))
    row = campaign["rows"][plan["row_name"]]["row"]
    if (attachment["status"] != "diagnostic_complete" or attachment["row"] != row
            or parity["status"] != "diagnostic_complete"
            or parity["rows"][plan["row_name"]]["row"] != row
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned public binary or diagnostic inputs differ")
    dummy = dict(dt_geom=plan["output_step_M"], time=np.array([0., 1.]),
                 mode=np.ones(2, dtype=complex), peak_time_M=0.)
    _, _, rt, rm, mass_seconds = public_mode(row, dummy)
    peak = fitted_peak(rt, rm, "quadratic_power")
    time_M = (rt-peak["time_s"])/mass_seconds
    nu_tensor, chi_tensor = seobnrv4_nqc_parameters(*(
        torch.tensor(row[key], dtype=torch.float64)
        for key in ("mass1", "mass2", "chi1", "chi2")))
    nu, chi = float(nu_tensor), float(chi_tensor)
    qnm = parity["rows"][plan["row_name"]]["public_binary"]
    sigma = torch.tensor(complex(qnm["frequency_M"], -qnm["damping_M"]),
                         dtype=torch.complex128)
    shape = seobnrv4_peak_preserving_shape(
        sigma, seobnrv4_merger_shape(torch.tensor(nu, dtype=torch.float64),
                                      torch.tensor(chi, dtype=torch.float64)))
    shape = tuple(float(x) for x in shape)
    amplitude = _fit(time_M, rm, shape, qnm["frequency_M"], qnm["damping_M"],
                     plan["fit_window_M"], plan["time_origin_bounds_M"], "amplitude")
    phase = _fit(time_M, rm, shape, qnm["frequency_M"], qnm["damping_M"],
                 plan["fit_window_M"], plan["time_origin_bounds_M"], "phase")
    native = attachment["native"]["boundary_inputs"]
    native_peak_M = campaign["rows"][plan["row_name"]]["runs"]["ode_2"]["native"]["peak_time_M"]
    scale = mass_seconds*float(C_SI)/(100*float(MPC_SI))
    scope = dict(q.SCOPE)
    scope["description"] = "Public q4 ringdown shape inversion for diagnosis only; no native fitting or promotion."
    scope["known_limitations"] = [
        "Public postpeak fitting does not reveal the private LAL attachment state uniquely.",
        "The inverse uses a single pinned q4 row and cannot establish domain-wide agreement.",
        "No training, production registration, derivatives, batched execution, or JAX qualification.",
    ]
    output = dict(schema="diffgw.seobnrv4.q4-public-ringdown-inverse.v1",
                  scope=scope, status="diagnostic_complete", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], source_sha256=plan["source_sha256"],
                  lal_binary=campaign["lal_binary"],
                  row=row, public_fitted_peak=peak, shape=list(shape), qnm=qnm,
                  public_amplitude_fit=amplitude, public_phase_fit=phase,
                  native_boundary=dict(
                      time_from_fitted_peak_M=attachment["native"]["boundary_time_M"]
                          -native_peak_M-attachment["sampled_peak_proxies"]["native_fitted_peak_time_s"]/mass_seconds,
                      amplitude=scale*native["amplitude"], frequency_M=native["frequency"]),
                  holdout=[])
    for lo, hi in plan["holdout_windows_M"]:
        mask = (time_M > lo) & (time_M <= hi)
        predicted_amp, predicted_phase = _predictions(
            time_M[mask], amplitude, phase, shape, qnm["frequency_M"], qnm["damping_M"])
        observed_phase = -np.unwrap(np.angle(rm[mask]))
        phase_error = np.angle(np.exp(1j*(observed_phase-predicted_phase)))
        output["holdout"].append(dict(window_M=[lo, hi], samples=int(mask.sum()),
            log_amplitude_rms=float(np.sqrt(np.mean((np.log(abs(rm[mask]))-np.log(predicted_amp))**2))),
            phase_rms_rad=float(np.sqrt(np.mean(phase_error**2)))))
    if (amplitude["fit_rms"] >= plan["fit_rms_limit"]
            or phase["fit_rms"] >= plan["fit_rms_limit"]
            or abs(amplitude["t0_M"]-phase["t0_M"]) >= plan["origin_agreement_limit_M"]
            or any(item["log_amplitude_rms"] >= plan["holdout_rms_limit"]
                   or item["phase_rms_rad"] >= plan["holdout_rms_limit"]
                   for item in output["holdout"])):
        raise ValueError("public q4 ringdown ansatz failed frozen fit or holdout limits")
    output["ansatz_holdout_passed"] = True
    q.write_evidence(args.json_out, output)
    print(json.dumps({k: output[k] for k in ("status", "public_amplitude_fit", "public_phase_fit", "native_boundary", "holdout")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
