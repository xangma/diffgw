"""Held-out public q8 ringdown inversion against the frozen native boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_peak, public_mode
from .diagnose_q4_public_ringdown_inverse import _fit, _predictions
from torchwave.approximants.eob.merger_ringdown import (
    seobnrv4_merger_shape, seobnrv4_peak_preserving_shape,
)
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters
from torchwave.constants import C_SI, MPC_SI


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_public_ringdown_inverse_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.q8-public-ringdown-inverse-plan.v1"
            or plan["row_name"] != "q8_high_aligned" or plan["output_step_M"] != 0.5
            or plan["fit_window_M"] != [0.5, 30]
            or plan["holdout_windows_M"] != [[30, 80], [80, 120]]
            or plan["time_origin_bounds_M"] != [-0.5, 0.0]
            or plan["fit_rms_limit"] != 1e-8
            or plan["holdout_rms_limit"] != 1e-8
            or plan["origin_agreement_limit_M"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()} != plan["input_sha256"]
            or {path: q.sha256(q.ROOT / path) for path in plan["source_sha256"]}
               != plan["source_sha256"]):
        parser.error("frozen inverse plan or inputs differ")
    campaign, native_result, parity = (json.loads(paths[name].read_text()) for name in
                                       ("campaign", "independent_qnm", "qnm_parity"))
    row = campaign["rows"][plan["row_name"]]["row"]
    pin = campaign["lal_binary"]
    if (native_result["status"] != "diagnostic_complete"
            or native_result["row"] != row or native_result["lal_binary"] != pin
            or parity["status"] != "diagnostic_complete"
            or parity["rows"][plan["row_name"]]["row"] != row
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("pinned q8 inputs or public binary differ")
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
    anchor = native_result["frequency_anchored"]["native_anchor"]
    fitted_native_peak_s = anchor["time_s"]+anchor["peak_lag_M"]*mass_seconds
    trajectory_peak_M = campaign["rows"][plan["row_name"]]["runs"]["ode_2"]["native"]["peak_time_M"]
    native_boundary = native_result["native"]["boundary_inputs"]
    scale = mass_seconds*float(C_SI)/(100*float(MPC_SI))
    scope = dict(q.SCOPE)
    scope["description"] = "Public q8 ringdown shape inversion for diagnosis only; no native fitting or promotion."
    scope["known_limitations"] = [
        "Public postpeak fitting does not reveal the private LAL attachment state uniquely.",
        "The inversion covers one q8 row and compares with an independent-QNM native counterfactual.",
        "No training, production registration, derivatives, batched execution, or JAX qualification.",
    ]
    output = dict(schema="diffgw.seobnrv4.q8-public-ringdown-inverse.v1",
                  scope=scope, status="diagnostic_complete", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], source_sha256=plan["source_sha256"],
                  lal_binary=pin, row=row,
                  public_fitted_peak=peak, shape=list(shape), qnm=qnm,
                  public_amplitude_fit=amplitude, public_phase_fit=phase,
                  native_boundary=dict(
                      time_from_fitted_peak_M=native_result["native"]["boundary_time_M"]
                        -trajectory_peak_M-fitted_native_peak_s/mass_seconds,
                      amplitude=scale*native_boundary["amplitude"],
                      frequency_M=native_boundary["frequency"]), holdout=[])
    for lo, hi in plan["holdout_windows_M"]:
        mask = (time_M > lo) & (time_M <= hi)
        predicted_amp, predicted_phase = _predictions(
            time_M[mask], amplitude, phase, shape, qnm["frequency_M"], qnm["damping_M"])
        phase_error = np.angle(np.exp(1j*(-np.unwrap(np.angle(rm[mask]))-predicted_phase)))
        output["holdout"].append(dict(window_M=[lo, hi], samples=int(mask.sum()),
            log_amplitude_rms=float(np.sqrt(np.mean((np.log(abs(rm[mask]))-np.log(predicted_amp))**2))),
            phase_rms_rad=float(np.sqrt(np.mean(phase_error**2)))))
    if (amplitude["fit_rms"] >= plan["fit_rms_limit"]
            or phase["fit_rms"] >= plan["fit_rms_limit"]
            or abs(amplitude["t0_M"]-phase["t0_M"]) >= plan["origin_agreement_limit_M"]
            or any(item["log_amplitude_rms"] >= plan["holdout_rms_limit"]
                   or item["phase_rms_rad"] >= plan["holdout_rms_limit"]
                   for item in output["holdout"])):
        raise ValueError("public q8 ringdown ansatz failed frozen fit or holdout limits")
    output["ansatz_holdout_passed"] = True
    q.write_evidence(args.json_out, output)
    print(json.dumps({k: output[k] for k in ("status", "public_amplitude_fit", "public_phase_fit", "native_boundary", "holdout")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
