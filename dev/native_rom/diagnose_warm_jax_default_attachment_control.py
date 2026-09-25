"""Dev-only same-array default Torch/JAX NQC attachment control."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "warm_jax_default_attachment_control_plan.json"
OUTPUT = EVIDENCE / "warm_jax_default_attachment_control.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def relative_l2(left, right):
    return float(np.linalg.norm(left-right)/np.linalg.norm(right))


def metric(left, right):
    left, right = arr(left), arr(right)
    if left.shape != right.shape:
        raise ValueError("attachment comparison charts differ")
    return {"relative_l2": relative_l2(left, right),
            "max_abs": float(np.max(abs(left-right))), "samples": len(left)}


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.warm-jax-default-attachment-plan.v1"
            or plan["release_qualified"] is not False
            or sha(__file__) != plan["script_sha256"]):
        raise ValueError("frozen default attachment runner changed")
    for item in plan["inputs"].values():
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("frozen input changed")
    v1 = json.loads((ROOT/plan["inputs"]["trajectory_v1"]["path"]).read_text())
    v2 = json.loads((ROOT/plan["inputs"]["attachment_v2"]["path"]).read_text())
    if (v1["status"] != "diagnostic_complete"
            or v2["status"] != "diagnostic_complete"
            or tuple(v1["row_tuple"]) != tuple(plan["row_tuple"])):
        raise ValueError("trajectory/attachment controls changed")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, _omega = plan["row_tuple"]
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    ham = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = driver.TorchIntegrator(ham, flux).vector_field
    settings = plan["settings"]
    dt = settings["coarse_output_step_M"]
    state = torch.tensor(v1["warm_state"], dtype=torch.float64)
    result = {
        "schema": "diffgw.seobnrv4.warm-jax-default-attachment-control.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha(PLAN), "script_sha256": sha(__file__),
        "row_tuple": plan["row_tuple"], "settings": settings,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    started = time.monotonic()
    coarse = integrate_adaptive(
        field, state, dt, t_end=settings["max_duration_M"],
        event=AlignedSpinV4Event("coarse"), initial_step=dt,
        max_step=settings["max_step_M"], controller="gsl_state_scaled",
        atol=settings["atol"], rtol=settings["rtol"])
    restart = len(coarse.output_time)-math.ceil(150/dt)
    fine_dt = dt/settings["fine_refinement_factor"]
    fine = integrate_adaptive(
        field, coarse.output_state[restart], fine_dt,
        t_end=settings["max_duration_M"], event=AlignedSpinV4Event("fine"),
        initial_step=fine_dt, max_step=settings["max_step_M"],
        controller="gsl_state_scaled", atol=settings["atol"], rtol=settings["rtol"])
    if (restart != v1["stages"]["coarse"]["torch_restart_index"]
            or len(fine.output_time) != v1["stages"]["fine"]["torch_output_samples"]):
        raise ValueError("Torch fine trajectory does not replay v1")
    ftime, fstate = fine.output_time, fine.output_state
    omega = field(fstate)[:, 1]
    raw, raw_frequency = driver.torch_mode(
        fstate, ham, flux, **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
    tnu, tchi = driver.torch_parameters(*map(tensor, (m1, m2, chi1, chi2)))
    tspin = driver.torch_final_spin(*map(tensor, (m1, m2, chi1, chi2)), fit_order=(3, 4))
    tw, tdamp = driver.torch_qnm(tspin)
    tmass = driver.torch_final_mass(*map(tensor, (m1, m2, chi1, chi2)))
    sigma = torch.complex(tw, -tdamp)/tmass
    match = tensor(v2["stages"]["torch_match_time_M"])
    output = torch.arange(plan["output_samples"], dtype=torch.float64)*fine_dt
    tref = driver.torch_attach(
        ftime, fstate, omega, raw, raw_frequency, tnu, tchi, sigma, output,
        match_time=match)
    jref = driver.jax_attach(
        *map(lambda value: jnp.asarray(arr(value)),
             (ftime, fstate, omega, raw, raw_frequency, tnu, tchi, sigma, output)),
        match_time=jnp.asarray(arr(match)))
    tcoef, jcoef = tref.calibration.coefficients, jref.calibration.coefficients
    tmode, jmode = arr(tref.mode), arr(jref.mode)
    boundary = min(float(tref.boundary_time), float(jref.boundary_time))
    before, after = arr(output) < boundary, arr(output) >= boundary
    result["stages"] = {
        "match_time_abs_M": abs(float(tref.match_time)-float(jref.match_time)),
        "boundary_time_abs_M": abs(float(tref.boundary_time)-float(jref.boundary_time)),
        "boundary_index_abs": abs(int(tref.boundary_index)-int(jref.boundary_index)),
        "coefficient_amplitude_max_abs": float(np.max(abs(arr(tcoef.amplitude)-arr(jcoef.amplitude)))),
        "coefficient_phase_max_abs": float(np.max(abs(arr(tcoef.phase)-arr(jcoef.phase)))),
        "corrected_inspiral": metric(jref.corrected_inspiral, tref.corrected_inspiral),
        "whole_attached_mode": metric(jmode, tmode),
        "pre_boundary_mode": metric(jmode[before], tmode[before]),
        "post_boundary_mode": metric(jmode[after], tmode[after]),
        "same_default_route_vs_asymmetric_v2_ratio":
            metric(jmode, tmode)["relative_l2"]/
            v2["stages"]["whole_attached_mode"]["relative_l2"],
    }
    result["checks"] = {
        "same_boundary": result["stages"]["boundary_time_abs_M"] <= 1e-12
                         and result["stages"]["boundary_index_abs"] == 0,
        "default_attached_mode_parity":
            result["stages"]["whole_attached_mode"]["relative_l2"] <= 1e-8,
        "default_reduces_asymmetric_gap_by_100x":
            result["stages"]["same_default_route_vs_asymmetric_v2_ratio"] <= 0.01,
    }
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
