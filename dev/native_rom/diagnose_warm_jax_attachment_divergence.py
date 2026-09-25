"""Dev-only frozen warm-state stagewise Torch/JAX attachment diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from torchwave.approximants.eob import numerical_attachment as torch_attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "warm_jax_attachment_divergence_plan.json"
OUTPUT = EVIDENCE / "warm_jax_attachment_divergence.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def scalar(value):
    return float(arr(value))


def relative_l2(left, right):
    return float(np.linalg.norm(left-right)/np.linalg.norm(right))


def coeffs(value):
    return {"amplitude": arr(value.amplitude).tolist(),
            "phase": arr(value.phase).tolist()}


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.warm-jax-attachment-divergence-plan.v1"
            or plan["release_qualified"] is not False
            or sha(__file__) != plan["script_sha256"]):
        raise ValueError("frozen attachment runner changed")
    for item in plan["inputs"].values():
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("frozen attachment input changed")
    v3 = json.loads((ROOT/plan["inputs"]["warm_whole_v3"]["path"]).read_text())
    v1 = json.loads((ROOT/plan["inputs"]["trajectory_v1"]["path"]).read_text())
    v3_plan = json.loads((ROOT/plan["inputs"]["warm_whole_v3_plan"]["path"]).read_text())
    if (v3["status"] != "diagnostic_overlap_only"
            or v1["status"] != "diagnostic_complete"
            or tuple(v3["row_tuple"]) != tuple(plan["row_tuple"])):
        raise ValueError("prior warm controls changed")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, omega = plan["row_tuple"]
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)
    th = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    warm = np.asarray(v3["field_parity"]["warm"]["state"], dtype=np.float64)
    tstate = torch.as_tensor(warm)
    boundary_plan = json.loads((ROOT/v3_plan["inputs"]["boundary_plan"]["path"]).read_text())
    native_source_plan = json.loads((ROOT/boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT/v3_plan["inputs"]["ladder_plan"]["path"]).read_text())
    level = next(row for row in ladder["levels"] if row["name"] == "refined")
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    if settings != plan["settings"]:
        raise ValueError("refined numerical settings changed")
    identity = driver._prepare_public(native_source_plan)
    field = _source_field()
    result = {
        "schema": "diffgw.seobnrv4.warm-jax-attachment-divergence.v1",
        "release_qualified": False, "status": "running", "host": platform.node(),
        "cwd": os.getcwd(), "pid": os.getpid(), "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__), "row_tuple": plan["row_tuple"],
        "settings": settings, "stages": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    started = time.monotonic()
    captures = {}
    original_tattach, original_jattach = driver.torch_attach, driver.jax_attach

    def capture_tattach(*args, **kwargs):
        attached = original_tattach(*args, **kwargs)
        captures["torch"] = {"args": args, "kwargs": kwargs, "attached": attached}
        return attached

    def capture_jattach(*args, **kwargs):
        attached = original_jattach(*args, **kwargs)
        captures["jax"] = {"args": args, "kwargs": kwargs, "attached": attached}
        return attached

    def torch_initial(_ham, _flux, frequency):
        if float(frequency) != omega:
            raise ValueError("unexpected Torch initial frequency")
        return tstate[0], tstate[2], tstate[3]

    def jax_initial(_ham, _flux, frequency):
        if float(frequency) != omega:
            raise ValueError("unexpected JAX initial frequency")
        return tuple(jnp.asarray(warm[index]) for index in (0, 2, 3))

    with (patch.object(driver, "torch_attach", capture_tattach),
          patch.object(driver, "jax_attach", capture_jattach),
          patch.object(driver, "find_quasicircular_initial_conditions", torch_initial),
          patch.object(driver, "jax_initial_conditions", jax_initial)):
        diagnostic, series, stage, _inputs, _controls = _capture_level(
            "refined", settings, native_source_plan, identity, boundary_plan,
            field, th, tf, torch_attachment._full_series_nqc,
            torch_attachment._eval_batch_scalar)
    if set(captures) != {"torch", "jax"}:
        raise ValueError("both attachment calls must be captured")
    t, j = captures["torch"], captures["jax"]
    ta, ja = t["attached"], j["attached"]
    targs, jargs = t["args"], j["args"]
    def metric(left, right):
        left, right = arr(left), arr(right)
        if left.shape != right.shape:
            raise ValueError(f"captured stage shapes differ: {left.shape} {right.shape}")
        return {"relative_l2": relative_l2(left, right),
                "max_abs": float(np.max(abs(left-right))), "samples": len(left)}
    tinput, jinput = arr(targs[0]), arr(jargs[0])
    if len(tinput) != len(jinput) or np.max(abs(tinput-jinput)) > 1e-12:
        raise ValueError("fine input chart differs")
    output_t, output_j = arr(targs[8]), arr(jargs[8])
    if len(output_t) != len(output_j) or np.max(abs(output_t-output_j)) > 1e-12:
        raise ValueError("attachment output chart differs")
    boundary = min(scalar(ta.boundary_time), scalar(ja.boundary_time))
    before = output_t < boundary
    after = output_t >= boundary
    tmode, jmode = arr(ta.mode), arr(ja.mode)
    tc, jc = coeffs(ta.calibration.coefficients), coeffs(ja.calibration.coefficients)
    result["stages"] = {
        "input_chart_max_abs_M": float(np.max(abs(tinput-jinput))),
        "input_state": metric(jargs[1], targs[1]),
        "input_orbital_frequency": metric(jargs[2], targs[2]),
        "input_base_mode": metric(jargs[3], targs[3]),
        "input_base_frequency": metric(jargs[4], targs[4]),
        "match_time_abs_M": abs(scalar(ta.match_time)-scalar(ja.match_time)),
        "torch_match_time_M": scalar(ta.match_time),
        "jax_match_time_M": scalar(ja.match_time),
        "torch_diagnostic_control_supplied": {
            "phase_matrix": t["kwargs"].get("diagnostic_phase_matrix") is not None,
            "rhs_derivatives": t["kwargs"].get("diagnostic_matching_rhs_derivatives") is not None},
        "jax_diagnostic_control_supplied": {
            "phase_matrix": j["kwargs"].get("diagnostic_phase_matrix") is not None,
            "rhs_derivatives": j["kwargs"].get("diagnostic_matching_rhs_derivatives") is not None},
        "torch_coefficients": tc, "jax_coefficients": jc,
        "amplitude_coefficient_max_abs": float(np.max(abs(np.asarray(tc["amplitude"])-np.asarray(jc["amplitude"])))),
        "phase_coefficient_max_abs": float(np.max(abs(np.asarray(tc["phase"])-np.asarray(jc["phase"])))),
        "corrected_inspiral": metric(ja.corrected_inspiral, ta.corrected_inspiral),
        "boundary_time_abs_M": abs(scalar(ta.boundary_time)-scalar(ja.boundary_time)),
        "boundary_index_abs": abs(int(arr(ta.boundary_index))-int(arr(ja.boundary_index))),
        "boundary_amplitude_abs": abs(scalar(ta.boundary_amplitude)-scalar(ja.boundary_amplitude)),
        "boundary_frequency_abs": abs(scalar(ta.boundary_frequency)-scalar(ja.boundary_frequency)),
        "whole_attached_mode": metric(jmode, tmode),
        "pre_boundary_mode": metric(jmode[before], tmode[before]),
        "post_boundary_mode": metric(jmode[after], tmode[after]),
        "pre_boundary_samples": int(before.sum()), "post_boundary_samples": int(after.sum()),
    }
    result["checks"] = {
        "warm_state_replay": np.array_equal(warm, np.asarray(v1["warm_state"])),
        "initial_state_parity_replay": diagnostic["initial_state_max_abs"] <= 1e-12,
        "whole_mode_parity_failure_replay": abs(
            diagnostic["whole_mode_relative_l2"]
            -v3["levels"]["refined"]["torch_jax_whole_l2"]) <= 1e-12,
        "input_base_mode_same": result["stages"]["input_base_mode"]["relative_l2"] <= 1e-10,
        "asymmetric_matching_controls":
            all(result["stages"]["torch_diagnostic_control_supplied"].values())
            and not any(result["stages"]["jax_diagnostic_control_supplied"].values()),
    }
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
