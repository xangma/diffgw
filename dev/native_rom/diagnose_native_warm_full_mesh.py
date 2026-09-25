"""Dev-only detached accepted-mesh discovery from the SEOBNRv4 cap to 0.04."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from dev.native_rom.eob.torch.warm_start import _FrequencyCrossing
from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER,
    integrate_adaptive,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_full_mesh_plan.json"
OUTPUT = PLAN.with_name("native_warm_full_mesh.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen full-mesh script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    for key in ("cap_source", "cap_jacobian_source"):
        reference = plan[key]
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"frozen prior evidence changed: {key}")
    cap_source = json.loads((ROOT / plan["cap_source"]["path"]).read_text())
    case = plan["case"]
    if case["frozen_state"] != cap_source["case"]["frozen_state"]:
        raise ValueError("cap state differs from prior passing tangent evidence")
    if case["state_tangent_per_chi1"] != cap_source["case"]["state_tangent_per_chi1"]:
        raise ValueError("cap tangent differs from prior passing tangent evidence")
    result = {
        "schema": "diffgw.seobnrv4.native-warm-full-mesh.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"],
        "cap_source": plan["cap_source"],
        "cap_jacobian_source": plan["cap_jacobian_source"],
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64)
        m1, m2 = case["mass1"], case["mass2"]
        nu = m1 * m2 / (m1 + m2) ** 2
        delta = (m1 - m2) / (m1 + m2)
        ham = TorchHamiltonian(tensor(nu), tensor(case["chi1"]),
                               tensor(case["chi2"]), delta=tensor(delta))
        field = EOBTrajectoryIntegrator(
            ham, SEOBNRv4PublishedFlux(**SOURCE_FLUX_OPTIONS),
        ).vector_field
        initial = tensor(case["frozen_state"])
        target = tensor(case["target_Momega"])
        settings = plan["solver"]
        initial_frequency = float(field(initial)[1].detach())
        result["initial_frequency"] = initial_frequency
        write(result)
        if not initial_frequency < case["target_Momega"]:
            raise RuntimeError("frozen cap state is not below target frequency")
        run = integrate_adaptive(
            field, initial, settings["output_step_M"],
            event=_FrequencyCrossing(target),
            atol=settings["atol"], rtol=settings["rtol"],
            initial_step=settings["initial_step_M"],
            controller=GSL_STATE_SCALED_CONTROLLER,
            max_accepted_steps=settings["max_accepted_steps"],
            max_rejected_steps=settings["max_rejected_steps"],
            max_evaluations=settings["max_evaluations"],
            max_duration=settings["max_duration_M"],
        )
        times = np.asarray(run.time.detach()).copy()
        states = np.asarray(run.state.detach()).copy()
        derivatives = np.asarray(run.derivative.detach()).copy()
        frequencies = derivatives[:, 1]
        statistics = run.statistics
        accepted = statistics.total_accepted
        rejected = statistics.total_rejected
        evaluations = statistics.total_evaluations
        pre_crossing_steps = accepted - 1
        phase_b_segments = math.ceil(pre_crossing_steps /
                                     plan["checkpoint_planning"]["intervals_per_segment_at_most"])
        result["mesh"] = {
            "accepted_time_M": times.tolist(),
            "accepted_state": states.tolist(),
            "accepted_derivative": derivatives.tolist(),
            "accepted_valid": np.asarray(run.valid.detach()).tolist(),
            "orbital_frequency": frequencies.tolist(),
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "attempted_steps": accepted + rejected,
            "evaluations": evaluations,
            "termination": list(statistics.termination),
            "event_time_M": float(statistics.event_time.detach()),
            "last_bracket_time_M": times[-2:].tolist(),
            "last_bracket_frequency": frequencies[-2:].tolist(),
            "pre_crossing_intervals": pre_crossing_steps,
            "phase_b_segment_count": phase_b_segments,
        }
        write(result)
        gates = {
            "termination_event": statistics.termination == ("event",),
            "accepted_steps_at_most": accepted <= plan["checks"]["accepted_steps_at_most"],
            "duration_M_at_most": float(times[-1]) <= plan["checks"]["duration_M_at_most"],
            "all_prior_accepted_frequencies_below_target": bool(
                np.all(frequencies[:-1] < case["target_Momega"])),
            "last_accepted_frequency_at_least_target": bool(
                frequencies[-1] >= case["target_Momega"]),
            "positive_radius_all_accepted": bool(np.all(states[:, 0] > 0)),
            "finite_and_strictly_increasing_accepted_times": bool(
                np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)),
            "accepted_row_count_equals_steps_plus_one": len(times) == accepted + 1,
            "finite_states_derivatives": bool(
                np.all(np.isfinite(states)) and np.all(np.isfinite(derivatives))),
            "phase_b_segment_count_within_cap": (
                phase_b_segments
                <= plan["checkpoint_planning"]["maximum_pre_crossing_segments"]),
        }
        result["gates"] = gates
        result["diagnostic_checks_passed"] = all(gates.values())
        result["status"] = "diagnostic_complete"
    except Exception:
        result["status"] = "diagnostic_failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(result)


if __name__ == "__main__":
    main()
