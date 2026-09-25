"""Frozen read-only audit of the v4 cap-frequency validity failure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_native_cap_frequency_v4_audit_plan.json"
OUTPUT = EVIDENCE / "source_aligned_native_cap_frequency_v4_audit.json"


def main() -> None:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.native-cap-frequency-v4-audit-plan.v1"
            or plan["release_qualified"] is not False
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen cap-frequency audit plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen cap-frequency input changed: {name}")
    v4 = json.loads((ROOT / plan["inputs"]["v4_result"]["path"]).read_text())
    v4_plan = json.loads((ROOT / plan["inputs"]["v4_plan"]["path"]).read_text())
    v5 = json.loads((ROOT / plan["inputs"]["v5_result"]["path"]).read_text())
    if (v4["status"] != "failed_validity"
            or v4["release_qualified"] is not False
            or v4["plan_sha256"] != q.sha256(ROOT / plan["inputs"]["v4_plan"]["path"])
            or v4["script_sha256"] != v4_plan["script_sha256"]
            or v5["status"] != "diagnostic_complete"
            or v5["release_qualified"] is not False):
        raise ValueError("v4/v5 frozen status or identity changed")

    cap = np.asarray(v4["cap_state"], dtype=np.float64)
    v5_cap = np.asarray(v5["native_warm_initial_state"], dtype=np.float64)
    source_cap = np.asarray(v5["source_initial_state"], dtype=np.float64)
    circular = cap.copy()
    circular[2] = 0.0
    field = _source_field()
    with torch.no_grad():
        circular_omega = float(field(torch.as_tensor(circular))[1])
        cap_omega = float(field(torch.as_tensor(cap))[1])
    target = float(v4_plan["cap_Momega"])
    observed = {
        "nominal_circular_target_Momega": target,
        "v4_dissipative_cap_Momega": float(v4["cap_orbital_Momega"]),
        "v5_native_dissipative_cap_Momega": float(v5["native_warm_initial_omega_M"]),
        "v5_source_dissipative_cap_Momega": float(v5["source_initial_omega_M"]),
        "reevaluated_v4_dissipative_cap_Momega": cap_omega,
        "zero_radial_momentum_circular_Momega": circular_omega,
        "dissipative_minus_nominal_Momega": cap_omega-target,
        "circular_minus_nominal_Momega": circular_omega-target,
        "v4_minus_v5_native_Momega": cap_omega-v5["native_warm_initial_omega_M"],
        "v4_minus_v5_source_Momega": cap_omega-v5["source_initial_omega_M"],
        "v5_native_minus_source_Momega": (
            v5["native_warm_initial_omega_M"]-v5["source_initial_omega_M"]),
        "v4_v5_native_state_max_abs": float(np.max(abs(cap-v5_cap))),
        "v4_v5_source_state_max_abs": float(np.max(abs(cap-source_cap))),
        "v4_failed_validity_checks": sorted(
            name for name, passed in v4["validity_checks"].items() if not passed),
        "v4_fixed_chart_samples": int(v4["fixed_chart"]["samples"]),
        "v4_hypothesis_passed": bool(v4["hypothesis_passed"]),
    }
    gates = plan["gates"]
    checks = {
        "v4_only_cap_frequency_failed": observed["v4_failed_validity_checks"]
            == ["cap_frequency"],
        "v4_target_gate_failure_replays": abs(observed["dissipative_minus_nominal_Momega"])
            > v4_plan["gates"]["cap_frequency_max_abs"],
        "v4_field_replay": abs(cap_omega-v4["cap_orbital_Momega"])
            <= gates["field_replay_max_abs"],
        "v4_v5_initial_state_replay": observed["v4_v5_native_state_max_abs"]
            <= gates["initial_state_replay_max_abs"],
        "v4_v5_native_frequency_parity": abs(observed["v4_minus_v5_native_Momega"])
            <= gates["dissipative_frequency_parity_max_abs"],
        "v4_v5_source_frequency_parity": abs(observed["v4_minus_v5_source_Momega"])
            <= gates["dissipative_frequency_parity_max_abs"],
        "v5_native_source_frequency_parity": abs(observed["v5_native_minus_source_Momega"])
            <= gates["dissipative_frequency_parity_max_abs"],
        "circular_target_replay": abs(observed["circular_minus_nominal_Momega"])
            <= gates["circular_target_max_abs"],
        "v4_full_fixed_chart_replay": observed["v4_fixed_chart_samples"] == 1169,
        "v4_hypothesis_replay": observed["v4_hypothesis_passed"],
    }
    result = {
        "schema": "diffgw.seobnrv4.native-cap-frequency-v4-audit.v1",
        "status": "audit_complete" if all(checks.values()) else "audit_failed",
        "release_qualified": False,
        "plan_sha256": q.sha256(PLAN),
        "script_sha256": q.sha256(Path(__file__)),
        "observed": observed,
        "checks": checks,
        "future_rerun_criteria": plan["future_rerun_criteria"],
        "interpretation": (
            "The nominal cap frequency is solved at zero radial momentum before "
            "the quasicircular initializer adds dissipative radial momentum. "
            "The original v4 validity gate therefore fails on an expected "
            "frequency shift; this audit does not revise or qualify v4."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "observed": observed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
