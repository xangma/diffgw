"""Post-hoc binary witnesses for the dense-spin QNM discrepancy.

The pinned binary is a historical comparator. The high-q rows are outside
the explicitly NR-calibrated configurations in Bohé et al. (2017).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lal
import lalsimulation as lalsim
import torch

from . import qualify_generator as q
from torchwave.approximants.eob.remnant import hofmann_final_spin
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from torchwave.qnm.solver import solve


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/qnm_binary_reachability_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    expected_rows = [
        {"q": 8, "chi1": 0.85, "chi2": 0.85},
        *({"q": ratio, "chi1": 1, "chi2": -1}
          for ratio in (8, 20, 50, 68, 70, 75, 76, 79, 80)),
    ]
    if (plan["schema"] != "diffgw.seobnrv4.qnm-binary-reachability-plan.v1"
            or plan["release_qualified"] is not False
            or plan["lal_version"] != "7.7.1"
            or plan["lalsimulation_version"] != "6.2.1"
            or plan["final_spin_fit_order"] != [3, 4]
            or plan["rows"] != expected_rows
            or plan["remnant_spin_absolute_tolerance"] != 1e-12
            or plan["qnm_dimensionless_absolute_tolerance"] != 6e-7
            or plan["posthoc_solver_control_row_indices"] != [0, 7]
            or plan["solver_control_absolute_tolerance"] != 1e-10
            or plan["pinned_qnm_spin_upper_bound"] != 0.9996):
        parser.error("Frozen binary-reachability plan changed")
    identity = q.lal_binary_identity(plan["lal_version"], plan["lalsimulation_version"])
    rows = []
    for index, inputs in enumerate(plan["rows"]):
        ratio, chi1, chi2 = inputs["q"], inputs["chi1"], inputs["chi2"]
        native_spin = float(hofmann_final_spin(
            float(ratio), 1., float(chi1), float(chi2), fit_order=(3, 4)))
        status, wheel_mass, wheel_spin = lalsim.SimIMREOBFinalMassSpin(
            float(ratio), 1., [0., 0., float(chi1)],
            [0., 0., float(chi2)], lalsim.SEOBNRv4)
        if status != 0:
            raise RuntimeError(f"Pinned wheel remnant call failed: {inputs}")
        wheel_mass, wheel_spin = float(wheel_mass), float(wheel_spin)
        item = {
            "input": inputs, "native_final_spin": native_spin,
            "wheel_final_spin": wheel_spin,
            "spin_absolute_error": abs(native_spin - wheel_spin),
            "wheel_final_mass_fraction": wheel_mass,
            "wheel_qnm_in_domain": bool(wheel_spin <= plan["pinned_qnm_spin_upper_bound"]),
        }
        if item["wheel_qnm_in_domain"]:
            native_frequency, native_damping = (
                float(value) for value in signed_qnm_220(
                    torch.tensor(wheel_spin, dtype=torch.float64)))
            vector = lal.CreateCOMPLEX16Vector(1)
            qnm_status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
                vector, 1., wheel_spin, 2, 2, 1)
            if qnm_status != 0:
                raise RuntimeError(f"Pinned wheel QNM call failed: {inputs}")
            wheel_qnm = vector.data[0] * lal.MTSUN_SI
            wheel_frequency, wheel_damping = float(wheel_qnm.real), float(wheel_qnm.imag)
            item["qnm"] = {
                "native_frequency": native_frequency,
                "native_damping": native_damping,
                "wheel_frequency": wheel_frequency,
                "wheel_damping": wheel_damping,
                "frequency_absolute_error": abs(native_frequency - wheel_frequency),
                "damping_absolute_error": abs(native_damping - wheel_damping),
            }
            if index in plan["posthoc_solver_control_row_indices"]:
                root = solve(wheel_spin, 2, 2, 0,
                             omega_seed=complex(native_frequency, -native_damping),
                             strict=True)
                if not root.converged:
                    raise RuntimeError(f"Original QNM solver did not converge: {inputs}")
                item["qnm"]["native_minus_solver_abs"] = abs(
                    root.omega-complex(native_frequency, -native_damping))
        rows.append(item)
    max_spin_error = max(item["spin_absolute_error"] for item in rows)
    max_qnm_error = max(
        max(item["qnm"]["frequency_absolute_error"],
            item["qnm"]["damping_absolute_error"])
        for item in rows if "qnm" in item)
    max_solver_error = max(
        item["qnm"]["native_minus_solver_abs"]
        for item in rows if "native_minus_solver_abs" in item.get("qnm", {}))
    report = {
        "schema": "diffgw.seobnrv4.qnm-binary-reachability.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(q.SCOPE,
                      description="Post-hoc binary witnesses for QNM component error; no waveform qualification.",
                      known_limitations=[
                          "The high-mass-ratio extreme-spin rows are extrapolations beyond the explicitly NR-calibrated configurations in Bohé et al. (2017).",
                          "These rows witness reachability in the public remnant fit; they do not bound the full input domain or waveform error.",
                          "The pinned wheel's QNM interpolation is a historical comparator, not independent physical truth.",
                      ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "binary": identity, "rows": rows,
        "max_spin_absolute_error": max_spin_error,
        "max_qnm_absolute_error_within_wheel_domain": max_qnm_error,
        "max_native_minus_solver_abs": max_solver_error,
        "gates": {
            "remnant_spin_matches_pinned_binary": max_spin_error < plan["remnant_spin_absolute_tolerance"],
            "qnm_near_pinned_binary_on_all_evaluable_rows": max_qnm_error < plan["qnm_dimensionless_absolute_tolerance"],
            "native_qnm_matches_original_solver_at_control_rows": max_solver_error < plan["solver_control_absolute_tolerance"],
            "all_witnesses_inside_pinned_qnm_domain": all(item["wheel_qnm_in_domain"] for item in rows),
        },
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
