"""Dev-only source q4 local-NQC and fine-RKF numerical attribution."""

import json
import os
import platform
import time
from pathlib import Path

import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, public_mode

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_numerical_attribution_plan.json"
OUTPUT = EVIDENCE / "source_q4_numerical_attribution.json"


def main():
    plan = json.loads(PLAN.read_text())
    baseline_path = EVIDENCE / "source_q4_route_attribution.json"
    anchor_path = EVIDENCE / "source_spliced_anchor.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-numerical-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["peak_fit"] != "quadratic_power"
            or q.sha256(baseline_path) != plan["baseline_evidence_sha256"]
            or q.sha256(anchor_path) != plan["anchor_evidence_sha256"]):
        raise ValueError("frozen attribution inputs changed")
    baseline = json.loads(baseline_path.read_text())
    anchor = json.loads(anchor_path.read_text())
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    n = settings_plan["numerical_settings"]
    m1, m2, chi1, chi2, omega = settings_plan["case"]
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega)
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != baseline["public_binary"] or identity != anchor["public_binary"]:
        raise ValueError("public binary identity changed")
    level = dict(name="source_q4_numerical_attribution", dt_geom=n["dt_geom_M"],
                 atol=n["atol"], rtol=n["rtol"])
    settings = q.Settings(ringdown_duration_M=n["ringdown_duration_M"])
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-numerical-attribution.v1",
                  status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
                  public_binary=identity, host=platform.node(), cwd=os.getcwd(),
                  pid=os.getpid(), row=row, interventions={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for intervention in plan["interventions"]:
        started = time.monotonic()
        run = q.native_run(
            row, level, settings, diagnostic_spin_truncation="tabulated",
            diagnostic_mode_21_higher_spin="rho_hybrid",
            diagnostic_mode_31_higher_spin="lal_v4_rho",
            diagnostic_mode_32_spin_square="lal_v4_zero",
            diagnostic_mode_33_higher_spin="lal_v4_rho",
            diagnostic_mode_43_higher_spin="lal_v4_rho",
            diagnostic_hamiltonian_log_derivative="lal_v4",
            diagnostic_hamiltonian_calibration="lal_v4_rounded",
            diagnostic_qnm_source="independent_kerr",
            diagnostic_fine_max_step_M=n["fine_max_step_M"],
            diagnostic_coarse_initial_step_M=n["coarse_initial_step_M"],
            diagnostic_local_nqc_derivatives=intervention["local_nqc_phase_derivatives"],
            diagnostic_fine_rkf45_resampling=intervention["fine_rkf45_resampling"],
        )
        native = q.summarize_run(run)
        if (intervention["name"] == "baseline"
                and native["array_sha256"] != baseline["previous_route"]["array_sha256"]):
            raise ValueError("baseline waveform differs from frozen q4 route")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        if (intervention["name"] == "baseline"
                and abs(fitted["metrics"]["complex_relative_l2"]
                        - anchor["cases"]["q4_nonspinning"]["fitted"][plan["peak_fit"]]
                        ["metrics"]["complex_relative_l2"]) > 1e-12):
            raise ValueError("baseline fitted comparison changed")
        result["interventions"][intervention["name"]] = dict(
            configuration=intervention, native=native, fitted_public=fitted,
            elapsed_s=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(intervention["name"], fitted["metrics"]["complex_relative_l2"],
              flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
