"""Dev-only q4 warm-start attribution against a pinned public waveform."""

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
PLAN = EVIDENCE / "source_q4_warm_start_plan.json"
OUTPUT = EVIDENCE / "source_q4_warm_start.json"


def main():
    plan = json.loads(PLAN.read_text())
    numerical_path = EVIDENCE / "source_q4_numerical_attribution.json"
    source_path = EVIDENCE / "source_q4_full_mode_control.json"
    historical_path = ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics_plan.json"
    historical = json.loads(historical_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-warm-start-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(numerical_path) != plan["numerical_attribution_sha256"]
            or q.sha256(source_path) != plan["source_full_mode_control_sha256"]
            or q.sha256(historical_path) != plan["historical_warm_start_plan_sha256"]
            or plan["warm_start_Momega"] != historical["warm_start_Momega"]):
        raise ValueError("frozen warm-start attribution inputs changed")
    numerical = json.loads(numerical_path.read_text())
    source = json.loads(source_path.read_text())
    if (numerical["status"] != "diagnostic_complete"
            or source["status"] != "diagnostic_complete"):
        raise ValueError("prerequisite attribution is incomplete")
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    n = settings_plan["numerical_settings"]
    row = numerical["row"]
    if row != source["row"]:
        raise ValueError("source and native rows differ")
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != numerical["public_binary"] or identity != source["public_binary"]:
        raise ValueError("pinned public binary changed")
    torch.set_num_threads(1)
    level = dict(name="source_q4_warm_start", dt_geom=n["dt_geom_M"],
                 atol=n["atol"], rtol=n["rtol"])
    settings = q.Settings(ringdown_duration_M=n["ringdown_duration_M"])
    result = dict(schema="diffgw.seobnrv4.source-q4-warm-start.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, interventions={})
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
            diagnostic_warm_start_Momega=plan["warm_start_Momega"],
            diagnostic_fine_rkf45_resampling=intervention["fine_rkf45_resampling"],
            return_dynamics=True,
        )
        initial = run.pop("coarse_dynamics")["state"][0].tolist()
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        result["interventions"][intervention["name"]] = dict(
            configuration=intervention, initial_state=dict(zip(
                ("r", "phi", "pr_star", "pphi"), initial)),
            native=q.summarize_run(run), fitted_public=fitted,
            elapsed_s=time.monotonic()-started)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(intervention["name"], fitted["metrics"]["complex_relative_l2"],
              flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
