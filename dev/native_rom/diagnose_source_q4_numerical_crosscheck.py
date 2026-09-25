"""Dev-only crosscheck of output grid versus attachment numerics on q4."""

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
PLAN = EVIDENCE / "source_q4_numerical_crosscheck_plan.json"
OUTPUT = EVIDENCE / "source_q4_numerical_crosscheck.json"


def main():
    plan = json.loads(PLAN.read_text())
    historical_path = EVIDENCE / "source_q4_historical_numerics.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-numerical-crosscheck-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(historical_path) != plan["historical_numerics_sha256"]):
        raise ValueError("frozen numerical-crosscheck inputs changed")
    historical = json.loads(historical_path.read_text())
    old = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics.json").read_text())
    if historical["status"] != "diagnostic_complete":
        raise ValueError("high-accuracy baseline is incomplete")
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != historical["public_binary"]:
        raise ValueError("public binary changed")
    row = historical["row"]
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-q4-numerical-crosscheck.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, interventions={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for item in plan["interventions"]:
        started = time.monotonic()
        level = dict(name=item["name"], dt_geom=item["dt_geom_M"],
                     atol=1e-14, rtol=1e-13)
        run = q.native_run(
            row, level, q.Settings(**old["settings"]),
            diagnostic_spin_truncation="tabulated",
            diagnostic_mode_21_higher_spin="rho_hybrid",
            diagnostic_mode_31_higher_spin="lal_v4_rho",
            diagnostic_mode_32_spin_square="lal_v4_zero",
            diagnostic_mode_33_higher_spin="lal_v4_rho",
            diagnostic_mode_43_higher_spin="lal_v4_rho",
            diagnostic_hamiltonian_log_derivative="lal_v4",
            diagnostic_hamiltonian_calibration="lal_v4_rounded",
            diagnostic_qnm_source="independent_kerr",
            diagnostic_hermite_resampling=item["hermite_resampling"],
            diagnostic_local_nqc_derivatives=item["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=item["matching_second_derivatives"],
            diagnostic_restart_time_M=item["restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
        )
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        fitted = fitted_metrics(nt, nm, rt, rm, mass_seconds, plan["peak_fit"])
        result["interventions"][item["name"]] = dict(
            configuration=item, native=q.summarize_run(run), fitted_public=fitted,
            elapsed_s=time.monotonic()-started)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(item["name"], fitted["metrics"]["complex_relative_l2"],
              flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
