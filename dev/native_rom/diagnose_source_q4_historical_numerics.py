"""Dev-only source-backed q4 replay with frozen historical numerics."""

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
PLAN = EVIDENCE / "source_q4_historical_numerics_plan.json"
OUTPUT = EVIDENCE / "source_q4_historical_numerics.json"


def main():
    plan = json.loads(PLAN.read_text())
    warm_path = EVIDENCE / "source_q4_warm_start.json"
    historical_path = ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-historical-numerics-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(warm_path) != plan["warm_start_result_sha256"]
            or q.sha256(historical_path) != plan["historical_numerics_result_sha256"]):
        raise ValueError("frozen historical-numerics inputs changed")
    warm = json.loads(warm_path.read_text())
    historical = json.loads(historical_path.read_text())
    old = historical["rows"]["mass_ratio_four_nonspinning"]["runs"][plan["historical_run"]]
    n = historical["numerical_choice"]
    if (not historical["rows"]["mass_ratio_four_nonspinning"]["check"]
            ["numerical_check_passed"]
            or warm["status"] != "diagnostic_complete"
            or old["level"] != dict(name=plan["historical_run"],
                                      dt_geom=plan["output_step_M"],
                                      atol=plan["atol"], rtol=plan["rtol"])
            or n["fixed_restart_time_M"] != plan["restart_time_M"]
            or n["hermite_resampling"] != plan["hermite_resampling"]
            or n["local_nqc_phase_derivatives"] != plan["local_nqc_phase_derivatives"]
            or historical["settings"]["ringdown_duration_M"]
                != plan["ringdown_duration_M"]):
        raise ValueError("historical numerical settings changed")
    settings_plan = json.loads((EVIDENCE / "source_q4_route_attribution_plan.json").read_text())
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    if identity != warm["public_binary"] or identity != historical["lal_binary"]:
        raise ValueError("pinned public binary changed")
    row = warm["row"]
    if row != historical["rows"]["mass_ratio_four_nonspinning"]["row"]:
        raise ValueError("q4 row changed")
    result = dict(schema="diffgw.seobnrv4.source-q4-historical-numerics.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), public_binary=identity,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(), row=row)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.set_num_threads(1)
    started = time.monotonic()
    run = q.native_run(
        row, old["level"], q.Settings(**historical["settings"]),
        diagnostic_spin_truncation="tabulated",
        diagnostic_mode_21_higher_spin="rho_hybrid",
        diagnostic_mode_31_higher_spin="lal_v4_rho",
        diagnostic_mode_32_spin_square="lal_v4_zero",
        diagnostic_mode_33_higher_spin="lal_v4_rho",
        diagnostic_mode_43_higher_spin="lal_v4_rho",
        diagnostic_hamiltonian_log_derivative="lal_v4",
        diagnostic_hamiltonian_calibration="lal_v4_rounded",
        diagnostic_qnm_source="independent_kerr",
        diagnostic_hermite_resampling=plan["hermite_resampling"],
        diagnostic_local_nqc_derivatives=plan["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=plan["matching_second_derivatives"],
        diagnostic_restart_time_M=plan["restart_time_M"],
        diagnostic_coarse_initial_step_M=plan["initial_step_M"],
        diagnostic_warm_start_Momega=plan["warm_start_Momega"],
    )
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    result.update(status="diagnostic_complete", native=q.summarize_run(run),
                  fitted_public=fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                               plan["peak_fit"]),
                  elapsed_s=time.monotonic()-started)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["fitted_public"]["metrics"]["complex_relative_l2"], flush=True)


if __name__ == "__main__":
    main()
