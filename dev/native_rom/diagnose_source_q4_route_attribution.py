"""Dev-only source-backed q4 replay through the previous Torch adaptive route."""

import json
import os
import platform
import time
from pathlib import Path

from dev.native_rom import qualify_generator as qualification
from dev.native_rom.compare_whole_waveforms import compare_public_binary

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_route_attribution_plan.json"
OUTPUT = ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_route_attribution.json"


def main():
    plan = json.loads(PLAN.read_text())
    if plan["schema"] != "diffgw.seobnrv4.source-q4-route-attribution-plan.v1":
        raise ValueError("wrong frozen plan")
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public

    identity = _prepare_public(plan)
    m1, m2, s1, s2, omega = plan["case"]
    row = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2, omega_start=omega)
    n = plan["numerical_settings"]
    level = dict(name="source_q4_route", dt_geom=n["dt_geom_M"],
                 atol=n["atol"], rtol=n["rtol"])
    settings = qualification.Settings(ringdown_duration_M=n["ringdown_duration_M"])
    started = time.monotonic()
    run = qualification.native_run(
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
    )
    comparison = compare_public_binary(row, run)
    prior = json.loads((ROOT / "dev/native_rom/evidence/2026-09-25/"
                       "jax_source_spliced_whole_waveform_q4_nonspinning.json").read_text())
    result = dict(
        schema="diffgw.seobnrv4.source-q4-route-attribution.v1",
        status="diagnostic_complete", release_qualified=False,
        host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
        plan_sha256=qualification.sha256(PLAN), public_binary=identity,
        previous_route=qualification.summarize_run(run),
        previous_route_public=comparison,
        new_spliced_route_public=prior["torch_public"],
        new_spliced_route_evidence_sha256=qualification.sha256(
            ROOT / "dev/native_rom/evidence/2026-09-25/"
            "jax_source_spliced_whole_waveform_q4_nonspinning.json"),
        elapsed_s=time.monotonic()-started,
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(previous_public_complex_l2=comparison[
        "amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
                          new_spliced_public_complex_l2=prior["torch_public"][
        "amplitude_peak_anchored"]["metrics"]["complex_relative_l2"])), flush=True)


if __name__ == "__main__":
    main()
