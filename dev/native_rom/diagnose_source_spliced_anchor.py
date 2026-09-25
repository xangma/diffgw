"""Dev-only sub-sample and frequency-anchor check of source-backed waveforms."""

import json
import os
import platform
import time
from pathlib import Path

import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import (
    METHODS, aligned_metrics, fitted_metrics, public_mode,
)
from dev.native_rom.diagnose_q8_ringdown_peak_bound import _frequency_metrics

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_spliced_anchor_plan.json"
OUTPUT = EVIDENCE / "source_spliced_anchor.json"


def replay(row, settings):
    level = dict(name="anchor_replay", dt_geom=settings["dt_geom_M"],
                 atol=settings["atol"], rtol=settings["rtol"])
    return q.native_run(
        row, level, q.Settings(ringdown_duration_M=settings["ringdown_duration_M"]),
        diagnostic_spin_truncation="tabulated",
        diagnostic_mode_21_higher_spin="rho_hybrid",
        diagnostic_mode_31_higher_spin="lal_v4_rho",
        diagnostic_mode_32_spin_square="lal_v4_zero",
        diagnostic_mode_33_higher_spin="lal_v4_rho",
        diagnostic_mode_43_higher_spin="lal_v4_rho",
        diagnostic_hamiltonian_log_derivative="lal_v4",
        diagnostic_hamiltonian_calibration="lal_v4_rounded",
        diagnostic_qnm_source="independent_kerr",
        diagnostic_fine_max_step_M=settings["fine_max_step_M"],
        diagnostic_coarse_initial_step_M=settings["coarse_initial_step_M"],
    )


def main():
    plan = json.loads(PLAN.read_text())
    prior_q4 = EVIDENCE / "source_q4_route_attribution.json"
    if (plan["schema"] != "diffgw.seobnrv4.source-spliced-anchor-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["peak_fit_methods"]) != METHODS
            or q.sha256(prior_q4) != plan["source_q4_route_sha256"]):
        raise ValueError("frozen anchor plan or q4 route changed")
    settings_plan = json.loads((EVIDENCE / plan["source_choices"]).read_text())
    settings = settings_plan["numerical_settings"]
    from dev.native_rom.diagnose_jax_spliced_whole_waveform import _prepare_public
    identity = _prepare_public(settings_plan)
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.source-spliced-anchor.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), host=platform.node(), cwd=os.getcwd(),
                  pid=os.getpid(), public_binary=identity, cases={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, expected_hash in plan["cases"].items():
        started = time.monotonic()
        prior_path = EVIDENCE / f"jax_source_spliced_whole_waveform_{name}.json"
        if q.sha256(prior_path) != expected_hash:
            raise ValueError(f"frozen spliced waveform evidence changed: {name}")
        prior = json.loads(prior_path.read_text())
        if (prior["variant"] != "source" or not prior["diagnostic_checks_passed"]
                or prior["public_binary"] != identity):
            raise ValueError(f"spliced diagnostic invalid: {name}")
        m1, m2, chi1, chi2, omega_start = prior["input"]["case_parameters"]
        row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                   omega_start=omega_start)
        run = replay(row, settings)
        if name == "q4_nonspinning":
            original_hash = json.loads(prior_q4.read_text())["previous_route"]["array_sha256"]
            if q.summarize_run(run)["array_sha256"] != original_hash:
                raise ValueError("q4 replay waveform changed")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        sampled = aligned_metrics(nt, nm, rt, rm, mass_seconds)
        fitted = {method: fitted_metrics(nt, nm, rt, rm, mass_seconds, method)
                  for method in METHODS}
        try:
            frequency = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                           plan["frequency_anchor_angular_M"])
        except ValueError as exc:
            frequency = {"available": False, "reason": str(exc)}
        result["cases"][name] = dict(
            row=row, native=q.summarize_run(run), sampled=sampled,
            spliced_sampled_complex_l2=prior["torch_public"]["amplitude_peak_anchored"][
                "metrics"]["complex_relative_l2"],
            fitted=fitted, frequency_anchor=frequency,
            elapsed_s=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, json.dumps(dict(sampled=sampled["metrics"]["complex_relative_l2"],
                                    fitted={k: v["metrics"]["complex_relative_l2"]
                                            for k, v in fitted.items()},
                                    frequency=frequency.get("metrics", {}).get(
                                        "complex_relative_l2", frequency))),
              flush=True)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
