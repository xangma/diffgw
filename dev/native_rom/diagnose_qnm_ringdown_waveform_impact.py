"""Fixed-boundary 220 QNM substitution on two frozen development waveforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_ringdown_waveform_impact_plan.json"


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _relative_l2(candidate, reference):
    return float(np.linalg.norm(candidate-reference) / np.linalg.norm(reference))


def _ringdown(time, sigma, boundary, nu, chi, internal_step, match, duration):
    from torchwave.approximants.eob.merger_ringdown import (
        merger_ringdown_22, seobnrv4_merger_shape, seobnrv4_peak_preserving_shape)
    from torchwave.approximants.eob.numerical_attachment import seobnrv4_ringdown_support_samples

    real = lambda value: torch.tensor(value, dtype=torch.float64)
    sigma_tensor = torch.tensor(sigma, dtype=torch.complex128)
    elapsed = real(time-boundary["time"])
    shape = seobnrv4_peak_preserving_shape(sigma_tensor, seobnrv4_merger_shape(real(nu), real(chi)))
    wave = merger_ringdown_22(
        elapsed, sigma_tensor, real(boundary["amplitude"]), real(0),
        real(boundary["frequency"]), real(boundary["phase"]), shape)
    count = seobnrv4_ringdown_support_samples(-sigma_tensor.imag, real(internal_step))
    support_end = float(count * real(internal_step))
    wave = torch.where((elapsed < support_end) & (real(time) <= match+duration),
                       wave, torch.zeros_like(wave))
    return wave.detach().cpu().numpy(), support_end


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-ringdown-waveform-impact-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["level"] != "ode_2"
            or plan["postattachment_window_M"] != [0.0, 100.0]
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False):
        parser.error("frozen plan differs")
    campaign_path = q.ROOT / plan["campaign"]
    wheel_path = q.ROOT / plan["wheel_qnm_evidence"]
    for path, expected in ((campaign_path, plan["campaign_sha256"]),
                           (wheel_path, plan["wheel_qnm_evidence_sha256"])):
        if q.sha256(path) != expected:
            parser.error(f"pinned evidence changed: {path}")
    for name, expected in plan["source_sha256"].items():
        if q.sha256(q.ROOT / name) != expected:
            parser.error(f"pinned source changed: {name}")
    for entry in plan["parity_baselines"].values():
        if q.sha256(q.ROOT / entry["path"]) != entry["sha256"]:
            parser.error("pinned parity baseline changed")
    campaign = json.loads(campaign_path.read_text())
    wheel = json.loads(wheel_path.read_text())
    if (campaign["status"] != "diagnostic_complete"
            or wheel["status"] != "diagnostic_complete"
            or wheel["campaign_sha256"] != plan["campaign_sha256"]
            or wheel["lal_binary"] != campaign["lal_binary"]):
        parser.error("campaign or wheel QNM witness differs")
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.qnm-ringdown-waveform-impact.v1", scope=q.SCOPE,
                  status="running", release_qualified=False, production_change=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  source_sha256=plan["source_sha256"],
                  source_revision=plan["source_revision"],
                  source_qnm_file_sha256=plan["source_qnm_file_sha256"],
                  campaign_sha256=plan["campaign_sha256"],
                  wheel_qnm_evidence_sha256=plan["wheel_qnm_evidence_sha256"],
                  wheel_binary=dict(lal_version=campaign["lal_binary"]["lal_version"],
                                    lalsimulation_version=campaign["lal_binary"]["lalsimulation_version"],
                                    lalsimulation_extension_sha256=campaign["lal_binary"]["binaries"][next(
                                        key for key in campaign["lal_binary"]["binaries"]
                                        if "/lalsimulation/_lalsimulation." in key)]),
                  rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters
        from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
        from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention

        for name in plan["rows"]:
            item = campaign["rows"][name]
            row = item["row"]
            started = time.monotonic()
            run = _replay(row, item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), campaign["choice"],
                          item["fixed_restart_time_M"], return_boundary_data=True)
            summary = q.summarize_run(run)
            expected = item["runs"]["ode_2"]["native"]["array_sha256"]
            if summary["array_sha256"] != expected:
                raise ValueError(f"{name}: baseline waveform differs from frozen campaign")
            witness = wheel["rows"][name]
            if (run["final_spin"] != witness["final_spin"]
                    or run["final_mass_fraction"] != witness["final_mass_fraction"]
                    or row != witness["row"]):
                raise ValueError(f"{name}: pinned remnant or row differs")
            spin = torch.tensor(run["final_spin"], dtype=torch.float64)
            historical = signed_qnm_220_table_convention(spin)
            direct = signed_qnm_220(spin)
            mass = run["final_mass_fraction"]
            sigma_h = complex(float(historical[0]/mass), -float(historical[1]/mass))
            sigma_d = complex(float(direct[0]/mass), -float(direct[1]/mass))
            boundary = dict(time=run["boundary_time_M"],
                            amplitude=run["boundary_data"]["amplitude"],
                            phase=run["boundary_data"]["phase"],
                            frequency=run["boundary_data"]["frequency"])
            if (abs(sigma_h.real-witness["native"]["frequency_M"]) > 3e-15
                    or abs(-sigma_h.imag-witness["native"]["damping_M"]) > 3e-15
                    or abs(sigma_h.real-witness["public_binary"]["frequency_M"]) > 3e-15
                    or abs(-sigma_h.imag-witness["public_binary"]["damping_M"]) > 3e-15):
                raise ValueError(f"{name}: historical QNM differs from pinned wheel")
            m1, m2 = row["mass1"], row["mass2"]
            nu, chi = seobnrv4_nqc_parameters(
                torch.tensor(m1/(m1+m2), dtype=torch.float64),
                torch.tensor(m2/(m1+m2), dtype=torch.float64),
                torch.tensor(row["chi1"], dtype=torch.float64),
                torch.tensor(row["chi2"], dtype=torch.float64))
            times, baseline = run["time"], run["mode"]
            post = times >= boundary["time"]
            if not post.any() or not (~post).any():
                raise ValueError(f"{name}: no pre/post attachment samples")
            kwargs = dict(boundary=boundary, nu=float(nu), chi=float(chi),
                          internal_step=run["internal_step_M"],
                          match=run["match_time_M"],
                          duration=campaign["settings"]["ringdown_duration_M"])
            reconstructed, historical_support = _ringdown(times[post], sigma_h, **kwargs)
            counterfactual_post, direct_support = _ringdown(times[post], sigma_d, **kwargs)
            window = post & (times <= boundary["time"]+plan["postattachment_window_M"][1])
            window_post = times[post] <= boundary["time"]+plan["postattachment_window_M"][1]
            reconstructed_l2 = _relative_l2(reconstructed[window_post], baseline[window])
            if reconstructed_l2 > plan["baseline_reconstruction_complex_relative_l2_max"]:
                raise ValueError(f"{name}: historical postattachment replay did not reproduce baseline: {reconstructed_l2}")
            counterfactual = baseline.copy()
            counterfactual[post] = counterfactual_post
            post_reference, post_direct = baseline[window], counterfactual[window]
            phase_mask = (np.abs(post_reference) >=
                          plan["phase_amplitude_floor_relative_to_window_peak"]
                          * np.max(np.abs(post_reference)))
            phase = np.unwrap(np.angle(post_direct[phase_mask]*np.conj(post_reference[phase_mask])))
            full_l2 = _relative_l2(counterfactual, baseline)
            parity = plan["parity_baselines"][name]["complex_relative_l2"]
            fraction = full_l2/parity
            report["rows"][name] = dict(
                row=row, elapsed_seconds=time.monotonic()-started,
                samples=dict(full=len(times), postattachment=int(post.sum()),
                             fixed_window=int(window.sum()), phase=int(phase_mask.sum())),
                time_M=dict(start=float(times[0]), end=float(times[-1]),
                            boundary=boundary["time"], match=run["match_time_M"],
                            postattachment_window=[boundary["time"],
                                                   boundary["time"]+plan["postattachment_window_M"][1]]),
                baseline_array_sha256=summary["array_sha256"],
                preattachment_mode_sha256=_array_hash(baseline[~post]),
                fixed_inputs=dict(final_spin=run["final_spin"],
                                  final_mass_fraction=mass, boundary=boundary,
                                  internal_step_M=run["internal_step_M"],
                                  nu=float(nu), chi=float(chi)),
                qnm_initial_mass_units=dict(historical_frequency=sigma_h.real,
                                            historical_damping=-sigma_h.imag,
                                            direct_frequency=sigma_d.real,
                                            direct_damping=-sigma_d.imag),
                support_end_M=dict(historical=historical_support, direct=direct_support),
                baseline_reconstruction_complex_relative_l2=reconstructed_l2,
                postattachment=dict(complex_relative_l2=_relative_l2(post_direct, post_reference),
                                    amplitude_relative_l2=_relative_l2(np.abs(post_direct), np.abs(post_reference)),
                                    phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
                                    phase_max_abs_rad=float(np.max(np.abs(phase)))),
                whole_mode_complex_relative_l2=full_l2,
                existing_frequency_anchored_parity_complex_relative_l2=parity,
                fraction_of_existing_parity_residual=fraction,
                material_at_predeclared_one_percent=fraction >= plan["whole_mode_materiality_fraction_of_existing_parity_residual"],
                exceeds_numerical_budget_reference=full_l2 >= plan["whole_mode_numerical_budget_reference"])
            q.write_evidence(args.json_out, report)
            print(f"{name} complete", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
