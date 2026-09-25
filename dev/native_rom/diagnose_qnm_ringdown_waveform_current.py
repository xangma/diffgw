"""Measure a 220 QNM-only substitution on pinned current-source waveforms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_qnm_ringdown_waveform_impact import _array_hash, _relative_l2, _ringdown


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_ringdown_waveform_current_plan.json"


def _pinned(root, name, digest):
    path = root / name
    if q.sha256(path) != digest:
        raise ValueError(f"pinned input changed: {name}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-ringdown-waveform-current-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["level"] != "ode_2"
            or plan["postattachment_window_M"] != [0.0, 100.0]
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False):
        parser.error("frozen plan differs")
    for key in ("baseline_plan", "baseline_evidence", "archive", "campaign",
                "wheel_qnm_evidence", "failed_attempt_plan", "failed_attempt_evidence"):
        _pinned(q.ROOT, plan[key], plan[key+"_sha256"])
    for name, digest in plan["source_sha256"].items():
        _pinned(q.ROOT, name, digest)
    for name, entry in plan["historical_parity_scale"].items():
        _pinned(q.ROOT, entry["path"], entry["sha256"])
    baseline = json.loads((q.ROOT / plan["baseline_evidence"]).read_text())
    wheel = json.loads((q.ROOT / plan["wheel_qnm_evidence"]).read_text())
    failed = json.loads((q.ROOT / plan["failed_attempt_evidence"]).read_text())
    if (baseline["status"] != "diagnostic_complete"
            or baseline["plan_sha256"] != plan["baseline_plan_sha256"]
            or baseline["archive_sha256"] != plan["archive_sha256"]
            or baseline["campaign_sha256"] != plan["campaign_sha256"]
            or baseline["wheel_qnm_evidence_sha256"] != plan["wheel_qnm_evidence_sha256"]
            or wheel["status"] != "diagnostic_complete"
            or failed["status"] != "failed"
            or failed["plan_sha256"] != plan["failed_attempt_plan_sha256"]):
        parser.error("pinned baseline, wheel witness, or failed attempt differs")
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.qnm-ringdown-waveform-current.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  production_change=False, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), source_sha256=plan["source_sha256"],
                  baseline_evidence_sha256=plan["baseline_evidence_sha256"],
                  archive_sha256=plan["archive_sha256"],
                  failed_attempt_evidence_sha256=plan["failed_attempt_evidence_sha256"],
                  historical_scale_is_current_parity=False,
                  rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters
        from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
        from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention

        with np.load(q.ROOT / plan["archive"], allow_pickle=False) as archived:
            if set(archived.files) != {name+"_"+kind for name in plan["rows"]
                                      for kind in ("time", "mode")}:
                raise ValueError("archive members differ from frozen two-row plan")
            for name in plan["rows"]:
                metadata = baseline["rows"][name]
                row = metadata["row"]
                witness = wheel["rows"][name]
                times = archived[name+"_time"]
                original = archived[name+"_mode"]
                if (times.dtype != np.float64 or original.dtype != np.complex128
                        or times.shape != original.shape or times.ndim != 1
                        or not np.all(np.diff(times) > 0)
                        or not np.all(np.isfinite(original))
                        or len(times) != metadata["samples"]
                        or _array_hash(times) != metadata["current_array_sha256"]["time"]
                        or _array_hash(original) != metadata["current_array_sha256"]["mode"]):
                    raise ValueError(f"{name}: archived arrays differ from frozen baseline")
                if (row != witness["row"]
                        or metadata["final_spin"] != witness["final_spin"]
                        or metadata["final_mass_fraction"] != witness["final_mass_fraction"]):
                    raise ValueError(f"{name}: row or remnant differs from wheel witness")
                spin = torch.tensor(metadata["final_spin"], dtype=torch.float64)
                historical = signed_qnm_220_table_convention(spin)
                direct = signed_qnm_220(spin)
                mass = metadata["final_mass_fraction"]
                sigma_h = complex(float(historical[0]/mass), -float(historical[1]/mass))
                sigma_d = complex(float(direct[0]/mass), -float(direct[1]/mass))
                if (abs(sigma_h.real-witness["public_binary"]["frequency_M"]) > 3e-15
                        or abs(-sigma_h.imag-witness["public_binary"]["damping_M"]) > 3e-15
                        or abs(sigma_h.real-metadata["boundary_data"]["qnm_frequency"]) > 3e-15
                        or abs(-sigma_h.imag-metadata["boundary_data"]["qnm_damping"]) > 3e-15):
                    raise ValueError(f"{name}: historical QNM differs from wheel witness")
                boundary = dict(time=metadata["boundary_time_M"],
                                amplitude=metadata["boundary_data"]["amplitude"],
                                phase=metadata["boundary_data"]["phase"],
                                frequency=metadata["boundary_data"]["frequency"])
                m1, m2 = row["mass1"], row["mass2"]
                nu, chi = seobnrv4_nqc_parameters(
                    torch.tensor(m1/(m1+m2), dtype=torch.float64),
                    torch.tensor(m2/(m1+m2), dtype=torch.float64),
                    torch.tensor(row["chi1"], dtype=torch.float64),
                    torch.tensor(row["chi2"], dtype=torch.float64))
                post = times >= boundary["time"]
                window = post & (times <= boundary["time"]+plan["postattachment_window_M"][1])
                if not post.any() or not (~post).any() or window.sum() < 2:
                    raise ValueError(f"{name}: incomplete attachment window")
                kwargs = dict(boundary=boundary, nu=float(nu), chi=float(chi),
                              internal_step=metadata["internal_step_M"],
                              match=metadata["match_time_M"],
                              duration=plan["ringdown_duration_M"])
                historical_post, historical_support = _ringdown(times[post], sigma_h, **kwargs)
                direct_post, direct_support = _ringdown(times[post], sigma_d, **kwargs)
                window_post = times[post] <= boundary["time"]+plan["postattachment_window_M"][1]
                reconstruction = _relative_l2(historical_post[window_post], original[window])
                if reconstruction > plan["baseline_reconstruction_complex_relative_l2_max"]:
                    raise ValueError(f"{name}: fixed-boundary historical reconstruction differs: {reconstruction}")
                counterfactual = original.copy()
                counterfactual[post] = direct_post
                if _array_hash(counterfactual[~post]) != _array_hash(original[~post]):
                    raise ValueError(f"{name}: preattachment mode changed")
                reference, candidate = original[window], counterfactual[window]
                mask = (np.abs(reference) >=
                        plan["phase_amplitude_floor_relative_to_window_peak"]
                        * np.max(np.abs(reference)))
                phase = np.unwrap(np.angle(candidate[mask]*np.conj(reference[mask])))
                full_l2 = _relative_l2(counterfactual, original)
                old_scale = plan["historical_parity_scale"][name]["complex_relative_l2"]
                report["rows"][name] = dict(
                    row=row, samples=dict(full=len(times), postattachment=int(post.sum()),
                                          fixed_window=int(window.sum()), phase=int(mask.sum())),
                    time_M=dict(start=float(times[0]), end=float(times[-1]),
                                boundary=boundary["time"], match=metadata["match_time_M"],
                                fixed_window=[boundary["time"],
                                              boundary["time"]+plan["postattachment_window_M"][1]]),
                    baseline_array_sha256=metadata["current_array_sha256"],
                    preattachment_mode_sha256=_array_hash(original[~post]),
                    fixed_inputs=dict(final_spin=metadata["final_spin"],
                                      final_mass_fraction=mass, boundary=boundary,
                                      internal_step_M=metadata["internal_step_M"],
                                      nu=float(nu), chi=float(chi)),
                    qnm_initial_mass_units=dict(historical_frequency=sigma_h.real,
                                                historical_damping=-sigma_h.imag,
                                                direct_frequency=sigma_d.real,
                                                direct_damping=-sigma_d.imag),
                    support_end_M=dict(historical=historical_support, direct=direct_support),
                    baseline_reconstruction_complex_relative_l2=reconstruction,
                    postattachment=dict(complex_relative_l2=_relative_l2(candidate, reference),
                                        amplitude_relative_l2=_relative_l2(np.abs(candidate), np.abs(reference)),
                                        phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
                                        phase_max_abs_rad=float(np.max(np.abs(phase)))),
                    whole_mode_complex_relative_l2=full_l2,
                    historical_2026_09_24_parity_scale_complex_relative_l2=old_scale,
                    ratio_to_historical_parity_scale=full_l2/old_scale,
                    exceeds_one_percent_of_historical_scale=(full_l2/old_scale >=
                        plan["historical_scale_materiality_fraction"]),
                    exceeds_numerical_reference=(full_l2 >=
                        plan["whole_mode_numerical_reference"]))
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
