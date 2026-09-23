"""Development-only phase/amplitude profile for frozen peak-aligned v4 rows.

Each waveform is anchored to its own local amplitude peak. No relative shift,
phase, amplitude, or model parameter is fitted to the public binary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode


CENTERS_M = tuple(float(value) for value in range(-300, 0, 25)) + (-10., 0.)
HALF_WIDTH_M = 5.


def residual_profile(native_time, native_mode, reference_time, reference_mode,
                     mass_seconds):
    """Measure peak-anchored residual and its exact squared-error split."""
    npk = fitted_peak(native_time, native_mode, "quadratic_power")
    rpk = fitted_peak(reference_time, reference_mode, "quadratic_power")
    nt = (np.asarray(native_time) - npk["time_s"]) / mass_seconds
    rt = (np.asarray(reference_time) - rpk["time_s"]) / mass_seconds
    nm = np.asarray(native_mode) * np.exp(-1j*npk["phase_rad"])
    rm = np.asarray(reference_mode) * np.exp(-1j*rpk["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    time, reference = rt[common], rm[common]
    candidate = np.interp(time, nt, nm.real) + 1j*np.interp(time, nt, nm.imag)
    if time.size < 3:
        raise ValueError("too few common samples")
    active = ((abs(candidate) > max(abs(candidate))*1e-4)
              & (abs(reference) > max(abs(reference))*1e-4))
    if active.sum() < 3:
        raise ValueError("too few samples with measurable phase")
    phase = np.full(time.shape, np.nan)
    phase[active] = np.unwrap(np.angle(candidate[active]*reference[active].conj()))
    amplitude_ratio = abs(candidate)/np.maximum(abs(reference), np.finfo(float).tiny)
    frequency_difference = np.full(time.shape, np.nan)
    frequency_difference[active] = np.gradient(phase[active], time[active], edge_order=2)
    # The identity is pointwise: |a-b|² = (|a|-|b|)² + 2|a||b|(1-cos dphi).
    amplitude_error = (abs(candidate)-abs(reference))**2
    phase_error = 2*abs(candidate)*abs(reference)*(
        1-np.cos(np.angle(candidate*reference.conj())))
    total_error = abs(candidate-reference)**2
    if not np.allclose(amplitude_error+phase_error, total_error,
                       rtol=1e-8, atol=float(np.max(total_error))*1e-12):
        raise ValueError("complex-error decomposition failed")
    bins = []
    for center in CENTERS_M:
        region = ((time >= center-HALF_WIDTH_M) & (time < center+HALF_WIDTH_M)
                  & active)
        if region.sum() < 3:
            continue
        bins.append(dict(center_M=center, samples=int(region.sum()),
                         time_M=[float(time[region][0]), float(time[region][-1])],
                         phase_mean_rad=float(np.mean(phase[region])),
                         phase_rms_rad=float(np.sqrt(np.mean(phase[region]**2))),
                         mode_frequency_difference_mean_per_M=float(np.mean(
                             frequency_difference[region])),
                         amplitude_ratio_mean=float(np.mean(amplitude_ratio[region])),
                         complex_relative_l2=q.mode_metrics(
                             candidate[region], reference[region])["complex_relative_l2"]))
    windows = {}
    for label, region in (("inspiral", time < -20),
                          ("late_inspiral", (time >= -20) & (time < 0)),
                          ("post_peak", time >= 0)):
        if region.sum() < 3:
            continue
        norm = float(np.sum(abs(reference[region])**2))
        windows[label] = dict(samples=int(region.sum()),
                              complex_relative_l2=float(np.sqrt(np.sum(
                                  total_error[region])/norm)),
                              amplitude_error_squared_fraction=float(np.sum(
                                  amplitude_error[region])/np.sum(total_error[region])),
                              phase_error_squared_fraction=float(np.sum(
                                  phase_error[region])/np.sum(total_error[region])))
    return dict(native_peak=npk, reference_peak=rpk, bins=bins, windows=windows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--whole-waveform", required=True, type=Path)
    parser.add_argument("--peak-anchor", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    prior = json.loads(args.whole_waveform.read_text())
    peaks = json.loads(args.peak_anchor.read_text())
    if (campaign.get("schema") != "diffgw.seobnrv4.hermite_local_nqc_fixed_restart_diagnostic.v1"
            or prior.get("status") != "diagnostic_complete"
            or peaks.get("status") != "diagnostic_complete"
            or peaks.get("campaign", {}).get("sha256") != q.sha256(args.campaign)
            or peaks.get("prior_whole_waveform", {}).get("sha256") != q.sha256(args.whole_waveform)
            or peaks.get("lal_binary") != campaign.get("lal_binary")
            or prior.get("rows") != [entry["row"] for entry in peaks["results"]]):
        parser.error("incompatible frozen diagnostic evidence")
    if q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                             campaign["lal_binary"]["lalsimulation_version"]) != campaign["lal_binary"]:
        parser.error("public binary identity changed")
    expected = q.source_snapshot()
    old = campaign["source_sha256_after"]
    changed = [name for name in set(expected) | set(old)
               if expected.get(name) != old.get(name)
               and name != "dev/native_rom/diagnose_hermite_campaign.py"]
    if changed:
        parser.error(f"generator source changed: {changed}")
    settings = q.Settings(**prior["settings"])
    import torch
    torch.set_num_threads(1)
    output = dict(schema="diffgw.seobnrv4.inspiral-residual-diagnostic.v1",
                  scope=q.SCOPE, status="running", whole_waveform_parity_validated=False,
                  campaign_sha256=q.sha256(args.campaign),
                  whole_waveform_sha256=q.sha256(args.whole_waveform),
                  peak_anchor_sha256=q.sha256(args.peak_anchor),
                  source_manifest_sha256=campaign["source_manifest_sha256"],
                  script_sha256=q.sha256(__file__), lal_binary=campaign["lal_binary"],
                  level=prior["level"], settings=asdict(settings),
                  numerical_choice=prior["numerical_choice"],
                  process=dict(host=socket.gethostname(), cwd=str(Path.cwd()),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, output)
    for index, row in enumerate(prior["rows"]):
        print(f"row={index} inspiral_residual", flush=True)
        try:
            choice = prior["numerical_choice"]
            run = q.native_run(row, prior["level"], settings,
                               diagnostic_hermite_resampling=choice["hermite_resampling"],
                               diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                               diagnostic_restart_time_M=choice["fixed_restart_time_M"])
            native = q.summarize_run(run)
            if native["array_sha256"] != peaks["results"][index]["native"]["array_sha256"]:
                raise ValueError("native waveform differs from frozen peak diagnostic")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            metrics = fitted_metrics(nt, nm, rt, rm, mass_seconds, "quadratic_power")
            old_metric = peaks["results"][index]["fitted_peaks"]["quadratic_power"]["metrics"]["complex_relative_l2"]
            if not math.isclose(metrics["metrics"]["complex_relative_l2"], old_metric,
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError("public waveform or peak metric differs from frozen diagnostic")
            output["results"].append(dict(row=row, native=native,
                                          profile=residual_profile(nt, nm, rt, rm, mass_seconds)))
        except Exception as exc:
            output["errors"].append(dict(row=index, type=type(exc).__name__, message=str(exc)))
            break
        q.write_evidence(args.json_out, output)
    output["status"] = "diagnostic_complete" if len(output["results"]) == len(prior["rows"]) else "failed"
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], evidence=str(args.json_out.resolve()))))
    return 0 if output["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
