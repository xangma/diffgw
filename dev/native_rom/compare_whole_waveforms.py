"""Development-only, amplitude-peak-anchored whole-waveform LAL diagnostic.

The peak and its complex phase define coordinates; neither shift is optimized.
This diagnostic does not establish SEOBNRv4 fidelity or ROM readiness.
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
import time

import numpy as np

from . import qualify_generator as q


def aligned_metrics(native_time, native_mode, reference_time, reference_mode, mass_seconds):
    """Compare on common support after each waveform's own amplitude peak."""
    nt, nm = np.asarray(native_time), np.asarray(native_mode)
    rt, rm = np.asarray(reference_time), np.asarray(reference_mode)
    if (nt.ndim != 1 or rt.ndim != 1 or nm.shape != nt.shape or rm.shape != rt.shape
            or min(nt.size, rt.size) < 3 or not np.all(np.diff(nt) > 0)
            or not np.all(np.diff(rt) > 0) or not all(np.isfinite(x).all() for x in (nt, nm, rt, rm))
            or max(abs(nm)) <= 0 or max(abs(rm)) <= 0):
        raise ValueError("expected finite, increasing, nonzero complex waveforms")
    ni, ri = int(np.argmax(abs(nm))), int(np.argmax(abs(rm)))
    native_peak_s, reference_peak_s = float(nt[ni]), float(rt[ri])
    native_phase, reference_phase = float(np.angle(nm[ni])), float(np.angle(rm[ri]))
    nt = nt-native_peak_s
    rt = rt-reference_peak_s
    nm = nm*np.exp(-1j*native_phase)
    rm = rm*np.exp(-1j*reference_phase)
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t, reference = rt[common], rm[common]
    if t.size < 3:
        raise ValueError("too few samples on common peak-anchored support")
    candidate = np.interp(t, nt, nm.real) + 1j*np.interp(t, nt, nm.imag)
    windows = {}
    for name, mask in (
        ("inspiral", t < -20*mass_seconds),
        ("late_inspiral", (t >= -20*mass_seconds) & (t < 0)),
        ("post_peak", t >= 0),
    ):
        if mask.sum() >= 3 and min(np.linalg.norm(candidate[mask]), np.linalg.norm(reference[mask])) > 0:
            windows[name] = dict(time_M=[float(t[mask][0]/mass_seconds),
                                         float(t[mask][-1]/mass_seconds)],
                                 metrics=q.mode_metrics(candidate[mask], reference[mask]))
    active = ((abs(candidate) > max(abs(candidate))*1e-4)
              & (abs(reference) > max(abs(reference))*1e-4))
    if active.sum() < 3:
        raise ValueError("too few common samples with measurable phase")
    phase = np.unwrap(np.angle(candidate[active]*reference[active].conj()))
    return dict(
        anchor=dict(native_amplitude_peak_s=native_peak_s,
                    reference_amplitude_peak_s=reference_peak_s,
                    native_peak_phase_rad=native_phase,
                    reference_peak_phase_rad=reference_phase),
        common_support_M=[float(t[0]/mass_seconds), float(t[-1]/mass_seconds)],
        native_support_M=[float(nt[0]/mass_seconds), float(nt[-1]/mass_seconds)],
        reference_support_M=[float(rt[0]/mass_seconds), float(rt[-1]/mass_seconds)],
        common_samples=int(t.size), active_phase_samples=int(active.sum()),
        metrics=q.mode_metrics(candidate, reference), windows=windows,
        phase_drift_rad=dict(start_to_end=float(phase[-1]-phase[0]),
                             range=float(np.ptp(phase))),
    )


def compare_public_binary(row, run):
    """Obtain a pinned public SEOBNRv4 waveform and compare full complex modes."""
    import lal
    import lalsimulation as lalsim
    from scripts.verify_eob_published_waveforms import _invert_22
    from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

    mass_seconds = (row["mass1"]+row["mass2"])*float(MSUN_SEC)
    dt = run["dt_geom"]*mass_seconds
    lower = row["omega_start"]/(math.pi*mass_seconds)
    distance, inclination = 100.0, 0.8
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"]*lal.MSUN_SI, row["mass2"]*lal.MSUN_SI,
        0., 0., row["chi1"], 0., 0., row["chi2"],
        distance*1e6*lal.PC_SI, inclination, 0., 0., 0., 0.,
        dt, lower, 0., lal.CreateDict(), lalsim.GetApproximantFromString("SEOBNRv4"))
    if len(hp.data.data) != len(hc.data.data) or hp.epoch != hc.epoch or hp.deltaT != hc.deltaT:
        raise ValueError("inconsistent LAL polarization grids")
    rt = float(hp.epoch)+np.arange(len(hp.data.data))*float(hp.deltaT)
    rm = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), inclination)
    nt = (run["time"]-run["peak_time_M"])*mass_seconds
    nm = run["mode"]*mass_seconds*float(C_SI)/(distance*float(MPC_SI))
    aligned = aligned_metrics(nt, nm, rt, rm, mass_seconds)
    raw_sample = np.interp(rt, nt, nm.real, left=0., right=0.) + 1j*np.interp(
        rt, nt, nm.imag, left=0., right=0.)
    return dict(api="SimInspiralChooseTDWaveform", approximant="SEOBNRv4",
                delta_t_s=dt, f_lower_Hz=lower, distance_Mpc=distance,
                inclination_rad=inclination, raw_full_support=q.mode_metrics(raw_sample, rm),
                amplitude_peak_anchored=aligned,
                alignment="Each mode's measured amplitude peak is t=0 and its phase at that peak is zero. "
                          "No optimized shift, phase, amplitude scale, or coefficients. "
                          "Common support only; raw full-reference-support metric retained.",
                whole_waveform_parity_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    prerequisite = parser.add_mutually_exclusive_group(required=True)
    prerequisite.add_argument("--qualification", type=Path)
    prerequisite.add_argument("--diagnostic-campaign", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--max-run-seconds", type=float, default=600.)
    args = parser.parse_args(argv)
    prior_path = args.diagnostic_campaign or args.qualification
    prior = json.loads(prior_path.read_text())
    diagnostic = args.diagnostic_campaign is not None
    if diagnostic:
        if (prior.get("schema") != "diffgw.seobnrv4.hermite_local_nqc_campaign_diagnostic.v1"
                or prior.get("status") not in {"diagnostic_numerical_gates_passed", "failed"}
                or not prior.get("sources_unchanged")
                or len(prior.get("result", {}).get("runs", [])) != 5):
            parser.error("the diagnostic numerical campaign is incomplete or sources changed")
        numerical_result = prior["result"]
        numerical_choice = prior["numerical_choice"]
        if not numerical_choice.get("hermite_resampling") or not numerical_choice.get(
                "local_nqc_phase_derivatives"):
            parser.error("the diagnostic campaign has unexpected numerical choices")
    else:
        if (prior.get("status") not in {"numerical_checks_passed", "failed"}
                or not prior.get("sources_unchanged")
                or len(prior.get("results", [{}])[0].get("runs", [])) != 5):
            parser.error("the prerequisite numerical campaign is incomplete or sources changed")
        numerical_result = prior["results"][0]
        numerical_choice = dict(hermite_resampling=False, local_nqc_phase_derivatives=False)
    expected_sources = q.source_snapshot()
    if diagnostic:
        campaign_source = q.ROOT / "dev/native_rom/diagnose_hermite_campaign.py"
        expected_sources[str(campaign_source.relative_to(q.ROOT))] = q.sha256(campaign_source)
    if expected_sources != prior["source_sha256_after"]:
        parser.error("generator sources differ from the numerical campaign")
    if not math.isfinite(args.max_run_seconds) or args.max_run_seconds <= 0:
        parser.error("max-run-seconds must be finite and positive")
    rows = q.validate_rows(json.loads(args.inputs.read_text()))
    if rows[0] != (prior["row"] if diagnostic else prior["config"]["rows"][0]):
        parser.error("the first comparison row must be the converged campaign row")
    pins = (dict(lal=prior["lal_binary"]["lal_version"],
                 lalsimulation=prior["lal_binary"]["lalsimulation_version"])
            if diagnostic else prior["config"]["reference_pins"])
    identity = q.lal_binary_identity(pins["lal"], pins["lalsimulation"])
    if identity != prior["lal_binary"]:
        parser.error("public binary identity differs from the numerical campaign")
    settings = q.Settings(**{**(prior["settings"] if diagnostic else prior["config"]["settings"]),
                             "max_run_seconds": args.max_run_seconds})
    level = dict(name="grid_2", atol=settings.atol_levels[-1],
                 rtol=settings.rtol_levels[-1], dt_geom=settings.dt_levels_M[-1])
    import torch
    torch.set_num_threads(1)
    output = dict(schema="diffgw.whole-waveform-diagnostic.v1", scope=q.SCOPE,
                  whole_waveform_parity_validated=False, prerequisite=str(prior_path),
                  prerequisite_numerical_checks_passed=numerical_result["numerical_checks_passed"],
                  prerequisite_sha256=q.sha256(prior_path),
                  numerical_choice=numerical_choice,
                  source_manifest_sha256=prior["source_manifest_sha256"],
                  script_sha256=q.sha256(__file__), lal_binary=identity,
                  process=dict(host=socket.gethostname(), cwd=str(Path.cwd()), pid=os.getpid(),
                               argv=sys.argv), rows=rows, level=level, settings=asdict(settings),
                  status="running", results=[], errors=[])
    q.write_evidence(args.json_out, output)
    for index, row in enumerate(rows):
        print(f"row={index} whole_waveform", flush=True)
        started = time.monotonic()
        try:
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=numerical_choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical_choice["local_nqc_phase_derivatives"])
            comparison = compare_public_binary(row, run)
            output["results"].append(dict(row=row, native=q.summarize_run(run),
                                          comparison=comparison,
                                          elapsed_seconds=time.monotonic()-started,
                                          numerical_convergence_prechecked=(
                                              index == 0 and numerical_result["numerical_checks_passed"])))
        except Exception as exc:
            output["errors"].append(dict(row=index, type=type(exc).__name__, message=str(exc)))
            break
        q.write_evidence(args.json_out, output)
    output["status"] = "diagnostic_complete" if len(output["results"]) == len(rows) else "failed"
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], evidence=str(args.json_out.resolve()))))
    return 0 if output["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
