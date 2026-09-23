"""Development-only check of sampled versus sub-sample amplitude-peak anchoring.

The input comparison and numerical campaign freeze the waveform choices.
Local fits locate each waveform's own amplitude peak; no relative time, phase,
amplitude, or model parameter is optimized against the public binary.
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
from .compare_whole_waveforms import aligned_metrics


METHODS = ("quadratic_power", "quadratic_amplitude", "quartic_power")


def fitted_peak(time, mode, method):
    """Return local amplitude-peak time, phase and offset in grid samples."""
    t, h = np.asarray(time), np.asarray(mode)
    if (method not in METHODS or t.ndim != 1 or h.shape != t.shape or t.size < 5
            or not np.isfinite(t).all() or not np.isfinite(h).all()
            or not np.all(np.diff(t) > 0)):
        raise ValueError("expected a supported fit and a finite increasing complex mode")
    i = int(np.argmax(abs(h)))
    if i < 2 or i > t.size - 3:
        raise ValueError("amplitude peak requires two samples on each side")
    if method == "quartic_power":
        y = abs(h[i-2:i+3])**2
        polynomial = np.polynomial.Polynomial.fit(np.arange(-2., 3.), y, 4).convert()
        roots = polynomial.deriv().roots()
        candidates = [float(z.real) for z in roots
                      if abs(z.imag) < 1e-10 and abs(z.real) <= 1]
        offset = max(candidates, key=polynomial) if candidates else 0.
    else:
        y = abs(h[i-1:i+2])
        if method == "quadratic_power":
            y = y**2
        offset = 0.5*(y[0]-y[2])/(y[0]-2*y[1]+y[2])
    offset = float(np.clip(offset, -1., 1.))
    peak_time = float(t[i]+offset*(t[i+1]-t[i]))
    peak_mode = np.interp(peak_time, t, h.real)+1j*np.interp(peak_time, t, h.imag)
    return dict(time_s=peak_time, phase_rad=float(np.angle(peak_mode)),
                offset_samples=offset)


def fitted_metrics(native_time, native_mode, reference_time, reference_mode,
                   mass_seconds, method):
    """Compare complex modes after independent local peak fits."""
    npk = fitted_peak(native_time, native_mode, method)
    rpk = fitted_peak(reference_time, reference_mode, method)
    nt = native_time-npk["time_s"]
    rt = reference_time-rpk["time_s"]
    nm = native_mode*np.exp(-1j*npk["phase_rad"])
    rm = reference_mode*np.exp(-1j*rpk["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t, reference = rt[common], rm[common]
    candidate = np.interp(t, nt, nm.real)+1j*np.interp(t, nt, nm.imag)
    if t.size < 3:
        raise ValueError("too few common peak-anchored samples")
    windows = {}
    for name, mask in (("inspiral", t < -20*mass_seconds),
                       ("late_inspiral", (t >= -20*mass_seconds) & (t < 0)),
                       ("post_peak", t >= 0)):
        if mask.sum() >= 3:
            windows[name] = dict(time_M=[float(t[mask][0]/mass_seconds),
                                         float(t[mask][-1]/mass_seconds)],
                                 metrics=q.mode_metrics(candidate[mask], reference[mask]))
    return dict(native_peak=npk, reference_peak=rpk,
                common_support_M=[float(t[0]/mass_seconds), float(t[-1]/mass_seconds)],
                metrics=q.mode_metrics(candidate, reference), windows=windows)


def public_mode(row, run):
    """Return the same pinned public binary and units as the prior diagnostic."""
    import lal
    import lalsimulation as lalsim
    from scripts.verify_eob_published_waveforms import _invert_22
    from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

    mass_seconds = (row["mass1"]+row["mass2"])*float(MSUN_SEC)
    dt = run["dt_geom"]*mass_seconds
    lower = row["omega_start"]/(math.pi*mass_seconds)
    distance, inclination = 100., 0.8
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"]*lal.MSUN_SI, row["mass2"]*lal.MSUN_SI,
        0., 0., row["chi1"], 0., 0., row["chi2"],
        distance*1e6*lal.PC_SI, inclination, 0., 0., 0., 0.,
        dt, lower, 0., lal.CreateDict(), lalsim.GetApproximantFromString("SEOBNRv4"))
    if len(hp.data.data) != len(hc.data.data) or hp.epoch != hc.epoch or hp.deltaT != hc.deltaT:
        raise ValueError("inconsistent public polarization grids")
    rt = float(hp.epoch)+np.arange(len(hp.data.data))*float(hp.deltaT)
    rm = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), inclination)
    nt = (run["time"]-run["peak_time_M"])*mass_seconds
    nm = run["mode"]*mass_seconds*float(C_SI)/(distance*float(MPC_SI))
    return nt, nm, rt, rm, mass_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--whole-waveform", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    prior = json.loads(args.whole_waveform.read_text())
    if (campaign.get("schema") != "diffgw.seobnrv4.hermite_local_nqc_fixed_restart_diagnostic.v1"
            or not campaign.get("sources_unchanged")
            or campaign.get("status") not in {"diagnostic_numerical_gates_passed", "failed"}
            or prior.get("schema") != "diffgw.whole-waveform-diagnostic.v1"
            or prior.get("status") != "diagnostic_complete"
            or prior.get("prerequisite_sha256") != q.sha256(args.campaign)
            or prior.get("rows") != [item["row"] for item in prior["results"]]
            or prior.get("numerical_choice") != campaign.get("numerical_choice")):
        parser.error("incompatible frozen campaign or whole-waveform evidence")
    expected_sources = q.source_snapshot()
    script = q.ROOT / "dev/native_rom/diagnose_hermite_campaign.py"
    expected_sources[str(script.relative_to(q.ROOT))] = q.sha256(script)
    changed_sources = [name for name in set(expected_sources) | set(campaign["source_sha256_after"])
                       if expected_sources.get(name) != campaign["source_sha256_after"].get(name)]
    # The older nonspinning campaign predates --inputs on its diagnostic
    # runner. That CLI-only edit leaves the frozen generator sources intact.
    if changed_sources not in ([], [str(script.relative_to(q.ROOT))]):
        parser.error("generator sources differ from the frozen campaign")
    pin = campaign["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs from the frozen campaign")
    settings = q.Settings(**prior["settings"])
    level = prior["level"]
    choice = prior["numerical_choice"]
    import torch
    torch.set_num_threads(1)
    output = dict(schema="diffgw.seobnrv4.peak-anchor-diagnostic.v1",
                  scope=q.SCOPE, status="running", whole_waveform_parity_validated=False,
                  campaign=dict(path=str(args.campaign), sha256=q.sha256(args.campaign)),
                  prior_whole_waveform=dict(path=str(args.whole_waveform),
                                            sha256=q.sha256(args.whole_waveform)),
                  source_manifest_sha256=campaign["source_manifest_sha256"],
                  diagnostic_runner_changed_since_campaign=bool(changed_sources),
                  script_sha256=q.sha256(__file__), lal_binary=pin,
                  level=level, settings=asdict(settings), numerical_choice=choice,
                  process=dict(host=socket.gethostname(), cwd=str(Path.cwd()),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, output)
    for index, row in enumerate(prior["rows"]):
        print(f"row={index} peak_anchor", flush=True)
        try:
            run = q.native_run(row, level, settings,
                               diagnostic_hermite_resampling=choice["hermite_resampling"],
                               diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                               diagnostic_restart_time_M=choice["fixed_restart_time_M"])
            native = q.summarize_run(run)
            if native["array_sha256"] != prior["results"][index]["native"]["array_sha256"]:
                raise ValueError("native waveform differs from the prior comparison")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            sampled = aligned_metrics(nt, nm, rt, rm, mass_seconds)
            old = prior["results"][index]["comparison"]["amplitude_peak_anchored"]["metrics"]
            if not math.isclose(sampled["metrics"]["complex_relative_l2"],
                                old["complex_relative_l2"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("sampled-peak metric differs from the prior comparison")
            output["results"].append(dict(row=row, native=native,
                                          sampled_peak=sampled,
                                          fitted_peaks={method: fitted_metrics(
                                              nt, nm, rt, rm, mass_seconds, method)
                                              for method in METHODS}))
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
