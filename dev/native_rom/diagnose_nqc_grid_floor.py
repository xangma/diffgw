"""Locate the aligned-spin candidate's frozen grid-monotonicity failure.

Every waveform is replayed with its campaign hash. The declared candidate and
qualification gates are unchanged; this is a numerical attribution only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_nqc_matching_inputs import _decompose
from .diagnose_trajectory_sensitivity import _array, _hash, _inspiral, _regions
from .eob.torch.adaptive_driver_clean import (
    _diagnostic_local_matching_second, reconstruct_seobnrv4_22_adaptive,
)


LEVELS = ("ode_2", "grid_1", "grid_2")


def _matching(result, targets):
    from torchwave.approximants.eob.numerical_attachment import (
        _eval_batch_scalar, natural_cubic_spline,
    )

    fine = result.attachment.fine
    nqc = fine.nqc
    run = result.trajectory.fine
    count = int(run.output_valid.sum().detach())
    time_axis = run.output_time[:count]
    match = fine.match_time
    combined = torch.cat((nqc.sampled_amplitude.unsqueeze(-1),
                          nqc.sampled_products, nqc.sampled_phase_basis), -1)
    spline = natural_cubic_spline(time_axis, combined)
    value = _eval_batch_scalar(spline, match)
    first = _eval_batch_scalar(spline, match, derivative=1)
    second = _diagnostic_local_matching_second(time_axis, nqc, match)
    amplitude_matrix = torch.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_base = torch.stack((value[0], first[0], second[0]))
    amplitude_target = torch.stack(targets[:3])
    amplitude_rhs = amplitude_target-amplitude_base

    window = 0.5
    selected = (time_axis-match).detach().abs() <= window
    chart = (time_axis[selected]-match)/window
    design = torch.stack([chart.pow(k) for k in range(7)], -1)
    polynomial = torch.linalg.lstsq(
        design, nqc.sampled_phase_basis[selected], driver="gels").solution
    phase_matrix = torch.stack((polynomial[1]/window,
                                2*polynomial[2]/window**2))
    phase_base = torch.stack((
        _eval_batch_scalar(nqc.waveform_phase_spline, match, derivative=1),
        second[4],
    ))
    phase_target = torch.stack(targets[3:])
    phase_rhs = phase_base-phase_target
    systems = {}
    for name, matrix, rhs, base, target, stored in (
        ("amplitude", amplitude_matrix, amplitude_rhs, amplitude_base,
         amplitude_target, fine.calibration.coefficients.amplitude),
        ("phase", phase_matrix, phase_rhs, phase_base,
         phase_target, fine.calibration.coefficients.phase),
    ):
        residual = float((torch.linalg.solve(matrix, rhs)-stored).abs().max())
        if residual > 1e-8:
            raise ValueError(f"{name} candidate matching reconstruction differs: {residual}")
        systems[name] = dict(matrix=_array(matrix), rhs=_array(rhs),
                             base=_array(base), target=_array(target),
                             coefficients=_array(stored),
                             reconstruction_max_abs_error=residual,
                             condition=float(torch.linalg.cond(matrix)))
    return dict(match_time_local_M=float(match), fine_step_M=float(time_axis[1]-time_axis[0]),
                local_fit_samples=int(chart.numel()),
                matching_second=_array(second).tolist(), systems=systems)


def _replay(row, prior, settings, targets):
    from torchwave.approximants.eob.adaptive_driver_clean import _apply_calibrated_nqc
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES, **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.,
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps)
        hashes = dict(time=_hash(_array(result.time)), mode=_hash(_array(result.mode)))
        if hashes != prior["array_sha256"]:
            raise ValueError(f"{prior['name']} replay differs from frozen waveform")
        m1, m2 = row["mass1"], row["mass2"]
        ham = SEOBNRv4Hamiltonian(
            torch.tensor(m1*m2/(m1+m2)**2, dtype=torch.float64),
            torch.tensor(row["chi1"], dtype=torch.float64),
            torch.tensor(row["chi2"], dtype=torch.float64),
            delta=torch.tensor((m1-m2)/(m1+m2), dtype=torch.float64))
        flux = SEOBNRv4PublishedFlux(
            spin_truncation=q.CHOICES["spin_truncation"],
            retain_pan_cubic_spin=True)
        trajectory = result.trajectory
        valid = _array(trajectory.valid).astype(bool)
        raw_mode = np.zeros_like(_array(result.mode))
        raw, _ = published_mode22_and_frequency(
            trajectory.state[trajectory.valid], ham, flux,
            spin_v6=q.CHOICES["spin_v6"], include_pan_v8=False,
            include_positive_chi_540=False, retain_pan_cubic_spin=True)
        raw_mode[valid] = _array(raw)
        corrected = np.zeros_like(raw_mode)
        corrected[valid] = _array(_apply_calibrated_nqc(
            raw, trajectory.state[trajectory.valid],
            trajectory.frequency[trajectory.valid],
            result.attachment.fine.calibration.coefficients))
        matching = _matching(result, targets)
    return dict(name=prior["name"], dt_geom=prior["dt_geom"],
                time=_array(result.time), mode=_array(result.mode),
                state=_array(trajectory.state), frequency=_array(trajectory.frequency),
                valid=valid, raw_mode=raw_mode, corrected_mode=corrected,
                peak_time_M=float(_array(trajectory.peak.time)),
                match_time_M=float(_array(result.attachment.match_time)),
                boundary_time_M=float(_array(result.attachment.boundary_time)),
                restart_time_M=float(_array(trajectory.restart_time).reshape(-1)[0]),
                matching=matching, hashes=hashes,
                elapsed_seconds=time.monotonic()-started)


def _pair(coarse, fine):
    from torchwave.approximants.eob.nqc import NQCCoefficients, nqc_22_factor

    boundary = fine["boundary_time_M"]
    windows = (("coarse_prefix", 0., 165.),
               ("fine_early", 165., boundary-30.),
               ("fine_late", boundary-30., boundary),
               ("attachment", boundary, boundary+10.),
               ("ringdown", boundary+10., float("inf")))
    pre = _inspiral(coarse, fine, boundary)
    dt = coarse["dt_geom"]
    ratio = round(dt/fine["dt_geom"])
    n = min(len(coarse["time"]), len(fine["time"][::ratio]))
    mask = (coarse["valid"][:n] & fine["valid"][::ratio][:n]
            & (coarse["time"][:n] < boundary))
    raw_coarse = coarse["raw_mode"][:n][mask]
    raw_fine = fine["raw_mode"][::ratio][:n][mask]
    state_coarse = coarse["state"][:n][mask]
    state_fine = fine["state"][::ratio][:n][mask]
    omega_coarse = coarse["frequency"][:n][mask]
    omega_fine = fine["frequency"][::ratio][:n][mask]
    coefficients = fine["matching"]["systems"]
    fixed = NQCCoefficients(
        torch.from_numpy(coefficients["amplitude"]["coefficients"]),
        torch.from_numpy(coefficients["phase"]["coefficients"]))
    with torch.no_grad():
        c_factor = nqc_22_factor(torch.from_numpy(state_coarse[:, 0]),
                                 torch.from_numpy(state_coarse[:, 2]),
                                 torch.from_numpy(omega_coarse), fixed, sample_dims=1)
        f_factor = nqc_22_factor(torch.from_numpy(state_fine[:, 0]),
                                 torch.from_numpy(state_fine[:, 2]),
                                 torch.from_numpy(omega_fine), fixed, sample_dims=1)
    fixed_coarse = raw_coarse*_array(c_factor)
    fixed_fine = raw_fine*_array(f_factor)
    native_coarse = coarse["corrected_mode"][:n][mask]
    native_fine = fine["corrected_mode"][::ratio][:n][mask]
    return dict(whole=q.compare_runs(coarse, fine),
                regions=_regions(coarse, fine, windows),
                pre_attachment=pre,
                corrected_pre_boundary=q.mode_metrics(native_coarse, native_fine),
                common_coefficients_pre_boundary=q.mode_metrics(fixed_coarse, fixed_fine),
                coefficient_only_at_coarse_state=q.mode_metrics(native_coarse, fixed_coarse),
                matching={name: _decompose(coarse["matching"]["systems"][name],
                                           fine["matching"]["systems"][name])
                          for name in ("amplitude", "phase")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("schema") != "diffgw.seobnrv4.nqc-second-derivative-campaign.v1"
            or not campaign.get("sources_unchanged")
            or campaign["rows"]["aligned_spin"]["checks"]["output_grid"]["gates"]["error_decreased"] is not False):
        parser.error("expected the frozen aligned-spin candidate grid failure")
    sources = q.source_snapshot()
    if any(campaign["source_sha256_before"].get(key) != value
           for key, value in sources.items()):
        parser.error("generator sources differ from frozen campaign")
    torch.set_num_threads(1)
    from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets
    row = campaign["rows"]["aligned_spin"]["inputs"]
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    stored = {run["name"]: run for run in campaign["rows"]["aligned_spin"]["runs"]}
    output = dict(schema="diffgw.seobnrv4.nqc-grid-floor.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             numerical_generator_qualified=False),
                  status="running", campaign_sha256=q.sha256(args.campaign),
                  script_sha256=q.sha256(__file__), source_sha256_before=sources,
                  process=dict(argv=sys.argv, cwd=os.getcwd(),
                               host=socket.gethostname(), pid=os.getpid()),
                  runs=[], pairs={})
    q.write_evidence(args.json_out, output)
    runs = []
    for name in LEVELS:
        run = _replay(row, stored[name], settings, targets)
        runs.append(run)
        output["runs"].append(copy.deepcopy({
            key: value for key, value in run.items()
            if key not in ("time", "mode", "state", "frequency", "valid",
                           "raw_mode", "corrected_mode")}))
        for system in output["runs"][-1]["matching"]["systems"].values():
            for key in ("matrix", "rhs", "base", "target", "coefficients"):
                system[key] = system[key].tolist()
        q.write_evidence(args.json_out, output)
        print(json.dumps(dict(run=name, hashes_verified=True,
                              elapsed_seconds=run["elapsed_seconds"])), flush=True)
    for coarse, fine in zip(runs, runs[1:]):
        output["pairs"][f"{coarse['name']}_to_{fine['name']}"] = _pair(coarse, fine)
    output["source_sha256_after"] = q.source_snapshot()
    output["sources_unchanged"] = sources == output["source_sha256_after"]
    output["status"] = "diagnostic_complete" if output["sources_unchanged"] else "failed"
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0 if output["sources_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
