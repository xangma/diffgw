"""Development-only decomposition of a frozen SEOBNRv4 numerical failure.

Replay the refined ODE or grid pair from a campaign, verify its waveform
hashes, and compare trajectory, pre-NQC mode, and attached mode separately.
No comparison here qualifies the provisional generator.
"""

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

from . import qualify_generator as q


PAIRS = {"ode": ("ode_1", "ode_2"), "grid": ("grid_1", "grid_2")}


def _array(value):
    return value.detach().cpu().numpy()


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _nested(candidate, reference, key):
    dt = max(candidate["dt_geom"], reference["dt_geom"])
    arrays = []
    for run in (candidate, reference):
        ratio = dt / run["dt_geom"]
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-12):
            raise ValueError("comparison grids are not nested")
        arrays.append(run[key][::round(ratio)])
    return dt, arrays


def _regions(candidate, reference, windows):
    dt, (lhs, rhs) = _nested(candidate, reference, "mode")
    count = max(len(lhs), len(rhs))
    lhs, rhs = (np.pad(value, (0, count-len(value))) for value in (lhs, rhs))
    time_axis = np.arange(count) * dt
    error = abs(lhs-rhs)**2
    power = abs(rhs)**2
    total_error, total_power = error.sum(), power.sum()
    result = {}
    for name, start, end in windows:
        mask = (time_axis >= start) & (time_axis < end)
        e, p = error[mask].sum(), power[mask].sum()
        result[name] = dict(samples=int(mask.sum()),
                            error_energy_fraction=float(e/total_error),
                            reference_energy_fraction=float(p/total_power),
                            local_relative_l2=float(np.sqrt(e/p)) if p else None)
    return result


def _inspiral(candidate, reference, boundary):
    dt, (left_valid, right_valid) = _nested(candidate, reference, "valid")
    _, (left_state, right_state) = _nested(candidate, reference, "state")
    _, (left_frequency, right_frequency) = _nested(candidate, reference, "frequency")
    _, (left_raw, right_raw) = _nested(candidate, reference, "raw_mode")
    count = min(len(left_valid), len(right_valid))
    mask = (left_valid[:count] & right_valid[:count]
            & (np.arange(count)*dt < boundary))
    if not mask.any():
        raise ValueError("no common pre-attachment trajectory")
    delta = left_state[:count][mask] - right_state[:count][mask]
    frequency_delta = left_frequency[:count][mask]-right_frequency[:count][mask]
    raw_left, raw_right = left_raw[:count][mask], right_raw[:count][mask]
    return dict(samples=int(mask.sum()),
                last_common_time_M=float(np.arange(count)[mask][-1]*dt),
                state_max_abs={name: float(abs(delta[:, index]).max())
                               for index, name in enumerate(("r", "phi", "pr_star", "pphi"))},
                state_rms={name: float(np.sqrt(np.mean(delta[:, index]**2)))
                           for index, name in enumerate(("r", "phi", "pr_star", "pphi"))},
                frequency_max_abs=float(abs(frequency_delta).max()),
                raw_mode=q.mode_metrics(raw_left, raw_right))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--pair", required=True, choices=PAIRS)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    if campaign.get("status") != "failed" or len(campaign.get("results", [])) != 1:
        parser.error("expected one failed frozen campaign row")
    sources = q.source_snapshot()
    if (sources != campaign["source_sha256_before"]
            or q.canonical_hash(sources) != campaign["source_manifest_sha256"]):
        parser.error("generator sources differ from the frozen campaign")

    import torch
    from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    torch.set_num_threads(1)
    row = campaign["results"][0]["inputs"]
    settings = q.Settings(**campaign["config"]["settings"])
    stored = {item["name"]: item for item in campaign["results"][0]["runs"]}
    runs = []
    with torch.no_grad():
        for name in PAIRS[args.pair]:
            prior = stored[name]
            level = {key: prior[key] for key in ("name", "atol", "rtol", "dt_geom")}
            started = time.monotonic()
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **(q.CHOICES | {"spin_truncation": "tabulated"}),
                **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_evaluations=settings.max_evaluations,
                max_rejected_steps=settings.max_rejected_steps,
            )
            trajectory = result.trajectory
            time_axis, mode = _array(result.time), _array(result.mode)
            hashes = {"time": _hash(time_axis), "mode": _hash(mode)}
            if hashes != prior["array_sha256"]:
                raise ValueError(f"{name} replay differs from frozen waveform")
            m1, m2 = row["mass1"], row["mass2"]
            nu = m1*m2/(m1+m2)**2
            ham = SEOBNRv4Hamiltonian(
                torch.tensor(nu, dtype=torch.float64),
                torch.tensor(row["chi1"], dtype=torch.float64),
                torch.tensor(row["chi2"], dtype=torch.float64),
                delta=torch.tensor((m1-m2)/(m1+m2), dtype=torch.float64),
            )
            flux = SEOBNRv4PublishedFlux(
                spin_truncation="tabulated",
                retain_pan_cubic_spin=True)
            valid = _array(trajectory.valid).astype(bool)
            raw_mode = np.zeros_like(mode)
            raw, _ = published_mode22_and_frequency(
                trajectory.state[trajectory.valid], ham, flux,
                spin_v6=q.CHOICES["spin_v6"], include_pan_v8=False,
                include_positive_chi_540=False, retain_pan_cubic_spin=True)
            raw_mode[valid] = _array(raw)
            coefficients = result.attachment.fine.calibration.coefficients
            calibration = result.attachment.fine.calibration
            run = dict(**level, time=time_axis, mode=mode,
                       state=_array(trajectory.state), frequency=_array(trajectory.frequency),
                       valid=valid, raw_mode=raw_mode,
                       peak_time_M=float(_array(trajectory.peak.time)),
                       match_time_M=float(_array(result.attachment.match_time)),
                       boundary_time_M=float(_array(result.attachment.boundary_time)),
                       restart_time_M=float(_array(trajectory.restart_time).reshape(-1)[0]),
                       nqc_coefficients=dict(amplitude=_array(coefficients.amplitude).tolist(),
                                             phase=_array(coefficients.phase).tolist()),
                       nqc_matching=dict(
                           sampling_condition=float(_array(calibration.sampling_condition)),
                           amplitude_condition=float(_array(calibration.amplitude_condition)),
                           phase_condition=float(_array(calibration.phase_condition)),
                           amplitude_residual=_array(calibration.amplitude_residual).tolist(),
                           phase_residual=_array(calibration.phase_residual).tolist()),
                       hashes=hashes, elapsed_seconds=time.monotonic()-started)
            runs.append(run)
            print(json.dumps(dict(run=name, seconds=run["elapsed_seconds"],
                                  hashes_verified=True)), flush=True)

    candidate, reference = runs
    boundary = reference["boundary_time_M"]
    restart = reference["restart_time_M"]
    windows = (("coarse_prefix", 0., restart),
               ("fine_early", restart, boundary-30.),
               ("fine_late", boundary-30., boundary),
               ("attachment", boundary, boundary+10.),
               ("ringdown", boundary+10., float("inf")))
    comparison = dict(whole=q.compare_runs(candidate, reference),
                      regions=_regions(candidate, reference, windows),
                      pre_attachment=_inspiral(candidate, reference, boundary))
    output = dict(schema="diffgw.seobnrv4.trajectory-sensitivity.v1",
                  scope=q.SCOPE, status="diagnostic_complete",
                  campaign_sha256=q.sha256(args.campaign), script_sha256=q.sha256(__file__),
                  source_manifest_sha256=q.canonical_hash(sources),
                  row=row, pair=args.pair,
                  levels=[{key: value for key, value in run.items()
                           if key not in ("time", "mode", "state", "frequency", "valid", "raw_mode")}
                          for run in runs],
                  windows_M=[dict(name=name, start=start, end=end if np.isfinite(end) else None)
                             for name, start, end in windows],
                  comparison=comparison,
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(),
                               pid=os.getpid()))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], pair=args.pair,
                          complex_relative_l2=comparison["whole"]["complex_relative_l2"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
