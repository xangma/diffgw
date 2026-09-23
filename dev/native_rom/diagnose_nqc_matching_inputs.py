"""Development-only NQC input attribution for a frozen SEOBNRv4 pair.

Replay two exact-hash waveforms, reconstruct their stored NQC linear systems,
and decompose coefficient drift into RHS and matching-matrix contributions.
This is numerical diagnosis, not model qualification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from . import qualify_generator as q
from .diagnose_trajectory_sensitivity import PAIRS, _array, _hash


def _tolist(value):
    return _array(value).tolist()


def _system(result, targets):
    import torch
    from torchwave.approximants.eob.numerical_attachment import (
        _eval_batch_scalar, natural_cubic_spline,
    )

    fine = result.attachment.fine
    nqc = fine.nqc
    run = result.trajectory.fine
    count = int(run.output_valid.sum().detach())
    time_axis = run.output_time[:count]
    match = fine.match_time
    if nqc.sampled_amplitude.shape != time_axis.shape:
        raise ValueError("fine NQC samples differ from the fine trajectory")

    combined = torch.cat((nqc.sampled_amplitude.unsqueeze(-1),
                          nqc.sampled_products, nqc.sampled_phase_basis), -1)
    spline = natural_cubic_spline(time_axis, combined)
    value = _eval_batch_scalar(spline, match)
    first = _eval_batch_scalar(spline, match, derivative=1)
    second = _eval_batch_scalar(spline, match, derivative=2)
    amplitude_matrix = torch.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_base = torch.stack((value[0], first[0], second[0]))
    amplitude_target = torch.stack(targets[:3])
    amplitude_rhs = amplitude_target - amplitude_base

    # This frozen route passes a local-polynomial diagnostic phase matrix to
    # the attachment, so the ordinary spline derivative is not its matrix.
    window = 0.5
    selected = (time_axis - match).detach().abs() <= window
    chart = (time_axis[selected] - match) / window
    if chart.numel() < 9 or not bool((chart.min() < 0) & (chart.max() > 0)):
        raise ValueError("local NQC fit lacks two-sided samples")
    design = torch.stack([chart.pow(k) for k in range(7)], -1)
    polynomial = torch.linalg.lstsq(
        design, nqc.sampled_phase_basis[selected], driver="gels").solution
    phase_matrix = torch.stack((polynomial[1] / window,
                                2 * polynomial[2] / window**2))
    phase_base = torch.stack((
        _eval_batch_scalar(nqc.waveform_phase_spline, match, derivative=1),
        _eval_batch_scalar(nqc.waveform_phase_spline, match, derivative=2),
    ))
    phase_target = torch.stack(targets[3:])
    phase_rhs = phase_base - phase_target

    systems = {}
    for name, matrix, rhs, base, target, stored in (
        ("amplitude", amplitude_matrix, amplitude_rhs, amplitude_base,
         amplitude_target, fine.calibration.coefficients.amplitude),
        ("phase", phase_matrix, phase_rhs, phase_base, phase_target,
         fine.calibration.coefficients.phase),
    ):
        reconstructed = torch.linalg.solve(matrix, rhs)
        error = (reconstructed - stored).abs().max().item()
        if error > 1e-7:
            raise ValueError(f"{name} system does not recover stored coefficients: {error}")
        systems[name] = dict(matrix=_array(matrix), rhs=_array(rhs),
                             base=_array(base), target=_array(target),
                             coefficients=_array(stored),
                             reconstruction_max_abs_error=error,
                             condition=float(torch.linalg.cond(matrix)),
                             singular_values=_array(torch.linalg.svdvals(matrix)))
    return dict(match_time_local_M=float(match), fine_step_M=float(time_axis[1]-time_axis[0]),
                local_phase_fit_samples=int(chart.numel()), systems=systems)


def _decompose(candidate, reference):
    matrix1, matrix2 = candidate["matrix"], reference["matrix"]
    rhs1, rhs2 = candidate["rhs"], reference["rhs"]
    coeff1, coeff2 = candidate["coefficients"], reference["coefficients"]
    delta_rhs, delta_matrix = rhs2-rhs1, matrix2-matrix1
    rhs_contribution = np.linalg.solve(matrix2, delta_rhs)
    matrix_contribution = -np.linalg.solve(matrix2, delta_matrix @ coeff1)
    observed = coeff2-coeff1
    closure = rhs_contribution+matrix_contribution-observed
    if np.max(np.abs(closure)) > 1e-7:
        raise ValueError(f"matching attribution does not close: {closure}")
    rhs_rows = [np.linalg.solve(matrix2, np.eye(len(rhs2))[k]*delta_rhs[k])
                for k in range(len(rhs2))]
    matrix_rows = [-np.linalg.solve(matrix2,
                    np.eye(len(rhs2))[k]*(delta_matrix[k] @ coeff1))
                   for k in range(len(rhs2))]
    return dict(delta_coefficients=observed.tolist(),
                rhs_contribution=rhs_contribution.tolist(),
                matrix_contribution=matrix_contribution.tolist(),
                rhs_row_contributions=[item.tolist() for item in rhs_rows],
                matrix_row_contributions=[item.tolist() for item in matrix_rows],
                delta_rhs=delta_rhs.tolist(), delta_matrix=delta_matrix.tolist(),
                rhs_contribution_l2=float(np.linalg.norm(rhs_contribution)),
                matrix_contribution_l2=float(np.linalg.norm(matrix_contribution)),
                coefficient_delta_l2=float(np.linalg.norm(observed)),
                closure_max_abs=float(np.max(np.abs(closure))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--pair", choices=PAIRS, default="ode")
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("status") != "failed"
            or campaign.get("row") != campaign.get("result", {}).get("inputs")
            or campaign.get("numerical_choice", {}).get("hermite_resampling") is not True
            or campaign.get("numerical_choice", {}).get("local_nqc_phase_derivatives") is not True
            or campaign.get("numerical_choice", {}).get("fixed_restart_time_M") is None):
        parser.error("expected a frozen Hermite/local-NQC/fixed-restart campaign")
    sources = q.source_snapshot()
    campaign_script = "dev/native_rom/diagnose_hermite_campaign.py"
    sources[campaign_script] = q.sha256(campaign_script)
    if (sources != campaign["source_sha256_before"]
            or q.canonical_hash(sources) != campaign["source_manifest_sha256"]):
        parser.error("generator sources differ from the frozen campaign")

    import torch
    from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
    from torchwave.approximants.eob.nqc import (
        seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
    )

    torch.set_num_threads(1)
    row = campaign["row"]
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    settings = q.Settings(**campaign["settings"])
    stored = {item["name"]: item for item in campaign["result"]["runs"]}
    runs = []
    with torch.no_grad():
        for name in PAIRS[args.pair]:
            prior = stored[name]
            started = time.monotonic()
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **q.CHOICES,
                **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
                diagnostic_hermite_resampling=True,
                diagnostic_local_nqc_derivatives=True,
                diagnostic_restart_time_M=campaign["numerical_choice"]["fixed_restart_time_M"],
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_evaluations=settings.max_evaluations,
                max_rejected_steps=settings.max_rejected_steps,
            )
            hashes = dict(time=_hash(_array(result.time)), mode=_hash(_array(result.mode)))
            if hashes != prior["array_sha256"]:
                raise ValueError(f"{name} replay differs from frozen waveform")
            systems = _system(result, targets)
            runs.append(dict(name=name, level={key: prior[key] for key in
                                               ("atol", "rtol", "dt_geom")},
                             hashes=hashes, **systems,
                             elapsed_seconds=time.monotonic()-started))
            print(json.dumps(dict(run=name, seconds=runs[-1]["elapsed_seconds"],
                                  hashes_verified=True)), flush=True)

    comparison = {name: _decompose(runs[0]["systems"][name], runs[1]["systems"][name])
                  for name in ("amplitude", "phase")}
    for run in runs:
        for system in run["systems"].values():
            for key in ("matrix", "rhs", "base", "target", "coefficients",
                        "singular_values"):
                system[key] = system[key].tolist()
    output = dict(schema="diffgw.seobnrv4.nqc-matching-inputs.v1",
                  scope=q.SCOPE, status="diagnostic_complete",
                  campaign_sha256=q.sha256(args.campaign), script_sha256=q.sha256(__file__),
                  source_manifest_sha256=q.canonical_hash(sources),
                  row=row, pair=args.pair, runs=runs, comparison=comparison,
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(),
                               pid=os.getpid()))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"], pair=args.pair,
                          amplitude=comparison["amplitude"]["delta_coefficients"],
                          phase=comparison["phase"]["delta_coefficients"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
