"""Development-only attribution of the remaining Hermite v4 waveform error."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from dev.native_rom.diagnose_ode_tolerance import regional_metrics
from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from dev.native_rom.qualify_generator import canonical_hash, compare_runs, sha256, source_snapshot
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets
from torchwave.approximants.eob.numerical_attachment import _eval_batch_scalar


ROW = dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0)
LEVELS = ((1e-11, 1e-12), (1e-12, 1e-13), (1e-13, 1e-14))


def snapshot():
    sources = source_snapshot()
    for path in (Path(__file__).resolve(),
                 Path(regional_metrics.__code__.co_filename).resolve()):
        sources[str(path.relative_to(path.parents[2]))] = sha256(path)
    return canonical_hash(sources)


def array(value):
    return value.detach().cpu().numpy()


def scalar(value):
    data = array(value)
    if data.size != 1:
        raise ValueError("expected scalar diagnostic value")
    return float(data.reshape(-1)[0])


def state_difference(a, b):
    first, second = array(a.output_state), array(b.output_state)
    count = min(len(first), len(second))
    delta = first[:count] - second[:count]
    return dict(common_samples=count,
                phi_max_absolute=float(np.max(abs(delta[:, 1]))),
                phi_rms_absolute=float(np.sqrt(np.mean(delta[:, 1]**2))),
                state_max_absolute=np.max(abs(delta), axis=0).tolist())


def phase_attribution(a, b):
    """Compare raw and NQC-corrected phases on identical fine-grid samples."""
    first, second = a.attachment.fine, b.attachment.fine
    boundary = min(int(first.boundary_index), int(second.boundary_index))
    count = boundary + 1
    first_time = array(a.trajectory.fine.output_time)[:count]
    second_time = array(b.trajectory.fine.output_time)[:count]
    if not np.array_equal(first_time, second_time):
        raise ValueError("phase attribution requires identical fine-grid times")
    first_nqc, second_nqc = first.nqc, second.nqc
    raw_first = array(first_nqc.sampled_waveform_phase)[:count]
    raw_second = array(second_nqc.sampled_waveform_phase)[:count]
    basis_first = array(first_nqc.sampled_phase_basis)[:count]
    basis_second = array(second_nqc.sampled_phase_basis)[:count]
    coefficients_first = array(first.calibration.coefficients.phase)
    coefficients_second = array(second.calibration.coefficients.phase)
    raw_delta = raw_first - raw_second
    shared_delta = raw_delta - (basis_first - basis_second) @ coefficients_first
    coefficient_delta = basis_second @ (coefficients_second - coefficients_first)
    corrected_delta = shared_delta + coefficient_delta
    actual_delta = (array(first_nqc.sampled_corrected_phase)[:count]
                    - array(second_nqc.sampled_corrected_phase)[:count])
    if not np.allclose(actual_delta, corrected_delta, rtol=0, atol=1e-10):
        raise ValueError("corrected phase attribution disagrees with sampled series")
    return dict(
        common_samples=count,
        raw_phase_max_abs_rad=float(np.max(abs(raw_delta))),
        shared_coefficients_max_abs_rad=float(np.max(abs(shared_delta))),
        coefficient_change_max_abs_rad=float(np.max(abs(coefficient_delta))),
        corrected_phase_max_abs_rad=float(np.max(abs(corrected_delta))),
        boundary_raw_phase_difference_rad=float(raw_delta[-1]),
        boundary_shared_coefficients_difference_rad=float(shared_delta[-1]),
        boundary_coefficient_change_difference_rad=float(coefficient_delta[-1]),
        boundary_corrected_phase_difference_rad=float(corrected_delta[-1]),
    )


def phase_matching_system(result, targets):
    fine = result.attachment.fine
    nqc = fine.nqc
    time = fine.match_time
    matrix = np.stack((
        array(_eval_batch_scalar(nqc.phase_basis_spline, time, derivative=1)),
        array(_eval_batch_scalar(nqc.phase_basis_spline, time, derivative=2)),
    ))
    rhs = np.asarray([
        scalar(_eval_batch_scalar(nqc.waveform_phase_spline, time, derivative=1)
               - targets.frequency),
        scalar(_eval_batch_scalar(nqc.waveform_phase_spline, time, derivative=2)
               - targets.frequency_derivative),
    ])
    coefficients = array(fine.calibration.coefficients.phase)
    if not np.allclose(np.linalg.solve(matrix, rhs), coefficients, rtol=0, atol=1e-8):
        raise ValueError("diagnostic phase system differs from calibration")
    return matrix, rhs


def phase_system_attribution(first, second, targets):
    first_matrix, first_rhs = phase_matching_system(first, targets)
    second_matrix, second_rhs = phase_matching_system(second, targets)
    rhs_contribution = np.linalg.solve(first_matrix, second_rhs - first_rhs)
    first_coefficients = array(first.attachment.fine.calibration.coefficients.phase)
    second_coefficients = array(second.attachment.fine.calibration.coefficients.phase)
    matrix_contribution = second_coefficients - first_coefficients - rhs_contribution
    return dict(
        first_matrix=first_matrix.tolist(), second_matrix=second_matrix.tolist(),
        first_rhs=first_rhs.tolist(), second_rhs=second_rhs.tolist(),
        rhs_difference_max_absolute=float(np.max(abs(second_rhs - first_rhs))),
        matrix_difference_max_absolute=float(np.max(abs(second_matrix - first_matrix))),
        coefficient_difference_max_absolute=float(np.max(abs(second_coefficients - first_coefficients))),
        rhs_change_coefficient_contribution=rhs_contribution.tolist(),
        matrix_change_coefficient_contribution=matrix_contribution.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    before = snapshot()
    torch.set_num_threads(1)
    mass1 = torch.tensor(ROW["mass1"], dtype=torch.float64)
    mass2 = torch.tensor(ROW["mass2"], dtype=torch.float64)
    spin1 = torch.tensor(ROW["chi1"], dtype=torch.float64)
    spin2 = torch.tensor(ROW["chi2"], dtype=torch.float64)
    nu, chi = seobnrv4_nqc_parameters(mass1, mass2, spin1, spin2)
    targets = seobnrv4_nqc_targets(nu, chi)
    runs = []
    for rtol, atol in LEVELS:
        started = time.monotonic()
        with torch.no_grad():
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(ROW["omega_start"], dtype=torch.float64),
                ROW["mass1"], ROW["mass2"], chi1=ROW["chi1"], chi2=ROW["chi2"],
                dt_geom=0.5, rtol=rtol, atol=atol,
                diagnostic_hermite_resampling=True,
            )
        runs.append(result)
        print(json.dumps({"rtol": rtol, "seconds": time.monotonic() - started,
                          "restart_time_M": scalar(result.trajectory.restart_time),
                          "peak_time_M": scalar(result.trajectory.peak.time)}), flush=True)
    summaries = []
    comparable = []
    for (rtol, atol), result in zip(LEVELS, runs):
        attachment = result.attachment
        calibration = attachment.fine.calibration
        coefficients = calibration.coefficients
        summaries.append(dict(
            rtol=rtol, atol=atol, restart_index=result.trajectory.restart_index,
            restart_time_M=scalar(result.trajectory.restart_time),
            coarse_samples=len(result.trajectory.coarse.output_time),
            fine_samples=len(result.trajectory.fine.output_time),
            peak_time_M=scalar(result.trajectory.peak.time),
            match_time_M=scalar(attachment.match_time),
            boundary_time_M=scalar(attachment.boundary_time),
            boundary_phase=scalar(attachment.boundary_phase),
            boundary_frequency=scalar(attachment.boundary_frequency),
            match_phase=scalar(attachment.fine.match_phase),
            nqc_amplitude_coefficients=array(coefficients.amplitude).tolist(),
            nqc_phase_coefficients=array(coefficients.phase).tolist(),
            nqc_sampling_condition=scalar(calibration.sampling_condition),
            nqc_amplitude_condition=scalar(calibration.amplitude_condition),
            nqc_phase_condition=scalar(calibration.phase_condition),
        ))
        comparable.append(dict(dt_geom=0.5, time=array(result.time), mode=array(result.mode),
                               peak_time_M=summaries[-1]["peak_time_M"],
                               match_time_M=summaries[-1]["match_time_M"],
                               boundary_time_M=summaries[-1]["boundary_time_M"]))
    comparisons = {}
    for index in range(2):
        first, second = runs[index], runs[index+1]
        key = f"level_{index}_to_{index+1}"
        comparisons[key] = dict(
            coarse_state=state_difference(first.trajectory.coarse, second.trajectory.coarse),
            fine_state=state_difference(first.trajectory.fine, second.trajectory.fine),
            whole=compare_runs(comparable[index], comparable[index+1]),
            regions=regional_metrics(comparable[index], comparable[index+1]),
            nqc_amplitude_coefficient_max_difference=float(np.max(abs(
                np.asarray(summaries[index]["nqc_amplitude_coefficients"])
                - np.asarray(summaries[index+1]["nqc_amplitude_coefficients"])))),
            nqc_phase_coefficient_max_difference=float(np.max(abs(
                np.asarray(summaries[index]["nqc_phase_coefficients"])
                - np.asarray(summaries[index+1]["nqc_phase_coefficients"])))),
            phase_attribution=phase_attribution(first, second),
            phase_system_attribution=phase_system_attribution(first, second, targets),
        )
    after = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.hermite_error_diagnostic.v1",
        scope=dict(physics_validated=False, default_generator_qualified=False,
                   development_only=True, no_coefficient_fitting=True),
        row=ROW, levels=[dict(rtol=r, atol=a) for r, a in LEVELS],
        numerical_choice=dict(hermite_resampling=True, default_restart_rule=True),
        runs=summaries, comparisons=comparisons,
        source_sha256_before=before, source_sha256_after=after,
        sources_unchanged=before == after,
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.json_out}", flush=True)
    if before != after:
        raise RuntimeError("sources changed during the diagnostic")


if __name__ == "__main__":
    main()
