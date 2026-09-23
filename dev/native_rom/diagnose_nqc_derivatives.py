"""Development-only check of local NQC phase-basis derivative sensitivity.

This compares natural-cubic derivative matching with independent local
polynomial differentiation of the *same* provisional fine-grid samples. It
does not select a new waveform algorithm or claim SEOBNRv4 fidelity.
"""

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

from dev.native_rom.diagnose_hermite_error import ROW, array, phase_matching_system, scalar
from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from dev.native_rom.qualify_generator import canonical_hash, sha256, source_snapshot
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets


LEVELS = ((1e-12, 1e-13), (1e-13, 1e-14))
WINDOWS_M = (0.25, 0.5, 1.0, 2.0)
DEGREES = (4, 6)


def snapshot():
    sources = source_snapshot()
    for path in (Path(__file__).resolve(),
                 Path(phase_matching_system.__code__.co_filename).resolve()):
        sources[str(path.relative_to(path.parents[2]))] = sha256(path)
    return canonical_hash(sources)


def local_polynomial_matrix(nqc, match_time, window_M, degree):
    knots = array(nqc.phase_basis_spline.knots)
    samples = array(nqc.sampled_phase_basis)
    local = abs(knots - match_time) <= window_M
    chart = (knots[local] - match_time) / window_M
    values = samples[local]
    if len(chart) < degree + 3 or not (np.min(chart) < 0 < np.max(chart)):
        raise ValueError("insufficient two-sided samples for local polynomial")
    coefficients = np.polynomial.polynomial.polyfit(chart, values, degree)
    fitted = np.polynomial.polynomial.polyval(chart, coefficients).T
    matrix = np.stack((coefficients[1] / window_M,
                       2 * coefficients[2] / window_M**2))
    return matrix, dict(samples=int(len(chart)),
                        fit_residual_rms=float(np.sqrt(np.mean((values - fitted)**2))),
                        condition=float(np.linalg.cond(matrix)))


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
        print(json.dumps({"rtol": rtol, "seconds": time.monotonic() - started}), flush=True)
        nqc = result.attachment.fine.nqc
        match_time = scalar(result.attachment.fine.match_time)
        native_matrix, rhs = phase_matching_system(result, targets)
        native_coefficients = array(result.attachment.fine.calibration.coefficients.phase)
        methods = {}
        for window in WINDOWS_M:
            for degree in DEGREES:
                matrix, details = local_polynomial_matrix(nqc, match_time, window, degree)
                coefficients = np.linalg.solve(matrix, rhs)
                methods[f"window_{window:g}_degree_{degree}"] = dict(
                    matrix=matrix.tolist(), coefficients=coefficients.tolist(), **details)
        runs.append(dict(rtol=rtol, atol=atol, match_time_M=match_time,
                         restart_time_M=scalar(result.trajectory.restart_time),
                         peak_time_M=scalar(result.trajectory.peak.time),
                         native_matrix=native_matrix.tolist(),
                         native_rhs=rhs.tolist(),
                         native_coefficients=native_coefficients.tolist(),
                         methods=methods))
    comparisons = {}
    first, second = runs
    for method in first["methods"]:
        first_value, second_value = first["methods"][method], second["methods"][method]
        comparisons[method] = dict(
            matrix_max_absolute_difference=float(np.max(abs(
                np.asarray(first_value["matrix"]) - np.asarray(second_value["matrix"])))),
            coefficient_max_absolute_difference=float(np.max(abs(
                np.asarray(first_value["coefficients"])
                - np.asarray(second_value["coefficients"])))),
        )
    comparisons["natural_cubic"] = dict(
        matrix_max_absolute_difference=float(np.max(abs(
            np.asarray(first["native_matrix"]) - np.asarray(second["native_matrix"])))),
        coefficient_max_absolute_difference=float(np.max(abs(
            np.asarray(first["native_coefficients"])
            - np.asarray(second["native_coefficients"])))),
    )
    after = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.nqc_derivative_diagnostic.v1",
        scope=dict(development_only=True, physics_validated=False,
                   numerical_generator_qualified=False,
                   algorithm_choice_made=False, no_coefficient_fitting=True),
        row=ROW, levels=[dict(rtol=r, atol=a) for r, a in LEVELS],
        method=dict(fine_grid_step_M=0.015625, windows_M=WINDOWS_M,
                    polynomial_degrees=DEGREES,
                    same_fine_grid_samples=True, same_phase_rhs=True),
        runs=runs, comparisons=comparisons,
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
