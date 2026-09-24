"""Compare native NQC coefficients with the public SEOBNRv4 diagnostic oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from . import qualify_generator as q
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/nqc_coefficients_plan.json"


def _source_targets(nu, chi):
    """Pinned LAL non-HM v4 (2,2) NQC fit literals, diagnostic only."""
    chi2, chi3 = chi * chi, chi * chi * chi

    def combine(linear, equal_mass, test_particle):
        constant = (-0.00099601593625498 * linear
                    - 0.00001600025600409607 * equal_mass
                    + 1.000016000256004 * test_particle)
        quadratic = (-3.984063745019967 * linear
                     + 16.00025600409607 * equal_mass
                     - 16.0002560041612 * test_particle)
        return constant + linear * nu + quadratic * nu * nu

    tpl = (1.4528573105413543 + 0.16613449160880395 * chi
           + 0.027355646661735258 * chi2 - 0.020072844926136438 * chi3)
    eq = (1.577457498227 - 0.0076949474494639085 * chi
          + 0.02188705616693344 * chi2 + 0.023268366492696667 * chi3)
    linear = (-0.03442402416125921 - 1.218066264419839 * chi
              - 0.5683726304811634 * chi2 + 0.4011143761465342 * chi3)
    amplitude = nu * combine(linear, eq, tpl)

    shifted = chi - 1
    tpl = (0.002395610769995033 * shifted
           - 0.00019273850675004356 * shifted**2
           - 0.00029666193167435337 * shifted**3)
    eq = -0.004126509071377509 + 0.002223999138735809 * chi
    linear = -0.005776537350356959 + 0.001030857482885267 * chi
    curvature = nu * combine(linear, eq, tpl)

    c0, c1 = 0.5626787200433265, -0.08706198756945482
    c2, c3 = 25.81979479453255, 25.85037751197443
    d2, d3 = 7.629921628648589, 10.26207326082448
    a4 = d2 + 4 * (d2 - c2) * (nu - 0.25)
    a3 = d3 + 4 * (d3 - c3) * (nu - 0.25)
    frequency = c0 + (c1 + 0.00174345193125868 * chi) * math.log(a3 - a4 * chi)

    tpl = (-0.011209791668428353
           + (0.0040867958978563915 + 0.0006333925136134493 * chi)
           * math.log(68.47466578100956 - 58.301487557007206 * chi))
    eq = 0.01128156666995859 + 0.0002869276768158971 * chi
    linear = 0.01574321112717377 + 0.02244178140869133 * chi
    slope = combine(linear, eq, tpl)
    return [amplitude, 0.0, curvature, frequency, slope]


def _target_counterfactual(row, run, public):
    m1 = torch.tensor(row["mass1"], dtype=torch.float64)
    m2 = torch.tensor(row["mass2"], dtype=torch.float64)
    chi1 = torch.tensor(row["chi1"], dtype=torch.float64)
    chi2 = torch.tensor(row["chi2"], dtype=torch.float64)
    nu, chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    printed = np.asarray([float(value) for value in seobnrv4_nqc_targets(nu, chi)])
    source = np.asarray(_source_targets(float(nu), float(chi)))
    difference = source - printed
    matrices = run["nqc_matching_matrices"]
    amplitude_matrix = np.asarray(matrices["amplitude"])
    phase_matrix = np.asarray(matrices["phase"])
    shift = np.concatenate((np.linalg.solve(amplitude_matrix, difference[:3]),
                            -np.linalg.solve(phase_matrix, difference[3:])))
    native = np.asarray(run["nqc_coefficients"]["amplitude"]
                        + run["nqc_coefficients"]["phase"])
    original_gap = native - np.asarray(public)
    source_gap = native + shift - np.asarray(public)
    return dict(
        nu=float(nu), chi=float(chi), printed_targets=printed.tolist(),
        source_targets=source.tolist(), source_minus_printed_targets=difference.tolist(),
        native_matching_matrices=matrices,
        native_amplitude_matrix_condition=float(np.linalg.cond(amplitude_matrix)),
        native_phase_matrix_condition=float(np.linalg.cond(phase_matrix)),
        source_target_coefficient_shift=shift.tolist(),
        source_targets_on_native_coefficients=(native + shift).tolist(),
        source_targets_on_native_minus_public=source_gap.tolist(),
        source_target_shift_over_original_gap=(np.abs(shift) / np.abs(original_gap)).tolist(),
    )


def _public_coefficients(row, step_M):
    mass_s = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    coefficients = lal.CreateREAL8Vector(10)
    output = lalsim.SimIMRSpinAlignedEOBModes(
        step_M * mass_s, row["mass1"] * lal.MSUN_SI,
        row["mass2"] * lal.MSUN_SI, row["omega_start"] / (math.pi * mass_s),
        100e6 * lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        coefficients, 1, lal.CreateDict(), lal.CreateDict(),
    )
    values = np.asarray(coefficients.data, dtype=float).copy()
    if (len(output) != 3 or values.shape != (10,) or not np.isfinite(values).all()
            or np.any(values[[3, 4, 5, 8, 9]] != 0)):
        raise ValueError("unexpected public NQC result or inactive coefficients")
    return values[[0, 1, 2, 6, 7]].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.nqc-coefficients-plan.v1"
            or plan["level"] != "ode_2"
            or plan["source_reference"]["git_revision"] != "53acc0c2cd7ea2d3c83fc71a7bd88035c2861d6c"
            or plan["source_reference"]["sha256"] != "be34dbe154439439b6a7dc37494cdc963dd4c29ded42d887251efe16f974899e"
            or [item["name"] for item in plan["rows"]] != [
                "strong_aligned_spin", "strong_antialigned_spin"]):
        parser.error("frozen NQC plan differs")
    prerequisites = []
    for item in plan["rows"]:
        path = q.ROOT / item["waveform"]
        if q.sha256(path) != item["waveform_sha256"]:
            parser.error("frozen waveform evidence differs")
        prior = json.loads(path.read_text())
        if (prior["status"] != "diagnostic_complete"
                or not prior["check"]["numerical_check_passed"]
                or prior["row"]["chi1"] * (1 if item["name"] == "strong_aligned_spin" else -1) <= 0):
            parser.error("waveform prerequisite failed")
        prerequisites.append(prior)
    binary = prerequisites[0]["lal_binary"]
    if (prerequisites[1]["lal_binary"] != binary
            or q.lal_binary_identity(binary["lal_version"],
                                    binary["lalsimulation_version"]) != binary):
        parser.error("public binary identity differs")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.nqc-coefficients.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        source_sha256={path: q.sha256(q.ROOT / path) for path in (
            "dev/native_rom/qualify_generator.py",
            "dev/native_rom/eob/torch/adaptive_driver_clean.py",
            "src/torchwave/approximants/eob/nqc.py",
            "src/torchwave/approximants/eob/numerical_attachment.py",
        )}, lal_binary=binary, source_reference=plan["source_reference"],
        rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        for item, prior in zip(plan["rows"], prerequisites):
            level = prior["candidate"][plan["level"]]["level"]
            numerical, physical = prior["choice"]["numerical"], prior["choice"]["physical"]
            flux = prior["choice"]["flux_placements"]
            run = q.native_run(
                prior["row"], level, q.Settings(**prior["settings"]),
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=(
                    numerical["fixed_restart_time_M"] if item["name"] == "strong_aligned_spin"
                    else prior["choice"]["fixed_restart_time_M"]),
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=item["nqc_phase_window_M"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
                diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
                return_nqc_coefficients=True,
            )
            replay = q.summarize_run(run)["array_sha256"]
            if replay != prior["candidate"][plan["level"]]["native"]["array_sha256"]:
                raise ValueError(f"{item['name']} native waveform no longer reproduces")
            native = run["nqc_coefficients"]["amplitude"] + run["nqc_coefficients"]["phase"]
            public = _public_coefficients(prior["row"], level["dt_geom"])
            delta = (np.asarray(native) - np.asarray(public)).tolist()
            counterfactual = _target_counterfactual(prior["row"], run, public)
            report["rows"].append(dict(
                name=item["name"], row=prior["row"], output_step_M=level["dt_geom"],
                native_waveform_sha256=replay, public_coefficients=public,
                native_coefficients=native, native_minus_public=delta,
                target_counterfactual=counterfactual,
                peak_time_M=run["peak_time_M"], match_time_M=run["match_time_M"],
            ))
            q.write_evidence(args.json_out, report)
            print(f"row={item['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
