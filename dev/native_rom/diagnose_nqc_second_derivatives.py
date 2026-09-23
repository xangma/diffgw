"""Compare NQC matching second derivatives with DOP853 dense output.

The native and DOP853 paths use the same provisional EOB equations. This
checks numerical differentiation at two frozen match times, not physics.
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
from .diagnose_nqc_matching_inputs import _system
from .diagnose_trajectory_sensitivity import _array, _hash


CAMPAIGNS = {
    "nonspinning": "dev/native_rom/evidence/2026-09-23/hermite_local_nqc_fixed_restart_campaign.json",
    "aligned_spin": "dev/native_rom/evidence/2026-09-23/aligned_spin_fixed_restart_campaign.json",
}
STEPS_M = (0.2, 0.1, 0.05, 0.025)
DOP_LEVELS = ((1e-11, 1e-12), (1e-12, 1e-13))  # rtol, atol
LOCAL_WINDOW_M = 0.5
LOCAL_DEGREE = 6


def _local_second(time_axis, values, match):
    selected = abs(time_axis-match) <= LOCAL_WINDOW_M
    chart = (time_axis[selected]-match)/LOCAL_WINDOW_M
    if len(chart) < LOCAL_DEGREE+3 or not (chart.min() < 0 < chart.max()):
        raise ValueError("local derivative fit lacks two-sided samples")
    local = values[selected]
    coefficients = np.polynomial.polynomial.polyfit(
        chart, local-local[len(local)//2], LOCAL_DEGREE)
    return 2*coefficients[2]/LOCAL_WINDOW_M**2, int(len(chart))


def _native(campaign):
    import torch
    from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
    from torchwave.approximants.eob.nqc import (
        seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
    )

    row = campaign["row"]
    prior = next(run for run in campaign["result"]["runs"] if run["name"] == "ode_2")
    settings = q.Settings(**campaign["settings"])
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES, **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
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
        raise ValueError("native replay differs from frozen waveform")
    matching = _system(result, targets)
    global_match = float(result.attachment.match_time)
    if abs(global_match - prior["match_time_M"]) > 1e-12:
        raise ValueError("native replay differs from frozen match time")
    nqc = result.attachment.fine.nqc
    time_axis = _array(nqc.amplitude_spline.knots)
    local_match = matching["match_time_local_M"]
    values = np.column_stack((_array(nqc.sampled_amplitude),
                              _array(nqc.sampled_products)))
    amplitude_local, count = _local_second(time_axis, values, local_match)
    phase_local, phase_count = _local_second(
        time_axis, _array(nqc.sampled_waveform_phase), local_match)
    if count != phase_count:
        raise ValueError("candidate fits selected different samples")
    systems = matching["systems"]
    spline_second = np.concatenate((systems["amplitude"]["base"][2:3],
                                    systems["amplitude"]["matrix"][2],
                                    systems["phase"]["base"][1:2]))
    candidate_second = np.concatenate((amplitude_local, np.atleast_1d(phase_local)))
    return dict(level={key: prior[key] for key in ("atol", "rtol", "dt_geom")},
                hashes=hashes, match_time_global_M=global_match,
                match_time_local_M=local_match, local_fit_samples=count,
                spline_second=spline_second.tolist(),
                candidate_second=candidate_second.tolist())


def _dense(campaign, match_time):
    import torch
    from scipy.integrate import solve_ivp
    from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
    from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
    from torchwave.approximants.eob.nqc import nqc_22_basis
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    row = campaign["row"]
    m1, m2, chi1, chi2 = (torch.tensor(row[key], dtype=torch.float64)
                          for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SEOBNRv4Hamiltonian(x1*x2, chi1, chi2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(spin_truncation=q.CHOICES["spin_truncation"],
                                retain_pan_cubic_spin=True)
    with torch.no_grad():
        initial_values = find_quasicircular_initial_conditions(
            ham, flux, torch.tensor(row["omega_start"], dtype=torch.float64))
        initial = torch.stack((initial_values[0], torch.zeros_like(m1),
                               initial_values[1], initial_values[2]))
    field = EOBTrajectoryIntegrator(ham, flux).vector_field

    def scipy_field(_time, state):
        with torch.no_grad():
            return _array(field(torch.from_numpy(state)))

    def quantities(solution, times):
        amplitudes, products, modes = [], [], []
        for coordinate in times:
            state = torch.from_numpy(solution.sol(coordinate))
            with torch.no_grad():
                omega = field(state)[1]
                mode, _ = published_mode22_and_frequency(
                    state, ham, flux, spin_v6=q.CHOICES["spin_v6"],
                    include_pan_v8=False, include_positive_chi_540=False,
                    retain_pan_cubic_spin=True)
                basis, _ = nqc_22_basis(state[0], state[2], omega)
            amplitude = abs(complex(_array(mode)))
            amplitudes.append(amplitude)
            products.append(amplitude*_array(basis))
            modes.append(complex(_array(mode)))
        phases = -np.angle(np.asarray(modes)*np.conj(modes[len(modes)//2]))
        if np.max(abs(np.diff(phases))) >= np.pi:
            raise ValueError("dense waveform phase crosses a local branch")
        return np.column_stack((amplitudes, np.asarray(products), phases))

    runs = []
    for rtol, atol in DOP_LEVELS:
        started = time.monotonic()
        solution = solve_ivp(
            scipy_field, (0., match_time + 2*max(STEPS_M)), _array(initial),
            method="DOP853", rtol=rtol, atol=atol, first_step=0.5,
            dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        stencils = {}
        for step in STEPS_M:
            values = quantities(solution, match_time + np.arange(-2, 3)*step)
            second = (-values[0]+16*values[1]-30*values[2]
                      +16*values[3]-values[4])/(12*step**2)
            stencils[f"{step:g}"] = second.tolist()
        runs.append(dict(rtol=rtol, atol=atol, evaluations=solution.nfev,
                         seconds=time.monotonic()-started, second=stencils))
        print(json.dumps(dict(dop_rtol=rtol, seconds=runs[-1]["seconds"],
                              evaluations=solution.nfev)), flush=True)
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch = __import__("torch")
    torch.set_num_threads(1)
    before = q.source_snapshot()
    rows = {}
    for name, path in CAMPAIGNS.items():
        campaign = json.loads(Path(path).read_text())
        if (campaign["numerical_choice"]["fixed_restart_time_M"] != 165.
                or not campaign["numerical_choice"]["local_nqc_phase_derivatives"]
                or any(campaign["source_sha256_before"].get(key) != value
                       for key, value in before.items())):
            parser.error(f"{name} frozen generator sources or choices differ")
        started = time.monotonic()
        native = _native(campaign)
        print(json.dumps(dict(row=name, native_seconds=time.monotonic()-started,
                              hash_verified=True)), flush=True)
        dense = _dense(campaign, native["match_time_global_M"])
        reference = np.asarray(dense[-1]["second"][f"{min(STEPS_M):g}"])
        rows[name] = dict(row=campaign["row"], campaign_sha256=q.sha256(path),
                          native=native, dop853=dense,
                          comparison=dict(reference_step_M=min(STEPS_M),
                                          reference=reference.tolist(),
                                          spline_error=(np.asarray(native["spline_second"])-reference).tolist(),
                                          candidate_error=(np.asarray(native["candidate_second"])-reference).tolist(),
                                          dop_tolerance_spread=(np.asarray(dense[0]["second"][f"{min(STEPS_M):g}"])-reference).tolist(),
                                          dop_step_spread=(np.asarray(dense[-1]["second"][f"{STEPS_M[-2]:g}"])-reference).tolist()))
    after = q.source_snapshot()
    output = dict(schema="diffgw.seobnrv4.nqc-second-derivative-diagnostic.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             same_provisional_vector_field=True,
                             independent_integrator=True, independent_physics=False,
                             numerical_generator_qualified=False, algorithm_choice_made=False),
                  predeclared_candidate=dict(window_M=LOCAL_WINDOW_M,
                                             polynomial_degree=LOCAL_DEGREE,
                                             quantities=["A", "A*q1", "A*q2", "A*q3", "waveform_phase"],
                                             operation="replace only second derivatives"),
                  steps_M=STEPS_M, dop_levels=DOP_LEVELS, rows=rows,
                  source_sha256_before=before, source_sha256_after=after,
                  sources_unchanged=before == after,
                  script_sha256=q.sha256(__file__),
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(),
                               pid=os.getpid()))
    q.write_evidence(args.json_out, output)
    if before != after:
        raise RuntimeError("generator sources changed during diagnostic")
    print(json.dumps(dict(status="diagnostic_complete", evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
