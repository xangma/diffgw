"""Check aligned-spin Hermite output sampling against DOP853 dense output.

The native solver's first step and vector field stay fixed across output grids.
This tests one provisional coarse inspiral through 165 M, not the full model.
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
import scipy
from scipy.integrate import solve_ivp
import torch

from . import qualify_generator as q
from .diagnose_orbital_phase import state_error
from .eob.torch.adaptive_driver_clean import _diagnostic_hermite_output
from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER, integrate_adaptive,
)
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROW = dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)
GRIDS_M = (0.5, 0.25, 0.125)
END_M = 165.0
ATOL, RTOL = 1e-14, 1e-13


def _array(value):
    data = value.detach().cpu().numpy()
    if not np.isfinite(data).all():
        raise ValueError("nonfinite trajectory")
    return data


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    before = q.source_snapshot()
    torch.set_num_threads(1)
    m1, m2 = ROW["mass1"], ROW["mass2"]
    nu = m1*m2/(m1+m2)**2
    ham = SEOBNRv4Hamiltonian(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(ROW["chi1"], dtype=torch.float64),
        torch.tensor(ROW["chi2"], dtype=torch.float64),
        delta=torch.tensor((m1-m2)/(m1+m2), dtype=torch.float64))
    flux = SEOBNRv4PublishedFlux(
        spin_truncation=q.CHOICES["spin_truncation"], retain_pan_cubic_spin=True)
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(
            ham, flux, torch.tensor(ROW["omega_start"], dtype=torch.float64))
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0))
    field = EOBTrajectoryIntegrator(ham, flux).vector_field
    report = dict(schema="diffgw.seobnrv4.output-sampling.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             numerical_generator_qualified=False,
                             same_provisional_vector_field=True,
                             coarse_inspiral_only=True),
                  status="running", row=ROW, state_order=["r", "phi", "pr_star", "pphi"],
                  t_end_M=END_M, grids_M=GRIDS_M, atol=ATOL, rtol=RTOL,
                  initial_step_M=0.5, script_sha256=q.sha256(__file__),
                  source_sha256_before=before,
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
                  versions=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  native={}, reference={}, comparisons={})
    q.write_evidence(args.json_out, report)
    samples = {}
    accepted_hashes = []
    accepted = None
    baseline_run = None
    for dt in GRIDS_M:
        started = time.monotonic()
        with torch.no_grad():
            run = integrate_adaptive(
                field, initial, dt, t_end=END_M, atol=ATOL, rtol=RTOL,
                initial_step=0.5, controller=GSL_STATE_SCALED_CONTROLLER,
                max_rejected_steps=10000, max_evaluations=200000,
                max_accepted_steps=20000)
            hermite = _diagnostic_hermite_output(run)
        time_axis = _array(run.output_time)
        expected = np.arange(round(END_M/dt)+1)*dt
        if not np.array_equal(time_axis, expected):
            raise ValueError("native sampler did not cover the expected grid")
        hashes = {key: _hash(_array(getattr(run, key)))
                  for key in ("time", "state", "derivative")}
        accepted_hashes.append(hashes)
        if accepted is None:
            accepted = (_array(run.time), _array(run.state))
            baseline_run = run
        samples[dt] = dict(time=time_axis, hermite=_array(hermite.output_state),
                           natural_cubic=_array(run.output_state))
        stats = run.statistics
        report["native"][str(dt)] = dict(
            accepted_hashes=hashes, accepted=stats.total_accepted,
            rejected=stats.total_rejected, evaluations=stats.total_evaluations,
            samples=len(time_axis), seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print(json.dumps(dict(grid_M=dt, **report["native"][str(dt)]),
                         sort_keys=True), flush=True)
    report["accepted_trajectories_identical"] = all(
        item == accepted_hashes[0] for item in accepted_hashes[1:])
    if not report["accepted_trajectories_identical"]:
        raise ValueError("output-grid change altered the accepted trajectory")

    def scipy_field(_time, state):
        with torch.no_grad():
            return _array(field(torch.from_numpy(state)))

    solutions = {}
    for rtol, atol in ((1e-11, 1e-12), (1e-12, 1e-13)):
        started = time.monotonic()
        solution = solve_ivp(
            scipy_field, (0., END_M), _array(initial), method="DOP853",
            rtol=rtol, atol=atol, first_step=0.5, dense_output=True)
        if not solution.success:
            raise RuntimeError(f"DOP853 failed: {solution.message}")
        key = f"rtol_{rtol:.0e}"
        solutions[key] = solution
        report["reference"][key] = dict(rtol=rtol, atol=atol,
                                        evaluations=solution.nfev,
                                        accepted_nodes=len(solution.t),
                                        seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print(json.dumps(dict(reference=key, **report["reference"][key])), flush=True)
    reference = solutions["rtol_1e-12"]
    loose = solutions["rtol_1e-11"]
    report["comparisons"]["accepted_vs_dop853"] = state_error(
        accepted[1], reference.sol(accepted[0]).T)
    for dt in GRIDS_M:
        record = samples[dt]
        t = record["time"]
        target = reference.sol(t).T
        report["comparisons"][str(dt)] = dict(
            hermite_vs_dop853=state_error(record["hermite"], target),
            natural_cubic_vs_dop853=state_error(record["natural_cubic"], target),
            dop853_tolerance_spread=state_error(loose.sol(t).T, target))
    for coarse, fine in zip(GRIDS_M, GRIDS_M[1:]):
        ratio = round(coarse/fine)
        lhs, rhs = samples[coarse]["hermite"], samples[fine]["hermite"][::ratio]
        report["comparisons"][f"nested_{coarse}_to_{fine}"] = state_error(lhs, rhs)
    # This fixed off-grid probe measures the same Hermite interpolant without
    # allowing any of the three output grids to choose the comparison points.
    probe = np.arange(0.0625, END_M, 0.125)
    from dataclasses import replace
    with torch.no_grad():
        # The accepted history is identical at all three levels; replay the
        # baseline once on the independent probe grid.
        probe_run = replace(baseline_run, output_time=torch.from_numpy(probe))
        probe_state = _array(_diagnostic_hermite_output(probe_run).output_state)
    report["comparisons"]["fixed_off_grid_probe"] = dict(
        start_M=float(probe[0]), step_M=0.125, samples=len(probe),
        hermite_vs_dop853=state_error(probe_state, reference.sol(probe).T),
        dop853_tolerance_spread=state_error(loose.sol(probe).T, reference.sol(probe).T))
    report["source_sha256_after"] = q.source_snapshot()
    report["sources_unchanged"] = report["source_sha256_after"] == before
    report["status"] = "diagnostic_complete" if report["sources_unchanged"] else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["sources_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
