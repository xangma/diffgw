"""Development-only comparison of v4 orbital phase integrators.

Both integrators use the same provisional Hamiltonian, flux, and initial
condition. This checks numerical integration, not independent physics parity.
Run with ``PYTHONPATH=src:. python -m dev.native_rom.diagnose_orbital_phase
--t-end 310 --json-out /tmp/orbital-phase.json``. The default endpoint is 165 M.
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
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
import torch

from dev.native_rom.qualify_generator import canonical_hash, sha256, source_snapshot
from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER, integrate_adaptive,
)
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROW = dict(omega_start=0.04, mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0)
T_END_M = 165.0
OUTPUT_STEP_M = 0.5
ADAPTIVE_LEVELS = ((1e-13, 1e-14), (1e-14, 1e-15), (1e-15, 1e-16))
SCIPY_LEVELS = ((1e-11, 1e-12), (1e-12, 1e-13))


def snapshot():
    sources = source_snapshot()
    source = Path(__file__).resolve()
    sources[str(source.relative_to(source.parents[2]))] = sha256(source)
    return canonical_hash(sources)


def state_error(candidate, reference):
    difference = candidate - reference
    return dict(endpoint_absolute_by_state=np.abs(difference[-1]).tolist(),
                rms_absolute_by_state=np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                max_absolute_by_state=np.max(abs(difference), axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--t-end", type=float, default=T_END_M)
    args = parser.parse_args()
    if not np.isfinite(args.t_end) or args.t_end <= 0 or args.t_end / OUTPUT_STEP_M != round(args.t_end / OUTPUT_STEP_M):
        parser.error("--t-end must be positive, finite, and on the output grid")
    t_end = args.t_end
    source_before = snapshot()
    torch.set_num_threads(1)
    m1 = torch.tensor(ROW["mass1"], dtype=torch.float64)
    m2 = torch.tensor(ROW["mass2"], dtype=torch.float64)
    chi1 = torch.tensor(ROW["chi1"], dtype=torch.float64)
    chi2 = torch.tensor(ROW["chi2"], dtype=torch.float64)
    start = torch.tensor(ROW["omega_start"], dtype=torch.float64)
    x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
    nu = x1 * x2
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=x1 - x2)
    rr = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8",
                              retain_pan_cubic_spin=True)
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(ham, rr, start)
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0))
    field = EOBTrajectoryIntegrator(ham, rr).vector_field
    output_times = np.arange(round(t_end / OUTPUT_STEP_M) + 1) * OUTPUT_STEP_M
    initial_numpy = initial.detach().cpu().numpy()

    adaptive = {}
    accepted = {}
    hermite = {}
    adaptive_stats = {}
    for rtol, atol in ADAPTIVE_LEVELS:
        name = f"rtol_{rtol:.0e}"
        started = time.monotonic()
        with torch.no_grad():
            run = integrate_adaptive(
                field, initial, OUTPUT_STEP_M, t_end=t_end,
                atol=atol, rtol=rtol, initial_step=OUTPUT_STEP_M,
                controller=GSL_STATE_SCALED_CONTROLLER,
            )
        times = run.output_time.detach().cpu().numpy()
        if not np.array_equal(times, output_times):
            raise ValueError("adaptive output did not cover the exact comparison grid")
        adaptive[name] = run.output_state.detach().cpu().numpy()
        accepted_times = run.time.detach().cpu().numpy()
        accepted_states = run.state.detach().cpu().numpy()
        accepted_derivatives = run.derivative.detach().cpu().numpy()
        accepted[name] = (accepted_times, accepted_states)
        hermite[name] = CubicHermiteSpline(
            accepted_times, accepted_states, accepted_derivatives)(output_times)
        stats = run.statistics
        adaptive_stats[name] = dict(rtol=rtol, atol=atol,
                                    accepted=stats.total_accepted,
                                    rejected=stats.total_rejected,
                                    evaluations=stats.total_evaluations,
                                    termination=list(stats.termination),
                                    seconds=time.monotonic() - started)
        print(json.dumps({"run": name, **adaptive_stats[name]}), flush=True)

    def scipy_field(_time, state):
        with torch.no_grad():
            value = field(torch.from_numpy(state))
        return value.detach().cpu().numpy()

    scipy_runs = {}
    scipy_stats = {}
    for rtol, atol in SCIPY_LEVELS:
        name = f"dop853_rtol_{rtol:.0e}"
        started = time.monotonic()
        solution = solve_ivp(
            scipy_field, (0.0, t_end), initial_numpy, method="DOP853",
            rtol=rtol, atol=atol, t_eval=output_times, first_step=OUTPUT_STEP_M,
            dense_output=True,
        )
        if not solution.success or not np.array_equal(solution.t, output_times):
            raise RuntimeError(f"DOP853 failed: {solution.message}")
        scipy_runs[name] = solution
        scipy_stats[name] = dict(rtol=rtol, atol=atol, evaluations=solution.nfev,
                                 seconds=time.monotonic() - started,
                                 message=solution.message)
        print(json.dumps({"run": name, **scipy_stats[name]}), flush=True)

    reference_solution = scipy_runs["dop853_rtol_1e-12"]
    reference = reference_solution.y.T
    comparisons = {name: state_error(state, reference)
                   for name, state in adaptive.items()}
    accepted_comparisons = {name: state_error(
        states, reference_solution.sol(times).T)
        for name, (times, states) in accepted.items()}
    hermite_comparisons = {name: state_error(state, reference)
                           for name, state in hermite.items()}
    comparisons["dop853_1e-11_to_1e-12"] = state_error(
        scipy_runs["dop853_rtol_1e-11"].y.T, reference)
    adaptive_names = list(adaptive)
    for first, second in zip(adaptive_names, adaptive_names[1:]):
        comparisons[f"{first}_to_{second}"] = state_error(adaptive[first], adaptive[second])
    source_after = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.orbital_phase_diagnostic.v1",
        scope=dict(physics_validated=False, default_generator_qualified=False,
                   development_only=True, same_vector_field=True,
                   independent_integrator=True),
        row=ROW, state_order=["r", "phi", "pr_star", "pphi"],
        t_end_M=t_end, output_step_M=OUTPUT_STEP_M,
        initial_state=initial_numpy.tolist(),
        adaptive=adaptive_stats, scipy=scipy_stats, comparisons=comparisons,
        accepted_state_comparisons=accepted_comparisons,
        hermite_output_comparisons=hermite_comparisons,
        source_sha256_before=source_before, source_sha256_after=source_after,
        sources_unchanged=source_before == source_after,
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.json_out}", flush=True)
    if source_before != source_after:
        raise RuntimeError("sources changed during the diagnostic")


if __name__ == "__main__":
    main()
