"""Development-only DOP853 check of SEOBNRv4 NQC phase-basis derivatives.

DOP853 integrates the same provisional Hamiltonian and flux as the native
generator. Dense states are differentiated with a five-point centered stencil
at the previously frozen NQC match time. This is a numerical cross-check,
not independent physics validation or a waveform-algorithm choice.
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
from scipy.integrate import solve_ivp
import torch

from dev.native_rom.diagnose_hermite_error import ROW, array
from dev.native_rom.qualify_generator import canonical_hash, sha256, source_snapshot
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


LEVELS = ((1e-11, 1e-12), (1e-12, 1e-13))
STEPS_M = (0.2, 0.1, 0.05, 0.025)


def snapshot():
    sources = source_snapshot()
    for path in (Path(__file__).resolve(), Path(array.__code__.co_filename).resolve()):
        sources[str(path.relative_to(path.parents[2]))] = sha256(path)
    return canonical_hash(sources)


def five_point_matrix(solution, match_time_M, step_M, phase_basis):
    offsets = np.arange(-2, 3, dtype=np.float64) * step_M
    values = np.stack([phase_basis(solution.sol(match_time_M + offset))
                       for offset in offsets])
    first = (values[0] - 8*values[1] + 8*values[3] - values[4]) / (12*step_M)
    second = (-values[0] + 16*values[1] - 30*values[2]
              + 16*values[3] - values[4]) / (12*step_M**2)
    return np.stack((first, second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--hermite-evidence", type=Path, default=Path(
        "dev/native_rom/evidence/2026-09-23/nqc_derivatives.json"))
    args = parser.parse_args()
    before = snapshot()
    hermite_evidence_sha256 = sha256(args.hermite_evidence)
    hermite = json.loads(args.hermite_evidence.read_text())
    if hermite["schema"] != "diffgw.seobnrv4.nqc_derivative_diagnostic.v1":
        raise ValueError("expected frozen NQC derivative evidence")
    first_run = hermite["runs"][0]
    match_time_M = first_run["restart_time_M"] + first_run["match_time_M"]
    if not all(row["restart_time_M"] + row["match_time_M"] == match_time_M
               for row in hermite["runs"]):
        raise ValueError("prior match times differ")
    torch.set_num_threads(1)
    m1 = torch.tensor(ROW["mass1"], dtype=torch.float64)
    m2 = torch.tensor(ROW["mass2"], dtype=torch.float64)
    chi1 = torch.tensor(ROW["chi1"], dtype=torch.float64)
    chi2 = torch.tensor(ROW["chi2"], dtype=torch.float64)
    start = torch.tensor(ROW["omega_start"], dtype=torch.float64)
    x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
    hamiltonian = SEOBNRv4Hamiltonian(x1*x2, chi1, chi2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8",
                                retain_pan_cubic_spin=True)
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(hamiltonian, flux, start)
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0))
    field = EOBTrajectoryIntegrator(hamiltonian, flux).vector_field

    def scipy_field(_time, state):
        with torch.no_grad():
            derivative = field(torch.from_numpy(state))
        return array(derivative)

    def phase_basis(state):
        tensor = torch.from_numpy(np.asarray(state))
        with torch.no_grad():
            orbital_frequency = field(tensor)[1]
            _, basis = nqc_22_basis(tensor[0], tensor[2], orbital_frequency)
        return array(basis)

    runs = []
    for rtol, atol in LEVELS:
        started = time.monotonic()
        solution = solve_ivp(
            scipy_field, (0.0, match_time_M + 2*max(STEPS_M)), array(initial),
            method="DOP853", rtol=rtol, atol=atol, first_step=0.5,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        matrices = {f"step_{step:g}": five_point_matrix(
            solution, match_time_M, step, phase_basis).tolist()
            for step in STEPS_M}
        runs.append(dict(rtol=rtol, atol=atol, evaluations=solution.nfev,
                         seconds=time.monotonic()-started,
                         match_state=solution.sol(match_time_M).tolist(),
                         match_phase_basis=phase_basis(solution.sol(match_time_M)).tolist(),
                         matrices=matrices))
        print(json.dumps({"rtol": rtol, "seconds": runs[-1]["seconds"],
                          "evaluations": solution.nfev}), flush=True)
    comparisons = {}
    for step in STEPS_M:
        name = f"step_{step:g}"
        matrices = [np.asarray(run["matrices"][name]) for run in runs]
        comparisons[name] = dict(
            dop853_tolerance_matrix_max_absolute_difference=float(np.max(abs(
                matrices[0] - matrices[1]))),
            natural_cubic_matrix_max_absolute_difference=[float(np.max(abs(
                matrices[index] - np.asarray(hermite["runs"][index]["native_matrix"]))))
                for index in range(2)],
            local_polynomial_matrix_max_absolute_difference=[float(np.max(abs(
                matrices[index] - np.asarray(hermite["runs"][index]["methods"][
                    "window_0.25_degree_6"]["matrix"])))) for index in range(2)],
        )
    after = snapshot()
    report = dict(
        schema="diffgw.seobnrv4.nqc_dense_reference_diagnostic.v1",
        scope=dict(development_only=True, same_provisional_vector_field=True,
                   independent_integrator=True, independent_physics=False,
                   numerical_generator_qualified=False, algorithm_choice_made=False),
        row=ROW, match_time_M=match_time_M, steps_M=STEPS_M,
        hermite_evidence=str(args.hermite_evidence),
        hermite_evidence_sha256=hermite_evidence_sha256,
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
