"""Frozen-row NQC chain-rule and independent dense-trajectory diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_trajectory_sensitivity import _array, _hash
from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import nqc_22_basis, seobnrv4_nqc_parameters, seobnrv4_nqc_targets
from torchwave.approximants.eob.nqc_analytic_clean import directional_derivatives
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-29/ode_gate_slice"
LEVELS = ((1e-11, 1e-12), (1e-12, 1e-13))
STEPS = (0.1, 0.05, 0.025)
NAMES = ("A", "Aq1", "Aq2", "Aq3", "p1", "p2", "psi")


def _values(state, field, ham, rr):
    omega = field(state)[1]
    mode, _ = published_mode22_and_frequency(
        state, ham, rr, spin_v6="bohe_total_energy", include_pan_v8=False,
        include_positive_chi_540=False, retain_pan_cubic_spin=True)
    amplitude = mode.abs()
    aq, p = nqc_22_basis(state[0], state[2], omega)
    return torch.cat((amplitude.reshape(1), amplitude * aq, p,
                      (-torch.angle(mode)).reshape(1)))


def _matching(value, first, second, targets):
    amplitude_matrix = np.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_rhs = np.asarray(targets[:3]) - np.array((value[0], first[0], second[0]))
    phase_matrix = np.stack((first[4:6], second[4:6]))
    phase_rhs = np.array((first[6], second[6])) - np.asarray(targets[3:])
    return dict(amplitude_matrix=amplitude_matrix.tolist(), amplitude_rhs=amplitude_rhs.tolist(),
                phase_matrix=phase_matrix.tolist(), phase_rhs=phase_rhs.tolist(),
                amplitude_coefficients=np.linalg.solve(amplitude_matrix, amplitude_rhs).tolist(),
                phase_coefficients=np.linalg.solve(phase_matrix, phase_rhs).tolist(),
                amplitude_condition=float(np.linalg.cond(amplitude_matrix)),
                phase_condition=float(np.linalg.cond(phase_matrix)))


def _match_vector(state, field, ham, rr):
    value, first, second = directional_derivatives(
        lambda x: _values(x, field, ham, rr), state, field)
    return tuple(_array(item) for item in (value, first, second))


def _native(receipt, row, settings):
    prior = next(run for run in receipt["results"][0]["runs"] if run["name"] == "ode_2")
    choices = dict(q.CHOICES, spin_truncation="tabulated")
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **choices, **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps,
        )
    hashes = dict(time=_hash(_array(result.time)), mode=_hash(_array(result.mode)))
    if hashes != prior["array_sha256"]:
        raise ValueError(f"frozen ode_2 waveform hash changed: {hashes}")
    if abs(float(result.attachment.match_time) - prior["match_time_M"]) > 1e-12:
        raise ValueError("frozen match time changed")
    run = result.trajectory.fine
    count = int(run.output_valid.sum())
    time_axis = run.output_time[:count]
    state_axis = run.output_state[:count]
    match = result.attachment.fine.match_time
    state = natural_cubic_spline(time_axis, state_axis).evaluate(match)
    nqc = result.attachment.fine.nqc
    sampled = torch.cat((nqc.sampled_amplitude[:, None], nqc.sampled_products,
                         nqc.sampled_phase_basis), -1)
    sampled_spline = natural_cubic_spline(time_axis, sampled)
    values = [_array(sampled_spline.evaluate(match, derivative=k)) for k in (0, 1, 2)]
    phase = [_array(nqc.waveform_phase_spline.evaluate(match, derivative=k)) for k in (0, 1, 2)]
    for k in range(3):
        values[k] = np.r_[values[k], phase[k]]
    return result, state.detach(), values, hashes


def _dense(initial, global_match, field, ham, rr):
    def scipy_field(_t, y):
        with torch.no_grad():
            return _array(field(torch.from_numpy(y)))

    def stencil(solution, step):
        times = global_match + np.arange(-2, 3) * step
        rows = []
        for t in times:
            with torch.no_grad():
                rows.append(_array(_values(torch.from_numpy(solution.sol(t)), field, ham, rr)))
        rows = np.asarray(rows)
        # The phase is a local chart about the center; all sample separations
        # are far smaller than pi, so unwrap changes only its additive offset.
        rows[:, 6] = np.unwrap(rows[:, 6])
        first = (rows[0] - 8*rows[1] + 8*rows[3] - rows[4]) / (12*step)
        second = (-rows[0] + 16*rows[1] - 30*rows[2] + 16*rows[3] - rows[4]) / (12*step**2)
        return rows[2], first, second

    out = []
    for rtol, atol in LEVELS:
        start = time.monotonic()
        solution = solve_ivp(scipy_field, (0., global_match + 0.25), initial,
                             method="DOP853", rtol=rtol, atol=atol,
                             first_step=0.5, dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.sol(global_match)
        chain = _match_vector(torch.from_numpy(state), field, ham, rr)
        stencils = {str(step): [item.tolist() for item in stencil(solution, step)]
                    for step in STEPS}
        entry = dict(rtol=rtol, atol=atol, nfev=solution.nfev,
                     elapsed_seconds=time.monotonic()-start, state=state.tolist(),
                     chain=[item.tolist() for item in chain], stencils=stencils)
        out.append(entry)
        print(json.dumps(dict(dop_rtol=rtol, seconds=entry["elapsed_seconds"],
                              evaluations=solution.nfev)), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=("nonspin", "spin"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((EVIDENCE / f"rerun_{args.row}.json").read_text())
    if q.source_snapshot() != receipt["source_sha256_before"]:
        parser.error("frozen generator source changed")
    row = receipt["results"][0]["inputs"]
    settings = q.Settings(**receipt["config"]["settings"])
    m1, m2, s1, s2 = (torch.tensor(row[key], dtype=torch.float64)
                      for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SEOBNRv4Hamiltonian(x1*x2, s1, s2, delta=x1-x2)
    rr = SEOBNRv4PublishedFlux(spin_truncation="tabulated", retain_pan_cubic_spin=True)
    field = EOBTrajectoryIntegrator(ham, rr).vector_field
    targets = _array(torch.stack(seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(x1, x2, s1, s2))))
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(
            ham, rr, torch.tensor(row["omega_start"], dtype=torch.float64))
        initial = _array(torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)))
    started = time.monotonic()
    result, native_state, spline, hashes = _native(receipt, row, settings)
    native_chain = _match_vector(native_state, field, ham, rr)
    print(json.dumps(dict(native_seconds=time.monotonic()-started,
                          hash_verified=True, match_time_M=float(result.attachment.match_time))),
          flush=True)
    dense = _dense(initial, float(result.attachment.match_time), field, ham, rr)
    tight = tuple(np.asarray(x) for x in dense[-1]["chain"])
    stencil = tuple(np.asarray(x) for x in dense[-1]["stencils"][str(STEPS[-1])])
    # Phase value is chart-dependent and not a matching entry.
    mask = np.array([True]*6+[False])
    def relative(x, y, selected):
        return float(np.max(np.abs(x[selected]-y[selected]) /
                            np.maximum(np.abs(y[selected]), 1e-12)))
    comparison = dict(
        dop_center_state_max_abs=float(np.max(np.abs(np.asarray(dense[0]["state"])
                                                  - np.asarray(dense[1]["state"])))),
        native_vs_dop_state_max_abs=float(np.max(np.abs(_array(native_state)
                                                   - np.asarray(dense[-1]["state"])))),
        native_vs_dop_chain_max_rel=max(relative(a,b,mask if k == 0 else np.ones(7, bool))
                                         for k,(a,b) in enumerate(zip(native_chain,tight))),
        dop_stencil_vs_chain_max_rel=max(relative(a,b,mask if k == 0 else np.ones(7, bool))
                                          for k,(a,b) in enumerate(zip(stencil,tight))),
        spline_vs_native_chain=[(a-b).tolist() for a,b in zip(spline,native_chain)],
    )
    output = dict(schema="diffgw.seobnrv4.ode-chain-derivatives.v1", row=row,
                  status="diagnostic_complete", scope=dict(q.SCOPE, development_only=True),
                  source_manifest_sha256=receipt["source_manifest_sha256"],
                  receipt_sha256=q.sha256(EVIDENCE / f"rerun_{args.row}.json"),
                  script_sha256=q.sha256(__file__), hashes=hashes,
                  native=dict(state=_array(native_state).tolist(),
                              match_time_M=float(result.attachment.match_time),
                              spline=[x.tolist() for x in spline],
                              chain=[x.tolist() for x in native_chain],
                              spline_system=_matching(*spline, targets),
                              chain_system=_matching(*native_chain, targets)),
                  dop853=dense, comparison=comparison,
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status="done", comparison=comparison)), flush=True)


if __name__ == "__main__":
    main()
