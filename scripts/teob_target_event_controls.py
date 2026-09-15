#!/usr/bin/env python3
"""Published-target precision and neighboring-knot controls on frozen inputs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

import numpy as np
import torch

from teob_grid_derivative_controls import (
    OUT as NOMINAL, INPUT, ROOT, CASES, digest, source_hashes, case_for, json_value,
    apply_coefficients, assemble_teob_merger_nqc, published_2017_nqc_targets,
    PUBLISHED_2017_NQC_PROVENANCE, assemble_teob_ringdown_22,
    uib2016_nonspinning_remnant, project_teob22_generic, condition_td_to_fd,
    plan_td_support, evaluate_cached,
)
from torchwave.approximants.eob.teob_merger_nqc_clean import (
    TEOBMergerNQCTargets, quartic_derivative,
)
from torchwave.approximants.eob.teob_nqc_clean import NQCCoefficients, NQCMatchResult

OUT = ROOT / 'docs/reviews/2026-09-15-teob-target-audit'
TARGETS = ('amplitude', 'amplitude_derivative', 'frequency', 'frequency_derivative')


def solve_at_index(base, index):
    """Change the solve index only; reuse the same auxiliary sampled arrays."""
    aw, aux, h = base.auxiliary_amplitude_rw, base.auxiliary, base.auxiliary_spacing
    d = lambda y: quartic_derivative(y, h, sample_dim=0)
    da, dprod, dp, dphase = d(aw), d(aw[:, None]*aux.q), d(aux.p), d(aux.phase)
    am = torch.stack((aw[index]*aux.q[index], dprod[index]))
    bm = torch.stack((dp[index], d(dp)[index]))
    targets = base.targets
    ar = torch.stack((targets.amplitude-aw[index], targets.amplitude_derivative-da[index]))
    br = torch.stack((dphase[index]-targets.frequency, d(dphase)[index]-targets.frequency_derivative))
    coeff = NQCCoefficients(torch.linalg.solve(am, ar), torch.linalg.solve(bm, br))
    match = NQCMatchResult(coeff, am, bm, torch.linalg.cond(am), torch.linalg.cond(bm),
                          torch.linalg.vector_norm(am@coeff.amplitude-ar),
                          torch.linalg.vector_norm(bm@coeff.phase-br))
    result = replace(apply_coefficients(base, coeff), matching_index=index, match=match)
    return result, {'amplitude_solve_residual': match.amplitude_residual,
                    'phase_solve_residual': match.phase_residual,
                    'amplitude_condition': match.amplitude_condition, 'phase_condition': match.phase_condition}


def target_box(base, radius):
    """Exact linear coefficient response to a target box, at fixed matrices."""
    eye = torch.eye(2, dtype=torch.float64)
    am, bm = base.amplitude_matrix, base.phase_matrix
    response_a = torch.linalg.solve(am, eye)
    response_b = -torch.linalg.solve(bm, eye)
    bound_a = response_a.abs() @ radius[:2]
    bound_b = response_b.abs() @ radius[2:]
    max_a = torch.zeros(2, dtype=torch.float64)
    max_b = torch.zeros_like(max_a)
    for signs in itertools.product((-1., 1.), repeat=4):
        delta = radius * torch.tensor(signs, dtype=torch.float64)
        max_a = torch.maximum(max_a, torch.linalg.solve(am, delta[:2]).abs())
        max_b = torch.maximum(max_b, torch.linalg.solve(bm, -delta[2:]).abs())
    torch.testing.assert_close(max_a, bound_a, rtol=2e-14, atol=1e-18)
    torch.testing.assert_close(max_b, bound_b, rtol=2e-14, atol=1e-18)
    return {'target_radii': radius, 'coefficient_radius_amplitude': bound_a,
            'coefficient_radius_phase': bound_b, 'all_16_vertices_verified': True,
            'response_amplitude': response_a, 'response_phase': response_b}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=CASES, required=True)
    args = ap.parse_args()
    out = OUT/args.case
    out.mkdir(parents=True, exist_ok=True)
    path = out/'controls.json'
    if path.exists():
        ap.error('output report already exists; preserve completed experiments')
    case = case_for(args.case)
    p, g = case['params'], case['grid']
    cache = INPUT/args.case/'continuous.npz'
    nominal_path = NOMINAL/args.case/'nominal.npz'
    frozen_report = json.loads((NOMINAL/'derivative-controls.json').read_text())['cases'][args.case]
    assert digest(cache) == frozen_report['input_sha256']
    assert digest(nominal_path) == frozen_report['variants']['nominal']['output_sha256']
    z = np.load(cache, allow_pickle=False)
    t = lambda k: torch.as_tensor(z[k], dtype=torch.float64)
    m1, m2 = p['mass1'], p['mass2']
    nu = torch.tensor(m1*m2/(m1+m2)**2, dtype=torch.float64)
    report = {'status': 'running', 'case': case, 'host': socket.gethostname(),
              'cwd': str(ROOT), 'pid': os.getpid(), 'command': shlex.join([sys.executable, *sys.argv]),
              'input_sha256': digest(cache), 'nominal_sha256': digest(nominal_path),
              'script_sha256': digest(Path(__file__)), 'protocol_sha256': digest(OUT/'protocol.md'),
              'helper_sha256': digest(ROOT/'scripts/teob_grid_derivative_controls.py'),
              'source_hashes_before': source_hashes(), 'variants': {}}
    def save():
        path.write_text(json.dumps(json_value(report), indent=2)+'\n')
    save()
    print(json.dumps({k: report[k] for k in ('host', 'cwd', 'pid', 'command')}), flush=True)
    begun = time.time()
    try:
        with torch.no_grad():
            target = published_2017_nqc_targets(nu, torch.zeros_like(nu))
            def assemble(values):
                return assemble_teob_merger_nqc(t('nqc_original_time'), t('nqc_original_state'),
                    amplitude=t('nqc_original_amplitude'), phase=t('nqc_original_phase'),
                    orbital_frequency=t('nqc_original_orbital_frequency'),
                    radial_acceleration=t('nqc_original_radial_acceleration'), nu=nu,
                    fixed_step=.1, targets=values, target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
                    matching_method='grid')
            base = assemble(target)
            assert torch.equal(base.auxiliary.time, t('nqc_auxiliary_time'))
            radius = torch.full((4,), 5e-9 if nu == .25 else (1+float(nu))*5e-9, dtype=torch.float64)
            report.update(targets=target, target_box=target_box(base, radius), fixed_step_M=.1,
                          sampled_peak_M=base.t_peak, requested_nqc_time_M=base.t_nqc,
                          nominal_index=base.matching_index)
            def variants():
                yield 'nominal', base, {}
                for j, key in enumerate(TARGETS):
                    for sign in (-1, 1):
                        values = torch.stack(target).clone()
                        values[j] += sign*radius[j]
                        nqc = assemble(TEOBMergerNQCTargets(*values.unbind()))
                        expected_a = report['target_box']['response_amplitude'] @ (values[:2]-torch.stack(target)[:2])
                        expected_b = report['target_box']['response_phase'] @ (values[2:]-torch.stack(target)[2:])
                        error = max(float((nqc.coefficients.amplitude-base.coefficients.amplitude-expected_a).abs().max()),
                                    float((nqc.coefficients.phase-base.coefficients.phase-expected_b).abs().max()))
                        assert error < 2e-10
                        yield f'{key}_{"minus" if sign < 0 else "plus"}', nqc, {
                            'target_delta': values-torch.stack(target), 'coefficient_response_check_max_error': error}
                for offset in (-1, 1):
                    nqc, detail = solve_at_index(base, base.matching_index+offset)
                    yield f'knot_{"previous" if offset < 0 else "next"}', nqc, detail

            mf, spin = [x.double() for x in uib2016_nonspinning_remnant(m1, m2)]
            support = plan_td_support(m1, m2, g['flow'], dtype=torch.float64)
            nominal = np.load(nominal_path, allow_pickle=False)
            base_fd = base_rd = None
            responses = {}
            for name, nqc, detail in variants():
                start = time.time()
                rd = assemble_teob_ringdown_22(nqc.original.time, nqc.original.corrected_amplitude,
                    nqc.original.corrected_phase, nqc.auxiliary.time, nqc.auxiliary.corrected_amplitude,
                    nqc.auxiliary.corrected_phase, nqc.auxiliary.orbital_frequency, nu, mf, spin,
                    output_step=t('sampling_dt_seconds')/t('total_mass_seconds'), matching_method='grid')
                hp, hc = project_teob22_generic(rd.mode, torch.conj(rd.mode), mass1=m1, mass2=m2,
                    distance=p['distance'], inclination=p['inclination'], phic=p['phic'])
                fd = condition_td_to_fd(hp, hc, dt=t('sampling_dt_seconds'), delta_f=g['df'],
                    f_max=g['fhigh'], support=support, tc=p['tc'], return_metadata=True)
                arrays = dict(frequencies=fd.frequencies.numpy(), h_plus_fd=fd.h_plus.numpy(),
                              h_cross_fd=fd.h_cross.numpy(), td_time=rd.time.numpy(),
                              h_plus_td=hp.numpy(), h_cross_td=hc.numpy())
                if name == 'nominal':
                    detail['nominal_bitwise'] = all(np.array_equal(v, nominal[k]) for k, v in arrays.items())
                    assert detail['nominal_bitwise']
                    base_fd, base_rd = fd, rd
                detail.update(coefficients=nqc.coefficients, targets=nqc.targets,
                    selected_index=nqc.matching_index, selected_time_M=nqc.auxiliary.time[nqc.matching_index],
                    attachment_time_M=rd.events.attachment_time,
                    requested_rd_match_M=rd.t_match, rd_peak_M=rd.t_peak,
                    td_grid_identical=torch.equal(rd.time, base_rd.time),
                    attachment_identical=torch.equal(rd.events.attachment_time, base_rd.events.attachment_time))
                assert detail['td_grid_identical'] and detail['attachment_identical']
                mask = (fd.frequencies >= g['flow']) & (fd.frequencies < g['fhigh'])
                detail['fd_change_vs_nominal_in_band'] = {
                    pol: torch.linalg.vector_norm((getattr(fd, pol)-getattr(base_fd, pol))[mask]) /
                    torch.linalg.vector_norm(getattr(base_fd, pol)[mask]) for pol in ('h_plus', 'h_cross')}
                responses[name] = {pol: getattr(fd, pol)[mask].clone() for pol in ('h_plus', 'h_cross')}
                output = out/f'{name}.npz'
                np.savez_compressed(output, **arrays)
                detail.update(output_sha256=digest(output), acceptance=evaluate_cached(fd.frequencies, fd.h_plus, fd.h_cross, case))
                detail['seconds'] = time.time()-start
                report['variants'][name] = detail
                save()
                print(json.dumps({'case': args.case, 'variant': name,
                    'plus': detail['acceptance']['plus']['complex_relative_l2'],
                    'cross': detail['acceptance']['cross']['complex_relative_l2']}), flush=True)
            report['linearized_rounding_fd_triangle_envelope'] = {
                pol: sum(torch.linalg.vector_norm((responses[key+'_plus'][pol]-responses[key+'_minus'][pol])/2)
                         for key in TARGETS) / torch.linalg.vector_norm(responses['nominal'][pol])
                for pol in ('h_plus', 'h_cross')}
            report['rounding_center_symmetry_residual'] = {
                pol: max(torch.linalg.vector_norm((responses[key+'_plus'][pol]+responses[key+'_minus'][pol])/2-responses['nominal'][pol])
                         for key in TARGETS) / torch.linalg.vector_norm(responses['nominal'][pol])
                for pol in ('h_plus', 'h_cross')}
            report['status'] = 'finished'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.time()-begun
        report['source_hashes_after'] = source_hashes()
        report['source_changes'] = [k for k in report['source_hashes_before'].keys() | report['source_hashes_after'].keys()
                                    if report['source_hashes_before'].get(k) != report['source_hashes_after'].get(k)]
        save()
    assert not report['source_changes']


if __name__ == '__main__':
    main()
