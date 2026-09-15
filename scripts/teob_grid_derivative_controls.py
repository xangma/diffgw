#!/usr/bin/env python3
"""Frozen-time sampled NQC derivative controls; no trajectory regeneration."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

import numpy as np
import torch

from teob_continuous_fd_attribution import (
    OUT as INPUT, ROOT, case_for, evaluate_cached, json_value, source_hashes,
)
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.teob_merger_nqc_clean import (
    assemble_teob_merger_nqc, published_2017_nqc_targets,
    PUBLISHED_2017_NQC_PROVENANCE, _solve_two_by_two,
)
from torchwave.approximants.eob.teob_nqc_clean import NQCCoefficients, nqc_basis_values
from torchwave.approximants.eob.teob_ringdown_assembly_clean import assemble_teob_ringdown_22
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant
from torchwave.approximants.eob.teob_conditioned_waveform_clean import project_teob22_generic
from torchwave.approximants.eob.td_fd_conditioning_clean import condition_td_to_fd, plan_td_support

OUT = ROOT / 'docs/reviews/2026-09-15-teob-grid-nqc'
CASES = ('catalog', 'bbh_reference', 'equal_bbh', 'unequal_bns')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational_first(y, h):
    """Centered degree-four derivative at every node with two neighbors."""
    return ((y[:-4] - y[4:]) + 8 * (y[3:-1] - y[1:-3])) / (12 * h)


def local_derivatives(y, h, direct_second=False):
    if y.shape[0] != 9:
        raise ValueError('nine centered samples required')
    dy = rational_first(y, h)
    if direct_second:
        # Subtract the center first to preserve constant-offset invariance.
        z = y[2:7] - y[4]
        d2 = (-z[0] + 16*z[1] + 16*z[3] - z[4]) / (12*h*h)
    else:
        d2 = rational_first(dy, h)[0]
    return dy[2], d2


def coefficients(nqc, derivatives):
    i = nqc.matching_index
    aw = nqc.auxiliary_amplitude_rw
    target = nqc.targets
    amp_matrix = torch.stack((aw[i] * nqc.auxiliary.q[i], derivatives['product'][0]))
    amp_rhs = torch.stack((target.amplitude - aw[i],
                           target.amplitude_derivative - derivatives['amplitude'][0]))
    phase_matrix = torch.stack(derivatives['basis'])
    phase_rhs = torch.stack((derivatives['phase'][0] - target.frequency,
                            derivatives['phase'][1] - target.frequency_derivative))
    a, ac, ar = _solve_two_by_two(amp_matrix, amp_rhs, max_condition=1e10, name='amplitude')
    b, bc, br = _solve_two_by_two(phase_matrix, phase_rhs, max_condition=1e10, name='phase')
    return NQCCoefficients(a, b), {'amplitude_condition': ac, 'phase_condition': bc,
                                 'amplitude_solve_residual': ar, 'phase_solve_residual': br}


def apply_coefficients(nqc, coeff):
    def corrected(series):
        return replace(series,
            corrected_amplitude=series.amplitude*(1+(series.q*coeff.amplitude).sum(-1)),
            corrected_phase=series.phase-(series.p*coeff.phase).sum(-1))
    return replace(nqc, coefficients=coeff, original=corrected(nqc.original),
                   auxiliary=corrected(nqc.auxiliary))


def make_variants(base, nu):
    i = base.matching_index
    if not 4 <= i < len(base.auxiliary.time)-4:
        raise ValueError('matching point needs nine interior auxiliary nodes')
    aux = base.auxiliary
    values = {'amplitude': base.auxiliary_amplitude_rw,
              'product': base.auxiliary_amplitude_rw[:, None]*aux.q,
              'phase': aux.phase, 'basis': aux.p}
    h = base.auxiliary_spacing
    tm = aux.time[i]
    for name, direct in [('rational', False), ('direct_second', True)]:
        derivatives = {k: local_derivatives(v[i-4:i+5], h, direct) for k, v in values.items()}
        coeff, info = coefficients(base, derivatives)
        yield name, apply_coefficients(base, coeff), info

    derivatives = {}
    for key, value in values.items():
        spline = natural_cubic_spline(aux.time, value, scalar_output=value.ndim == 1)
        derivatives[key] = (spline.derivative(tm), spline.derivative(tm, 2))
    coeff, info = coefficients(base, derivatives)
    yield 'auxiliary_cubic', apply_coefficients(base, coeff), info

    s = slice(base.cut_start_index, base.cut_last_index+1)
    original = base.original
    splines = {}
    for key in ('state', 'amplitude', 'phase', 'orbital_frequency', 'radial_acceleration'):
        value = getattr(original, key)[s]
        splines[key] = natural_cubic_spline(original.time[s], value, scalar_output=value.ndim == 1)
    for step in (.05, .1, .2, .5):
        query = tm + torch.arange(-4, 5, dtype=torch.float64)*step
        sampled = {key: spline(query) for key, spline in splines.items()}
        basis = nqc_basis_values(sampled['state'][:, 0], sampled['state'][:, 2],
                                sampled['orbital_frequency'], sampled['radial_acceleration'])
        aw = sampled['amplitude']/(nu*torch.sqrt(nu.new_tensor(24.)))
        values_local = {'amplitude': aw, 'product': aw[:, None]*basis.amplitude,
                        'phase': sampled['phase'], 'basis': basis.phase}
        derivatives = {k: local_derivatives(v, step) for k, v in values_local.items()}
        coeff, info = coefficients(base, derivatives)
        info['derivative_spacing_M'] = step
        info['matching_time_M'] = tm
        if step == .1:
            info['max_coordinate_delta_vs_auxiliary_M'] = (query-aux.time[i-4:i+5]).abs().max()
            info['max_sample_delta_vs_auxiliary'] = {
                k: (v-values[k][i-4:i+5]).abs().max() for k, v in values_local.items()}
        yield f'local_{step:g}', apply_coefficients(base, coeff), info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = ap.parse_args()
    report = {'status': 'running', 'host': socket.gethostname(), 'cwd': str(ROOT),
              'pid': os.getpid(), 'command': shlex.join([sys.executable, *sys.argv]),
              'source_hashes_before': source_hashes(), 'script_sha256': digest(Path(__file__)),
              'protocol_sha256': digest(OUT/'protocol.md'), 'cases': {},
              'unsupported': {'unequal_bbh': 'nu=0.16 excluded by frozen published target domain'}}
    path = OUT/'derivative-controls.json'
    def save():
        path.write_text(json.dumps(json_value(report), indent=2)+'\n')
    save()
    print(json.dumps({k: report[k] for k in ('status','host','cwd','pid','command')}), flush=True)
    start = time.time()
    try:
        with torch.no_grad():
            for name in args.cases:
                case = case_for(name)
                p, g = case['params'], case['grid']
                cache = INPUT/name/'continuous.npz'
                z = np.load(cache, allow_pickle=False)
                t = lambda key: torch.as_tensor(z[key], dtype=torch.float64)
                old = np.load(INPUT/name/'matching-grid.npz', allow_pickle=False)
                m1, m2 = p['mass1'], p['mass2']
                nu = torch.tensor(m1*m2/(m1+m2)**2, dtype=torch.float64)
                mf, spin = [x.double() for x in uib2016_nonspinning_remnant(m1, m2)]
                support = plan_td_support(m1, m2, g['flow'], dtype=torch.float64)
                def assemble(step):
                    return assemble_teob_merger_nqc(t('nqc_original_time'), t('nqc_original_state'),
                        amplitude=t('nqc_original_amplitude'), phase=t('nqc_original_phase'),
                        orbital_frequency=t('nqc_original_orbital_frequency'),
                        radial_acceleration=t('nqc_original_radial_acceleration'), nu=nu,
                        fixed_step=step, targets=published_2017_nqc_targets(nu, torch.zeros_like(nu)),
                        target_provenance=PUBLISHED_2017_NQC_PROVENANCE, matching_method='grid')
                recovered = t('nqc_auxiliary_time')[1]-t('nqc_auxiliary_time')[0]
                base = assemble(.1)
                item = {'case': case, 'input_sha256': digest(cache),
                        'previous_grid_sha256': digest(INPUT/name/'matching-grid.npz'),
                        'reconstructed_spacing_M': recovered, 'nominal_spacing_M': .1,
                        'selected_index': base.matching_index,
                        'selected_time_M': base.auxiliary.time[base.matching_index],
                        'sampled_peak_M': base.t_peak, 'variants': {}}
                report['cases'][name] = item
                baseline_fd = baseline_time = baseline_rd = None
                variants = [('previous_replay', assemble(recovered), {}), ('nominal', base, {})]
                variants.extend(make_variants(base, nu))
                case_out = OUT/name
                case_out.mkdir(exist_ok=True)
                for variant, nqc, detail in variants:
                    begun = time.time()
                    rd = assemble_teob_ringdown_22(nqc.original.time, nqc.original.corrected_amplitude,
                        nqc.original.corrected_phase, nqc.auxiliary.time, nqc.auxiliary.corrected_amplitude,
                        nqc.auxiliary.corrected_phase, nqc.auxiliary.orbital_frequency, nu, mf, spin,
                        output_step=t('sampling_dt_seconds')/t('total_mass_seconds'), matching_method='grid')
                    hp, hc = project_teob22_generic(rd.mode, torch.conj(rd.mode), mass1=m1, mass2=m2,
                        distance=p['distance'], inclination=p['inclination'], phic=p['phic'])
                    fd = condition_td_to_fd(hp, hc, dt=t('sampling_dt_seconds'), delta_f=g['df'],
                        f_max=g['fhigh'], support=support, tc=p['tc'], return_metadata=True)
                    detail.update(coefficients=nqc.coefficients,
                        coefficient_delta_vs_nominal=[a-b for a,b in zip(nqc.coefficients,base.coefficients)],
                        td_samples=len(hp), rd_peak_M=rd.t_peak, attachment_time_M=rd.events.attachment_time,
                        matching_time_M=nqc.auxiliary.time[nqc.matching_index])
                    if variant == 'previous_replay':
                        detail['previous_replay_bitwise'] = all(np.array_equal(getattr(fd,pol).numpy(),old[pol+'_fd'])
                            for pol in ('h_plus','h_cross')) and np.array_equal(rd.time.numpy(),old['td_time'])
                        assert detail['previous_replay_bitwise'], 'previous baseline replay differs'
                    if variant == 'nominal':
                        baseline_fd, baseline_time, baseline_rd = fd, rd.time, rd
                    if baseline_fd is not None:
                        detail['td_grid_identical'] = torch.equal(rd.time,baseline_time)
                        detail['attachment_identical'] = torch.equal(rd.events.attachment_time,baseline_rd.events.attachment_time)
                        assert detail['td_grid_identical'] and detail['attachment_identical']
                        detail['fd_change_vs_nominal'] = {pol: torch.linalg.vector_norm(getattr(fd,pol)-getattr(baseline_fd,pol))/torch.linalg.vector_norm(getattr(baseline_fd,pol))
                            for pol in ('h_plus','h_cross')}
                    output = case_out/f'{variant}.npz'
                    np.savez_compressed(output, frequencies=fd.frequencies.numpy(), h_plus_fd=fd.h_plus.numpy(),
                        h_cross_fd=fd.h_cross.numpy(), td_time=rd.time.numpy(), h_plus_td=hp.numpy(), h_cross_td=hc.numpy())
                    detail['output_sha256'] = digest(output)
                    detail['acceptance'] = evaluate_cached(fd.frequencies, fd.h_plus, fd.h_cross, case)
                    detail['seconds'] = time.time()-begun
                    item['variants'][variant] = detail
                    save()
                    print(json.dumps({'case': name, 'variant': variant,
                        'plus': detail['acceptance']['plus']['complex_relative_l2'],
                        'cross': detail['acceptance']['cross']['complex_relative_l2'],
                        'seconds': detail['seconds']}), flush=True)
        report['status'] = 'finished'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.time()-start
        report['source_hashes_after'] = source_hashes()
        report['source_changes'] = [p for p,h in report['source_hashes_before'].items()
                                    if report['source_hashes_after'].get(p) != h]
        save()
    assert not report['source_changes']


if __name__ == '__main__':
    main()
