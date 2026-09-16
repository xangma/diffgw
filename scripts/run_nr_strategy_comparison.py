"""Prospectively selected, NR-only conditional complex-waveform experiment.

Train and evaluate are separate commands. Evaluation reads frozen model state
and cannot call a fitter. All scores use one common orbital/time gauge.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_nr33_amplitude_pilot import (
    FFTControls, fit_parameter_space, independent_qnm33, ringdown_shape,
)
from scripts.nr33_phase_data import (
    load_complex_case, transform_complex, interpolate_complex_common_grid,
)
from scripts.nr33_phase_metrics import phase_metrics
from scripts.nr_complex_reduced_basis import ComplexReducedBasisModel, fit_complex_basis

BASE = Path('docs/reviews/2026-09-15-parity-strategy-experiment')
OLD = Path('docs/reviews/2026-09-15-independent-calibration-v3-search')
N_BINS = 512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_freeze():
    for path, digest in read(BASE / 'protocol-freeze.json')['sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'pre-experiment frozen input changed: {path}')


def canonical_modes(time, h22, h33):
    """A single orbital rotation; the pi branch also rotates all odd modes."""
    peak_index = int(np.argmax(np.abs(h22)))
    orbital = float(np.angle(h22[peak_index]) / 2)
    rotated33 = h33[peak_index] * np.exp(-3j * orbital)
    if abs(rotated33.imag) < 1e-8 * abs(rotated33):
        raise ValueError('orbital pi branch is ill-conditioned at h22 peak')
    if rotated33.imag < 0:
        orbital += np.pi
    modes = np.stack((h22 * np.exp(-2j * orbital), h33 * np.exp(-3j * orbital)))
    return modes, float(time[peak_index]), orbital


def frequency_and_factor(q, qnm):
    fr, fd = qnm['f_ring_33'], qnm['f_damp_33']
    # The upper endpoint is omitted so interpolation never extrapolates an FFT.
    f = .95 * fr + np.arange(N_BINS) / N_BINS * (.3 - .95 * fr)
    eta = q / (1 + q)**2
    a0 = np.pi * np.sqrt(2 * eta / 3) * (np.pi * f)**(-7 / 6)
    oscillator = fd / (fd + 1j * (f - fr))
    factor = np.stack((a0 * oscillator, a0 * oscillator * (q - 1) / (q + 1)))
    return f, factor, a0


def prepare(row, *, qnm=None, controls=None):
    controls = controls or FFTControls()
    data = load_complex_case(row['path'], expected_sha256=row['npz_sha256'], metadata=row['metadata'])
    if data.h22 is None:
        raise ValueError('both modes are needed for a shared gauge')
    modes, peak, orbital = canonical_modes(data.time, data.h22, data.h33)
    qnm = independent_qnm33(row['metadata']) if qnm is None else qnm
    f, factor, a0 = frequency_and_factor(float(row['q']), qnm)
    spectra = []
    for mode in modes:
        transformed = transform_complex(data.time, mode, controls,
            relaxation_time=row['metadata']['relaxation_time'], h22_peak_time=peak)
        spectra.append(interpolate_complex_common_grid(
            transformed.frequency_Mf, transformed.spectrum, f))
    h = np.stack(spectra)
    return {'id': row['id'], 'sxs_id': row['sxs_id'], 'q': float(row['q']),
            'f': f, 'h': h, 'factor': factor, 'values': h / factor, 'a0': a0,
            'qnm': qnm, 'peak_time': peak, 'orbital_rotation': orbital,
            'source_sha256': row['npz_sha256']}


def relative_l2(model, target):
    return float(np.linalg.norm(model - target) / np.linalg.norm(target))


def scores(model, target):
    per_mode = {}
    for i, mode in enumerate(('22', '33')):
        per_mode[mode] = {'complex_l2': relative_l2(model[i], target[i]),
                         'amplitude_l2': relative_l2(abs(model[i]), abs(target[i]))}
    # Ratio of ratios preserves inter-mode phase; no phase/time optimization.
    mask = np.all(abs(target) >= 1e-3 * abs(target).max(axis=1, keepdims=True), axis=0)
    phase_delta = np.angle(model[1] * np.conj(target[1]) * np.conj(model[0]) * target[0])
    weights = abs(target[1, mask])**2
    phase_rms = float(np.sqrt(np.average(phase_delta[mask]**2, weights=weights)))
    return {'modes': per_mode,
            'mean_mode_complex_l2': float(np.mean([v['complex_l2'] for v in per_mode.values()])),
            'relative_mode_phase_rms_rad': phase_rms,
            'coherent_equal_weight_sum_l2': relative_l2(model.sum(axis=0), target.sum(axis=0))}


def quality(primary, siblings):
    rows = []
    for row in siblings:
        if row['id'] == primary['id']:
            continue
        control = prepare(row, qnm=primary['qnm'])
        # Tiny metadata-q differences change f grids: explicitly resample.
        h = np.stack([interpolate_complex_common_grid(control['f'], v, primary['f'])
                      for v in control['h']])
        rows.append((row['id'], h, .01))
    source = next(r for r in siblings if r['id'] == primary['id'])
    for name, controls in [('dt_0.05', replace(FFTControls(), sample_dt=.05)),
                           ('tail_40', replace(FFTControls(), end_taper_duration=40))]:
        rows.append((name, prepare(source, qnm=primary['qnm'], controls=controls)['h'], .002))
    reports = []
    for name, h, threshold in rows:
        result = scores(h, primary['h'])
        phase = {}
        for i, mode in enumerate(('22', '33')):
            mask = abs(primary['h'][i]) >= 1e-3 * abs(primary['h'][i]).max()
            phase[mode] = phase_metrics(primary['f'], h[i], primary['h'][i], phase_mask=mask)
        result.update(control=name, phase=phase, phase_threshold_rad=threshold,
                      phase_diagnostic_pass=all(v['affine_phase_rms'] <= threshold for v in phase.values()))
        reports.append(result)
    return {'id': primary['id'], 'comparisons': reports,
            'phase_diagnostic_pass': all(r['phase_diagnostic_pass'] for r in reports)}


def mode_scales(cases):
    values = np.stack([c['values'] for c in cases])
    return np.sqrt(np.mean(abs(values)**2, axis=(0, 2)))[:, None]


def basis_fit(cases, rank, degree):
    scales = mode_scales(cases)
    model = fit_complex_basis(torch.tensor([c['q'] for c in cases], dtype=torch.float64),
        torch.tensor(np.stack([c['values'] / scales for c in cases])), rank=rank, degree=degree)
    return model, scales


def basis_predict(model, scales, case):
    # Endpoint CV folds and q10 regression intentionally report extrapolation.
    return model.predict(torch.tensor(case['q'], dtype=torch.float64),
                         allow_extrapolation=True).detach().numpy() * scales * case['factor']


def polar_fit(cases, direct, degree):
    ordered = sorted(cases, key=lambda c: c['q'])
    q = np.array([c['q'] for c in ordered])
    x = 4 * q / (1 + q)**2 - 1
    values = np.stack([c['values'] for c in ordered])
    phase = np.unwrap(np.unwrap(np.angle(values), axis=-1), axis=0)
    logamp = np.log(np.maximum(abs(values), np.finfo(float).tiny))
    design = np.vander(x, degree + 1, increasing=True)
    coefficients = [np.linalg.lstsq(design, v.reshape(len(cases), -1), rcond=None)[0]
                    .reshape(degree + 1, 2, N_BINS).tolist() for v in (logamp, phase)]
    ids = {c['id'] for c in cases}
    return {'degree': degree, 'logamp': coefficients[0], 'phase': coefficients[1],
            'amplitude': fit_parameter_space([d for d in direct if d['case_id'] in ids], degree=2)}


def polar_predict(model, case):
    q = case['q']
    x = 4 * q / (1 + q)**2 - 1
    powers = x**np.arange(model['degree'] + 1)
    logamp = np.einsum('k,kmb->mb', powers, model['logamp'])
    phase = np.einsum('k,kmb->mb', powers, model['phase'])
    result = np.exp(logamp + 1j * phase) * case['factor']
    params = model['amplitude']
    log_a = np.polynomial.polynomial.polyval(x, params['log_abs_a_lambda_over_delta_coefficients'])
    lam = np.polynomial.polynomial.polyval(x, params['lambda_coefficients'])
    amplitude = case['a0'] * (q - 1) / (q + 1) * np.exp(log_a) * ringdown_shape(
        case['f'], case['qnm']['f_ring_33'], case['qnm']['f_damp_33'], lam, 1.3)
    result[1] = amplitude * np.exp(1j * (phase[1] + np.angle(case['factor'][1])))
    return result


def source_freeze():
    paths = [Path(__file__), Path('scripts/nr_complex_reduced_basis.py'),
             Path('scripts/nr33_phase_data.py'), Path('scripts/nr33_phase_metrics.py'),
             Path('scripts/fit_nr33_amplitude_pilot.py'), Path('scripts/kerr_leaver_solver.py'),
             BASE / 'protocol.md', BASE / 'protocol-freeze.json']
    return {str(p): sha(p) for p in paths}


def train(out):
    verify_freeze()
    rows = [r for r in read(OLD / 'nr33-qualified-manifest.json')['cases'] if r['split'] == 'training']
    if len(rows) != 5:
        raise ValueError('experiment requires the five predeclared training families')
    controls = read(OLD / 'nr33-manifest-round1.json')['cases']
    direct = read(OLD / 'nr33-calibration.json')['direct_train']
    if {r['id'] for r in rows} != {d['case_id'] for d in direct}:
        raise ValueError('baseline direct fits differ from training identities')
    cases, diagnostics = [], []
    for row in rows:
        print('prepare training', row['id'], flush=True)
        case = prepare(row)
        diagnostics.append(quality(case, [r for r in controls if r['sxs_id'] == row['sxs_id']]))
        cases.append(case)
    write_new(out / 'training-quality.json', diagnostics)
    trials = []
    for kind, rank, degree in ([('complex_basis', r, d) for r in (1, 2, 3) for d in (1, 2)]
                               + [('eq42_polar', None, d) for d in (1, 2)]):
        folds = []
        for i, target in enumerate(cases):
            subset = cases[:i] + cases[i+1:]
            if kind == 'complex_basis':
                model, scales = basis_fit(subset, rank, degree)
                prediction = basis_predict(model, scales, target)
            else:
                model = polar_fit(subset, direct, degree)
                prediction = polar_predict(model, target)
            result = scores(prediction, target['h'])
            result.update(id=target['id'], q=target['q'],
                          extrapolation=not min(c['q'] for c in subset) <= target['q'] <= max(c['q'] for c in subset))
            folds.append(result)
        trials.append({'kind': kind, 'rank': rank, 'degree': degree, 'folds': folds,
                       'selection_score': float(np.mean([f['mean_mode_complex_l2'] for f in folds]))})
    winners = {kind: min((t for t in trials if t['kind'] == kind),
                         key=lambda t: (t['selection_score'], t['rank'] or 0, t['degree']))
               for kind in ('complex_basis', 'eq42_polar')}
    winner = winners['complex_basis']
    model, scales = basis_fit(cases, winner['rank'], winner['degree'])
    polar = polar_fit(cases, direct, winners['eq42_polar']['degree'])
    # Preserve the exact full-training amplitude calibration for final scoring.
    polar['amplitude'] = read(OLD / 'nr33-calibration.json')['parameter_fit']
    gradient = []
    for q in (2.5, 3.5, 6.5):
        query = torch.tensor(q, dtype=torch.float64)
        _, jvp = torch.func.jvp(model.predict, (query,), (torch.ones_like(query),))
        eps = 1e-5
        fd = (model.predict(query + eps) - model.predict(query - eps)) / (2 * eps)
        gradient.append({'q': q, 'relative_jvp_error': float(torch.linalg.norm(jvp - fd) / torch.linalg.norm(fd))})
    flat = np.stack([c['values'] / scales for c in cases]).reshape(len(cases), -1)
    u, s, vh = np.linalg.svd(flat, full_matrices=False)
    projection = (u[:, :model.rank] * s[:model.rank]) @ vh[:model.rank]
    artifact = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_freeze(),
                'training_ids': [c['id'] for c in cases], 'training_npz_sha256': {r['path']: r['npz_sha256'] for r in rows},
                'basis': model.to_dict(), 'mode_scales': scales.tolist(), 'polar': polar,
                'winners': winners, 'gradient_checks': gradient,
                'training_projection_relative_l2': relative_l2(projection, flat),
                'full_rank_reconstruction_relative_l2': relative_l2((u * s) @ vh, flat),
                'all_training_phase_diagnostics_pass': all(d['phase_diagnostic_pass'] for d in diagnostics)}
    write_new(out / 'cross-validation.json', trials)
    write_new(out / 'models.json', artifact)
    write_new(out / 'model-freeze.json', {'model_sha256': sha(out / 'models.json'),
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': artifact['source_sha256']})
    print(json.dumps({k: {f: v[f] for f in ('rank', 'degree', 'selection_score')} for k, v in winners.items()}), flush=True)


def evaluate(out, manifest, label):
    verify_freeze()
    artifact = read(out / 'models.json')
    if sha(out / 'models.json') != read(out / 'model-freeze.json')['model_sha256']:
        raise ValueError('frozen model changed')
    for path, digest in artifact['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('frozen executable source changed: ' + path)
    payload = read(manifest)
    rows = payload['cases']
    primary_ids = payload.get('primary_ids') or payload.get('selected_primary_case_ids')
    if primary_ids is None:
        primary_ids = [r['id'] for r in rows if r.get('split') == 'holdout']
    if not primary_ids or len(set(primary_ids)) != len(primary_ids):
        raise ValueError('need distinct declared evaluation primary IDs')
    train_families = {i.split('/')[0] for i in artifact['training_ids']}
    selected = [r for r in rows if r['id'] in primary_ids]
    if len(selected) != len(primary_ids) or any(r['sxs_id'] in train_families for r in selected):
        raise ValueError('evaluation overlaps training or misses declared IDs')
    model = ComplexReducedBasisModel.from_dict(artifact['basis'])
    scales = np.array(artifact['mode_scales'])
    results = []
    for row in selected:
        print('evaluate', label, row['id'], flush=True)
        case = prepare(row)
        results.append({'id': row['id'], 'q': row['q'], 'input_sha256': row['npz_sha256'],
            'extrapolation': not model.q_min <= row['q'] <= model.q_max,
            'basis': scores(basis_predict(model, scales, case), case['h']),
            'baseline': scores(polar_predict(artifact['polar'], case), case['h']),
            'quality': quality(case, [r for r in rows if r['sxs_id'] == row['sxs_id']]),
            'gauge': {'peak_time': case['peak_time'], 'orbital_rotation': case['orbital_rotation']},
            'frequency_band': [float(case['f'][0]), float(case['f'][-1])]})
    write_new(out / (label + '.json'), {'label': label, 'manifest_sha256': sha(manifest),
        'model_sha256': sha(out / 'models.json'), 'results': results})
    print(json.dumps([{k: r[k] for k in ('id', 'q', 'basis', 'baseline')} for r in results]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--output', type=Path, default=BASE / 'nr')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--label', default='regression')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train':
        train(args.output)
    else:
        if args.manifest is None:
            parser.error('evaluate requires --manifest')
        evaluate(args.output, args.manifest, args.label)


if __name__ == '__main__':
    main()
