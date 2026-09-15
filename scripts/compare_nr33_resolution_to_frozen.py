#!/usr/bin/env python3
"""Post-hoc q3 resolution comparison against frozen predictions and public LAL."""

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude
from scripts.validate_nr33_amplitude_pilot import (
    lal_one_mode_amplitude, physical_to_geometric, total_mass_seconds,
)


BASE = Path('docs/reviews/2026-09-15-independent-calibration')
OUT = BASE / 'extraction-diagnosis'
FROZEN_SHA = 'f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_l2(prediction, target):
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def main():
    assert sha(BASE / 'nr33-calibration.json') == FROZEN_SHA
    fit = json.loads((BASE / 'nr33-calibration.json').read_text())
    assert sha(BASE / 'nr33-data-manifest.json') == fit['provenance']['manifest_sha256']
    assert sha('scripts/fit_nr33_amplitude_pilot.py') == fit['provenance']['fitter_script_sha256']
    prediction_path = BASE / 'nr33-frozen-predictions.json'
    validation = json.loads((BASE / 'nr33-validation.json').read_text())
    assert sha(prediction_path) == validation['run']['predictions_sha256']
    prediction = next(r for r in json.loads(prediction_path.read_text())['predictions']
                      if r['case_id'] == 'SXS:BBH:0030/Lev3/N3')
    frequency = np.asarray(prediction['frequency_Mf'])
    frozen = np.asarray(prediction['prediction_amplitude_geometric'])
    import lal
    distance = 100 * 1e6 * lal.PC_SI
    physical, api = lal_one_mode_amplitude(
        frequency, m1_msun=prediction['m1_msun'], m2_msun=prediction['m2_msun'],
        distance_m=distance)
    assert api['xhm_release_version'] == 122022
    lal_amp = physical_to_geometric(
        physical, total_mass_seconds(prediction['m1_msun'], prediction['m2_msun']), distance)
    resolution_path = OUT / 'resolution-diagnosis.json'
    resolution = json.loads(resolution_path.read_text())
    metrics, plot_curves = [], []
    for row in resolution['cases']:
        assert sha(row['npz']) == row['npz_sha256']
        with np.load(row['npz'], allow_pickle=False) as z:
            f, nr = frequency_amplitude(z['time'], z['h33'], FFTControls(**fit['controls']),
                                       relaxation_time=260., peak_time=row['peak_time'])
        keep = (frequency >= f[0]) & (frequency <= f[-1])
        grid = frequency[keep]
        target = np.interp(grid, f, nr)
        metrics.append({'id': row['id'], 'npz_sha256': row['npz_sha256'],
                        'bins': int(keep.sum()), 'frequency_Mf_range': [float(grid[0]), float(grid[-1])],
                        'fit_vs_NR': relative_l2(frozen[keep], target),
                        'NR_vs_LAL': relative_l2(target, lal_amp[keep]),
                        'fit_vs_LAL': relative_l2(frozen[keep], lal_amp[keep])})
        if row['id'] in ('SXS:BBH:0030/Lev3/N3', 'SXS:BBH:0030/Lev5/N3'):
            plot_curves.append((row['short'], grid, target))
    baseline = next(r for r in metrics if r['id'] == prediction['case_id'])
    prior = next(r for r in validation['cases'] if r['case_id'] == prediction['case_id'])
    assert abs(baseline['fit_vs_NR'] - prior['relative_amp_l2']) < 1e-12
    assert abs(baseline['NR_vs_LAL'] - prior['lal_validation']['target_vs_lal']['relative_amp_l2']) < 1e-12
    assert sha(BASE / 'nr33-calibration.json') == FROZEN_SHA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, f, a in plot_curves:
        ax.semilogy(f, a, label='NR ' + label)
    ax.semilogy(frequency, frozen, '--', label='Frozen independent fit')
    ax.semilogy(frequency, lal_amp, ':', label='LAL XHM 122022')
    ax.set(xlabel='Dimensionless frequency Mf', ylabel='Geometric Fourier amplitude |h33|',
           title='q≈3: changing NR resolution, with predictions fixed', ylim=(1e-5, 2))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    plot_path = OUT / 'resolution-versus-frozen.png'
    fig.savefig(plot_path, dpi=160); plt.close(fig)
    report = {'purpose': 'post-hoc diagnosis, not a fresh independent validation; no refitting',
              'metric': 'geometric amplitude ||prediction-target||/||target||; target is second named operand',
              'controls': fit['controls'], 'relaxation_time_fixed_M': 260.,
              'limitations': ['fixed original q/spins/frequency band/relaxation test resolution sensitivity; record duration and native relaxation differ, so this is not formal convergence',
                             'linear amplitude interpolation on frozen frequency bins, no extrapolation',
                             'Lev4 lacks the last frozen frequency bin; each row records its exact overlap',
                             'higher resolution is not exact NR truth or an independent simulation'],
              'cases': metrics, 'lal_api': api, 'calibration_sha256': FROZEN_SHA,
              'baseline_metrics_reproduced_within': 1e-12,
              'input_sha256': {str(p): sha(p) for p in (prediction_path, resolution_path, BASE / 'nr33-validation.json')},
              'run': {'host': platform.node(), 'cwd': str(Path.cwd()), 'command': sys.argv, 'python': sys.executable},
              'source_sha256': {str(p): sha(p) for p in (Path(__file__), Path('scripts/validate_nr33_amplitude_pilot.py'))},
              'plot': {'path': str(plot_path), 'sha256': sha(plot_path)}}
    (OUT / 'resolution-versus-frozen.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
