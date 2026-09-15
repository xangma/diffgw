#!/usr/bin/env python3
"""Compare NR preprocessing choices after calibration, without refitting."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.fit_nr33_amplitude_pilot import (
    FFTControls, frequency_amplitude, independent_qnm33,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(row, controls):
    if digest(row['path']) != row['npz_sha256']:
        raise ValueError('NR input differs from manifest: ' + row['id'])
    with np.load(row['path'], allow_pickle=False) as data:
        return frequency_amplitude(
            data['time'], data['h33'], controls,
            relaxation_time=row['relaxation_time'],
            peak_time=row['peak_time'],
        )


def compare(base_f, base_a, other_f, other_a, lower):
    # Compare on baseline FFT bins within both spectra; never extrapolate.
    keep = ((base_f >= max(lower, other_f[0])) &
            (base_f <= other_f[-1]))
    f, base = base_f[keep], base_a[keep]
    if len(f) < 8 or np.linalg.norm(base) == 0:
        raise ValueError('insufficient nonzero common-band NR data')
    other = np.interp(f, other_f, other_a)
    return {
        'frequency_Mf_range': [float(f[0]), float(f[-1])],
        'bins': len(f),
        'relative_amplitude_l2': float(np.linalg.norm(other-base)/np.linalg.norm(base)),
        'norm_ratio_variant_over_baseline': float(np.linalg.norm(other)/np.linalg.norm(base)),
        'max_absolute_amplitude_difference': float(np.max(abs(other-base))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    before = digest(args.calibration)
    if before != args.expected_sha256:
        raise ValueError('calibration differs from the independently recorded frozen hash')
    fit = json.loads(args.calibration.read_text())
    controls = FFTControls(**fit['controls'])
    rows = json.loads(args.manifest.read_text())['cases']
    baseline_rows = [r for r in rows if r['split'] in ('training', 'holdout')]
    variants = {
        'start_taper_250M': replace(controls, start_taper_duration=250.),
        'start_taper_1000M': replace(controls, start_taper_duration=1000.),
        'end_taper_10M': replace(controls, end_taper_duration=10.),
        'end_taper_40M': replace(controls, end_taper_duration=40.),
        'resample_dt_0p05M': replace(controls, sample_dt=.05),
    }
    results = []
    extraction_curves = []
    for row in baseline_rows:
        metadata = dict(row['metadata'])
        metadata['final_spin'] = row['final_spin_magnitude']
        qnm = independent_qnm33(metadata)
        lower = .95*qnm['f_ring_33']
        base_f, base_a = spectrum(row, controls)
        comparisons = {}
        for name, variant in variants.items():
            other_f, other_a = spectrum(row, variant)
            comparisons[name] = compare(base_f, base_a, other_f, other_a, lower)
        alternatives = [r for r in rows if r['split'] == 'validation_only'
                        and r['sxs_id'] == row['sxs_id']]
        for alternative in alternatives:
            other_f, other_a = spectrum(alternative, controls)
            key = 'NR_extraction_N'+str(alternative['extrapolation_order'])
            comparisons[key] = compare(base_f, base_a, other_f, other_a, lower)
            comparisons[key]['input_sha256'] = digest(alternative['path'])
            keep = (base_f >= max(lower, other_f[0])) & (base_f <= other_f[-1])
            extraction_curves.append((row['sxs_id'], base_f[keep], base_a[keep],
                                      np.interp(base_f[keep], other_f, other_a)))
        results.append({'case_id': row['sxs_id'], 'q': row['q'], 'split': row['split'],
                        'input_sha256': digest(row['path']), 'comparisons': comparisons})
    after = digest(args.calibration)
    if before != after:
        raise RuntimeError('calibration changed during NR audit')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(extraction_curves), figsize=(10, 3.5), squeeze=False)
        for ax, (case_id, f, n3, n2) in zip(axes[0], extraction_curves):
            ax.semilogy(f, n3, label='N3 (pilot baseline)')
            ax.semilogy(f, n2, label='N2 (validation only)')
            ax.set(title=case_id, xlabel='Dimensionless frequency Mf',
                   ylabel='Geometric Fourier amplitude |h33|')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    report = {'purpose': 'post-freeze NR sensitivity; no fitting or LAL output',
              'baseline_controls': asdict(controls),
              'calibration_before_sha256': before, 'calibration_after_sha256': after,
              'manifest_sha256': digest(args.manifest), 'cases': results,
              'plot': None if not args.plot else {'path': str(args.plot), 'sha256': digest(args.plot)},
              'source_sha256': {str(Path(__file__).name): digest(__file__),
                                'fit_nr33_amplitude_pilot.py': digest(Path(__file__).with_name('fit_nr33_amplitude_pilot.py'))},
              'limitations': ['N2 versus N3 does not bound all NR errors',
                              'no numerical-resolution convergence campaign',
                              'linear spectral interpolation onto common baseline bins']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'cases': len(results), 'calibration_unchanged': before == after}))


if __name__ == '__main__':
    main()
