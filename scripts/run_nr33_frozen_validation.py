#!/usr/bin/env python3
"""Evaluate the frozen NR pilot on its declared holdouts, then optionally LAL."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from scripts.fit_nr33_amplitude_pilot import (
    FFTControls, frequency_amplitude, independent_qnm33, load_case,
    read_manifest, ringdown_shape,
)
from scripts.validate_nr33_amplitude_pilot import sha256_file, validate_frozen_predictions


def prepare_inputs(calibration, manifest, expected_sha256, output_dir):
    """Read fixed coefficients and holdout waveforms; never call a fitter."""
    if sha256_file(calibration) != expected_sha256:
        raise ValueError('calibration differs from frozen SHA256')
    fit = json.loads(calibration.read_text())
    if sha256_file(manifest) != fit['provenance']['manifest_sha256']:
        raise ValueError('manifest differs from frozen calibration input')
    if sha256_file(Path(__file__).with_name('fit_nr33_amplitude_pilot.py')) != fit['provenance']['fitter_script_sha256']:
        raise ValueError('frozen numerical helper source changed')
    if sha256_file(fit['provenance']['qnm_helper_path']) != fit['provenance']['qnm_helper_sha256']:
        raise ValueError('frozen Kerr helper source changed')
    cases = {case.case_id: case for case in read_manifest(manifest)}
    controls = FFTControls(**fit['controls'])
    parameters = fit['parameter_fit']
    predictions, targets = [], []
    for case_id in fit['split']['holdout_case_ids']:
        case = cases[case_id]
        if case.metadata['split'] != 'holdout':
            raise ValueError('case is not a declared NR holdout')
        time, h33, metadata = load_case(case)
        qnm = independent_qnm33(metadata)
        f, nr, fft = frequency_amplitude(
            time, h33, controls, metadata['relaxation_time'],
            metadata['peak_time'], return_metadata=True)
        band = f >= .95 * qnm['f_ring_33']
        f, nr = f[band], nr[band]
        eta, delta = case.q/(1+case.q)**2, (case.q-1)/(case.q+1)
        x = 4*eta - 1
        log_a = np.polynomial.polynomial.polyval(x, parameters['log_abs_a_lambda_over_delta_coefficients'])
        lam = np.polynomial.polynomial.polyval(x, parameters['lambda_coefficients'])
        a022 = np.pi*np.sqrt(2*eta/3)*(np.pi*f)**(-7/6)
        predicted = a022*delta*np.exp(log_a)*ringdown_shape(
            f, qnm['f_ring_33'], qnm['f_damp_33'], lam, fit['sigma_33'])
        common = {'case_id': case_id, 'split': 'holdout', 'q': case.q,
                  'frequency_Mf': f.tolist(), 'amplitude_units': 'geometric',
                  'm1_msun': 50*case.q/(1+case.q), 'm2_msun': 50/(1+case.q),
                  'chi1': 0., 'chi2': 0., 'distance_mpc': 100.,
                  'spin_approximation': 'near-nonspinning NR treated as zero spin',
                  'qnm': qnm, 'fft': fft, 'input_npz_sha256': sha256_file(case.path)}
        predictions.append({**common, 'prediction_amplitude_geometric': predicted.tolist()})
        targets.append({**common, 'target_amplitude_geometric': nr.tolist()})
    if sha256_file(calibration) != expected_sha256:
        raise ValueError('calibration changed while preparing holdouts')
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, target_path = output_dir/'nr33-frozen-predictions.json', output_dir/'nr33-heldout-targets.json'
    shared = {'frozen': True, 'split': fit['split'],
              'calibration_before_sha256': expected_sha256,
              'calibration_after_sha256': expected_sha256,
              'manifest_sha256': sha256_file(manifest),
              'adapter_sha256': sha256_file(__file__)}
    for path, payload in ((prediction_path, {**shared, 'predictions': predictions}),
                          (target_path, {**shared, 'targets': targets})):
        path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return prediction_path, target_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--with-lal', action='store_true')
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    prediction_path, target_path = prepare_inputs(args.calibration, args.manifest,
                                                 args.expected_sha256, args.output_dir)
    report = validate_frozen_predictions(
        prediction_path, target_path, include_lal=args.with_lal,
        calibration_path=args.calibration, expected_calibration_sha256=args.expected_sha256,
        plot_path=args.output_dir/'nr33-heldout-comparison.png')
    report['run'] = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                     'host': platform.node(), 'cwd': str(Path.cwd()),
                     'python': sys.executable, 'command': sys.argv,
                     'adapter_sha256': sha256_file(__file__),
                     'predictions_sha256': sha256_file(prediction_path),
                     'targets_sha256': sha256_file(target_path)}
    (args.output_dir/'nr33-validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'cases': len(report['cases']), 'calibration_unchanged':
                     report['calibration_before_sha256'] == report['calibration_after_sha256']}))


if __name__ == '__main__':
    main()
