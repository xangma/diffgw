#!/usr/bin/env python3
"""Run the unchanged NR33 fitter only after the predeclared quality gates pass."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from scripts import fit_nr33_amplitude_pilot as pilot
from scripts import kerr_leaver_solver


POLICY_SHA = '1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d'
FITTER_SHA = 'e307ef505094a5f3d568690d3a9f44cff58d9683d3c29f91e62cc7d2f17ed554'
KERR_SHA = 'a7b07b28962b676271645fd60cf6bd7c2684c8c92e801579897f2d8f84698fc9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Refuse to overwrite either a completed freeze or partial historical run.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_inputs(policy_path, quality_path, manifest_path):
    if sha(policy_path) != POLICY_SHA:
        raise ValueError('predeclared quality policy changed')
    if sha(pilot.__file__) != FITTER_SHA:
        raise ValueError('frozen numerical fitter changed')
    if sha(kerr_leaver_solver.__file__) != KERR_SHA:
        raise ValueError('frozen independent QNM solver changed')
    policy = json.loads(Path(policy_path).read_text())
    quality = json.loads(Path(quality_path).read_text())
    if quality.get('overall_pass') is not True:
        raise ValueError('NR quality screen did not pass; no fit allowed')
    if quality.get('policy_sha256') != POLICY_SHA:
        raise ValueError('quality report used another policy')
    qualified = quality.get('qualified_manifest') or {}
    if qualified.get('sha256') != sha(manifest_path):
        raise ValueError('qualified manifest hash mismatch')
    payload = json.loads(Path(manifest_path).read_text())
    rows = [r for r in payload['cases'] if r['split'] != 'validation_only']
    if len(rows) != 7 or len({r['sxs_id'] for r in rows}) != 7:
        raise ValueError('exactly seven distinct selected simulations required')
    if sorted(float(r['target_q']) for r in rows) != policy['target_mass_ratios']:
        raise ValueError('selected target mass ratios differ from predeclared design')
    if set(r['id'] for r in rows) != set(quality['selected_primary_case_ids']):
        raise ValueError('manifest primary cases differ from quality report')
    passed = {r['primary_case_id'] for r in quality['simulations'] if r.get('passed') is True}
    if not {r['id'] for r in rows}.issubset(passed):
        raise ValueError('a selected simulation has not passed its quality gates')
    if any(r['sxs_id'] in policy['excluded_sxs_ids'] for r in rows):
        raise ValueError('historical pilot simulation cannot enter new fit')
    for row in rows:
        split = 'holdout' if row['target_q'] in policy['holdout_target_q'] else 'training'
        if row['split'] != split or row['extrapolation_order'] != 3:
            raise ValueError('primary extraction order or declared split changed')
    cases = pilot.read_manifest(Path(manifest_path))
    train, holdout = pilot.split_by_q(cases)
    expected_train = {r['id'] for r in rows if r['split'] == 'training'}
    expected_holdout = {r['id'] for r in rows if r['split'] == 'holdout'}
    if {c.case_id for c in train} != expected_train or {c.case_id for c in holdout} != expected_holdout:
        raise ValueError('legacy fitter split differs from predeclared simulation split')
    return policy, cases, train, holdout


def run(policy_path, quality_path, manifest_path, output):
    policy_path, quality_path, manifest_path, output = map(Path, (policy_path, quality_path, manifest_path, output))
    policy, cases, train, holdout = check_inputs(policy_path, quality_path, manifest_path)
    controls = pilot.FFTControls(**policy['controls'])
    before = {str(p): sha(p) for p in (policy_path, quality_path, manifest_path,
                                      Path(pilot.__file__), Path(kerr_leaver_solver.__file__))}
    split_path = output.with_name(output.stem + '-split.json')
    freeze_path = output.with_name(output.stem + '-freeze.json')
    sidecar = output.with_suffix(output.suffix + '.sha256')
    if any(p.exists() for p in (output, split_path, freeze_path, sidecar)):
        raise FileExistsError('new calibration output paths must not already exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    split = {'rule': 'predeclared target q=3,6 holdout; q=1.5,2,4,5,8 training; NR-only quality selection',
             'train_case_ids': [c.case_id for c in train], 'holdout_case_ids': [c.case_id for c in holdout],
             'excluded_case_ids': [c.case_id for c in cases if c not in train and c not in holdout],
             'frozen_at_utc': started, 'input_sha256': before, 'controls': asdict(controls)}
    write_json(split_path, split)
    fit = pilot.fit_manifest(manifest_path, controls)
    if {c['case_id'] for c in fit['direct_train']} != set(split['train_case_ids']):
        raise RuntimeError('actual fitting inputs differ from frozen split')
    if any(sha(p) != h for p, h in before.items()):
        raise RuntimeError('input changed while fitting')
    fit['split']['rule'] = split['rule']
    fit['provenance']['quality_report_sha256'] = sha(quality_path)
    fit['provenance']['quality_policy_sha256'] = POLICY_SHA
    fit['provenance']['orchestrator_sha256'] = sha(__file__)
    write_json(output, fit)
    digest = sha(output)
    with sidecar.open('x') as stream:
        stream.write(digest + '  ' + output.name + '\n')
    write_json(freeze_path, {'status': 'fit_complete', 'inputs_frozen_at_utc': started,
        'fit_completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'calibration_path': str(output.resolve()), 'calibration_sha256': digest,
        'calibration_sidecar': str(sidecar.resolve()), 'split_path': str(split_path.resolve()),
        'split_sha256': sha(split_path), 'manifest_path': str(manifest_path.resolve()),
        'manifest_sha256': sha(manifest_path), 'fitter_script_sha256': FITTER_SHA,
        'input_sha256': before, 'controls': asdict(controls), 'host': platform.node(),
        'cwd': str(Path.cwd()), 'command': sys.argv, 'python_executable': sys.executable,
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'scipy_version': scipy.__version__, 'orchestrator_sha256': sha(__file__)})
    return {'calibration': str(output), 'sha256': digest, 'training': len(train), 'holdout': len(holdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.policy, args.quality, args.manifest, args.output)))


if __name__ == '__main__':
    main()
