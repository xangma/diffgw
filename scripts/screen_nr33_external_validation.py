"""Apply the unchanged per-simulation NR quality screen to external holdouts."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import screen_nr33_calibration_v2 as original


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def screen(manifest_path, protocol_path, expected_protocol_sha256, output, qualified_output):
    manifest_path, protocol_path, output, qualified_output = map(
        Path, (manifest_path, protocol_path, output, qualified_output))
    if output.exists() or qualified_output.exists():
        raise FileExistsError('external screen outputs must be new')
    if original.sha256(protocol_path) != expected_protocol_sha256:
        raise ValueError('external validation protocol changed')
    protocol = json.loads(protocol_path.read_text())
    policy_path = Path(protocol['quality_policy_path'])
    policy = original.load_policy(policy_path)
    if original.sha256(policy_path) != protocol['quality_policy_sha256']:
        raise ValueError('numerical quality policy changed')
    calibration_path = Path(protocol['calibration_path'])
    if original.sha256(calibration_path) != protocol['calibration_sha256']:
        raise ValueError('frozen calibration changed')
    original_manifest = Path(protocol['original_fit_manifest_path'])
    if original.sha256(original_manifest) != protocol['original_fit_manifest_sha256']:
        raise ValueError('original fit manifest changed')
    prior = json.loads(original_manifest.read_text())['cases']
    excluded = set(protocol['excluded_sxs_ids']) | {r['sxs_id'] for r in prior}
    rows = original.read_manifest(manifest_path)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate external waveform ID')
    grouped = defaultdict(list)
    targets = set(protocol['target_mass_ratios'])
    for row in rows:
        if row['sxs_id'] in excluded:
            raise ValueError('external data overlap prior simulations')
        if row['target_q'] not in targets or int(row['candidate_rank']) < 1:
            raise ValueError('undeclared target or candidate rank')
        if row['split'] not in {'holdout', 'validation_only'}:
            raise ValueError('external data must not contain training rows')
        if original.sha256(Path(row['path'])) != row['npz_sha256']:
            raise ValueError('external waveform hash mismatch')
        grouped[row['sxs_id']].append(row)
    simulations = []
    by_id = {r['id']: r for r in rows}
    for sid, group in grouped.items():
        declarations = {(r['target_q'], r['candidate_rank']) for r in group}
        if len(declarations) != 1:
            raise ValueError('inconsistent target or rank within simulation')
        target, rank = declarations.pop()
        result = original.screen_simulation(group, policy)
        result.update(target_q=target, candidate_rank=rank)
        primary = by_id.get(result['primary_case_id'])
        declared_primaries = [r for r in group if r['split'] == 'holdout']
        split_pass = primary is not None and declared_primaries == [primary] and primary['split'] == 'holdout'
        q_pass = primary is not None and abs(float(primary['q']) / target - 1) <= protocol['max_relative_q_deviation']
        result['gates'].extend([
            {'name': 'external_holdout_split', 'pass': split_pass, 'required': True},
            {'name': 'declared_target_q', 'pass': q_pass, 'required': True,
             'evidence': primary['q'] if primary else None,
             'limit_relative': protocol['max_relative_q_deviation']},
        ])
        result['passed'] = bool(result['passed'] and split_pass and q_pass)
        result['evaluation_status'] = 'passed' if result['passed'] else 'failed_quality_gate'
        simulations.append(result)
    selected = []
    for target in protocol['target_mass_ratios']:
        passed = [s for s in simulations if s['target_q'] == target and s['passed']]
        if passed:
            selected.append(min(passed, key=lambda s: (s['candidate_rank'], s['sxs_id'])))
    chosen = [by_id[s['primary_case_id']] for s in selected]
    info = None
    if chosen:
        payload = {'cases': chosen, 'screening': {
            'policy_sha256': protocol['quality_policy_sha256'],
            'protocol_sha256': expected_protocol_sha256,
            'manifest_sha256': original.sha256(manifest_path),
            'selected_primary_case_ids': [r['id'] for r in chosen]}}
        write_new(qualified_output, payload)
        info = {'path': str(qualified_output), 'sha256': original.sha256(qualified_output), 'case_count': len(chosen)}
    missing = sorted(targets - {s['target_q'] for s in selected})
    report = {'schema_version': 1, 'purpose': 'external holdouts; no training or refit',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'overall_pass': not missing, 'overall_pass_definition': 'all requested targets have a qualified case',
              'policy_sha256': protocol['quality_policy_sha256'], 'protocol_sha256': expected_protocol_sha256,
              'input_manifest_sha256': original.sha256(manifest_path),
              'selected_primary_case_ids': [r['id'] for r in chosen], 'simulations': simulations,
              'qualified_manifest': info, 'missing_or_failed_targets': missing,
              'counts': {'selected': len(chosen), 'training': 0, 'holdout': len(chosen), 'required_targets': len(targets)},
              'source_sha256': {'screen_adapter': original.sha256(Path(__file__)),
                                'original_screen': original.sha256(Path(original.__file__))}}
    if original.sha256(calibration_path) != protocol['calibration_sha256']:
        raise ValueError('calibration changed during screening')
    write_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'output', 'qualified-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    args = parser.parse_args()
    report = screen(args.manifest, args.protocol, args.expected_protocol_sha256, args.output, args.qualified_output)
    print(json.dumps({'counts': report['counts'], 'missing_or_failed_targets': report['missing_or_failed_targets']}))


if __name__ == '__main__':
    main()
