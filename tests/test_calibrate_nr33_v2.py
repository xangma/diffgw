"""Protect the no-fit boundary and the predeclared held-out simulations."""
import json
from pathlib import Path

import pytest

from scripts import calibrate_nr33_v2 as runner


POLICY = Path(__file__).resolve().parents[1] / 'docs/reviews/2026-09-15-independent-calibration-v2/quality-policy.json'

pytestmark = pytest.mark.skipif(
    not POLICY.exists(),
    reason="Review audit artifact not present (archived)",
)


def inputs(tmp_path):
    rows = []
    for index, q in enumerate([1.5, 2, 3, 4, 5, 6, 8]):
        rows.append({'id': f'SXS:BBH:{8000 + index}/Lev5/N3',
                     'sxs_id': f'SXS:BBH:{8000 + index}', 'q': q,
                     'target_q': q, 'split': 'holdout' if q in (3, 6) else 'training',
                     'extrapolation_order': 3, 'path': 'unopened.npz'})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': rows}))
    quality = tmp_path / 'quality.json'
    report = {'overall_pass': True, 'policy_sha256': runner.POLICY_SHA,
              'qualified_manifest': {'sha256': runner.sha(manifest)},
              'selected_primary_case_ids': [r['id'] for r in rows],
              'simulations': [{'primary_case_id': r['id'], 'passed': True} for r in rows]}
    quality.write_text(json.dumps(report))
    return quality, manifest, report


def test_missing_quality_never_calls_fitter(tmp_path, monkeypatch):
    quality, manifest, report = inputs(tmp_path)
    report['overall_pass'] = False
    quality.write_text(json.dumps(report))
    monkeypatch.setattr(runner.pilot, 'fit_manifest', lambda *a: pytest.fail('fit called'))
    with pytest.raises(ValueError, match='no fit allowed'):
        runner.run(POLICY, quality, manifest, tmp_path / 'fit.json')
    assert not (tmp_path / 'fit-split.json').exists()


def test_manifest_cannot_change_after_screen(tmp_path):
    quality, manifest, _ = inputs(tmp_path)
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        runner.check_inputs(POLICY, quality, manifest)


def test_aggregate_pass_cannot_hide_failed_simulation(tmp_path):
    quality, manifest, report = inputs(tmp_path)
    report['simulations'][2]['passed'] = False
    quality.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='has not passed'):
        runner.check_inputs(POLICY, quality, manifest)


def test_holdouts_survive_legacy_fitter_split(tmp_path):
    quality, manifest, _ = inputs(tmp_path)
    _, _, train, holdout = runner.check_inputs(POLICY, quality, manifest)
    assert {c.q for c in holdout} == {3, 6}
    assert {c.q for c in train} == {1.5, 2, 4, 5, 8}


def test_freeze_refuses_existing_output(tmp_path, monkeypatch):
    quality, manifest, _ = inputs(tmp_path)
    output = tmp_path / 'fit.json'
    output.write_text('existing frozen result')
    monkeypatch.setattr(runner.pilot, 'fit_manifest', lambda *a: pytest.fail('fit called'))
    with pytest.raises(FileExistsError):
        runner.run(POLICY, quality, manifest, output)
    assert output.read_text() == 'existing frozen result'
