import json

import pytest

from scripts import screen_nr33_external_validation as ext


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    wave = tmp_path / 'wave.npz'
    wave.write_bytes(b'checksum fixture')
    original = save('original.json', {'cases': [{'sxs_id': 'SXS:BBH:0001'}]})
    calibration = save('fit.json', {})
    policy = save('policy.json', {})
    protocol = save('protocol.json', {
        'quality_policy_path': str(policy), 'quality_policy_sha256': ext.original.sha256(policy),
        'calibration_path': str(calibration), 'calibration_sha256': ext.original.sha256(calibration),
        'original_fit_manifest_path': str(original), 'original_fit_manifest_sha256': ext.original.sha256(original),
        'excluded_sxs_ids': [], 'target_mass_ratios': [2.5, 7], 'max_relative_q_deviation': .005})
    row = {'id': 'SXS:BBH:0002/Lev3/N3', 'sxs_id': 'SXS:BBH:0002', 'target_q': 2.5,
           'candidate_rank': 1, 'split': 'holdout', 'q': 2.5, 'path': str(wave),
           'npz_sha256': ext.original.sha256(wave)}
    monkeypatch.setattr(ext.original, 'load_policy', lambda _: {})
    monkeypatch.setattr(ext.original, 'screen_simulation', lambda rows, _: {
        'sxs_id': rows[0]['sxs_id'], 'primary_case_id': rows[0]['id'], 'gates': [], 'passed': True})
    def run(rows=None):
        manifest = save('manifest.json', {'cases': rows or [row]})
        return ext.screen(manifest, protocol, ext.original.sha256(protocol), tmp_path / 'quality.json', tmp_path / 'qualified.json')
    return row, run, tmp_path


def test_partial_coverage_keeps_qualified_holdout_and_records_missing(setup):
    _, run, path = setup
    report = run()
    assert not report['overall_pass'] and report['missing_or_failed_targets'] == [7]
    assert report['counts']['holdout'] == 1 and (path / 'qualified.json').exists()
    with pytest.raises(FileExistsError):
        run()


@pytest.mark.parametrize('changes,message', [
    ({'sxs_id': 'SXS:BBH:0001'}, 'overlap'),
    ({'split': 'training'}, 'training'),
    ({'npz_sha256': '0' * 64}, 'hash mismatch'),
])
def test_rejects_contamination_and_mutated_inputs(setup, changes, message):
    row, run, _ = setup
    with pytest.raises(ValueError, match=message):
        run([{**row, **changes}])


def test_mass_ratio_outside_declared_window_is_not_qualified(setup):
    row, run, path = setup
    report = run([{**row, 'q': 2.6}])
    assert report['selected_primary_case_ids'] == []
    assert not (path / 'qualified.json').exists()
    assert report['simulations'][0]['gates'][-1]['pass'] is False
