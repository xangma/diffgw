"""Guard the scientific input and holdout boundary of the NR pilot."""

import hashlib
import json

import numpy as np
import pytest

from scripts import fit_nr33_amplitude_pilot as pilot


def test_training_input_replacement_is_rejected(tmp_path):
    path = tmp_path / 'case.npz'
    np.savez(path, time=np.arange(32.), h33=np.ones(32, dtype=complex))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    case = pilot.Case('fixture', str(path), 2., {'npz_sha256': expected})
    pilot.load_case(case)
    np.savez(path, time=np.arange(32.), h33=np.zeros(32, dtype=complex))
    with pytest.raises(ValueError, match='SHA256|sha256|hash'):
        pilot.load_case(case)


def test_real_pilot_split_never_loads_holdouts_or_alternate_extraction(tmp_path, monkeypatch):
    ids = [('0008', 1.5), ('0169', 2.), ('0030', 3.), ('0182', 4.),
           ('0056', 5.), ('0166', 6.), ('0063', 8.)]
    entries = []
    for suffix, q in ids:
        entries.append({'id': f'SXS:BBH:{suffix}/Lev3/N3', 'path': 'absent.npz',
                        'sxs_id': f'SXS:BBH:{suffix}', 'q': q,
                        'split': 'holdout' if suffix in {'0030', '0166'} else 'training'})
    entries += [{**row, 'id': row['id'].replace('/N3', '/N2'), 'split': 'validation_only'}
                for row in entries if row['split'] == 'holdout']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': entries}))
    seen = []

    def direct(case, controls):
        assert case.metadata['split'] == 'training'
        seen.append(case.case_id)
        eta = case.q / (1 + case.q)**2
        delta = (case.q - 1) / (case.q + 1)
        return {'eta': eta, 'delta': delta, 'log_abs_a_lambda': np.log(delta),
                'lambda': .5}

    monkeypatch.setattr(pilot, '_case_direct', direct)
    result = pilot.fit_manifest(manifest)
    assert len(seen) == 5
    assert seen == result['split']['train_case_ids']
    assert len(result['split']['holdout_case_ids']) == 2
    assert len(result['split']['excluded_case_ids']) == 2
