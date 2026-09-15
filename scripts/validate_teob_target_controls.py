#!/usr/bin/env python3
"""Independent NumPy checks of frozen target/event artifacts; no oracle calls."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/reviews/2026-09-15-teob-target-audit'
BASE = ROOT/'docs/reviews/2026-09-15-teob-grid-nqc'
INPUT = ROOT/'docs/reviews/2026-09-15-teob-fd-attribution'
CASES = ('catalog', 'bbh_reference', 'equal_bbh', 'unequal_bns')
TARGETS = ('amplitude', 'amplitude_derivative', 'frequency', 'frequency_derivative')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    previous = read(BASE/'artifact-manifest.json')
    assert all(sha(ROOT/item['path']) == item['sha256'] for item in previous['artifacts'])
    baseline_sources = read(BASE/'derivative-controls.json')['source_hashes_after']
    original_manifest = read(INPUT/'artifact-manifest.json')
    reports = {}
    for name in CASES:
        d = read(OUT/name/'controls.json')
        assert d['status'] == 'finished' and not d['source_changes']
        assert len(d['variants']) == 11
        assert d['source_hashes_before'] == d['source_hashes_after']
        assert all(sha(ROOT/p) == h for p, h in d['source_hashes_before'].items())
        assert all(d['source_hashes_before'][p] == h for p, h in baseline_sources.items() if p.startswith('src/'))
        for path, field in ((INPUT/name/'continuous.npz', 'input_sha256'),
                            (BASE/name/'nominal.npz', 'nominal_sha256'),
                            (ROOT/'scripts/teob_target_event_controls.py', 'script_sha256'),
                            (OUT/'protocol.md', 'protocol_sha256'),
                            (ROOT/'scripts/teob_grid_derivative_controls.py', 'helper_sha256')):
            assert sha(path) == d[field]
        params, grid = d['case']['params'], d['case']['grid']
        assert all(params[k] == 0 for k in ('chi1', 'chi2', 'lambda1', 'lambda2'))
        m1, m2 = (Decimal(str(params[k])) for k in ('mass1', 'mass2'))
        nu = m1*m2/(m1+m2)**2
        # Independent decimal transcription of the published zero-spin formulas.
        if m1 == m2:
            exact = list(map(Decimal, ('.31973334', '-.00195014', '.38321834', '.01190548')))
            radius = 5e-9
        else:
            exact = [Decimal(a)*nu+Decimal(b) for a, b in
                     (('.15597465', '.28034978'), ('-.00362779', '-.00105397'),
                      ('.46908067', '.27022141'), ('.02504442', '.00548217'))]
            radius = float((1+nu)*Decimal('5e-9'))
        np.testing.assert_allclose(d['targets'], list(map(float, exact)), rtol=0, atol=6e-17)
        np.testing.assert_allclose(d['target_box']['target_radii'], radius, rtol=3e-16, atol=0)
        baseline = np.load(BASE/name/'nominal.npz', allow_pickle=False)
        oracle_path = INPUT/name/'public-fd.npz'
        assert original_manifest[str(oracle_path)] == sha(oracle_path)
        oracle = np.load(oracle_path, allow_pickle=False)
        frequencies = baseline['frequencies']
        mask = (frequencies >= grid['flow']) & (frequencies < grid['fhigh'])
        oracle_idx = np.searchsorted(oracle['frequencies'], frequencies[mask])
        assert np.array_equal(oracle['frequencies'][oracle_idx], frequencies[mask])
        arrays, maximum_norm_error, scores = {}, 0., {}
        for variant, detail in d['variants'].items():
            path = OUT/name/f'{variant}.npz'
            assert sha(path) == detail['output_sha256']
            a = dict(np.load(path, allow_pickle=False))
            arrays[variant] = a
            assert all(np.isfinite(x).all() for x in a.values())
            assert np.array_equal(a['frequencies'], frequencies)
            assert np.array_equal(a['td_time'], baseline['td_time'])
            if variant == 'nominal':
                assert all(np.array_equal(a[k], baseline[k]) for k in baseline.files)
                assert detail['nominal_bitwise']
            assert detail['attachment_identical'] and detail['td_grid_identical']
            assert detail['attachment_time_M'] == d['variants']['nominal']['attachment_time_M']
            offset = {'knot_previous': -1, 'knot_next': 1}.get(variant, 0)
            assert detail['selected_index'] == d['nominal_index']+offset
            dt = detail['selected_time_M']-d['variants']['nominal']['selected_time_M']
            assert abs(dt-offset*.1) < 1e-9
            acc = detail['acceptance']
            assert acc['strict'] and acc['status'] == 'FAILED'
            assert acc['gate'] == {'complex_relative_l2': 1e-6, 'mismatch': 1e-8}
            assert len(acc['projections']) == 6
            scores[variant] = {}
            for pol, label in (('h_plus', 'plus'), ('h_cross', 'cross')):
                value, ref = a[pol+'_fd'][mask], oracle[pol+'_fd'][oracle_idx]
                measured = float(np.linalg.norm(value-ref)/np.linalg.norm(ref))
                maximum_norm_error = max(maximum_norm_error, abs(measured-acc[label]['complex_relative_l2']))
                np.testing.assert_allclose(measured, acc[label]['complex_relative_l2'], rtol=1e-8, atol=2e-14)
                own = baseline[pol+'_fd'][mask]
                change = np.linalg.norm(value-own)/np.linalg.norm(own)
                np.testing.assert_allclose(change, detail['fd_change_vs_nominal_in_band'][pol], rtol=1e-12, atol=1e-17)
                assert all(acc[label][k] for k in ('finite', 'dc_zero', 'out_of_band_zero'))
                scores[variant][label] = measured
        envelopes, symmetries = {}, {}
        for pol in ('h_plus', 'h_cross'):
            key = pol+'_fd'
            norm = np.linalg.norm(baseline[key][mask])
            envelopes[pol] = sum(np.linalg.norm((arrays[t+'_plus'][key][mask]-arrays[t+'_minus'][key][mask])/2)
                                 for t in TARGETS)/norm
            symmetries[pol] = max(np.linalg.norm((arrays[t+'_plus'][key][mask]+arrays[t+'_minus'][key][mask])/2-baseline[key][mask])
                                  for t in TARGETS)/norm
            np.testing.assert_allclose(envelopes[pol], d['linearized_rounding_fd_triangle_envelope'][pol], rtol=1e-12, atol=1e-17)
            np.testing.assert_allclose(symmetries[pol], d['rounding_center_symmetry_residual'][pol], rtol=1e-12, atol=1e-17)
        reports[name] = {'outputs_verified': 11, 'nominal_bitwise': True,
                         'frozen_public_fd_sha256': sha(oracle_path),
                         'max_independent_oracle_norm_difference': maximum_norm_error,
                         'linearized_rounding_fd_triangle_envelope': envelopes,
                         'rounding_center_symmetry_residual': symmetries,
                         'complex_relative_l2': scores}
    result = {'status': 'passed', 'outputs_verified': 44,
              'previous_manifest_entries_unchanged': len(previous['artifacts']),
              'source_files_unchanged': len(d['source_hashes_before']),
              'runtime_files_unchanged': sum(p.startswith('src/') for p in d['source_hashes_before']),
              'validation_script_sha256': sha(Path(__file__)), 'cases': reports}
    (OUT/'artifact-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
