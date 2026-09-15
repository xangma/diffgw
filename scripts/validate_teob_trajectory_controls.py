#!/usr/bin/env python3
"""Check saved trajectory controls with NumPy/SciPy, without native assembly."""
import hashlib
import json
import math
from pathlib import Path

import lal
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/2026-09-15-teob-trajectory-controls'
LEVELS = ('tight_0p1', 'tight_0p01')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(candidate, reference):
    assert candidate.shape == reference.shape
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def main():
    report = json.loads((OUT / 'controls.json').read_text())
    assert report['status'] == 'finished'
    before = report['source_and_input_hashes_before']
    assert before == report['source_and_input_hashes_after']
    assert all(sha(ROOT / p) == h for p, h in before.items())
    assert all(report['native_nqc_clone_bitwise'].values())
    cache = dict(np.load(ROOT / 'docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean/convergence-arrays.npz', allow_pickle=False))
    history = json.loads((ROOT / 'docs/reviews/2026-09-15-parity-strategy-experiment/eob/native-fixed-event-reassembly.json').read_text())
    reference = cache['public_mode']
    projection = -math.sqrt(5 / (4 * math.pi)) * (50 * lal.MTSUN_SI * lal.C_SI / (100e6 * lal.PC_SI))
    arrays, checks = {}, {}
    max_recomposition, max_score_difference = 0., 0.
    for variant, rows in report['rows'].items():
        arrays[variant] = {}
        for level, row in rows.items():
            path = OUT / f'{variant}--{level}.npz'
            assert sha(path) == row['output_sha256']
            a = dict(np.load(path, allow_pickle=False))
            arrays[variant][level] = a
            assert all(np.isfinite(v).all() for v in a.values())
            assert np.array_equal(a['public_time'], cache['public_time'])
            t = a['public_time']
            assert t[0] >= a['extended_time'][0] and t[-1] <= a['extended_time'][-1]
            amplitude = CubicSpline(a['extended_time'], a['extended_amplitude'], bc_type='natural')(t)
            phase = CubicSpline(a['extended_time'], a['extended_phase'], bc_type='natural')(t)
            rebuilt = projection * amplitude * np.exp(-1j * phase)
            difference = norm(rebuilt, a['mode'])
            max_recomposition = max(max_recomposition, difference)
            assert difference < 1e-12
            score = norm(a['mode'], reference)
            error = abs(score - row['events']['metrics_vs_public']['relative_l2'])
            max_score_difference = max(max_score_difference, error)
            assert error < 1e-15
            assert row['nqc_selected_time_M'] == a['aux_time'][row['nqc_selected_index']]
            event = row['events']
            assert event['attachment_time_initial_mass'] == a['joined_time'][event['attachment_sampled_index_joined']]
            for name in ('amplitude_coefficients', 'phase_coefficients'):
                assert np.array_equal(row[name], a[name])
            if variant == 'native':
                assert norm(a['mode'], cache[level + '_processed_mode']) < 1e-13
            else:
                assert np.array_equal(a['aux_time'], history['controlled_pair']['common_auxiliary_grid'])
            if variant != 'native' and variant != 'legacy_common_fixed':
                assert row['nqc_derivative_spacing_M'] == .1
        left, right = (arrays[variant][level] for level in LEVELS)
        measured = norm(right['mode'], left['mode'])
        assert abs(measured - report['pairwise_mode_metrics'][variant]['relative_l2']) < 1e-15
        checks[variant] = {'pair_relative_l2_reference_tight_0p1': measured}
        if variant in ('native', 'legacy_common_fixed'):
            old_variant = 'native' if variant == 'native' else 'common_grid_fixed_events'
            # The historical report used the other member as denominator.
            old_measured = norm(left['mode'], right['mode'])
            np.testing.assert_allclose(old_measured,
                history['controlled_pair']['pairwise_mode_metrics'][old_variant]['relative_l2'], rtol=1e-11, atol=1e-15)
            checks[variant]['historical_norm_reproduced'] = True
        if variant.startswith('prefix_'):
            for key in ('aux_time', 'joined_time', 'extended_time', 'time', 'public_time'):
                assert np.array_equal(left[key], right[key])
            ra, rb = (rows[level] for level in LEVELS)
            assert ra['nqc_selected_time_M'] == rb['nqc_selected_time_M']
            assert ra['events']['attachment_time_initial_mass'] == rb['events']['attachment_time_initial_mass']
            assert ra['events']['peak_time_initial_mass'] == rb['events']['peak_time_initial_mass']
            for level, a in arrays[variant].items():
                base = arrays['explicit_common_fixed'][level]
                assert all(np.array_equal(a[k], base[k]) for k in base
                           if k.startswith('aux_') or k.endswith('_coefficients'))
            grid_level = variant.removeprefix('prefix_')
            assert np.array_equal(arrays[variant][grid_level]['mode'], arrays['explicit_common_fixed'][grid_level]['mode'])
            prefix = cache[grid_level + '_trajectory_time']
            prefix = prefix[prefix < left['aux_time'][0]]
            assert np.array_equal(left['joined_time'], np.concatenate((prefix, left['aux_time'][1:])))
            checks[variant]['all_grids_and_actual_events_equal'] = True
            checks[variant]['own_prefix_output_bitwise'] = True
    for variant, rows in report['changes_against_previous_control'].items():
        base = 'legacy_common_fixed' if variant == 'explicit_common_fixed' else 'explicit_common_fixed'
        for level, detail in rows.items():
            measured = norm(arrays[variant][level]['mode'], arrays[base][level]['mode'])
            assert abs(measured - detail['relative_l2']) < 1e-15
    result = dict(status='passed', outputs_verified=sum(len(v) for v in arrays.values()),
        source_and_input_files_unchanged=len(before),
        runtime_files_unchanged=sum(p.startswith('src/') for p in before),
        previous_manifest_entries_unchanged=report['previous_manifest_entries_unchanged'],
        max_scipy_recomposition_relative_l2=max_recomposition,
        max_independent_public_score_difference=max_score_difference,
        checks=checks, controls_sha256=sha(OUT / 'controls.json'),
        validation_script_sha256=sha(Path(__file__)))
    (OUT / 'artifact-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
