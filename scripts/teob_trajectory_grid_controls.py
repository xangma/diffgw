#!/usr/bin/env python3
"""Close the final-interpolation-grid control in the historical TEOB replay."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time
from types import FunctionType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import teob_fixed_event_replay as old

driver, nqc_module, ringdown = old.driver, old.nqc_module, old.ringdown
OUT = ROOT / 'docs/reviews/2026-09-15-teob-trajectory-controls'
OLD = ROOT / 'docs/reviews/2026-09-15-parity-strategy-experiment/eob/native-fixed-event-reassembly.json'
PRIOR = ROOT / 'docs/reviews/2026-09-15-teob-target-audit/artifact-manifest.json'
LEVELS = ('tight_0p1', 'tight_0p01')
VARIANTS = ('native', 'legacy_common_fixed', 'explicit_common_fixed',
            'prefix_tight_0p1', 'prefix_tight_0p01')


def explicit_spacing_nqc(*args, **kwargs):
    """Run the existing grid/event clone with its six derivatives at known dt.

    A private function namespace replaces only the derivative callable. The
    old clone otherwise uses spacing only for grid construction and returned
    metadata. No module globals or runtime files are changed.
    """
    if kwargs.get('matching_method', 'grid') != 'grid':
        raise ValueError('this diagnostic requires grid matching')
    source = old._fixed_nqc_function()
    base = args[0]
    step = torch.minimum(base.new_tensor(.5), torch.as_tensor(
        kwargs['fixed_step'], dtype=base.dtype, device=base.device))
    query = kwargs.get('auxiliary_time_override')
    if query is not None:
        expected = query[0] + torch.arange(query.numel(), dtype=query.dtype, device=query.device) * step
        atol = 4 * torch.finfo(query.dtype).eps * max(1., float(query.abs().max()))
        if not torch.allclose(query, expected, rtol=0, atol=atol):
            raise ValueError('auxiliary grid does not follow the authoritative step')
    calls = 0

    def derivative(values, unused_spacing, **options):
        nonlocal calls
        calls += 1
        return nqc_module.quartic_derivative(values, step, **options)

    namespace = source.__globals__.copy()
    namespace['quartic_derivative'] = derivative
    builder = FunctionType(source.__code__, namespace, source.__name__, source.__defaults__)
    builder.__kwdefaults__ = source.__kwdefaults__
    result = builder(*args, **kwargs)
    if calls != 6:
        raise RuntimeError(f'NQC derivative contract changed: {calls} != 6')
    return replace(result, auxiliary_spacing=step)


def resample_prefix(source_time, values, query):
    """Evaluate the native cubic without changing values at exact source knots."""
    if query.ndim != 1 or query.numel() < 1 or not bool(torch.isfinite(query).all()):
        raise ValueError('prefix must be finite and one dimensional')
    if bool((query[1:] <= query[:-1]).any()):
        raise ValueError('prefix must increase')
    if bool((query < source_time[0]).any() | (query > source_time[-1]).any()):
        raise ValueError('prefix extrapolation is forbidden')
    index = torch.searchsorted(source_time, query)
    exact = source_time[index] == query
    if bool(exact.all()):
        return values[index].clone()
    answer = ringdown._natural_cubic_interpolate(source_time, values, query)
    return torch.where(exact, values[index], answer)


def trajectory_inputs(data, level):
    t = torch.tensor(data[level + '_trajectory_time'], dtype=torch.float64)
    state = torch.tensor(data[level + '_trajectory_state'], dtype=torch.float64)
    h = driver._hamiltonian()
    radiation = driver._radiation_reaction(h, 'current')
    frequency = old._native_frequency(h, radiation, state)
    mode = driver.teob_waveform_state_mode(
        state, h, 2, 2, Omega=frequency, residual_phase='zero-unpublished')
    kwargs = dict(amplitude=mode.abs(), phase=driver.phase_from_mode(mode, state[:, 1]),
                  orbital_frequency=frequency,
                  radial_acceleration=driver.conservative_trajectory_quantities(h, state).radial_acceleration,
                  nu=t.new_tensor(driver.NU_VALUE), fixed_step=t.new_tensor(.1),
                  targets=driver.published_2017_nqc_targets(t.new_tensor(driver.NU_VALUE), t.new_tensor(0.)),
                  target_provenance=driver.PUBLISHED_2017_NQC_PROVENANCE)
    return t, state, kwargs


def assemble(t, nqc, peak=None, prefix=None):
    mass, spin = (x.to(dtype=t.dtype) for x in driver.uib2016_nonspinning_remnant(
        driver.MASS1_MSUN, driver.MASS2_MSUN))
    full_time = t if prefix is None else prefix
    amplitude, phase = nqc.original.corrected_amplitude, nqc.original.corrected_phase
    if prefix is not None:
        amplitude = resample_prefix(t, amplitude, prefix)
        phase = resample_prefix(t, phase, prefix)
    with old._forced_peak(None if peak is None else t.new_tensor(peak)):
        result = ringdown.assemble_teob_ringdown_22(
            full_time, amplitude, phase, nqc.auxiliary.time,
            nqc.auxiliary.corrected_amplitude, nqc.auxiliary.corrected_phase,
            nqc.auxiliary.orbital_frequency, nu=t.new_tensor(driver.NU_VALUE),
            final_mass=mass, final_spin=spin, output_step=t.new_tensor(driver.OUTPUT_STEP))
    return result, mass


def save_output(folder, variant, level, nqc, result, mass, data):
    output, event = old._score(result, data['public_time'], data['public_mode'],
                               old._projection_scale(), mass)
    assert event['full_public_support']
    arrays = {'public_time': data['public_time'], 'mode': output}
    for name in ('time', 'amplitude', 'phase', 'joined_time', 'joined_amplitude',
                 'joined_phase', 'extended_time', 'extended_amplitude', 'extended_phase'):
        arrays[name] = old.arr(getattr(result, name))
    for name in ('time', 'state', 'amplitude', 'phase', 'orbital_frequency',
                 'radial_acceleration', 'corrected_amplitude', 'corrected_phase'):
        arrays['aux_' + name] = old.arr(getattr(nqc.auxiliary, name))
    arrays['amplitude_coefficients'] = old.arr(nqc.coefficients.amplitude)
    arrays['phase_coefficients'] = old.arr(nqc.coefficients.phase)
    assert all(np.isfinite(a).all() for a in arrays.values())
    path = folder / f'{variant}--{level}.npz'
    np.savez_compressed(path, **arrays)
    row = dict(output_sha256=old.sha256(path), events=event,
               nqc_derivative_spacing_M=float(nqc.auxiliary_spacing),
               nqc_peak_time_M=float(nqc.t_peak_grid), nqc_requested_time_M=float(nqc.t_nqc),
               nqc_selected_index=int(nqc.matching_index),
               nqc_selected_time_M=float(nqc.auxiliary.time[nqc.matching_index]),
               amplitude_coefficients=arrays['amplitude_coefficients'].tolist(),
               phase_coefficients=arrays['phase_coefficients'].tolist())
    return arrays, row


def hashes(paths):
    return {str(p.relative_to(ROOT)): old.sha256(p) for p in sorted(set(paths))}


def run(folder):
    started = time.monotonic()
    if (folder / 'controls.json').exists():
        raise FileExistsError('refusing to overwrite a completed experiment')
    folder.mkdir(parents=True, exist_ok=True)
    history = json.loads(OLD.read_text())
    assert old.sha256(old.DEFAULT_CACHE) == history['inputs']['cache']
    assert old.sha256(old.DEFAULT_REPORT) == history['inputs']['prior_report']
    # NQC/ringdown gained the separately audited continuous branch since the
    # historical replay. Pin today's files to the latest frozen target audit;
    # require old grid behavior to reproduce numerically below.
    latest_path = PRIOR.parent / 'catalog/controls.json'
    latest_sources = json.loads(latest_path.read_text())['source_hashes_before']
    compatibility = {}
    for name, entry in history['inputs']['source_files'].items():
        path = Path(entry['path'])
        current = old.sha256(path)
        compatibility[name] = dict(historical=entry['sha256'], current=current,
                                   historical_hash_equal=current == entry['sha256'])
        if name in ('native_nqc_module', 'native_ringdown_module'):
            assert current == latest_sources[str(path.relative_to(ROOT))]
        else:
            assert current == entry['sha256'], entry['path']
    assert all(old.sha256(ROOT / p) == h for p, h in latest_sources.items() if p.startswith('src/'))
    previous = json.loads(PRIOR.read_text())['artifacts']
    prior_paths = [ROOT / e['path'] for e in previous]
    assert all(old.sha256(ROOT / e['path']) == e['sha256'] for e in previous)
    paths = list((ROOT / 'src').rglob('*.py')) + prior_paths + [
        OLD, PRIOR, latest_path, old.DEFAULT_CACHE, old.DEFAULT_REPORT, Path(__file__),
        folder / 'protocol.md', Path(old.__file__), Path(driver.__file__)]
    before = hashes(paths)
    data = dict(np.load(old.DEFAULT_CACHE, allow_pickle=False))
    pair = history['controlled_pair']
    grid = old._common_auxiliary_grid(data, LEVELS)
    assert np.array_equal(old.arr(grid), pair['common_auxiliary_grid'])
    prefixes = {level: torch.tensor(data[level + '_trajectory_time'][
        data[level + '_trajectory_time'] < float(grid[0])], dtype=torch.float64) for level in LEVELS}
    controls = dict(auxiliary_time_override=grid,
                    fixed_peak_time=grid.new_tensor(pair['fixed_nqc_peak_time_M']),
                    fixed_nqc_time=grid.new_tensor(pair['fixed_nqc_time_M']))
    arrays, rows, identity = {}, {}, {}
    with torch.no_grad():
        for level in LEVELS:
            t, state, kwargs = trajectory_inputs(data, level)
            native = nqc_module.assemble_teob_merger_nqc(t, state, **kwargs)
            own = explicit_spacing_nqc(t, state, **kwargs, auxiliary_time_override=native.auxiliary.time)
            identity[level] = all(torch.equal(a, b) for a, b in (
                (native.coefficients.amplitude, own.coefficients.amplitude),
                (native.coefficients.phase, own.coefficients.phase),
                (native.original.corrected_amplitude, own.original.corrected_amplitude),
                (native.original.corrected_phase, own.original.corrected_phase),
                (native.auxiliary.corrected_amplitude, own.auxiliary.corrected_amplitude),
                (native.auxiliary.corrected_phase, own.auxiliary.corrected_phase)))
            assert identity[level], 'explicit derivative clone does not reproduce native NQC'
            legacy = old._fixed_nqc_function()(t, state, **kwargs, **controls)
            explicit = explicit_spacing_nqc(t, state, **kwargs, **controls)
            for variant in VARIANTS:
                nqc = native if variant == 'native' else legacy if variant == 'legacy_common_fixed' else explicit
                prefix = prefixes[variant.removeprefix('prefix_')] if variant.startswith('prefix_') else None
                peak = None if variant == 'native' else pair['fixed_rd_peak_time_M']
                result, mass = assemble(t, nqc, peak, prefix)
                a, row = save_output(folder, variant, level, nqc, result, mass, data)
                arrays.setdefault(variant, {})[level] = a
                rows.setdefault(variant, {})[level] = row
                if variant == 'native':
                    row['cache_reproduction'] = old.metrics(a['mode'], data[level + '_processed_mode'])
                    assert row['cache_reproduction']['relative_l2'] < 1e-13
                if variant in ('native', 'legacy_common_fixed'):
                    prior_variant = 'native' if variant == 'native' else 'common_grid_fixed_events'
                    prior_row = pair['variants'][prior_variant][level]
                    np.testing.assert_allclose(row['events']['metrics_vs_public']['relative_l2'],
                        prior_row['events']['metrics_vs_public']['relative_l2'], rtol=1e-11, atol=1e-15)
                    assert row['events']['attachment_time_initial_mass'] == prior_row['events']['attachment_time_initial_mass']
                print(f'{variant} {level}: saved', flush=True)
    checks, metrics = {}, {}
    for variant, by_level in arrays.items():
        a, b = (by_level[level] for level in LEVELS)
        ra, rb = (rows[variant][level] for level in LEVELS)
        checks[variant] = {key: np.array_equal(a[key], b[key]) for key in
                           ('aux_time', 'joined_time', 'extended_time', 'time', 'public_time')}
        checks[variant].update(nqc_selected_time_equal=ra['nqc_selected_time_M'] == rb['nqc_selected_time_M'],
            rd_peak_equal=ra['events']['peak_time_initial_mass'] == rb['events']['peak_time_initial_mass'],
            attachment_time_equal=ra['events']['attachment_time_initial_mass'] == rb['events']['attachment_time_initial_mass'])
        if variant.startswith('prefix_'):
            assert all(checks[variant].values())
            for level in LEVELS:
                baseline = arrays['explicit_common_fixed'][level]
                assert all(np.array_equal(by_level[level][k], baseline[k]) for k in baseline
                           if k.startswith('aux_') or k.endswith('_coefficients'))
            reference_level = variant.removeprefix('prefix_')
            checks[variant]['reference_output_bitwise'] = np.array_equal(
                by_level[reference_level]['mode'], arrays['explicit_common_fixed'][reference_level]['mode'])
            assert checks[variant]['reference_output_bitwise']
        metrics[variant] = old.metrics(b['mode'], a['mode'])
    changes = {variant: {level: old.metrics(arrays[variant][level]['mode'],
                arrays['legacy_common_fixed' if variant == 'explicit_common_fixed' else 'explicit_common_fixed'][level]['mode'])
                for level in LEVELS} for variant in VARIANTS[2:]}
    after = hashes(paths)
    assert before == after, 'source or historical input changed during replay'
    report = dict(status='finished', experiment='historical TD trajectory and final interpolation grid controls',
        run=dict(host=socket.gethostname(), cwd=str(ROOT), pid=os.getpid(),
                 command=shlex.join([sys.executable] + sys.argv), elapsed_seconds=time.monotonic()-started),
        source_and_input_hashes_before=before, source_and_input_hashes_after=after,
        previous_manifest_entries_unchanged=len(previous), native_nqc_clone_bitwise=identity,
        historical_source_compatibility=compatibility,
        common_grid_source='previous experiment common support intersection, no oracle selection',
        rows=rows, fixed_control_checks=checks, pairwise_mode_metrics=metrics,
        changes_against_previous_control=changes)
    (folder / 'controls.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'finished', 'elapsed_seconds': report['run']['elapsed_seconds'],
                      'pairwise_mode_metrics': metrics}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUT)
    run(parser.parse_args().output_root.resolve())
