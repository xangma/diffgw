#!/usr/bin/env python3
"""Prepare a versioned runtime input or validate a closed performance segment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import yaml

ROOT = Path('/campaign')
PROV = ROOT / 'provenance'
RUN = ROOT / 'performance/Lev1'
INPUT = RUN / 'Segment_0000/Inspiral.48pe.yaml'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    manifest = read(PROV / 'performance-evolution-manifest.json')
    assert read(PROV / 'independent-p11-comparison.json')['comparison_passed']
    assert read(PROV / 'controlled-p9-lev0-smoke-quality.json')['validation_passed']
    original = Path(manifest['rendered_input'])
    assert digest(original) == manifest['rendered_input_sha256']
    assert digest(Path(manifest['source']['final_quality'])) == manifest['source']['final_quality_sha256']
    assert manifest['configuration']['num_procs'] == 48
    assert manifest['configuration']['final_time'] == 100.0
    docs = list(yaml.safe_load_all(original.read_text()))
    original_docs = copy.deepcopy(docs)
    changed = 0
    for item in docs[1]['PhaseChangeAndTriggers']:
        for phase in item['PhaseChanges']:
            if isinstance(phase, dict) and 'CheckpointAndExitAfterWallclock' in phase:
                assert phase['CheckpointAndExitAfterWallclock']['WallclockHours'] == 23.5
                assert item['Trigger'] == {'Slabs': {'EvenlySpaced': {'Interval': 10, 'Offset': 0}}}
                phase['CheckpointAndExitAfterWallclock']['WallclockHours'] = 11.5
                item['Trigger']['Slabs']['EvenlySpaced']['Interval'] = 1
                changed += 1
    assert changed == 1
    # Preserve every physics and output option, and retain the original input.
    reverted = copy.deepcopy(docs)
    reverted[1]['PhaseChangeAndTriggers'] = original_docs[1]['PhaseChangeAndTriggers']
    assert reverted == original_docs
    assert not INPUT.exists()
    INPUT.write_text(yaml.safe_dump_all(docs, sort_keys=False, explicit_start=True))
    command = ['EvolveGhBinaryBlackHole', '+p1', '--input-file', str(INPUT), '--check-options']
    subprocess.run(command, check=True)
    write(PROV / 'performance-48pe-input.json', {
        'original_input': str(original), 'original_sha256': digest(original),
        'runtime_input': str(INPUT), 'runtime_sha256': digest(INPUT),
        'checkpoint_wall_hours': 11.5, 'checkpoint_trigger_slabs': 1,
        'check_options_command': command, 'check_options_passed': True,
        'physics_and_output_options_preserved': True,
    })


def inspect(index):
    stage = f'performance-48pe-segment-{index:04d}'
    metrics = read(PROV / (stage + '-metrics.json'))
    assert metrics['accepted_exit'] and metrics['watchdog_reason'] is None
    assert metrics['exit_code'] in (0, 2)
    assert digest(INPUT) == read(PROV / 'performance-48pe-input.json')['runtime_sha256']
    segment = RUN / f'Segment_{index:04d}'
    tables = {}
    with h5py.File(segment / 'BbhReductions.h5', 'r') as f:
        def visit(name, dataset):
            if not name.endswith('.dat') or not isinstance(dataset, h5py.Dataset) or not np.issubdtype(dataset.dtype, np.number):
                return
            assert dataset.ndim == 2 and dataset.shape[0] > 0, name
            for begin in range(0, dataset.shape[0], 4096):
                assert np.isfinite(dataset[begin:begin+4096]).all(), name
        f.visititems(visit)
        for name in ('Norms.dat', 'TimeSteps.dat', 'ObservationAhA.dat', 'ObservationAhB.dat'):
            ds = f[name]
            legend = [x.decode() if isinstance(x, bytes) else str(x) for x in ds.attrs['Legend']]
            times = np.asarray(ds[:, legend.index('Time')])
            assert len(times) and np.all(np.diff(times) >= 0), name
            tables[name] = {'rows': len(times), 'first_time': float(times[0]),
                            'last_time': float(times[-1]),
                            'last': dict(zip(legend, ds[-1].tolist()))}
    last = tables['Norms.dat']['last_time']
    assert last > 1e-10, 'Segment made no simulation-time progress'
    cce = {}
    for radius in (200, 300, 400):
        filename = f'BondiSachsCceR{radius:04d}.h5'
        cce[filename] = {}
        with h5py.File(segment / filename, 'r') as f:
            f.visititems(visit)
            for name in ('Beta', 'DrJ', 'DuR', 'H', 'J', 'Q', 'R', 'U', 'W'):
                ds = f[name + '.dat']
                times = np.asarray(ds[:, 0])
                assert len(times) and np.all(np.diff(times) >= 0), (filename, name)
                assert times[-1] >= last - 0.100000001, (filename, name, times[-1], last)
                cce[filename][name] = {'rows': len(times), 'first_time': float(times[0]),
                                       'last_time': float(times[-1])}
    previous = read(PROV / f'performance-48pe-segment-{index-1:04d}-quality.json') if index else None
    if previous:
        assert last > previous['last_time'] + 1e-10, 'Checkpoint made no progress'
        for name, table in tables.items():
            assert table['first_time'] >= previous['tables'][name]['last_time'] - 1e-10
    else:
        assert abs(tables['Norms.dat']['first_time']) < 1e-10
    checkpoint = None
    checkpoint_files = {}
    if metrics['exit_code'] == 2:
        checkpoints = sorted((segment / 'Checkpoints').glob('Checkpoint_[0-9][0-9][0-9][0-9]'))
        assert len(checkpoints) == 1, checkpoints
        checkpoint = checkpoints[0]
        checkpoint_files = {str(p.relative_to(checkpoint)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                            for p in sorted(checkpoint.rglob('*')) if p.is_file()}
        assert {'MainChares.dat', 'RO.dat'} <= checkpoint_files.keys()
        assert all(x['bytes'] > 0 for x in checkpoint_files.values())
    else:
        assert last >= 100.0 - 1e-9, last
        assert tables['TimeSteps.dat']['last_time'] >= 100.0 - 1e-9
        assert all(tables[name]['last_time'] >= 99.5 - 1e-9
                   for name in ('ObservationAhA.dat', 'ObservationAhB.dat'))
    wall = metrics['wall_seconds']
    elapsed_sim = last - (previous['last_time'] if previous else 0.0)
    write(PROV / (stage + '-quality.json'), {
        'validation_passed': True, 'segment': str(segment), 'last_time': last,
        'tables': tables, 'cce': cce, 'exit_code': metrics['exit_code'],
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_files': checkpoint_files, 'wall_seconds': wall,
        'observed_M_per_wall_hour': elapsed_sim * 3600 / wall,
        'scope': '100 M Lev1 performance run; no waveform accuracy or eccentricity certification',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'inspect'))
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else inspect(args.index)
