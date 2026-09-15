#!/usr/bin/env python3
"""Finite 48-worker evolution with verified checkpoint continuation."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

ROOT = Path('/home/xangma/simulations/torchwave-nr/2026-09-12-pilot')
PROV = ROOT / 'provenance'
STATE = PROV / 'performance-48pe-state.json'
UNIT = 'torchwave-nr-performance-48pe.service'
IMAGE = 'sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77'
INPUT = '/campaign/performance/Lev1/Segment_0000/Inspiral.48pe.yaml'
active = None
stopping = False
state = {'status': 'starting', 'completed_segments': []}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(**updates):
    state.update(updates, updated_utc=now())
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(state), flush=True)


def stop(signum, frame):
    global stopping
    stopping = True


def resources(memory):
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    assert available >= (memory + 16) * 2**30, f'Insufficient available RAM: {available}'
    assert shutil.disk_usage(ROOT).free >= 80 * 2**30, 'Insufficient free disk'
    total = sum(p.stat().st_size for p in (ROOT / 'performance').rglob('*')
                if p.is_file() and not p.is_symlink())
    assert total < 192 * 2**30, 'Performance output cap reached'


def stage(name, command, cwd, cpus=1, memory=4, timeout=1800, exits=(0,)):
    global active
    assert not stopping, 'Stop requested'
    resources(memory)
    metrics_path = PROV / (name + '-metrics.json')
    job_path = PROV / (name + '-job.json')
    assert not metrics_path.exists() and not job_path.exists(), name
    active = 'torchwave-nr-' + name + '.service'
    save(status='running', active_stage=name, active_unit=active,
         active_command=command, active_cwd=cwd)
    command_line = ['/usr/bin/python3', str(ROOT / 'scripts/launch-performance-stage.py'), name,
                    '--image', IMAGE, '--workdir', cwd, '--output-dir', '/campaign/performance',
                    '--cpus', str(cpus), '--memory-gib', str(memory),
                    '--timeout-seconds', str(timeout), '--parent-unit', UNIT,
                    '--command-json', json.dumps(command)]
    for code in exits:
        command_line += ['--expect-exit', str(code)]
    subprocess.run(command_line, check=True, timeout=60)
    deadline = time.monotonic() + timeout + 90
    while True:
        assert not stopping, 'Stop requested'
        assert time.monotonic() < deadline, 'Missing terminal stage metrics'
        status = subprocess.check_output(['systemctl', '--user', 'show', active,
                                          '-p', 'ActiveState', '--value'], text=True).strip()
        if status in ('inactive', 'failed'):
            assert metrics_path.exists(), f'{name} ended without metrics'
            assert status == 'inactive', f'{name} systemd service failed'
            break
        time.sleep(10)
    metrics = read(metrics_path)
    job = read(job_path)
    assert job['command'] == metrics['command'] == command
    assert job['image_id'] == IMAGE and job['cpus'] == cpus and job['memory_gib'] == memory
    assert metrics['cwd'] == cwd and job['container_cwd'] == cwd
    assert metrics['started_utc'] >= job['requested_utc']
    assert metrics['accepted_exit'] and metrics['watchdog_reason'] is None
    assert job['expected_exit_codes'] == metrics['expected_exit_codes'] == list(exits)
    assert metrics['exit_code'] in exits
    counters = dict(line.split() for line in metrics['cgroup_memory_events'].splitlines())
    assert all(int(counters.get(key, -1)) == 0 for key in ('oom', 'oom_kill', 'oom_group_kill'))
    active = None
    return metrics


def main():
    assert socket.gethostname() == 'len'
    assert not STATE.exists()
    plan = read(PROV / 'performance-48pe-plan.json')
    job = read(PROV / 'performance-48pe-job.json')
    assert digest(PROV / 'performance-48pe-plan.json') == job['plan_sha256']
    assert plan['image_id'] == IMAGE and plan['num_procs'] == 48
    for path, expected in plan['locked_files'].items():
        assert digest(ROOT / path) == expected, path
    assert read(PROV / 'performance-48pe-input.json')['check_options_passed']
    assert read(PROV / 'independent-p11-comparison.json')['comparison_passed']
    assert read(PROV / 'controlled-p9-lev0-smoke-quality.json')['validation_passed']
    save(started_utc=now(), plan_sha256=job['plan_sha256'])
    checkpoint = None
    for index in range(8):
        name = f'performance-48pe-segment-{index:04d}'
        cwd = f'/campaign/performance/Lev1/Segment_{index:04d}'
        if index:
            (ROOT / cwd.removeprefix('/campaign/')).mkdir(exist_ok=False)
        command = ['EvolveGhBinaryBlackHole', '+p48', '--input-file', INPUT]
        if checkpoint:
            command += ['+restart', checkpoint]
        metrics = stage(name, command, cwd, cpus=48, memory=160, timeout=45000, exits=(0, 2))
        stage(name + '-validation', ['python', '/campaign/scripts/performance-48pe-artifacts.py',
                                    'inspect', '--index', str(index)], cwd)
        quality_path = PROV / (name + '-quality.json')
        quality = read(quality_path)
        assert quality['validation_passed'] and quality['exit_code'] == metrics['exit_code']
        state['completed_segments'].append({'stage': name, 'quality_file': str(quality_path),
                                             'last_time': quality['last_time'],
                                             'wall_seconds': metrics['wall_seconds']})
        if metrics['exit_code'] == 0:
            assert quality['last_time'] >= 100.0 - 1e-9
            save(status='completed', active_stage=None, active_unit=None, ended_utc=now(),
                 final_time=quality['last_time'])
            return
        checkpoint = quality['checkpoint']
        assert checkpoint and checkpoint.startswith(cwd + '/Checkpoints/') and '..' not in Path(checkpoint).parts
        save(status='checkpoint_verified', active_stage=None, active_unit=None,
             next_checkpoint=checkpoint)
    raise RuntimeError('Eight-segment budget exhausted; all checkpoints retained')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        save(status='stopped' if stopping else 'failed', error=str(error), ended_utc=now())
        raise
    finally:
        if active:
            subprocess.run(['systemctl', '--user', 'stop', active], timeout=60, check=False)
