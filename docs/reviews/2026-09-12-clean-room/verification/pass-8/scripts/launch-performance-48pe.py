#!/usr/bin/env python3
"""Record and launch the bounded 100 M / 48-worker performance run."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/xangma/simulations/torchwave-nr/2026-09-12-pilot')
PROV = ROOT / 'provenance'
NAME = 'torchwave-nr-performance-48pe'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname() == 'len'
    log = ROOT / 'logs/performance-48pe.log'
    job = PROV / 'performance-48pe-job.json'
    plan_path = PROV / 'performance-48pe-plan.json'
    assert not any(p.exists() for p in (log, job, plan_path, PROV / 'performance-48pe-state.json'))
    files = ['scripts/' + name for name in (
        'measure-performance-stage.py', 'launch-performance-stage.py',
        'performance-48pe-artifacts.py', 'run-performance-48pe.py', 'launch-performance-48pe.py')]
    files += ['provenance/' + name for name in (
        'performance-evolution-manifest.json', 'performance-48pe-input.json',
        'independent-p11-comparison.json', 'controlled-p9-lev0-smoke-quality.json')]
    files += ['performance/Lev1/Segment_0000/Inspiral.yaml',
              'performance/Lev1/Segment_0000/Inspiral.48pe.yaml']
    plan = {'image_id': 'sha256:0ba0f3bb70b90e83c9b50316ecd37f1b55af4537160eafbe3392a6f67c4e9d77',
            'num_procs': 48, 'final_time': 100, 'lev': 1, 'memory_gib': 160,
            'checkpoint_wall_hours': 11.5, 'segment_timeout_seconds': 45000,
            'max_segments': 8, 'driver_timeout_seconds': 374400,
            'max_output_gib': 192, 'min_free_gib': 80,
            'locked_files': {name: digest(ROOT / name) for name in files},
            'scope': '100 M performance evolution only; no further campaigns launched'}
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    command = ['/usr/bin/python3', '-u', str(ROOT / 'scripts/run-performance-48pe.py')]
    launch = ['systemd-run', '--user', '--unit=' + NAME,
              '--property=RuntimeMaxSec=374400', '--property=TimeoutStopSec=90',
              '--property=MemoryMax=256M', '--property=CPUQuota=20%',
              '--property=StandardOutput=append:' + str(log), '--property=StandardError=inherit',
              '--working-directory=' + str(ROOT), *command]
    record = {'host': 'len', 'cwd': str(ROOT), 'command': command, 'systemd_command': launch,
              'unit': NAME + '.service', 'requested_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'log_path': str(log), 'state_path': str(PROV / 'performance-48pe-state.json'),
              'stop_command': 'systemctl --user stop ' + NAME + '.service',
              'expected_next_check': 'within one minute; driver polls every 10 seconds',
              'plan_sha256': digest(plan_path)}
    job.write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(launch, check=True)
    record['initial_systemd_state'] = subprocess.check_output(
        ['systemctl', '--user', 'show', NAME + '.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState'], text=True)
    job.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
