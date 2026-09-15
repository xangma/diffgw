#!/usr/bin/env python3
"""One bounded writable build from the preserved original runtime image."""

import datetime
import json
from pathlib import Path
import shlex
import socket
import subprocess


assert socket.gethostname() == 'len'
root = Path('/home/xangma/simulations/torchwave-nr/2026-09-12-pilot')
image = 'sha256:4d03ce10bf6deb922444b293b39e0c3753d92ac69c70f99ecb9cef49ba64b42f'
name = 'torchwave-nr-build-alignment-2'
log = root / 'logs/build-alignment-2.log'
script = root / 'build-alignment-2.sh'
job = root / 'provenance/build-alignment-2-job.json'
assert not any(p.exists() for p in (log, script, job))
output = root / 'diagnostics/build-alignment-2'
output.mkdir()
output.chmod(0o777)
(root / 'diagnostics/charm7-alignment').chmod(0o755)
command = ['python', '-u', '/campaign/scripts/fix-charm-alignment.py']
docker = ['/usr/bin/docker', 'run', '--name', name, '--cpus', '8',
          '--memory', '32g', '--memory-swap', '32g', '--pids-limit', '512',
          '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
          '--mount', f'type=bind,src={root},dst=/campaign,readonly',
          '--mount', f'type=bind,src={output},dst=/campaign/diagnostics/build-alignment-2',
          '--workdir', '/work/spectre/build', '--entrypoint', 'python', image,
          '-u', '/campaign/scripts/measure-stage.py', '--metrics',
          '/campaign/diagnostics/build-alignment-2/metrics.json', '--output-dir',
          '/campaign/diagnostics/build-alignment-2', '--timeout-seconds', '3300',
          '--', *command]
script.write_text('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(docker)
                  + ' > ' + shlex.quote(str(log)) + ' 2>&1\n')
unit = ['systemd-run', '--user', '--unit=' + name,
        '--property=RuntimeMaxSec=3360', '--property=TimeoutStopSec=60s',
        '--property=ExecStop=/usr/bin/docker stop -t 30 ' + name,
        '--working-directory=' + str(root), '/bin/bash', str(script)]
record = dict(host='len', cwd=str(root), container_cwd='/work/spectre/build',
              command=command, docker_command=docker, image_id=image,
              unit=name + '.service', log_path=str(log),
              stop_command='systemctl --user stop ' + name + '.service',
              expected_next_check='within one minute',
              requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
job.write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(unit, check=True)
record['initial_systemd_state'] = subprocess.check_output(
    ['systemctl', '--user', 'show', name + '.service', '-p', 'MainPID',
     '-p', 'ActiveState', '-p', 'SubState'], text=True)
job.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
