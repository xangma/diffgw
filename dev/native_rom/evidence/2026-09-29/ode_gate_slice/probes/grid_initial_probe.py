import hashlib
import json
import os
import socket
import time
from dev.native_rom import qualify_generator as q

row = dict(mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0, omega_start=0.04)
settings = q.Settings(atol_levels=(1e-12, 1e-13, 1e-14), rtol_levels=(1e-11, 1e-12, 1e-13))
levels = [dict(name='ode_2', atol=1e-14, rtol=1e-13, dt_geom=0.5),
          dict(name='grid_1', atol=1e-14, rtol=1e-13, dt_geom=0.25),
          dict(name='grid_2', atol=1e-14, rtol=1e-13, dt_geom=0.125)]
source_before = q.source_snapshot()
runs = []
summary = []
for level in levels:
    start = time.monotonic()
    run = q.native_run(row, level, settings, diagnostic_spin_truncation='tabulated',
                       diagnostic_restart_time_M=165.0,
                       diagnostic_coarse_initial_step_M=0.5,
                       return_nqc_coefficients=True)
    runs.append(run)
    item = dict(name=level['name'], dt_geom=level['dt_geom'], seconds=time.monotonic()-start,
                mode_sha256=hashlib.sha256(run['mode'].tobytes()).hexdigest(),
                restart_time_M=run['restart_time_M'], nqc_coefficients=run['nqc_coefficients'])
    summary.append(item)
    print(json.dumps(item), flush=True)
first = q.compare_runs(runs[0], runs[1])
last = q.compare_runs(runs[1], runs[2])
source_after = q.source_snapshot()
output = dict(schema='diffgw.seobnrv4.grid-initial-step-control.v1', row=row,
              fixed_restart_time_M=165.0, fixed_coarse_initial_step_M=0.5,
              levels=summary, first=first, last=last,
              source_manifest_sha256=q.canonical_hash(source_before),
              sources_unchanged=source_before == source_after,
              process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid()))
with open('grid-initial-step-probe.json', 'w') as stream:
    json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(status='done', first=first['complex_relative_l2'],
                      last=last['complex_relative_l2'])), flush=True)
