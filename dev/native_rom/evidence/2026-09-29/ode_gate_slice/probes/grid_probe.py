import hashlib
import json
import os
import socket
import sys
import time

from dev.native_rom import qualify_generator as q

row = dict(mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0, omega_start=0.04)
settings = q.Settings(atol_levels=(1e-12, 1e-13, 1e-14), rtol_levels=(1e-11, 1e-12, 1e-13))
levels = [dict(name='grid_1', atol=1e-14, rtol=1e-13, dt_geom=0.25),
          dict(name='grid_2_default', atol=1e-14, rtol=1e-13, dt_geom=0.125),
          dict(name='grid_2_fixed', atol=1e-14, rtol=1e-13, dt_geom=0.125)]
source_before = q.source_snapshot()
runs = {}
summary = {}
for level in levels:
    name = level['name']
    start = time.monotonic()
    kwargs = {'diagnostic_spin_truncation': 'tabulated'}
    if name == 'grid_2_fixed':
        kwargs['diagnostic_restart_time_M'] = 165.0
    run = q.native_run(row, level, settings, **kwargs, return_nqc_coefficients=True)
    runs[name] = run
    summary[name] = dict(restart_time_M=run['restart_time_M'],
                         peak_time_M=run['peak_time_M'],
                         match_time_M=run['match_time_M'],
                         mode_sha256=hashlib.sha256(run['mode'].tobytes()).hexdigest(),
                         nqc_coefficients=run['nqc_coefficients'],
                         stages=run['stages'], seconds=time.monotonic()-start)
    print(json.dumps(dict(run=name, seconds=summary[name]['seconds'],
                          restart_time_M=run['restart_time_M'],
                          mode_sha256=summary[name]['mode_sha256'])), flush=True)
comparisons = {name: q.compare_runs(runs[name], runs['grid_1'])
               for name in ('grid_2_default', 'grid_2_fixed')}
source_after = q.source_snapshot()
output = dict(schema='diffgw.seobnrv4.grid-restart-control.v1',
              row=row, settings=q.asdict(settings), source_manifest_sha256=q.canonical_hash(source_before),
              sources_unchanged=source_before == source_after,
              cases=summary, comparisons=comparisons,
              process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
with open('grid-restart-probe.json', 'w') as stream:
    json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(status='done', comparisons={k: v['complex_relative_l2'] for k,v in comparisons.items()})), flush=True)
