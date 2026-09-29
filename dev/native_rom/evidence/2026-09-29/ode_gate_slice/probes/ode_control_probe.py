import argparse
import hashlib
import json
import os
import socket
import time

from dev.native_rom import qualify_generator as q

parser = argparse.ArgumentParser()
parser.add_argument('--row', choices=('nonspin', 'spin'), required=True)
parser.add_argument('--control', choices=('hermite', 'second'), required=True)
args = parser.parse_args()
row = dict(mass1=30.0, mass2=20.0, chi1=0.0 if args.row == 'nonspin' else 0.2,
           chi2=0.0 if args.row == 'nonspin' else -0.1, omega_start=0.04)
settings = q.Settings(atol_levels=(1e-12, 1e-13, 1e-14), rtol_levels=(1e-11, 1e-12, 1e-13))
levels = [dict(name='ode_1', atol=1e-13, rtol=1e-12, dt_geom=0.5),
          dict(name='ode_2', atol=1e-14, rtol=1e-13, dt_geom=0.5)]
kwargs = dict(diagnostic_spin_truncation='tabulated',
              diagnostic_hermite_resampling=args.control == 'hermite',
              diagnostic_matching_second_derivatives=args.control == 'second')
source_before = q.source_snapshot()
runs = []
summary = []
for level in levels:
    start = time.monotonic()
    run = q.native_run(row, level, settings, **kwargs, return_nqc_coefficients=True,
                       return_accepted_hashes=True)
    runs.append(run)
    item = dict(name=level['name'], seconds=time.monotonic()-start,
                mode_sha256=hashlib.sha256(run['mode'].tobytes()).hexdigest(),
                restart_time_M=run['restart_time_M'], peak_time_M=run['peak_time_M'],
                match_time_M=run['match_time_M'], nqc_coefficients=run['nqc_coefficients'],
                accepted_array_sha256=run['accepted_array_sha256'])
    summary.append(item)
    print(json.dumps(dict(run=item['name'], seconds=item['seconds'],
                          mode_sha256=item['mode_sha256'])), flush=True)
comparison = q.compare_runs(runs[0], runs[1])
source_after = q.source_snapshot()
output = dict(schema='diffgw.seobnrv4.ode-control.v1', row=row, control=args.control,
              levels=summary, comparison=comparison,
              source_manifest_sha256=q.canonical_hash(source_before),
              sources_unchanged=source_before == source_after,
              process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid()))
path=f'ode-control-{args.row}-{args.control}.json'
with open(path, 'w') as stream:
    json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(status='done', complex_relative_l2=comparison['complex_relative_l2'],
                      amplitude_relative_l2=comparison['amplitude_relative_l2'])), flush=True)
