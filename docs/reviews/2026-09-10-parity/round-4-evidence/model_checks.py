"""Focused round-four model measurements; public LAL binary oracle only.

Reuses the original review's measurement definitions, not its case execution.
These overlap measurements allow independent time/phase optimization of each
polarization and are diagnostic only, not parity certification. Amplitude
relative L2 is invariant under those alignments.
"""
import ast
import json
from pathlib import Path
import platform
import sys

ROOT = Path('/Users/xangma/repos/torchwave')
original = ROOT / 'docs/reviews/2026-09-10-parity/probe.py'
tree = ast.parse(original.read_text())
definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(original), 'exec'))
rows = []
for m1,m2,c1,c2 in [(30.,30.,.8,.8),(36.,29.,.3,-.2),(30.,10.,.5,-.3)]:
    run_case(f'D {m1}/{m2} {c1}/{c2}', 'IMRPhenomD', m1=m1,m2=m2,c1=c1,c2=c2,
             distance=410.,flow=20.,fhigh=1024.,tw_extra={'f_ref':20.})
for m1,m2,c1,c2 in [(35.,30.,.2,-.3),(50.,15.,.6,.4),(60.,7.5,-.4,.1),(75.,5.,.5,-.2),(30.,30.,.8,.8)]:
    run_case(f'XAS {m1}/{m2} {c1}/{c2}', 'IMRPhenomXAS', m1=m1,m2=m2,c1=c1,c2=c2,
             distance=500.,flow=20.,fhigh=512.,df=.125,tw_extra={'f_ref':20.})
for name in ['IMRPhenomHM','IMRPhenomXHM']:
    run_case(f'{name} inclined', name, m1=40.,m2=20.,c1=.3,c2=-.1,inc=1.,
             distance=500.,flow=20.,fhigh=512.,df=.125,tw_extra={'f_ref':20.})
out = {'metadata':{'host':platform.node(),'python':sys.executable,'torch':torch.__version__,
       'numpy':np.__version__,'lal':lal.__version__,'lalsimulation':lalsim.__version__,
       'pycbc':pycbc.__version__,'precision':'float64/complex128 CPU','psd':'flat',
       'band':'flow <= f < fhigh','shared_fref':'both=flow',
       'raw_l2_caveat':'Raw errors include phase/time convention differences; these are diagnostic overlaps, not certification.',
       'optimized_match_caveat':'Time/phase maximized independently per polarization, using PyCBC optimized_match.'},
       'results': rows}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
