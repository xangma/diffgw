#!/usr/bin/env python3
"""Diagnostic: six-significant-digit Kerr QNM tables and candidate spin grids.

The Berti page documents the tabular columns and Leaver method; it does not
state a grid or rounding rule.  Candidate grids below are therefore explicit
diagnostic hypotheses, never runtime data or fitted constants.
"""
import hashlib,json,sys
from pathlib import Path
import lal,lalsimulation as ls,numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0,str(Path(__file__).parent)); from kerr_leaver_solver import solve
MSUN_SEC=4.925490947641267e-6
def pub(a):
 v=lal.CreateCOMPLEX16Vector(1); ls.SimIMREOBGenerateQNMFreqV2fromFinal(v,1.,float(a),2,2,1)
 z=complex(v.data[0]); return complex(z.real,-z.imag)*MSUN_SEC
def six(x):
 return float(f'{x:.6g}')
def run(step, spins):
 # Independent continued-fraction solver is restricted below the extremal
 # region; public API remains the validation oracle at withheld spins.
 grid=np.arange(0.,.9989000001+step/2,step)
 grid=grid[grid < .999]
 vals=[]
 for a in grid:
  z=solve(float(a),2,2,0,strict=True).omega
  vals.append([six(z.real),six(z.imag)])
 vals=np.asarray(vals)
 spl=[CubicSpline(grid,vals[:,j],bc_type='natural') for j in range(2)]
 rows=[]
 for a in spins:
  z=complex(float(spl[0](a)),float(spl[1](a))); p=pub(a)
  rows.append({'spin':a,'abs_table_public':abs(z-p),'table_omega':[z.real,z.imag],'public_omega':[p.real,p.imag]})
 return {'step':step,'grid_count':len(grid),'rows':rows,'max_abs_table_public':max(r['abs_table_public'] for r in rows)}
def main():
 spins=[.137,.6921198429,.8731664442,.317, .641, .918]
 negative=[]
 for a in (-.9,-.5,-.1):
  z=solve(a,2,2,0,strict=True).omega; p=pub(a)
  negative.append({'spin':a,'solver_omega':[z.real,z.imag],'public_omega':[p.real,p.imag],'abs_solver_public':abs(z-p)})
 out={'method':'Leaver values independently generated on regular positive-spin grids, each real/damping component rounded to six significant digits, then natural cubic spline in a; public API is validation only','references':['https://pages.jh.edu/eberti2/ringdown/','https://arxiv.org/abs/0905.2975'],'public_convention':'public positive damping is negated; all rows use exp(-i omega t)','grid_scope':'Berti author data page lists a/M as a table column but does not specify a regular grid, interpolation, or decimal precision. Candidate regular grids are diagnostic hypotheses. Negative-spin probes are reported separately because no negative-spin table grid was documented.','negative_spin_probes':negative,'candidate_grids':[run(s,spins) for s in (.05,.01,.001)],'source_hashes':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/kerr_leaver_solver.py')]},'versions':{'python':sys.version,'lal':lal.__version__,'lalsimulation':ls.__version__}}
 Path('docs/diagnose_qnm_rounding_grids.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
