#!/usr/bin/env python3
"""Black-box public QNM versus independent Kerr solver/interpolants."""
import hashlib,json,sys
from pathlib import Path
import lal,lalsimulation as ls,numpy as np,torch
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
import torchwave.approximants.kerr_qnm_generated as generated
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm as london
MSUN_SEC=4.925490947641267e-6
sys.path.insert(0,str(Path(__file__).parent)); from kerr_leaver_solver import solve

def public(a):
 v=lal.CreateCOMPLEX16Vector(1); ls.SimIMREOBGenerateQNMFreqV2fromFinal(v,1.,float(a),2,2,1)
 raw=complex(v.data[0])
 # Public API reports positive damping; normalize to exp(-i omega t).
 return complex(raw.real,-raw.imag)*MSUN_SEC
def row(a):
 root=solve(float(a),2,2,0,strict=True); gg=dimensionless_qnm(2,2,0,torch.tensor(a,dtype=torch.float64)); g=complex(float(gg[0]),-float(gg[1])); l=london(torch.tensor(a,dtype=torch.float64),2,2); l=complex(l); l=complex(l.real,-l.imag); p=public(a)
 return {'spin':a,'solver_omega':[root.omega.real,root.omega.imag],'generated_omega':[g.real,g.imag],'london_omega':[l.real,l.imag],'public_omega':[p.real,p.imag],'abs_generated_public':abs(g-p),'abs_solver_public':abs(root.omega-p),'abs_generated_solver':abs(g-root.omega),'abs_london_public':abs(l-p)}
def main():
 # Probe exact clean-table knots and an interval midpoint, in addition to
 # conventional and canonical noninteger spins.  The clean table is indexed
 # in x=-log(1-a), so convert knots back to physical spin without fitting.
 knots=np.asarray(generated._SPLINES[(2,2,0)][0],dtype=float)
 # The public black-box API documents a practical upper bound near .9996;
 # exclude clean-table extrapolation knots above that bound.
 knot_indices=(0,len(knots)//4,len(knots)//2,250)
 knot_spins=[float(1.0-np.exp(-knots[i])) for i in knot_indices]
 mid_x=0.5*(knots[len(knots)//2]+knots[len(knots)//2+1])
 spins=[0.,.1,.2,.5,.7,.9,.95,.99,.137,.6921198429,.8731664442]+knot_spins+[float(1-np.exp(-mid_x))]
 spins=list(dict.fromkeys(spins))
 out={'method':'Independent Leaver eigenproblem and generated spline/table interpolants compared with public Berti-style QNM API; no waveform fitting','convention':'All complex frequencies use negative imaginary damping for exp(-i omega t). The public API reports positive damping, so its imaginary part is negated before multiplying by M_sun.','public_api':'SimIMREOBGenerateQNMFreqV2fromFinal(mass=1 solar mass, spin, l=2,m=2,n=0); output is angular frequency per supplied mass','table_coordinate':'x=-log(1-a); exact clean-table knot probes are listed separately from conventional spins','knot_spins':knot_spins,'midpoint_spin':float(1-np.exp(-mid_x)),'references':['https://arxiv.org/abs/0905.2975','https://arxiv.org/abs/1512.02248'],'spins': [row(a) for a in spins],'source_hashes':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/kerr_leaver_solver.py'),Path('src/torchwave/approximants/kerr_qnm_generated.py'),Path('src/torchwave/approximants/kerr_qnm_london_published.py')]},'versions':{'python':sys.version,'lal':lal.__version__,'lalsimulation':ls.__version__,'torch':torch.__version__}}
 Path('docs/diagnose_kerr_qnm_interpolation.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
