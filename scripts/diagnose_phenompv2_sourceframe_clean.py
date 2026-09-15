"""Compare clean Pv2 source-frame geometry with LAL's public black box."""
import json, pathlib, math
import torch
import lal, lalsimulation as ls
from torchwave.approximants.phenompv2_waveform_clean import _geometry, effective_chip

CASES=[(35,25,.2,.1,.2,-.1,.15,-.1,.7,.3),(20,10,.3,0,.2,0,0,0,.5,0),(15,12,.2,.2,-.1,.1,0,.2,.3,0),
       (20,10,.35,-.15,.1,.1,.2,-.25,1.1,-.2),(12,11,-.2,.25,-.1,.3,.05,.2,.4,.8)]
def main():
 rows=[]
 for c in CASES:
  m1,m2,x,y,z,X,Y,Z,inc,phi=c
  got=_geometry(*[torch.tensor(v,dtype=torch.float64) for v in (*c[:8],inc,phi,20.)])
  chip=float(effective_chip(*[torch.tensor(float(v)) for v in (m1,m2,x,y,X,Y)]))
  ref=ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(m1*lal.MSUN_SI,m2*lal.MSUN_SI,20.,phi,inc,x,y,z,X,Y,Z,ls.IMRPhenomPv2_V)
  # API order: chi1_l, chi2_l, chi_p, theta_JN, alpha0, phi_aligned, zeta.
  clean={'theta_JN':float(got[1]),'alpha0':float(got[0]),'phi_aligned':float(got[3]),'zeta_mod':float((got[2]+math.pi)%(2*math.pi)-math.pi),'chi_p':chip}
  public={'theta_JN':ref[3],'alpha0':ref[4],'phi_aligned':ref[5],'zeta':ref[6],'chi_p':ref[2]}
  rows.append({'input':c,'clean':clean,'public':public,'abs_errors':{k:abs(clean[k]-public[k]) for k in ('theta_JN','alpha0','phi_aligned','chi_p')},'zeta_periodic_error':abs(((clean['zeta_mod']-public['zeta']+math.pi)%(2*math.pi))-math.pi)})
 out=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean');out.mkdir(parents=True,exist_ok=True)
 (out/'sourceframe-geometry.json').write_text(json.dumps({'provenance':'public SimIMRPhenomPCalculateModelParametersFromSourceFrame black box; clean 2PN J geometry','api_enum':'IMRPhenomPv2_V','cases':rows},indent=2))
 print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
