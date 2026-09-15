"""Paper-literal Pv2 L variants versus the public source-frame API."""
import json, math, pathlib, torch, lal, lalsimulation as ls
import torchwave.approximants.phenompv2_waveform_clean as pv
CASES=[(35,25,.2,.1,.2,-.1,.15,-.1,.7,.3),(20,10,.3,0,.2,0,0,0,.5,0),(15,12,.2,.2,-.1,.1,0,.2,.3,0),(20,10,.35,-.15,.1,.1,.2,-.25,1.1,-.2),(12,11,-.2,.25,-.1,.3,.05,.2,.4,.8)]
def lfun(sign,lead):
 def fun(m1,m2,f):
  M=(m1+m2)*pv.MSUN_SEC;v=(math.pi*M*f).pow(1/3);e=m1*m2/(m1+m2).square()
  return e/v*(1+(1.5+e/6)*v.square()+(lead-19*e/8+sign*e.square()/24)*v.pow(4))
 return fun
def main():
 rows=[]
 for name,sg,lead in [('current',1,27/8),('literal',-1,3.373),('27/8_minus',-1,27/8)]:
  pv._L2=lfun(sg,lead); errs=[]
  for c in CASES:
   m1,m2,x,y,z,X,Y,Z,inc,phi=c;g=pv._geometry(*[torch.tensor(float(v),dtype=torch.float64) for v in (*c[:8],inc,phi,20.)]);r=ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(m1*lal.MSUN_SI,m2*lal.MSUN_SI,20.,phi,inc,x,y,z,X,Y,Z,ls.IMRPhenomPv2_V)
   errs.append({'alpha0':abs(float(g[0])-r[4]),'theta_JN':abs(float(g[1])-r[3]),'phi_aligned':abs(float(g[3])-r[5]),'zeta_periodic':abs(((float(g[2])-r[6]+math.pi)%(2*math.pi))-math.pi)})
  rows.append({'variant':name,'cases':errs})
 out=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean');out.mkdir(parents=True,exist_ok=True);(out/'l2-variants.json').write_text(json.dumps({'provenance':'paper-motivated finite variants; no fitting','results':rows},indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
