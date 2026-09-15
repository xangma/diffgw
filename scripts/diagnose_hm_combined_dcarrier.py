"""Combined process-local HM diagnostic; no production edits."""
import hashlib,inspect,json,pathlib
import numpy as np,torch,lal,lalsimulation as ls
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants import imrphenom_hm as nmod
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm_clean import pn_hhat_lm
from torchwave.waveforms import get_fd_waveform
MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=({'id':'catalog','mass1':40.,'mass2':15.,'chi1':.3,'chi2':-.2,'distance':500.,'inclination':.8,'phic':.2,'f_ref':20.,'flow':20.,'fhigh':512.,'df':.25},{'id':'holdout','mass1':55.,'mass2':11.,'chi1':-.4,'chi2':.2,'distance':680.,'inclination':1.1,'phic':.71,'f_ref':27.,'flow':20.,'fhigh':512.,'df':.25})
def metric(x,y):return {'raw_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'magnitude_l2':float(np.linalg.norm(np.abs(x)-np.abs(y))/np.linalg.norm(np.abs(y)))}
def binary(c,a):
 ma=ls.SimInspiralCreateModeArray()
 for l,m in a:ls.SimInspiralModeArrayActivateMode(ma,l,m)
 d=lal.CreateDict();ls.SimInspiralWaveformParamsInsertModeArray(d,ma)
 return ls.SimInspiralChooseFDWaveform(c['mass1']*lal.MSUN_SI,c['mass2']*lal.MSUN_SI,0,0,c['chi1'],0,0,c['chi2'],c['distance']*1e6*lal.PC_SI,c['inclination'],c['phic'],0,0,0,c['df'],c['flow'],c['fhigh'],c['f_ref'],d,ls.GetApproximantFromString('IMRPhenomHM'))
def wave(c,a):
 f=torch.arange(int(c['fhigh']/c['df'])+1,dtype=torch.float64)*c['df'];h,hc=get_fd_waveform(approximant='IMRPhenomHM',mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],f_lower=c['flow'],f_final=c['fhigh'],sample_frequencies=f,modes=a);b,bc=binary(c,a);k=int(c['flow']/c['df']);return h.detach().numpy()[k:],hc.detach().numpy()[k:],b.data.data[k:],bc.data.data[k:]
def run():
 path=pathlib.Path(inspect.getfile(IMRPhenomHM)); oldh=nmod.h_hat_multipole; oldq=IMRPhenomHM.mode_qnm_frequencies; out={'convention':'Eq.4 clean amplitude factors are represented here by the published PN substitutions; the D carrier substitution is isolated through the documented QNM API. Higher-mode QNMs remain native.','native_source_sha256_before':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':[]}
 for c in CASES:
  a=[torch.tensor(c[k],dtype=torch.float64) for k in ('mass1','mass2','chi1','chi2')];mf,af,fr,fd=IMRPhenomD().remnant_properties(*a); calls={'h':0,'q':0}
  def hsub(l,m,v,d,e,cs,ca):
   calls['h']+=1
   if (l,m)==(2,1): return -(2**.5/3)*(d*v-1.5*(ca+d*cs)*v**2)
   if (l,m)==(3,2): return -(1/3)*(5/7)**.5*(1-3*e)*v**2
   if (l,m)==(4,3): return -(3/4)*(3/35)**.5*d*(1-2*e)*v**3
   return oldh(l,m,v,d,e,cs,ca)
  def qsub(l,m,m1,m2,s1,s2):
   calls['q']+=1
   if (l,m)==(2,2): return m1.new_tensor(float(fr)),m1.new_tensor(float(fd))
   return oldq(l,m,m1,m2,s1,s2)
  try:
   base={};
   for label,a0 in [('mode21',((2,1),)),('mode32',((3,2),)),('mode43',((4,3),)),('all',MODES)]:
    h,hc,b,bc=wave(c,a0);base[label]={'hp':metric(h,b),'hc':metric(hc,bc)}
   nmod.h_hat_multipole=hsub;IMRPhenomHM.mode_qnm_frequencies=staticmethod(qsub);changed={}
   for label,a0 in [('mode21',((2,1),)),('mode32',((3,2),)),('mode43',((4,3),)),('all',MODES)]:
    h,hc,b,bc=wave(c,a0); low=slice(0,int(20/c['df'])); changed[label]={'hp':metric(h,b),'hc':metric(hc,bc),'20_40Hz':{'hp':metric(h[low],b[low]),'hc':metric(hc[low],bc[low])}}
   out['cases'].append({'id':c['id'],'d_carrier':{'Mf':float(mf),'af':float(af),'f_rd_M':float(fr),'f_damp_M':float(fd)},'baseline':base,'combined':changed,'calls':calls})
  finally: nmod.h_hat_multipole=oldh;IMRPhenomHM.mode_qnm_frequencies=staticmethod(oldq)
 out['native_source_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest();p=pathlib.Path(__file__);out['script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();p.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
