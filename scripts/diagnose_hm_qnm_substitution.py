"""Process-local exact-Kerr QNM substitution, one mode at a time."""
import hashlib, inspect, json, pathlib
import numpy as np, torch, lal, lalsimulation as lalsim
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm, SUPPORTED_MODES
from torchwave.waveforms import get_fd_waveform

MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=({'id':'catalog','mass1':40.,'mass2':15.,'chi1':.3,'chi2':-.2,'distance':500.,'inclination':.8,'phic':.2,'f_ref':20.,'flow':20.,'fhigh':512.,'df':.25},{'id':'holdout','mass1':55.,'mass2':11.,'chi1':-.4,'chi2':.2,'distance':680.,'inclination':1.1,'phic':.71,'f_ref':27.,'flow':20.,'fhigh':512.,'df':.25})
def binary(c,active):
 ma=lalsim.SimInspiralCreateModeArray()
 for l,m in active:lalsim.SimInspiralModeArrayActivateMode(ma,l,m)
 d=lal.CreateDict();lalsim.SimInspiralWaveformParamsInsertModeArray(d,ma)
 return lalsim.SimInspiralChooseFDWaveform(c['mass1']*lal.MSUN_SI,c['mass2']*lal.MSUN_SI,0,0,c['chi1'],0,0,c['chi2'],c['distance']*1e6*lal.PC_SI,c['inclination'],c['phic'],0,0,0,c['df'],c['flow'],c['fhigh'],c['f_ref'],d,lalsim.GetApproximantFromString('IMRPhenomHM'))
def metric(x,y):return {'raw_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'magnitude_l2':float(np.linalg.norm(np.abs(x)-np.abs(y))/np.linalg.norm(np.abs(y)))}
def waveform(c,active):
 f=torch.arange(int(round(c['fhigh']/c['df']))+1,dtype=torch.float64)*c['df']; h,hc=get_fd_waveform(approximant='IMRPhenomHM',mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],f_lower=c['flow'],f_final=c['fhigh'],sample_frequencies=f,modes=active); b,bc=binary(c,active); k=int(round(c['flow']/c['df'])); return h.detach().numpy()[k:],hc.detach().numpy()[k:],b.data.data[k:],bc.data.data[k:]
def run():
 path=pathlib.Path(inspect.getfile(IMRPhenomHM)); report={'native_qnm_signature':str(inspect.signature(IMRPhenomHM.mode_qnm_frequencies)),'source_provenance':'Exact Kerr values use independent kerr_qnm_generated eigenvalues; native method docstring identifies Berti-Cardoso-Will (2006) Table VIII fits. Remnant inputs come from the exposed IMRPhenomD.remnant_properties API.','native_source_sha256_before':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':[]}
 for c in CASES:
  args=[torch.tensor(c[k],dtype=torch.float64) for k in ('mass1','mass2','chi1','chi2')]; mf,af,_,_=IMRPhenomD().remnant_properties(*args); scale=(c['mass1']+c['mass2'])/float(mf); old=IMRPhenomHM.mode_qnm_frequencies
  for target in MODES:
   if (target[0],target[1],0) not in SUPPORTED_MODES:continue
   def exact(l,m,m1,m2,chi1,chi2,target=target):
    if (l,m)!=target:return old(l,m,m1,m2,chi1,chi2)
    wr,wi=dimensionless_qnm(l,m,0,float(af)); return (m1.new_tensor(float(wr)/(2*np.pi)*scale),m1.new_tensor(float(wi)/(2*np.pi)*scale))
   base={}; changed={}
   try:
    for label,active in [('mode',(target,)),('all',MODES)]:
     h,hc,b,bc=waveform(c,active); base[label]={'hp':metric(h,b),'hc':metric(hc,bc)}
    IMRPhenomHM.mode_qnm_frequencies=staticmethod(exact)
    for label,active in [('mode',(target,)),('all',MODES)]:
     h,hc,b,bc=waveform(c,active); changed[label]={'hp':metric(h,b),'hc':metric(hc,bc)}
   finally: IMRPhenomHM.mode_qnm_frequencies=staticmethod(old)
   # Regenerate native baseline after restoration; this catches leakage.
   h,hc,b,bc=waveform(c,(target,)); restored={'hp':metric(h,b),'hc':metric(hc,bc)}
   report['cases'].append({'id':c['id'],'target':f'{target[0]}{target[1]}','remnant':{'Mf':float(mf),'af':float(af),'M_over_Mf':scale},'baseline':base,'exact_kerr_substitution':changed,'restored_baseline':restored})
 report['native_source_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest();p=pathlib.Path(__file__);report['script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run()
