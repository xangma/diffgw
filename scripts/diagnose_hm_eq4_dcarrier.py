"""Direct London Eq.(4) amplitude reassembly with a PhenomD carrier."""
import hashlib,json,pathlib,inspect
import numpy as np,torch,lal,lalsimulation as ls
from torchwave.approximants import imrphenom_hm as nm
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm,SUPPORTED_MODES
from torchwave.approximants.imrphenom_hm_clean import map_amplitude_frequency,pn_hhat_lm
from torchwave.waveforms import get_fd_waveform
MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=({'id':'catalog','mass1':40.,'mass2':15.,'chi1':.3,'chi2':-.2,'distance':500.,'inclination':.8,'phic':.2,'f_ref':20.,'flow':20.,'fhigh':512.,'df':.25},{'id':'holdout','mass1':55.,'mass2':11.,'chi1':-.4,'chi2':.2,'distance':680.,'inclination':1.1,'phic':.71,'f_ref':27.,'flow':20.,'fhigh':512.,'df':.25},{'id':'independent_spin','mass1':32.,'mass2':18.,'chi1':-.6,'chi2':.45,'distance':430.,'inclination':.6,'phic':.4,'f_ref':25.,'flow':20.,'fhigh':512.,'df':.25},{'id':'independent_mass_ratio','mass1':70.,'mass2':8.,'chi1':.1,'chi2':-.5,'distance':900.,'inclination':1.,'phic':1.1,'f_ref':30.,'flow':20.,'fhigh':512.,'df':.25})
def metric(x,y):return {'raw_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'magnitude_l2':float(np.linalg.norm(np.abs(x)-np.abs(y))/np.linalg.norm(np.abs(y)))}
def binary(c,a):
 ma=ls.SimInspiralCreateModeArray()
 for l,m in a:ls.SimInspiralModeArrayActivateMode(ma,l,m)
 d=lal.CreateDict();ls.SimInspiralWaveformParamsInsertModeArray(d,ma)
 return ls.SimInspiralChooseFDWaveform(c['mass1']*lal.MSUN_SI,c['mass2']*lal.MSUN_SI,0,0,c['chi1'],0,0,c['chi2'],c['distance']*1e6*lal.PC_SI,c['inclination'],c['phic'],0,0,0,c['df'],c['flow'],c['fhigh'],c['f_ref'],d,ls.GetApproximantFromString('IMRPhenomHM'))
def wave(c,a):
 f=torch.arange(int(c['fhigh']/c['df'])+1,dtype=torch.float64)*c['df'];h,hc=get_fd_waveform(approximant='IMRPhenomHM',mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],f_lower=c['flow'],f_final=c['fhigh'],sample_frequencies=f,modes=a);b,bc=binary(c,a);k=int(c['flow']/c['df']);return f.numpy()[k:],h.detach().numpy()[k:],hc.detach().numpy()[k:],b.data.data[k:],bc.data.data[k:]
def run_eval(c,use_d_carrier=False,use_exact_other=False):
 M=(c['mass1']+c['mass2'])*lal.MTSUN_SI; eta=(c['mass1']*c['mass2'])/(c['mass1']+c['mass2'])**2;delta=(c['mass1']-c['mass2'])/(c['mass1']+c['mass2']);cs=(c['chi1']+c['chi2'])/2;ca=(c['chi1']-c['chi2'])/2; f,h22,hc22,b22,bc22=wave(c,((2,2),)); f,hall,hcall,ball,bcall=wave(c,MODES); out={}; model=IMRPhenomHM(); d=IMRPhenomD(); args=[torch.tensor(c[k],dtype=torch.float64) for k in ('mass1','mass2','chi1','chi2')];mf,af,fr,fd=d.remnant_properties(*args); fr22=float(fr) if use_d_carrier else float(model.mode_qnm_frequencies(2,2,*args)[0]); scale=(c['mass1']+c['mass2'])/float(mf); ftorch=torch.tensor(f,dtype=torch.float64); sum_hp=np.zeros_like(f,dtype=complex);sum_hc=np.zeros_like(f,dtype=complex);alt_hp=np.zeros_like(f,dtype=complex);alt_hc=np.zeros_like(f,dtype=complex)
 for l,m in MODES:
  frlm,fdlm=model.mode_qnm_frequencies(l,m,*args); frlm=fr22 if (l,m)==(2,2) else float(frlm); 
  if use_exact_other and (l,m,0) in SUPPORTED_MODES and (l,m)!=(2,2):frlm=float(dimensionless_qnm(l,m,0,float(af))[0])/(2*np.pi)*scale
  f0=.018*frlm/fr22; fa=map_amplitude_frequency(ftorch*M,f0,frlm,fr22,m).numpy()/M
  _,_,modes=model(sample_frequencies=torch.tensor(fa,dtype=torch.float64),mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],return_unprojected_modes=True)
  def hh(v):
   if (l,m)==(2,1): z=-(2**.5/3)*(delta*v-1.5*(ca+delta*cs)*v**2)
   elif (l,m)==(3,2): z=-(1/3)*(5/7)**.5*(1-3*eta)*v**2
   elif (l,m)==(4,3): z=-(3/4)*(3/35)**.5*delta*(1-2*eta)*v**3
   else:z=pn_hhat_lm((l,m),v,eta=eta,delta=delta,chi_s=cs,chi_a=ca)
   return z*v**-3.5
  vf=(2*np.pi*M*ftorch/m)**(1/3);v2=(2*np.pi*M*(2*ftorch/m)/m)**(1/3);va=(2*np.pi*M*torch.tensor(fa,dtype=torch.float64)/m)**(1/3);qf=hh(vf).numpy();q2=hh(v2).numpy();qa=hh(va).numpy();h22a=-((2*np.pi*M*torch.tensor(fa,dtype=torch.float64)/2)**(1/3)).numpy()**-3.5; beta=np.abs((qa/h22a)*(qf/q2)); beta_alt=np.abs(qf/h22a); native=np.abs(model(sample_frequencies=ftorch,mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],return_unprojected_modes=True)[2][(l,m)].detach().numpy()); carrier=np.abs(modes[(2,2)].detach().numpy()); ratio=np.divide(carrier*beta,native,out=np.zeros_like(native),where=native>1e-300); ratio_alt=np.divide(carrier*beta_alt,native,out=np.zeros_like(native),where=native>1e-300); _,h,hc,b,bc=wave(c,((l,m),)); ah=h*ratio;ac=hc*ratio;alt_h=h*ratio_alt;alt_c=hc*ratio_alt;sum_hp+=ah;sum_hc+=ac;alt_hp+=alt_h;alt_hc+=alt_c; low=f<40;out[f'{l}{m}']={'hp':metric(ah,b),'hc':metric(ac,bc),'low_hp':metric(ah[low],b[low]),'low_hc':metric(ac[low],bc[low]),'alternative_hp':metric(alt_h,b),'alternative_hc':metric(alt_c,bc)}
 assert np.isfinite(sum_hp).all() and np.isfinite(sum_hc).all(); assert np.max(np.abs(beta[native>1e-300]-1.0))<1e-10 if (l,m)==(2,2) else True
 low=f<40;out['all']={'hp':metric(sum_hp,ball),'hc':metric(sum_hc,bcall),'alternative_hp':metric(alt_hp,ball),'alternative_hc':metric(alt_hc,bcall),'low_hp':metric(sum_hp[low],ball[low]),'low_hc':metric(sum_hc[low],bcall[low])};return out
def run():
 p=pathlib.Path(inspect.getfile(IMRPhenomHM));r={'provenance':'London arXiv:1708.00404 Eqs.(3)-(4), clean Mishra Eq.(11)-(12) factors, PhenomD remnant carrier f22; native phase retained by amplitude rescaling. exact mode QNMs use independent Kerr where supported.','native_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':[]}
 for c in CASES:r['cases'].append({'id':c['id'],'dcarrier':run_eval(c,True,False),'berti22_control':run_eval(c,False,False),'dcarrier_exact_other':run_eval(c,True,True)})
 r['script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':run()
