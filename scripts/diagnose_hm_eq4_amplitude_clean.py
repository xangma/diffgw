"""Clean-room London Eq.(4) amplitude reconstruction control.

PN factors are the fresh Mishra et al. Eqs. 11--12 transcription. Public HM
modes are black-box targets; no scale or phase is fitted.
"""
import hashlib,json,pathlib
import numpy as np
import torch, lal, lalsimulation
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm_clean import map_amplitude_frequency,pn_hhat_lm
from torchwave.approximants.hm_21_pn_clean import hhat21_eq12b_v4
from torchwave.approximants.imrphenom_hm_phase_clean import mapped_phase_c1, phenomd_eq14_phase_delta_c1
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm
MTSUN=4.925490947641267e-6
MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=((40.,15.,.3,-.2,500.,.8,.2,20.),(55.,11.,-.4,.2,680.,1.1,.71,27.),(32.,18.,-.6,.45,430.,.6,.4,25.),(70.,8.,.1,-.5,900.,1.,1.1,30.))
def public_modes(f,c):
 m1,m2,s1,s2,dist,inc,ph,rf=c; v=lal.CreateREAL8Vector(len(f));v.data[:]=f
 h=lalsimulation.SimIMRPhenomHMGethlmModes(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,s1,0.,0.,s2,ph,float(f[1]-f[0]) if len(f)>1 else 0.,rf,lal.CreateDict());o={};n=h
 while n:o[(int(n.l),int(n.m))]=torch.tensor(n.mode.data.data,dtype=torch.complex128);n=n.next
 return o
def native(c,f):
 m1,m2,s1,s2,dist,inc,ph,rf=c
 return IMRPhenomHM()(f,mass1=m1,mass2=m2,chi1=s1,chi2=s2,distance=dist,inclination=inc,phic=ph,tc=0.,f_ref=rf,f_lower=20.,return_unprojected_modes=True)
def projector(z,inc):
 hp=torch.zeros_like(next(iter(z.values())));hc=hp.clone()
 for (ell,m),q in z.items():
  yp=complex(lal.SpinWeightedSphericalHarmonic(inc,0.,-2,ell,m));yn=complex(lal.SpinWeightedSphericalHarmonic(inc,0.,-2,ell,-m));p=(-1)**ell
  hp=hp+.5*q*(yp+p*np.conj(yn));hc=hc-.5j*q*(yp-p*np.conj(yn))
 return hp,hc
def d_amplitude(d,fa,m1,m2,s1,s2,dist):
 # D black-box complex strain at face-on; h22=2(h+ - i hc)/Y22.
 hp,hc=d(fa/MTSUN if False else fa,m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=None)
 y22=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2))
 return torch.abs((hp+1j*hc)/y22)

def metric(a,b,mask):
 n=int(mask.sum())
 if n==0:return {'sample_count':0}
 aa,bb=a[mask],b[mask];den=torch.linalg.vector_norm(bb)
 if not bool(torch.isfinite(den)) or float(den)==0:return {'sample_count':n,'relative_l2':None,'magnitude_l2':None}
 return {'sample_count':n,'relative_l2':float(torch.linalg.vector_norm(aa-bb)/den),'magnitude_l2':float(torch.linalg.vector_norm(torch.abs(aa)-torch.abs(bb))/torch.linalg.vector_norm(torch.abs(bb)))}
def pn_variant(mode,v,eta,delta,cs,ca,variant):
 if variant=='eq12b_v4' and mode==(2,1):return hhat21_eq12b_v4(v,eta,delta,cs,ca)
 if variant=='full':return pn_hhat_lm(mode,v,eta,delta,cs,ca)
 if mode==(2,1):
  if variant=='leading':return -(torch.sqrt(v.new_tensor(2.))/3.)*delta*v
  if variant=='spin':return -(torch.sqrt(v.new_tensor(2.))/3.)*(delta*v-1.5*(ca+delta*cs)*v**2)
 if mode==(3,2):
  base=-(torch.sqrt(v.new_tensor(5./7.))/3.)*(1.-3.*eta)*v**2
  if variant=='leading':return base
  if variant=='spin':return base-(torch.sqrt(v.new_tensor(5./7.))/3.)*4.*eta*cs*v**3
 return pn_hhat_lm(mode,v,eta,delta,cs,ca)
def main():
 f=torch.linspace(20.,512.,257,dtype=torch.float64);rows=[]
 for c in CASES:
  m1,m2,s1,s2,dist,inc,ph,rf=c;mt=m1+m2;M=mt*MTSUN;eta=torch.tensor(m1*m2/mt**2,dtype=torch.float64);delta=torch.tensor((m1-m2)/mt,dtype=torch.float64);cs=torch.tensor((s1+s2)/2,dtype=torch.float64);ca=torch.tensor((s1-s2)/2,dtype=torch.float64)
  target=public_modes(f.numpy(),c);dmodel=IMRPhenomD();hp_native,hc_native,native_modes=native(c,f);S=mt*lal.MTSUN_SI*(mt*lal.MRSUN_SI)/(dist*1e6*lal.PC_SI);target={k:q*complex(S) for k,q in target.items()}
  native_for_projection={k:(-1)**k[0]*native_modes[k] for k in MODES};hp_rec,hc_rec=projector(native_for_projection,inc);selfcheck={'hp':float(torch.max(torch.abs(hp_rec-hp_native))/torch.max(torch.abs(hp_native))),'hc':float(torch.max(torch.abs(hc_rec-hc_native))/torch.max(torch.abs(hc_native)))}
  mf,af,dfr22,dfd22=map(float,IMRPhenomD().remnant_properties(torch.tensor(m1,dtype=torch.float64),torch.tensor(m2,dtype=torch.float64),torch.tensor(s1,dtype=torch.float64),torch.tensor(s2,dtype=torch.float64)));q22c=dimensionless_qnm(af,2,2);q22=(q22c.real/(2*np.pi)*mt/mf,q22c.imag/(2*np.pi)*mt/mf);slope=float(IMRPhenomD().compute_slope_peak(torch.tensor(m1,dtype=torch.float64),torch.tensor(m2,dtype=torch.float64),torch.tensor(s1,dtype=torch.float64),torch.tensor(s2,dtype=torch.float64)));chip=(m1*s1+m2*s2)/mt-38*(m1*m2/mt**2)/113*(s1+s2);a4=float(IMRPhenomD().eval_table_v('alpha4',eta,torch.tensor(chip,dtype=torch.float64)));a5=float(IMRPhenomD().eval_table_v('alpha5',eta,torch.tensor(chip,dtype=torch.float64)));mode_rows={};amp_before={};amp_after={}
  for ell,m in MODES:
   fr=dimensionless_qnm(af,ell,m).real/(2*np.pi)*mt/mf;f0=.018*fr/q22[0];fa=map_amplitude_frequency(f*M,f0,fr,q22[0],m)/M
   def c0(x): return IMRPhenomD()(x/M,m1,m2,s1,s2,phic=0.,tc=0.,f_ref=rf,return_phase=True)[2]
   def cp(x): return c0(x)+phenomd_eq14_phase_delta_c1(x,eta,torch.tensor(chip,dtype=torch.float64),q22[0],fr,q22[1],dimensionless_qnm(af,ell,m).imag/(2*np.pi)*mt/mf,a4,a5,qnm_ratio=q22[0]/fr)
   phase=-mapped_phase_c1(f*M,f0,fr,q22[0],m,cp,ell=ell)+(m/2.-1.)*slope*M*rf+m*ph
   v=(2*np.pi*M*f/m).pow(1/3);v2=(2*np.pi*M*(2*f/m)/m).pow(1/3);va=(2*np.pi*M*fa/m).pow(1/3);va22=(2*np.pi*M*fa/2).pow(1/3)
   def H(vv,mode,variant): # Eq.11 prefactor is mode-independent; Eq.12 is in pn_hhat_lm.
    return pn_variant(mode,vv,eta,delta,cs,ca,variant)*vv.pow(-3.5)
   variants=['leading','full'] if (ell,m) not in ((2,1),(3,2)) else ['leading','spin','full'];
   if (ell,m)==(2,1): variants.append('eq12b_v4')
   native_amp=torch.abs(native_modes[(ell,m)]);target_mode=target[(ell,m)];target_amp=torch.abs(target_mode);valid0=torch.isfinite(native_amp)&torch.isfinite(target_amp)&(native_amp>0)&(target_amp>0);amp_before[(ell,m)]=native_amp*torch.exp(1j*phase);best={}
   for variant in variants:
    beta=torch.abs((H(va,(ell,m),variant)/H(va22,(2,2),'full'))*(H(v,(ell,m),variant)/H(v2,(ell,m),variant)));A22=d_amplitude(dmodel,fa,m1,m2,s1,s2,dist);pred=A22*beta;valid=valid0&torch.isfinite(pred)&(pred>0)
    amp_after[(ell,m,variant)]=pred*torch.exp(1j*phase);bands={b:valid&(f*M<cut) for b,cut in [('inspiral',f0)]};bands.update({'full':valid,'intermediate':valid&(f*M>=f0)&(f*M<=fr),'mr':valid&(f*M>fr)}); f014=.014*fr/q22[0]; fa14=map_amplitude_frequency(f*M,f014,fr,q22[0],m)/M; va14=(2*np.pi*M*fa14/m).pow(1/3); va2214=(2*np.pi*M*fa14/2).pow(1/3); beta14=torch.abs((H(va14,(ell,m),variant)/H(va2214,(2,2),'full'))*(H(v,(ell,m),variant)/H(v2,(ell,m),variant))); pred14=d_amplitude(dmodel,fa14,m1,m2,s1,s2,dist)*beta14; valid14=valid0&torch.isfinite(pred14)&(pred14>0); best[variant]={'beta_source':'London Eq.4: [Hlm(fA)/H22(fA)] [Hlm(f)/Hlm(2f/m)], Hhat=pn_hhat*v^-7/2 ; map Eq.5','published_.018':{b:metric(pred,target_amp,mm) for b,mm in bands.items()},'compatibility_.014_full':metric(pred14,target_amp,valid14)}
   mode_rows[f'{ell}{m}']={'f0_amplitude':f0,'join_controls_Mf':{'published_amplitude':.018*fr,'compatibility_phase':.014*fr},'f_rd_lm':fr,'map_jacobian_low_mid_high':(2/m,(q22[0]-2*f0/m)/(fr-f0),1.),'truncations':best}
  pt=projector({k:q for k,q in target.items() if k in MODES},inc);variant_names=sorted({k[2] for k in amp_after});whole={}
  for vn in variant_names:
   zz={mode:(amp_after[(mode[0],mode[1],vn)] if (mode[0],mode[1],vn) in amp_after else amp_before[mode]) for mode in amp_before};pp=projector(zz,inc);valid=torch.isfinite(pp[0])&torch.isfinite(pp[1])&torch.isfinite(pt[0])&torch.isfinite(pt[1]); bands={'full':valid,'low_20_80':valid&(f<80),'mid_80_200':valid&(f>=80)&(f<200),'high_200_512':valid&(f>=200)}; whole[vn]={b:{'hp':metric(pp[0],pt[0],mm),'hc':metric(pp[1],pt[1],mm)} for b,mm in bands.items()}
  rows.append({'inputs':c,'normalization_S':S,'phase_convention':'physical clean phase = -mapped(D positive-p + London Eq14 delta) + (m/2-1)*slope_peak*M*f_ref + m*phic','native_reconstruction_selfcheck':selfcheck,'native_vs_public_target':{f'{k[0]}{k[1]}':metric((-1)**k[0]*native_modes[k],target[k],torch.ones_like(f,dtype=torch.bool)) for k in MODES},'modes':mode_rows,'whole_polarization_clean_phase_eq4_amplitude':whole})
 def finite_json(x):
  if isinstance(x,float) and not np.isfinite(x): return None
  if isinstance(x,dict): return {k:finite_json(v) for k,v in x.items()}
  if isinstance(x,list): return [finite_json(v) for v in x]
  return x
 out={'paper_eq4':'https://arxiv.org/html/1708.00404 (Eq.4; Eq.5 amplitude map, .018 join)','pn_source':'Mishra et al., arXiv:1601.05588, Eqs.11-12; 21 leading+spin+nonspin v^3 (1.5PN), 32 leading or spin v^3, 33/44/43 leading terms; London arXiv:1708.00404 Eq.4','cases':rows,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'clean_room':'public callables/black boxes and printed primary equations; no production edits'}
 pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(finite_json(out),indent=2,allow_nan=False)+'\n');print(pathlib.Path(__file__).with_suffix('.json'))
if __name__=='__main__':main()
