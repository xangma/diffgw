"""Low-frequency clean-room test of the Mishra London (2,1) PN factor.

At these frequencies London Eq.5 is exactly f^A_22=2 f/m, so QNM joins do
not enter. Public HM mode amplitudes and public IMRPhenomD 22 amplitudes are
black-box observations; the three predictions are the printed Mishra
Eq.12b truncations, with no fitted multiplier.
"""
import json,hashlib,pathlib
import numpy as np,torch,lal,lalsimulation
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm_clean import pn_hhat_lm
from torchwave.approximants.hm_21_pn_clean import hhat21_eq12b_v4,hhat21_spa_spin_v4,hhat21_nonspin_v4,hhat21_arun_compat_v4
MTSUN=4.925490947641267e-6
BASE=((40.,15.,.3,-.2,500.),(55.,11.,-.4,.2,680.),(32.,18.,-.6,.45,430.),(70.,8.,.1,-.5,900.))
EXTRA=((48.,13.,.72,-.31,610.),(63.,19.,-.27,.58,740.),(36.,9.,-.81,-.12,520.),(82.,7.,.44,-.66,1020.))
FREQ=(1.,2.,5.,10.,20.)
def modes(f,c):
 m1,m2,s1,s2,d=c;v=lal.CreateREAL8Vector(len(f));v.data[:]=f
 h=lalsimulation.SimIMRPhenomHMGethlmModes(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,s1,0.,0.,s2,0.,0.,20.,lal.CreateDict());o={};n=h
 while n:o[(int(n.l),int(n.m))]=torch.tensor(n.mode.data.data,dtype=torch.complex128);n=n.next
 return o
def d22amp(d,f,m1,m2,s1,s2,dist):
 hp,hc=d(torch.as_tensor(f,dtype=torch.float64),m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=None)
 y=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2))
 # This public D convention has hplus+i hcross = Y22 h22.
 return torch.abs((hp+1j*hc)/y)
def one(c):
 m1,m2,s1,s2,dist=c;mt=m1+m2;M=mt*MTSUN;eta=torch.tensor(m1*m2/mt**2,dtype=torch.float64);delta=torch.tensor((m1-m2)/mt,dtype=torch.float64);cs=torch.tensor((s1+s2)/2,dtype=torch.float64);ca=torch.tensor((s1-s2)/2,dtype=torch.float64);d=IMRPhenomD();ff=torch.tensor(FREQ,dtype=torch.float64);raw=modes(FREQ,c);scale=mt*lal.MTSUN_SI*(mt*lal.MRSUN_SI)/(dist*1e6*lal.PC_SI);obs=torch.abs(raw[(2,1)])*scale/d22amp(d,2*ff,m1,m2,s1,s2,dist)
 v=(2*np.pi*M*ff/1).pow(1/3);v2=(2*np.pi*M*(2*ff/1)/2).pow(1/3)
 def H(x,y): return y*x.pow(-3.5)
 leading=-(torch.sqrt(v.new_tensor(2.))/3.)*delta*v;spin=-(torch.sqrt(v.new_tensor(2.))/3.)*(delta*v-1.5*(ca+delta*cs)*v**2);full=pn_hhat_lm((2,1),v,eta,delta,cs,ca);full_v4=hhat21_eq12b_v4(v,eta,delta,cs,ca);old_spa=full+hhat21_spa_spin_v4(v,eta,delta,cs,ca);nonspin_v4=full+hhat21_nonspin_v4(v,eta,delta,cs,ca);compat_v4=hhat21_arun_compat_v4(v,eta,delta,cs,ca)
 # H21(f)/H22(2f), with H22=-v_22^-7/2.
 pred=lambda q:torch.abs(H(v,q)/H(v2,-torch.ones_like(v2)))
 vals={'published_leading':pred(leading),'published_leading_spin':pred(spin),'published_full_1p5pn':pred(full),'arun_eq12b_v4':pred(full_v4),'arun_legacy_spa_spin_v4':pred(old_spa),'nonspin_tail_v4_control':pred(nonspin_v4),'arun_td_spin_sign_control':pred(compat_v4)}
 return [{'f_hz':x,'observed_ratio':float(obs[i]),**{k:float(q[i]) for k,q in vals.items()},**{k+'_relative_error':float(torch.abs(q[i]-obs[i])/torch.abs(obs[i])) for k,q in vals.items()}} for i,x in enumerate(FREQ)]
def main():
 cases=[]
 for c in BASE+EXTRA:
  cases.append({'label':'catalog_or_holdout','inputs':c,'samples':one(c)})
  ns=(c[0],c[1],0.,0.,c[4]); cases.append({'label':'nonspin_control','inputs':ns,'samples':one(ns)})
  rev=(c[0],c[1],-c[2],-c[3],c[4]); cases.append({'label':'spin_reversal_control','inputs':rev,'samples':one(rev)})
 out={'primary_equations':['https://arxiv.org/html/1708.00404 Eq.4-5','https://arxiv.org/abs/1601.05588 Eq.12b (H21)','https://arxiv.org/abs/0810.5336 Appendix D'],'contract':'low branch fA_22=2f/m; H_lm=Hhat_lm*v^-7/2 with mode-independent Eq.11 prefactor','v4_provenance':'Eq.12b/H21 displayed coefficient: chi_a*(4771/1344-11941 eta/336)+delta chi_s*(4771/1344-2549 eta/336)+delta*(-pi-i/2-2 i log 2); no inferred coefficients','arun_0810_5336_audit':'Arun Appendix D gives the frequency-domain C_k coefficients. Its displayed S_3 spin term times Newtonian C_1^(1) is included as arun_legacy_spa_spin_v4; this is only the inherited SPA product, not a new direct mode coefficient.','arun_td_sign_control':'Diagnostic control: c_TD=(1/2-205 eta/14)chi_a+delta(1/2-33 eta/14)chi_s, so c_tail+c_SPA-c_TD reverses only the displayed TD spin contribution; no fitted coefficients.','cases':cases,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'clean_room':'public callable/black-box outputs and primary equations; no legacy or implementation bodies; no fitted constants'}
 pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(pathlib.Path(__file__).with_suffix('.json'))
if __name__=='__main__':main()
