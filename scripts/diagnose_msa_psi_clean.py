"""Clean-room Appendix-C psi diagnostic with the MSA PN-beta closure."""
import json, math, hashlib
import lal, lalsimulation, torch
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation
from torchwave.approximants.precession_msa_pn_clean import precession_averages, flux_coefficients, reciprocal_coefficients, anchored_psi
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
FREQS=[20.+180*i/80 for i in range(81)]
CASES=[{"name":"35_25","m1":35.,"m2":25.,"chi1":[.2,.1,.2],"chi2":[-.1,.15,-.1]}, {"name":"45_15","m1":45.,"m2":15.,"chi1":[.45,-.2,.3],"chi2":[.1,.2,-.25]}]
def public(p, version):
 fv=lal.CreateREAL8Vector(len(FREQS)); fv.data[:]=FREQS; d=lal.CreateDict()
 lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,version)
 lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
 out=lalsimulation.SimIMRPhenomXPMSAAngles(fv,p['m1']*lal.MSUN_SI,p['m2']*lal.MSUN_SI,*p['chi1'],*p['chi2'],.7,20.,2,d)
 return [float(x) for x in out[2].data]
def appendix_delta(eta,q,delta,c1,xi,s1sq,s2sq,second_eta):
 d4=delta**4
 left=c1*c1*eta/(q*d4)-2*c1*eta**3*(1+q)*xi/(q*d4)-eta**2*(delta**2*s1sq-eta**2*xi*xi)/d4
 coeff=eta*q if second_eta else eta**2
 right=coeff*c1*c1/d4-2*c1*eta**3*(1+q)*xi/d4-eta**2*(delta**2*s2sq-eta**2*xi*xi)/d4
 return math.sqrt(left*right)
def appendix_psi2(eta,q,delta,c1,xi,sav,s1sq,s2sq,g2g0,second_eta):
 de=appendix_delta(eta,q,delta,c1,xi,s1sq,s2sq,second_eta)
 body=(2*de-2*eta**2/delta**2*sav-10*eta/delta**4*c1*c1+2*eta**2/delta**2*(7+6*q+7*q*q)/(1-q)**2*c1*xi-eta**3/delta**2*(3+4*q+3*q*q)/(1-q)**2*xi*xi+eta/(1-q)**2*(q*(2+q)*s1sq+(1+2*q)*s2sq))
 return 3*g2g0+3*body/(2*eta**3)
def run(p):
 f=torch.tensor(FREQS,dtype=torch.float64); s1=torch.tensor(p['chi1'],dtype=torch.float64); s2=torch.tensor(p['chi2'],dtype=torch.float64)
 M=p['m1']+p['m2']; x1=p['m1']/M; x2=p['m2']/M; eta=x1*x2; q=x2/x1; delta=x1-x2
 base=assemble_precession_msa_angles(f,p['m1'],p['m2'],s1,s2,f_ref=20.,order=5)
 pn=assemble_precession_msa_angles(f,p['m1'],p['m2'],s1,s2,f_ref=20.,order=5,beta_pn=True,beta_compat_102=True)
 S1=x1*x1*s1; S2=x2*x2*s2; Ssum2=float(torch.dot(S1+S2,S1+S2)); roots=base['roots']; v=(math.pi*M*lal.MTSUN_SI*f).pow(1/3); Lpn=torch.stack([orbital_angular_momentum(torch.tensor(p['m1'],dtype=torch.float64),torch.tensor(p['m2'],dtype=torch.float64),f[i],S1[2]/x1**2,S2[2]/x2**2,pn_version=102,compat_102=True) for i in range(len(FREQS))]); Jpn=torch.sqrt(Lpn.square()+2*base['c1']*Lpn/eta+base['sav']); Ssq=spin_oscillation(base['psi'],roots[...,2],roots[...,1],roots[...,0])
 assert abs(float(Ssq[0]-Ssum2))<1e-12
 assert torch.max(torch.abs(pn['cos_beta_pn']))<=1+1e-12
 # Independently reconstruct g2/g0 and the root-derived current coefficients.
 s1sq=float((S1*S1).sum()); s2sq=float((S2*S2).sum()); xi=float((1+q)*S1[2]+(1+1/q)*S2[2]); c1=float(base['c1']); sav=float(base['sav'])
 av=precession_averages(c1,xi,q,eta,base['roots'][0,2],base['roots'][0,1],s1sq,s2sq,float(v[0]))
 fc=flux_coefficients(eta,q,av['s1l'],av['s2l'],s1sq,s2sq,av['s12'],av['s1l2'],av['s2l2'],av['s1ls2l']); gg=reciprocal_coefficients(fc); g0=float(gg['g0']); g2g0=float(gg['g2']/gg['g0'])
 bm2=eta*delta**2; bm1=2*eta*xi-4*c1; b0=q*s1sq+s2sq/q-s1sq-s2sq-2*sav; cm2=4*c1*c1-4*eta*c1*xi+4*eta**3*xi**2-2*eta*delta**2*(s1sq+s2sq); dm2=4*(1-q)/q*(s2sq-q*s1sq)*c1*c1+eta*delta**2*(s1sq-s2sq)**2+4*delta*eta*c1*xi*(s1sq-s2sq); disc=cm2*cm2-4*bm2*dm2; z1=(-cm2+math.sqrt(disc))/(2*bm2); z2=(-cm2-math.sqrt(disc))/(2*bm2); pp=max(z1,z2); nn=min(z1,z2); r1=bm1/(2*bm2); r2=(b0+2*pp+nn)/(2*bm2)-bm1*bm1/(8*bm2*bm2); psi1=1.5*(r1-xi); psi2=3*(r2-xi*r1+g2g0)
 raw=lambda p1,p2: anchored_psi(v,v[0],base['psi_initial'],delta,g0,p1,p2)
 psi_variants={'current':raw(psi1,psi2), 'literal':raw(psi1,appendix_psi2(eta,q,delta,c1,xi,sav,s1sq,s2sq,g2g0,False)), 'delta_etaq':raw(psi1,appendix_psi2(eta,q,delta,c1,xi,sav,s1sq,s2sq,g2g0,True))}
 assert set(psi_variants)=={'current','literal','delta_etaq'}
 assert torch.max(torch.abs(psi_variants['current']-pn['psi'])) < 1e-12
 assert abs(float(Ssq[0]-(S1+S2).square().sum()))<1e-12
 pub={str(v):public(p,v) for v in (220,223)}
 assert abs(pn['cos_beta_pn'][0].item()-pub['220'][0])<1e-12
 inferred=(Jpn.square()+Lpn.square()-2*Jpn*Lpn*torch.tensor(pub['220'],dtype=torch.float64)).tolist()
 out={'parameters':p,'public_cos_beta':pub,'inferred_S2_public220':inferred,'coefficients':{'g0':g0,'g2':float(gg['g2']),'current':[psi1,psi2],'literal':[psi1,appendix_psi2(eta,q,delta,c1,xi,sav,s1sq,s2sq,g2g0,False)],'delta_etaq':[psi1,appendix_psi2(eta,q,delta,c1,xi,sav,s1sq,s2sq,g2g0,True)]},'variants':{}}
 for name,psi in psi_variants.items():
  ss=spin_oscillation(psi,roots[...,2],roots[...,1],roots[...,0]); beta=(Jpn.square()+Lpn.square()-ss)/(2*Jpn*Lpn); assert abs(float(ss[0]-Ssum2))<1e-12; assert torch.max(torch.abs(beta))<=1+1e-12
  if name == 'current': assert torch.max(torch.abs(beta-pn['cos_beta_pn'])) < 1e-12
  out['variants'][name]={str(ver):{'psi':[float(x) for x in psi],'S2':[float(x) for x in ss],'cosbeta':[float(x) for x in beta],'max_abs_error':max(abs(float(beta[i])-pub[str(ver)][i]) for i in range(81))} for ver in (220,223)}
 return out
if __name__=='__main__':
 out={'provenance':'Clean-room PN-beta guard diagnostic; three independent psi prescriptions, compat102 3PN L, c1/Sav J, subtracting evolving S2; no runtime changes/fitting.','script_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'frequencies_hz':FREQS,'sources':['1703.03967v3 Appendix C','fresh precession_msa_angles_clean.py','fresh precession_msa_pn_clean.py'],'cases':[run(c) for c in CASES]}
 open('docs/reviews/2026-09-13-parity-review/msa-psi-reference.json','w').write(json.dumps(out,indent=2)+'\n'); print({c['parameters']['name']:{k:{v:row[v]['max_abs_error'] for v in ('220','223')} for k,row in c['variants'].items()} for c in out['cases']}, {c['parameters']['name']:c['coefficients'] for c in out['cases']})
