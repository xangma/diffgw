"""Independent clean-room controls for the MSA Jacobi initialization (Eqs. 23-24)."""
import json,math,torch,hashlib
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation,jacobi_sn,initial_phase,elliptic_K_agm,elliptic_E_quadrature
CASES=[(35.,25.,[.2,.1,.2],[-.1,.15,-.1]),(45.,15.,[.45,-.2,.3],[.1,.2,-.25]),(30.,10.,[.12,-.31,.44],[-.22,.18,-.07])]
def run(m1,m2,c1,c2):
 f=torch.tensor([20.,30.,50.,100.,200.],dtype=torch.float64); s1=torch.tensor(c1,dtype=torch.float64);s2=torch.tensor(c2,dtype=torch.float64)
 a=assemble_precession_msa_angles(f,m1,m2,s1,s2,f_ref=20.,order=5); r=a['roots']; psi=a['psi']; evolving_psi=psi.clone(); sp=r[:,2];sm=r[:,1];s3=r[:,0];ss=spin_oscillation(psi,sp,sm,s3);m=(sp-sm)/(sp-s3)
 # Independent local Jacobi derivative check: d(sn²)/dψ=2 sn cn dn.
 h=1e-6; probe_psi=torch.full_like(m,0.37); psi=probe_psi; sn=jacobi_sn(psi,m); snp=jacobi_sn(psi+h,m);snm=jacobi_sn(psi-h,m);cn=torch.sqrt(torch.clamp(1-sn.square(),min=0));dn=torch.sqrt(torch.clamp(1-m*sn.square(),min=0));analytic=2*(sm-sp)*sn*cn*dn; numeric=(spin_oscillation(psi+h,sp,sm,s3)-spin_oscillation(psi-h,sp,sm,s3))/(2*h)
 err=float(torch.max(torch.abs(analytic-numeric)))
 # Periodic quadrature independently checks the elliptic branch and its mean.
 N=20000; K=elliptic_K_agm(m); u=torch.arange(N,dtype=torch.float64)[...,None]*2*K/N; # period is 2K(m), Eq. (m-def)
 means=[]
 for i in range(5): means.append(float(sp[i]+(sm[i]-sp[i])*jacobi_sn(u[:,i],m[i]).square().mean()))
 return {'masses':[m1,m2],'spins':[c1,c2],'roots_s3_sm_sp':r.tolist(),'psi':evolving_psi.tolist(),'S2':ss.tolist(),'jacobi_derivative_max_abs':err,'periodic_sample_means':means,'leading_means':((sp+sm)/2).tolist(),'elliptic_mean_identity_max_abs':max(abs(means[i]-float(sp[i]-(sp[i]-sm[i])*(elliptic_K_agm(m[i])-elliptic_E_quadrature(m[i]))/(m[i]*elliptic_K_agm(m[i])))) for i in range(5))}
if __name__=='__main__':
 out={'source_sha256':{n:hashlib.sha256(open('src/torchwave/approximants/'+n,'rb').read()).hexdigest() for n in ('precession_msa_initial_clean.py','precession_msa_angles_clean.py')},'equations':['1703.03967v3 Eqs. (Ssq-sol),(m-def),(psidot)'],'provenance':'Fresh clean modules only; independent Jacobi derivative and periodic quadrature controls for 1703.03967v3 Eqs. (23),(24),(Ssq-sol). No oracle fitting or runtime changes.','cases':[run(*x) for x in CASES]};open('docs/reviews/2026-09-13-parity-review/msa-initial-equation-audit.json','w').write(json.dumps(out,indent=2)+'\n');print(json.dumps([(x['masses'],x['jacobi_derivative_max_abs']) for x in out['cases']]))
