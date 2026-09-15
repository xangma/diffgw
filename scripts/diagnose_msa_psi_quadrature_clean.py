"""Direct-root quadrature of 1703.03967v3 Eqs. (psidot),(vdot),(Ssq-sol)."""
import hashlib,json,math
import lal,lalsimulation,torch
from scipy.integrate import quad
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation,cubic_coefficients,ordered_roots
from torchwave.approximants.precession_msa_pn_clean import precession_averages,flux_coefficients,reciprocal_coefficients
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
F=[20.+180*i/80 for i in range(81)]; CASES=[(35.,25.,[.2,.1,.2],[-.1,.15,-.1]),(45.,15.,[.45,-.2,.3],[.1,.2,-.25]),(30.,10.,[.12,-.31,.44],[-.22,.18,-.07])]
def public(m1,m2,c1,c2):
 x=lal.CreateREAL8Vector(len(F));x.data[:]=F;d=lal.CreateDict();lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,220);lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5);o=lalsimulation.SimIMRPhenomXPMSAAngles(x,m1*lal.MSUN_SI,m2*lal.MSUN_SI,*c1,*c2,.7,20.,2,d);return [float(z) for z in o[2].data]
def run(m1,m2,chi1,chi2,eps):
 f=torch.tensor(F,dtype=torch.float64);s1=torch.tensor(chi1,dtype=torch.float64);s2=torch.tensor(chi2,dtype=torch.float64);M=m1+m2;x1=m1/M;x2=m2/M;eta=x1*x2;q=x2/x1;delta=x1-x2
 a=assemble_precession_msa_angles(f,m1,m2,s1,s2,f_ref=20.,order=5);v=(math.pi*M*4.925490947641267e-6*f).pow(1/3);roots=a['roots'];sp,sm,s3=roots[:,2],roots[:,1],roots[:,0]; S1=x1*x1*s1;S2=x2*x2*s2;s1sq=float((S1*S1).sum());s2sq=float((S2*S2).sum());xi=float((1+q)*S1[2]+(1+1/q)*S2[2]);sav=float(a['sav']);c1=float(a['c1']);av=precession_averages(c1,xi,q,eta,sp[0],sm[0],s1sq,s2sq,v[0]);fc=flux_coefficients(eta,q,av['s1l'],av['s2l'],s1sq,s2sq,av['s12'],av['s1l2'],av['s2l2'],av['s1ls2l']);g=reciprocal_coefficients(fc);
 def sr(x):
  xx=torch.tensor(x,dtype=torch.float64);L=eta/xx;J=torch.sqrt(L*L+2*c1*L/eta+sav);z=ordered_roots(*cubic_coefficients(torch.tensor(q,dtype=torch.float64),torch.tensor(eta,dtype=torch.float64),torch.tensor(delta,dtype=torch.float64),torch.tensor(s1sq,dtype=torch.float64),torch.tensor(s2sq,dtype=torch.float64),L,J,torch.tensor(xi,dtype=torch.float64)));return [float(z[i]) for i in range(3)]
 def rhs(x):
  T,N,P=sr(x);G=sum(float(g[f'g{i}'])*x**i for i in range(6));return 2.25/math.sqrt(eta)*(1-xi*x)*x**-3*math.sqrt(P-T)*G
 qpsi=[float(a['psi_initial'])]
 for i in range(1,len(F)):qpsi.append(qpsi[-1]+quad(rhs,float(v[i-1]),float(v[i]),epsabs=eps,epsrel=eps,limit=100)[0])
 qpsi=torch.tensor(qpsi,dtype=torch.float64);S2q=spin_oscillation(qpsi,sp,sm,s3);S2ser=spin_oscillation(a['psi'],sp,sm,s3);Lpn=torch.stack([orbital_angular_momentum(torch.tensor(m1,dtype=torch.float64),torch.tensor(m2,dtype=torch.float64),f[i],S1[2]/x1**2,S2[2]/x2**2,pn_version=102,compat_102=True) for i in range(81)]);Jpn=torch.sqrt(Lpn*Lpn+2*a['c1']*Lpn/eta+a['sav']);bq=(Jpn*Jpn+Lpn*Lpn-S2q)/(2*Jpn*Lpn);bs=(Jpn*Jpn+Lpn*Lpn-S2ser)/(2*Jpn*Lpn);p220=public(m1,m2,chi1,chi2);s220=Jpn*Jpn+Lpn*Lpn-2*Jpn*Lpn*torch.tensor(p220,dtype=torch.float64)
 return {'masses':[m1,m2],'Lpn':Lpn.tolist(),'Jpn':Jpn.tolist(),'public220_cosbeta':p220,'S2_public220':s220.tolist(),'psi_quadrature':qpsi.tolist(),'psi_series':a['psi'].tolist(),'S2_quadrature':S2q.tolist(),'S2_series':S2ser.tolist(),'beta_quadrature':bq.tolist(),'beta_series':bs.tolist(),'max_differences':{'psi':float((qpsi-a['psi']).abs().max()),'S2':float((S2q-S2ser).abs().max()),'beta':float((bq-bs).abs().max()),'beta_public220':float((bq-torch.tensor(p220,dtype=torch.float64)).abs().max())},'vdot':'v^9/[3 sum(g_n v^n)]','normalization':'a0=96 eta/5, g0=1/a0'}
if __name__=='__main__':
 runs={'1e-9':[run(*c,1e-9) for c in CASES],'1e-11':[run(*c,1e-11) for c in CASES]}
 conv=[]
 for i,c in enumerate(CASES):
  lo,hi=runs['1e-9'][i],runs['1e-11'][i]
  vals={k:max(abs(a-b) for a,b in zip(lo[k],hi[k])) for k in ('psi_quadrature','S2_quadrature','beta_quadrature')}
  conv.append({'masses':[c[0],c[1]],'max_abs_1e-9_minus_1e-11':vals})
  assert max(vals.values()) < 1e-8, vals
 out={'equations':['1703.03967v3 psidot','1703.03967v3 vdot','1703.03967v3 Ssq-sol'],'tolerances':[1e-9,1e-11],'convergence':conv,'source_sha256':{n:hashlib.sha256(open('src/torchwave/approximants/'+n,'rb').read()).hexdigest() for n in ('precession_msa_initial_clean.py','precession_msa_pn_clean.py','precession_beta_clean.py')},'cases':runs}
 out['script_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
 open('docs/reviews/2026-09-13-parity-review/msa-psi-quadrature-reference.json','w').write(json.dumps(out,indent=2)+'\n')
 print([(x['masses'],x['max_differences']) for x in runs['1e-11']],conv)
