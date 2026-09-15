"""Clean-room f_ref phase/S^2 derivative control against public PhenomX MSA 220."""
import hashlib,json,math
import lal,lalsimulation,torch
from scipy.special import ellipj
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation,jacobi_sn,cubic_coefficients,ordered_roots
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
F=[19.9,19.99,19.999,20.,20.001,20.01,20.1]+[float(x) for x in range(30,201,10)]
CASES=[(35.,25.,[.2,.1,.2],[-.1,.15,-.1]),(45.,15.,[.45,-.2,.3],[.1,.2,-.25]),(30.,10.,[.12,-.31,.44],[-.22,.18,-.07])]
def pub(m1,m2,c1,c2):
 x=lal.CreateREAL8Vector(len(F));x.data[:]=F;d=lal.CreateDict();lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,220);lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
 o=lalsimulation.SimIMRPhenomXPMSAAngles(x,m1*lal.MSUN_SI,m2*lal.MSUN_SI,*c1,*c2,.7,20.,2,d);return torch.tensor(list(o[2].data),dtype=torch.float64)
def one(m1,m2,c1,c2):
 f=torch.tensor(F,dtype=torch.float64); s1=torch.tensor(c1,dtype=torch.float64);s2=torch.tensor(c2,dtype=torch.float64);M=m1+m2;x1=m1/M;x2=m2/M;eta=x1*x2;q=x2/x1;delta=x1-x2
 a=assemble_precession_msa_angles(f,m1,m2,s1,s2,f_ref=20.,order=5);r=a['roots'];s3,sm,sp=r[:,0],r[:,1],r[:,2];S1=x1*x1*s1;S2=x2*x2*s2;Ssum=S1+S2;ss=sum(Ssum.square());xi=float((1+q)*S1[2]+(1+1/q)*S2[2]);L=torch.stack([orbital_angular_momentum(torch.tensor(m1,dtype=torch.float64),torch.tensor(m2,dtype=torch.float64),z,S1[2]/x1**2,S2[2]/x2**2,pn_version=102,compat_102=True) for z in f]);J=torch.sqrt(L.square()+2*a['c1']*L/eta+a['sav']);p=pub(m1,m2,c1,c2);spub=J.square()+L.square()-2*J*L*p;sclean=spin_oscillation(a['psi'],sp,sm,s3)
 # Full analytic total derivative at f_ref: psi_dot includes the PN flux series;
 # root derivatives are centered clean roots, while sn partials use scipy ellipj.
 i=3; v=(math.pi*M*4.925490947641267e-6*f).pow(1/3); m=(sp-sm)/(sp-s3)
 from torchwave.approximants.precession_msa_pn_clean import flux_coefficients,reciprocal_coefficients
 av=__import__('torchwave.approximants.precession_msa_pn_clean',fromlist=['precession_averages']).precession_averages(float(a['c1']),xi,q,eta,sp[i],sm[i],float((S1*S1).sum()),float((S2*S2).sum()),float(v[i]))
 g=reciprocal_coefficients(flux_coefficients(eta,q,av['s1l'],av['s2l'],float((S1*S1).sum()),float((S2*S2).sum()),(float(a['sav'])-float((S1*S1).sum())-float((S2*S2).sum()))/2,av['s1l2'],av['s2l2'],av['s1ls2l']))
 vv=float(v[i]); G=sum(float(g[f'g{k}'])*vv**k for k in range(6)); dpsi_quad=2.25/math.sqrt(eta)*(1-xi*vv)*vv**-3*math.sqrt(float(sp[i]-s3[i]))*G; g0=float(g['g0']); psi1=float(a['psi1']); psi2=float(a['psi2']); dpsi_series=(3*g0*delta/4)*(3*vv**-4+2*psi1*vv**-3+psi2*vv**-2)
 # scipy's sn(u|m) supplies independent partial derivatives
 u=float(a['psi'][i]); mm=float(m[i]); dh=1e-6
 def sn2(uu,mu): return float(ellipj(uu,mu)[0])**2
 Sfun=lambda uu,mu: float(sp[i]+(sm[i]-sp[i])*sn2(uu,mu))
 # Independent scipy partials of sn^2(u|m).
 dSdu=(Sfun(u+dh,mm)-Sfun(u-dh,mm))/(2*dh); dSdm=(Sfun(u,mm+dh)-Sfun(u,mm-dh))/(2*dh)
 # centered root drift and total m/root contribution for each requested h
 slopes=[]
 for h in (.1,.01,.001):
  lo,hi=F.index(20.-h),F.index(20.+h); dr=(r[hi]-r[lo])/(2*h); dm=((m[hi]-m[lo])/(2*h)); droot=float((dr[0]*0 + 0))
  # S3 is included through explicit root derivative; partial derivative formula
  dSroot=float(dr[2] + (dr[1]-dr[2])*sn2(u,mm) + dSdm*float(dm))
  root_df=dSroot; clean_total=dSdu*dpsi_series*vv/(3*20.)+root_df; quad_total=dSdu*dpsi_quad*vv/(3*20.)+root_df; total=clean_total
  pub_slope=float((spub[hi]-spub[lo])/(2*h)); clean_slope=float((sclean[hi]-sclean[lo])/(2*h))
  slopes.append({'h_hz':h,'analytic_series_total_dS2_df':clean_total,'analytic_quadrature_total_dS2_df':quad_total,'clean_centered_dS2_df':clean_slope,'public_inferred_centered_dS2_df':pub_slope,'root_term_df':root_df,'inferred_public_dpsi_df':(pub_slope-root_df)/dSdu,'series_dpsi_df':dpsi_series*vv/(3*20.),'quadrature_dpsi_df':dpsi_quad*vv/(3*20.),'root_derivative_s3_sm_sp':dr.tolist(),'m_derivative':float(dm)})
  branch_minus=spin_oscillation(a['psi']-2*a['psi'][i],sp,sm,s3)
 return {'masses':[m1,m2],'frequencies_hz':F,'public220_cosbeta':p.tolist(),'S2_public220':spub.tolist(),'S2_clean':sclean.tolist(),'roots_s3_sm_sp':r.tolist(),'Lpn':L.tolist(),'Jpn':J.tolist(),'metrics':{'ref_public_clean_S2_abs':float(abs(spub[i]-sclean[i])),'analytic_dS2_df_ref':slopes[1]['analytic_series_total_dS2_df'],'centered_dS2_df_ref':slopes[1]['clean_centered_dS2_df'],'slopes':slopes,'branch_minus_S2':branch_minus.tolist(),'max_centered_root_derivative_per_Hz':max(abs(z) for z in slopes[1]['root_derivative_s3_sm_sp']),'public_jump_19.999_to_20.001':float(abs(p[4]-p[2])),'clean_S2_jump':float(abs(sclean[4]-sclean[2]))},'branch_note':'Opposite branch is reanchored as psi_minus=psi_clean-2*psi_initial, preserving the same forward dpsi/df.'}
if __name__=='__main__':
 out={'equations':['1703.03967v3 (Ssq-sol)','1703.03967v3 (psidot)','1703.03967v3 (m-def)'],'parameters':{'f_ref_hz':20.,'version':220,'expansion_order':5,'mprime':2},'cases':[one(*c) for c in CASES]};out['script_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest();out['clean_sha256']={n:hashlib.sha256(open('src/torchwave/approximants/'+n,'rb').read()).hexdigest() for n in ('precession_msa_initial_clean.py','precession_msa_angles_clean.py','precession_beta_clean.py')};open('docs/reviews/2026-09-13-parity-review/msa-phase-derivative-reference.json','w').write(json.dumps(out,indent=2)+'\n');print(json.dumps([(x['masses'],x['metrics']) for x in out['cases']]))
