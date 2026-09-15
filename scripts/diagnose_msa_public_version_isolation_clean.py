"""Isolate public PhenomX MSA version 220/223/224 angle differences."""
import json
from unittest.mock import patch
import lal, lalsimulation, torch
import torchwave.approximants.precession_msa_angles_clean as a

F=[float(x) for x in torch.linspace(20.,200.,81,dtype=torch.float64)]
CASES=[(35.,25.,(.2,.1,.2),(-.1,.15,-.1)),(45.,15.,(.45,-.2,.3),(.1,.2,-.25))]
def pub(m1,m2,s1,s2,ver):
 v=lal.CreateREAL8Vector(len(F)); v.data[:]=F; d=lal.CreateDict()
 lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,ver); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
 o=lalsimulation.SimIMRPhenomXPMSAAngles(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,*s1,*s2,.7,20.,2,d)
 return [[float(x) for x in z.data] for z in o]
def clean(m1,m2,s1,s2,kind):
 orig=a.oscillatory_correction
 def wrap(psi,J,psidot,aa,c0,c2,c4,d0,d2,d4,at,bt):
  phi,z=orig(psi,J,psidot,aa,c0,c2,c4,d0,d2,d4,at,bt)
  if kind=='noJ': return phi/J,z/J
  if kind=='omit_zeta_Q': return phi,z-J/psidot*(aa+c4/d4)*bt*(-torch.sin(2*psi)/4)
  return phi,z
 with patch.object(a,'oscillatory_correction',wrap):
  o=a.assemble_precession_msa_angles(torch.tensor(F),m1,m2,s1,s2,f_ref=20.,order=5,beta_pn=True,beta_pn_version=102,beta_compat_102=False)
 return o
def anchored_diff(x,y):
 d=[u-v for u,v in zip(x,y)]; return [z-d[0] for z in d]
def case(m1,m2,s1,s2):
 p={v:pub(m1,m2,s1,s2,v) for v in (220,223,224)}
 c={k:clean(m1,m2,s1,s2,k) for k in ('baseline','noJ','omit_zeta_Q')}
 public_isolation={'alpha':anchored_diff(p[220][0],p[224][0]),'gamma':anchored_diff(p[220][1],p[224][1])}
 public_223_224={'alpha':anchored_diff(p[223][0],p[224][0]),'gamma':anchored_diff(p[223][1],p[224][1])}
 smooth={k:{'max_abs_first_difference':max(abs(x-y) for x,y in zip(public_223_224[k][:-1],public_223_224[k][1:])),'max_abs_second_difference':max(abs(public_223_224[k][i+1]-2*public_223_224[k][i]+public_223_224[k][i-1]) for i in range(1,len(public_223_224[k])-1))} for k in ('alpha','gamma')}
 rows=[]
 for k in c:
  oz=[float(x) for x in c[k]['oscillatory_zeta']]; op=[float(x) for x in c[k]['oscillatory_phi']]
  candz=anchored_diff(oz,[0.]*len(oz)); candp=anchored_diff(op,[0.]*len(op))
  rows.append({'recipe':k,'clean_raw_components':{'secular_phi':[float(x) for x in c[k]['secular_phi']],'secular_zeta':[float(x) for x in c[k]['secular_zeta']],'oscillatory_phi':op,'oscillatory_zeta':oz},'relative_to_baseline':{'alpha':anchored_diff([float(x) for x in c[k]['alpha']], [float(x) for x in c['baseline']['alpha']]),'gamma':anchored_diff([float(x) for x in c[k]['gamma']], [float(x) for x in c['baseline']['gamma']])},'candidate_against_public_220_minus_224':{'clean_oscillatory_alpha':max(abs(x-y) for x,y in zip(candp,public_isolation['alpha'])),'clean_oscillatory_gamma_zeta_sign_plus':max(abs(x-y) for x,y in zip(candz,public_isolation['gamma'])),'clean_oscillatory_gamma_zeta_sign_minus':max(abs(x+y) for x,y in zip(candz,public_isolation['gamma']))},'maxabs_vs_public_220_minus_224':{'alpha':max(abs(x-y) for x,y in zip(anchored_diff([float(x) for x in c[k]['alpha']],[float(x) for x in c['baseline']['alpha']]),public_isolation['alpha'])),'gamma':max(abs(x-y) for x,y in zip(anchored_diff([float(x) for x in c[k]['gamma']],[float(x) for x in c['baseline']['gamma']]),public_isolation['gamma']))}})
 return {'parameters':{'mass1_msun':m1,'mass2_msun':m2,'chi1':list(s1),'chi2':list(s2),'f_ref_hz':20.,'inclination':.7,'mprime':2,'expansion_order':5},'public_arrays':{str(v):p[v] for v in (220,223,224)},'public_anchored_220_minus_224':public_isolation,'public_anchored_223_minus_224':public_223_224,'public_223_minus_224_smoothness':smooth,'clean_recipes':rows}
if __name__=='__main__':
 out={'provenance':'Public version 224 is documented as version 220 with the phi_z0/zeta0 prescription from 223 (Pratten et al. primary table, lines 1949-1953). Clean recipes change only the oscillatory wrapper; no fitting or runtime changes.','frequencies_hz':F,'public_callable':'SimIMRPhenomXPMSAAngles','public_versions':[220,223,224],'public_expansion_order':5,'public_mprime':2,'cases':[case(*x) for x in CASES]}
 with open('docs/reviews/2026-09-13-parity-review/msa-public-version-isolation-reference.json','w') as h: json.dump(out,h,indent=2); h.write('\n')
 print(json.dumps(out,indent=2))
