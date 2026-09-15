"""Clean-room diagnostic of PN ``J`` closures with evolving MSA spin magnitude."""
from __future__ import annotations
import hashlib, json, math, sys
import lal, lalsimulation, torch
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation, initial_invariants, cubic_coefficients, ordered_roots
FREQS=[20.+180.*i/80 for i in range(81)]
CASES=[{"name":"35_25","m1":35.,"m2":25.,"chi1":[.2,.1,.2],"chi2":[-.1,.15,-.1]},
       {"name":"45_15","m1":45.,"m2":15.,"chi1":[.45,-.2,.3],"chi2":[.1,.2,-.25]}]
def public(p, version):
 fv=lal.CreateREAL8Vector(len(FREQS)); fv.data[:]=FREQS; d=lal.CreateDict()
 lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,version); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
 o=lalsimulation.SimIMRPhenomXPMSAAngles(fv,p["m1"]*lal.MSUN_SI,p["m2"]*lal.MSUN_SI,*p["chi1"],*p["chi2"],.7,20.,2,d)
 return [float(x) for x in o[2].data]
def l2pn(eta,v):
 x=v*v; return eta/v*(1+(1.5+eta/6)*x+(27/8-19*eta/8+eta*eta/24)*x*x)
def run(p):
 f=torch.tensor(FREQS,dtype=torch.float64); s1=torch.tensor(p["chi1"],dtype=torch.float64); s2=torch.tensor(p["chi2"],dtype=torch.float64)
 M=p["m1"]+p["m2"]; x1=p["m1"]/M; x2=p["m2"]/M; eta=x1*x2; S1=x1*x1*s1; S2=x2*x2*s2; st=S1+S2; st2=float(torch.dot(st,st))
 a=assemble_precession_msa_angles(f,p["m1"],p["m2"],s1,s2,f_ref=20.,order=5);
 orb=torch.tensor([0.,0.,eta/float((math.pi*M*lal.MTSUN_SI*20.)**(1/3))],dtype=torch.float64)
 qi,ei,di,s1i,s2i,_,ji,xii=initial_invariants(torch.tensor(p["m1"],dtype=torch.float64),torch.tensor(p["m2"],dtype=torch.float64),s1,s2,orb)
 assert abs(float(ei-eta))<1e-14 and abs(float(qi-x2/x1))<1e-14 and abs(float(xii-((1+x2/x1)*S1[2]+(1+x1/x2)*S2[2])))<1e-14
 assert abs(float(ji*ji-(orb[2]**2+2*a["c1"]/float((math.pi*M*lal.MTSUN_SI*20.)**(1/3))+a["sav"])))<1e-12
 v=(math.pi*M*lal.MTSUN_SI*f).pow(1/3); ln=eta/v; jn=torch.sqrt(ln.square()+2*a["c1"]/v+a["sav"])
 s2e=spin_oscillation(a["psi"],a["roots"][...,2],a["roots"][...,1],a["roots"][...,0])
 assert abs(float(s2e[0]-st2)) < 1e-11, (s2e[0],st2); assert float(torch.max(s2e)-torch.min(s2e)) > 1e-10; assert abs(float((jn-ln)[-1]-(jn[0]-ln[0]))) > 1e-10
 pub={str(ver):public(p,ver) for ver in (220,223)}; rows=[]; q=x2/x1; vector_reference=[]
 for label,compat in (("2PN_nonspin",None),("3PN_standard",False),("3PN_compat102",True),("4PN_standard",False),("4PN_compat102",True)):
  vals={"c1Sav":[],"additive":[]}
  for i,ff in enumerate(FREQS):
   vv=float(v[i]); L=l2pn(eta,vv) if label=="2PN_nonspin" else float(orbital_angular_momentum(torch.tensor(p["m1"],dtype=torch.float64),torch.tensor(p["m2"],dtype=torch.float64),torch.tensor(ff,dtype=torch.float64),S1[2]/x1**2,S2[2]/x2**2,pn_version=("full" if label.startswith("4PN") else 102),compat_102=compat))
   if i == 0:
    jv=math.sqrt(float(st[0])**2+float(st[1])**2+(L+float(st[2]))**2)
    vector_reference.append({"L_formula":label,"J_vector":jv,"cosbeta_vector":(jv*jv+L*L-st2)/(2*jv*L),"error_public220":abs((jv*jv+L*L-st2)/(2*jv*L)-pub["220"][0]),"error_public223":abs((jv*jv+L*L-st2)/(2*jv*L)-pub["223"][0])})
   vals["c1Sav"].append(( (L*L+2*float(a["c1"])*L/eta+float(a["sav"])) + L*L-float(s2e[i]) )/(2*math.sqrt(L*L+2*float(a["c1"])*L/eta+float(a["sav"]))*L))
   J=float(jn[i])+L-float(ln[i]); vals["additive"].append((J*J+L*L-float(s2e[i]))/(2*J*L))
  for jk,arr in vals.items(): rows.append({"L_formula":label,"J_formula":jk,"cosbeta":arr,"max_abs_error_public220":max(abs(x-y) for x,y in zip(arr,pub["220"])),"max_abs_error_public223":max(abs(x-y) for x,y in zip(arr,pub["223"]))})
 return {"parameters":p,"public_cos_beta":pub,"reference":{"eta":eta,"v_ref":float(v[0]),"Sav":float(a["sav"]),"c1":float(a["c1"]),"S2_ref":float(s2e[0]),"S2_range":[float(torch.min(s2e)),float(torch.max(s2e))],"JN_ref":float(jn[0]),"JN_200":float(jn[-1]),"vector_reference":vector_reference},"variants":rows}
if __name__=='__main__':
 out={"provenance":"Clean-room PN-beta diagnostic; evolving S² and J_N from assemble_precession_msa_angles, public API only, no fitting/runtime changes.","public_callable":"SimIMRPhenomXPMSAAngles","public_mprime":2,"public_expansion_order":5,"frequencies_hz":FREQS,"cases":[run(p) for p in CASES],"source_sha256":{n:hashlib.sha256(open('src/torchwave/approximants/'+n,'rb').read()).hexdigest() for n in ('precession_beta_clean.py','precession_msa_angles_clean.py')}}
 with open('docs/reviews/2026-09-13-parity-review/msa-beta-jpn-reference.json','w') as h: json.dump(out,h,indent=2); h.write('\n')
 print(json.dumps(out,indent=2))
