"""Diagnostic comparison of leading and elliptic Sav choices in clean MSA."""
import json, sys
import lal, lalsimulation, torch
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles, elliptic_spin_average

F=[20.,30.,50.,100.,200.]
CASES=[(35.,25.,(.2,.1,.2),(-.1,.15,-.1)),(45.,15.,(.45,-.2,.3),(.1,.2,-.25))]
def average_self_check():
 # Independent dense uniform phase quadrature for x=sp-(sp-sm)sn².
 import scipy.special as ss
 checks=[]
 for m in (.2,.5,.8):
  p,n=1.0,.25; s3=p-(p-n)/m; K=ss.ellipk(m)
  u=torch.arange(200000,dtype=torch.float64)*4*K/200000
  sn=ss.ellipj(u.numpy(),m)[0]
  dense=float((p-(p-n)*torch.tensor(sn*sn)).mean())
  exact=float(p-(p-n)*(K-ss.ellipe(m))/(m*K))
  clean=float(elliptic_spin_average(torch.tensor(p),torch.tensor(n),torch.tensor(s3)))
  checks.append({"m":m,"dense_uniform_phase":dense,"elliptic_formula":exact,"clean_helper":clean,"dense_formula_abs_error":abs(dense-exact),"helper_formula_abs_error":abs(clean-exact)})
 return {"periodic_quadrature_checks":checks,"m0_exact":float(elliptic_spin_average(torch.tensor(.625),torch.tensor(.625),torch.tensor(.25))),"m0_limit_factor":.5,"m1e-9":float(elliptic_spin_average(torch.tensor(1.),torch.tensor(.25),torch.tensor(1.-.75/1e-9)))}
def public(m1,m2,s1,s2,ver=223):
 v=lal.CreateREAL8Vector(len(F)); v.data[:]=F; d=lal.CreateDict(); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,ver); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
 o=lalsimulation.SimIMRPhenomXPMSAAngles(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,*s1,*s2,.7,20.,2,d)
 return [[float(x) for x in z.data] for z in o]
def run(m1,m2,s1,s2):
 ref=public(m1,m2,s1,s2); rows=[]
 for mode in ("leading","exact"):
  out=assemble_precession_msa_angles(torch.tensor(F),m1,m2,s1,s2,f_ref=20.,order=5,average_mode=mode,beta_pn=True)
  ae=[float(x-y) for x,y in zip(out["alpha"],ref[0])]; ge=[float(x-y) for x,y in zip(out["gamma"],ref[1])]
  aa=[x-ae[0] for x in ae]; gg=[x-ge[0] for x in ge]
  rows.append({"Sav_mode":mode,"Sav":float(out["sav"]),"c1":float(out["c1"]),"alpha_raw_abs_error":list(map(abs,ae)),"gamma_raw_abs_error":list(map(abs,ge)),"alpha_anchored_error":aa,"gamma_anchored_error":gg,"alpha_anchored_maxabs":max(map(abs,aa)),"gamma_anchored_maxabs":max(map(abs,gg)),"PNbeta_abs_error":[abs(float(x-y)) for x,y in zip(out["cos_beta_pn"],ref[2])]})
 return {"parameters":{"mass1_msun":m1,"mass2_msun":m2,"chi1":s1,"chi2":s2,"f_ref_hz":20.,"iota":.7},"public_arrays":{"alpha":ref[0],"gamma":ref[1],"cos_beta":ref[2]},"variants":rows}
if __name__=="__main__":
 out={"provenance":"Clean-room MSA diagnostic: only Sav averaging selector changes (leading (sp+sm)/2 versus elliptic cycle average); PN flux, phase, and cubic approximations remain as implemented.","average_self_check":average_self_check(),"public_callable":"SimIMRPhenomXPMSAAngles","public_prec_version":223,"public_expansion_order":5,"public_mprime":2,"frequencies_hz":F,"python":sys.version,"cases":[run(*x) for x in CASES]}
 with open("docs/reviews/2026-09-13-parity-review/msa-exact-average-reference.json","w") as f: json.dump(out,f,indent=2); f.write("\n")
 print(json.dumps(out,indent=2))
