"""Clean-room double-spin diagnostic against public PhenomX MSA angles."""
from __future__ import annotations
import json, math, sys
import lal, lalsimulation, torch
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum
from torchwave.approximants.precession_msa_initial_clean import cubic_coefficients, ordered_roots
from torchwave.approximants.precession_msa_pn_clean import precession_averages

FREQS=[20.,30.,50.,100.,200.]
CASES=[{"name":"35_25","m1":35.,"m2":25.,"chi1":[.2,.1,.3],"chi2":[.05,-.1,-.2]},
       {"name":"45_15","m1":45.,"m2":15.,"chi1":[.4,-.2,.5],"chi2":[-.1,.2,-.3]}]

def public(p, version):
    fv=lal.CreateREAL8Vector(len(FREQS)); fv.data[:]=FREQS
    d=lal.CreateDict(); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,version)
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d,5)
    out=lalsimulation.SimIMRPhenomXPMSAAngles(fv,p["m1"]*lal.MSUN_SI,p["m2"]*lal.MSUN_SI,*p["chi1"],*p["chi2"],.7,20.,2,d)
    return [float(x) for x in out[2].data]

def run(p):
    m1,m2=p["m1"],p["m2"]; M=m1+m2; x1,x2=m1/M,m2/M; eta=x1*x2; q=x2/x1; delta=x1-x2
    s1=torch.tensor(p["chi1"],dtype=torch.float64)*x1**2; s2=torch.tensor(p["chi2"],dtype=torch.float64)*x2**2
    v0=(math.pi*M*20*lal.MTSUN_SI)**(1/3); LN=eta/v0; JN=torch.linalg.vector_norm(torch.tensor([s1[0]+s2[0],s1[1]+s2[1],LN+s1[2]+s2[2]]))
    xi=(1+q)*s1[2]+(1+1/q)*s2[2]
    B,C,D=cubic_coefficients(q,eta,delta,(s1*s1).sum(),(s2*s2).sum(),LN,JN,xi); roots=ordered_roots(B,C,D); s3,sm,sp=roots
    sav=(sp+sm)/2; c1=.5*v0*(JN**2-LN**2-sav)
    av=precession_averages(c1,xi,q,eta,sp,sm,(s1*s1).sum(),(s2*s2).sum(),v0)
    pub={str(v):public(p,v) for v in (220,223)}; rows=[]
    s1sq,s2sq=(s1*s1).sum(),(s2*s2).sum(); stot=torch.linalg.vector_norm(s1+s2)
    for label in ("2PN_nonspin","3PN_spin","full4PN"):
      vals=[]
      for f in [20.]:
        v=(math.pi*M*f*lal.MTSUN_SI)**(1/3); x=v*v
        for proj,ss1,ss2 in (("initial",s1[2]/x1**2,s2[2]/x2**2),("average",av["s1l"]/x1**2,av["s2l"]/x2**2)):
          L=eta/v*(1+(eta/6+1.5)*x+(eta**2/24-19*eta/8+27/8)*x*x) if label=="2PN_nonspin" else orbital_angular_momentum(m1,m2,f,float(ss1),float(ss2),pn_version=("full" if label=="full4PN" else 102),compat_102=False)
          jvec=torch.linalg.vector_norm(torch.stack((s1[0]+s2[0],s1[1]+s2[1],L+s1[2]+s2[2])))
          jpn=torch.sqrt(L*L+2*c1*L/eta+sav); stot=torch.linalg.vector_norm(s1+s2)
          Bpn,Cpn,Dpn=cubic_coefficients(q,eta,delta,s1sq,s2sq,L,jvec,xi); rootspn=ordered_roots(Bpn,Cpn,Dpn)
          savpn=(rootspn[1]+rootspn[2])/2; c1pn=eta*(jvec**2-L**2-savpn)/(2*L); Jrec=torch.sqrt(L**2+2*c1pn*L/eta+savpn)
          beta_vec=(jvec*jvec+L*L-stot*stot)/(2*jvec*L); beta_rec=(Jrec**2+L*L-stot*stot)/(2*Jrec*L)
          assert abs(float(beta_rec-beta_vec)) < 1e-12
          rows.append({"formula":label,"spin_projection":proj,"L_reference":float(L),"cosbeta_vectorJ":float(beta_vec),"cosbeta_c1SavJ":float((jpn*jpn+L*L-stot*stot)/(2*jpn*L)),"PNroots":rootspn.tolist(),"savpn":float(savpn),"c1pn":float(c1pn),"Jrec":float(Jrec),"beta_rec":float(beta_rec),"absolute_error_vectorJ_public_220":abs(float(beta_vec)-pub["220"][0]),"absolute_error_vectorJ_public_223":abs(float(beta_vec)-pub["223"][0]),"absolute_error_c1SavJ_public_220":abs(float((jpn*jpn+L*L-stot*stot)/(2*jpn*L))-pub["220"][0]),"absolute_error_c1SavJ_public_223":abs(float((jpn*jpn+L*L-stot*stot)/(2*jpn*L))-pub["223"][0])})
    return {"parameters":p,"public_cos_beta":pub,"reference_geometry":{"q":q,"eta":eta,"delta":delta,"v_ref":v0,"L_N":float(LN),"J_N":float(JN),"xi":float(xi),"c1":float(c1),"roots_s3_sm_sp":[float(s3),float(sm),float(sp)],"Sav":float(sav)},"variants":rows}

if __name__=='__main__':
 out={"provenance":"Fresh double-spin Newtonian geometry, clean cubic roots and Eq.17/26 L variants compared against public SimIMRPhenomXPMSAAngles expansionorder=5 at prec_version 220/223; mprime=2; no fitting.","public_callable":"SimIMRPhenomXPMSAAngles","public_mprime":2,"public_expansion_order":5,"python":sys.version,"frequencies_hz":FREQS,"cases":[run(p) for p in CASES]}
 with open("docs/reviews/2026-09-13-parity-review/msa-beta-reference.json","w") as f: json.dump(out,f,indent=2); f.write("\n")
 print(json.dumps(out,indent=2))
