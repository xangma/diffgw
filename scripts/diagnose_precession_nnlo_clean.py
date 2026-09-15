"""Black-box comparison of public Appendix-G NNLO series and native MSA angles."""
import hashlib,json,platform,sys
import torch
import lal, lalsimulation
from torchwave.approximants.precession_msa import evaluate_msa_precession
from torchwave.approximants.precession_nnlo_clean import raw_angles,effective_chip

def public_angles(p, version=102):
    f=lal.CreateREAL8Vector(5); f.data[:]=[20.,25.,40.,80.,160.]
    out=lalsimulation.SimIMRPhenomXPPNAngles(f,float(p["m1"]*lal.MSUN_SI),float(p["m2"]*lal.MSUN_SI),*map(float,(p["s1x"],p["s1y"],p["s1z"],p["s2x"],p["s2y"],p["s2z"])),.7,40,version,lal.CreateDict())
    return [[float(x) for x in y.data] for y in out]

def run(p):
    f=torch.tensor([20.,25.,40.,80.,160.],dtype=torch.float64); M=p['m1']+p['m2']; v=(torch.pi*M*f*4.92549095e-6)**(1/3); v0=v[2]
    native=evaluate_msa_precession(v,v0,**p,iota=torch.tensor(.7,dtype=torch.float64),phi_ref=torch.tensor(0.,dtype=torch.float64),prec_version=102,convention=7)
    eta=p['m1']*p['m2']/M**2; d=(p['m1']-p['m2'])/M; m1=p['m1']/M
    cl=(p['m1']*p['s1z']+p['m2']*p['s2z'])/M
    cp=effective_chip(p['m1'],p['m2'],p['s1x'],p['s1y'],p['s2x'],p['s2y'])
    a,e=raw_angles(v,eta,d,m1,cl,cp)
    # compare frequency evolution only; constants are intentionally absent.
    da=a-a[2]; de=e-e[2]; na=native[0]-native[0][2]; ne=-(native[2]-native[2][2])
    pub=public_angles(p)
    pubs={str(k):public_angles(p,k) for k in (101,102,103,104)}
    return {'frequencies_hz':f.tolist(),'public_alpha':pub[0],'public_second':pub[1],'public_cosbeta':pub[2],'public_versions':pubs,'clean_alpha_delta':da.tolist(),'native_alpha_delta':na.tolist(),'clean_epsilon_delta':de.tolist(),'native_epsilon_delta':ne.tolist(),'alpha_delta_error':(da-na).tolist(),'epsilon_delta_error':(de-ne).tolist(),'eta':float(eta),'delta':float(d),'chi_l':float(cl),'chi_p':float(cp)}

cases=[dict(m1=torch.tensor(40.,dtype=torch.float64),m2=torch.tensor(20.,dtype=torch.float64),s1x=torch.tensor(.3,dtype=torch.float64),s1y=torch.tensor(.2,dtype=torch.float64),s1z=torch.tensor(.4,dtype=torch.float64),s2x=torch.tensor(-.2,dtype=torch.float64),s2y=torch.tensor(.1,dtype=torch.float64),s2z=torch.tensor(-.2,dtype=torch.float64)),dict(m1=torch.tensor(30.,dtype=torch.float64),m2=torch.tensor(10.,dtype=torch.float64),s1x=torch.tensor(.1,dtype=torch.float64),s1y=torch.tensor(-.25,dtype=torch.float64),s1z=torch.tensor(-.4,dtype=torch.float64),s2x=torch.tensor(.15,dtype=torch.float64),s2y=torch.tensor(.05,dtype=torch.float64),s2z=torch.tensor(.2,dtype=torch.float64)),dict(m1=torch.tensor(20.,dtype=torch.float64),m2=torch.tensor(20.,dtype=torch.float64),s1x=torch.tensor(.2,dtype=torch.float64),s1y=torch.tensor(.1,dtype=torch.float64),s1z=torch.tensor(.3,dtype=torch.float64),s2x=torch.tensor(-.1,dtype=torch.float64),s2y=torch.tensor(.2,dtype=torch.float64),s2z=torch.tensor(-.1,dtype=torch.float64)),dict(m1=torch.tensor(50.,dtype=torch.float64),m2=torch.tensor(5.,dtype=torch.float64),s1x=torch.tensor(-.2,dtype=torch.float64),s1y=torch.tensor(.3,dtype=torch.float64),s1z=torch.tensor(.1,dtype=torch.float64),s2x=torch.tensor(.1,dtype=torch.float64),s2y=torch.tensor(-.1,dtype=torch.float64),s2z=torch.tensor(-.3,dtype=torch.float64))]
if __name__=='__main__':
 out={'paper':'Pratten et al. arXiv:2004.06503v2 Appendix G.1 Eqs (120)-(122j); v=(pi M f)^(1/3); native API black-box evaluate_msa_precession prec_version=102 convention=7','python':sys.version,'torch':torch.__version__,'cases':[run(x) for x in cases]}
 out['helper_sha256']=hashlib.sha256(open('src/torchwave/approximants/precession_nnlo_clean.py','rb').read()).hexdigest()
 out['script_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
 open('docs/reviews/2026-09-13-parity-review/precession-nnlo-clean.json','w').write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
