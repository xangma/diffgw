"""Process-local public-convention NNLO adapter; production routing is untouched."""
import hashlib,json,torch
import torchwave.approximants.precession_msa as msa
import torchwave.approximants.imrphenom_xp as xpmod
from torchwave.constants import MSUN_SEC
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.precession_nnlo_clean import raw_angles,effective_chip
DT=torch.float64
ADAPTER_CALL_LOG=[]
CASES=[dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=0.,f_ref=40.),dict(mass1=30.,mass2=10.,spin1x=.1,spin1y=-.25,spin1z=-.4,spin2x=.15,spin2y=.05,spin2z=.2,inclination=.5,phic=.2,f_ref=40.),dict(mass1=20.,mass2=20.,spin1x=.2,spin1y=.1,spin1z=.3,spin2x=-.1,spin2y=.2,spin2z=-.1,inclination=.8,phic=-.1,f_ref=40.),dict(mass1=50.,mass2=5.,spin1x=-.2,spin1y=.3,spin1z=.1,spin2x=.1,spin2y=-.1,spin2z=-.3,inclination=.6,phic=.3,f_ref=40.)]

def adapter(v, v0, **p):
    """Substitute only the published NNLO angles, anchored by exact black-box values."""
    incoming_version = p.pop("prec_version", 102)
    incoming_convention = p.pop("convention", 1)
    p.pop("return_carrier_phase", None)
    ADAPTER_CALL_LOG.append({"version": int(incoming_version), "convention": int(incoming_convention), "v0": float(torch.as_tensor(v0).reshape(-1)[0])})
    native = ORIG(v, v0, **p, prec_version=incoming_version,
                  convention=incoming_convention, return_carrier_phase=True)
    v0_scalar = torch.as_tensor(v0, dtype=v.dtype, device=v.device).reshape(-1)[0]
    m_total = p["m1"] + p["m2"]
    eta = p["m1"] * p["m2"] / m_total**2
    delta = (p["m2"] - p["m1"]) / m_total
    m1_norm = p["m1"] / m_total
    chi_l = (p["m1"] * p["s1z"] + p["m2"] * p["s2z"]) / p["m1"]
    chi_p = effective_chip(p["m1"], p["m2"], p["s1x"], p["s1y"],
                           p["s2x"], p["s2y"])
    alpha, epsilon = raw_angles(v, eta, delta, m1_norm, chi_l, chi_p)
    alpha0, epsilon0 = raw_angles(v0_scalar, eta, delta, m1_norm, chi_l, chi_p)
    # A singleton black-box call supplies the exact native anchor at v0;
    # this avoids nearest-grid interpolation and retains the model gauge.
    native0 = ORIG(v0_scalar.reshape(1), v0_scalar, **p,
                   prec_version=incoming_version, convention=incoming_convention,
                   return_carrier_phase=True)
    alpha_anchor = native0[0].reshape(-1)[0]
    gamma_anchor = native0[2].reshape(-1)[0]
    alpha = alpha - alpha0 + alpha_anchor
    gamma = -(epsilon - epsilon0) + gamma_anchor
    return alpha, native[1], gamma, native[3], native[4], native[5]

def run(c):
    f=torch.tensor([25.,30.,40.,60.,80.,120.,160.],dtype=DT); model=IMRPhenomXP(); kw={k:torch.tensor(v,dtype=DT) if isinstance(v,(float,int)) else v for k,v in c.items()};
    b=model(f,**kw,prec_version=102)
    global ORIG; ORIG=msa.evaluate_msa_precession; msa.evaluate_msa_precession=adapter; ORIG_XP=xpmod.evaluate_msa_precession; xpmod.evaluate_msa_precession=adapter
    try: q=model(f,**kw,prec_version=102)
    finally: msa.evaluate_msa_precession=ORIG; xpmod.evaluate_msa_precession=ORIG_XP
    return {'baseline_norm':[float(torch.linalg.vector_norm(x)) for x in b],'substitution_relative_l2':[float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)) for x,y in zip(b,q)],'baseline_substitution_max':[float(torch.max(torch.abs(x-y))) for x,y in zip(b,q)]}

if __name__=='__main__':
 out={'adapter':'published public-contract delta=(m2-m1)/M, chi_l=(m1 chi1z+m2 chi2z)/m1; geometry/beta/anchors retained from native blackbox; no production edits','cases':[run(c) for c in CASES], 'helper_sha256':hashlib.sha256(open('src/torchwave/approximants/precession_nnlo_clean.py','rb').read()).hexdigest()}
 print(json.dumps(out,indent=2)); open('docs/reviews/2026-09-13-parity-review/precession-nnlo-waveform.json','w').write(json.dumps(out,indent=2)+'\n')
