"""Opt-in clean NNLO/convention-0 Eq.45 XP waveform driver."""
import math
import torch
from torchwave.constants import MSUN_SEC
from .imrphenom_xas import IMRPhenomXAS, XASParameterFits
from .precession_nnlo_clean import raw_angles, effective_chip
from .precession_beta_clean import beta
from .precession_source_frame_clean import source_frame_geometry
from .precession_twisting_clean import project_twisted_positive_modes

def _scalar_wave(f, m1, m2, sx1, sy1, sz1, sx2, sy2, sz2, distance, inclination, phic, tc, f_ref, convention):
    # The published single-spin reduction uses the larger body as label 1.
    # Canonicalize a reversed input together with its complete spin vectors.
    reverse = m1 < m2
    cm1, cm2 = torch.where(reverse, m2, m1), torch.where(reverse, m1, m2)
    csx1, csx2 = torch.where(reverse, sx2, sx1), torch.where(reverse, sx1, sx2)
    csy1, csy2 = torch.where(reverse, sy2, sy1), torch.where(reverse, sy1, sy2)
    csz1, csz2 = torch.where(reverse, sz2, sz1), torch.where(reverse, sz1, sz2)
    m1, m2, sx1, sx2, sy1, sy2, sz1, sz2 = cm1, cm2, csx1, csx2, csy1, csy2, csz1, csz2
    flow = torch.where(f > 0, f, torch.ones_like(f)); M=(m1+m2)*MSUN_SEC
    v=(math.pi*M*flow).pow(1/3); v0=(math.pi*M*f_ref).pow(1/3)
    eta=m1*m2/(m1+m2)**2; delta=(m1-m2)/(m1+m2); x1=m1/(m1+m2)
    chip=effective_chip(m1,m2,sx1,sy1,sx2,sy2); chieff=(m1*sz1+m2*sz2)/(m1+m2)
    a,e=raw_angles(v,eta,-delta,x1,chieff/x1,chip); a0,e0=raw_angles(v0,eta,-delta,x1,chieff/x1,chip)
    geom=source_frame_geometry(m1,m2,sx1,sy1,sz1,sx2,sy2,sz2,inclination,phic,f_ref)
    gamma = -e + e0
    zeta = geom[2]
    if convention == 1:
        gamma = gamma + math.pi - geom[3]
        zeta = zeta - math.pi / 2
    angles=(a-a0+geom[0],gamma,beta(m1,m2,flow,sx1,sy1,sz1,sx2,sy2,sz2,compat_102=False),geom[1],zeta)
    af=XASParameterFits.fit_final_spin_nnlo(eta,sz1,sz2,chip)
    carrier_phase = -geom[3] if convention == 0 else torch.zeros_like(geom[3])
    hp0,_=IMRPhenomXAS()(f=f,mass1=m1,mass2=m2,chi1=sz1,chi2=sz2,distance=distance,inclination=torch.zeros_like(inclination),phi_c=carrier_phase,t_c=tc,f_ref=f_ref,_final_spin_override=af)
    hneg=2*hp0/math.sqrt(5/(4*math.pi)); hp,hc=project_twisted_positive_modes({(2,2):hneg},*angles)
    valid=f>0; return torch.where(valid,hp,torch.zeros_like(hp)),torch.where(valid,hc,torch.zeros_like(hc))

def waveform(frequency,mass1,mass2,spin1x=0.,spin1y=0.,spin1z=0.,spin2x=0.,spin2y=0.,spin2z=0.,distance=100.,inclination=0.,phic=0.,tc=0.,f_ref=None,prec_version=102,convention=0,final_spin_mod=0):
    """Return clean ``(h_plus,h_cross)``; parameter axes precede frequency."""
    if int(prec_version) != 102 or int(final_spin_mod) != 0 or int(convention) not in (0, 1):
        raise ValueError("clean NNLO waveform requires prec_version=102, convention in {0,1}, final_spin_mod=0")
    f_in=torch.as_tensor(frequency); f_in=f_in if f_in.is_floating_point() else f_in.to(torch.float64)
    out_dtype=torch.complex64 if f_in.dtype==torch.float32 else torch.complex128; f=f_in.to(torch.float64).reshape(-1)
    if not torch.isfinite(f).all(): raise ValueError("frequency grid must be finite")
    positive=f[f>0]
    if positive.numel()==0: raise ValueError("frequency grid must contain a positive frequency")
    vals=[torch.as_tensor(x,dtype=torch.float64,device=f.device) for x in (mass1,mass2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,distance,inclination,phic,tc)]
    # Arithmetic construction also works when the broadcastable parameter is
    # a scalar (``full_like`` rejects a tensor fill value in that case).
    ref_default = vals[0] * 0.0 + positive.min()
    refs = ref_default if f_ref is None else torch.as_tensor(f_ref, dtype=torch.float64, device=f.device)
    if not torch.isfinite(refs).all() or torch.any(refs < 0):
        raise ValueError("f_ref must be finite and nonnegative")
    vals.append(torch.where(refs > 0, refs, ref_default))
    vals=torch.broadcast_tensors(*vals)
    if not torch.isfinite(torch.stack(vals, dim=0)).all(): raise ValueError("physical parameters must be finite")
    if torch.any(vals[0]<=0) or torch.any(vals[1]<=0): raise ValueError("masses must be positive")
    if torch.any(torch.linalg.vector_norm(torch.stack(vals[2:5], dim=-1), dim=-1) > 1.0) or torch.any(torch.linalg.vector_norm(torch.stack(vals[5:8], dim=-1), dim=-1) > 1.0): raise ValueError("spin-vector norm must not exceed 1")
    if torch.any(vals[8] <= 0): raise ValueError("distance must be positive")
    shape=vals[0].shape; flat=[x.reshape(-1) for x in vals]; hp=[]; hc=[]
    for i in range(flat[0].numel()):
        p,c=_scalar_wave(f,*(x[i] for x in flat),int(convention)); hp.append(p); hc.append(c)
    return torch.stack(hp).reshape(shape+(f.numel(),)).to(out_dtype),torch.stack(hc).reshape(shape+(f.numel(),)).to(out_dtype)
__all__=["waveform"]
