"""Clean Kerr (2,2) ringdown TD/FD pair, diagnostic-only and opt in.

Equations are the analytic Fourier transform of a damped complex mode and its
mirror. The default angular shape uses the separately documented inferred
public-compatible convention; ``source_spectral`` selects the angular
Teukolsky eigensolution. No public implementation or calibration values are used.
"""
import math
import torch
from torchwave.constants import MSUN_SEC
from .kerr_qnm_generated import signed_qnm_220
from .kerr_ringdown_energy import ringdown_amplitude_from_energy_fraction
from .kerr_angular_table_convention import spin_weighted_spheroidal_harmonic_table_compat


def _angular_pair(a, wr, damp, inc, pc, angular_convention):
    if angular_convention == "public_compat":
        return (spin_weighted_spheroidal_harmonic_table_compat(a, inc, pc),
                spin_weighted_spheroidal_harmonic_table_compat(a, math.pi-inc, pc))
    if angular_convention != "source_spectral":
        raise ValueError("angular_convention must be 'public_compat' or 'source_spectral'")
    from .kerr_angular import spin_weighted_spheroidal_harmonic as spectral
    c = a * (wr - 1j*damp)
    equator = spectral(2, 2, c, torch.full_like(inc, math.pi/2), 0., ell_max=18)
    gauge = equator.conj() / equator.abs()
    return (spectral(2, 2, c, inc, pc, ell_max=18) * gauge,
            spectral(2, 2, c, math.pi-inc, pc, ell_max=18) * gauge)


def _params(final_mass, final_spin, distance, energy_fraction, inclination, phic, tc, grid=None):
    vals=[final_mass,final_spin,distance,energy_fraction,inclination,phic,tc]
    ts=[v for v in vals+[grid] if isinstance(v,torch.Tensor)]
    device=ts[0].device if ts else torch.device('cpu')
    dtype=None
    for v in ts:
        if v.device!=device: raise ValueError('tensor inputs must share device')
        if v.is_floating_point(): dtype=v.dtype if dtype is None else torch.promote_types(dtype,v.dtype)
    dtype=dtype or torch.float64
    out=[torch.as_tensor(v,dtype=dtype,device=device) for v in vals]
    if not torch.compiler.is_compiling():
        if not all(torch.isfinite(v).all() for v in out):
            raise ValueError('physical parameters must be finite')
    params=list(torch.broadcast_tensors(*[torch.as_tensor(v,dtype=dtype,device=device) for v in out]))
    return params

def _grid(g, params):
    g = torch.as_tensor(g, dtype=params[0].dtype, device=params[0].device)
    if not torch.compiler.is_compiling():
        if not torch.isfinite(g).all():
            raise ValueError('time/frequency grid must be finite')
    params=list(torch.broadcast_tensors(*params))
    if g.ndim:
        grid_ndim=g.ndim
        batch=params[0].shape
        g=g.reshape((1,)*len(batch)+tuple(g.shape))
        params=[v.reshape(v.shape+(1,)*grid_ndim) for v in params]
    return g, params


def kerr220_td(t, final_mass, final_spin, distance, energy_fraction=.01,
               inclination=0., phic=0., tc=0., *, angular_convention="public_compat"):
    """Return ``(hp,hc)`` for u=t-tc>=0; zero before onset."""
    M,a,D,e,inc,pc,t0=_params(final_mass,final_spin,distance,energy_fraction,inclination,phic,tc,t)
    t, (M,a,D,e,inc,pc,t0)=_grid(t,[M,a,D,e,inc,pc,t0])
    wr, damp=signed_qnm_220(a); sec=M*MSUN_SEC; omega=wr/sec; lam=damp/sec
    A=ringdown_amplitude_from_energy_fraction(M,D,e,omega,1/lam,convention='single_mode')
    Sp,Sm=_angular_pair(a,wr,damp,inc,pc,angular_convention)
    u=t-t0; up=torch.clamp(u,min=0); z=torch.exp(-lam*up-1j*omega*up); mask=u>=0
    hp=-A/math.sqrt(2)*torch.real(z*(Sp+Sm))*mask
    hc=A/math.sqrt(2)*torch.imag(z*(Sp-Sm))*mask
    return hp,hc


def kerr220_fd(f, final_mass, final_spin, distance, energy_fraction=.01,
               inclination=0., phic=0., tc=0., *, angular_convention="public_compat"):
    """Return Fourier-minus-frequency ``(hp,hc)`` for frequency f in Hz."""
    M,a,D,e,inc,pc,t0=_params(final_mass,final_spin,distance,energy_fraction,inclination,phic,tc,f)
    f, (M,a,D,e,inc,pc,t0)=_grid(f,[M,a,D,e,inc,pc,t0])
    wr,damp=signed_qnm_220(a); sec=M*MSUN_SEC; omega=wr/sec; lam=damp/sec
    A=ringdown_amplitude_from_energy_fraction(M,D,e,omega,1/lam,convention='single_mode')
    Sp,Sm=_angular_pair(a,wr,damp,inc,pc,angular_convention)
    P=Sp+Sm; C=Sp-Sm
    zp=1/(lam+1j*(2*math.pi*f+omega)); zm=1/(lam+1j*(2*math.pi*f-omega))
    phase=torch.exp(-2j*math.pi*f*t0)
    hp=-A/(2*math.sqrt(2))*(P*zp+torch.conj(P)*zm)*phase
    hc=A/(2j*math.sqrt(2))*(C*zp-torch.conj(C)*zm)*phase
    return hp,hc
