"""Focused follow-up audit; LAL accessed only through its public binary API."""
import json
import platform
from pathlib import Path
import numpy as np
import torch
import lal
import lalsimulation as ls
from torchwave import get_fd_waveform

out = {"versions": {"host": platform.node(), "torch": torch.__version__,
                    "lal": lal.__version__, "lalsimulation": ls.__version__}}

def error(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a))

def oracle_case(name, target, m1=1.4, m2=1.4, s1=0., s2=0., e=0., tides=None,
                tidal_order=12, fref=30., flow=30., fhigh=500., phase=.37, inc=.7):
    pars = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPNSpinOrder(pars, 7)
    ls.SimInspiralWaveformParamsInsertPNTidalOrder(pars, tidal_order)
    opts = dict(spin_order=7, tidal_order=tidal_order)
    if tides:
        ls.SimInspiralWaveformParamsInsertTidalLambda1(pars, tides[0])
        ls.SimInspiralWaveformParamsInsertTidalLambda2(pars, tides[1])
        opts.update(lambda1=tides[0], lambda2=tides[1])
    if e:
        ls.SimInspiralWaveformParamsInsertEccentricityFreq(pars, flow)
        ls.SimInspiralWaveformParamsInsertPNEccentricityOrder(pars, 6)
        opts.update(e0=e, f0_ecc=flow)
        opts.pop("tidal_order")
    a = ls.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0.,0.,s1, 0.,0.,s2,
        100e6*lal.PC_SI, inc, phase, 0., e, 0., .25, flow, fhigh, fref,
        pars, ls.GetApproximantFromString(target))
    f = torch.arange(len(a[0].data.data), dtype=torch.float64) * .25
    b = get_fd_waveform(name, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                        distance=100., inclination=inc, phic=phase,
                        f_ref=fref, sample_frequencies=f, **opts)
    sel = (f >= flow) & (f < fhigh)
    return {label: {"complex_relative_l2": error(torch.tensor(x.data.data)[sel], y[sel]),
                    "dc_zero": bool(y[0] == 0), "finite": bool(torch.isfinite(y).all())}
            for label, x, y in zip(("plus", "cross"), a, b)}

out["explicit_option_oracle"] = {}
for label, opts in [
    ("TF2 point particle", {}),
    ("TF2 6PN tides", dict(tides=(400.,400.), fhigh=1000.)),
    ("TF2 6.5PN tides", dict(tides=(400.,400.), tidal_order=13, fhigh=1000.)),
    ("TF2 unequal spinning", dict(m1=1.7,m2=1.2,s1=.15,s2=-.08,fref=50.)),
    ("Ecc explicit references", dict(e=.05,fref=50.)),
    ("Ecc unequal spinning", dict(e=.03,m1=1.7,m2=1.2,s1=.15,s2=-.08,fref=50.)),
]:
    try:
        names = ("TaylorF2_Eccentric", "TaylorF2Ecc") if "e" in opts else ("TaylorF2", "TaylorF2")
        out["explicit_option_oracle"][label] = oracle_case(*names, **opts)
    except Exception as exc:
        out["explicit_option_oracle"][label] = {"error": str(exc)}

f = torch.arange(0.,500.25,.25,dtype=torch.float64)
cfg = dict(mass1=1.4,mass2=1.4,e0=.05,f_lower=30.,sample_frequencies=f)
a = get_fd_waveform("TaylorF2_Eccentric", **cfg)[0]
b = get_fd_waveform("TaylorF2_Eccentric", **cfg, f0_ecc=.25)[0]
c = get_fd_waveform("TaylorF2_Eccentric", **cfg, f0_ecc=30.)[0]
sel = f >= 30.
out["ecc_default_on_dc_grid"] = {"equals_explicit_f0_0p25": bool(torch.equal(a,b)),
                                 "rel_l2_vs_explicit_f0_30_in_band": error(c[sel],a[sel])}

try:
    get_fd_waveform("TaylorF2_Eccentric", mass1=torch.tensor([1.4,1.7]),
                    mass2=torch.tensor([1.3,1.2]), e0=torch.tensor([.05,.03]),
                    f_ref=torch.tensor([30.,50.]), sample_frequencies=f)
    out["ecc_vector_fref"] = {"error": None}
except Exception as exc:
    out["ecc_vector_fref"] = {"error": str(exc)}

out["spin_alias"] = {}
for name in ["TaylorF2", "IMRPhenomD"]:
    cfg = dict(mass1=30.,mass2=20.,f_ref=30.,sample_frequencies=f)
    a = get_fd_waveform(name, **cfg)[0]
    b = get_fd_waveform(name, **cfg, spin1z=.3)[0]
    c = get_fd_waveform(name, **cfg, chi1=.3)[0]
    out["spin_alias"][name] = {"spin1z_equals_zero_spin": bool(torch.equal(a,b)),
                               "chi1_changes_waveform": not bool(torch.equal(a,c))}

out["ecc_dc_gradient"] = {}
for dtype in [torch.float32, torch.float64]:
    e = torch.tensor(.05,dtype=dtype,requires_grad=True)
    f = torch.tensor([0.,30.,50.,100.],dtype=dtype)
    h = get_fd_waveform("TaylorF2_Eccentric",mass1=1.4,mass2=1.4,e0=e,
                        f0_ecc=30.,f_ref=30.,sample_frequencies=f,dtype=dtype)[0]
    grad, = torch.autograd.grad((h.real*1e23).sum(),e)
    out["ecc_dc_gradient"][str(dtype)] = {"forward_finite": bool(torch.isfinite(h).all()),
                                         "gradient_finite": bool(torch.isfinite(grad))}

from torchwave.approximants.imrphenom_hm import IMRPhenomHM
f = torch.arange(20.,1000.25,.25,dtype=torch.float64)
cfg = dict(f=f,mass1=30.,mass2=10.,chi1=.2,chi2=-.1,f_ref=30.,
           return_unprojected_modes=True)
a = IMRPhenomHM()(**cfg,tc=0.)[2]
b = IMRPhenomHM()(**cfg,tc=.003)[2]
shift = torch.exp(-2j*torch.pi*f*.003)
out["hm_common_time_shift"] = {str(mode): error(a[mode]*shift,b[mode]) for mode in a}

Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
