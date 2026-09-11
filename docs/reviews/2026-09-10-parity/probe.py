"""Read-only waveform audit. Uses LAL only through its public black-box API."""
import json
import math
import os
import platform
from pathlib import Path
import sys
import numpy as np
import torch
import lal
import lalsimulation as lalsim
import pycbc
from pycbc.filter import match, optimized_match
from pycbc.types import FrequencySeries
from torchwave import get_fd_waveform

OUT = Path(__file__).resolve().parent
rows = []

def reference(name, m1=1.4, m2=1.4, c1=0., c2=0., s1x=0., s1y=0.,
              s2x=0., s2y=0., inc=0., distance=100., flow=30., fhigh=500.,
              df=.25, l1=None, l2=None, eccentricity=0., dict_ecc=None, phic=0., **ignored):
    params = lal.CreateDict()
    if l1 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, l1)
    if l2 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, l2)
    if dict_ecc is not None:
        lalsim.SimInspiralWaveformParamsInsertEccentricity(params, dict_ecc)
    approx = lalsim.GetApproximantFromString(name)
    assert lalsim.GetStringFromApproximant(approx) == name
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x,s1y,c1,s2x,s2y,c2,
        distance*1e6*lal.PC_SI,inc,phic,0.,eccentricity,0.,
        df,flow,fhigh,flow,params,approx)
    return np.array(hp.data.data), np.array(hc.data.data)

def tensor_wave(name, f, m1=1.4,m2=1.4,c1=0.,c2=0.,s1x=0.,s1y=0.,s2x=0.,
                s2y=0.,inc=0.,distance=100.,l1=None,l2=None,tw_extra=None,phic=0.,**ignored):
    return tuple(x.detach().cpu().numpy() for x in get_fd_waveform(
        name,mass1=m1,mass2=m2,chi1=c1,chi2=c2,spin1x=s1x,spin1y=s1y,
        spin2x=s2x,spin2y=s2y,distance=distance,inclination=inc,lambda1=l1,lambda2=l2,
        phic=phic,sample_frequencies=torch.tensor(f,dtype=torch.float64),
        **(tw_extra or {})))

def metrics(ref, tw, df, flow, fhigh):
    f = np.arange(len(ref))*df
    sel = (f >= flow) & (f < fhigh)
    a,b = ref[sel],tw[sel]
    norm = np.linalg.norm(a)
    if norm == 0:
        return {'zero_reference':True,'torch_norm':float(np.linalg.norm(b))}
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite waveform inside comparison band'
    overlap, shift = match(FrequencySeries(ref,delta_f=df),FrequencySeries(tw,delta_f=df),
                           low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
    # Zero only excluded bins; the original arrays remain available for finite checks.
    # In-band finiteness was asserted above. This avoids propagating DC NaNs
    # through the sub-sample time-shift operation.
    opt_overlap, opt_shift = optimized_match(
        FrequencySeries(np.where(sel,ref,0.),delta_f=df),
        FrequencySeries(np.where(sel,tw,0.),delta_f=df),
        low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
    return dict(match=float(overlap),sample_shift=int(shift),
                optimized_match=float(opt_overlap),optimized_sample_shift=float(opt_shift),
                reference_nonfinite_bins=np.flatnonzero(~np.isfinite(ref)).tolist(),
                torch_nonfinite_bins=np.flatnonzero(~np.isfinite(tw)).tolist(),
                norm_ratio=float(np.linalg.norm(b)/norm),
                magnitude_relative_l2=float(np.linalg.norm(np.abs(b)-np.abs(a))/norm),
                raw_relative_l2=float(np.linalg.norm(b-a)/norm))

def case(label, tw_name, lal_name=None, **cfg):
    opts={**dict(flow=30.,fhigh=500.,df=.25),**cfg}
    ref=reference(lal_name or tw_name,**opts)
    f=np.arange(len(ref[0]))*opts['df']
    tw=tensor_wave(tw_name,f,**opts)
    row=dict(label=label,torch_model=tw_name,lal_model=lal_name or tw_name,parameters=opts,
             plus=metrics(ref[0],tw[0],opts['df'],opts['flow'],opts['fhigh']),
             cross=metrics(ref[1],tw[1],opts['df'],opts['flow'],opts['fhigh']))
    rows.append(row)
    print(json.dumps(row),flush=True)

def run_case(label, tw_name, lal_name=None, **cfg):
    # Defaults are merged here so callers can override frequency bounds.
    try:
        case(label,tw_name,lal_name,**cfg)
    except Exception as exc:
        row={'label':label,'error':repr(exc)}
        rows.append(row)
        print(json.dumps(row),flush=True)

print(json.dumps(dict(host=platform.node(),cwd=os.getcwd(),pid=os.getpid(),
                     python=sys.executable,versions=dict(torch=torch.__version__,
                     numpy=np.__version__,lal=lal.__version__,lalsimulation=lalsim.__version__,
                     pycbc=pycbc.__version__))),flush=True)

run_case('TaylorF2 circular','TaylorF2')
run_case('TaylorF2 tidal','TaylorF2',l1=400.,l2=400.,fhigh=1000.)
for m1,m2,c1,c2 in [(10.,10.,0.,0.),(30.,30.,0.,0.),(30.,30.,.8,.8),
                   (36.,29.,.3,-.2),(30.,10.,.5,-.3),(20.,10.,.4,0.),
                   (50.,10.,.3,.2),(40.,10.,.6,-.4),(45.,5.,.5,0.),(80.,10.,.7,.1)]:
    run_case(f'D {m1}/{m2} {c1}/{c2}','IMRPhenomD',m1=m1,m2=m2,c1=c1,c2=c2,
             distance=410.,flow=20.,fhigh=1024.)
run_case('NRTidal v1','IMRPhenomD_NRTidal',l1=400.,l2=400.,fhigh=1000.)
run_case('NRTidal v2','IMRPhenomD_NRTidalv2',m1=1.4,m2=1.3,c1=.02,c2=-.01,l1=350.,l2=420.,
         distance=40.,flow=25.,fhigh=1200.)
run_case('NSBH','IMRPhenomNSBH',m1=6.,m2=1.4,c1=.2,l2=450.,fhigh=1024.)
run_case('Eccentric original call','TaylorF2_Eccentric','EccentricFD',dict_ecc=.05,
         tw_extra={'eccentricity':.05})
for lal_name in ['EccentricFD','TaylorF2Ecc']:
    run_case(f'Eccentric correct args vs {lal_name}','TaylorF2_Eccentric',lal_name,
             eccentricity=.05,tw_extra={'e0':.05,'f0_ecc':30.})

# Explicit controls: the original eccentric call must actually perturb both models.
f=np.arange(30.,500.25,.25)
tw0=tensor_wave('TaylorF2_Eccentric',f,tw_extra={'f0_ecc':30.})
twbad=tensor_wave('TaylorF2_Eccentric',f,tw_extra={'f0_ecc':30.,'eccentricity':.05})
twgood=tensor_wave('TaylorF2_Eccentric',f,tw_extra={'f0_ecc':30.,'e0':.05})
assert all(np.isfinite(x).all() for pair in (tw0,twbad,twgood) for x in pair)
lal0=reference('EccentricFD')
lalbad=reference('EccentricFD',dict_ecc=.05)
controls=dict(ignored_torch_eccentricity=np.array_equal(tw0[0],twbad[0]),
              ignored_lal_dict_eccentricity=np.array_equal(lal0[0],lalbad[0]),
              torch_e0_changes_waveform=not np.array_equal(tw0[0],twgood[0]))
common=dict(m1=35.,m2=25.,c1=.2,c2=-.1,inc=.7)
f=np.arange(25.,500.25,.25)
d=tensor_wave('IMRPhenomD',f,tw_extra={'window_type':'planck'},**common)
xas=tensor_wave('IMRPhenomXAS',f,**common)
hm=tensor_wave('IMRPhenomHM',f,**common)
xhm=tensor_wave('IMRPhenomXHM',f,**common)
controls.update(xas_equals_d_with_same_window=all(np.array_equal(a,b) for a,b in zip(d,xas)),
                hm_equals_xhm=all(np.array_equal(a,b) for a,b in zip(hm,xhm)))
for prefix,pair in [('torch',tensor_wave('IMRPhenomD',f,m1=30.,m2=10.,c1=.5,c2=-.3)),
                    ('lal',reference('IMRPhenomD',m1=30.,m2=10.,c1=.5,c2=-.3))]:
    hp,hc=pair
    controls[prefix+'_faceon_cross_residual_vs_minus_i_plus']=float(np.linalg.norm(hc+1j*hp)/np.linalg.norm(hp))
    controls[prefix+'_faceon_cross_residual_vs_minus_i_conjugate_plus']=float(np.linalg.norm(hc+1j*np.conj(hp))/np.linalg.norm(hp))
print(json.dumps({'controls':controls}),flush=True)
metadata=dict(host=platform.node(),python=sys.executable,torch=torch.__version__,
              numpy=np.__version__,lal=lal.__version__,lalsimulation=lalsim.__version__,
              pycbc=pycbc.__version__,precision='float64/complex128 CPU',psd='None (flat)',
              band='flow <= f < fhigh',phase_time='PyCBC match and optimized_match independently maximize time and phase for each polarization; raw error does not align',
              defaults=dict(mass1=1.4,mass2=1.4,chi1=0.,chi2=0.,distance_Mpc=100.,inclination=0.,phic=0.,
                            torch_f_ref=None,lal_f_ref='flow'),
              caveat='Smoke-test replication; raw residuals include uncorrected convention differences. Not a complete parity harness.')
(OUT/'results.json').write_text(json.dumps(dict(metadata=metadata,controls=controls,results=rows),indent=2)+'\n')
