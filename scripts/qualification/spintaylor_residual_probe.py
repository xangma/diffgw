"""Black-box LALSuite vs native XHM/XPHM SpinTaylor waveform diagnostics.

Uses public waveform APIs and binary outputs only; does not inspect LAL code.
Run with the pinned PyCBC/LALSuite environment on len.
"""

import json
import numpy as np
import scipy.optimize
import torch
import lal
import lalsimulation as ls

from torchwave import get_fd_waveform


M1, M2 = 36.0, 29.0
SPINS = (0.3, 0.2, -0.1, 0.1, -0.2, 0.2)
INC, DIST, F_REF = 0.6, 100.0, 37.3
DF, F_MIN, F_MAX = 0.25, 20.0, 300.0


def lal_waveform(precessing, spin_scale=1.0, mode=None, convention=None, f_ref=F_REF):
    p = lal.CreateDict()
    if precessing:
        ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(p, 320)
        ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(p, 2)
        if convention is not None:
            ls.SimInspiralWaveformParamsInsertPhenomXPConvention(p, convention)
    if mode is not None:
        arr = ls.SimInspiralCreateModeArray()
        ls.SimInspiralModeArrayActivateMode(arr, *mode)
        ls.SimInspiralWaveformParamsInsertModeArray(p, arr)
    sx1, sy1, sz1, sx2, sy2, sz2 = SPINS
    hp, hc = ls.SimInspiralChooseFDWaveform(
        M1*lal.MSUN_SI, M2*lal.MSUN_SI,
        sx1*spin_scale if precessing else 0.0,
        sy1*spin_scale if precessing else 0.0,
        sz1,
        sx2*spin_scale if precessing else 0.0,
        sy2*spin_scale if precessing else 0.0,
        sz2,
        DIST*1e6*lal.PC_SI, INC, 0.0, 0.0, 0.0, 0.0,
        DF, F_MIN, F_MAX, f_ref, p,
        ls.IMRPhenomXPHM if precessing else ls.IMRPhenomXHM,
    )
    return np.asarray(hp.data.data), np.asarray(hc.data.data)


def native_waveform(precessing, size, spin_scale=1.0, mode=None, convention=None, f_ref=F_REF):
    freq = torch.arange(size, dtype=torch.float64)*DF
    kwargs = dict(mass1=M1, mass2=M2, distance=DIST,
                  inclination=INC, f_ref=f_ref, sample_frequencies=freq)
    if precessing:
        sx1, sy1, sz1, sx2, sy2, sz2 = SPINS
        kwargs.update(spin1x=sx1*spin_scale, spin1y=sy1*spin_scale,
                      spin1z=sz1, spin2x=sx2*spin_scale,
                      spin2y=sy2*spin_scale, spin2z=sz2)
        if convention is not None:
            kwargs['convention'] = convention
        if mode is not None:
            kwargs['modes'] = (mode,)
        model = 'IMRPhenomXPHM-SpinTaylor'
    else:
        kwargs.update(chi1=SPINS[2], chi2=SPINS[5])
        if mode is not None:
            kwargs['modes'] = (mode,)
        model = 'IMRPhenomXHM'
    hp, hc = get_fd_waveform(model, **kwargs)
    return hp.detach().numpy(), hc.detach().numpy()


def metrics(ref, pred, f, lo=F_MIN, hi=F_MAX):
    mask = (f>=lo)&(f<=hi)
    x, y, ff = ref[mask], pred[mask], f[mask]
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    raw = np.linalg.norm(y-x)/nx
    amp = np.linalg.norm(np.abs(y)-np.abs(x))/np.linalg.norm(np.abs(x))
    correlation = lambda dt: np.vdot(y*np.exp(2j*np.pi*ff*dt), x)
    grid = np.linspace(-0.1, 0.1, 801)
    vals = np.array([abs(correlation(t)) for t in grid])
    k = int(vals.argmax())
    half = grid[1]-grid[0]
    best = scipy.optimize.minimize_scalar(lambda t: -abs(correlation(t)),
                                          bounds=(grid[k]-half,grid[k]+half),
                                          method='bounded',options={'xatol':1e-12})
    c = abs(correlation(best.x))
    best_error = np.sqrt(max(0.0, 1.0-c*c/(nx*nx*ny*ny)))
    return {'raw_l2':float(raw),'amplitude_l2':float(amp),
            'time_phase_scale_l2':float(best_error),'best_dt_s':float(best.x),
            'norm_ratio':float(ny/nx)}


def main():
    import sys
    if '--alpha-shift' in sys.argv:
        import torchwave.approximants.precession_spintaylor as st
        original = st.evaluate_spintaylor_precession
        shift = 1.5743964450330335
        def shifted(*args, **kwargs):
            a, b, g, theta, zeta, carrier = original(*args, **kwargs)
            return a-shift, b, g, theta, zeta, carrier
        for mode in [None,(2,2)]:
            ref = lal_waveform(True, mode=mode)
            st.evaluate_spintaylor_precession = shifted
            try:
                pred = native_waveform(True,len(ref[0]),mode=mode)
            finally:
                st.evaluate_spintaylor_precession = original
            f = np.arange(len(ref[0]))*DF
            result = {pol:metrics(x,y,f) for pol,x,y in zip(('plus','cross'),ref,pred)}
            print(json.dumps({'alpha_shift':shift,'mode':mode,'result':result}),flush=True)
        return
    if '--mode-sum' in sys.argv:
        modes = [(2,1),(2,2),(3,2),(3,3),(4,4)]
        for precessing in [False, True]:
            f_ref = 50.0 if precessing else F_REF
            ref_all = lal_waveform(precessing,f_ref=f_ref)
            pred_all = native_waveform(precessing,len(ref_all[0]),f_ref=f_ref)
            ref_modes = [lal_waveform(precessing,mode=mode,f_ref=f_ref) for mode in modes]
            pred_modes = [native_waveform(precessing,len(ref_all[0]),mode=mode,f_ref=f_ref) for mode in modes]
            f = np.arange(len(ref_all[0]))*DF
            valid = (f>=F_MIN)&(f<=F_MAX)
            result = {}
            for k,pol in enumerate(('plus','cross')):
                rsum = sum(x[k][valid] for x in ref_modes)
                psum = sum(x[k][valid] for x in pred_modes)
                rall = ref_all[k][valid]
                pall = pred_all[k][valid]
                result[pol] = {
                    'lal_mode_sum_l2':float(np.linalg.norm(rsum-rall)/np.linalg.norm(rall)),
                    'native_mode_sum_l2':float(np.linalg.norm(psum-pall)/np.linalg.norm(pall)),
                    'relative_mode_norms_lal':[float(np.linalg.norm(x[k][valid])/np.linalg.norm(rall)) for x in ref_modes],
                    'relative_mode_norms_native':[float(np.linalg.norm(x[k][valid])/np.linalg.norm(pall)) for x in pred_modes],
                }
            print(json.dumps({'precessing':precessing,'f_ref':f_ref,'modes':modes,'result':result}),flush=True)
        return
    if '--modes' in sys.argv:
        for mode in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]:
            ref = lal_waveform(False, mode=mode)
            pred = native_waveform(False, len(ref[0]), mode=mode)
            f = np.arange(len(ref[0]))*DF
            result = {pol:metrics(x,y,f) for pol,x,y in zip(('plus','cross'),ref,pred)}
            print(json.dumps({str(mode):result}),flush=True)
        ref = lal_waveform(True,mode=(2,2))
        pred = native_waveform(True,len(ref[0]),mode=(2,2))
        f = np.arange(len(ref[0]))*DF
        result = {f'{lo}-{hi}':{
            pol:metrics(x,y,f,lo,hi) for pol,x,y in zip(('plus','cross'),ref,pred)
        } for lo,hi in [(20,60),(60,150),(150,300)]}
        print(json.dumps({'xphm_22_bands':result}),flush=True)
        return
    if '--isolation' in sys.argv:
        for label, precessing, scale, mode, convention, f_ref in [
            ('xhm_22', False, 0.0, (2, 2), None, F_REF),
            ('xphm_22_c1', True, 1.0, (2, 2), 1, F_REF),
            ('xphm_22_c0', True, 1.0, (2, 2), 0, F_REF),
            ('xphm_all_c0', True, 1.0, None, 0, F_REF),
            ('xhm_all_fref20', False, 0.0, None, None, 20.0),
        ]:
            ref = lal_waveform(precessing, scale, mode, convention, f_ref)
            pred = native_waveform(precessing, len(ref[0]), scale, mode, convention, f_ref)
            f = np.arange(len(ref[0]))*DF
            result = {pol:metrics(x,y,f) for pol,x,y in zip(('plus','cross'),ref,pred)}
            print(json.dumps({label:result}),flush=True)
        return
    rows = {}
    for label, precessing, spin_scale in [
        ('xhm_aligned',False,0.0),
        ('xphm_aligned',True,0.0),
        ('xphm_weak',True,0.01),
        ('xphm_half',True,0.5),
        ('xphm_full',True,1.0),
    ]:
        ref = lal_waveform(precessing,spin_scale)
        pred = native_waveform(precessing,len(ref[0]),spin_scale)
        f = np.arange(len(ref[0]))*DF
        rows[label] = {pol:metrics(x,y,f) for pol,x,y in zip(('plus','cross'),ref,pred)}
        if label == 'xphm_full':
            rows[label]['bands'] = {f'{lo}-{hi}':{
                pol:metrics(x,y,f,lo,hi) for pol,x,y in zip(('plus','cross'),ref,pred)
            } for lo,hi in [(20,60),(60,150),(150,300)]}
        print(json.dumps({label:rows[label]}),flush=True)
    print(json.dumps({'versions':{'lal':lal.__version__,'lalsimulation':ls.__version__,
                                  'torch':torch.__version__}}),flush=True)

if __name__ == '__main__':
    main()
