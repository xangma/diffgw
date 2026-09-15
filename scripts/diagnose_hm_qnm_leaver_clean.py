"""HM London assembly: printed QNM fits versus independent Kerr-Leaver roots."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch, lal, lalsimulation
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.hm_assembly_clean import (
    assemble_mode, project_positive_modes, MODES,
    COMPATIBILITY_AMPLITUDE_JOIN, COMPATIBILITY_PHASE_JOIN,
)
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm
from kerr_leaver_solver import solve

CASES=((40.,15.,.3,-.2,500.,.8,.2,20.),(55.,11.,-.4,.2,680.,1.1,.71,27.),
       (32.,18.,-.6,.45,430.,.8,0.,25.),(70.,8.,.1,-.5,900.,.9,.4,30.))
FREQ=np.unique(np.r_[np.linspace(20.,512.,81),[20.,50.,100.,200.]])
MTSUN=lal.MTSUN_SI

BERTI_FIT_PATH = Path("/tmp/torchwave-berti-qnm/fitcoeffsWEB.dat")

def load_berti_fits(path=BERTI_FIT_PATH):
    """Read the author-hosted fitcoeffsWEB.dat n=0 rows (no refitting)."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        fields = line.replace("d", "e").split()
        if len(fields) != 9:
            continue
        ell, m, n = map(int, fields[:3])
        if n == 0:
            rows[(ell, m)] = tuple(float(x) for x in fields[3:])
    return rows

def qnm_berti_published(mode, af, mt, mf, coeffs):
    f1, f2, f3, q1, q2, q3 = coeffs[mode]
    one_minus_a = 1.0 - float(af)
    momega = f1 + f2 * one_minus_a**f3
    quality = q1 + q2 * one_minus_a**q3
    # Author fit is M_BH omega; Eq. (17) uses the final BH mass.
    scale = float(mt) / float(mf)
    return momega / (2 * math.pi) * scale, momega / (4 * math.pi * quality) * scale

def qnm_fit(mode,af,mt,mf):
    z=dimensionless_qnm(af,*mode)
    return z.real/(2*math.pi)*mt/mf, abs(z.imag)/(2*math.pi)*mt/mf

def qnm_leaver(mode,af,mt,mf):
    root=solve(float(af),mode[0],mode[1],0,strict=False)
    z=root.omega
    return z.real/(2*math.pi)*mt/mf, abs(z.imag)/(2*math.pi)*mt/mf, root.cf_depth,root.cf_error

def qnm_public_eob(mode, af, final_mass_msun):
    """Public EOB final-state API returns complex angular frequency (rad/s)."""
    v=lal.CreateCOMPLEX16Vector(1)
    rc=lalsimulation.SimIMREOBGenerateQNMFreqV2fromFinal(v,float(final_mass_msun),float(af),int(mode[0]),int(mode[1]),1)
    if rc != 0:
        raise RuntimeError(f"EOB QNM API returned {rc} for mode {mode}")
    z=complex(v.data[0])
    return z.real/(2*math.pi),abs(z.imag)/(2*math.pi)

def qnm_ratio_samples(qsets, total_mass_msun, sample_hz=(20., 50., 100., 200.)):
    """Report dimensionless QNM ratios at fixed physical frequencies.

    ``qsets`` stores dimensionless ``M_total*f`` values.  Convert the sample
    frequencies to that same convention before forming ratios; this routine
    is diagnostic only and does not select a prescription.
    """
    out = {}
    for mode in MODES:
        out[str(mode)] = {}
        for name, q in qsets.items():
            if mode not in q:
                out[str(mode)][name] = {'unsupported': True}
                continue
            value = q[mode]
            fr, fd = value[:2]
            carrier = q.get((2, 2))
            mf_samples = [float(f * MTSUN * total_mass_msun) for f in sample_hz]
            out[str(mode)][name] = {
                'frequencies_hz': list(sample_hz),
                'sample_Mf_total_mass_msun': mf_samples,
                'fr_over_sample_Mf': [float(fr / x) for x in mf_samples],
                'fdamp_over_sample_Mf': [float(fd / x) for x in mf_samples],
                'fr_over_22_Mf': (float(fr / carrier[0]) if carrier else None),
                'fdamp_over_22_Mf': (float(fd / carrier[1]) if carrier else None),
            }
    return out

def direct_22_identity_path(mf, carrier_positive, amplitude_22):
    """Independent mode-22 identity: carrier times its amplitude."""
    return carrier_positive(mf) * amplitude_22(mf)

def public_modes(freqs,c):
    m1,m2,s1,s2,_,_,phic,fref=c
    v=lal.CreateREAL8Vector(len(freqs)); v.data[:]=freqs
    h=lalsimulation.SimIMRPhenomHMGethlmModes(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,s1,0,0,s2,phic,float(freqs[1]-freqs[0]) if len(freqs)>1 else 0.,fref,lal.CreateDict())
    out={}; n=h
    while n: out[(int(n.l),int(n.m))]=np.asarray(n.mode.data.data,dtype=np.complex128); n=n.next
    return out

def public_mode_schema(freqs, c, raw):
    """Check sequence/scalar correspondence; this does not infer normalization."""
    freqs = np.asarray(freqs, dtype=float)
    uniform = np.linspace(freqs[0], freqs[-1], len(freqs))
    u = public_modes(uniform, c)
    picks = [float(x) for x in (20., 50., 100., 200.)]
    scalar_err = {}
    scalar_lengths = {}
    for mode in MODES:
        errs = []
        scalar_lengths[str(mode)] = []
        for f in picks:
            j = int(np.argmin(np.abs(freqs-f)))
            if abs(freqs[j]-f) > 1e-12:
                continue
            one = public_modes(np.array([f]), c)[mode]
            scalar_lengths[str(mode)].append(len(one))
            errs.append(abs(one[0]-raw[mode][j]))
        scalar_err[str(mode)] = max(errs, default=0.)
    return {
        'requested_nonuniform_length': len(freqs),
        'returned_nonuniform_lengths': {str(k): len(v) for k,v in raw.items()},
        'requested_uniform_length': len(uniform),
        'returned_uniform_lengths': {str(k): len(v) for k,v in u.items()},
        'scalar_lengths': scalar_lengths,
        'scalar_max_abs_error_at_20_50_100_200': scalar_err,
    }

def public_d_parts(x,m1,m2,s1,s2,dist,fref):
    v=lal.CreateREAL8Vector(len(x)); v.data[:]=np.asarray(x,dtype=float)
    # The public D sequence takes luminosity distance in SI metres; case inputs
    # store Mpc, as does the HM comparison harness.
    z=lalsimulation.SimIMRPhenomDFrequencySequence(v,0.,fref,m1*lal.MSUN_SI,m2*lal.MSUN_SI,s1,s2,dist*1e6*lal.PC_SI,lal.CreateDict(),lalsimulation.NRTidal_V)
    y=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2)); h=np.asarray(z.data.data,dtype=np.complex128)/y
    # The positive-frequency strain convention is h_+-i h_x =
    # sum h_lm * {}_{-2}Y_lm; the D callable's face-on sequence carries the
    # standard 1/2 polarization prefactor, so its recovered 22 coefficient is
    # doubled when compared with the unprojected HM coefficient.
    h = 2.0*h
    # Keep the principal phase here.  Mapped-frequency callers may pass sparse
    # or non-monotone arrays; unwrapping such arrays would manufacture branch
    # jumps.  Identity checks use the complex values directly.
    return torch.tensor(np.abs(h),dtype=torch.float64),torch.tensor(-np.angle(h),dtype=torch.float64)

def public_d_complex(x,m1,m2,s1,s2,dist,fref):
    """Return the doubled positive-frequency D 22 coefficient directly."""
    x=np.atleast_1d(np.asarray(x,dtype=float))
    v=lal.CreateREAL8Vector(len(x)); v.data[:]=x
    z=lalsimulation.SimIMRPhenomDFrequencySequence(v,0.,fref,m1*lal.MSUN_SI,m2*lal.MSUN_SI,s1,s2,dist*1e6*lal.PC_SI,lal.CreateDict(),lalsimulation.NRTidal_V)
    y=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2))
    return 2.0*np.asarray(z.data.data,dtype=np.complex128)/y

def public_d_schema(m1,m2,s1,s2,dist,fref):
    fs = np.array([20., 50., 100., 200.])
    va, vp = public_d_parts(fs,m1,m2,s1,s2,dist,fref)
    errs_a, errs_p = [], []
    for i, f in enumerate(fs):
        a, p = public_d_parts(np.array([f]),m1,m2,s1,s2,dist,fref)
        errs_a.append(abs(float(a[0])-float(va[i])))
        errs_p.append(abs(float(p[0])-float(vp[i])))
    deriv = {}
    for f in (50., 100., 200.):
        vals = []
        for h in (1e-2, 1e-3, 1e-4):
            _, pm = public_d_parts(np.array([f-h]),m1,m2,s1,s2,dist,fref)
            _, pp = public_d_parts(np.array([f+h]),m1,m2,s1,s2,dist,fref)
            # derivative of the positive-phase convention, with branch wrapping.
            dp = float(torch.remainder(pp[0]-pm[0]+math.pi, 2*math.pi)-math.pi)
            vals.append(dp/(2*h))
        deriv[str(f)] = vals
    return {'scalar_max_abs_amp_error': max(errs_a),
            'scalar_max_abs_phase_error': max(errs_p),
            'phase_derivative_h_1e-2_1e-3_1e-4': deriv,
            'mode22_scalar_identity_checked': True}

def public_d_mode22_identity(c, raw, scale):
    """Compare public D's face-on 22 coefficient with HM's unprojected 22.

    The factor two is the documented positive-frequency polarization prefactor;
    the remaining ratio is reported as a circular phase/amplitude diagnostic.
    """
    m1,m2,s1,s2,dist,_,_,fref=c
    idx=[int(np.argmin(abs(FREQ-f))) for f in (20.,50.,100.,200.)]
    a,_=public_d_parts(FREQ,m1,m2,s1,s2,dist,fref)
    # Recover the complex D mode before magnitude/phase conversion.
    v=lal.CreateREAL8Vector(len(FREQ)); v.data[:]=FREQ
    z=lalsimulation.SimIMRPhenomDFrequencySequence(v,0.,fref,m1*lal.MSUN_SI,m2*lal.MSUN_SI,s1,s2,dist*1e6*lal.PC_SI,lal.CreateDict(),lalsimulation.NRTidal_V)
    y=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2))
    dmode=2*np.asarray(z.data.data,dtype=np.complex128)/y
    hmode=np.asarray(raw[(2,2)],dtype=np.complex128)*scale
    ratios=dmode[idx]/hmode[idx]
    amp=np.abs(ratios); phase=np.angle(ratios)
    # The public HM/D face-on relation carries exp(-2 i phic) in this
    # positive-frequency convention. Remove that documented extrinsic gauge
    # factor before testing the intrinsic 22 identity.
    corrected = ratios * np.exp(2j * float(c[6]))
    corrected_phase = np.angle(corrected)
    if float(np.max(np.abs(corrected_phase))) >= 1e-12:
        raise AssertionError(f"public D/HM22 intrinsic identity failed: {corrected_phase}")
    deriv=[]
    for f in (50.,100.,200.):
        vals=[]
        for h in (1e-2,1e-3,1e-4):
            def cm(ff):
                vv=lal.CreateREAL8Vector(1); vv.data[0]=ff
                zz=lalsimulation.SimIMRPhenomDFrequencySequence(vv,0.,fref,m1*lal.MSUN_SI,m2*lal.MSUN_SI,s1,s2,dist*1e6*lal.PC_SI,lal.CreateDict(),lalsimulation.NRTidal_V)
                return 2*complex(zz.data.data[0])/y
            vals.append(np.angle(cm(f+h)*np.conj(cm(f-h)))/(2*h))
        deriv.append({'f':f,'values':vals})
    return {'frequencies_hz':[float(FREQ[i]) for i in idx],
            'amplitude_ratio_D_over_HM22':amp.tolist(),
            'circular_phase_ratio_rad':phase.tolist(),
            'extrinsic_phase_correction_exp_plus_2i_phic': [float(c[6]), 2.0],
            'corrected_circular_phase_ratio_rad':corrected_phase.tolist(),
            'max_abs_corrected_circular_phase_ratio_rad':float(np.max(np.abs(corrected_phase))),
            'phase_derivative_convergence':deriv,
            'max_abs_amplitude_ratio_minus_one':float(np.max(np.abs(amp-1))),
            'max_abs_circular_phase_ratio':float(np.max(np.abs(phase)))}

def row(c):
    m1,m2,s1,s2,dist,iota,phic,fref=c; mt=m1+m2; M=mt*MTSUN
    eta=m1*m2/mt**2; delta=(m1-m2)/mt; cs=(s1+s2)/2; ca=(s1-s2)/2
    d=IMRPhenomD(); args=[torch.tensor(x,dtype=torch.float64) for x in (m1,m2,s1,s2)]
    mf,af,frd,fdamp=d.remnant_properties(*args); mf=float(mf); af=float(af)
    frd=float(frd); fdamp=float(fdamp)
    chipn=(m1*s1+m2*s2)/mt-38*eta/113*(s1+s2)
    a4=float(d.eval_table_v('alpha4',torch.tensor(eta,dtype=torch.float64),torch.tensor(chipn,dtype=torch.float64)))
    a5=float(d.eval_table_v('alpha5',torch.tensor(eta,dtype=torch.float64),torch.tensor(chipn,dtype=torch.float64)))
    slope=float(d.compute_slope_peak(*args)); scale=mt*lal.MTSUN_SI*(mt*lal.MRSUN_SI)/(dist*1e6*lal.PC_SI)
    vec=lal.CreateREAL8Vector(len(FREQ)); vec.data[:]=FREQ
    raw=public_modes(FREQ,c)
    schema = public_mode_schema(FREQ, c, raw)
    d_schema = public_d_schema(m1,m2,s1,s2,dist,fref)
    d_identity = public_d_mode22_identity(c, raw, scale)
    def carrier(x): return d(x/M,m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=fref,return_phase=True)[2]
    def amp22(x):
        hp,hc=d(x/M,m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=None)
        return torch.abs((hp+1j*hc)/complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2)))
    native_carrier, native_amplitude = carrier, amp22
    def public_phase_branch(x):
        xx=np.atleast_1d(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x,dtype=float))
        p_native=np.atleast_1d(np.asarray(native_carrier(torch.tensor(xx,dtype=torch.float64))))
        a_native=np.atleast_1d(np.asarray(native_amplitude(torch.tensor(xx,dtype=torch.float64))))
        h_native=a_native*np.exp(-1j*p_native)
        h_public=public_d_complex(xx/M,m1,m2,s1,s2,dist,fref)
        valid=np.abs(h_native)>np.finfo(float).tiny
        denom=np.where(valid,h_native,1.0+0j)
        ratio=np.where(valid,h_public/denom,1.0+0j)
        return torch.tensor(p_native-np.angle(ratio),dtype=torch.float64)
    def public_amplitude(x):
        xx=np.atleast_1d(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x,dtype=float))
        return public_d_parts(xx/M,m1,m2,s1,s2,dist,fref)[0]
    harm=lambda p,t,s,l,m: lal.SpinWeightedSphericalHarmonic(float(t),float(p),s,l,m)
    qfit={(2,2):qnm_fit((2,2),af,mt,mf)}; qlev={(2,2):qnm_leaver((2,2),af,mt,mf)}
    berti_coeffs=load_berti_fits(); qberti={mode:qnm_berti_published(mode,af,mt,mf,berti_coeffs) for mode in MODES}
    for mode in MODES[1:]: qfit[mode]=qnm_fit(mode,af,mt,mf); qlev[mode]=qnm_leaver(mode,af,mt,mf)
    # The London HM paper uses the PhenomD 22 carrier.  This control keeps
    # that carrier's actual black-box remnant frequencies and changes only
    # the higher-mode QNM recipe.
    qd22=(frd,fdamp)
    qd_london=dict(qfit); qd_london[(2,2)]=qd22
    qd_leaver=dict(qlev); qd_leaver[(2,2)]=qd22
    qeob={}; qeob_errors={}
    for mode in MODES:
        try:
            hz=qnm_public_eob(mode,af,mf)
            # EOB API returns Hz; assembly uses dimensionless Mf.
            qeob[mode]=(hz[0]*M,hz[1]*M)
        except Exception as exc: qeob_errors[str(mode)]=str(exc)
    qeob_d22=dict(qeob); qeob_d22[(2,2)]=qd22
    qnm_sets={'London_fit': qfit, 'D22_London_higher': qd_london,
              'D22_Leaver_higher': qd_leaver,
              'Leaver': {k:v[:2] for k,v in qlev.items()},
              'Berti_author_fit_n0': qberti,
              'public_EOB': qeob}
    qnm_samples=qnm_ratio_samples(qnm_sets, mt)
    variants={}
    for name,q in [('london_fit',qfit),('D22_london_higher',qd_london),
                   ('D22_leaver_higher',qd_leaver),
                   ('leaver', {k:v[:2] for k,v in qlev.items()}),
                   ('berti_author_fit_n0', qberti),
                   ('public_EOB_all',qeob), ('D22_public_EOB_higher',qeob_d22),
                   ('london_fit_public_D_phase',qfit), ('london_fit_public_D_amp',qfit),
                   ('london_fit_public_D_both',qfit)]:
        if 'public_EOB' in name and len(q) < len(MODES):
            continue
        carrier_v=native_carrier; amplitude_v=native_amplitude
        if name.endswith('public_D_phase') or name.endswith('public_D_amp') or name.endswith('public_D_both'):
            # assemble_mode receives dimensionless Mf; the public D API receives Hz.
            # The identity check below records the HM/D phic factor.  Keep the
            # D carrier at phi0=0 here; assemble_mode applies the requested
            # harmonic m*phic transformation exactly once.
            carrier_v=public_phase_branch if name.endswith('public_D_phase') or name.endswith('public_D_both') else native_carrier
            amplitude_v=public_amplitude if name.endswith('public_D_amp') or name.endswith('public_D_both') else native_amplitude
        else:
            carrier_v=native_carrier; amplitude_v=native_amplitude
        carrier=carrier_v; amplitude=amplitude_v
        pred={}
        for mode in MODES:
            gauge=((mode[1]/2)-1)*slope*M*fref
            pred[mode]=assemble_mode(torch.tensor(FREQ*M,dtype=torch.float64),mode,eta=eta,delta=delta,chi_s=cs,chi_a=ca,carrier_positive=carrier,amplitude_22=amplitude,amp_qnm_22=q[(2,2)][0],amp_qnm_lm=q[mode][0],phase_qnm_22=q[(2,2)][0],phase_qnm_lm=q[mode][0],phase_damp_22=q[(2,2)][1],phase_damp_lm=q[mode][1],alpha4=a4,alpha5=a5,chi_pn=chipn,phic=phic,reference_gauge=gauge,use_eq14=True,amplitude_join=COMPATIBILITY_AMPLITUDE_JOIN,phase_join=COMPATIBILITY_PHASE_JOIN).detach()
        errs={}
        for mode in MODES:
            tar=torch.tensor(raw[mode]*scale,dtype=torch.complex128); e=pred[mode]-tar; ph=torch.angle(pred[mode]*torch.conj(tar)); da=pred[mode].abs()-tar.abs()
            ridx=[int(np.argmin(abs(FREQ-f))) for f in (20.,50.,100.,200.)]
            rr=(pred[mode][ridx]/tar[ridx]).detach().numpy()
            errs[str(mode)]={'raw_relative_l2':float(torch.linalg.vector_norm(e)/torch.linalg.vector_norm(tar)),'amplitude_relative_l2':float(torch.linalg.vector_norm(da)/torch.linalg.vector_norm(tar)),'phase_rms':float(torch.sqrt(torch.mean(ph.square()))),'ratio_samples_real':rr.real.tolist(),'ratio_samples_imag':rr.imag.tolist()}
        php,phc=project_positive_modes(pred,iota,harm); thp,thc=project_positive_modes({k:torch.tensor(raw[k]*scale) for k in MODES},iota,harm)
        variants[name]={'per_mode':errs,'hp_relative_l2':float(torch.linalg.vector_norm(php-thp)/torch.linalg.vector_norm(thp)),'hc_relative_l2':float(torch.linalg.vector_norm(phc-thc)/torch.linalg.vector_norm(thc))}
    supported=(2,2),(2,1),(3,3),(4,4)
    eob_supported={}
    eob_guard=[]
    for mode in supported:
        egauge=((mode[1]/2)-1)*slope*M*fref
        pred=assemble_mode(torch.tensor(FREQ*M,dtype=torch.float64),mode,eta=eta,delta=delta,chi_s=cs,chi_a=ca,carrier_positive=native_carrier,amplitude_22=native_amplitude,amp_qnm_22=qeob[(2,2)][0],amp_qnm_lm=qeob[mode][0],phase_qnm_22=qeob[(2,2)][0],phase_qnm_lm=qeob[mode][0],phase_damp_22=qeob[(2,2)][1],phase_damp_lm=qeob[mode][1],alpha4=a4,alpha5=a5,chi_pn=chipn,phic=phic,reference_gauge=egauge,use_eq14=True,amplitude_join=COMPATIBILITY_AMPLITUDE_JOIN,phase_join=COMPATIBILITY_PHASE_JOIN).detach()
        tar=torch.tensor(raw[mode]*scale,dtype=torch.complex128); e=pred-tar; da=pred.abs()-tar.abs()
        # Independent mode-22 identity, rather than calling assemble_mode a
        # second time.  Other supported modes have no tautological guard.
        if mode == (2, 2):
            check=direct_22_identity_path(torch.tensor(FREQ*M,dtype=torch.float64), native_carrier, native_amplitude).detach()
            eob_guard.append(float(torch.max(torch.abs(pred-check))))
        eob_supported[str(mode)]={'raw_relative_l2':float(torch.linalg.vector_norm(e)/torch.linalg.vector_norm(tar)),'amplitude_relative_l2':float(torch.linalg.vector_norm(da)/torch.linalg.vector_norm(tar)),'phase_rms':float(torch.sqrt(torch.mean(torch.angle(pred*torch.conj(tar)).square())))}
    # This diagnostic intentionally retains the former compatibility phase
    # join (the clean paper default is .014).  Report its physical location
    # and the residual immediately below it; this separates inspiral mapping
    # from post-join QNM behavior.
    join_diagnostics={}
    for mode in MODES[1:]:
        f0=COMPATIBILITY_PHASE_JOIN*qfit[mode][0]/qfit[(2,2)][0]
        x=.99*f0
        pred0=assemble_mode(torch.tensor([x],dtype=torch.float64),mode,
            eta=eta,delta=delta,chi_s=cs,chi_a=ca,carrier_positive=native_carrier,
            amplitude_22=native_amplitude,amp_qnm_22=qfit[(2,2)][0],amp_qnm_lm=qfit[mode][0],
            phase_qnm_22=qfit[(2,2)][0],phase_qnm_lm=qfit[mode][0],phase_damp_22=qfit[(2,2)][1],
            phase_damp_lm=qfit[mode][1],alpha4=a4,alpha5=a5,chi_pn=chipn,phic=phic,
            reference_gauge=((mode[1]/2)-1)*slope*M*fref,use_eq14=True,
            amplitude_join=COMPATIBILITY_AMPLITUDE_JOIN,
            phase_join=COMPATIBILITY_PHASE_JOIN).detach()[0]
        ref0=torch.tensor(public_modes(np.array([x/M]),c)[mode][0]*scale,dtype=torch.complex128)
        join_diagnostics[str(mode)]={'amplitude_map_join_hz':float(COMPATIBILITY_AMPLITUDE_JOIN*qfit[mode][0]/qfit[(2,2)][0]/M),
            'phase_map_join_hz':float(f0/M),'ringdown_join_hz':float(qfit[mode][0]/M),
            'below_phase_join_hz':float(x/M),
            'below_phase_join_raw_relative_error':float(torch.abs(pred0-ref0)/torch.abs(ref0))}
    return {'inputs':c,'remnant':{'Mf_solar_mass':mf,'af':af,'D_fRD':frd,'D_fdamp':fdamp},
            'public_frequency_schema': schema, 'public_D_schema': d_schema,
            'public_D_vs_HM22_identity': d_identity,
            'qnm_london':{str(k):qfit[k] for k in MODES},'qnm_leaver':{str(k):qlev[k] for k in MODES},
            'qnm_berti_author_fit_n0':{str(k):qberti[k] for k in MODES},
            'qnm_public_eob':{str(k):qeob[k] for k in qeob},
            'qnm_public_eob_errors':qeob_errors,
            'qnm_ratio_samples':qnm_samples,
            'join_diagnostics':join_diagnostics,
            'public_EOB_supported_mode_metrics':eob_supported,
            'public_EOB_22_identity_guard_max_abs':max(eob_guard),
            'variants':variants}

def main():
    out={'provenance':'London HM Eq3/Eq4/Eq14 assembly; printed QNM fits versus independent Kerr Leaver solver; diagnostic only','frequency_hz':FREQ.tolist(),'cases':[row(c) for c in CASES],'versions':{'python':sys.version,'torch':torch.__version__,'lal':lal.__version__,'lalsimulation':lalsimulation.__version__},'sha256':{}}
    for p in (__file__,'scripts/kerr_leaver_solver.py','src/torchwave/approximants/hm_assembly_clean.py','src/torchwave/approximants/kerr_qnm_london_published.py'): out['sha256'][p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out['berti_fit_provenance']={'url':'https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat','path':str(BERTI_FIT_PATH),'sha256':hashlib.sha256(BERTI_FIT_PATH.read_bytes()).hexdigest(),'retrieval_note':'Author page text retrieved independently; direct shell download returned HTTP 403, so this external copy is preserved only in /tmp.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
