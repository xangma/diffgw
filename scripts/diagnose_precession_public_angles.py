"""Process-local XP102 public-angle attribution; never used by production."""
import hashlib,json,sys
from pathlib import Path
import lal,lalsimulation,torch
import torchwave.approximants.precession_msa as msa
import torchwave.approximants.imrphenom_xp as xpmod
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.precession_nnlo_clean import coefficients
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry
from parity_harness import get_canonical_catalog_cases,get_canonical_holdout_cases,direct_complex_relative_l2
from diagnose_precession_nnlo_paired import public_fd

ORIG=None; CALLS=0
def coeff(eta,delta,m1_norm,chip,chi_l):
    aa,ee=coefficients(eta,-delta,m1_norm,chi_l/m1_norm,chip)
    return tuple(aa.unbind(-1))+tuple(ee.unbind(-1))

def public_adapter(v,v0,**p):
    global CALLS
    CALLS+=1; version=p.pop('prec_version',102); conv=p.pop('convention',1); p.pop('return_carrier_phase',None); p.pop('phi_ref',None)
    native=ORIG(v,v0,**p,phi_ref=torch.tensor(0.,dtype=v.dtype,device=v.device),prec_version=version,convention=conv,return_carrier_phase=True)
    M=p['m1']+p['m2']; f=(v**3/(torch.pi*M*lal.MTSUN_SI)).detach().cpu().reshape(-1)
    d=lal.CreateDict(); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,int(version))
    fv=lal.CreateREAL8Vector(f.numel()); fv.data[:]=f.tolist()
    out=lalsimulation.SimIMRPhenomXPPNAngles(fv,float(p['m1']*lal.MSUN_SI),float(p['m2']*lal.MSUN_SI),float(p['s1x']),float(p['s1y']),float(p['s1z']),float(p['s2x']),float(p['s2y']),float(p['s2z']),float(p['iota']),float((v0**3/(torch.pi*M*lal.MTSUN_SI)).detach()),2,d)
    a=torch.tensor(list(out[0].data),dtype=v.dtype,device=v.device); g=torch.tensor(list(out[1].data),dtype=v.dtype,device=v.device); b=torch.acos(torch.tensor(list(out[2].data),dtype=v.dtype,device=v.device))
    return a,b,g,native[3],native[4],native[5]

def evaluate(c, angles):
    global ORIG
    import copy
    cc=copy.deepcopy(c)
    # Public PNAngles has no phiRef argument; use the same phic=0 control for
    # both baseline and substitution so the comparison is a gauge-matched one.
    cc['params']['phic']=0.
    f,refp,refc=public_fd(cc,0); g=cc['grid']; kw=dict(cc['params']); kw.pop('f_lower',None); kw.pop('f_final',None); kw.update(prec_version=102,convention=1)
    old=IMRPhenomP.__dict__['nnlo_angle_coefficients']; ORIG=msa.evaluate_msa_precession; oldxp=xpmod.evaluate_msa_precession
    IMRPhenomP.nnlo_angle_coefficients=staticmethod(coeff)
    if angles: msa.evaluate_msa_precession=public_adapter; xpmod.evaluate_msa_precession=public_adapter
    try: hp,hc=get_fd_waveform('IMRPhenomXP',sample_frequencies=f,f_lower=float(g['flow']),f_final=float(g['fhigh']),dtype=torch.float64,**kw)
    finally: IMRPhenomP.nnlo_angle_coefficients=old; msa.evaluate_msa_precession=ORIG; xpmod.evaluate_msa_precession=oldxp
    sel=(f>=float(g['flow']))&(f<float(g['fhigh']))
    return {'plus_raw_l2':direct_complex_relative_l2(hp[sel],refp[sel]),'cross_raw_l2':direct_complex_relative_l2(hc[sel],refc[sel]),'n':int(sel.sum())}

def pointwise_angle_control(c):
    """Compare intrinsic angles at explicit frequencies in the phic=0 gauge."""
    global ORIG
    import copy
    q = c['params']
    p = {'m1': q['mass1'], 'm2': q['mass2'], 's1x': q['spin1x'], 's1y': q['spin1y'],
         's1z': q['spin1z'], 's2x': q['spin2x'], 's2y': q['spin2y'], 's2z': q['spin2z'],
         'iota': q['inclination'], 'f_ref': q['f_ref'], 'phic': 0.0, 'tc': q.get('tc', 0.0)}
    p['phic'] = 0.0
    freqs = torch.tensor([float(p['f_ref']), 50.0, 100.0, 200.0], dtype=torch.float64)
    M = float(p['m1'] + p['m2'])
    v = (torch.pi * M * lal.MTSUN_SI * freqs) ** (1.0 / 3.0)
    v0 = (torch.pi * M * lal.MTSUN_SI * float(p['f_ref'])) ** (1.0 / 3.0)
    old = IMRPhenomP.__dict__['nnlo_angle_coefficients']
    ORIG = msa.evaluate_msa_precession
    IMRPhenomP.nnlo_angle_coefficients = staticmethod(coeff)
    p_call = dict(p)
    p_call.pop('f_ref', None)
    p_call.pop('phic', None)
    p_call.pop('tc', None)
    p_call = {k: torch.as_tensor(x, dtype=torch.float64) for k, x in p_call.items()}
    try:
        native = ORIG(v, v0, **p_call, phi_ref=torch.tensor(0., dtype=torch.float64), prec_version=102, convention=1, return_carrier_phase=True)
        public = public_adapter(v, v0, **p_call, prec_version=102, convention=1)
    finally:
        IMRPhenomP.nnlo_angle_coefficients = old
    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))
    da = wrap(native[0] - public[0])
    dg = wrap(native[2] - public[2])
    db = native[1] - public[1]
    geom = source_frame_geometry(p['m1'], p['m2'], p['s1x'], p['s1y'], p['s1z'],
                                 p['s2x'], p['s2y'], p['s2z'], p['iota'],
                                 0.0, p['f_ref'], compat_102=True)
    return {
        'frequencies_hz': freqs.tolist(),
        'native_minus_public_alpha_wrapped_rad': da.tolist(),
        'native_minus_public_gamma_wrapped_rad': dg.tolist(),
        'native_minus_public_beta_rad': db.tolist(),
        'alpha_anchor_removed_rad': (da - da[0]).tolist(),
        'gamma_anchor_removed_rad': (dg - dg[0]).tolist(),
        'source_geometry_at_fref': {
            'alpha0_rad': float(geom[0]),
            'theta_jn_rad': float(geom[1]),
            'zeta_rad': float(geom[2]),
            'phi_j_rad': float(geom[3]),
            'native_theta_minus_clean_rad': float(native[3] - geom[1]),
            'native_zeta_wrapped_minus_clean_rad': float(wrap(native[4] - geom[2])),
        },
    }

if __name__=='__main__':
    cs=[next(c for c in get_canonical_catalog_cases() if c['torch_name']=='IMRPhenomXP'),next(c for c in get_canonical_holdout_cases() if c['torch_name']=='IMRPhenomXP')]
    out={'provenance':'Public SimIMRPhenomXPPNAngles alpha/gamma/beta with exact case inclination; native theta,zeta,carrier; XP102 convention1; both waveform controls phic=0. Public output[1] is gamma=-epsilon.','python':sys.version,'versions':{'lal':lal.__version__,'lalsimulation':lalsimulation.__version__},'cases':[]}
    for c in cs:
        out['cases'].append({'id':c['id'],'clean_coeff':evaluate(c,False),'clean_coeff_public_angles':evaluate(c,True),'pointwise_angle_control':pointwise_angle_control(c),'adapter_calls':CALLS})
    for p in ['scripts/diagnose_precession_public_angles.py','src/torchwave/approximants/precession_beta_clean.py']:
        out.setdefault('sha256',{})[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    Path('docs/reviews/2026-09-13-parity-review/precession-public-angle-attribution.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
