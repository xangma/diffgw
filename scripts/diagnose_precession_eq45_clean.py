"""Standalone Eq.45 positive-mode twist using an XAS face-on carrier.

The prediction path calls only the clean XAS aligned carrier and the fresh
angle/frame/projector helpers; native XP is used solely for the comparison.
"""
import copy, hashlib, json, sys
from pathlib import Path
import lal, lalsimulation as ls, torch
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS, XASParameterFits
from torchwave.approximants.precession_nnlo_clean import raw_angles, effective_chip
from torchwave.approximants.precession_beta_clean import beta
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry
from torchwave.approximants.precession_twisting_clean import project_twisted_positive_modes
from torchwave.waveforms import get_fd_waveform
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases, direct_complex_relative_l2
from diagnose_precession_nnlo_paired import public_fd
from compare_precession_source_frame_public import one as public_geometry_record

def clean_wave(case, f=None, public_geometry=None, angle_override=None):
    p, g = case['params'], case['grid']
    if f is None:
        f = torch.arange(0., float(g.get('oracle_fhigh', g['fhigh'])) + float(g['df']), float(g['df']), dtype=torch.float64)
    M = torch.tensor(float(p['mass1'] + p['mass2']) * lal.MTSUN_SI, dtype=torch.float64)
    support = f >= float(g['flow'])
    # PN angles are undefined at DC and below the requested flow.  Evaluate
    # those bins at the flow solely to keep tensor arithmetic finite, then
    # explicitly zero unsupported output bins.
    f_safe = torch.where(f > 0, f, torch.full_like(f, float(g['flow'])))
    v = (torch.pi * M * f_safe).pow(1/3)
    v0 = (torch.pi * M * float(p['f_ref'])).pow(1/3)
    m1,m2 = (torch.tensor(float(p[k]),dtype=torch.float64) for k in ('mass1','mass2'))
    s = [torch.tensor(float(p.get(k,0.)),dtype=torch.float64) for k in ('spin1x','spin1y','spin1z','spin2x','spin2y','spin2z')]
    eta=m1*m2/(m1+m2)**2; delta=(m1-m2)/(m1+m2); x1=m1/(m1+m2)
    chip=effective_chip(m1,m2,s[0],s[1],s[3],s[4]); chieff=(m1*s[2]+m2*s[5])/(m1+m2)
    aa,ee=raw_angles(v,eta,-delta,x1,chieff/x1,chip); a0,e0=raw_angles(v0,eta,-delta,x1,chieff/x1,chip)
    geom=source_frame_geometry(m1,m2,*s,float(p['inclination']),float(p.get('phic',0.)),float(p['f_ref']),compat_102=False)
    if public_geometry is not None:
        # Public tuple is (chi1,chi2,chip,thetaJN,alpha0,phiJL0,zeta).
        geom = (torch.tensor(public_geometry[4], dtype=torch.float64),
                torch.tensor(public_geometry[3], dtype=torch.float64),
                torch.tensor(public_geometry[6] + torch.pi / 2, dtype=torch.float64),
                torch.tensor(public_geometry[5], dtype=torch.float64))
    b=beta(m1,m2,f_safe,*s,compat_102=False)
    xas=IMRPhenomXAS(); af=XASParameterFits.fit_final_spin_nnlo(eta, s[2], s[5], chip)
    # Diagnostic carrier convention under test: use the public phiJL0 when
    # supplied; the clean baseline uses the analytic positive-frequency h22
    # sign and observer phase.  This is not asserted as a general gauge fix.
    phi_c = geom[3] if public_geometry is not None else -geom[3]
    hp0,hc0=xas(f=f,mass1=m1,mass2=m2,chi1=s[2],chi2=s[5],distance=float(p.get('distance',100.)),inclination=0.,phi_c=phi_c,f_ref=float(p['f_ref']),_final_spin_override=af)
    hneg=2*hp0/torch.sqrt(torch.tensor(5/(4*torch.pi),dtype=torch.float64))
    if angle_override is None:
        alpha, gamma, beta_value = aa-a0+geom[0], -ee+e0, b
    else:
        alpha, gamma, beta_value = angle_override
    hp, hc = project_twisted_positive_modes({(2,2): hneg}, alpha, gamma, beta_value, geom[1], geom[2])
    return (torch.where(support, hp, torch.zeros_like(hp)),
            torch.where(support, hc, torch.zeros_like(hc))), f

def public_angle_series(case, f):
    """Public PNAngles black-box, anchored at an exact f_ref singleton."""
    p = case['params']; d = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, 102)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(d, 0)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(d, 0)
    assert ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(d) == 102
    assert ls.SimInspiralWaveformParamsLookupPhenomXPConvention(d) == 0
    assert ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(d) == 0
    fs = torch.where(f > 0, f, torch.full_like(f, float(case['grid']['flow'])))
    def call(x):
        fv = lal.CreateREAL8Vector(int(x.numel())); fv.data[:] = x.detach().cpu().numpy()
        r = ls.SimIMRPhenomXPPNAngles(fv, p['mass1']*lal.MSUN_SI, p['mass2']*lal.MSUN_SI,
            p.get('spin1x',0.),p.get('spin1y',0.),p.get('spin1z',0.),p.get('spin2x',0.),p.get('spin2y',0.),p.get('spin2z',0.),
            p['inclination'],p['f_ref'],2,d)
        return [torch.tensor(list(x.data),dtype=torch.float64) for x in r]
    alpha_public, gamma_public, cosbeta_public = call(fs)
    # Keep the first anchor exactly at f_ref while avoiding the public
    # routine's zero-width-vector warning.
    fr=float(p['f_ref'])
    alpha_ref, gamma_ref, cosbeta_ref = call(torch.tensor([fr,fr+1e-9],dtype=torch.float64))
    m1,m2 = (torch.tensor(float(p[k]),dtype=torch.float64) for k in ('mass1','mass2'))
    s = [torch.tensor(float(p.get(k,0.)),dtype=torch.float64) for k in
         ('spin1x','spin1y','spin1z','spin2x','spin2y','spin2z')]
    g = source_frame_geometry(m1,m2,*s,float(p['inclination']),float(p.get('phic',0.)),float(p['f_ref']),compat_102=False)
    public_result = (alpha_public-alpha_ref[0]+g[0], gamma_public-gamma_ref[0],
                     torch.acos(torch.clamp(cosbeta_public,-1.,1.)))
    m1,m2 = (torch.tensor(float(p[k]),dtype=torch.float64) for k in ('mass1','mass2'))
    M=m1+m2; v=(torch.pi*M*lal.MTSUN_SI*fs).pow(1/3); v0=(torch.pi*M*lal.MTSUN_SI*fr).pow(1/3)
    eta=m1*m2/M**2; delta=(m1-m2)/M; x1=m1/M
    chip=effective_chip(m1,m2,s[0],s[1],s[3],s[4]); chieff=(m1*s[2]+m2*s[5])/M
    ca,ce=raw_angles(v,eta,-delta,x1,chieff/x1,chip); ca0,ce0=raw_angles(v0,eta,-delta,x1,chieff/x1,chip)
    clean_result=(ca-ca0+g[0], -ce+ce0, beta(m1,m2,fs,*s,compat_102=False))
    return public_result, {"alpha_max_abs":float((public_result[0]-clean_result[0]).abs().max()),
                           "gamma_max_abs":float((public_result[1]-clean_result[1]).abs().max()),
                           "beta_max_abs":float((public_result[2]-clean_result[2]).abs().max())}

def main():
    cases=[next(c for c in get_canonical_catalog_cases() if c['torch_name']=='IMRPhenomXP'),next(c for c in get_canonical_holdout_cases() if c['torch_name']=='IMRPhenomXP')]
    for i, q in enumerate(((40.,20.,.3,.2,.4,-.2,.1,-.2,.7,40.),
                           (30.,10.,.1,-.25,-.4,.15,.05,.2,.5,40.),
                           (50.,15.,-.2,.3,.1,.1,-.2,.25,1.0,30.),
                           (28.,12.,.05,-.1,.6,-.3,.2,-.4,.4,25.))):
        m1,m2,x1,y1,z1,x2,y2,z2,inc,rf=q
        cases.append({'id':f'IMRPhenomXP_independent_{i}','torch_name':'IMRPhenomXP','lal_name':'IMRPhenomXP',
            'params':{'mass1':m1,'mass2':m2,'spin1x':x1,'spin1y':y1,'spin1z':z1,'spin2x':x2,'spin2y':y2,'spin2z':z2,'inclination':inc,'phic':0.,'f_ref':rf,'distance':100.,'prec_version':102,'convention':0,'final_spin_mod':0},'grid':{'flow':25.,'fhigh':160.,'df':.25}})
    base = copy.deepcopy(cases[0])
    for i, inc in enumerate((0., .7, 1.1)):
        q = copy.deepcopy(base); q['id'] = f'IMRPhenomXP_aligned_{i}'
        q['params'].update(spin1x=0., spin1y=0., spin2x=0., spin2y=0., inclination=inc, phic=0.3)
        cases.append(q)
    out={'python':sys.version,'cases':[],'provenance':'standalone clean Eq45 projector with XAS face-on h22 carrier; public/native XP comparison only'}
    for c in cases:
        c = copy.deepcopy(c)
        c['params'].update(prec_version=102, convention=0, final_spin_mod=0)
        options = {k: c['params'][k] for k in ('prec_version', 'convention', 'final_spin_mod')}
        assert options == {'prec_version': 102, 'convention': 0, 'final_spin_mod': 0}
        f,pubp,pubc=public_fd(c,0)
        public_record = public_geometry_record(c)
        assert public_record['options'] == public_record['looked_up_options'] == options
        public_tuple = public_record['public']
        pred,f=clean_wave(c, f); pred_public_geom,_=clean_wave(c, f, public_geometry=public_tuple)
        public_angles, angle_diagnostics = public_angle_series(c, f)
        pred_public_angles,_ = clean_wave(c, f, angle_override=public_angles)
        kw=copy.deepcopy(c['params']); kw.pop('f_lower',None);kw.pop('f_final',None)
        nh,nc=get_fd_waveform('IMRPhenomXP',sample_frequencies=f,f_lower=c['grid']['flow'],f_final=c['grid']['fhigh'],dtype=torch.float64,**kw)
        sel=(f>=c['grid']['flow'])&(f<c['grid']['fhigh'])
        if not (torch.isfinite(pred[0]).all() and torch.isfinite(pred[1]).all()):
            out['cases'].append({'id':c['id'],'options':options,'n':int(sel.sum()),'status':'unsupported clean source-frame geometry on requested physical support'})
            continue
        row={'id':c['id'],'options':options,'n':int(sel.sum()),'angle_max_abs_rad':angle_diagnostics,'clean_plus':direct_complex_relative_l2(pred[0][sel],pubp[sel]),'clean_cross':direct_complex_relative_l2(pred[1][sel],pubc[sel]),'public_geometry_plus':direct_complex_relative_l2(pred_public_geom[0][sel],pubp[sel]),'public_geometry_cross':direct_complex_relative_l2(pred_public_geom[1][sel],pubc[sel]),'public_angles_plus':direct_complex_relative_l2(pred_public_angles[0][sel],pubp[sel]),'public_angles_cross':direct_complex_relative_l2(pred_public_angles[1][sel],pubc[sel]),'native_plus':direct_complex_relative_l2(nh[sel],pubp[sel]),'native_cross':direct_complex_relative_l2(nc[sel],pubc[sel])}
        for label, got, target in (('plus',pred[0],pubp),('cross',pred[1],pubc)):
            ratio=got[sel]/torch.where(target[sel].abs()>1e-300,target[sel],torch.ones_like(target[sel]))
            ph=torch.angle(ratio); amp=ratio.abs()
            constant=torch.angle(torch.mean(torch.exp(1j*ph)))
            row[f'{label}_ratio_phase_rms_rad']=float(torch.sqrt(torch.mean(ph**2)))
            row[f'{label}_ratio_phase_variation_rad']=float(torch.std(ph))
            row[f'{label}_ratio_amplitude_rms']=float(torch.sqrt(torch.mean((amp-1)**2)))
            row[f'{label}_diagnostic_constant_phase_rad']=float(constant)
        out['cases'].append(row)
    assert sum('clean_plus' in r for r in out['cases'][:6]) == 6
    for q in ('src/torchwave/approximants/precession_twisting_clean.py','src/torchwave/approximants/precession_source_frame_clean.py','scripts/diagnose_precession_eq45_clean.py'): out.setdefault('sha256',{})[q]=hashlib.sha256(Path(q).read_bytes()).hexdigest()
    Path('docs/reviews/2026-09-13-parity-review/precession-eq45-clean.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
