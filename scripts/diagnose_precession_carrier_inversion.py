"""Diagnostic inversion of the fresh Eq45 projector against public XP waveforms."""
import copy, hashlib, json
from pathlib import Path
import lal, torch
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS, XASParameterFits
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry
from torchwave.approximants.precession_twisting_clean import project_twisted_positive_modes
from torchwave.approximants.precession_nnlo_clean import raw_angles, effective_chip
from torchwave.approximants.precession_beta_clean import beta
from parity_harness import direct_complex_relative_l2
from diagnose_precession_eq45_clean import public_angle_series
from compare_precession_source_frame_public import cases
from diagnose_precession_nnlo_paired import public_fd


def factors(c, f, b, angles):
    p=c['params']; m1=torch.tensor(float(p['mass1']),dtype=torch.float64); m2=torch.tensor(float(p['mass2']),dtype=torch.float64)
    h=torch.ones_like(f,dtype=torch.complex128)
    hp,hc=project_twisted_positive_modes({(2,2):h}, angles[0], angles[1], b, angles[2], angles[3])
    return hp,hc


def one(c):
    c=copy.deepcopy(c); c['params'].update(prec_version=102, convention=0, final_spin_mod=0)
    p,g=c['params'],c['grid']; f,pubp,pubc=public_fd(c,0)
    sel=(f>=float(g['flow']))&(f<float(g['fhigh']))
    m1=torch.tensor(float(p['mass1']),dtype=torch.float64); m2=torch.tensor(float(p['mass2']),dtype=torch.float64)
    s=[torch.tensor(float(p.get(k,0.)),dtype=torch.float64) for k in ('spin1x','spin1y','spin1z','spin2x','spin2y','spin2z')]
    eta=m1*m2/(m1+m2)**2; delta=(m1-m2)/(m1+m2); x1=m1/(m1+m2); M=(m1+m2)*lal.MTSUN_SI
    f_safe=torch.where(f>0,f,torch.full_like(f,float(g['flow'])))
    v=(torch.pi*M*f_safe).pow(1/3); v0=(torch.pi*M*float(p['f_ref'])).pow(1/3)
    chip=effective_chip(m1,m2,s[0],s[1],s[3],s[4]); chieff=(m1*s[2]+m2*s[5])/(m1+m2)
    aa,ee=raw_angles(v,eta,-delta,x1,chieff/x1,chip); a0,e0=raw_angles(v0,eta,-delta,x1,chieff/x1,chip)
    geom=source_frame_geometry(m1,m2,*s,float(p['inclination']),float(p.get('phic',0.)),float(p['f_ref']),compat_102=True)
    clean_angles=(aa-a0+geom[0],-ee+e0,geom[1],geom[2])
    public_angles,_=public_angle_series(c,f)
    public_geom_angles=(public_angles[0],public_angles[1],geom[1],geom[2])
    bclean=beta(m1,m2,f_safe,*s,compat_102=True); bprint=beta(m1,m2,f_safe,*s,compat_102=False); bpub=public_angles[2]
    # The XAS carrier is an independent clean quantity, not inferred from the
    # target.  Its phase convention follows the clean source-frame geometry.
    af=XASParameterFits.fit_final_spin_nnlo(eta,s[2],s[5],chip); xas=IMRPhenomXAS()
    hp0,_=xas(f=f_safe,mass1=m1,mass2=m2,chi1=s[2],chi2=s[5],distance=float(p.get('distance',100.)),inclination=0.,phi_c=-geom[3],f_ref=float(p['f_ref']),_final_spin_override=af)
    h22=2*hp0/torch.sqrt(torch.tensor(5/(4*torch.pi),dtype=torch.float64)); h22=torch.where(f>=float(g['flow']),h22,torch.zeros_like(h22))
    rows=[]
    for label,bv,ang in (('clean_beta_compat102',bclean,clean_angles),('clean_beta_printed3PN',bprint,clean_angles),('public_beta',bpub,public_geom_angles)):
        Fp,Fc=factors(c,f,bv,ang); den=Fp.abs()**2+Fc.abs()**2; good=sel&(den>1e-24)
        hest=(torch.conj(Fp)*pubp+torch.conj(Fc)*pubc)/den
        recp, recc=Fp*hest,Fc*hest
        ratio=hest[good]/torch.where(h22[good].abs()>1e-300,h22[good],torch.ones_like(h22[good]))
        sample={}
        for hz in (30.,80.,140.):
            j=torch.argmin((f[good]-hz).abs()); sample[str(hz)]={'amplitude_ratio':float(ratio.abs()[j]),'phase_rad':float(torch.angle(ratio)[j])}
        target_pol=pubc[good]/torch.where(pubp[good].abs()>1e-300,pubp[good],torch.ones_like(pubp[good])); projected_pol=Fc[good]/torch.where(Fp[good].abs()>1e-300,Fp[good],torch.ones_like(Fp[good]))
        rows.append({'variant':label,'closure_plus':direct_complex_relative_l2(recp[good],pubp[good]),'closure_cross':direct_complex_relative_l2(recc[good],pubc[good]),'hest_finite':bool(torch.isfinite(hest[good]).all()),'support_n':int(good.sum()),'carrier_vs_xas_l2':direct_complex_relative_l2(hest[good],h22[good]),'carrier_amp_rms':float(torch.sqrt(torch.mean((ratio.abs()-1)**2))),'carrier_phase_rms':float(torch.sqrt(torch.mean(torch.angle(ratio)**2))),'carrier_samples':sample,'projected_vs_target_cross_over_plus_rms':float(torch.sqrt(torch.mean(torch.abs(projected_pol-target_pol)**2)))})
    return {'id':c['id'],'options':{'prec_version':102,'convention':0,'final_spin_mod':0},'variants':rows}


def main():
    cs=cases()
    for c in cs:
        c.setdefault('grid', {'flow':25.,'fhigh':160.,'df':.25})
        c.setdefault('lal_name', 'IMRPhenomXP')
    out={'provenance':'linear Eq45 carrier inversion; public waveform is target; no fitted correction','cases':[one(c) for c in cs], 'sha256':{}}
    for path in ('scripts/diagnose_precession_carrier_inversion.py','src/torchwave/approximants/precession_twisting_clean.py','src/torchwave/approximants/precession_source_frame_clean.py'):
        out['sha256'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dst=Path('docs/reviews/2026-09-13-parity-review/precession-carrier-inversion.json'); dst.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
