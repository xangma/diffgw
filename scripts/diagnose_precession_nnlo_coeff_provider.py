"""Process-local published NNLO coefficient-provider comparison.

The descriptor is patched only for this process and restored in ``finally``.
The public reference is SimInspiralChooseFDWaveform with explicit XP version 102.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch
from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.imrphenomp import IMRPhenomP
import torchwave.approximants.precession_msa as msa
import torchwave.approximants.imrphenom_xp as xpmod
from torchwave.constants import MSUN_SEC
from torchwave.approximants.precession_beta_clean import beta
from torchwave.approximants.precession_nnlo_clean import coefficients
from parity_harness import get_canonical_catalog_cases,get_canonical_holdout_cases,direct_complex_relative_l2,direct_magnitude_relative_l2
from diagnose_precession_nnlo_paired import public_fd

CALLS=[]
PROVIDER_MODEL='XP'
ORIG_MSA=None; ORIG_XP=None

def beta_adapter(v,v0,**p):
    version=p.pop('prec_version',102); convention=p.pop('convention',1)
    p.pop('return_carrier_phase',None)
    native=ORIG_MSA(v,v0,**p,prec_version=version,convention=convention,return_carrier_phase=True)
    M=p['m1']+p['m2']; f=v**3/(torch.pi*M*MSUN_SEC)
    bb=beta(p['m1'],p['m2'],f,p['s1x'],p['s1y'],p['s1z'],p['s2x'],p['s2y'],p['s2z'],compat_102=True)
    return (native[0],bb,native[2],native[3],native[4],native[5])
def provider(eta,delta,m1_norm,chip,chi_l):
    CALLS.append({'eta_shape':list(eta.shape),'delta_shape':list(delta.shape)})
    if PROVIDER_MODEL == 'Pv2':
        # PhenomPv2 technical note T1500602 Eq. (32): chi_l is the
        # dimensionless aligned spin of the larger black hole (m1).
        chi_for_series=chi_l
    else:
        chi_for_series=chi_l/m1_norm
    aa,ee=coefficients(eta,-delta,m1_norm,chi_for_series,chip)
    return tuple(aa.unbind(-1))+tuple(ee.unbind(-1))

def evaluate(c, patched, beta_patch=False):
    global PROVIDER_MODEL, ORIG_MSA, ORIG_XP
    PROVIDER_MODEL='Pv2' if c['torch_name']=='IMRPhenomPv2' else 'XP'
    f,refp,refc=public_fd(c,0); g=c['grid']; kw=dict(c['params']); kw.pop('f_lower',None); kw.pop('f_final',None)
    if c['torch_name']=='IMRPhenomXP': kw.update(prec_version=102,convention=1)
    old=IMRPhenomP.__dict__['nnlo_angle_coefficients']; old_l=IMRPhenomP.__dict__['orbital_angular_momentum']
    ORIG_MSA=msa.evaluate_msa_precession; ORIG_XP=xpmod.evaluate_msa_precession
    if patched: IMRPhenomP.nnlo_angle_coefficients=staticmethod(provider)
    if beta_patch:
        msa.evaluate_msa_precession=beta_adapter; xpmod.evaluate_msa_precession=beta_adapter
    try:
        hp,hc=get_fd_waveform(c['torch_name'],sample_frequencies=f,f_lower=float(g['flow']),f_final=float(g['fhigh']),dtype=torch.float64,**kw)
    finally:
        IMRPhenomP.nnlo_angle_coefficients=old; IMRPhenomP.orbital_angular_momentum=old_l
        msa.evaluate_msa_precession=ORIG_MSA; xpmod.evaluate_msa_precession=ORIG_XP
    sel=(f>=float(g['flow']))&(f<float(g['fhigh']))
    return {'plus_raw_l2':direct_complex_relative_l2(hp[sel],refp[sel]),'cross_raw_l2':direct_complex_relative_l2(hc[sel],refc[sel]),'plus_mag_l2':direct_magnitude_relative_l2(hp[sel],refp[sel]),'cross_mag_l2':direct_magnitude_relative_l2(hc[sel],refc[sel]),'n':int(sel.sum())}

def selected():
    cat=get_canonical_catalog_cases();hold=get_canonical_holdout_cases();out=[]
    for name in ('IMRPhenomXP','IMRPhenomPv2'):
        out += [next(c for c in cat if c['torch_name']==name),next(c for c in hold if c['torch_name']==name)]
    import copy
    xp=next(c for c in cat if c['torch_name']=='IMRPhenomXP')
    for suffix,m1,m2,z1,z2 in [('independent_a',28.,11.,.35,-.25),('independent_b',61.,19.,-.2,.3)]:
        a=copy.deepcopy(xp); a['id']='IMRPhenomXP_'+suffix
        a['params'].update(mass1=m1,mass2=m2,spin1z=z1,spin2z=z2,chi1=z1,chi2=z2)
        out.append(a)
    # Pv2 aligned-in-plane controls isolate the aligned-spin coefficient path.
    for c in list(out):
        if c['torch_name']=='IMRPhenomPv2':
            a=copy.deepcopy(c); a['id']=c['id']+'_aligned_inplane_zero'
            for k in ('spin1x','spin1y','spin2x','spin2y'): a['params'][k]=0.
            out.append(a)
    return out

def main():
    out={'public_api':'SimInspiralChooseFDWaveform; XP PrecVersion=102, Convention=1','provider':'XP: chi_l/m1_norm; Pv2: chi1z (larger-BH spin) per T1500602 Eq.32; coefficients(eta,-delta,m1_norm,chi,chip)','cases':[],'versions':{'python':sys.version,'torch':torch.__version__},'source_sha256':{}}
    for p in ('src/torchwave/approximants/precession_nnlo_clean.py','scripts/diagnose_precession_nnlo_coeff_provider.py'):
        out['source_sha256'][p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for c in selected():
        CALLS.clear(); b=evaluate(c,False); coeff=evaluate(c,True); q=evaluate(c,True,beta_patch=True)
        row={'id':c['id'],'torch_name':c['torch_name'],'baseline':b,'coeff_only':coeff,'coeff_plus_beta':q};
        if c['torch_name']=='IMRPhenomXP': row['beta_only']=evaluate(c,False,beta_patch=True)
        row['provider_calls']=list(CALLS); out['cases'].append(row); print(row,flush=True)
    Path('docs/reviews/2026-09-13-parity-review/precession-nnlo-coeff-provider.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
