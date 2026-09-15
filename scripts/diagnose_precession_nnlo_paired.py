"""Paired public XP/Pv2 waveform comparison for the clean NNLO angle adapter.

The binary side is the public LALSimulation generic FD API; no implementation
source is inspected.  The adapter is process-local and restored in finally.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import lal, lalsimulation as ls
import torch
from parity_harness import (get_canonical_catalog_cases,
    get_canonical_holdout_cases, direct_complex_relative_l2,
    direct_magnitude_relative_l2)
from torchwave.waveforms import get_fd_waveform
from torchwave.approximants import precession_msa as msa
from torchwave.approximants import imrphenom_xp as xpmod
import diagnose_precession_nnlo_waveform as clean_wave
adapter=clean_wave.adapter

PYTHON = sys.executable

def public_fd(case, nfreq):
    p, g = case['params'], case['grid']
    df, flow, fhigh = float(g['df']), float(g['flow']), float(g['fhigh'])
    extra = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(extra, int(p.get('prec_version', 102)))
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(extra, int(p.get('convention', 1)))
    if p.get('final_spin_mod') is not None:
        ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(extra, int(p['final_spin_mod']))
    assert ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(extra) == int(p.get('prec_version', 102))
    assert ls.SimInspiralWaveformParamsLookupPhenomXPConvention(extra) == int(p.get('convention', 1))
    if p.get('final_spin_mod') is not None:
        assert ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(extra) == int(p['final_spin_mod'])
    approx = ls.GetApproximantFromString(case['lal_name'])
    hp, hc = ls.SimInspiralChooseFDWaveform(
        float(p['mass1'])*lal.MSUN_SI, float(p['mass2'])*lal.MSUN_SI,
        float(p.get('spin1x',0)), float(p.get('spin1y',0)), float(p.get('spin1z',p.get('chi1',0))),
        float(p.get('spin2x',0)), float(p.get('spin2y',0)), float(p.get('spin2z',p.get('chi2',0))),
        float(p.get('distance',100))*1e6*lal.PC_SI, float(p.get('inclination',0)),
        float(p.get('phic',0)), 0., float(p.get('e0',0)), 0., df, flow,
        float(g.get('oracle_fhigh', fhigh)), float(p.get('f_ref',flow)), extra, approx)
    n = len(hp.data.data)
    return torch.arange(n,dtype=torch.float64)*df, torch.tensor(hp.data.data,dtype=torch.complex128), torch.tensor(hc.data.data,dtype=torch.complex128)

def torch_fd(case, patched=False):
    p, g = case['params'], case['grid']
    freqs, refp, refc = public_fd(case, 0)
    kw = dict(p); kw.pop('f_lower',None); kw.pop('f_final',None)
    if case['torch_name'] == 'IMRPhenomXP':
        kw['prec_version'] = int(p.get('prec_version', 102))
        kw['convention'] = int(p.get('convention', 1))
        if p.get('final_spin_mod') is not None:
            kw['final_spin_mod'] = int(p['final_spin_mod'])
    old1, old2 = msa.evaluate_msa_precession, xpmod.evaluate_msa_precession
    clean_wave.ORIG = old1
    if patched:
        msa.evaluate_msa_precession = adapter
        xpmod.evaluate_msa_precession = adapter
    try:
        hp,hc = get_fd_waveform(case['torch_name'], sample_frequencies=freqs,
            f_lower=float(g['flow']), f_final=float(g['fhigh']), dtype=torch.float64, **kw)
    finally:
        msa.evaluate_msa_precession, xpmod.evaluate_msa_precession = old1, old2
    sel=(freqs>=float(g['flow']))&(freqs<float(g['fhigh']))
    return {"plus_raw_l2":direct_complex_relative_l2(hp[sel],refp[sel]),
            "cross_raw_l2":direct_complex_relative_l2(hc[sel],refc[sel]),
            "plus_mag_l2":direct_magnitude_relative_l2(hp[sel],refp[sel]),
            "cross_mag_l2":direct_magnitude_relative_l2(hc[sel],refc[sel]),
            "n":int(sel.sum())}

def cases():
    allc=get_canonical_catalog_cases(); hold=get_canonical_holdout_cases()
    chosen=[]
    for name in ('IMRPhenomXP','IMRPhenomPv2'):
        chosen += [next(c for c in allc if c['torch_name']==name)]
        chosen += [next(c for c in hold if c['torch_name']==name)]
    # Two independent XP probes, separate from frozen catalog/holdout cases.
    for i, vals in enumerate(((40.,20.,.3,.2,.4,-.2,.1,-.2,.7,40.),
                              (30.,10.,.1,-.25,-.4,.15,.05,.2,.5,40.))):
        m1,m2,x1,y1,z1,x2,y2,z2,inc,ref=vals
        chosen.append({'id':f'IMRPhenomXP_independent_{i}', 'torch_name':'IMRPhenomXP', 'lal_name':'IMRPhenomXP',
            'params':dict(mass1=m1,mass2=m2,spin1x=x1,spin1y=y1,spin1z=z1,spin2x=x2,spin2y=y2,spin2z=z2,
                          inclination=inc,phic=.0,f_ref=ref,distance=100.,prec_version=102,convention=1),
            'grid':{'flow':25.,'fhigh':160.,'df':.25}})
    return chosen

def main():
    out={'python':PYTHON,'platform':platform.platform(),'lal_version':getattr(lal,'__version__','unknown'),
         'lalsimulation_version':getattr(ls,'__version__','unknown'),'prec_version':102,
         'adapter_calls':clean_wave.ADAPTER_CALL_LOG,
         'adapter':'clean NNLO; delta=(m2-m1)/M, chi_l=(m1*s1z+m2*s2z)/m1; gamma=-epsilon; anchors native; process-local',
         'source_sha256':{},'cases':[]}
    for path in ('src/torchwave/approximants/precession_nnlo_clean.py','scripts/diagnose_precession_nnlo_paired.py','scripts/diagnose_precession_nnlo_waveform.py','tests/parity_harness.py'):
        out['source_sha256'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for c in cases():
        clean_wave.ADAPTER_CALL_LOG.clear()
        row={'id':c['id'],'torch_name':c['torch_name'],'lal_name':c['lal_name'],'params':c['params'],'grid':c['grid']}
        row['baseline']=torch_fd(c,False); row['clean_substitution']=torch_fd(c,True)
        row['adapter_calls']=list(clean_wave.ADAPTER_CALL_LOG)
        out['cases'].append(row); print(row['id'],row['baseline'],row['clean_substitution'],flush=True)
    dest=Path('docs/reviews/2026-09-13-parity-review/precession-nnlo-paired.json'); dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(dest)
if __name__=='__main__': main()
