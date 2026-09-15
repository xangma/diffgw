#!/usr/bin/env python3
"""Clean-room ROM glue-node experiment (Purrer 1512.02248 IV.1).

The only changed quantity between variants is the documented phase fit-node
set.  The binary comparison uses the public SEOBNRv4_ROM API and its public
QNM frequency, with masses converted to Mf using total detector-frame mass.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, os
from pathlib import Path
import lal, lalsimulation as ls, numpy as np, torch
from torchwave.approximants.rom.bspline import evaluate_stitched_frequency, load_official_rom

MSUN_SEC = 4.925490947641267e-6

def sha256(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def public_reference(case):
    m1,m2,c1,c2 = case
    hp,hc = ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,c1,0,0,c2,
        500*lal.PC_SI*1e6,0,0,0,0,0,1,20,512,0,None,
        ls.GetApproximantFromString("SEOBNRv4_ROM"))
    f=np.arange(hp.data.length)*hp.deltaF
    return f,hp.data.data.copy(),hc.data.data.copy()

def public_sequence(case, frequencies):
    """Public sequence comparison at genuinely non-grid frequencies.

    The installed ROM sequence binding returns zero-valued series for this
    asset, so the public ChooseFD API is used with ``deltaF=.01`` to obtain
    exact requested bins; this is recorded explicitly in the artifact.
    """
    m1,m2,c1,c2=case; df=.01; fmax=max(frequencies)+df
    hp=ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,c1,0,0,c2,
        500*lal.PC_SI*1e6,0,0,0,0,0,df,20.,fmax,20.,None,
        ls.GetApproximantFromString("SEOBNRv4_ROM"))[0]
    idx=np.rint(np.asarray(frequencies)/df).astype(int)
    if not np.allclose(idx*df,frequencies,atol=1e-12): raise ValueError("sample frequencies are not exact .01-Hz bins")
    return np.asarray(hp.data.data,dtype=np.complex128)[idx]

def qnm_mf(case):
    m1,m2,c1,c2=case; q=lal.CreateCOMPLEX16Vector(1)
    ls.SimIMREOBGenerateQNMFreqV2(q,m1,m2,[0,0,c1],[0,0,c2],2,2,1,ls.GetApproximantFromString("SEOBNRv4"))
    return float(q.data[0].real*(m1+m2)*MSUN_SEC/(2*np.pi))

def metrics(z,r,mf,band):
    valid=np.isfinite(z)&np.isfinite(r)&(np.abs(z)>0)&(np.abs(r)>0)
    if not np.any(valid): return {"count":0,"support_mismatch":int(np.size(valid))}
    x=mf[valid]; order=np.argsort(x); x=x[order]
    ph=np.unwrap(np.angle(z[valid][order]*np.conj(r[valid][order])))
    # phase intercept is removed only at the declared 20 Hz reference when present.
    ph-=ph[np.argmin(np.abs(x-band[2]))]
    resid=ph-np.polyval(np.polyfit(x,ph,1),x) if len(x)>1 else ph*0
    sel=(x>=band[0])&(x<band[1])
    if not np.any(sel): return {"count":0,"support_mismatch":int(np.sum(~valid))}
    zz=z[valid][order][sel]; rr=r[valid][order][sel]; pp=ph[sel]; qq=resid[sel]
    return {"count":int(sel.sum()),"support_mismatch":int(np.sum(~valid)),
      "amplitude_relative_rms":float(np.linalg.norm(np.abs(zz)-np.abs(rr))/np.linalg.norm(np.abs(rr))),
      "complex_relative_rms":float(np.linalg.norm(zz-rr)/np.linalg.norm(rr)),
      "phase_rms":float(np.sqrt(np.mean(pp*pp))),
      "phase_affine_removed_rms":float(np.sqrt(np.mean(qq*qq))),
      "phase_slope_rad_per_Mf":float(np.polyfit(x[sel],pp,1)[0]) if np.sum(sel)>1 else 0.0}

def variant_nodes(high):
    n=high.mf_grid_phase.detach().cpu().numpy(); inx=n[(n>=.0074)&(n<=.014)]
    near=inx[np.argsort(np.abs(inx-.01))[:15]]; near.sort()
    below=inx[inx<.01]; above=inx[inx>=.01]
    split=np.concatenate((below[np.argsort(np.abs(below-.01))[:15]],above[np.argsort(np.abs(above-.01))[:15]])); split.sort()
    # Two 31-node interpretations of “15 below + 15 above”, retaining the
    # central tabulated knot closest to .01 or the first knot above it.
    ia=int(np.searchsorted(inx,.01)); ic=ia-1
    return {"all66_interval":inx,"15_nearest_total":near,"15_below_plus_15_above":split,
            "31_centered_closest_to_0p01":inx[ic-15:ic+16],
            "31_centered_first_above_0p01":inx[ia-15:ia+16]}

def run_case(case,low,high):
    f,rp,rc=public_reference(case); mass=sum(case[:2]); mf=f*mass*MSUN_SEC; rd=qnm_mf(case)
    ok=(mf>=float(low.mf_grid_amp[0]))&(mf<=float(high.mf_grid_amp[-1])); mf0=mf[ok]
    scale=mass*MSUN_SEC*(lal.C_SI*mass*MSUN_SEC)/(500*lal.PC_SI*1e6)*.5
    eta=case[0]*case[1]/mass**2; ref=rp[ok]; nodes=variant_nodes(high); out={}
    for name,sel in nodes.items():
        amp,phase=evaluate_stitched_frequency(low,high,torch.as_tensor(mf0,dtype=torch.float64),eta,case[2],case[3],transition=.01,overlap=(.0074,.014),ringdown_frequency=rd,reference_mf=20*mass*MSUN_SEC,fit_nodes=torch.as_tensor(sel,dtype=torch.float64))
        z=torch.polar(amp,phase).detach().numpy()*scale
        out[name]={"fit_node_count":int(len(sel)),"fit_nodes_Mf":sel.tolist(),"metrics":{"all":metrics(z,ref,mf0,(mf0[0],mf0[-1]+1e-15,20*mass*MSUN_SEC)),"low":metrics(z,ref,mf0,(mf0[0],.009,20*mass*MSUN_SEC)),"high":metrics(z,ref,mf0,(.011,mf0[-1]+1e-15,20*mass*MSUN_SEC))}}
        samples=[]; sample_hz=(23.17,37.61,82.43,173.29)
        seq=public_sequence(case,np.asarray(sample_hz)); mf_s=np.asarray(sample_hz)*mass*MSUN_SEC
        aa,pp=evaluate_stitched_frequency(low,high,torch.as_tensor(mf_s,dtype=torch.float64),eta,case[2],case[3],transition=.01,overlap=(.0074,.014),ringdown_frequency=rd,reference_mf=20*mass*MSUN_SEC,fit_nodes=torch.as_tensor(sel,dtype=torch.float64))
        zz=torch.polar(aa,pp).detach().numpy()*scale
        for hz,zn,rn in zip(sample_hz,zz,seq):
            samples.append({"frequency_hz":hz,"complex_relative":float(abs(zn-rn)/max(abs(rn),np.finfo(float).tiny)),"amplitude_relative":float(abs(abs(zn)-abs(rn))/max(abs(rn),np.finfo(float).tiny))})
        out[name]["independent_frequency_samples_hz"]=samples
    return {"case":{"m1_msun":case[0],"m2_msun":case[1],"chi1":case[2],"chi2":case[3],"eta":eta,"mass_conversion":"Mf=f_Hz*(m1+m2)*MSUN_SEC","qnm_Mf":rd},"variants":out}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--rom",type=Path,default=Path("/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5")); p.add_argument("--output",type=Path,default=Path("scripts/diagnose_rom_glue_nodes.json")); a=p.parse_args()
    os.environ["LAL_DATA_PATH"] = str(a.rom.parent)
    low,high=load_official_rom(a.rom,"sub1"),load_official_rom(a.rom,"sub2")
    cases=[(50.,25.,.2,.1),(35.,35.,-.4,.3),(80.,20.,.6,-.2),(42.,18.,-.3,.45)]
    native=[Path("src/torchwave/approximants/rom/bspline.py"),Path("src/torchwave/approximants/rom/official.py")]
    result={"method":"Purrer arXiv:1512.02248 IV.1; node variants are diagnostic interpretations, not fitted runtime choices","references":["https://arxiv.org/abs/1512.02248","https://arxiv.org/abs/1611.03703"],"qnm_contract":"SimIMREOBGenerateQNMFreqV2 real angular frequency; divide 2*pi, then multiply total mass in seconds to obtain Mf","independent_sampling":{"requested_hz":[23.17,37.61,82.43,173.29],"backend":"public SimInspiralChooseFDWaveform SEOBNRv4_ROM at exact 0.01-Hz bins; the installed SimIMRSEOBNRv4ROMFrequencySequence binding returned zero-valued series for this asset and was therefore not used for numerical samples"},"source":{"rom":str(a.rom),"rom_sha256_before":sha256(a.rom),"native_hashes_before":{str(x):sha256(x) for x in native},"script_sha256_before":sha256(Path(__file__)) ,"versions":{"python":sys.version,"torch":torch.__version__,"lal":getattr(lal,"__version__","unknown"),"lalsimulation":getattr(ls,"__version__","unknown")}},"cases":[run_case(c,low,high) for c in cases]}
    result["source"].update({"rom_sha256_after":sha256(a.rom),"native_hashes_after":{str(x):sha256(x) for x in native},"script_sha256_after":sha256(Path(__file__))})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
