"""Black-box NSBH tidal-ratio test against published NRTidalv2 Eq. 18."""
import hashlib, json, pathlib, platform, sys
import lal, lalsimulation as ls, numpy as np, torch
from torchwave import get_fd_waveform
from torchwave.approximants import imrphenom_nsbh
from torchwave.approximants.nsbh_clean_phase import nrtidalv2_frequency_phase, nrtidalv2_frequency_phase_derivative

ROOT = pathlib.Path(__file__).resolve().parents[1]
MSUN_S = lal.G_SI * lal.MSUN_SI / lal.C_SI**3
CAPTURED = []
CAPTURE_LABEL = None
OFFSET_MODE = "original"
_ORIGINAL_OFFSETS = getattr(imrphenom_nsbh, "_legacy_phase_offsets", None)

def _offset_probe(*args, **kwargs):
    args = list(args)
    if OFFSET_MODE != "original" and CAPTURE_LABEL and CAPTURE_LABEL.get("lambda2",0.) != 0.:
        key = tuple((k, CAPTURE_LABEL[k]) for k in ("case","f_final","f_ref"))
        base = next((x for x in CAPTURED if x["label"] and tuple((k,x["label"][k]) for k in ("case","f_final","f_ref"))==key and x["label"].get("lambda2")==0.), None)
        if base:
            if OFFSET_MODE in ("freeze_mass","freeze_mass_spin"):
                args[6] = torch.tensor(base["args"][6], dtype=args[6].dtype)
            if OFFSET_MODE == "freeze_mass_spin":
                args[7] = torch.tensor(base["args"][7], dtype=args[7].dtype)
    out = _ORIGINAL_OFFSETS(*args, **kwargs)
    def conv(x):
        if isinstance(x, (tuple,list)): return [conv(y) for y in x]
        if isinstance(x, torch.Tensor) and x.numel() == 1: return float(x.detach().cpu().item())
        return float(x) if np.isscalar(x) else repr(x)
    CAPTURED.append({"label": CAPTURE_LABEL, "args": [conv(x) for x in args], "output": conv(out)})
    return out

def native(case, f, lo, final, ref, lam):
    m1,m2,chi,inc,phic=case
    h,_=get_fd_waveform("IMRPhenomNSBH", mass1=m1,mass2=m2,chi1=chi,chi2=0.,lambda1=0.,lambda2=lam,
        distance=100.,inclination=inc,phic=phic,f_ref=ref,f_lower=lo,f_final=final,
        sample_frequencies=torch.tensor(f,dtype=torch.float64),dtype=torch.float64)
    return h.detach().numpy()

def binary(case, lo, final, df, ref, lam):
    m1,m2,chi,inc,phic=case; d=lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertTidalLambda1(d,0.); ls.SimInspiralWaveformParamsInsertTidalLambda2(d,lam)
    a=[m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,chi,0.,0.,0.,100e6*lal.PC_SI,inc,phic,0.,0.,0.,df,lo,final,ref,d,ls.IMRPhenomNSBH]
    h,_=ls.SimInspiralChooseFDWaveform(*a); return np.asarray(h.data.data,dtype=np.complex128)

def lamtilde(m1,m2,l2):
    M=m1+m2
    return 16./13.*((m2+12*m1)*m2**4*l2)/M**5

def fit(a,b,f):
    ok=(f>=30)&np.isfinite(a)&np.isfinite(b)&(abs(a)>0)&(abs(b)>0)
    ph=np.unwrap(np.angle(a[ok]/b[ok])); ff=f[ok]; c=np.linalg.lstsq(np.c_[np.ones(len(ff)),ff],ph,rcond=None)[0]
    r=ph-np.c_[np.ones(len(ff)),ff]@c; scale=np.sqrt(np.mean(abs(b[ok])**2))
    ua=a[ok]/abs(a[ok]); ub=b[ok]/abs(b[ok])
    return dict(complex_l2=float(np.sqrt(np.mean(abs(a[ok]-b[ok])**2))/scale),
        unit_phase_complex_l2=float(np.sqrt(np.mean(abs(ua-ub)**2))),
        max_abs_normalized=float(max(abs(a[ok]-b[ok]))/scale),
        affine_intercept_rad=float(c[0]),affine_slope_rad_per_hz=float(c[1]),nonaffine_phase_rms_rad=float(np.sqrt(np.mean(r*r))))

def main(out):
    if _ORIGINAL_OFFSETS is not None:
        setattr(imrphenom_nsbh, "_legacy_phase_offsets", _offset_probe)
    cases=[("catalog",(5.,1.4,.2,.5,.3),400.,(30.,27.,60.)),("holdout",(7.,1.4,.4,.9,.3),2600.,(40.,27.,60.))]
    rec=[]
    for name,case,lam,refs in cases:
      m1,m2,*_=case; M=(m1+m2)*MSUN_S; eta=m1*m2/(m1+m2)**2; lt=lamtilde(m1,m2,lam)
      endpoint_ratios={}
      for final in (512.,1024.,2048.):
       f=np.arange(int(final/.25)+1)*.25; mask=f>=30
       td=lambda x: torch.tensor(x,dtype=torch.float64)
       psi=nrtidalv2_frequency_phase(td(f[mask]),td(M),td(eta),td(lt)).numpy()
       dpsi=nrtidalv2_frequency_phase_derivative(td(f[mask]),td(M),td(eta),td(lt)).numpy()
       for ref in refs:
        for backend in ("native","binary"):
         global CAPTURE_LABEL
         CAPTURE_LABEL=dict(case=name,backend=backend,f_final=final,f_ref=ref,lambda2=0.)
         h0=native(case,f,30,final,ref,0.) if backend=="native" else binary(case,30,final,.25,ref,0.)
         modes=("original","freeze_mass","freeze_mass_spin") if backend=="native" else ("original",)
         for mode in modes:
          global OFFSET_MODE
          OFFSET_MODE=mode
          CAPTURE_LABEL=dict(case=name,backend=backend,f_final=final,f_ref=ref,lambda2=lam,mode=mode)
          hl=native(case,f,30,final,ref,lam) if backend=="native" else binary(case,30,final,.25,ref,lam)
          ratio=np.full_like(hl,np.nan); np.divide(hl,h0,out=ratio,where=abs(h0)>0)
          if mode=="original": endpoint_ratios[(backend,ref,final)] = (f, ratio)
          for sign in (1.,-1.):
           target=np.exp(1j*sign*psi)
           rec.append(dict(case=name,backend=backend,mode=mode,f_final=final,f_ref=ref,sign=sign,lambda_tilde=lt,
              derivative_mean=float(np.mean(dpsi)),metrics=fit(ratio[mask],target,f[mask])))
         OFFSET_MODE="original"
      for backend in ("native","binary"):
       for ref in refs:
        f512,r512=endpoint_ratios[(backend,ref,512.)]; f1024,r1024=endpoint_ratios[(backend,ref,1024.)]
        mask=(f512>=30)&(f512<512)&(abs(r512)>0)&(abs(r1024[:len(r512)])>0)
        # Eq.18 is endpoint independent, so its predicted endpoint ratio is 1.
        rec.append(dict(case=name,backend=backend,f_final="512_vs_1024",f_ref=ref,lambda_tilde=lt,
          derivative_difference_prediction="zero (same Eq.18 phase at both endpoints)",
          metrics=fit(r512[mask],r1024[:len(r512)][mask],f512[mask])))
    # Keep the native callable black-boxed: inspect signature, probe calls,
    # and restore the descriptor before returning.
    anchor_records=[]
    if _ORIGINAL_OFFSETS is not None:
        setattr(imrphenom_nsbh, "_legacy_phase_offsets", _ORIGINAL_OFFSETS)
        # Native calls are labelled above; expose actual Mf_ref/Mf_anchor and
        # derivative differences without inspecting the callable body.
        for name,case,lam,refs in cases:
            m1,m2,*_=case; M=(m1+m2)*MSUN_S; eta=m1*m2/(m1+m2)**2; lt=lamtilde(m1,m2,lam)
            for ref in refs:
                for f1,f2 in ((512.,1024.),(1024.,2048.)):
                    rows=[]
                    for ff in (f1,f2):
                        hit=next((x for x in CAPTURED if x["label"]=={"case":name,"backend":"native","f_final":ff,"f_ref":ref,"lambda2":0.}),None)
                        if hit: rows.append((ff,hit["args"][4],hit["args"][5],hit["args"][0]+hit["args"][1]))
                    if len(rows)==2:
                        def deriv(hz): return float(nrtidalv2_frequency_phase_derivative(torch.tensor(hz,dtype=torch.float64),torch.tensor(M,dtype=torch.float64),torch.tensor(eta,dtype=torch.float64),torch.tensor(lt,dtype=torch.float64)).item())
                        candidates={"endpoint_hz":(f1,f2),"last_sample_hz":(f1,f2),
                                    "native_mf_anchor":(rows[0][2]/(rows[0][3]*MSUN_S),rows[1][2]/(rows[1][3]*MSUN_S))}
                        anchor_records.append({"case":name,"f_ref":ref,"endpoints":(f1,f2),"captured_mf_ref":(rows[0][1],rows[1][1]),
                          "captured_mf_anchor":(rows[0][2],rows[1][2]),"anchor_hz":candidates,
                          "derivative_difference_rad_per_hz":{k:deriv(v[0])-deriv(v[1]) for k,v in candidates.items()}})
    meta={"python":sys.executable,"platform":platform.platform(),"lal":lal.__version__,"lalsimulation":ls.__version__,
      "legacy_phase_offsets_signature": str(__import__('inspect').signature(_ORIGINAL_OFFSETS)) if _ORIGINAL_OFFSETS else None,
      "legacy_phase_offsets_calls": CAPTURED,"anchor_derivative_records":anchor_records,
      "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/nsbh_tidal_derivative_diagnostic.py',ROOT/'src/torchwave/approximants/nsbh_clean_phase.py']}}
    pathlib.Path(out).write_text(json.dumps({"metadata":meta,"records":rec},indent=2)+"\n")
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "docs/nsbh_tidal_derivative_diagnostic.json")
