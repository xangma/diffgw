"""Black-box NSBH remnant-offset acceptance experiment."""
import hashlib, inspect, json, pathlib, platform, sys
import lal, lalsimulation as ls, numpy as np, torch
from torchwave import get_fd_waveform
from torchwave.approximants import imrphenom_nsbh, imrphenomd

ROOT=pathlib.Path(__file__).resolve().parents[1]; MSUN_S=lal.G_SI*lal.MSUN_SI/lal.C_SI**3
HASH_BEFORE={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ("scripts/nsbh_remnant_offset_diagnostic.py","src/torchwave/approximants/imrphenom_nsbh.py")}
ORIG=getattr(imrphenom_nsbh,"_legacy_phase_offsets",None); CAP=[]; LABEL=None; MODE="baseline"
def probe(*args,**kw):
    a=list(args); key=(LABEL["case"],LABEL["final"],LABEL["ref"])
    base=next((x for x in CAP if x["key"]==key and x["lambda2"]==0),None)
    if MODE in ("freeze_mass","freeze_mass_spin","freeze_phenomd_mass") and LABEL["lambda2"] and base:
        if MODE == "freeze_phenomd_mass":
            m1,m2,s1,s2=[v.detach().clone().to(dtype=torch.float64) if isinstance(v,torch.Tensor) else torch.tensor(v,dtype=torch.float64) for v in (a[0],a[1],a[2],a[3])]
            a[6]=imrphenomd.IMRPhenomD().remnant_properties(m1,m2,s1,s2)[0]
        else:
            a[6]=torch.tensor(base["args"][6],dtype=a[6].dtype)
        if MODE=="freeze_mass_spin": a[7]=torch.tensor(base["args"][7],dtype=a[7].dtype)
    out=ORIG(*a,**kw)
    def cv(x):
        if isinstance(x,torch.Tensor) and x.numel()==1:return float(x.detach().cpu().item())
        if isinstance(x,(tuple,list)):return [cv(y) for y in x]
        return float(x) if np.isscalar(x) else repr(x)
    CAP.append({"key":key,"lambda2":LABEL["lambda2"],"mode":MODE,"args":[cv(x) for x in a],"output":cv(out)})
    return out
def native(case,f,lo,final,ref,lam):
    m1,m2,chi,inc,ph=case
    return get_fd_waveform("IMRPhenomNSBH",mass1=m1,mass2=m2,chi1=chi,chi2=0.,lambda1=0.,lambda2=lam,distance=100.,inclination=inc,phic=ph,f_ref=ref,f_lower=lo,f_final=final,sample_frequencies=torch.tensor(f,dtype=torch.float64),dtype=torch.float64)
def binary(case,lo,final,df,ref,lam):
    m1,m2,chi,inc,ph=case; d=lal.CreateDict(); ls.SimInspiralWaveformParamsInsertTidalLambda1(d,0.); ls.SimInspiralWaveformParamsInsertTidalLambda2(d,lam)
    a=[m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,chi,0.,0.,0.,100e6*lal.PC_SI,inc,ph,0.,0.,0.,df,lo,final,ref,d,ls.IMRPhenomNSBH]; hp,hc=ls.SimInspiralChooseFDWaveform(*a)
    return np.asarray(hp.data.data,dtype=np.complex128),np.asarray(hc.data.data,dtype=np.complex128)
def metrics(a,b,f,lo):
    finite=(f>=lo)&np.isfinite(a)&np.isfinite(b); scale=np.sqrt(np.mean(abs(b[finite])**2)); raw=float(np.sqrt(np.mean(abs(a[finite]-b[finite])**2))/scale); mag=float(np.sqrt(np.mean((abs(a[finite])-abs(b[finite]))**2))/scale)
    ok=finite&(abs(a)>0)&(abs(b)>0); ff=f[ok]; ph=np.unwrap(np.angle(a[ok]/b[ok])); c=np.linalg.lstsq(np.c_[np.ones(len(ff)),ff],ph,rcond=None)[0]
    return {"support_mismatch_bins":int(np.count_nonzero(finite & ((abs(a)==0)!=(abs(b)==0)))),"raw_relative_l2":raw,"magnitude_relative_l2":mag,"unit_phase_l2":float(np.sqrt(np.mean(abs(a[ok]/abs(a[ok])-b[ok]/abs(b[ok]))**2))),"affine_slope_rad_per_hz":float(c[1])}
def main(out):
    global LABEL,MODE
    cases=[("catalog",(5.,1.4,.2,.5,.3),400.,30.,30.),("holdout",(7.,1.4,.4,.9,.3),2600.,30.,40.)]; rec=[]
    try:
      if ORIG is None: raise RuntimeError("_legacy_phase_offsets unavailable")
      setattr(imrphenom_nsbh,"_legacy_phase_offsets",probe)
      for name,case,lam,lo,ref in cases:
       for final in (512.,1024.):
        f=np.arange(int(final/.25)+1)*.25
        for rref in (ref,27.,60.):
         LABEL={"case":name,"final":final,"ref":rref,"lambda2":0}; MODE="baseline"; n0=native(case,f,lo,final,rref,0.); b0=binary(case,lo,final,.25,rref,0.)
         for mode in ("baseline","freeze_mass","freeze_mass_spin","freeze_phenomd_mass"):
          MODE=mode; LABEL={"case":name,"final":final,"ref":rref,"lambda2":lam}; n=native(case,f,lo,final,rref,lam); bl=binary(case,lo,final,.25,rref,lam)
          rec.append({"case":name,"f_final":final,"f_ref":rref,"lambda2":lam,"mode":mode,"matching_length":len(n[0])==len(bl[0])==len(f),"baseline_native_binary":{"hp":metrics(n0[0].detach().numpy(),b0[0],f,lo),"hc":metrics(n0[1].detach().numpy(),b0[1],f,lo)},"restored_native_binary":{"hp":metrics(n[0].detach().numpy(),bl[0],f,lo),"hc":metrics(n[1].detach().numpy(),bl[1],f,lo)}})
    finally:
      if ORIG is not None:setattr(imrphenom_nsbh,"_legacy_phase_offsets",ORIG)
    predictions=[]
    for x in CAP:
      if x["lambda2"] and x["mode"] in ("freeze_mass","freeze_mass_spin"):
       base=next((y for y in CAP if y["key"]==x["key"] and y["lambda2"]==0),None)
       if base:
        M=(base["args"][0]+base["args"][1])*MSUN_S; delta=x["output"][1]-next(y for y in CAP if y["key"]==x["key"] and y["lambda2"]==x["lambda2"] and y["mode"]=="baseline")["output"][1]
        predictions.append({"key":x["key"],"mode":x["mode"],"delta_second_output":delta,"predicted_slope_minus_delta_M":-delta*M,"predicted_slope_plus_delta_M":delta*M})
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in HASH_BEFORE}
    inventory={}
    inventory["IMRPhenomD.remnant_properties"]=str(inspect.signature(imrphenomd.IMRPhenomD.remnant_properties))
    inventory["IMRPhenomD.remnant_properties_doc"]=imrphenomd.IMRPhenomD.remnant_properties.__doc__
    for n in dir(imrphenomd):
      if "mass" in n.lower() or "remnant" in n.lower() or "final" in n.lower():
       try: inventory[n]=str(inspect.signature(getattr(imrphenomd,n)))
       except (TypeError,ValueError): inventory[n]=type(getattr(imrphenomd,n)).__name__
    pathlib.Path(out).write_text(json.dumps({"metadata":{"python":sys.executable,"platform":platform.platform(),"legacy_phase_offsets_signature":str(inspect.signature(ORIG)),"phenomd_remnant_callable_inventory":inventory,"phenoD_final_mass_api_available":True,"hook_restored":getattr(imrphenom_nsbh,"_legacy_phase_offsets",None) is ORIG,"calls":CAP,"offset_slope_predictions":predictions,"source_sha256_before":HASH_BEFORE,"source_sha256_after":hashes},"records":rec},indent=2)+"\n")
if __name__=="__main__":main(sys.argv[1] if len(sys.argv)>1 else "docs/nsbh_remnant_offset_diagnostic.json")
