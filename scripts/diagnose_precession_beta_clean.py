"""Eq.17/Eq.26 beta comparison using only public angle APIs and clean equations."""
from __future__ import annotations
import hashlib, json, platform, sys
import lal, lalsimulation, torch
from torchwave.approximants.precession_msa import evaluate_msa_precession
from torchwave.approximants.precession_beta_clean import beta, cos_beta

REQUESTED_FREQS = [20., 50., 100., 200.]
CASES = {
    "catalog_35_25": dict(m1=35.,m2=25.,s1x=.2,s1y=.1,s1z=.2,s2x=-.1,s2y=.15,s2z=-.1,iota=.7,fref=20.),
    "holdout_42_17": dict(m1=42.,m2=17.,s1x=.3,s1y=-.15,s1z=-.3,s2x=.12,s2y=.08,s2z=.2,iota=1.1,fref=35.),
    "withheld_28_11": dict(m1=28.,m2=11.,s1x=-.15,s1y=.22,s1z=.35,s2x=.08,s2y=-.12,s2z=-.25,iota=.9,fref=27.),
    "withheld_61_19": dict(m1=61.,m2=19.,s1x=.11,s1y=-.18,s1z=-.2,s2x=-.2,s2y=.06,s2z=.3,iota=1.2,fref=43.),
}

def public_angles(p, freqs):
    fv = lal.CreateREAL8Vector(len(freqs)); fv.data[:] = freqs
    params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, 102)
    out = lalsimulation.SimIMRPhenomXPPNAngles(
        fv, p["m1"]*lal.MSUN_SI, p["m2"]*lal.MSUN_SI,
        p["s1x"],p["s1y"],p["s1z"],p["s2x"],p["s2y"],p["s2z"],
        p["iota"],p["fref"],2,params)
    return [[float(x) for x in y.data] for y in out]

def clean_and_native(p, freqs):
    d={k:torch.tensor(v,dtype=torch.float64) for k,v in p.items() if k not in ("iota","fref")}
    f=torch.tensor(freqs,dtype=torch.float64); M=d["m1"]+d["m2"]
    v=(torch.pi*M*f*lal.MTSUN_SI)**(1/3); v0=(torch.pi*M*p["fref"]*lal.MTSUN_SI)**(1/3)
    native=evaluate_msa_precession(v,v0,**d,iota=torch.tensor(p["iota"],dtype=torch.float64),phi_ref=torch.tensor(0.,dtype=torch.float64),prec_version=102,convention=7)
    args=(p["m1"],p["m2"],f,p["s1x"],p["s1y"],p["s1z"],p["s2x"],p["s2y"],p["s2z"])
    clean=beta(*args); compat=beta(*args,compat_102=True)
    return clean.tolist(), compat.tolist(), native[1].tolist()

def run(name,p):
    freqs=sorted(set(REQUESTED_FREQS+[p["fref"]]))
    pub=public_angles(p,freqs); clean,compat,native=clean_and_native(p,freqs)
    public_beta=[float(torch.acos(torch.tensor(x,dtype=torch.float64))) for x in pub[2]]
    return {"parameters":p,"frequencies_hz":freqs,"fref_index":freqs.index(p["fref"]),"public_cos_beta":pub[2],
            "public_beta":public_beta,
            "clean_beta":clean,"native_msa_beta":native,
            "compat_102_beta":compat,
            "clean_minus_public_beta":[a-b for a,b in zip(clean,public_beta)],
            "compat_102_minus_public_beta":[a-b for a,b in zip(compat,public_beta)],
            "native_minus_public_beta":[a-b for a,b in zip(native,public_beta)]}

if __name__ == "__main__":
    out={"provenance":"Pratten et al. arXiv:2004.06503v2 Appendix G Eq.26/Table 3 version-102 truncation (L through 3PN, SO through 2.5PN), Eq.17 with single-effective-spin S_perp=x1^2 chi_p; f is GW Hz, x=(pi M f)^(2/3), public XPPNAngles prec_version=102","python":sys.version,"torch":torch.__version__,"lal":getattr(lal,"__version__","unknown"),"lalsimulation":getattr(lalsimulation,"__version__","unknown"),"cases":{n:run(n,p) for n,p in CASES.items()}}
    for path in ["src/torchwave/approximants/precession_beta_clean.py",__file__]:
        with open(path,"rb") as fh: out["sha256_"+path.split("/")[-1]]=hashlib.sha256(fh.read()).hexdigest()
    target="docs/reviews/2026-09-13-parity-review/precession-beta-clean.json"
    with open(target,"w") as fh: json.dump(out,fh,indent=2); fh.write("\n")
    print(json.dumps(out,indent=2))
