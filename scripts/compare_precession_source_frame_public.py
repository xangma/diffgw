"""Compare fresh source-frame geometry with public XP black-box parameters."""
import copy, hashlib, json
from pathlib import Path
import lal
import lalsimulation as ls
import torch
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry

def angle_diff(a, b):
    return (a - b + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793


def cases():
    out = [next(c for c in get_canonical_catalog_cases() if c["torch_name"] == "IMRPhenomXP"),
           next(c for c in get_canonical_holdout_cases() if c["torch_name"] == "IMRPhenomXP")]
    vals = ((40.,20.,.3,.2,.4,-.2,.1,-.2,.7,40.), (30.,10.,.1,-.25,-.4,.15,.05,.2,.5,40.),
            (50.,15.,-.2,.3,.1,.1,-.2,.25,1.,30.), (28.,12.,.05,-.1,.6,-.3,.2,-.4,.4,25.))
    for i, (m1,m2,x1,y1,z1,x2,y2,z2,inc,rf) in enumerate(vals):
        out.append({"id": f"IMRPhenomXP_independent_{i}", "params": {"mass1":m1,"mass2":m2,
            "spin1x":x1,"spin1y":y1,"spin1z":z1,"spin2x":x2,"spin2y":y2,"spin2z":z2,
            "inclination":inc,"phic":0.,"f_ref":rf,"prec_version":102,"convention":0,"final_spin_mod":0}})
    base = copy.deepcopy(out[0])
    for i, inc in enumerate((0., .7, 1.1)):
        q = copy.deepcopy(base); q["id"] = f"IMRPhenomXP_aligned_{i}"
        q["params"].update(spin1x=0., spin1y=0., spin2x=0., spin2y=0., inclination=inc, phic=.3)
        out.append(q)
    return out


def one(case):
    p = case["params"]
    options = {"prec_version": int(p.get("prec_version", 102)), "convention": int(p.get("convention", 0)),
               "final_spin_mod": int(p.get("final_spin_mod", 0))}
    d = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, options["prec_version"])
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(d, options["convention"])
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(d, options["final_spin_mod"])
    looked = {"prec_version": int(ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(d)),
              "convention": int(ls.SimInspiralWaveformParamsLookupPhenomXPConvention(d)),
              "final_spin_mod": int(ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(d))}
    assert looked == options
    pub = ls.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        p["mass1"]*lal.MSUN_SI, p["mass2"]*lal.MSUN_SI, p["f_ref"], p.get("phic", 0.), p["inclination"],
        p.get("spin1x", 0.), p.get("spin1y", 0.), p.get("spin1z", 0.),
        p.get("spin2x", 0.), p.get("spin2y", 0.), p.get("spin2z", 0.), d)
    q = [torch.tensor(float(p[k]), dtype=torch.float64) for k in
         ("mass1","mass2","spin1x","spin1y","spin1z","spin2x","spin2y","spin2z","inclination")]
    clean = source_frame_geometry(*q, torch.tensor(float(p.get("phic", 0.)), dtype=torch.float64),
                                  torch.tensor(float(p["f_ref"]), dtype=torch.float64), compat_102=True)
    return {"id": case["id"], "options": options, "looked_up_options": looked,
            "public_tuple_semantics": ["chi1","chi2","chip","thetaJN","alpha0","phiJL0","zeta"],
            "public": list(map(float, pub)), "clean_geometry": [float(x) for x in clean],
            "differences": {"alpha0_wrapped": float(angle_diff(float(clean[0]), pub[4])),
                            "thetaJN": float(clean[1]-pub[3]),
                            "zeta_wrapped": float(angle_diff(float(clean[2]), pub[6])),
                            "phiJL0_vs_neg_phiJ_wrapped": float(angle_diff(float(clean[3]), -pub[5]))}}


def main():
    out = {"api_doc": ls.SimIMRPhenomXPCalculateModelParametersFromSourceFrame.__doc__, "cases": [one(c) for c in cases()], "sha256": {}}
    for path in ("src/torchwave/approximants/precession_source_frame_clean.py", __file__):
        out["sha256"][str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dst = Path("docs/reviews/2026-09-13-parity-review/precession-source-frame-public.json")
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
