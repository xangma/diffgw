"""Black-box NSBH remnant-spin property and phase-hook diagnostic."""
import hashlib, inspect, json, pathlib, sys
import lal, lalsimulation as ls
import numpy as np
import torch
from torchwave.approximants import imrphenomd, imrphenom_nsbh
from nsbh_phase_residual_diagnostic import CASES, _native, _binary, _metric

ROOT = pathlib.Path(__file__).resolve().parents[1]


def properties(c, lam):
    m1 = torch.tensor(c["m1"], dtype=torch.float64)
    m2 = torch.tensor(c["m2"], dtype=torch.float64)
    s1 = torch.tensor(c["chi1"], dtype=torch.float64)
    s2 = torch.tensor(0., dtype=torch.float64)
    nm, ns = imrphenom_nsbh.zappa_remnant_properties(m1, m2, s1, s2, torch.tensor(lam, dtype=torch.float64))
    lp = ls.SimIMRPhenomNSBHProperties(c["m1"]*lal.MSUN_SI, c["m2"]*lal.MSUN_SI, c["chi1"], lam)
    return {"native_mass_msun": float(nm), "native_spin": float(ns),
            "lal_mass_msun": float(lp[4]/lal.MSUN_SI), "lal_spin": float(lp[5]),
            "mass_abs_diff_msun": float(abs(float(nm)-lp[4]/lal.MSUN_SI)),
            "spin_abs_diff": float(abs(float(ns)-lp[5]))}


def main(path="docs/nsbh_spin_convention_diagnostic.json"):
    f = np.arange(1024./.25 + 1.) * .25
    props, hooks = [], []
    original = imrphenomd.IMRPhenomD.remnant_properties

    for name, c in CASES.items():
        for lam in (0., c["lam"]):
            def zappa_qnm(*args, _lam=lam, _mode="zappa_spin"):
                if isinstance(args[0], imrphenomd.IMRPhenomD):
                    args = args[1:]
                m1, m2, s1, s2 = args
                old = original(m1, m2, s1, s2)
                lt = torch.as_tensor(_lam, dtype=m1.dtype, device=m1.device)
                zmass, zspin = imrphenom_nsbh.zappa_remnant_properties(m1, m2, s1, s2, lt)
                qre, qim = imrphenomd.evaluate_kerr_qnm_220(zspin)
                # remnant_properties documents M_total*f_RD; the Kerr helper
                # returns M_final*omega (angular), so convert both units.
                selected_mass = zmass if _mode == "zappa_mass" else old[0]
                fac = (m1 + m2) / selected_mass / (2 * torch.pi)
                assert torch.isfinite(fac).all() and (fac > 0).all(), "invalid QNM unit-conversion factor"
                return selected_mass, zspin, qre * fac, qim * fac
            props.append({"case": name, "lambda2": lam, "properties": properties(c, lam)})
            control = _native(c, f, lam)
            b = _binary(c, f, lam)
            for mode in ("zappa_spin", "zappa_mass"):
                # Diagnostic substitution is deliberately process-local and restored below.
                hook = lambda *args, _mode=mode: zappa_qnm(*args, _mode=_mode)
                try:
                    imrphenomd.IMRPhenomD.remnant_properties = staticmethod(hook)
                    n = _native(c, f, lam)
                finally:
                    imrphenomd.IMRPhenomD.remnant_properties = staticmethod(original)
                hooks.append({"case": name, "lambda2": lam, "mode": mode,
                          "finite": bool(torch.isfinite(n[0]).all() and torch.isfinite(n[1]).all()),
                          "support_length": [len(n[0]), len(b[0])],
                          "restoration_max_abs": float(max((control[0]-_native(c, f, lam)[0]).abs().max().item(),
                                                            (control[1]-_native(c, f, lam)[1]).abs().max().item())),
                          "hp": _metric(n[0].detach().numpy(), b[0], f),
                          "hc": _metric(n[1].detach().numpy(), b[1], f)})
    inv = {}
    for mod in (imrphenomd, imrphenom_nsbh):
        for n in dir(mod):
            if "phase" in n.lower() or "phi" in n.lower() or "qnm" in n.lower():
                o = getattr(mod, n)
                try: inv[f"{mod.__name__}.{n}"] = {"signature": str(inspect.signature(o)), "doc": inspect.getdoc(o)}
                except (TypeError, ValueError): pass
    for n in ("SimIMRPhenomNSBHProperties", "SimIMRPhenomDFinalSpin",
              "SimPhenomUtilsIMRPhenomDFinalMass"):
        o = getattr(ls, n)
        inv[f"lalsimulation.{n}"] = {"signature": "builtin (inspect.signature unavailable)",
                                      "doc": inspect.getdoc(o)}
    out = {"metadata": {"python": sys.executable,
        "qnm_unit_conversion": "evaluate_kerr_qnm_220 gives M_final*omega; remnant_properties wants M_total*f, so factor=(m1+m2)/(M_final*2*pi)",
        "unit_assertion": True,
        "source_sha256": {
        p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "scripts/nsbh_spin_convention_diagnostic.py", "src/torchwave/approximants/imrphenom_nsbh.py",
            "src/torchwave/approximants/imrphenomd.py")}, "hook_restored": inspect.unwrap(imrphenomd.IMRPhenomD.remnant_properties) is original,
        "public_callable_inventory": inv}, "property_comparisons": props, "zappa_spin_qnm_hook": hooks}
    pathlib.Path(path).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "docs/nsbh_spin_convention_diagnostic.json")
