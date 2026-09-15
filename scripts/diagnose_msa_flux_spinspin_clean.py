"""Clean-room diagnostic of the published MSA dissipative spin-spin choice."""
import hashlib
import json
from unittest.mock import patch

import lal
import lalsimulation
import torch

import torchwave.approximants.precession_msa_angles_clean as angle_module

F = [20.0, 30.0, 50.0, 100.0, 200.0]
CASES = [
    (35.0, 25.0, (0.2, 0.1, 0.2), (-0.1, 0.15, -0.1)),
    (45.0, 15.0, (0.45, -0.2, 0.3), (0.1, 0.2, -0.25)),
]


def public_reference(m1, m2, s1, s2):
    fv = lal.CreateREAL8Vector(len(F)); fv.data[:] = F
    d = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, 223)
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(d, 5)
    out = lalsimulation.SimIMRPhenomXPMSAAngles(
        fv, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2, 0.7, 20.0, 2, d)
    return [[float(x) for x in z.data] for z in out]


def run_variant(f, m1, m2, s1, s2, spin_spin):
    original = angle_module.flux_coefficients
    def omitted(*args, **kwargs):
        coeff = original(*args, **kwargs)
        coeff = dict(coeff)
        coeff["a4"] = coeff["a4"] + coeff["sigma4"]
        return coeff
    fn = omitted if spin_spin == "omitted" else original
    with patch.object(angle_module, "flux_coefficients", fn):
        out = angle_module.assemble_precession_msa_angles(
            torch.tensor(F), m1, m2, s1, s2, f_ref=20.0, order=5,
            beta_pn=True, beta_pn_version=102, beta_compat_102=False)
    return out


def run_case(m1, m2, s1, s2):
    ref = public_reference(m1, m2, s1, s2)
    variants = []
    for choice in ("baseline", "omitted"):
        out = run_variant(F, m1, m2, s1, s2, choice)
        ae = [float(x-y) for x, y in zip(out["alpha"], ref[0])]
        ge = [float(x-y) for x, y in zip(out["gamma"], ref[1])]
        aa = [x-ae[0] for x in ae]
        gg = [x-ge[0] for x in ge]
        be = [abs(float(x-y)) for x, y in zip(out["cos_beta_pn"], ref[2])]
        variants.append({
            "flux_choice": choice,
            "alpha_raw_error": ae, "gamma_raw_error": ge,
            "alpha_anchored_error": aa, "gamma_anchored_error": gg,
            "alpha_anchored_maxabs": max(map(abs, aa)),
            "gamma_anchored_maxabs": max(map(abs, gg)),
            "PNcosbeta_abs_error": be,
            "Sav": float(out["sav"]), "c1": float(out["c1"]),
        })
    return {
        "parameters": {"mass1_msun": m1, "mass2_msun": m2,
                        "chi1": list(s1), "chi2": list(s2),
                        "f_ref_hz": 20.0, "inclination": 0.7,
                        "mprime": 2, "expansion_order": 5,
                        "prec_version": 223},
        "frequencies_hz": F,
        "public_arrays": {"alpha": ref[0], "gamma": ref[1], "cos_beta": ref[2]},
        "variants": variants,
    }


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    report = {
        "provenance": "Clean-room diagnostic only. The omitted-spin-spin branch adds sigma4 to a4, removing the explicit spin-spin flux term while retaining all other clean MSA equations. No fitting or routing changes.",
        "public_callable": "SimIMRPhenomXPMSAAngles",
        "public_prec_version": 223, "public_expansion_order": 5, "public_mprime": 2,
        "cases": [run_case(*case) for case in CASES],
        "source_sha256": {
            "angle_module": sha256("src/torchwave/approximants/precession_msa_angles_clean.py"),
            "pn_module": sha256("src/torchwave/approximants/precession_msa_pn_clean.py"),
        },
    }
    with open("docs/reviews/2026-09-13-parity-review/msa-flux-spinspin-reference.json", "w") as h:
        json.dump(report, h, indent=2); h.write("\n")
    print(json.dumps(report, indent=2))
