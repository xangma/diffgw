"""Diagnostic of the literal versus derived MSA oscillatory prefactor."""
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


def evaluate(m1, m2, s1, s2, literal):
    original = angle_module.oscillatory_correction
    def paper_literal(psi, J, psidot, *args):
        phi, zeta = original(psi, J, psidot, *args)
        return phi / J, zeta / J
    fn = paper_literal if literal else original
    with patch.object(angle_module, "oscillatory_correction", fn):
        return angle_module.assemble_precession_msa_angles(
            torch.tensor(F), m1, m2, s1, s2, f_ref=20.0, order=5,
            beta_pn=True, beta_pn_version=102, beta_compat_102=False)


def run_case(m1, m2, s1, s2):
    ref = public_reference(m1, m2, s1, s2)
    variants = []
    for label, literal in (("derived_J_over_psidot", False), ("literal_1_over_psidot", True)):
        out = evaluate(m1, m2, s1, s2, literal)
        ae = [float(x-y) for x, y in zip(out["alpha"], ref[0])]
        ge = [float(x-y) for x, y in zip(out["gamma"], ref[1])]
        aa = [x-ae[0] for x in ae]; gg = [x-ge[0] for x in ge]
        variants.append({
            "factor_choice": label,
            "alpha_raw_error": ae, "gamma_raw_error": ge,
            "alpha_anchored_error": aa, "gamma_anchored_error": gg,
            "alpha_anchored_maxabs": max(map(abs, aa)),
            "gamma_anchored_maxabs": max(map(abs, gg)),
            "PNcosbeta_abs_error": [abs(float(x-y)) for x, y in zip(out["cos_beta_pn"], ref[2])],
            "Sav": float(out["sav"]), "c1": float(out["c1"]),
        })
    return {"parameters": {"mass1_msun": m1, "mass2_msun": m2,
                             "chi1": list(s1), "chi2": list(s2),
                             "f_ref_hz": 20.0, "inclination": 0.7,
                             "mprime": 2, "expansion_order": 5,
                             "prec_version": 223},
            "frequencies_hz": F, "public_arrays": {
                "alpha": ref[0], "gamma": ref[1], "cos_beta": ref[2]},
            "variants": variants}


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    report = {
        "provenance": "Clean-room paper diagnostic. Eq. (phiz0) prints 1/psidot, while the Appendix-B ratio construction used here yields J/psidot after inserting the derived C_phi,D_phi. No other MSA quantity is changed.",
        "paper_context": "The cached arXiv:1703.03967v3 transcription defines C_phi and D_phi from c_i/d_i ratios without a displayed J factor; this report preserves that literal reading as a discrete diagnostic and does not adopt it in runtime.",
        "public_callable": "SimIMRPhenomXPMSAAngles", "public_prec_version": 223,
        "public_expansion_order": 5, "public_mprime": 2,
        "cases": [run_case(*case) for case in CASES],
        "source_sha256": {
            "angle_module": digest("src/torchwave/approximants/precession_msa_angles_clean.py"),
            "oscillatory_module": digest("src/torchwave/approximants/precession_msa_oscillatory_clean.py"),
        },
    }
    with open("docs/reviews/2026-09-13-parity-review/msa-oscillatory-factor-reference.json", "w") as h:
        json.dump(report, h, indent=2); h.write("\n")
    print(json.dumps(report, indent=2))
