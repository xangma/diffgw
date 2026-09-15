"""Diagnostic for London's Eq. (14) higher-mode MR phase replacement."""
import hashlib
import json
import pathlib
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenom_hm_phase_clean import phenomd_eq14_phase_delta_c1

CASES = (("catalog", 40., 15., .3, -.2), ("holdout", 55., 11., -.4, .2),
         ("independent_spin", 32., 18., -.6, .45),
         ("independent_mass_ratio", 70., 8., .1, -.5))
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _case_row(d, hm, m1, m2, chi1, chi2):
    total = m1 + m2
    eta = _t(m1 * m2 / total ** 2)
    chi_eff = _t((m1 * chi1 + m2 * chi2) / total)
    # Khan et al. arXiv:1508.07253 Eq. (3), unscaled reduced PN spin.
    chi_pn = chi_eff - (38. * eta / 113.) * _t(chi1 + chi2)
    remnant = d.remnant_properties(_t(m1), _t(m2), _t(chi1), _t(chi2))
    fr22, fd22 = (float(z) for z in remnant[2:])
    alpha4 = d.eval_table_v("alpha4", eta, chi_pn)
    alpha5 = d.eval_table_v("alpha5", eta, chi_pn)
    f = torch.linspace(.5 * fr22, 4. * fr22, 1000, dtype=torch.float64)
    args = (_t(m1), _t(m2), _t(chi1), _t(chi2))
    modes = {}
    for ell, m in MODES:
        fr_lm, fd_lm = hm.mode_qnm_frequencies(ell, m, *args)
        delta = phenomd_eq14_phase_delta_c1(
            f, eta, chi_pn, fr22, fr_lm, fd22, fd_lm, alpha4, alpha5)
        modes[f"{ell}{m}"] = {
            "max_abs_rad": float(delta.abs().max()),
            "rms_rad": float(torch.sqrt(torch.mean(delta * delta))),
            "finite": bool(torch.isfinite(delta).all()),
            "join_Mf_carrier": .5 * fr22,
            "mode_qnm": {"f_rd": float(fr_lm), "f_damp": float(fd_lm)},
        }
    identity = phenomd_eq14_phase_delta_c1(
        f, eta, chi_pn, fr22, fr22, fd22, fd22, alpha4, alpha5)
    assert torch.equal(identity, torch.zeros_like(identity))
    return {"eta": float(eta), "chi_eff": float(chi_eff),
            "chi_pn_unscaled": float(chi_pn),
            "d_qnm_22": {"f_rd": fr22, "f_damp": fd22},
            "eq14_identical_22_max_abs": float(identity.abs().max()),
            "modes": modes}


def main():
    d, hm = IMRPhenomD(), IMRPhenomHM()
    out = {"provenance": (
        "London arXiv:1708.00404 Eq.(14); Khan et al. "
        "arXiv:1508.07253 Eqs.(2)-(3), Sec.V join. Frequencies are "
        "dimensionless total-M Mf. Eq.(14) attaches at 0.5*f_rd_22 "
        "in carrier coordinates. eval_table_v receives unscaled chi_PN."),
        "cases": []}
    for ident, m1, m2, chi1, chi2 in CASES:
        row = _case_row(d, hm, m1, m2, chi1, chi2)
        row["id"] = ident
        out["cases"].append(row)
    script = pathlib.Path(__file__)
    out["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    script.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
