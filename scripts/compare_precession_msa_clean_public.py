"""Black-box comparison of clean MSA angles with LALSuite's public callable."""
import json
from pathlib import Path
from unittest.mock import patch
import torch

from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
import torchwave.approximants.precession_msa_angles_clean as angle_module


def literal_appendix_c_psi_coefficients(eta, delta, q, xi, c1, sav,
                                        s1sq, s2sq, g0, g2):
    """Literal arXiv:1703.03967v3 Appendix-C ``Delta`` diagnostic.

    This deliberately retains the printed first factor ``c1**2*eta**2``;
    the clean baseline uses the independently reduced cubic expression.
    """
    vals = torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64)
                                      for x in (eta, delta, q, xi, c1, sav,
                                                s1sq, s2sq, g0, g2)))
    eta, delta, q, xi, c1, sav, s1sq, s2sq, g0, g2 = vals
    d4 = delta**4
    first = (c1**2 * eta / (q * d4)
             - 2*c1*eta**3*(1+q)*xi/(q*d4)
             - eta**2/d4*(delta**2*s1sq - eta**2*xi**2))
    second = (c1**2 * eta**2 / d4
              - 2*c1*eta**3*(1+q)*xi/d4
              - eta**2/d4*(delta**2*s2sq - eta**2*xi**2))
    Delta = torch.sqrt(first * second)
    ratio = (1-q)**2
    bracket = (2*Delta - 2*eta**2/delta**2*sav
               - 10*eta/d4*c1**2
               + 2*eta**2/delta**2*(7+6*q+7*q**2)/ratio*c1*xi
               - eta**3/delta**2*(3+4*q+3*q**2)/ratio*xi**2
               + eta/ratio*(q*(2+q)*s1sq + (1+2*q)*s2sq))
    psi1 = 3*(2*xi*eta**2-c1)/(eta*delta**2)
    psi2 = 3*g2/g0 + 3/(2*eta**3)*bracket
    return psi1, psi2


def run(output=None):
    f = torch.tensor([20., 30., 50., 100., 200.], dtype=torch.float64)
    cases = [(35., 25., [.2, .1, .2], [-.1, .15, -.1]),
             (45., 15., [.45, -.2, .3], [.1, .2, -.25])]
    result = {"frequencies_hz": f.tolist(), "cases": []}
    try:
        import lalsimulation as ls
        import lal
        public = ls.SimIMRPhenomXPMSAAngles
        doc = getattr(public, "__doc__", "")
    except ImportError:
        public, doc = None, ""
    result["public_callable"] = "SimIMRPhenomXPMSAAngles"
    result["public_callable_doc_available"] = bool(doc)
    if public is not None:
        result["versions"] = {"torch": torch.__version__,
                               "lal": getattr(lal, "__version__", "unknown"),
                               "lalsimulation": getattr(ls, "__version__", "unknown")}
    for m1, m2, s1, s2 in cases:
        item = {"m1": m1, "m2": m2, "chi1": s1, "chi2": s2,
                "inclination": .7, "f_ref": 20.0, "mprime": 2,
                "expansion_order": 5}
        for version in (220, 223):
            clean = assemble_precession_msa_angles(f, m1, m2, s1, s2, beta_pn=True)
            item[str(version)] = {k: v.detach().tolist() for k, v in clean.items()
                                  if k in ("alpha", "gamma", "cos_beta", "cos_beta_newtonian", "cos_beta_pn", "psi", "roots", "roots_reference", "c1", "sav", "psi1", "psi2", "secular_phi", "secular_zeta", "oscillatory_phi", "oscillatory_zeta")}
            if public is not None:
                params = lal.CreateDict()
                ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, version)
                ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(params, 5)
                fv = lal.CreateREAL8Vector(f.numel()); fv.data[:] = f.tolist()
                raw = public(fv, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                             *s1, *s2, .7, 20., 2, params)
                pub = [torch.tensor(list(x.data), dtype=torch.float64) for x in raw]
                raw_da = clean["alpha"].detach() - pub[0]
                raw_dg = clean["gamma"].detach() - pub[1]
                raw_db = clean["cos_beta"].detach() - pub[2]
                # Angles are gauge quantities: remove only each series' f_ref value.
                da = (clean["alpha"] - clean["alpha"][0]).detach() - (pub[0] - pub[0][0])
                dg = (clean["gamma"] - clean["gamma"][0]).detach() - (pub[1] - pub[1][0])
                db = clean["cos_beta"].detach() - pub[2]
                item[str(version)]["public"] = {"alpha": pub[0].tolist(), "gamma": pub[1].tolist(), "cos_beta": pub[2].tolist()}
                base_errors = {"alpha": raw_da.tolist(), "gamma": raw_dg.tolist(),
                               "cos_beta_newtonian": raw_db.tolist(),
                               "cos_beta_pn": (clean["cos_beta_pn"] - pub[2]).detach().tolist(),
                               "max_abs": {"alpha": float(raw_da.abs().max()), "gamma": float(raw_dg.abs().max()),
                                            "cos_beta_newtonian": float(raw_db.abs().max()),
                                            "cos_beta_pn": float((clean["cos_beta_pn"] - pub[2]).abs().max())}}
                item[str(version)]["raw_errors"] = base_errors
                item[str(version)]["reference_anchored_errors"] = {"alpha": da.tolist(), "gamma": dg.tolist(),
                                                  "cos_beta_newtonian": db.tolist(),
                                                  "cos_beta_pn": (clean["cos_beta_pn"] - pub[2]).detach().tolist(),
                                                  "max_abs": {"alpha": float(da.abs().max()), "gamma": float(dg.abs().max()),
                                                               "cos_beta_newtonian": float(db.abs().max()),
                                                               "cos_beta_pn": float((clean["cos_beta_pn"] - pub[2]).abs().max())}}
                item[str(version)]["errors"] = item[str(version)]["reference_anchored_errors"]
                with patch.object(angle_module, "psi_coefficients",
                                  literal_appendix_c_psi_coefficients):
                    literal = assemble_precession_msa_angles(
                        f, m1, m2, s1, s2, beta_pn=True)
                lit = {k: literal[k].detach().tolist() for k in
                       ("alpha", "gamma", "cos_beta_pn", "psi", "psi1", "psi2")}
                lit_da = literal["alpha"].detach() - pub[0]
                lit_dg = literal["gamma"].detach() - pub[1]
                lit_db = literal["cos_beta_pn"].detach() - pub[2]
                lit_db_newt = literal["cos_beta"].detach() - pub[2]
                lit["raw_errors"] = {"alpha": lit_da.tolist(), "gamma": lit_dg.tolist(),
                                     "cos_beta_newtonian": lit_db_newt.tolist(),
                                     "cos_beta_pn": lit_db.tolist(),
                                     "max_abs": {"alpha": float(lit_da.abs().max()), "gamma": float(lit_dg.abs().max()),
                                                  "cos_beta_newtonian": float(lit_db_newt.abs().max()),
                                                  "cos_beta_pn": float(lit_db.abs().max())}}
                lit_da_ref = (literal["alpha"] - literal["alpha"][0]).detach() - (pub[0] - pub[0][0])
                lit_dg_ref = (literal["gamma"] - literal["gamma"][0]).detach() - (pub[1] - pub[1][0])
                lit["reference_anchored_errors"] = {"alpha": lit_da_ref.tolist(), "gamma": lit_dg_ref.tolist(),
                                                     "cos_beta_newtonian": lit_db_newt.tolist(),
                                                     "cos_beta_pn": lit_db.tolist(),
                                                     "max_abs": {"alpha": float(lit_da_ref.abs().max()), "gamma": float(lit_dg_ref.abs().max()),
                                                                  "cos_beta_newtonian": float(lit_db_newt.abs().max()),
                                                                  "cos_beta_pn": float(lit_db.abs().max())}}
                item[str(version)]["literal_appendix_c"] = lit
        result["cases"].append(item)
    if output is None:
        output = Path("docs/reviews/2026-09-13-parity-review/precession-msa-angles-clean-public.json")
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    run()
