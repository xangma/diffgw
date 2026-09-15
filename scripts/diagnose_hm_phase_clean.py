"""Single-mode London phase-map diagnostic using native unprojected modes.

The native mode dictionary is obtained through the public ``forward`` option
``return_unprojected_modes``.  Binary polarization output is retained as a
separate projection check; its angular factors are not treated as unprojected
mode phases.
"""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np
import torch
import lal
import lalsimulation as ls
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenom_hm_phase_clean import (
    mapped_phase_c1, phenomd_eq14_phase_delta_c1,
)
from torchwave.approximants.imrphenomd import IMRPhenomD

MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
CASES = (
    {"id":"catalog", "mass1":40., "mass2":15., "chi1":.3, "chi2":-.2, "distance":500., "inclination":.8, "phic":.2, "f_ref":20.},
    {"id":"holdout", "mass1":55., "mass2":11., "chi1":-.4, "chi2":.2, "distance":680., "inclination":1.1, "phic":.71, "f_ref":27.},
    {"id":"independent_spin", "mass1":32., "mass2":18., "chi1":-.6, "chi2":.45, "distance":430., "inclination":.6, "phic":.4, "f_ref":25.},
    {"id":"independent_mass_ratio", "mass1":70., "mass2":8., "chi1":.1, "chi2":-.5, "distance":900., "inclination":1., "phic":1.1, "f_ref":30.},
)


def _binary(c, modes):
    ma = ls.SimInspiralCreateModeArray()
    for ell, m in modes:
        ls.SimInspiralModeArrayActivateMode(ma, ell, m)
    d = lal.CreateDict(); ls.SimInspiralWaveformParamsInsertModeArray(d, ma)
    return ls.SimInspiralChooseFDWaveform(
        c["mass1"]*lal.MSUN_SI, c["mass2"]*lal.MSUN_SI, 0, 0, c["chi1"], 0, 0, c["chi2"],
        c["distance"]*1e6*lal.PC_SI, c["inclination"], c["phic"], 0, 0, 0, .25, 20, 512,
        c["f_ref"], d, ls.GetApproximantFromString("IMRPhenomHM"))


def _rms_angle(z, mask):
    q = np.angle(z[mask])
    return float(np.sqrt(np.mean(q*q)))


def _run_case(c):
    model, d = IMRPhenomHM(), IMRPhenomD()
    f = torch.arange(2049, dtype=torch.float64) * .25
    native_hp, native_hc, native_modes = model.forward(
        sample_frequencies=f, mass1=c["mass1"], mass2=c["mass2"], chi1=c["chi1"], chi2=c["chi2"],
        distance=c["distance"], inclination=c["inclination"], phic=c["phic"], tc=0., f_ref=c["f_ref"],
        f_lower=20., return_unprojected_modes=True)
    args = [torch.tensor(c[k], dtype=torch.float64) for k in ("mass1", "mass2", "chi1", "chi2")]
    mf, _, fr22, fd22 = d.remnant_properties(*args)
    mf = float(mf); fr22 = float(fr22); fd22 = float(fd22)
    eta = c["mass1"]*c["mass2"]/(c["mass1"]+c["mass2"])**2
    chi_eff = (c["mass1"]*c["chi1"]+c["mass2"]*c["chi2"])/(c["mass1"]+c["mass2"])
    chi_pn = chi_eff - 38.*eta/113.*(c["chi1"]+c["chi2"])
    alpha4 = d.eval_table_v("alpha4", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64))
    alpha5 = d.eval_table_v("alpha5", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64))
    mf_grid = f * (mf * lal.MTSUN_SI)
    rows = {}
    mask = f.numpy() >= 20.
    for ell, m in MODES:
        frlm, fdlm = model.mode_qnm_frequencies(ell, m, *args)
        frlm, fdlm = float(frlm), float(fdlm)
        f0 = .014 * frlm / fr22
        def carrier(x):
            x = torch.as_tensor(x, dtype=torch.float64)
            return d(x, *args, distance=c["distance"], inclination=c["inclination"], phic=c["phic"], tc=0., f_ref=c["f_ref"], return_phase=True)[2]
        mapped = mapped_phase_c1(mf_grid, f0, frlm, fr22, m, carrier, ell=ell)
        delta = phenomd_eq14_phase_delta_c1(mf_grid, eta, chi_pn, fr22, frlm, fd22, fdlm, alpha4, alpha5)
        predicted = torch.exp(-1j * (mapped + delta)).detach().numpy()
        native = native_modes[(ell, m)].detach().numpy()
        native_unit = native / np.maximum(np.abs(native), 1e-300)
        b, bc = _binary(c, ((ell, m),))
        binary = b.data.data - 1j*bc.data.data
        binary_unit = binary / np.maximum(np.abs(binary), 1e-300)
        band = (f.numpy() >= 20.) & (f.numpy() < 40.)
        rows[f"{ell}{m}"] = {
            "native_unprojected_phase_rms_rad": _rms_angle(predicted*np.conj(native_unit), mask),
            "binary_projected_phase_rms_rad": _rms_angle(predicted*np.conj(binary_unit), mask),
            "native_unprojected_phase_20_40_rms_rad": _rms_angle(predicted*np.conj(native_unit), band),
            "binary_projected_phase_20_40_rms_rad": _rms_angle(predicted*np.conj(binary_unit), band),
            "eq14_delta_rms_rad": float(torch.sqrt(torch.mean(delta*delta))),
            "finite": bool(np.isfinite(predicted).all() and np.isfinite(native).all()),
        }
    # Matching carrier/mode QNM and zero low offset gives exact 22 map identity.
    identity = mapped_phase_c1(mf_grid, .014, fr22, fr22, 2, carrier, ell=2, low_offset=0.)
    carrier_phase = carrier(mf_grid)
    assert torch.allclose(identity, carrier_phase, atol=2e-12, rtol=2e-12)
    assert all(v["finite"] for v in rows.values())
    return {"id": c["id"], "mf": mf, "d_qnm_22": {"f_rd": fr22, "f_damp": fd22},
            "chi_pn_unscaled": chi_pn, "identity_max_abs": float((identity-carrier_phase).abs().max()), "modes": rows}


def main():
    p = pathlib.Path(__file__)
    native_path = pathlib.Path(__import__("inspect").getfile(IMRPhenomHM))
    out = {"provenance":"London arXiv:1708.00404 Eqs.(3),(5),(8)-(9),(13)-(14); Khan arXiv:1508.07253 Eqs.(2)-(3), Sec.V. Native phases are public unprojected mode outputs; Eq.14 is evaluated in total-M Mf carrier coordinates with exp(-i phase). Binary phase rows are projected controls and are not unprojected errors.", "native_source_sha256":hashlib.sha256(native_path.read_bytes()).hexdigest(), "cases":[]}
    for c in CASES: out["cases"].append(_run_case(c))
    out["script_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__": main()
