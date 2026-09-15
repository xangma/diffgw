"""Record equation-level IMRPhenomHM mapping diagnostics.

This calls the public TorchWave HM API and the documented binary
LALSimulation APIs, and compares exported affine maps with the independently
transcribed London et al. equations.  It does not inspect implementation
source or use PyCBC, calibration data, fitting, or NR data.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform

import torch
import numpy as np
from torchwave.approximants.imrphenom_hm import IMRPhenomHM

from torchwave.approximants.imrphenom_hm import (
    _swsh_projection_coefficients,
    frequency_mapping_amplitude,
    frequency_mapping_phase,
)
from torchwave.approximants.imrphenom_hm_clean import (
    map_amplitude_frequency,
    map_phase_frequency,
    pn_hhat_lm,
)
from torchwave.waveforms import get_fd_waveform


def _hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    f = torch.linspace(0.0, 0.16, 1001, dtype=torch.float64)
    cases = [(1, 0.021, 0.0589, 0.0596), (2, 0.018, 0.0954, 0.0596),
             (3, 0.019, 0.1286, 0.0596), (4, 0.019, 0.1290, 0.0596)]
    rows = []
    for m, f0, flm, f22 in cases:
        for kind, native, clean in (
            ("amplitude", frequency_mapping_amplitude, map_amplitude_frequency),
            ("phase", frequency_mapping_phase, map_phase_frequency),
        ):
            a = native(f, f0, flm, f22, m)
            b = clean(f, f0, flm, f22, m)
            rows.append({"m": m, "kind": kind, "max_abs_difference": float((a - b).abs().max()),
                         "finite": bool(torch.isfinite(a).all()),
                         "join_residual_f0": float((a[torch.argmin((f-f0).abs())] - 2*f0/m).abs()),
                         "join_residual_fr": float((a[torch.argmin((f-flm).abs())] - f22).abs())})
    src = pathlib.Path(__file__).resolve().parents[1] / "src/torchwave/approximants/imrphenom_hm_clean.py"
    native_root = src.parent
    native_sources = {
        "imrphenom_hm.py": native_root / "imrphenom_hm.py",
        "imrphenomd.py": native_root / "imrphenomd.py",
        "constants.py": native_root.parent / "constants.py",
        "conversions.py": native_root.parent / "conversions.py",
    }
    out = pathlib.Path(__file__).with_suffix(".json")
    model = IMRPhenomHM()
    frequencies = torch.tensor([25.0, 100.0, 300.0], dtype=torch.float64)
    _, _, mode_dict = model(sample_frequencies=frequencies, mass1=35.0, mass2=25.0,
                            chi1=0.1, chi2=-0.05, distance=400.0, inclination=0.7,
                            phic=0.2, tc=0.0, f_ref=25.0,
                            return_unprojected_modes=True)
    modes = {f"{l}{m}": {"real": [float(z.real) for z in v],
                           "imag": [float(z.imag) for z in v]}
             for (l, m), v in mode_dict.items()}
    case = {"mass1": 35.0, "mass2": 25.0, "chi1": 0.1, "chi2": -0.05,
            "distance_mpc": 400.0, "inclination_rad": 0.7, "phic_rad": 0.2,
            "tc_sec": 0.0, "f_ref_hz": 25.0, "sample_frequencies_hz": frequencies.tolist(),
            "api": "IMRPhenomHM.forward(sample_frequencies=..., return_unprojected_modes=True)"}
    case_hash = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Public lalsimulation mode API comparison.  The same physical parameters,
    # positive-frequency grid, f_ref and phiRef are passed to both models.
    binary = {"available": False}
    binary_cases_for_hash = [
        {"id": "IMRPhenomHM_diagnostic", "mass1": 40., "mass2": 15., "chi1": .3, "chi2": -.2,
         "f_ref": 20., "distance": 500., "inclination": .8, "phic": .2, "tc": 0., "flow": 20., "fhigh": 512., "df": .25},
        {"id": "IMRPhenomHM_holdout_5", "mass1": 55., "mass2": 11., "chi1": -.4, "chi2": .2,
         "f_ref": 27., "distance": 680., "inclination": 1.1, "phic": .71, "tc": 0., "flow": 20., "fhigh": 512., "df": .25},
    ]
    binary_case_hash = hashlib.sha256(json.dumps(binary_cases_for_hash, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    try:
        import lal
        import lalsimulation as lalsim
        binary["available"] = True
        binary["versions"] = {"lal": getattr(lal, "__version__", "unknown"),
                               "lalsimulation": getattr(lalsim, "__version__", "unknown")}
        bcases = binary_cases_for_hash
        for bc in bcases:
            freq = np.arange(int(round(bc["fhigh"] / bc["df"])) + 1) * bc["df"]
            keep = (freq >= bc["flow"]) & (freq <= bc["fhigh"])
            fe = freq[keep]
            fv = lal.CreateREAL8Vector(len(fe)); fv.data[:] = fe
            lm = lalsim.SimIMRPhenomHMGethlmModes(
                fv, bc["mass1"]*lal.MSUN_SI, bc["mass2"]*lal.MSUN_SI, 0., 0., bc["chi1"],
                0., 0., bc["chi2"], bc["phic"], bc["df"], bc["f_ref"], lal.CreateDict())
            lal_modes = {}
            while lm:
                if (lm.l, lm.m) in IMRPhenomHM.SUPPORTED_MODES:
                    # The public mode API returns exactly the supplied
                    # positive-frequency sequence (unlike a deltaF grid).
                    lal_modes[(lm.l, lm.m)] = np.asarray(lm.mode.data.data)
                lm = lm.next
            tf = torch.tensor(fe, dtype=torch.float64)
            _, _, tw_modes = IMRPhenomHM()(sample_frequencies=tf, mass1=bc["mass1"], mass2=bc["mass2"],
                chi1=bc["chi1"], chi2=bc["chi2"], distance=bc["distance"], inclination=bc["inclination"],
                phic=bc["phic"], tc=bc["tc"], f_ref=bc["f_ref"], return_unprojected_modes=True)
            mode_metrics = {}
            tw22 = tw_modes[(2, 2)].detach().cpu().numpy()
            la22 = lal_modes[(2, 2)]
            for mode in IMRPhenomHM.SUPPORTED_MODES:
                tw = tw_modes[mode].detach().cpu().numpy()
                la = lal_modes[mode]
                # The mode API has no distance argument.  Divide each mode by
                # its own 22 mode to cancel that documented normalization while
                # preserving the supplied phiRef/fRef convention.
                twr, lar = tw / tw22, la / la22
                den = np.linalg.norm(lar)
                z = twr / lar
                phase = np.angle(z)
                mode_metrics[f"{mode[0]}{mode[1]}"] = {
                    "relative_mode_to_22_amplitude_l2": float(np.linalg.norm(np.abs(twr)-np.abs(lar)) / den),
                    "relative_mode_to_22_complex_l2": float(np.linalg.norm(twr-lar) / den),
                    "phase_circular_rms_rad": float(np.sqrt(np.mean(np.angle(np.exp(1j*phase))**2))),
                    "phase_median_rad": float(np.median(np.angle(np.exp(1j*phase)))),
                }
            # Same spherical-harmonic projection for both mode sets.  This is
            # a reconstruction control, with all modes normalized by 22.
            hp_t = torch.zeros_like(torch.from_numpy(tw22), dtype=torch.complex128)
            hc_t = torch.zeros_like(hp_t)
            hp_l = np.zeros_like(la22)
            hc_l = np.zeros_like(la22)
            for mode in IMRPhenomHM.SUPPORTED_MODES:
                dp, dc = _swsh_projection_coefficients(mode[0], mode[1], torch.tensor(bc["inclination"], dtype=torch.float64), torch.tensor(bc["phic"], dtype=torch.float64))
                hp_t = hp_t + dp * torch.from_numpy(tw_modes[mode].detach().cpu().numpy() / tw22)
                hc_t = hc_t + dc * torch.from_numpy(tw_modes[mode].detach().cpu().numpy() / tw22)
                hp_l = hp_l + complex(dp.detach().cpu().numpy()) * lal_modes[mode] / la22
                hc_l = hc_l + complex(dc.detach().cpu().numpy()) * lal_modes[mode] / la22
            tw22_only_hp, tw22_only_hc = _swsh_projection_coefficients(2, 2, torch.tensor(bc["inclination"], dtype=torch.float64), torch.tensor(bc["phic"], dtype=torch.float64))
            record = {"id": bc["id"], "mode_metrics": mode_metrics,
                "mode_api": "SimIMRPhenomHMGethlmModes",
                "mode_api_signature": "SimIMRPhenomHMGethlmModes(REAL8Vector[f=20..512Hz,df=.25], m1_SI,m2_SI, S1x=0,S1y=0,S1z=chi1, S2x=0,S2y=0,S2z=chi2, phiRef, deltaF, f_ref, lal.Dict)",
                "polarization_api_signature": "SimInspiralChooseFDWaveform(same masses/spins, distance_Mpc*1e6*lal.PC_SI, inclination, phiRef, tc=0, deltaF=.25, f_min=20, f_max=512, f_ref, lal.Dict[ModeArray])",
                "frequency_count": len(fe),
                "phase_convention": "raw complex ratio; no time/phase optimization or alignment",
                "metric_scope": "Each reported mode is divided by the same model's (2,2) mode because the public mode API has no distance argument. These are relative-mode diagnostics, not raw individual-mode errors; (2,2) is identically zero by construction.",
                "reconstruction_control": {
                "summed_normalized_mode_hp_relative_l2": float(np.linalg.norm(hp_t.numpy()-hp_l) / np.linalg.norm(hp_l)),
                "summed_normalized_mode_hc_relative_l2": float(np.linalg.norm(hc_t.numpy()-hc_l) / np.linalg.norm(hc_l)),
                "22_only_scope": "public binary polarization waveform with ModeArray=(2,2); mode ratios are not individual-mode errors",
                "22_only_hp_norm": float(torch.linalg.vector_norm(tw22_only_hp).item()),
                "22_only_hc_norm": float(torch.linalg.vector_norm(tw22_only_hc).item()),
            }}
            binary.setdefault("cases", []).append(record)
            ma = lalsim.SimInspiralCreateModeArray()
            lalsim.SimInspiralModeArrayActivateMode(ma, 2, 2)
            lp = lal.CreateDict(); lalsim.SimInspiralWaveformParamsInsertModeArray(lp, ma)
            ahp, ahc = lalsim.SimInspiralChooseFDWaveform(
                bc["mass1"]*lal.MSUN_SI, bc["mass2"]*lal.MSUN_SI, 0., 0., bc["chi1"],
                0., 0., bc["chi2"], bc["distance"]*1e6*lal.PC_SI, bc["inclination"], bc["phic"],
                0., 0., 0., bc["df"], bc["flow"], bc["fhigh"], bc["f_ref"], lp,
                lalsim.GetApproximantFromString("IMRPhenomHM"))
            thp, thc = get_fd_waveform(approximant="IMRPhenomHM", mass1=bc["mass1"], mass2=bc["mass2"],
                chi1=bc["chi1"], chi2=bc["chi2"], distance=bc["distance"], inclination=bc["inclination"],
                phic=bc["phic"], tc=bc["tc"], f_ref=bc["f_ref"], sample_frequencies=torch.tensor(fe), modes=[(2, 2)])
            binary["cases"][-1]["reconstruction_control"].update({
                "22_only_hp_binary_relative_l2": float(np.linalg.norm(thp.detach().numpy()-ahp.data.data[keep]) / np.linalg.norm(ahp.data.data[keep])),
                "22_only_hc_binary_relative_l2": float(np.linalg.norm(thc.detach().numpy()-ahc.data.data[keep]) / np.linalg.norm(ahc.data.data[keep])),
            })
            # Physical single-mode controls: each call uses the documented
            # ModeArray API and the same distance, inclination, phase, f_ref,
            # and frequency grid.  These avoid assuming a normalization for
            # the unprojected h_lm returned above.
            mode_names = [f"{l}{m}" for l, m in IMRPhenomHM.SUPPORTED_MODES]
            def binary_polarization(active):
                am = lalsim.SimInspiralCreateModeArray()
                for l, m in active:
                    lalsim.SimInspiralModeArrayActivateMode(am, l, m)
                pd = lal.CreateDict()
                lalsim.SimInspiralWaveformParamsInsertModeArray(pd, am)
                return lalsim.SimInspiralChooseFDWaveform(
                    bc["mass1"]*lal.MSUN_SI, bc["mass2"]*lal.MSUN_SI, 0., 0., bc["chi1"],
                    0., 0., bc["chi2"], bc["distance"]*1e6*lal.PC_SI, bc["inclination"], bc["phic"],
                    0., 0., 0., bc["df"], bc["flow"], bc["fhigh"], bc["f_ref"], pd,
                    lalsim.GetApproximantFromString("IMRPhenomHM"))
            physical = {}
            native_single_hp, native_single_hc = [], []
            binary_single_hp, binary_single_hc = [], []
            for mode in IMRPhenomHM.SUPPORTED_MODES:
                bh, bcw = binary_polarization([mode])
                nh, nc = get_fd_waveform(approximant="IMRPhenomHM", mass1=bc["mass1"], mass2=bc["mass2"],
                    chi1=bc["chi1"], chi2=bc["chi2"], distance=bc["distance"], inclination=bc["inclination"],
                    phic=bc["phic"], tc=bc["tc"], f_ref=bc["f_ref"], sample_frequencies=torch.tensor(fe), modes=[mode])
                n_h, n_c = nh.detach().numpy(), nc.detach().numpy()
                p_h, p_c = bh.data.data[keep], bcw.data.data[keep]
                physical[f"{mode[0]}{mode[1]}"] = {
                    "hp_amplitude_relative_l2": float(np.linalg.norm(np.abs(n_h)-np.abs(p_h))/np.linalg.norm(np.abs(p_h))),
                    "hc_amplitude_relative_l2": float(np.linalg.norm(np.abs(n_c)-np.abs(p_c))/np.linalg.norm(np.abs(p_c))),
                    "hp_raw_complex_relative_l2": float(np.linalg.norm(n_h-p_h)/np.linalg.norm(p_h)),
                    "hc_raw_complex_relative_l2": float(np.linalg.norm(n_c-p_c)/np.linalg.norm(p_c)),
                }
                native_single_hp.append(n_h); native_single_hc.append(n_c)
                binary_single_hp.append(p_h); binary_single_hc.append(p_c)
            all_mode = IMRPhenomHM.SUPPORTED_MODES
            bh_all, bc_all = binary_polarization(all_mode)
            nh_all, nc_all = get_fd_waveform(approximant="IMRPhenomHM", mass1=bc["mass1"], mass2=bc["mass2"],
                chi1=bc["chi1"], chi2=bc["chi2"], distance=bc["distance"], inclination=bc["inclination"],
                phic=bc["phic"], tc=bc["tc"], f_ref=bc["f_ref"], sample_frequencies=torch.tensor(fe), modes=all_mode)
            ns_hp, ns_hc = np.sum(native_single_hp, axis=0), np.sum(native_single_hc, axis=0)
            bs_hp, bs_hc = np.sum(binary_single_hp, axis=0), np.sum(binary_single_hc, axis=0)
            physical["all"] = {
                "hp_amplitude_relative_l2": float(np.linalg.norm(np.abs(nh_all.numpy())-np.abs(bh_all.data.data[keep]))/np.linalg.norm(np.abs(bh_all.data.data[keep]))),
                "hc_amplitude_relative_l2": float(np.linalg.norm(np.abs(nc_all.numpy())-np.abs(bc_all.data.data[keep]))/np.linalg.norm(np.abs(bc_all.data.data[keep]))),
                "hp_raw_complex_relative_l2": float(np.linalg.norm(nh_all.numpy()-bh_all.data.data[keep])/np.linalg.norm(bh_all.data.data[keep])),
                "hc_raw_complex_relative_l2": float(np.linalg.norm(nc_all.numpy()-bc_all.data.data[keep])/np.linalg.norm(bc_all.data.data[keep])),
                "native_sum_single_vs_all_hp_relative_l2": float(np.linalg.norm(ns_hp-nh_all.numpy())/np.linalg.norm(nh_all.numpy())),
                "native_sum_single_vs_all_hc_relative_l2": float(np.linalg.norm(ns_hc-nc_all.numpy())/np.linalg.norm(nc_all.numpy())),
                "binary_sum_single_vs_all_hp_relative_l2": float(np.linalg.norm(bs_hp-bh_all.data.data[keep])/np.linalg.norm(bh_all.data.data[keep])),
                "binary_sum_single_vs_all_hc_relative_l2": float(np.linalg.norm(bs_hc-bc_all.data.data[keep])/np.linalg.norm(bc_all.data.data[keep])),
            }
            # Equation-level PN check: compare the clean Eq. (4) beta factor
            # with native mode amplitude divided by native 22 at mapped fA.
            # This is a black-box comparison, with no fitted scale/constants.
            Msec = (bc["mass1"] + bc["mass2"]) * lal.MTSUN_SI
            eta = bc["mass1"] * bc["mass2"] / (bc["mass1"] + bc["mass2"])**2
            delta = (bc["mass1"] - bc["mass2"]) / (bc["mass1"] + bc["mass2"])
            _, _, tw_all = IMRPhenomHM()(sample_frequencies=torch.tensor(fe), mass1=bc["mass1"], mass2=bc["mass2"],
                chi1=bc["chi1"], chi2=bc["chi2"], distance=bc["distance"], inclination=bc["inclination"],
                phic=bc["phic"], tc=bc["tc"], f_ref=bc["f_ref"], return_unprojected_modes=True)
            pn_diag = {}
            bands = [(20., 40.), (40., 80.), (80., 160.), (160., 320.)]
            for mode in IMRPhenomHM.SUPPORTED_MODES:
                l, m = mode
                if mode == (2, 2):
                    continue
                frlm, _ = IMRPhenomHM().mode_qnm_frequencies(l, m, torch.tensor(bc["mass1"]), torch.tensor(bc["mass2"]), torch.tensor(bc["chi1"]), torch.tensor(bc["chi2"]))
                fr22, _ = IMRPhenomHM().mode_qnm_frequencies(2, 2, torch.tensor(bc["mass1"]), torch.tensor(bc["mass2"]), torch.tensor(bc["chi1"]), torch.tensor(bc["chi2"]))
                frlm, fr22 = float(frlm), float(fr22)
                f0 = 0.018 * frlm / fr22
                fg = torch.tensor(fe * Msec, dtype=torch.float64)
                fa = map_amplitude_frequency(fg, f0, frlm, fr22, m).numpy() / Msec
                native_ref = np.interp(fa, fe, np.abs(tw_all[(2, 2)].detach().numpy()))
                v = (2.0 * np.pi * Msec * fe / m) ** (1.0 / 3.0)
                # Mishra Eq. (5): V_m(x)=(2*pi*M*x/m)^(1/3), for every
                # Hhat_lm argument in London's Eq. (4).
                va = (2.0 * np.pi * Msec * fa / m) ** (1.0 / 3.0)
                hfa = pn_hhat_lm(mode, torch.tensor(va), eta, delta, (bc["chi1"]+bc["chi2"])/2, (bc["chi1"]-bc["chi2"])/2).numpy()
                hf = pn_hhat_lm(mode, torch.tensor(v), eta, delta, (bc["chi1"]+bc["chi2"])/2, (bc["chi1"]-bc["chi2"])/2).numpy()
                v2 = (2.0 * np.pi * Msec * (2.0 * fe / m) / m) ** (1.0 / 3.0)
                h2 = pn_hhat_lm(mode, torch.tensor(v2), eta, delta, (bc["chi1"]+bc["chi2"])/2, (bc["chi1"]-bc["chi2"])/2).numpy()
                # Mishra Eq. (11) supplies the common SPA factor V_m^(-7/2)
                # for each Hhat_lm argument in London's Eq. (4).
                spa_fa = va ** (-3.5)
                spa_22_fa = (2.0 * np.pi * Msec * fa / 2.0) ** (-3.5 / 3.0)
                spa_f = v ** (-3.5)
                spa_2 = v2 ** (-3.5)
                beta = np.abs((hfa * spa_fa) / (-spa_22_fa) * (hf * spa_f) / (h2 * spa_2))
                native_ratio = np.abs(tw_all[mode].detach().numpy()) / native_ref
                band_values = {}
                for lo, hi in bands:
                    mask = (fe >= lo) & (fe < hi) & np.isfinite(beta) & (np.abs(h2) > 0)
                    band_values[f"{lo:g}-{hi:g}Hz"] = float(np.median(np.abs(native_ratio[mask] - beta[mask]))) if np.any(mask) else None
                pn_diag[f"{l}{m}"] = {"absolute_beta_minus_native_ratio_median_by_band": band_values}
            binary["cases"][-1]["pn_amplitude_diagnostic"] = {
                "provenance": "London arXiv:1708.00404 Eq. (4) and its text; Mishra et al. PRD 93 084054 (2016) Eq. (12)",
                "bands_hz": bands, "values": pn_diag,
            }
            binary["cases"][-1]["physical_mode_array_comparison"] = {
                "scope": "absolute physical hp/hc outputs from documented ModeArray calls; no mode normalization, alignment, or fitted scale",
                "modes": physical,
            }
    except Exception as exc:
        binary["error"] = {"type": type(exc).__name__, "message": str(exc),
                            "api": "SimIMRPhenomHMGethlmModes"}
    out.write_text(json.dumps({"provenance": {"paper": "London et al., arXiv:1708.00404, Eqs. (5)-(6), (13)-(14)",
        "source_sha256": _hash(src), "script_sha256": _hash(pathlib.Path(__file__)),
        "paper_mishra": "Mishra et al., arXiv:1601.05588 / PRD 93 084054 (2016), Eq. (12)",
        "native_source_sha256": {name: _hash(path) for name, path in native_sources.items()},
        "case_sha256": case_hash,
        "binary_case_sha256": binary_case_hash,
        "python": platform.python_version(), "torch": torch.__version__},
        "grid": {"start": 0.0, "stop": 0.16, "count": len(f)}, "rows": rows,
        "case": case, "mode_outputs": modes, "binary_mode_comparison": binary}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
