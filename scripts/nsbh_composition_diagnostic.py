"""Black-box NSBH/PhenomD composition and affine-phase diagnostic."""

import hashlib
import inspect
import json
import math
import pathlib
import platform
import sys

import lal
import lalsimulation as ls
import numpy as np
import torch

from torchwave import get_fd_waveform
from torchwave.approximants import imrphenom_nsbh, imrphenomd


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(approximant, case, f_low, f_final, df, f_ref, lam):
    m1, m2, chi1, inc, phic = case[:5]
    d = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertTidalLambda1(d, 0.0)
    ls.SimInspiralWaveformParamsInsertTidalLambda2(d, lam)
    approx = {"NSBH": ls.IMRPhenomNSBH, "D": ls.IMRPhenomD}[approximant]
    args = [m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0., chi1, 0., 0., 0.,
            100. * 1e6 * lal.PC_SI, inc, phic, 0., 0., 0., df, f_low,
            f_final, f_ref, d, approx]
    hp, _ = ls.SimInspiralChooseFDWaveform(*args)
    return np.asarray(hp.data.data, dtype=np.complex128)


def native(approximant, case, freqs, f_low, f_final, f_ref, lam):
    m1, m2, chi1, inc, phic = case[:5]
    out, _ = get_fd_waveform(
        "IMRPhenomNSBH" if approximant == "NSBH" else "IMRPhenomD",
        mass1=m1, mass2=m2, chi1=chi1, chi2=0., lambda1=0., lambda2=lam,
        distance=100., inclination=inc, phic=phic, f_ref=f_ref,
        f_lower=f_low, f_final=f_final, sample_frequencies=torch.tensor(freqs, dtype=torch.float64),
        dtype=torch.float64)
    return out.detach().numpy()


def metrics(a, b, freqs, f_low):
    common = (freqs >= f_low) & (np.abs(a) > 0) & (np.abs(b) > 0) & np.isfinite(a) & np.isfinite(b)
    aa, bb, ff = a[common], b[common], freqs[common]
    scale = np.sqrt(np.mean(np.abs(bb) ** 2))
    amp = np.sqrt(np.mean((np.abs(aa) - np.abs(bb)) ** 2)) / scale
    cx = np.sqrt(np.mean(np.abs(aa - bb) ** 2)) / scale
    phase = np.unwrap(np.angle(aa / bb))
    coeff = np.linalg.lstsq(np.c_[np.ones(len(ff)), ff], phase, rcond=None)[0]
    residual = phase - np.c_[np.ones(len(ff)), ff] @ coeff
    return {"n": int(len(ff)), "amplitude_l2": float(amp), "complex_l2": float(cx),
            "max_abs_normalized": float(np.max(np.abs(aa - bb)) / scale),
            "affine_intercept_rad": float(coeff[0]), "affine_slope_rad_per_hz": float(coeff[1]),
            "nonaffine_phase_rms_rad": float(np.sqrt(np.mean(residual ** 2))),
            "nonaffine_phase_max_rad": float(np.max(np.abs(residual)))}


def main():
    cases = [("catalog", (5., 1.4, .2, .5, .3), 30., 400.),
             ("holdout", (7., 1.4, .4, .9, .3), 30., 2600.)]
    records = []
    grid_records = []
    for name, case, f_low, lam in cases:
        for df in (.25, .5, 1.):
            for f_final in (512., 1024.):
                freqs = np.arange(int(round(f_final / df)) + 1, dtype=float) * df
                refs = (30., 27., 60.) if name == "catalog" else (40., 27., 60.)
                for f_ref in refs:
                    vals = {}
                    for backend in ("native", "binary"):
                        for model in ("NSBH", "D"):
                            # Public PhenomD has no tidal parameters; its
                            # Lambda-restored entry is therefore its same
                            # Lambda=0 carrier baseline.
                            lambda_cases = (("lambda0", 0.), ("lambda", lam)) if model == "NSBH" else (("lambda0", 0.), ("lambda", 0.))
                            for lkey, lv in lambda_cases:
                                vals[(backend, model, lkey)] = (
                                    native(model, case, freqs, f_low, f_final, f_ref, lv)
                                    if backend == "native" else
                                    binary(model, case, f_low, f_final, df, f_ref, lv)[:len(freqs)])
                    row = {"case": name, "physical": {"m1_msun": case[0], "m2_msun": case[1],
                             "chi_bh": case[2], "inclination": case[3], "phic": case[4],
                             "lambda2": lam}, "f_low": f_low, "f_final": f_final,
                           "df": df, "f_ref": f_ref, "metrics": {}}
                    for backend in ("native", "binary"):
                        row["metrics"][backend + "_nsbh_vs_d_lambda0"] = metrics(
                            vals[(backend, "NSBH", "lambda0")], vals[(backend, "D", "lambda0")], freqs, f_low)
                        row["metrics"][backend + "_nsbh_vs_d_lambda"] = metrics(
                            vals[(backend, "NSBH", "lambda")], vals[(backend, "D", "lambda")], freqs, f_low)
                        nsbh0 = vals[(backend, "NSBH", "lambda0")]
                        nsbhl = vals[(backend, "NSBH", "lambda")]
                        carrier = vals[(backend, "D", "lambda0")]
                        nsbh_ratio = np.full_like(nsbhl, np.nan)
                        carrier_ratio = np.full_like(carrier, np.nan)
                        np.divide(nsbhl, nsbh0, out=nsbh_ratio, where=np.abs(nsbh0) > 0)
                        np.divide(carrier, carrier, out=carrier_ratio, where=np.abs(carrier) > 0)
                        row["metrics"][backend + "_tidal_ratio_vs_carrier"] = metrics(
                            nsbh_ratio, carrier_ratio, freqs, f_low)
                    # Cross-backend isolation on the identical frequency grid.
                    for lkey in ("lambda0", "lambda"):
                        row["metrics"]["cross_native_binary_nsbh_" + lkey] = metrics(
                            vals[("native", "NSBH", lkey)], vals[("binary", "NSBH", lkey)], freqs, f_low)
                    row["metrics"]["cross_native_binary_d_lambda0"] = metrics(
                        vals[("native", "D", "lambda0")], vals[("binary", "D", "lambda0")], freqs, f_low)
                    native_comp = np.full_like(vals[("native", "NSBH", "lambda0")], np.nan)
                    binary_comp = np.full_like(native_comp, np.nan)
                    np.divide(vals[("native", "NSBH", "lambda0")], vals[("native", "D", "lambda0")], out=native_comp,
                              where=np.abs(vals[("native", "D", "lambda0")]) > 0)
                    np.divide(vals[("binary", "NSBH", "lambda0")], vals[("binary", "D", "lambda0")], out=binary_comp,
                              where=np.abs(vals[("binary", "D", "lambda0")]) > 0)
                    row["metrics"]["cross_native_binary_nsbh_over_d_lambda0"] = metrics(
                        native_comp, binary_comp, freqs, f_low)
                    native_ratio = np.full_like(native_comp, np.nan)
                    binary_ratio = np.full_like(binary_comp, np.nan)
                    np.divide(vals[("native", "NSBH", "lambda")], vals[("native", "NSBH", "lambda0")], out=native_ratio,
                              where=np.abs(vals[("native", "NSBH", "lambda0")]) > 0)
                    np.divide(vals[("binary", "NSBH", "lambda")], vals[("binary", "NSBH", "lambda0")], out=binary_ratio,
                              where=np.abs(vals[("binary", "NSBH", "lambda0")]) > 0)
                    row["metrics"]["cross_native_binary_tidal_ratio"] = metrics(
                        native_ratio, binary_ratio, freqs, f_low)
                    records.append(row)
        # Evaluate grid variants on explicitly identical frequency samples.
        refs = (30., 27., 60.) if name == "catalog" else (40., 27., 60.)
        for f_ref in refs:
            for backend in ("native", "binary"):
                for model in ("NSBH", "D"):
                    for lkey, lv in (("lambda0", 0.), ("lambda", lam)):
                        if model == "D" and lkey == "lambda":
                            lv = 0.
                        def get(freqs, final, spacing, decimate=1):
                            # Match binary semantics: evaluate native on its
                            # complete requested grid, then select samples.
                            full = np.arange(int(round(final / spacing)) + 1, dtype=float) * spacing
                            if backend == "native":
                                out = native(model, case, full, f_low, final, f_ref, lv)
                            else:
                                out = binary(model, case, f_low, final, spacing, f_ref, lv)
                            indices = np.rint(freqs / spacing).astype(int)
                            return out[indices]
                        # Different df, same explicit samples and endpoint.
                        common = np.arange(int(512. / .5), dtype=float) * .5
                        common = common[common >= f_low]
                        a, b = get(common, 512., .25, 2), get(common, 512., .5)
                        grid_records.append({"case": name, "f_ref": f_ref, "backend": backend,
                            "model": model, "lambda_case": lkey, "comparison": "df_common_samples",
                            "grid_a": {"df": .25, "f_final": 512.}, "grid_b": {"df": .5, "f_final": 512.},
                            "metrics": metrics(a, b, common, f_low)})
                        # Different f_final, fixed df, restricted to f<512.
                        common = np.arange(int(512. / .25), dtype=float) * .25
                        common = common[(common >= f_low) & (common < 512.)]
                        a, b = get(common, 512., .25), get(common, 1024., .25)
                        grid_records.append({"case": name, "f_ref": f_ref, "backend": backend,
                            "model": model, "lambda_case": lkey, "comparison": "f_final_common_samples",
                            "grid_a": {"df": .25, "f_final": 512.}, "grid_b": {"df": .25, "f_final": 1024.},
                            "restricted_frequency": "30<=f<512", "metrics": metrics(a, b, common, f_low)})
    source_paths = ("src/torchwave/approximants/nsbh_clean_phase.py", "src/torchwave/approximants/imrphenom_nsbh.py",
                    "src/torchwave/approximants/imrphenomd.py", "scripts/nsbh_composition_diagnostic.py")
    source_hashes_after = {p: sha256(ROOT / p) for p in source_paths}
    source_hashes_before = {
        "src/torchwave/approximants/nsbh_clean_phase.py": "a887f592b32a40a1d4903ce32bb97859bd06f5fa54e72396cb75c4d5f1fd90ec",
        "src/torchwave/approximants/imrphenom_nsbh.py": "5bd3750cdaba9ee4e0f7a469598676aadf1054e9de889878e47e72b42ac4b6eb",
        "src/torchwave/approximants/imrphenomd.py": "56c5c5c5ba40b73cf6ac73e69b8d6913678c10e7ca6aa3c2f72dbf06cebf055c",
        "scripts/nsbh_composition_diagnostic.py": "fb5eeb133a40db255a1385fe5d4ce3142607b1c646702e52877e204dfb01ab33",
    }
    result = json.dumps({"metadata": {"python": PYTHON, "platform": platform.platform(),
        "lal_version": lal.__version__, "lalsimulation_version": ls.__version__,
        "get_fd_waveform_signature": str(inspect.signature(get_fd_waveform)),
        "source_sha256": source_hashes_after, "source_sha256_before": source_hashes_before,
        "source_sha256_after": source_hashes_after},
        "records": records, "common_sample_grid_records": grid_records}, indent=2)
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
