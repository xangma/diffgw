"""Compact black-box NSBH phase residual report for frozen public cases."""
import hashlib, json, pathlib, platform, sys

import lal
import lalsimulation as ls
import numpy as np
import torch

from torchwave import get_fd_waveform
from torchwave.approximants import imrphenomd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = {
    "catalog": dict(m1=5., m2=1.4, chi1=.2, lam=400., fref=30., inc=.5),
    "holdout": dict(m1=7., m2=1.4, chi1=.4, lam=2600., fref=40., inc=.9),
    "independent_b": dict(m1=8., m2=1.5, chi1=.25, lam=1200., fref=45., inc=.5),
}
MSUN_S = lal.G_SI * lal.MSUN_SI / lal.C_SI**3


def _native(c, f, lam):
    return get_fd_waveform(
        "IMRPhenomNSBH", mass1=c["m1"], mass2=c["m2"], chi1=c["chi1"],
        chi2=0., lambda1=0., lambda2=lam, distance=100.,
        inclination=c["inc"], phic=.3, f_ref=c["fref"], f_lower=30.,
        f_final=1024., sample_frequencies=torch.tensor(f, dtype=torch.float64),
        dtype=torch.float64,
    )


def _binary(c, f, lam):
    d = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertTidalLambda1(d, 0.)
    ls.SimInspiralWaveformParamsInsertTidalLambda2(d, lam)
    args = [c["m1"]*lal.MSUN_SI, c["m2"]*lal.MSUN_SI, 0., 0., c["chi1"], 0.,
            0., 0., 100e6*lal.PC_SI, c["inc"], .3, 0., 0., 0., .25, 30.,
            1024., c["fref"], d, ls.IMRPhenomNSBH]
    hp, hc = ls.SimInspiralChooseFDWaveform(*args)
    return np.asarray(hp.data.data, dtype=np.complex128), np.asarray(hc.data.data, dtype=np.complex128)


def _metric(a, b, f):
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    if not finite_a.all() or not finite_b.all():
        raise ValueError("nonfinite waveform sample in diagnostic metric")
    support = f >= 30.
    nz = support & (np.abs(a) > 0) & (np.abs(b) > 0)
    phase = np.unwrap(np.angle(a[nz] / b[nz]))
    x = f[nz]
    coeff = np.polyfit(x, phase, 1)
    residual = phase - np.polyval(coeff, x)
    scale = np.sqrt(np.mean(np.abs(b[support])**2))
    return {
        "support_bins": int(np.count_nonzero(support)),
        "support_mismatch_bins": int(np.count_nonzero(support & ((a == 0) != (b == 0)))),
        "raw_relative_l2": float(np.sqrt(np.mean(np.abs(a[support]-b[support])**2))/scale),
        "magnitude_relative_l2": float(np.sqrt(np.mean((np.abs(a[support])-np.abs(b[support]))**2))/scale),
        "unit_phase_l2": float(np.sqrt(np.mean(np.abs(a[nz]/np.abs(a[nz])-b[nz]/np.abs(b[nz]))**2))),
        "phase_intercept_rad": float(coeff[1]),
        "phase_slope_rad_per_hz": float(coeff[0]),
        "phase_nonaffine_rms_rad": float(np.sqrt(np.mean(residual**2))),
        "fixed_reference_shift_removed_rad": 0.0,
    }


def main(path="docs/nsbh_phase_residual_diagnostic.json"):
    f = np.arange(1024./.25 + 1.) * .25
    records = []
    for name, c in CASES.items():
        for lam in (0., c["lam"]):
            n = _native(c, f, lam)
            b = _binary(c, f, lam)
            records.append({"case": name, "lambda2": lam, "f_ref": c["fref"],
                            "grid": {"flow": 30., "f_final": 1024., "df": .25},
                            "fixed_reference_shift_note": "same f_ref/phic; no shift applied",
                            "hp": _metric(n[0].detach().numpy(), b[0], f),
                            "hc": _metric(n[1].detach().numpy(), b[1], f)})
    inv = {"IMRPhenomD.remnant_properties": str(__import__("inspect").signature(imrphenomd.IMRPhenomD.remnant_properties)),
           "doc": imrphenomd.IMRPhenomD.remnant_properties.__doc__}
    out = {"metadata": {"python": sys.executable, "platform": platform.platform(),
                         "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                           for p in ("scripts/nsbh_phase_residual_diagnostic.py",
                                                     "src/torchwave/approximants/imrphenom_nsbh.py",
                                                     "src/torchwave/approximants/imrphenomd.py")},
                         "public_phenoD_inventory": inv}, "records": records}
    pathlib.Path(path).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/nsbh_phase_residual_diagnostic.json")
