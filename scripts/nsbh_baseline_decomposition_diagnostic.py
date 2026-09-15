"""Banded black-box decomposition of NSBH residuals against public binaries."""
import hashlib, inspect, json, pathlib, platform, sys
import lal
import lalsimulation as ls
import numpy as np
import torch
from torchwave import get_fd_waveform
from torchwave.approximants import imrphenomd, imrphenom_nsbh

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANDS = ((30., 60.), (60., 100.), (100., 200.), (200., 500.), (500., 1024.))
CASES = {
    "catalog": dict(m1=5., m2=1.4, chi1=.2, lam=400., fref=30., inc=.5),
    "independent_b": dict(m1=8., m2=1.5, chi1=.25, lam=1200., fref=45., inc=.5),
}


def native(name, c, f, lam=0.):
    kw = dict(mass1=c["m1"], mass2=c["m2"], chi1=c["chi1"], chi2=0.,
              distance=100., inclination=c["inc"], phic=.3, f_ref=c["fref"],
              f_lower=30., f_final=1024., sample_frequencies=torch.tensor(f, dtype=torch.float64),
              dtype=torch.float64)
    if name == "IMRPhenomNSBH":
        kw.update(lambda1=0., lambda2=lam)
    return get_fd_waveform(name, **kw)


def binary(name, c, f, lam=0.):
    d = lal.CreateDict()
    if name == "IMRPhenomNSBH":
        ls.SimInspiralWaveformParamsInsertTidalLambda1(d, 0.)
        ls.SimInspiralWaveformParamsInsertTidalLambda2(d, lam)
        approx = ls.IMRPhenomNSBH
    else:
        approx = ls.IMRPhenomD
    args = [c["m1"]*lal.MSUN_SI, c["m2"]*lal.MSUN_SI, 0., 0., c["chi1"], 0.,
            0., 0., 100e6*lal.PC_SI, c["inc"], .3, 0., 0., 0., .25, 30.,
            1024., c["fref"], d, approx]
    hp, hc = ls.SimInspiralChooseFDWaveform(*args)
    return np.asarray(hp.data.data, dtype=np.complex128), np.asarray(hc.data.data, dtype=np.complex128)


def metric(a, b, f, lo, hi):
    mask = (f >= lo) & (f < hi) & np.isfinite(a) & np.isfinite(b)
    nz = mask & (np.abs(a) > 0) & (np.abs(b) > 0)
    scale = np.sqrt(np.mean(np.abs(b[mask])**2))
    phase = np.unwrap(np.angle(a[nz] / b[nz]))
    coef = np.polyfit(f[nz], phase, 1) if np.count_nonzero(nz) > 2 else (np.nan, np.nan)
    rem = phase - np.polyval(coef, f[nz]) if np.count_nonzero(nz) > 2 else np.array([np.nan])
    return {"support_bins": int(mask.sum()),
            "support_mismatch_bins": int(np.count_nonzero(mask & ((a == 0) != (b == 0)))),
            "raw_relative_l2": float(np.sqrt(np.mean(np.abs(a[mask]-b[mask])**2))/scale),
            "magnitude_relative_l2": float(np.sqrt(np.mean((np.abs(a[mask])-np.abs(b[mask]))**2))/scale),
            "unit_phase_l2": float(np.sqrt(np.mean(np.abs(a[nz]/np.abs(a[nz])-b[nz]/np.abs(b[nz]))**2))),
            "phase_slope_rad_per_hz": float(coef[0]),
            "phase_nonaffine_rms_rad": float(np.sqrt(np.mean(rem**2))),
            "fixed_reference_shift_removed_rad": 0.0}


def main(path="docs/nsbh_baseline_decomposition_diagnostic.json"):
    f = np.arange(1024./.25 + 1.) * .25
    records = []
    for name, c in CASES.items():
        nd = native("IMRPhenomD", c, f)
        bd = binary("IMRPhenomD", c, f)
        for lam in (0., c["lam"]):
            nn = native("IMRPhenomNSBH", c, f, lam)
            bn = binary("IMRPhenomNSBH", c, f, lam)
            pairs = {"native_D_vs_binary_D": (nd, bd),
                     "native_NSBH_vs_binary_NSBH": (nn, bn),
                     "native_NSBH_vs_native_D": (nn, nd),
                     "binary_NSBH_vs_binary_D": (bn, bd)}
            for pair, (x, y) in pairs.items():
                for pol, i in (("hp", 0), ("hc", 1)):
                    for lo, hi in BANDS:
                        records.append({"case": name, "lambda2": lam, "pair": pair,
                                        "polarization": pol, "band_hz": [lo, hi],
                                        "metric": metric(np.asarray(x[i].detach().numpy() if isinstance(x[i], torch.Tensor) else x[i]),
                                                         np.asarray(y[i].detach().numpy() if isinstance(y[i], torch.Tensor) else y[i]), f, lo, hi)})
    inv = {}
    for module in (imrphenomd, imrphenom_nsbh):
        for n in dir(module):
            if "phase" in n.lower() or "phi" in n.lower():
                obj = getattr(module, n)
                try: inv[f"{module.__name__}.{n}"] = {"signature": str(inspect.signature(obj)), "doc": inspect.getdoc(obj)}
                except (TypeError, ValueError): pass
    out = {"metadata": {"python": sys.executable, "platform": platform.platform(),
                         "grid": {"flow": 30., "f_final": 1024., "df": .25},
                         "fixed_reference_shift_note": "same public f_ref/phic; no shift applied",
                         "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                           ("scripts/nsbh_baseline_decomposition_diagnostic.py",
                                            "src/torchwave/approximants/imrphenom_nsbh.py",
                                            "src/torchwave/approximants/imrphenomd.py")},
                         "module_phase_callable_inventory": inv}, "records": records}
    pathlib.Path(path).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "docs/nsbh_baseline_decomposition_diagnostic.json")
