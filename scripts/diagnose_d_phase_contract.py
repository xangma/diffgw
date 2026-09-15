"""Black-box audit of IMRPhenomD's returned phase/polarization contract.

This script intentionally uses only the public callable and helper APIs.  It
does not inspect implementation source (the imported module is hashed only).
"""
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD

MTSUN_SI = 4.925490947641267e-6
CASES = [
    (40.0, 15.0, 0.3, -0.2, 20.0),
    (55.0, 11.0, -0.4, 0.2, 27.0),
    (32.0, 18.0, -0.6, 0.45, 25.0),
    (70.0, 8.0, 0.1, -0.5, 30.0),
]


def cphase(z):
    """Wrapped phase residual of a unit-complex identity."""
    return np.angle(np.asarray(z))


def maxerr(z):
    return float(np.max(np.abs(cphase(z))))


def main():
    d = IMRPhenomD()
    rows = []
    for m1, m2, s1, s2, fref in CASES:
        mt = m1 + m2
        mf = 0.2 / (mt * MTSUN_SI)
        # Stay strictly inside the analytic supported band and include f_ref.
        f = np.unique(np.r_[fref, np.geomspace(20.0, mf * (1.0 - 1e-9), 4095)])
        ft = torch.tensor(f, dtype=torch.float64)
        kw = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                  distance=1000.0, inclination=0.0, f_ref=fref,
                  return_phase=True)
        hp0, hc0, p0 = d(ft, phic=0.0, tc=0.0, **kw)
        hp1, hc1, p1 = d(ft, phic=1.234567, tc=0.0, **kw)
        hpt, hct, pt = d(ft, phic=0.0, tc=0.371234, **kw)
        hpfull, hcfull, pfull = d(ft, phic=1.234567, tc=0.371234, **kw)
        # The phase returned by the callable is the corrected carrier:
        # q(f)=phi(f)-phi(f_ref)-slope_peak*(Mf-Mf_ref), then
        # p=q - 2 phic + 2 pi f tc.  Polarizations use exp(-i(q-2 phic)).
        slope = float(d.compute_slope_peak(m1, m2, s1, s2))
        # q is directly p0, because phic=tc=0 and the callable references fref.
        q = p0.detach().numpy()
        phase_phic_err = maxerr(np.exp(1j * (p1.detach().numpy() - p0.detach().numpy() + 2.0 * 1.234567)))
        phase_tc_err = maxerr(np.exp(1j * (pt.detach().numpy() - p0.detach().numpy() - 2.0 * np.pi * f * 0.371234)))
        pol_phic_err = max(maxerr(hp1.detach().numpy() / hp0.detach().numpy() * np.exp(-2j * 1.234567)),
                           maxerr(hc1.detach().numpy() / hc0.detach().numpy() * np.exp(-2j * 1.234567)))
        pol_tc_err = max(maxerr(hpt.detach().numpy() / hp0.detach().numpy() * np.exp(2j * np.pi * f * 0.371234)),
                         maxerr(hct.detach().numpy() / hc0.detach().numpy() * np.exp(2j * np.pi * f * 0.371234)))
        # Direct identity at nonzero phic and tc, and same identity at iota=.8.
        direct_full_err = max(maxerr(hpfull.detach().numpy() / np.abs(hpfull.detach().numpy()) * np.exp(1j * pfull.detach().numpy())),
                              maxerr(hcfull.detach().numpy() / np.abs(hcfull.detach().numpy()) * np.exp(1j * pfull.detach().numpy()) * 1j))
        inc = dict(kw, inclination=0.8)
        hpi, hci, pi = d(ft, phic=1.234567, tc=0.371234, **inc)
        inclination_err = max(maxerr(hpi.detach().numpy() / np.abs(hpi.detach().numpy()) * np.exp(1j * pi.detach().numpy())),
                              maxerr(hci.detach().numpy() / np.abs(hci.detach().numpy()) * np.exp(1j * pi.detach().numpy()) * 1j))
        # Direct carrier check, avoiding arbitrary fitted offsets/slopes.
        carrier_err = max(maxerr(hp0.detach().numpy() / np.abs(hp0.detach().numpy()) * np.exp(1j * q)),
                          maxerr(hc0.detach().numpy() / np.abs(hc0.detach().numpy()) * np.exp(1j * q) * (1j)))
        wrong_plus_sign = maxerr(hp0.detach().numpy() / np.abs(hp0.detach().numpy()) * np.exp(-1j * q))
        wrong_unreferenced_tc = maxerr(hpt.detach().numpy() / hp0.detach().numpy())
        rows.append({
            "case": {"m1": m1, "m2": m2, "chi1": s1, "chi2": s2, "f_ref": fref},
            "n_frequency": int(f.size), "f_max_hz": float(f.max()),
            "total_mass_time_s": mt * MTSUN_SI, "slope_peak": slope,
            "equation": "q=phi(f)-phi(f_ref)-slope_peak*(M f-M f_ref); p=q-2*phic+2*pi*f*tc; hp,hc carrier=exp(-i*p), including the tc rotation exp(-2*pi*i*f*tc)",
            "max_abs_wrapped_rad": {"phase_phic": phase_phic_err, "phase_tc": phase_tc_err,
                                    "polarization_phic": pol_phic_err, "polarization_tc": pol_tc_err,
                                    "direct_carrier": carrier_err, "direct_nonzero_phic_tc": direct_full_err,
                                    "inclination_0p8": inclination_err,
                                    "wrong_exp_plus_i_q": wrong_plus_sign,
                                    "wrong_no_tc_rotation": wrong_unreferenced_tc},
            "exact_below_1e-10": all(x < 1e-10 for x in (phase_phic_err, phase_tc_err, pol_phic_err, pol_tc_err, carrier_err, direct_full_err, inclination_err)),
        })
    mod_path = pathlib.Path(inspect.getfile(IMRPhenomD))
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = None
    report = {"api_signature": str(inspect.signature(IMRPhenomD.__call__)),
              "slope_signature": str(inspect.signature(IMRPhenomD.compute_slope_peak)),
              "torch_version": torch.__version__, "python": sys.version,
              "native_module_path": str(mod_path),
              "native_module_sha256": hashlib.sha256(mod_path.read_bytes()).hexdigest(),
              "git_head": git,
              "published_reference": "https://arxiv.org/html/1508.07253",
              "MTSUN_SI_used": MTSUN_SI, "cases": rows}
    out = pathlib.Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row["exact_below_1e-10"] for row in rows):
        raise SystemExit("phase contract assertion failed")


if __name__ == "__main__":
    main()
