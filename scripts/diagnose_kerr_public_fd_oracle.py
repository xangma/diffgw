#!/usr/bin/env python3
"""Convergence evidence for the public Kerr TD-to-FD adapter."""
from __future__ import annotations
import hashlib, importlib.metadata, json, math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
from kerr_public_oracle import public_ringdown_fd, public_ringdown_fd_converged

CASES = [
    {"name": "frozen", "mass": 65., "spin": .7, "distance": 100., "inclination": 0., "phase": 0.},
    {"name": "inclined_phase", "mass": 65., "spin": .7, "distance": 100., "inclination": .8, "phase": .2},
    *[{"name": f"spin_{s:g}", "mass": 65., "spin": s, "distance": 100., "inclination": .8, "phase": .2}
      for s in (0., .1, .3, .9, .95)],
    {"name": "mass_scaled", "mass": 130., "spin": .7, "distance": 100., "inclination": .8, "phase": .2},
]

def rel(a, b, mask):
    den = np.linalg.norm(b[mask])
    return float(np.linalg.norm(a[mask] - b[mask]) / den) if den else None

def main():
    rows = []
    for case in CASES:
        inputs = {"mass_solar": case["mass"], "spin": case["spin"],
                  "distance_mpc": case["distance"], "inclination": case["inclination"],
                  "phase": case["phase"]}
        f0, hp0, hc0 = public_ringdown_fd(**inputs, sample_rate=65536, delta_f=.25, f_max=500.)
        f1, hp1, hc1 = public_ringdown_fd(**inputs, sample_rate=131072, delta_f=.25, f_max=500.)
        if not np.array_equal(f0, f1):
            raise RuntimeError("coarse/fine frequency grids differ")
        mask = (f0 >= 50.) & (f0 <= 500.)
        rows.append({"case": case, "frequency": {"first": float(f0[0]), "last": float(f0[-1]), "delta_f": .25,
                    "comparison_band": [50., 500.], "support_bins": int(mask.sum())},
                    "coarse_vs_fine": {"hp_complex_l2": rel(hp0, hp1, mask), "hc_complex_l2": rel(hc0, hc1, mask),
                                        "hp_magnitude_l2": rel(abs(hp0), abs(hp1), mask), "hc_magnitude_l2": rel(abs(hc0), abs(hc1), mask)},
                    "epoch_applied": True, "sample_rates": [65536, 131072]})
    out = {"scope": "public SimBlackHoleRingdown Simpson FD adapter; diagnostic only",
           "energy_fraction": .01, "quadrature": "append zero for even causal sample count; composite Simpson; rfft/(3*fs)",
           "epoch_factor": "exp(-2*pi*i*f*epoch)", "cases": rows,
           "versions": {"numpy": np.__version__, "torchwave": importlib.metadata.version("torchwave")},
           "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "adapter_hash": hashlib.sha256((Path(__file__).parents[1] / "tests" / "kerr_public_oracle.py").read_bytes()).hexdigest()}
    # Tail samples are recorded from the adapter's public callable separately
    # so truncation can be reviewed without treating them as model constants.
    import lal, lalsimulation
    tails = []
    for case in CASES:
        td = lalsimulation.SimBlackHoleRingdown(lal.LIGOTimeGPS(0), case["phase"], 1/65536,
              case["mass"]*lal.MSUN_SI, case["spin"], .01, case["distance"]*1e6*lal.PC_SI, case["inclination"], 2, 2)
        plus = np.asarray(td[0].data.data, float)
        cross = np.asarray(td[1].data.data, float)
        scale = max(float(np.max(np.abs(plus))), float(np.max(np.abs(cross))), np.finfo(float).tiny)
        tails.append({"case": case["name"], "epoch": float(td[0].epoch), "length": int(td[0].data.length),
                      "tail_samples_plus": plus[-8:].tolist(), "tail_samples_cross": cross[-8:].tolist(),
                      "tail_bound_estimate": float(max(np.max(np.abs(plus[-8:])), np.max(np.abs(cross[-8:]))) / scale)})
    out["public_tail_samples"] = tails
    _, _, _, convergence = public_ringdown_fd_converged(
        65., .7, 100., inclination=0., phase=0., sample_rate=32768,
        max_sample_rate=131072, delta_f=.25, f_max=500.)
    out["adaptive_convergence"] = convergence
    out["public_api"] = {"name": "SimBlackHoleRingdown", "doc": lalsimulation.SimBlackHoleRingdown.__doc__, "lal": lal.__version__, "lalsimulation": lalsimulation.__version__}
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({r["case"]["name"]: r["coarse_vs_fine"] for r in rows}, indent=2))

if __name__ == "__main__": main()
