#!/usr/bin/env python3
"""Exercise the unrouted clean ROM glue on complete and independent grids."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import lal
import lalsimulation as ls
import torch
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from torchwave.approximants.rom.bspline import load_official_rom
from torchwave.approximants.rom.glue import OfficialROMGlue

ROM = Path("/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5")
MSUN_SEC = 4.925490947641267e-6

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def qnm(case):
    m1, m2, s1, s2 = case
    v = lal.CreateCOMPLEX16Vector(1)
    ls.SimIMREOBGenerateQNMFreqV2(v, m1, m2, [0., 0., s1], [0., 0., s2], 2, 2, 1,
                                  ls.GetApproximantFromString("SEOBNRv4"))
    public_mf = float((m1 + m2) * MSUN_SEC * v.data[0].real / (2 * torch.pi))
    _, final_fraction, final_spin = ls.SimIMREOBFinalMassSpin(
        m1, m2, [0., 0., s1], [0., 0., s2], ls.GetApproximantFromString("SEOBNRv4"))
    independent = float(signed_qnm_220(torch.tensor(final_spin, dtype=torch.float64))[0]) * final_fraction / (2 * 3.141592653589793)
    return public_mf, independent, float(final_fraction), float(final_spin)

def run_case(glue, case, frequencies):
    m1, m2, s1, s2 = case
    rd, independent_rd, final_fraction, final_spin = qnm(case)
    case_glue = OfficialROMGlue(glue.low, glue.high, ringdown_mf=rd)
    hp, hc = case_glue(torch.as_tensor(frequencies, dtype=torch.float64), m1, m2, s1, s2,
                  distance_mpc=500., inclination=.73, phic=.17, fref_hz=30., tc=.002,
                  f_lower_hz=float(frequencies[0]), f_final_hz=float(frequencies[-1]))
    finite = bool(torch.isfinite(hp.real).all() and torch.isfinite(hp.imag).all()
                  and torch.isfinite(hc.real).all() and torch.isfinite(hc.imag).all())
    return {"case": list(case), "ringdown_mf_public": rd,
            "ringdown_mf_independent": independent_rd,
            "final_mass_fraction": final_fraction, "final_spin": final_spin,
            "count": int(len(frequencies)),
            "first_hz": float(frequencies[0]), "last_hz": float(frequencies[-1]),
            "finite": finite, "hp_shape": list(hp.shape), "hc_shape": list(hc.shape),
            "hp_norm": float(torch.linalg.vector_norm(hp)),
            "hc_norm": float(torch.linalg.vector_norm(hc))}

def main():
    low, high = load_official_rom(ROM, "sub1"), load_official_rom(ROM, "sub2")
    glue = OfficialROMGlue(low, high, ringdown_mf=.012)
    # The catalog grid is deliberately exact quarter-Hz arithmetic.
    catalog = torch.arange(30., 1024.0001, .25, dtype=torch.float64)
    cases = [(5., 1.4, .2, .0), (35., 35., -.4, .3), (80., 20., .6, -.2),
             (23., 21., -.6, .2)]
    independent = torch.tensor([23.17, 37.61, 82.43, 173.29], dtype=torch.float64)
    result = {"source": {"rom": str(ROM), "rom_sha256": sha256(ROM),
                          "python": sys.version, "torch": torch.__version__,
                          "lal": getattr(lal, "__version__", "unknown"),
                          "lalsimulation": getattr(ls, "__version__", "unknown"),
                          "wrapper": sha256(Path(__file__)),
                          "node_policy": "31_first_above"},
              "catalog_grid": {"df_hz": .25, "f_lower_hz": 30., "f_final_hz": 1024.,
                               "records": [run_case(glue, cases[0], catalog)]},
              "independent_grid": {"frequencies_hz": independent.tolist(),
                                    "records": [run_case(glue, c, independent) for c in cases]}}
    out = Path("scripts/diagnose_rom_glue_wrapper.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
