#!/usr/bin/env python3
"""Public-binary ringdown diagnostics, separate from strict catalog acceptance.

The native API constructs an equal equatorial mirror pair. Its input amplitude
is set from the primary derivative-energy integral. The LAL API's phase/angular
conventions are not assumed equivalent: raw results and a separately labelled
fixed pi phase probe are both retained, without optimizing any alignment.
"""

import argparse
import json
import math
import platform
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.approximants.kerr_ringdown import (
    KerrRingdown, evaluate_qnm, sampled_kerr_single_mode_reference,
)
from torchwave.approximants.kerr_ringdown_energy import ringdown_amplitude_from_energy_fraction


def metrics(native, reference):
    native = native.detach().cpu().numpy()
    scale = np.linalg.norm(reference)
    return {
        "relative_l2": float(np.linalg.norm(native - reference) / scale),
        "relative_norm_error": float(abs(np.linalg.norm(native) / scale - 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "scope": "Unaligned public-binary TD diagnostics, not strict waveform certification",
        "host": platform.node(),
        "lal_version": lal.__version__,
        "lalsimulation_version": lalsim.__version__,
        "oracle_api": "SimBlackHoleRingdown",
        "oracle_inputs": "mass kg; distance m; deltaT seconds; final spin; fractional mass loss",
        "native_energy_convention": "Equal +m,-m pair; remnant rest-energy fraction; exact finite Q",
        "native_qnm_backend": "independent",
        "native_angular_basis": "spheroidal",
        "strict_reference_contract": {
            "mode": [2, 2, 0],
            "amplitude": "explicit native_pair amplitude from the published finite-Q energy integral",
            "energy": "epsilon is remnant rest-energy fraction; external fractional-mass-loss semantics remain unresolved",
            "angular_basis": "normalized spin-weighted spheroidal basis with equatorial mirror",
            "onset": "t < t0 is exactly zero; u=t-t0 for t >= t0",
            "phase": "phase is explicit positive-frequency carrier phase at onset",
            "sampling": "uniform samples returned by SimBlackHoleRingdown; native evaluated at the same t_k",
            "fourier_sign": "integral h(t) exp(-2*pi*i*f*t) dt",
        },
        "phase_probes": {
            "raw": "Both APIs supplied phase zero",
            "fixed_pi": "Native phase pi, reference phase zero; a fixed convention diagnostic, no optimized alignment",
        },
        "unresolved": [
            "The meaning of the reference phase zero and angular eigenfunction gauge is not established as equivalent to the native API.",
            "The scalar native energy conversion treats fractional mass loss as a fraction of the remnant rest energy; reference implementation semantics require independent documentation.",
            "No waveform coefficient or native default is selected from these diagnostics.",
        ],
        "cases": [],
    }
    final_mass, distance, energy_fraction = 65.0, 100.0, 0.01
    for spin in (0.0, 0.3, 0.7, 0.95):
        omega, tau, _ = evaluate_qnm(2, 2, 0, final_mass, spin)
        amplitude = ringdown_amplitude_from_energy_fraction(
            final_mass, distance, energy_fraction, omega, tau,
            convention="native_pair",
        )
        for inclination in (0.0, 0.8, 1.3):
            hp, hc = lalsim.SimBlackHoleRingdown(
                lal.LIGOTimeGPS(0), 0.0, 1.0 / 8192.0,
                final_mass * lal.MSUN_SI, spin, energy_fraction,
                distance * 1e6 * lal.PC_SI, inclination, 2, 2,
            )
            times = torch.arange(len(hp.data.data), dtype=torch.float64) * hp.deltaT
            row = {
                "final_mass_solar": final_mass, "final_spin": spin,
                "distance_mpc": distance, "energy_fraction": energy_fraction,
                "inclination": inclination, "mode": [2, 2, 0],
                "delta_t": hp.deltaT, "num_samples": len(times),
                "native_pair_amplitude": float(amplitude),
            }
            for label, phase in (("raw", 0.0), ("fixed_pi", math.pi)):
                native = sampled_kerr_single_mode_reference(
                    times, final_mass, spin, amplitude, mode=(2, 2, 0),
                    phase=phase, inclination=inclination,
                    angular_basis="spheroidal",
                )
                row[label] = {
                    "plus": metrics(native[0], hp.data.data),
                    "cross": metrics(native[1], hc.data.data),
                }
            result["cases"].append(row)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: max(row[label][p]["relative_l2"]
                                for row in result["cases"] for p in ("plus", "cross"))
                      for label in ("raw", "fixed_pi")}))


if __name__ == "__main__":
    main()
