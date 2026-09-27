"""Optional public-binary diagnostic for the direct-class Kerr 220 slice.

This invokes LAL only as a public waveform data provider through the existing
test oracle. The residual is reported, not used to calibrate the scalar model.
"""

import json

import numpy as np
import torch

from kerr_public_oracle import public_ringdown_fd_converged
from torchwave.approximants.kerr_ringdown import KerrRingdown


def main():
    import lalsimulation

    case = dict(final_mass=53.0, final_spin=0.35, distance=230.0,
                energy_fraction=0.017, inclination=1.1, phi=0.41, t0=0.0)
    frequencies, reference_plus, reference_cross, convergence = (
        public_ringdown_fd_converged(
            case["final_mass"], case["final_spin"], case["distance"],
            inclination=case["inclination"], phase=case["phi"],
            fractional_mass_loss=case["energy_fraction"],
            sample_rate=32768, max_sample_rate=131072, delta_f=1.0,
            comparison_band=(50.0, 500.0), f_max=500.0,
        )
    )
    native_plus, native_cross = KerrRingdown.frequency_domain(
        torch.from_numpy(frequencies), **case, energy_mode=(2, 2, 0),
        angular_basis="spheroidal", qnm_backend="independent",
    )
    band = (frequencies >= 50.0) & (frequencies <= 500.0)

    def relative_l2(actual, expected):
        return float(np.linalg.norm(actual[band] - expected[band])
                     / np.linalg.norm(expected[band]))

    print(json.dumps({
        "purpose": "diagnostic only; no public-binary or physical-accuracy claim",
        "lal_version": lalsimulation.__version__,
        "case": case,
        "band_hz": [50.0, 500.0],
        "frequency_spacing_hz": 1.0,
        "oracle_convergence": convergence,
        "relative_l2": {
            "plus": relative_l2(native_plus.detach().numpy(), reference_plus),
            "cross": relative_l2(native_cross.detach().numpy(), reference_cross),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
