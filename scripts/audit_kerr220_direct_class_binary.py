"""Staged public-binary diagnostic for the direct-class Kerr 220 slice.

LAL is used only through the public waveform oracle. The paper-consistent
carrier stage uses the independent scalar QNM/angular/energy references; its
residual is evidence about conventions, not a direct-class qualification.
"""

import argparse
import json
import math

import mpmath as mp
import numpy as np
import torch

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference.kerr_angular import spheroidal_coefficients, spherical_harmonic
from diffgw_reference.kerr_qnm import ScalarQNM220
from diffgw_reference.kerr_ringdown import energy_amplitude, frequency_polarizations
from kerr_public_oracle import public_ringdown_fd_converged
from torchwave.approximants.kerr_angular_table_convention import (
    spin_weighted_spheroidal_harmonic_table_compat,
)
from torchwave.approximants.kerr_ringdown import KerrRingdown
from torchwave.approximants.kerr_ringdown_public_clean import kerr220_fd


def _scalar_paper_carrier(frequencies, case):
    """Evaluate Cook's e^{-i omega t} mode with its positive-m mirror.

    The equatorial gauge is a separately declared phase choice: it makes
    S(pi/2, 0) real and positive without changing the angular eigenfunction.
    """
    with mp.workdps(40):
        spin = mp.mpf(str(case["final_spin"]))
        mass = mp.mpf(str(case["final_mass"])) * mp.mpf(str(MSUN_SEC))
        qnm, _ = ScalarQNM220(ell_max=10, order=32, dps=40).root(spin, depth=400)
        _, coefficients = spheroidal_coefficients(2, 2, spin * qnm, ell_max=10, order=32)

        def angular(theta):
            return sum(coef * spherical_harmonic(ell, 2, theta)
                       for ell, coef in enumerate(coefficients, start=2))

        equator = angular(mp.pi / 2)
        gauge = mp.conj(equator) / abs(equator)
        azimuth = mp.exp(2j * mp.mpf(str(case["phi"])))
        inclination = mp.mpf(str(case["inclination"]))
        positive = angular(inclination) * gauge * azimuth
        mirror = angular(mp.pi - inclination) * gauge * azimuth
        omega, decay = mp.re(qnm) / mass, -mp.im(qnm) / mass
        amplitude = energy_amplitude(
            mass, mp.mpf(str(case["distance"])) * mp.mpf(str(MPC_SI)) / mp.mpf(str(C_SI)),
            case["energy_fraction"], omega, decay,
        )
        waveform = np.array([
            tuple(complex(value) for value in frequency_polarizations(
                frequency, case["t0"], amplitude, omega, decay, positive, mirror,
            )) for frequency in frequencies
        ])
        angular_pair = (complex(angular(inclination) * gauge),
                        complex(angular(mp.pi - inclination) * gauge))
        return waveform[:, 0], waveform[:, 1], float(mp.arg(equator)), angular_pair


def main():
    import lalsimulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin", type=float, choices=(0.35, 0.7), default=0.35)
    args = parser.parse_args()
    case = dict(final_mass=53.0, final_spin=args.spin, distance=230.0,
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
    rephased_plus, rephased_cross = KerrRingdown.frequency_domain(
        torch.from_numpy(frequencies), **dict(case, phi=-case["phi"]),
        phases={(2, 2, 0): math.pi}, energy_mode=(2, 2, 0),
        angular_basis="spheroidal", qnm_backend="independent",
    )
    paper_plus, paper_cross, equatorial_phase, scalar_angular = (
        _scalar_paper_carrier(frequencies, case)
    )
    table_plus, table_cross = kerr220_fd(
        torch.from_numpy(frequencies), case["final_mass"], case["final_spin"],
        case["distance"], case["energy_fraction"], case["inclination"],
        case["phi"], case["t0"],
    )
    band = (frequencies >= 50.0) & (frequencies <= 500.0)

    def relative_l2(actual, expected):
        return float(np.linalg.norm(actual[band] - expected[band])
                     / np.linalg.norm(expected[band]))

    def residuals(plus, cross):
        return {"plus": relative_l2(np.asarray(plus), reference_plus),
                "cross": relative_l2(np.asarray(cross), reference_cross)}

    table_angular = tuple(complex(spin_weighted_spheroidal_harmonic_table_compat(
        torch.tensor(case["final_spin"], dtype=torch.float64),
        torch.tensor(theta, dtype=torch.float64),
    )) for theta in (case["inclination"], math.pi - case["inclination"]))

    print(json.dumps({
        "purpose": "diagnostic only; no public-binary or physical-accuracy claim",
        "lal_version": lalsimulation.__version__,
        "case": case,
        "band_hz": [50.0, 500.0],
        "frequency_spacing_hz": 1.0,
        "oracle_convergence": convergence,
        "relative_l2": {
            "direct_class_v1": residuals(native_plus.detach().numpy(), native_cross.detach().numpy()),
            "direct_class_empirical_phase_map": residuals(
                rephased_plus.detach().numpy(), rephased_cross.detach().numpy()),
            "independent_paper_carrier_equatorial_gauge": residuals(paper_plus, paper_cross),
            "separate_inferred_public_clean_route": residuals(
                table_plus.detach().numpy(), table_cross.detach().numpy()),
        },
        "empirical_phase_map": "class phi = -public phase; class intrinsic phase = pi",
        "equatorial_gauge_radians": equatorial_phase,
        "independent_spectral_vs_inferred_table_angular_relative": {
            name: abs(scalar - table) / abs(table)
            for name, scalar, table in zip(
                ("inclination", "mirror"), scalar_angular, table_angular)
        },
        "paper_carrier_source": "Cook and Zalutskiy, arXiv:1410.7698v2, Eq. (3)",
    }, indent=2))


if __name__ == "__main__":
    main()
