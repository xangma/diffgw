"""Ablate the inferred public angular cutoff against two equation solvers.

No binary waveform or private implementation is read. The scalar Galerkin
reference follows Cook--Zalutskiy Eq. (4); the full Frobenius series follows
the angular recurrence. The public-compatible cutoff is evaluated only to
identify the numerical source of its discrepancy from those solutions.
"""

import json
import math

import mpmath as mp
import numpy as np
import torch

from diffgw_reference.kerr_angular import spheroidal_coefficients, spherical_harmonic
from diffgw_reference.kerr_qnm import ScalarQNM220
from torchwave.approximants.kerr_angular import leaver_frobenius_coefficients
from torchwave.approximants.kerr_angular_table_convention import (
    _norm as inferred_cutoff_norm,
    _raw as inferred_cutoff_raw,
    spin_weighted_spheroidal_harmonic_table_compat,
)
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


ANGLES = (1.1, math.pi - 1.1, math.pi / 2)
LABELS = ("inclination", "mirror", "equator")


def _equatorial_gauge(values):
    equator = values[-1]
    return values * equator.conj() / abs(equator)


def _relative_errors(actual, expected):
    return {label: float(abs((a - b) / b))
            for label, a, b in zip(LABELS, actual, expected)}


def _scalar_spectral(spin, ell_max):
    with mp.workdps(40):
        qnm, _ = ScalarQNM220(ell_max=ell_max, order=32, dps=40).root(
            mp.mpf(str(spin)), depth=400,
        )
        _, coefficients = spheroidal_coefficients(
            2, 2, mp.mpf(str(spin)) * qnm, ell_max=ell_max, order=32,
        )
        values = np.array([
            complex(sum(coefficient * spherical_harmonic(ell, 2, mp.mpf(str(theta)))
                        for ell, coefficient in enumerate(coefficients, start=2)))
            for theta in ANGLES
        ])
        return complex(qnm), _equatorial_gauge(values)


def _case(spin):
    scalar_qnm, spectral = _scalar_spectral(spin, 12)
    _, spectral_coarse = _scalar_spectral(spin, 10)
    s = torch.tensor(spin, dtype=torch.float64)
    real, damping = signed_qnm_220(s)
    qnm = complex(torch.complex(real, -damping))
    c = s * torch.complex(real, -damping)
    eigenvalue, coefficients = leaver_frobenius_coefficients(2, c, n_terms=48)
    mu = torch.tensor([math.cos(theta) for theta in ANGLES], dtype=torch.float64)
    z = (1 + mu) / 2
    terms = coefficients * z[:, None] ** torch.arange(48)
    raw_full = torch.exp(c * mu) * (1 + mu).square() * terms.sum(-1)
    raw_cut = inferred_cutoff_raw(mu, c, coefficients)

    nodes, weights = np.polynomial.legendre.leggauss(256)
    x = torch.from_numpy(nodes)
    w = torch.from_numpy(weights)
    zn = (1 + x) / 2
    full_at_nodes = (torch.exp(c * x) * (1 + x).square()
                     * (coefficients * zn[:, None] ** torch.arange(48)).sum(-1))
    norm_full = float(torch.sqrt(2 * math.pi * (w * full_at_nodes.abs().square()).sum()))
    norm_cut = float(inferred_cutoff_norm(c, coefficients))
    stages = {
        "full_series_full_norm": _equatorial_gauge((raw_full / norm_full).numpy()),
        "cut_series_full_norm": _equatorial_gauge((raw_cut / norm_full).numpy()),
        "full_series_cut_norm": _equatorial_gauge((raw_full / norm_cut).numpy()),
        "cut_series_cut_norm": _equatorial_gauge((raw_cut / norm_cut).numpy()),
    }
    public_compat = spin_weighted_spheroidal_harmonic_table_compat(
        s, torch.tensor(ANGLES, dtype=torch.float64),
    ).numpy()
    first_small_term = {}
    for label, row in zip(LABELS, terms):
        candidates = torch.nonzero(row[1:].abs() < 0.01).flatten()
        first_small_term[label] = int(candidates[0]) + 1 if len(candidates) else None
    return {
        "spin": spin,
        "complex_oblateness": [complex(c).real, complex(c).imag],
        "angular_eigenvalue": [complex(eigenvalue).real, complex(eigenvalue).imag],
        "generated_vs_scalar_qnm_absolute": abs(qnm - scalar_qnm),
        "spectral_ell10_vs_ell12_relative": _relative_errors(spectral_coarse, spectral),
        "first_included_term_below_absolute_0p01": first_small_term,
        "full_norm": norm_full,
        "cut_norm": norm_cut,
        "cut_norm_over_full_norm_minus_one": norm_cut / norm_full - 1,
        "relative_to_scalar_spectral": {
            label: _relative_errors(values, spectral) for label, values in stages.items()
        },
        "cut_series_cut_norm_vs_public_compat_relative": _relative_errors(
            stages["cut_series_cut_norm"], public_compat,
        ),
    }


def main():
    print(json.dumps({
        "purpose": "diagnostic ablation; does not qualify a waveform",
        "source": "Cook and Zalutskiy, arXiv:1410.7698v2, Eqs. (4), (15b), (49), (51)",
        "phase": "each angular function made positive real at theta=pi/2",
        "angles_radians": dict(zip(LABELS, ANGLES)),
        "cases": [_case(spin) for spin in (0.35, 0.7)],
    }, indent=2))


if __name__ == "__main__":
    main()
