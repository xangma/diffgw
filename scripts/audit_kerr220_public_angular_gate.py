"""Reproduce the source-backed Kerr 220 opt-in public-clean qualification blocker.

This is a diagnostic, not a tolerance-adjusting parity test. It uses the
independent Cook--Zalutskiy scalar QNM and Galerkin angular references, then
compares both opt-in public-clean frontends after aligning their equator phase gauge.
"""

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import kerr_ringdown as scalar_waveform
from diffgw_reference.kerr_angular import spherical_harmonic, spheroidal_coefficients
from diffgw_reference.kerr_qnm import ScalarQNM220


CASES = (
    dict(spin=0.35, mass=53.0, distance=230.0, energy=0.017,
         inclination=1.1, phase=0.41, onset=0.002),
    dict(spin=0.7, mass=92.0, distance=430.0, energy=0.005,
         inclination=0.55, phase=-0.27, onset=-0.001),
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _angular_values(spin, frequency, inclination, phase, *, ell_max=10):
    _, coefficients = spheroidal_coefficients(
        2, 2, mp.mpf(str(spin)) * frequency, ell_max=ell_max, order=32
    )

    def value(theta, phi):
        return sum(coef * spherical_harmonic(ell, 2, theta, phi)
                   for ell, coef in zip(range(2, ell_max + 1), coefficients))

    equator = value(mp.pi / 2, 0)
    gauge = mp.conj(equator) / abs(equator)
    return (complex(value(inclination, phase) * gauge),
            complex(value(mp.pi - inclination, phase) * gauge))


def _native(backend, case):
    spin, angle, phase = case["spin"], case["inclination"], case["phase"]
    times = np.array([case["onset"] - 0.001, case["onset"],
                      case["onset"] + 0.002, case["onset"] + 0.008])
    frequencies = np.array([-130.0, 0.0, 35.0, 110.0, 270.0])
    if backend == "torch":
        import torch
        from torchwave.approximants.kerr_angular_table_convention import (
            spin_weighted_spheroidal_harmonic_table_compat as angular,
        )
        from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
        from torchwave.approximants.kerr_ringdown_public_clean import kerr220_td, kerr220_fd
        arr = lambda x: torch.as_tensor(x, dtype=torch.float64)
        w, decay = signed_qnm_220(arr(spin))
    else:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
        from jaxwave.approximants.kerr_ringdown_public_clean import (
            spin_weighted_spheroidal_harmonic_table_compat as angular,
            kerr220_td, kerr220_fd,
        )
        arr = lambda x: jnp.asarray(x, dtype=jnp.float64)
        w, decay = evaluate_kerr_qnm_220(arr(spin))
    args = dict(final_mass=case["mass"], final_spin=spin,
                distance=case["distance"], energy_fraction=case["energy"],
                inclination=angle, phic=phase, tc=case["onset"])
    plus = complex(angular(arr(spin), arr(angle), arr(phase)))
    mirror = complex(angular(arr(spin), arr(math.pi - angle), arr(phase)))
    td = tuple(np.asarray(x) for x in kerr220_td(arr(times), **args))
    fd = tuple(np.asarray(x) for x in kerr220_fd(arr(frequencies), **args))
    return complex(float(w), -float(decay)), (plus, mirror), times, frequencies, td, fd


def _relative_l2(native, scalar):
    actual = np.stack(native)
    expected = np.stack(scalar)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main():
    source_hashes = {
        name: _hash(name) for name in (
            "src/diffgw_reference/kerr_angular.py",
            "src/diffgw_reference/kerr_qnm.py",
            "src/torchwave/approximants/kerr_angular_table_convention.py",
            "src/jaxwave/approximants/kerr_ringdown_public_clean.py",
        )
    }
    rows = []
    with mp.workdps(40):
        scalar_qnm = ScalarQNM220(ell_max=10, order=32, dps=40)
        for case in CASES:
            frequency, _ = scalar_qnm.root(case["spin"], depth=400)
            angular = _angular_values(case["spin"], frequency,
                                      case["inclination"], case["phase"])
            angular_fine = _angular_values(case["spin"], frequency,
                                           case["inclination"], case["phase"],
                                           ell_max=12)
            angular_convergence = max(abs(a - b) for a, b in zip(angular, angular_fine))
            omega = float(mp.re(frequency)) / (case["mass"] * MSUN_SEC)
            damping = -float(mp.im(frequency)) / (case["mass"] * MSUN_SEC)
            amplitude = scalar_waveform.energy_amplitude(
                case["mass"] * MSUN_SEC, case["distance"] * MPC_SI / C_SI,
                case["energy"], omega, damping,
            )
            for backend in ("torch", "jax"):
                native_frequency, native_angular, times, frequencies, td, fd = _native(backend, case)
                expected_td = tuple(np.array([
                    float(scalar_waveform.time_polarizations(
                        t, case["onset"], amplitude, omega, damping, *angular
                    )[index]) for t in times
                ]) for index in (0, 1))
                expected_fd = tuple(np.array([
                    complex(scalar_waveform.frequency_polarizations(
                        f, case["onset"], amplitude, omega, damping, *angular
                    )[index]) for f in frequencies
                ]) for index in (0, 1))
                rows.append({
                    "backend": backend, "spin": case["spin"],
                    "qnm_absolute_error": abs(native_frequency - complex(frequency)),
                    "angular_ell10_vs_ell12_absolute_error": angular_convergence,
                    "angular_plus_absolute_error": abs(native_angular[0] - angular[0]),
                    "angular_mirror_absolute_error": abs(native_angular[1] - angular[1]),
                    "td_polarization_relative_l2": _relative_l2(td, expected_td),
                    "fd_polarization_relative_l2": _relative_l2(fd, expected_fd),
                })
    print(json.dumps({"method": "scalar QNM depth=400, Galerkin ell_max=10/order=32, equator-positive phase gauge, CPU FP64",
                      "source_sha256": source_hashes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
