"""Audit the direct KerrRingdown physical-energy 220 spectral variant.

The QNM, angular factors, energy amplitude, and TD/FD composition are computed
from independent scalar references. No native QNM or angular factor is supplied
to the scalar waveform.
"""

import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from diffgw_reference.kerr_ringdown import (
    direct_class_220_frequency, direct_class_220_inputs, direct_class_220_time,
)


CASES = (
    dict(spin=0.35, mass=53.0, distance=230.0, energy=0.017,
         inclination=1.1, phase=0.41, onset=0.002),
    dict(spin=0.7, mass=92.0, distance=430.0, energy=0.005,
         inclination=0.55, phase=-0.27, onset=-0.001),
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native(backend, case, times, frequencies):
    kwargs = dict(final_mass=case["mass"], final_spin=case["spin"],
                  distance=case["distance"], energy_fraction=case["energy"],
                  inclination=case["inclination"], phi=case["phase"],
                  t0=case["onset"], angular_basis="spheroidal",
                  qnm_backend="independent", energy_mode=(2, 2, 0))
    if backend == "torch":
        import torch
        from torchwave.approximants.kerr_ringdown import KerrRingdown
        arr = lambda value: torch.as_tensor(value, dtype=torch.float64)
    else:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.kerr_ringdown import KerrRingdown
        arr = lambda value: jnp.asarray(value, dtype=jnp.float64)
    td = tuple(np.asarray(x) for x in KerrRingdown.time_domain(arr(times), **kwargs))
    fd = tuple(np.asarray(x) for x in KerrRingdown.frequency_domain(arr(frequencies), **kwargs))
    return td, fd


def _relative_l2(native, scalar):
    actual = np.stack(native)
    expected = np.stack(scalar)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main():
    source_hashes = {
        name: _hash(name) for name in (
            "src/diffgw_reference/kerr_angular.py",
            "src/diffgw_reference/kerr_qnm.py",
            "src/diffgw_reference/kerr_ringdown.py",
            "src/torchwave/approximants/kerr_ringdown.py",
            "src/jaxwave/approximants/kerr_ringdown.py",
            "src/torchwave/approximants/kerr_angular.py",
        )
    }
    rows = []
    with mp.workdps(40):
        for case in CASES:
            inputs = direct_class_220_inputs(
                case["mass"], case["spin"], case["distance"], case["energy"],
                case["inclination"], case["phase"],
            )
            fine = direct_class_220_inputs(
                case["mass"], case["spin"], case["distance"], case["energy"],
                case["inclination"], case["phase"], ell_max=12,
            )
            convergence = max(abs(a - b) for a, b in zip(inputs[3:], fine[3:]))
            times = np.array([case["onset"] - 0.001, case["onset"],
                              case["onset"] + 0.002, case["onset"] + 0.008])
            frequencies = np.array([-130.0, 0.0, 35.0, 110.0, 270.0])
            expected_td = tuple(np.asarray([
                float(direct_class_220_time(t, case["onset"], *inputs)[i]) for t in times
            ]) for i in (0, 1))
            expected_fd = tuple(np.asarray([
                complex(direct_class_220_frequency(f, case["onset"], *inputs)[i]) for f in frequencies
            ]) for i in (0, 1))
            for backend in ("torch", "jax"):
                native_td, native_fd = _native(backend, case, times, frequencies)
                rows.append(dict(
                    backend=backend, spin=case["spin"],
                    angular_ell10_vs_ell12_absolute_error=float(convergence),
                    td_polarization_relative_l2=_relative_l2(native_td, expected_td),
                    fd_polarization_relative_l2=_relative_l2(native_fd, expected_fd),
                ))
    print(json.dumps({"method": "independent scalar QNM depth=400, angular Galerkin ell_max=10/order=32, CPU FP64",
                      "source_sha256": source_hashes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
