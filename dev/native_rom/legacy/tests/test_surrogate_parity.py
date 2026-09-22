"""Archived legacy surrogate parity check; not a release qualification test."""

import numpy as np
import torch
import jax.numpy as jnp
from torchwave.approximants.rom.surrogate import SEOBNR_ROM as TorchSEOBNR_ROM
from jaxwave.approximants.rom.surrogate import SEOBNR_ROM as JaxSEOBNR_ROM


def test_seobnr_rom_surrogate_parity():
    # Test SEOBNR_ROM class
    t_rom = TorchSEOBNR_ROM()
    j_rom = JaxSEOBNR_ROM()

    f = np.linspace(20.0, 500.0, 120)
    test_cases = [
        (30.0, 20.0, 0.0, 0.0),
        (35.0, 25.0, 0.3, -0.2),
        (50.0, 10.0, -0.5, 0.4),
    ]

    for m1, m2, s1, s2 in test_cases:
        t_hp, t_hc = t_rom(
            torch.tensor(f, dtype=torch.float64),
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=100.0,
            inclination=0.5,
            phic=0.2,
        )
        j_hp, j_hc = j_rom(
            jnp.array(f, dtype=jnp.float64),
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=100.0,
            inclination=0.5,
            phic=0.2,
        )

        np.testing.assert_allclose(np.asarray(j_hp), t_hp.numpy(), rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(np.asarray(j_hc), t_hc.numpy(), rtol=1e-12, atol=1e-25)


