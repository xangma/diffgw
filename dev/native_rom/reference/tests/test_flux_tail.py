"""Published factorized tail modulus with supplied orbital quantities."""

import jax
import jax.numpy as jnp
import mpmath as mp
import pytest
import torch

from dev.native_rom.reference.flux import _tail_modulus as scalar_tail
from jaxwave.approximants.eob.seobnrv4_amplitudes33_44 import _tail_modulus as jax_tail
from torchwave.approximants.eob.factorized_modes import _tail_abs_squared

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("ell,m,omega,hreal", [
    (2, 2, "0.000001", "1.0"),
    (3, 2, "0.02", "0.97"),
    (4, 1, "0.08", "1.03"),
    (7, 5, "0.04", "0.89"),
    (8, 8, "0.12", "1.01"),
])
def test_published_tail_modulus_against_scalar_torch_jax(ell, m, omega, hreal):
    # DIN (19), (59) and Pan (26): the arbitrary tail-phase scale drops out.
    with mp.workdps(80):
        w, energy = mp.mpf(omega), mp.mpf(hreal)
        khat = m*w*energy
        gamma_modulus = abs(mp.gamma(ell+1-2j*khat)
                            *mp.exp(mp.pi*khat)/mp.factorial(ell))
        reference = scalar_tail(ell, m, w, energy)
        assert abs(reference/gamma_modulus-1) < mp.mpf("1e-70")

    w_float, energy_float = float(omega), float(hreal)
    torch_khat = torch.tensor(m*w_float*energy_float, dtype=torch.float64)
    torch_value = _tail_abs_squared(ell, torch_khat).sqrt().item()
    jax_value = float(jax_tail(ell, m, jnp.float64(w_float), jnp.float64(energy_float)))
    expected = float(gamma_modulus)
    assert torch_value == pytest.approx(expected, rel=3e-14)
    assert jax_value == pytest.approx(expected, rel=3e-14)


def test_jax_tail_modulus_jit_and_gradient():
    def tail(omega, energy):
        return jax_tail(4, 1, omega, energy)

    omega, energy = jnp.float64("0.08"), jnp.float64("1.03")
    assert float(jax.jit(tail)(omega, energy)) == pytest.approx(float(tail(omega, energy)), rel=2e-14)
    with mp.workdps(80):
        def gamma_tail(w, hreal):
            khat = w*hreal
            return abs(mp.gamma(5-2j*khat)*mp.exp(mp.pi*khat)/mp.factorial(4))

        omega_reference = mp.diff(lambda w: gamma_tail(w, mp.mpf("1.03")), mp.mpf("0.08"))
        energy_reference = mp.diff(lambda h: gamma_tail(mp.mpf("0.08"), h), mp.mpf("1.03"))
    assert float(jax.grad(tail, argnums=0)(omega, energy)) == pytest.approx(
        float(omega_reference), rel=2e-12)
    assert float(jax.grad(tail, argnums=1)(omega, energy)) == pytest.approx(
        float(energy_reference), rel=2e-12)
