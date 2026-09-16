"""Parity tests: torchwave vs jaxwave for time-domain waveform generation (get_td_waveform)."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave import get_td_waveform as torch_get_td
from jaxwave import get_td_waveform as jax_get_td


@pytest.mark.parametrize(
    "approximant, params",
    [
        (
            "IMRPhenomD",
            {"mass1": 30.0, "mass2": 25.0, "chi1": 0.2, "chi2": -0.1, "distance": 100.0, "f_lower": 25.0},
        ),
        (
            "TaylorF2",
            {"mass1": 1.4, "mass2": 1.3, "chi1": 0.0, "chi2": 0.0, "distance": 50.0, "f_lower": 40.0},
        ),
        (
            "IMRPhenomXAS",
            {"mass1": 35.0, "mass2": 20.0, "chi1": 0.4, "chi2": -0.3, "distance": 200.0, "f_lower": 25.0},
        ),
    ],
)
def test_td_waveform_parity(approximant, params):
    """Verify exact parity between torchwave and jaxwave time-domain polarizations."""
    delta_t = 1.0 / 2048.0
    hp_t, hc_t = torch_get_td(approximant=approximant, delta_t=delta_t, **params)
    hp_j, hc_j = jax_get_td(approximant=approximant, delta_t=delta_t, **params)

    assert hp_t.shape == hp_j.shape
    assert hc_t.shape == hc_j.shape

    hp_t_np = hp_t.detach().numpy()
    hc_t_np = hc_t.detach().numpy()
    hp_j_np = np.asarray(hp_j)
    hc_j_np = np.asarray(hc_j)

    max_amp = max(np.max(np.abs(hp_t_np)), np.max(np.abs(hc_t_np)))
    rel_diff_p = np.max(np.abs(hp_j_np - hp_t_np)) / max_amp
    rel_diff_c = np.max(np.abs(hc_j_np - hc_t_np)) / max_amp

    assert rel_diff_p < 1e-10, f"h_plus relative difference {rel_diff_p} exceeds 1e-10"
    assert rel_diff_c < 1e-10, f"h_cross relative difference {rel_diff_c} exceeds 1e-10"


def test_td_waveform_return_times_parity():
    """Verify return_times flag parity."""
    hp_t, hc_t, times_t = torch_get_td(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=20.0,
        delta_t=1.0 / 2048.0,
        f_lower=30.0,
        return_times=True,
    )
    hp_j, hc_j, times_j = jax_get_td(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=20.0,
        delta_t=1.0 / 2048.0,
        f_lower=30.0,
        return_times=True,
    )

    np.testing.assert_allclose(np.asarray(times_j), times_t.numpy(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.asarray(hp_j), hp_t.numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.numpy(), rtol=1e-10, atol=1e-30)


def test_td_waveform_diffgw_dispatch():
    """Verify top-level diffgw unified dispatch for time-domain waveforms."""
    import diffgw

    hp_t, hc_t = diffgw.get_td_waveform(
        approximant="IMRPhenomD", mass1=30.0, mass2=20.0, delta_t=1.0 / 2048.0, f_lower=30.0, backend="torch"
    )
    hp_j, hc_j = diffgw.get_td_waveform(
        approximant="IMRPhenomD", mass1=30.0, mass2=20.0, delta_t=1.0 / 2048.0, f_lower=30.0, backend="jax"
    )

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.numpy(), rtol=1e-10, atol=1e-30)


def test_td_waveform_batched():
    """Verify batched time-domain waveform generation across mass vectors."""
    m1_t = torch.tensor([30.0, 40.0], dtype=torch.float64)
    m2_t = torch.tensor([20.0, 25.0], dtype=torch.float64)
    m1_j = jnp.array([30.0, 40.0], dtype=jnp.float64)
    m2_j = jnp.array([20.0, 25.0], dtype=jnp.float64)

    hp_t, hc_t = torch_get_td(
        approximant="IMRPhenomD", mass1=m1_t, mass2=m2_t, delta_t=1.0 / 2048.0, f_lower=30.0, duration=2.0
    )
    hp_j, hc_j = jax_get_td(
        approximant="IMRPhenomD", mass1=m1_j, mass2=m2_j, delta_t=1.0 / 2048.0, f_lower=30.0, duration=2.0
    )

    assert hp_t.shape == (2, int(2.0 * 2048))
    assert hp_j.shape == (2, int(2.0 * 2048))
    np.testing.assert_allclose(np.asarray(hp_j), hp_t.numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.numpy(), rtol=1e-10, atol=1e-30)

