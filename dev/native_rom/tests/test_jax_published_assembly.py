"""End-to-end backend parity for the unregistered fixed-step development route."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from dev.native_rom.eob.jax.published_waveform import (
    reconstruct_seobnrv4_22 as reconstruct_jax,
)
from dev.native_rom.eob.torch.published_waveform import (
    reconstruct_seobnrv4_22 as reconstruct_torch,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("m1,m2,chi1,chi2,num_steps", [
    (1., 1., 0., 0., 500),
    (4., 1., .4, -.2, 700),
])
def test_fixed_step_source_components_match_across_backends(
    m1, m2, chi1, chi2, num_steps,
):
    options = dict(
        spin_truncation="tabulated",
        spin_v6="bohe_total_energy",
        include_pan_v8=False,
        include_positive_chi_540=False,
        final_spin_fit_order=(3, 4),
        dt_geom=.5,
        num_steps=num_steps,
        ringdown_duration=20.,
        numerical_compatibility=False,
    )
    jax_result = reconstruct_jax(
        jnp.asarray(.06), m1, m2, chi1, chi2, **options,
    )
    torch_result = reconstruct_torch(
        torch.tensor(.06, dtype=torch.float64), m1, m2, chi1, chi2, **options,
    )

    def compare(jax_value, torch_value, *, rtol=2e-10, atol=2e-10):
        np.testing.assert_allclose(
            np.asarray(jax_value), torch_value.detach().numpy(), rtol=rtol, atol=atol,
        )

    compare(jax_result.trajectory.peak.time, torch_result.trajectory.peak.time)
    compare(jax_result.attachment.match_time, torch_result.attachment.match_time)
    compare(jax_result.trajectory.state, torch_result.trajectory.state)
    compare(jax_result.mode, torch_result.mode)
    assert bool(jnp.all(jnp.isfinite(jax_result.mode)))
