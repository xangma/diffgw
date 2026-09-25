"""Cross-backend checks for the nonpublic aligned native-MSA v223 endpoint.

The regularized root triple is a v223 compatibility prescription.  These
checks do not assert a strict beta=0 paper limit or public waveform parity.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.precession_msa_v223_endpoint import aligned_msa_v223_endpoint
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles_v223,
)
from torchwave.approximants.precession_msa_waveform_clean import _native_msa_final_spin


def _t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _torch_endpoint(f, m1, m2, chi1, chi2, f_ref):
    s1 = torch.stack((torch.zeros_like(chi1), torch.zeros_like(chi1), chi1), dim=-1)
    s2 = torch.stack((torch.zeros_like(chi2), torch.zeros_like(chi2), chi2), dim=-1)
    angles = assemble_precession_msa_angles_v223(
        f, m1, m2, s1, s2, f_ref=f_ref,
    )
    remnant, _ = _native_msa_final_spin(
        m1, m2, s1, s2, f_ref, angles, compatibility="v223",
    )
    return angles, remnant


@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (40.0, 20.0, 0.4, -0.2),
    (35.0, 25.0, 0.4, -0.2),
    (40.0, 20.0, 0.0, 0.0),
    (30.0, 30.0, 0.3, -0.1),
])
def test_aligned_endpoint_matches_torch_intermediates(m1, m2, chi1, chi2):
    frequencies = np.array([25.0, 40.0, 80.0])
    angles, remnant = _torch_endpoint(
        _t(frequencies), _t(m1), _t(m2), _t(chi1), _t(chi2), _t(40.0),
    )
    result = aligned_msa_v223_endpoint(
        jnp.asarray(frequencies), m1, m2, chi1, chi2, 40.0,
    )
    np.testing.assert_allclose(
        np.asarray(result["alpha"] + result["gamma"]),
        (angles["alpha"] + angles["gamma"]).detach().numpy(),
        rtol=0.0, atol=1e-9 if m1 == m2 else 2e-10,
    )
    for key in ("cos_beta", "sav", "c1"):
        np.testing.assert_allclose(
            np.asarray(result[key]), angles[key].detach().numpy(),
            rtol=0.0, atol=2e-14,
        )
    if m1 == m2:
        np.testing.assert_allclose(
            np.asarray(result["mean_s1_l"] + result["mean_s2_l"]),
            (angles["mean_s1_l"] + angles["mean_s2_l"]).detach().numpy(),
            rtol=0.0, atol=2e-14,
        )
    else:
        for key in ("mean_s1_l", "mean_s2_l"):
            np.testing.assert_allclose(
                np.asarray(result[key]), angles[key].detach().numpy(),
                rtol=0.0, atol=2e-14,
            )
    np.testing.assert_allclose(
        np.asarray(result["final_spin"]), remnant.detach().numpy(),
        rtol=0.0, atol=2e-13,
    )
    # The compatibility root split is an independent contract: it remains
    # present even though the physical aligned spin does not nutate.
    total = m1 + m2
    initial_spin = (m1 / total) ** 2 * chi1 + (m2 / total) ** 2 * chi2
    np.testing.assert_allclose(
        np.asarray(result["sav"]), initial_spin**2 + 5e-10,
        rtol=0.0, atol=2e-17,
    )
    assert np.all((1.0 - np.asarray(result["cos_beta"])) > 0.0)
    assert np.all((1.0 - np.asarray(result["cos_beta"])) < 1e-8)


def test_aligned_endpoint_jit_batch_and_mass_exchange():
    frequencies = jnp.asarray([25.0, 40.0, 80.0])
    evaluate = jax.jit(aligned_msa_v223_endpoint)
    batched = evaluate(
        frequencies, jnp.asarray([40.0, 35.0]), jnp.asarray([20.0, 25.0]),
        jnp.asarray([0.4, 0.0]), jnp.asarray([-0.2, 0.0]),
        jnp.asarray([40.0, 40.0]),
    )
    assert batched["alpha"].shape == (2, 3)
    for row, inputs in enumerate(((40.0, 20.0, 0.4, -0.2),
                                  (35.0, 25.0, 0.0, 0.0))):
        single = evaluate(frequencies, *inputs, 40.0)
        exchanged = evaluate(frequencies, inputs[1], inputs[0], inputs[3], inputs[2], 40.0)
        for key in ("alpha", "gamma", "cos_beta", "final_spin"):
            np.testing.assert_allclose(
                np.asarray(batched[key][row]), np.asarray(single[key]),
                rtol=0.0, atol=2e-13,
            )
            np.testing.assert_allclose(
                np.asarray(single[key]), np.asarray(exchanged[key]),
                rtol=0.0, atol=2e-13,
            )


def test_aligned_endpoint_gradients_match_torch():
    frequencies = np.array([25.0, 40.0, 80.0])
    weights = np.array([0.2, -0.1, 0.3])

    def jax_loss(parameters):
        chi1, chi2, f_ref = parameters
        result = aligned_msa_v223_endpoint(
            jnp.asarray(frequencies), 40.0, 20.0, chi1, chi2, f_ref,
        )
        return (jnp.sum(jnp.asarray(weights) * (result["alpha"] + result["gamma"]))
                + 0.3 * result["final_spin"])

    parameters = _t([0.4, -0.2, 40.0]).requires_grad_()
    angles, remnant = _torch_endpoint(
        _t(frequencies), _t(40.0), _t(20.0), parameters[0], parameters[1], parameters[2],
    )
    torch_loss = torch.sum(_t(weights) * (angles["alpha"] + angles["gamma"])) + 0.3 * remnant
    torch_gradient = torch.autograd.grad(torch_loss, parameters)[0]
    jax_gradient = jax.grad(jax_loss)(jnp.asarray([0.4, -0.2, 40.0]))
    assert bool(jnp.all(jnp.isfinite(jax_gradient)))
    np.testing.assert_allclose(
        np.asarray(jax_gradient), torch_gradient.detach().numpy(),
        rtol=2e-7, atol=2e-8,
    )
