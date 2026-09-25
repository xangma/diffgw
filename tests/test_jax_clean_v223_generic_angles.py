"""Two-spin intrinsic v223 JAX angles against the audited Torch math kernel.

This is a nonpublic source-frame diagnostic.  It deliberately makes no
observer-frame or public-waveform equivalence claim.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.precession_msa_v223_generic import generic_msa_v223_angles
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles_v223
from torchwave.approximants.precession_msa_waveform_clean import _native_msa_final_spin


CASES = [
    (40., 20., (.3, .2, .4), (-.2, .1, -.2), 40., (25., 40., 80.)),
    (48., 17., (-.18, .27, .52), (.15, -.21, .09), 30., (20., 30., 70.)),
]


def _torch_result(m1, m2, chi1, chi2, f_ref, frequencies):
    f = torch.tensor(frequencies, dtype=torch.float64)
    s1 = torch.as_tensor(chi1, dtype=torch.float64)
    s2 = torch.as_tensor(chi2, dtype=torch.float64)
    result = assemble_precession_msa_angles_v223(
        f, m1, m2, s1, s2, f_ref=f_ref,
    )
    remnant, _ = _native_msa_final_spin(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64), s1, s2,
        torch.tensor(f_ref, dtype=torch.float64), result,
        compatibility="v223",
    )
    return result, remnant


@pytest.mark.parametrize("case", CASES)
def test_generic_v223_intrinsic_values_match_clean_torch(case):
    m1, m2, s1, s2, f_ref, frequencies = case
    expected, expected_remnant = _torch_result(*case)
    actual = generic_msa_v223_angles(frequencies, m1, m2, s1, s2, f_ref)
    for key in (
        "roots_reference", "roots", "c1", "sav", "psi_initial",
        "psi1", "psi2", "psi", "spin_squared", "secular_phi",
        "secular_zeta", "oscillatory_phi", "oscillatory_zeta",
        "alpha", "gamma", "cos_beta", "mean_s1_l", "mean_s2_l",
    ):
        np.testing.assert_allclose(
            np.asarray(actual[key]), expected[key].detach().numpy(),
            rtol=0., atol=3e-12, err_msg=key,
        )
    np.testing.assert_allclose(
        np.asarray(actual["final_spin"]), expected_remnant.detach().numpy(),
        rtol=0., atol=2e-13,
    )
    assert np.asarray(actual["alpha"]).shape == (3,)


def test_generic_v223_jit_batch_and_mass_exchange():
    frequencies = jnp.array([25., 40., 80.])
    s1 = jnp.array([[.3, .2, .4], [-.18, .27, .52]])
    s2 = jnp.array([[-.2, .1, -.2], [.15, -.21, .09]])
    evaluate = jax.jit(generic_msa_v223_angles)
    batched = evaluate(frequencies, jnp.array([40., 48.]), jnp.array([20., 17.]),
                       s1, s2, jnp.array([40., 30.]))
    assert batched["alpha"].shape == (2, 3)
    for row, case in enumerate(CASES):
        m1, m2, chi1, chi2, f_ref, _ = case
        single = evaluate(frequencies, m1, m2, jnp.array(chi1), jnp.array(chi2), f_ref)
        exchanged = evaluate(frequencies, m2, m1, jnp.array(chi2), jnp.array(chi1), f_ref)
        for key in ("alpha", "gamma", "cos_beta", "final_spin"):
            np.testing.assert_allclose(np.asarray(batched[key][row]),
                                       np.asarray(single[key]), rtol=0., atol=4e-12)
            np.testing.assert_allclose(np.asarray(exchanged[key]),
                                       np.asarray(single[key]), rtol=0., atol=4e-12)


@pytest.mark.parametrize("case", CASES)
def test_generic_v223_intrinsic_gradients_match_clean_torch(case):
    m1, m2, chi1, chi2, f_ref, frequencies = case
    weights = jnp.array([.2, -.1, .3])

    def jax_loss(spins):
        result = generic_msa_v223_angles(
            frequencies, m1, m2, spins[:3], spins[3:], f_ref,
        )
        return (jnp.sum(weights*(result["alpha"] + .7*result["gamma"]
                                 + .2*result["cos_beta"]))
                + .3*result["final_spin"])

    spins = torch.tensor((*chi1, *chi2), dtype=torch.float64, requires_grad=True)
    angles = assemble_precession_msa_angles_v223(
        torch.tensor(frequencies, dtype=torch.float64), m1, m2,
        spins[:3], spins[3:], f_ref=f_ref,
    )
    remnant, _ = _native_msa_final_spin(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64), spins[:3], spins[3:],
        torch.tensor(f_ref, dtype=torch.float64), angles,
        compatibility="v223",
    )
    torch_loss = (torch.tensor(np.asarray(weights), dtype=torch.float64)
                  *(angles["alpha"] + .7*angles["gamma"]
                    + .2*angles["cos_beta"])).sum() + .3*remnant
    expected = torch.autograd.grad(torch_loss, spins)[0].detach().numpy()
    actual = np.asarray(jax.grad(jax_loss)(jnp.array((*chi1, *chi2))))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
