"""Intrinsic JAX zero-nutation MSA limit against independent component checks."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from diffgw_reference.xp_msa_regularized_beta import one_spin_aligned_beta as scalar_beta
from jaxwave.approximants.precession_msa_v223_endpoint import _opening_l_3pn
from jaxwave.approximants.precession_msa_v223_zero_nutation import one_spin_msa_v223_angles
from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles_v223
from torchwave.approximants.precession_msa_waveform_clean import _native_msa_final_spin


CASES = [
    (40., 20., (.2, -.1, .4), (0., 0., -.2), 40.),
    (20., 40., (0., 0., -.2), (.2, -.1, .4), 40.),
    (30., 30., (.15, .07, .35), (0., 0., -.1), 30.),
    (40., 20., (0., 0., .3), (0., 0., -.1), 40.),
    (40., 20., (1e-8, -1e-8, .4), (0., 0., -.2), 40.),
]


def _torch_angles(case, frequencies):
    m1, m2, chi1, chi2, f_ref = case
    s1 = torch.tensor(chi1, dtype=torch.float64)
    s2 = torch.tensor(chi2, dtype=torch.float64)
    angles = assemble_precession_msa_angles_v223(
        torch.tensor(frequencies, dtype=torch.float64), m1, m2, s1, s2,
        f_ref=f_ref,
    )
    # The angle assembler canonicalizes mass order internally; the lower-level
    # remnant helper expects the already canonicalized inputs from its caller.
    if m2 > m1:
        m1, m2, s1, s2 = m2, m1, s2, s1
    remnant, _ = _native_msa_final_spin(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64), s1, s2,
        torch.tensor(f_ref, dtype=torch.float64), angles,
        compatibility="v223",
    )
    return angles, remnant


@pytest.mark.parametrize("case", CASES)
def test_intrinsic_limit_matches_torch_secular_and_remnant(case):
    frequencies = np.array([25., case[-1], 80.])
    expected, remnant = _torch_angles(case, frequencies)
    actual = one_spin_msa_v223_angles(frequencies, *case)
    for key in ("c1", "sav", "alpha", "gamma"):
        np.testing.assert_allclose(
            np.asarray(actual[key]), expected[key].detach().numpy(),
            rtol=0., atol=2e-9 if key in ("alpha", "gamma") else 3e-14,
            err_msg=key,
        )
    # The Torch v223 diagnostic reports the split upper auxiliary root here;
    # the intrinsic primitive names the physical, unsplit spin norm.
    np.testing.assert_allclose(
        np.asarray(actual["spin_squared"] + 1.e-9),
        expected["spin_squared"].detach().numpy(), rtol=0., atol=3e-14,
    )
    np.testing.assert_allclose(
        np.asarray(actual["final_spin"]), remnant.detach().numpy(),
        rtol=0., atol=3e-13,
    )
    assert np.isfinite(np.asarray(actual["alpha"])).all()
    assert np.isfinite(np.asarray(actual["gamma"])).all()
    assert np.asarray(actual["valid"]).all()
    np.testing.assert_allclose(
        np.asarray(actual["alpha"] + actual["gamma"])[1], 0., atol=1e-10,
    )


@pytest.mark.parametrize("case", CASES)
def test_opening_angle_uses_unsplit_scalar_reference(case):
    m1, m2, chi1, chi2, f_ref = case
    frequencies = np.array([25., 40., 80.])
    actual = one_spin_msa_v223_angles(frequencies, *case)
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = abs(m1 - m2) / total
    heavy, light = (chi1, chi2) if m1 >= m2 else (chi2, chi1)
    n1, n2 = max(m1, m2) / total, min(m1, m2) / total
    spin_l = n1**2 * heavy[2] + n2**2 * light[2]
    sigma_l = n2 * light[2] - n1 * heavy[2]
    v_ref = (np.pi * total * MSUN_SEC * f_ref)**(1/3)
    for index, f in enumerate(frequencies):
        v = (np.pi * total * MSUN_SEC * f)**(1/3)
        l_beta = float(_opening_l_3pn(v, eta, delta, spin_l, sigma_l))
        reference = scalar_beta(m1, m2, chi1, chi2, v_ref=v_ref, l_beta=l_beta)
        np.testing.assert_allclose(
            float(actual["beta"][index]), reference.beta,
            rtol=0., atol=3e-14,
        )
        np.testing.assert_allclose(
            float(actual["cos_beta"][index]), reference.cos_beta,
            rtol=0., atol=3e-14,
        )
    if chi1[:2] == chi2[:2] == (0., 0.):
        np.testing.assert_allclose(np.asarray(actual["beta"]), 0., atol=0.)


def test_intrinsic_limit_jit_batch_and_mass_exchange():
    f = jnp.array([25., 40., 80.])
    evaluate = jax.jit(one_spin_msa_v223_angles)
    rows = CASES[:3]
    batched = evaluate(
        f, jnp.array([row[0] for row in rows]), jnp.array([row[1] for row in rows]),
        jnp.array([row[2] for row in rows]), jnp.array([row[3] for row in rows]),
        jnp.array([row[4] for row in rows]),
    )
    assert batched["alpha"].shape == (3, 3)
    for i, (m1, m2, s1, s2, f_ref) in enumerate(rows):
        single = evaluate(f, m1, m2, jnp.array(s1), jnp.array(s2), f_ref)
        exchanged = evaluate(f, m2, m1, jnp.array(s2), jnp.array(s1), f_ref)
        for key in ("alpha", "gamma", "cos_beta", "final_spin"):
            np.testing.assert_allclose(
                np.asarray(batched[key][i]), np.asarray(single[key]), atol=3e-12,
            )
            np.testing.assert_allclose(
                np.asarray(exchanged[key]), np.asarray(single[key]), atol=3e-12,
            )


def test_longitudinal_spin_gradients_match_torch():
    f = jnp.array([25., 40., 80.])
    def jax_loss(z):
        state = one_spin_msa_v223_angles(
            f, 40., 20., jnp.array([.2, -.1, z]), jnp.array([0., 0., -.2]), 40.,
        )
        return jnp.sum(state["alpha"] + .7 * state["gamma"] +
                       .2 * state["cos_beta"]) + .3 * state["final_spin"]

    z = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    chi1 = torch.stack((torch.tensor(.2, dtype=torch.float64),
                        torch.tensor(-.1, dtype=torch.float64), z))
    chi2 = torch.tensor([0., 0., -.2], dtype=torch.float64)
    angles = assemble_precession_msa_angles_v223(
        torch.tensor(np.asarray(f)), 40., 20., chi1, chi2, f_ref=40.,
    )
    remnant, _ = _native_msa_final_spin(
        torch.tensor(40.), torch.tensor(20.), chi1, chi2, torch.tensor(40.),
        angles, compatibility="v223",
    )
    # The public physical opening is a separate component from the auxiliary
    # root beta returned by the Torch angle diagnostic.
    from torchwave.approximants.precession_msa_waveform_clean import physical_zero_nutation_cos_beta
    cos_beta = physical_zero_nutation_cos_beta(
        torch.tensor(np.asarray(f)), torch.tensor(40.), torch.tensor(20.),
        chi1, chi2,
    )
    torch_loss = (angles["alpha"] + .7 * angles["gamma"] +
                  .2 * cos_beta).sum() + .3 * remnant
    expected = torch.autograd.grad(torch_loss, z)[0].item()
    actual = float(jax.grad(jax_loss)(.4))
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)


def test_equal_mass_remnant_is_finite_under_jit_and_cond():
    case = CASES[2]
    f = jnp.array([25., 40., 80.])

    def evaluate():
        return one_spin_msa_v223_angles(f, *case)

    eager = evaluate()
    compiled = jax.jit(evaluate)()
    conditional = jax.lax.cond(
        True, lambda _: evaluate(), lambda _: evaluate(), None,
    )
    for result in (compiled, conditional):
        for key in ("mean_s1_l", "mean_s2_l", "final_spin", "alpha", "gamma"):
            assert np.isfinite(np.asarray(result[key])).all(), key
        for key in ("final_spin", "alpha", "gamma"):
            np.testing.assert_allclose(
                np.asarray(result[key]), np.asarray(eager[key]), rtol=2e-10,
                atol=1e-12, err_msg=key,
            )
