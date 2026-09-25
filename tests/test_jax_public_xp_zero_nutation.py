"""Public JAX XP v223 zero-nutation route against clean Torch composition."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS as JAXXAS
from jaxwave.approximants.precession_msa_v223_one_spin_projection import (
    one_spin_msa_v223_waveform,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP


F = np.array([0.0, 25.0, 40.0, 80.0])
CASES = [
    dict(mass1=40., mass2=20., spin1x=.2, spin1y=-.1, spin1z=.4,
         spin2x=0., spin2y=0., spin2z=-.2),
    dict(mass1=20., mass2=40., spin1x=0., spin1y=0., spin1z=-.2,
         spin2x=.2, spin2y=-.1, spin2z=.4),
    dict(mass1=30., mass2=30., spin1x=.15, spin1y=.07, spin1z=.35,
         spin2x=0., spin2y=0., spin2z=-.1),
    dict(mass1=40., mass2=20., spin1x=0., spin1y=0., spin1z=.3,
         spin2x=0., spin2y=0., spin2z=-.1),
    dict(mass1=40., mass2=20., spin1x=1e-8, spin1y=-1e-8, spin1z=.4,
         spin2x=0., spin2y=0., spin2z=-.2),
]


def _relative_l2(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return np.linalg.norm(actual-expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("f_ref", (None, 20., 25., 40.))
def test_exact_aligned_xp_collapses_to_xas_on_carrier_phase_branch(f_ref):
    # Pratten et al. (arXiv:2004.06503v2), Eqs. (7), (17): beta=0 makes
    # alpha+gamma frequency-independent.  An omitted public f_ref leaves the
    # XAS carrier phase unanchored even though MSA angles use 20 Hz internally.
    f = jnp.linspace(25., 300., 24)
    params = dict(mass1=40., mass2=20., spin1z=.4, spin2z=-.2,
                  distance=100., inclination=.7, phic=.3, tc=0.,
                  f_ref=f_ref)
    hp, hc = JAXXP.evaluate(
        f, **params, prec_version=223, convention=1, final_spin_mod=3,
    )
    expected = JAXXAS.evaluate(
        f=f, mass1=40., mass2=20., chi1=.4, chi2=-.2,
        distance=100., inclination=.7, phi_c=.3, t_c=0., f_ref=f_ref,
    )
    for actual, target in zip((hp, hc), expected):
        # The unanchored XAS phase amplifies compiler-staging roundoff: on the
        # 24-bin grid, eager XP/XAS differed by 1.15e-11, whereas matched JIT
        # XP/XAS differed by 5.4e-15 at the same exact beta=0 endpoint.
        assert _relative_l2(actual, target) < (1e-10 if f_ref is None else 1e-12)
    if f_ref is None:
        compiled_xp = jax.jit(lambda: JAXXP.evaluate(
            f, **params, prec_version=223, convention=1, final_spin_mod=3,
        ))()
        compiled_xas = jax.jit(lambda: JAXXAS.evaluate(
            f=f, mass1=40., mass2=20., chi1=.4, chi2=-.2,
            distance=100., inclination=.7, phi_c=.3, t_c=0., f_ref=None,
        ))()
        for actual, target in zip(compiled_xp, compiled_xas):
            assert _relative_l2(actual, target) < 1e-12

    _, _, state = one_spin_msa_v223_waveform(
        f, 40., 20., jnp.array([0., 0., .4]), jnp.array([0., 0., -.2]),
        distance=100., inclination=.7, phic=.3, f_ref=f_ref,
        convention=1, return_state=True,
    )
    assert np.ptp(np.asarray(state["alpha"] + state["gamma"])) < 1e-11
    np.testing.assert_equal(np.asarray(state["endpoint_weight"]), 1.0)


def test_near_aligned_xp_converges_to_xas_with_jit():
    f = jnp.linspace(25., 300., 24)
    params = dict(mass1=40., mass2=20., spin1z=.4, spin2z=-.2,
                  distance=100., inclination=.7, phic=.3, tc=0.,
                  f_ref=None, prec_version=223, convention=1,
                  final_spin_mod=3)
    target = JAXXAS.evaluate(
        f=f, mass1=40., mass2=20., chi1=.4, chi2=-.2,
        distance=100., inclination=.7, phi_c=.3, t_c=0., f_ref=None,
    )
    evaluate = jax.jit(lambda spinx: JAXXP.evaluate(f, **dict(params, spin1x=spinx)))
    tiny = evaluate(jnp.asarray(1e-8))
    exact = evaluate(jnp.asarray(0.))
    for near, aligned, expected in zip(tiny, exact, target):
        assert _relative_l2(aligned, expected) < 1e-10
        assert _relative_l2(near, expected) < 1e-7


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", (0, 1, 6))
def test_public_one_spin_matches_torch_physical_boundary(case, convention):
    params = dict(case, distance=100., inclination=.7, phic=.3,
                  phi0=.2, tc=0., f_ref=40., convention=convention,
                  prec_version=223, final_spin_mod=3)
    jhp, jhc = JAXXP.evaluate(jnp.asarray(F), **params)
    thp, thc = TorchXP()(torch.as_tensor(F, dtype=torch.float64), **params)
    for actual, expected in zip((jhp, jhc), (thp, thc)):
        assert np.asarray(actual).shape == F.shape
        assert np.asarray(actual)[0] == 0.
        assert _relative_l2(actual, expected.detach().numpy()) < 2e-7


def test_public_one_spin_jit_batch_and_gradient():
    params = dict(distance=100., inclination=.7, phic=.3, phi0=.2,
                  f_ref=40., prec_version=223, final_spin_mod=3)
    f = jnp.asarray([25., 40., 80.])
    batch = dict(
        mass1=jnp.array([40., 40.]), mass2=jnp.array([20., 20.]),
        spin1x=jnp.array([0., .2]), spin1y=jnp.array([0., -.1]),
        spin1z=jnp.array([.3, .4]), spin2x=jnp.zeros(2),
        spin2y=jnp.zeros(2), spin2z=jnp.array([-.1, -.2]),
    )
    def evaluate_batch(sx):
        return JAXXP.evaluate(f, **dict(batch, spin1x=sx), **params)
    hp, hc = jax.jit(evaluate_batch)(batch["spin1x"])
    assert hp.shape == hc.shape == (2, 3)
    for i in range(2):
        row = {key: value[i] for key, value in batch.items()}
        single = JAXXP.evaluate(f, **dict(row, **params))
        np.testing.assert_allclose(np.asarray(hp[i]), np.asarray(single[0]), rtol=1e-8, atol=2e-31)
        np.testing.assert_allclose(np.asarray(hc[i]), np.asarray(single[1]), rtol=1e-8, atol=2e-31)
    def amplitude(sx):
        row = dict(CASES[0], spin1x=sx)
        strain = JAXXP.evaluate(f, **dict(row, **params))[0]
        return jnp.sum(jnp.abs(strain) ** 2)
    assert np.isfinite(float(jax.grad(amplitude)(jnp.asarray(.2))))


def test_generic_row_retains_legacy_route():
    f = jnp.asarray([25., 40., 80.])
    params = dict(mass1=40., mass2=20., spin1x=.2, spin1y=-.1,
                  spin1z=.4, spin2x=.1, spin2y=.05, spin2z=-.2,
                  distance=100., inclination=.7, phic=.3, f_ref=40.,
                  prec_version=223, final_spin_mod=3)
    actual = JAXXP.evaluate(f, **params)
    legacy = JAXXP.evaluate(f, **dict(params, _force_legacy=True))
    for left, right in zip(actual, legacy):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=0.)


def test_default_reference_frequency_matches_torch():
    params = dict(CASES[0], distance=100., inclination=.7, phic=.3,
                  phi0=.2, prec_version=223, final_spin_mod=3)
    actual = JAXXP.evaluate(jnp.asarray(F), **params)
    expected = TorchXP()(torch.as_tensor(F, dtype=torch.float64), **params)
    for left, right in zip(actual, expected):
        assert _relative_l2(left, right.detach().numpy()) < 2e-7


def test_jit_mixed_clean_and_generic_batch_keeps_row_routes():
    params = dict(distance=100., inclination=.7, phic=.3,
                  f_ref=40., prec_version=223, final_spin_mod=3)
    f = jnp.asarray([25., 40., 80.])
    rows = dict(
        mass1=jnp.array([40., 40.]), mass2=jnp.array([20., 20.]),
        spin1x=jnp.array([.2, .2]), spin1y=jnp.array([-.1, -.1]),
        spin1z=jnp.array([.4, .4]), spin2x=jnp.array([0., .1]),
        spin2y=jnp.array([0., .05]), spin2z=jnp.array([-.2, -.2]),
    )
    hp, hc = jax.jit(lambda spin2x: JAXXP.evaluate(
        f, **dict(rows, spin2x=spin2x), **params,
    ))(rows["spin2x"])
    for index in range(2):
        row = {name: value[index] for name, value in rows.items()}
        expected = JAXXP.evaluate(f, **dict(row, **params))
        np.testing.assert_allclose(np.asarray(hp[index]), np.asarray(expected[0]), rtol=1e-8, atol=2e-31)
        np.testing.assert_allclose(np.asarray(hc[index]), np.asarray(expected[1]), rtol=1e-8, atol=2e-31)
