"""Public JAX generic XP v223 qualification on the distinct-root domain.

Pratten et al. (arXiv:2004.06503v2), Eqs. (8)--(13), Appendix C give the
rotation and observer projection.  Chatziioannou et al. (arXiv:1703.03967v3)
give the unequal-mass two-spin nutation equations.  Equal-mass generic rows
and coalescing-root initializations use the published NNLO angle fallback;
an explicitly selected MSA remnant remains independently domain-checked.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP
from jaxwave.approximants.precession_msa_v223_generic_projection import (
    generic_msa_v223_waveform,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP


CASES = (
    dict(mass1=35., mass2=25., spin1x=.2, spin1y=.1, spin1z=.3,
         spin2x=-.1, spin2y=.05, spin2z=-.2),
    dict(mass1=48., mass2=17., spin1x=.13, spin1y=-.2, spin1z=.26,
         spin2x=.04, spin2y=.12, spin2z=-.1),
)
F = np.array([0., 25., 40., 80., 160.])


def _relative_l2(actual, expected):
    expected = np.asarray(expected)
    return np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", (0, 1, 6))
def test_public_generic_matches_torch_complex_strain_with_zero_bin(case, convention):
    params = dict(case, distance=150., inclination=.7, phic=.3, tc=.001,
                  f_ref=20., phi0=0., prec_version=223, convention=convention,
                  final_spin_mod=3)
    jax_strain = JAXXP.evaluate(jnp.asarray(F), **params)
    torch_strain = TorchXP()(torch.as_tensor(F, dtype=torch.float64), **params)
    for actual, expected in zip(jax_strain, torch_strain):
        assert np.asarray(actual)[0] == 0.
        assert np.all(np.isfinite(np.asarray(actual)))
        assert _relative_l2(actual, expected.detach().numpy()) < 2e-10


def test_helper_masks_zero_bin_before_intrinsic_phase_and_passes_options(monkeypatch):
    import jaxwave.approximants.precession_msa_v223_generic_projection as module

    seen = []
    original = module.IMRPhenomXAS.evaluate

    def record(*args, **kwargs):
        seen.append((kwargs.get("phasing_convention"), kwargs.get("reference_frame")))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.IMRPhenomXAS, "evaluate", record)
    case = CASES[0]
    hp, hc, state = generic_msa_v223_waveform(
        jnp.asarray(F), case["mass1"], case["mass2"],
        jnp.asarray((case["spin1x"], case["spin1y"], case["spin1z"])),
        jnp.asarray((case["spin2x"], case["spin2y"], case["spin2z"])),
        distance=150., inclination=.7, phic=.3, f_ref=20.,
        phasing_convention="passed-through", reference_frame="passed-through",
        return_state=True,
    )
    assert seen == [("passed-through", "passed-through")]
    assert np.asarray(hp)[0] == np.asarray(hc)[0] == 0.
    for name in ("alpha", "gamma", "cos_beta", "final_spin"):
        assert np.all(np.isfinite(np.asarray(state[name])))


def test_nonzero_azimuth_generic_rows_use_clean_source_chart():
    f = jnp.asarray([0., 25., 40., 80.])
    params = dict(CASES[0], phi0=.4, distance=150., inclination=.7, phic=.3,
                  f_ref=20., prec_version=223, convention=1,
                  final_spin_mod=3)
    fallback = dict(params, _force_legacy=True)
    actual = JAXXP.evaluate(f, **params)
    legacy = JAXXP.evaluate(f, **fallback)
    torch_f = torch.tensor(np.asarray(f), dtype=torch.float64)
    torch_actual = TorchXP()(torch_f, **params)
    torch_legacy = TorchXP()(torch_f, **fallback)
    for left, right, torch_left, torch_right in zip(
        actual, legacy, torch_actual, torch_legacy,
    ):
        assert _relative_l2(left, right) > 1e-5
        assert _relative_l2(torch_left.detach().numpy(), torch_right.detach().numpy()) > 1e-5
        assert _relative_l2(left, torch_left.detach().numpy()) < 2e-10
        assert np.all(np.isfinite(np.asarray(left)))


def test_near_aligned_coalescing_roots_use_nnlo_fallback():
    f = jnp.asarray([25., 40., 55., 80.])
    params = dict(
        mass1=36., mass2=22., spin1x=1e-10, spin1y=0., spin1z=.35,
        spin2x=-5e-11, spin2y=0., spin2z=-.25, distance=150.,
        inclination=.65, phic=.3, f_ref=40., phi0=0., prec_version=223,
        convention=1, final_spin_mod=3,
    )
    fallback = dict(params, _force_legacy=True)
    actual = JAXXP.evaluate(f, **params)
    legacy = JAXXP.evaluate(f, **fallback)
    torch_f = torch.tensor(np.asarray(f), dtype=torch.float64)
    torch_actual = TorchXP()(torch_f, **params)
    torch_legacy = TorchXP()(torch_f, **fallback)
    for left, right, torch_left, torch_right in zip(
        actual, legacy, torch_actual, torch_legacy,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   rtol=1e-12, atol=0.)
        torch.testing.assert_close(torch_left, torch_right, rtol=1e-12, atol=0.)
        assert _relative_l2(left, torch_left.detach().numpy()) < 2e-10
        assert np.all(np.isfinite(np.asarray(left)))


def test_generic_jit_mixed_batch_and_spin_gradient_match_torch():
    f = jnp.asarray([25., 40., 80.])
    common = dict(distance=150., inclination=.7, phic=.3, f_ref=20.,
                  prec_version=223, convention=1, final_spin_mod=3)
    rows = dict(
        mass1=jnp.array([35., 48., 35.]), mass2=jnp.array([25., 17., 25.]),
        spin1x=jnp.array([.2, .13, .2]), spin1y=jnp.array([.1, -.2, .1]),
        spin1z=jnp.array([.3, .26, .3]), spin2x=jnp.array([-.1, .04, 0.]),
        spin2y=jnp.array([.05, .12, 0.]), spin2z=jnp.array([-.2, -.1, -.2]),
    )
    def evaluate(sx):
        return JAXXP.evaluate(f, **dict(rows, spin1x=sx), **common)
    hp, hc = jax.jit(evaluate)(rows["spin1x"])
    assert hp.shape == hc.shape == (3, 3)
    for index in range(3):
        row = {name: value[index] for name, value in rows.items()}
        expected = JAXXP.evaluate(f, **dict(row, **common))
        for batched, scalar in zip((hp[index], hc[index]), expected):
            assert _relative_l2(batched, scalar) < 2e-10

    def jax_loss(sx):
        hp, hc = JAXXP.evaluate(f, **dict(CASES[0], spin1x=sx), **common)
        return 1e44 * jnp.sum(jnp.abs(hp)**2 + jnp.abs(hc)**2)
    sx = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    thp, thc = TorchXP()(torch.tensor(np.asarray(f), dtype=torch.float64),
                         **dict(CASES[0], spin1x=sx), **common)
    torch_loss = 1e44 * (thp.abs().square() + thc.abs().square()).sum()
    torch_gradient = torch.autograd.grad(torch_loss, sx)[0].detach().numpy()
    jax_gradient = np.asarray(jax.grad(jax_loss)(jnp.asarray(.2)))
    np.testing.assert_allclose(jax_gradient, torch_gradient, rtol=2e-8, atol=1e-10)
