"""All published pre-NQC modes against independent scalar paper equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dev.native_rom.reference import flux as scalar_flux
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux

jax.config.update("jax_enable_x64", True)


def _case(m1=30., m2=20., chi1=.2, chi2=-.1, omega=.06,
          omega_zero_radial=.057, pphi=3.2, heff=.93, hreal=.981):
    mass = m1+m2
    state = jnp.asarray([9., 0., -.01, pphi])
    args = dict(omega=jnp.asarray(omega), omega_zero_radial=jnp.asarray(omega_zero_radial),
                heff_over_mu=jnp.asarray(heff), hreal_over_M=jnp.asarray(hreal),
                x1=jnp.asarray(m1/mass), x2=jnp.asarray(m2/mass),
                chi1=jnp.asarray(chi1), chi2=jnp.asarray(chi2))
    return state, args


def _scalar_modes(m1, m2, s1, s2, w, w0, j, he, hr, *,
                  spin_truncation="tabulated", mode_21_higher_spin="taylor_f"):
    base = (m1, m2, s1, s2, w, w0)
    h_args, j_args = (*base, he, hr), (*base, j, hr)
    mode21 = (scalar_flux.factorized_21_spin_amplitude if mode_21_higher_spin == "taylor_f"
              else scalar_flux.factorized_21_rho_hybrid_amplitude)
    modes = {
        (2, 2): scalar_flux.factorized_22_amplitude(*h_args),
        (2, 1): mode21(*j_args),
        (3, 1): scalar_flux.factorized_31_amplitude(*h_args),
        (3, 2): scalar_flux.factorized_32_amplitude(*j_args),
        (4, 1): scalar_flux.factorized_41_amplitude(*j_args),
        (4, 2): scalar_flux.factorized_42_amplitude(*h_args),
        (4, 3): scalar_flux.factorized_43_amplitude(*j_args),
        (5, 5): scalar_flux.factorized_55_amplitude(*h_args),
        (6, 2): scalar_flux.factorized_62_amplitude(*h_args),
    }
    grouped = (
        (((3, 3), (4, 4)), scalar_flux.factorized_33_44_amplitudes(*h_args)),
        (((5, 4), (5, 2)), scalar_flux.factorized_54_52_amplitudes(*j_args)),
        (((5, 3), (5, 1)), scalar_flux.factorized_53_51_amplitudes(*h_args)),
        (((6, 6), (6, 4)), scalar_flux.factorized_66_64_amplitudes(*h_args)),
        (((6, 5), (6, 3), (6, 1)), scalar_flux.factorized_6_odd_amplitudes(*j_args)),
        (((7, 7), (7, 5), (7, 3), (7, 1)), scalar_flux.factorized_7_odd_amplitudes(*h_args)),
        (((7, 6), (7, 4), (7, 2)), scalar_flux.factorized_7_even_amplitudes(*j_args)),
        (((8, 8), (8, 6), (8, 4), (8, 2)),
         scalar_flux.factorized_8_even_amplitudes(*h_args, spin_truncation=spin_truncation)),
        (((8, 7), (8, 5), (8, 3), (8, 1)), scalar_flux.factorized_8_odd_amplitudes(*j_args)),
    )
    for keys, values in grouped:
        modes.update(zip(keys, values))
    return modes


@pytest.mark.parametrize("point", [
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.6, .89, .973),
])
@pytest.mark.parametrize("spin_truncation,mode_21_higher_spin", [
    ("tabulated", "taylor_f"),
    ("generic-leading-ell8", "taylor_f"),
    ("tabulated", "rho_hybrid"),
])
def test_all_35_frozen_scalar_modes_and_flux(point, spin_truncation, mode_21_higher_spin):
    m1, m2, s1, s2, w, w0, j, he, hr = point
    state, args = _case(m1, m2, s1, s2, w, w0, j, he, hr)
    model = SEOBNRv4PublishedFlux(
        spin_truncation=spin_truncation, mode_21_higher_spin=mode_21_higher_spin)
    expected = _scalar_modes(*point, spin_truncation=spin_truncation,
                             mode_21_higher_spin=mode_21_higher_spin)
    actual = model.amplitudes(state, **args)
    assert set(actual) == set(expected)
    for mode in expected:
        np.testing.assert_allclose(actual[mode], float(expected[mode]), rtol=4e-12, atol=2e-14,
                                   err_msg=f"mode {mode}")
    np.testing.assert_allclose(model.energy_flux(state, **args),
                               float(scalar_flux.emitted_flux(w, expected)), rtol=4e-12)


def test_batch_jit_and_gradient_match_scalar_evaluation():
    state, args = _case()
    model = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
    states = jnp.stack((state, state.at[3].set(2.9)))
    omegas = jnp.asarray([.06, .07])
    batched = jax.jit(model.energy_flux)(states, **(args | {"omega": omegas}))
    separate = jnp.stack([model.energy_flux(states[i], **(args | {"omega": omegas[i]}))
                          for i in range(2)])
    np.testing.assert_allclose(batched, separate, rtol=3e-14)
    derivative = jax.jit(jax.grad(lambda w: model.energy_flux(state, **(args | {"omega": w}))))(
        args["omega"])
    step = 2e-6
    finite_difference = (model.energy_flux(state, **(args | {"omega": .06+step}))
                         -model.energy_flux(state, **(args | {"omega": .06-step})))/(2*step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-8)


def test_equal_mass_odd_mode_flux_has_finite_hessian():
    state, args = _case(m1=20., m2=20., chi1=0., chi2=0.)
    model = SEOBNRv4PublishedFlux(spin_truncation="tabulated")

    def signed_mode(delta):
        return model._signed_amplitudes(
            state, **(args | {"x1": .5+delta, "x2": .5-delta}))[(2, 1)]

    def odd_flux(delta):
        return signed_mode(delta)**2

    d2 = jax.jit(jax.grad(jax.grad(odd_flux)))(jnp.asarray(0.))
    slope = jax.grad(signed_mode)(jnp.asarray(0.))
    np.testing.assert_allclose(d2, 2*slope**2, rtol=2e-13)
    assert np.isfinite(d2) and d2 > 0


def test_required_choices_and_compiled_invalid_row():
    with pytest.raises(TypeError):
        SEOBNRv4PublishedFlux()
    with pytest.raises(ValueError):
        SEOBNRv4PublishedFlux(spin_truncation="unknown")
    state, args = _case()
    model = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
    states = jnp.stack((state, state.at[0].set(-1.)))
    with pytest.raises(ValueError):
        model.energy_flux(states, **args)
    compiled = jax.jit(model.energy_flux)(states, **args)
    assert np.isfinite(compiled[0]) and np.isnan(compiled[1])
