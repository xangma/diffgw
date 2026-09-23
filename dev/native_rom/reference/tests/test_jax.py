"""Independent paper reference against the JAX component implementation."""

import jax
import jax.numpy as jnp
import pytest

from dev.native_rom.reference.hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, effective_spin_mapping, energy_map,
    nonspinning_effective_hamiltonian, radial_potentials,
)
from dev.native_rom.reference.initial_conditions import leading_adiabatic_state
from dev.native_rom.reference.published import calibration, evaluate, mass_spin, peak_targets, ringdown_shape
from dev.native_rom.reference.ringdown import h22
from dev.native_rom.reference.remnant import final_mass_fraction, final_spin_projection
from dev.native_rom.reference.qnm import BERTI_AUTHOR_220_SAMPLES, COOK_AUTHOR_220_SAMPLES
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from jaxwave.approximants.eob.merger_ringdown import merger_ringdown_22, seobnrv4_merger_shape
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets, seobnrv4_nqc_time_offset
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions

jax.config.update("jax_enable_x64", True)


def scalar(x):
    return jnp.asarray(x, dtype=jnp.float64)


@pytest.mark.parametrize("m1,m2,s1,s2,omega", [
    (30., 20., 0., 0., .04),
    (30., 20., .2, -.1, .04),
])
def test_leading_adiabatic_initial_state_against_scalar_reference(m1, m2, s1, s2, omega):
    torque = -.0007
    class ConstantTorque:
        def radiation_reaction_forces(self, frequency, pr_star, pphi, *, r, hamiltonian):
            return jnp.zeros_like(pphi), jnp.full_like(pphi, torque)

    n = m1*m2/(m1+m2)**2
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2),
                              delta=scalar((m1-m2)/(m1+m2)))
    actual = find_quasicircular_initial_conditions(ham, ConstantTorque(), scalar(omega))
    expected = leading_adiabatic_state(m1, m2, s1, s2, omega, torque)
    for value, reference in zip(actual, expected):
        assert float(value) == pytest.approx(float(reference), rel=2e-11, abs=2e-13)


@pytest.mark.parametrize("m1,m2,s1,s2,r", [
    (30., 20., 0., 0., 9.), (45., 5., .8, -.3, 7.), (20., 20., -.9, .7, 12.),
])
def test_radial_potentials_and_energy_map_against_reference(m1, m2, s1, s2, r):
    n, a, k = body_parameters(m1, m2, s1, s2)
    expected = radial_potentials(r, n, a, k)
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2))
    actual = ham.deformed_kerr_potentials(scalar(r))
    for key, value in zip(("delta_t", "delta_r", "delta_t_prime", "lambda_t",
                           "lambda_t_prime", "xi"), actual):
        assert float(value) == pytest.approx(float(expected[key]), rel=2e-12, abs=2e-12)
    lapse, g_rr, xi = ham.metric_potentials(scalar(r))
    for value, key in zip((lapse, g_rr, xi), ("lapse_squared", "g_rr", "xi")):
        assert float(value) == pytest.approx(float(expected[key]), rel=2e-12, abs=2e-12)
    h_eff = scalar(.94)
    total, _ = energy_map(n, .94)
    assert float(ham.real_hamiltonian_from_eff(h_eff)) == pytest.approx(float(total), rel=2e-14)
    actual_eff = ham.effective_hamiltonian(scalar(r), scalar(.01), scalar(3.))
    _, actual_reduced = energy_map(n, float(actual_eff))
    assert float(ham.reduced_binding_hamiltonian(scalar(r), scalar(.01), scalar(3.))) == pytest.approx(
        float(actual_reduced), rel=2e-13, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2", [(30., 20., 0., 0.), (45., 5., .8, -.3), (20., 20., -.9, .7)])
def test_fits_against_reference(m1, m2, s1, s2):
    n, c = mass_spin(m1, m2, s1, s2)
    params = seobnrv4_nqc_parameters(*map(scalar, (m1, m2, s1, s2)))
    assert abs(float(params[0])-n) < 1e-15
    assert abs(float(params[1])-c) < 1e-15
    expected = evaluate(calibration(), n, c)
    hamiltonian = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2))
    k, d_so, d_ss = hamiltonian.calibration_parameters(scalar(10.))
    for actual, key in zip((k, d_so, d_ss, seobnrv4_nqc_time_offset(*params)),
                           ("K", "d_SO", "d_SS", "peak_lag")):
        assert float(actual) == pytest.approx(float(expected[key]), rel=2e-14, abs=1e-12)
    targets = seobnrv4_nqc_targets(*params)
    for actual, key in zip(targets, peak_targets()):
        expected_value = evaluate(peak_targets(), n, c)[key]
        assert float(actual) == pytest.approx(float(expected_value), rel=2e-14, abs=1e-12)


@pytest.mark.parametrize("m1,m2,r,pr_star,pphi", [
    (30., 20., 9., .01, 3.), (20., 20., 6.5, -.08, 2.7),
])
def test_nonspinning_effective_hamiltonian_against_reference(m1, m2, r, pr_star, pphi):
    n, a, k = body_parameters(m1, m2, 0, 0)
    assert a == 0
    expected = nonspinning_effective_hamiltonian(r, pr_star, pphi, n, k)
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(0), scalar(0))
    actual = ham.effective_hamiltonian(scalar(r), scalar(pr_star), scalar(pphi))
    assert float(actual) == pytest.approx(float(expected), rel=2e-12)


@pytest.mark.parametrize("m1,m2,s1,s2,r,q,p", [
    (30., 20., .6, -.2, 9., .15, .002),
    (45., 5., -.8, .3, 7., .24, .01),
    (20., 20., -.9, .7, 12., .09, 0.),
])
def test_effective_spin_mapping_against_reference(m1, m2, s1, s2, r, q, p):
    n, _, _ = body_parameters(m1, m2, s1, s2)
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2))
    expected = effective_spin_mapping(m1, m2, s1, s2, r, q, p)
    actual = ham.effective_spin(scalar(r), scalar(q), scalar(p))
    assert float(actual) == pytest.approx(float(expected), rel=2e-12, abs=2e-13)


@pytest.mark.parametrize("m1,m2,s1,s2,r,pr_star,pphi", [
    (30., 20., .6, -.2, 9., .01, 3.),
    (45., 5., -.8, .3, 7., -.03, 2.6),
    (20., 20., -.9, .7, 12., 0., 3.5),
])
def test_aligned_effective_hamiltonian_against_reference(m1, m2, s1, s2, r, pr_star, pphi):
    n, _, _ = body_parameters(m1, m2, s1, s2)
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2))
    expected = aligned_effective_hamiltonian(m1, m2, s1, s2, r, pr_star, pphi)
    actual = ham.effective_hamiltonian(scalar(r), scalar(pr_star), scalar(pphi))
    assert float(actual) == pytest.approx(float(expected), rel=2e-12, abs=2e-13)


def test_ringdown_against_reference():
    n, c = 0.24, 0.15
    times = scalar([0., .01, .5, 2., 10.])
    sigma = scalar(.55) + 1j*scalar(-.08)
    shape = seobnrv4_merger_shape(scalar(n), scalar(c))
    printed = evaluate(ringdown_shape(), n, c)
    for actual, key in zip(shape, ("c1f", "c2f", "d1f", "d2f")):
        assert float(actual) == pytest.approx(float(printed[key]), rel=2e-14, abs=1e-14)
    actual = merger_ringdown_22(times, sigma, scalar(.35), scalar(0.), scalar(.32), scalar(.4), shape)
    for t, sample in zip(times.tolist(), actual.tolist()):
        expected = complex(h22(t, nu=n, chi=c, sigma=complex(.55, -.08),
                               amplitude=.35, frequency=.32, phase=.4))
        assert sample == pytest.approx(expected, rel=3e-13, abs=3e-14)


@pytest.mark.parametrize("m1,m2,s1,s2", [
    (30., 20., 0., 0.), (45., 5., .8, -.3),
    (20., 20., -.9, -.9), (20., 20., .8, .8),
])
def test_remnant_mass_against_reference(m1, m2, s1, s2):
    actual = eob_final_mass_fraction(*map(scalar, (m1, m2, s1, s2)))
    expected = final_mass_fraction(m1, m2, s1, s2)
    assert float(actual) == pytest.approx(float(expected), rel=2e-14, abs=1e-14)


@pytest.mark.parametrize("fit_order", [(1, 2), (3, 3), (3, 4)])
@pytest.mark.parametrize("m1,m2,s1,s2", [
    (30., 20., 0., 0.), (45., 5., .8, -.3), (20., 20., -.9, -.9),
])
def test_hofmann_spin_against_reference(fit_order, m1, m2, s1, s2):
    actual = hofmann_final_spin(*map(scalar, (m1, m2, s1, s2)), fit_order=fit_order)
    expected = final_spin_projection(m1, m2, s1, s2, fit_order=fit_order)
    assert float(actual) == pytest.approx(float(expected), rel=3e-13, abs=3e-14)


@pytest.mark.parametrize("spin,frequency,damping", BERTI_AUTHOR_220_SAMPLES)
def test_qnm_against_berti_numerical_table(spin, frequency, damping):
    # Berti author archive l2.tar.gz, n1l2m2.dat; not the printed four-decimal table.
    actual_frequency, actual_damping = evaluate_kerr_qnm_220(scalar(spin))
    assert float(actual_frequency) == pytest.approx(frequency, rel=0, abs=1e-9)
    assert float(actual_damping) == pytest.approx(damping, rel=0, abs=1e-9)


@pytest.mark.parametrize("spin,frequency,damping", COOK_AUTHOR_220_SAMPLES)
def test_qnm_against_cook_independent_author_data(spin, frequency, damping):
    # The signed-spin JAX frontend intentionally clamps above |spin|=0.9999.
    actual_frequency, actual_damping = evaluate_kerr_qnm_220(scalar(spin))
    assert float(actual_frequency) == pytest.approx(frequency, rel=0, abs=1e-10)
    assert float(actual_damping) == pytest.approx(damping, rel=0, abs=1e-10)
