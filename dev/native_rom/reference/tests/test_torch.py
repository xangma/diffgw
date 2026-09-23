"""Independent paper reference against the Torch component implementation."""

import pytest
import torch

from dev.native_rom.reference import flux as scalar_flux
from dev.native_rom.reference.flux import (
    emitted_flux, factorized_22_amplitude, factorized_33_44_amplitudes,
    factorized_21_nonspin_amplitude, factorized_21_spin_amplitude,
    factorized_21_rho_hybrid_amplitude,
    factorized_31_amplitude, factorized_32_amplitude, factorized_42_amplitude,
    factorized_43_amplitude, factorized_55_amplitude, factorized_62_amplitude,
    factorized_54_52_amplitudes, factorized_66_64_amplitudes,
    factorized_6_odd_amplitudes, factorized_7_even_amplitudes,
    factorized_8_even_amplitudes, factorized_8_odd_amplitudes,
    factorized_41_amplitude, factorized_53_51_amplitudes,
    factorized_7_odd_amplitudes,
    newtonian_22_mode,
)
from dev.native_rom.reference.hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, effective_spin_mapping, energy_map,
    nonspinning_effective_hamiltonian, radial_potentials,
)
from dev.native_rom.reference.initial_conditions import leading_adiabatic_state
from dev.native_rom.reference.published import calibration, evaluate, mass_spin, peak_targets, ringdown_shape
from dev.native_rom.reference.ringdown import h22
from dev.native_rom.reference.remnant import final_mass_fraction, final_spin_projection
from dev.native_rom.reference.qnm import (
    BERTI_AUTHOR_220_SAMPLES, COOK_AUTHOR_220_NEAR_EXTREMAL,
    COOK_AUTHOR_220_SAMPLES,
)
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.eob.merger_ringdown import merger_ringdown_22, seobnrv4_merger_shape
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_targets, seobnrv4_nqc_time_offset
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22
from torchwave.approximants.eob.factorized_modes import energy_flux_from_modes


def scalar(x):
    return torch.tensor(x, dtype=torch.float64)


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
    # Binding energy map is checked directly from the public effective H.
    actual_eff = ham.effective_hamiltonian(scalar(r), scalar(.01), scalar(3.))
    _, actual_reduced = energy_map(n, float(actual_eff))
    assert float(ham.reduced_binding_hamiltonian(scalar(r), scalar(.01), scalar(3.))) == pytest.approx(
        float(actual_reduced), rel=2e-13, abs=2e-14)


def test_mode_flux_normalization_against_reference():
    omega = scalar(.04)
    mode = complex(newtonian_22_mode(.24, .04, .37))
    modes = {(2, 2): torch.tensor(mode, dtype=torch.complex128),
             (3, 2): torch.tensor(2*mode, dtype=torch.complex128)}
    actual = energy_flux_from_modes(modes, omega)
    expected = emitted_flux(.04, {(2, 2): mode, (3, 2): 2*mode})
    assert float(actual) == pytest.approx(float(expected), rel=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .6, -.2, .07, .068, .91, .978),
    (20., 20., -.9, .7, .11, .105, .89, .972),
])
def test_factorized_22_amplitude_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_22_amplitude(m1, m2, s1, s2, w, w0, he, hr)
    actual = seobnrv4_published_amplitude22(
        scalar(w), omega_zero_radial=scalar(w0), heff_over_mu=scalar(he),
        hreal_over_M=scalar(hr), x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    assert float(actual) == pytest.approx(float(expected), rel=2e-12, abs=2e-14)
    from_flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., .01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(2, 2)]
    assert float(from_flux) == pytest.approx(float(expected), rel=2e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (20., 20., .7, -.4, .08, .077, .91, .975),
    (45., 5., -.8, .3, .10, .095, .89, .973),
])
def test_factorized_33_44_amplitudes_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_33_44_amplitudes(m1, m2, s1, s2, w, w0, he, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((3, 3), (4, 4)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_55_amplitude_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_55_amplitude(m1, m2, s1, s2, w, w0, he, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(5, 5)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_54_52_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_54_52_amplitudes(m1, m2, s1, s2, w, w0, j, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((5, 4), (5, 2)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_7_even_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_7_even_amplitudes(m1, m2, s1, s2, w, w0, j, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((7, 6), (7, 4), (7, 2)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_6_odd_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_6_odd_amplitudes(m1, m2, s1, s2, w, w0, j, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((6, 5), (6, 3), (6, 1)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_8_odd_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_8_odd_amplitudes(m1, m2, s1, s2, w, w0, j, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((8, 7), (8, 5), (8, 3), (8, 1)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_41_amplitude_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_41_amplitude(m1, m2, s1, s2, w, w0, j, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    assert float(modes[4, 1]) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_53_51_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_53_51_amplitudes(m1, m2, s1, s2, w, w0, he, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((5, 3), (5, 1)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_7_odd_amplitudes_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_7_odd_amplitudes(m1, m2, s1, s2, w, w0, he, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((7, 7), (7, 5), (7, 3), (7, 1)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_31_amplitude_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_31_amplitude(m1, m2, s1, s2, w, w0, he, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(3, 1)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_66_64_amplitudes_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_66_64_amplitudes(m1, m2, s1, s2, w, w0, he, hr)
    modes = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((6, 6), (6, 4)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_62_amplitude_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_62_amplitude(m1, m2, s1, s2, w, w0, he, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(6, 2)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("spin_truncation", ["tabulated", "generic-leading-ell8"])
@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_8_even_amplitudes_against_reference(
        m1, m2, s1, s2, w, w0, he, hr, spin_truncation):
    expected = factorized_8_even_amplitudes(
        m1, m2, s1, s2, w, w0, he, hr, spin_truncation=spin_truncation)
    modes = SEOBNRv4PublishedFlux(spin_truncation=spin_truncation).amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))
    for mode, value in zip(((8, 8), (8, 6), (8, 4), (8, 2)), expected):
        assert float(modes[mode]) == pytest.approx(float(value), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,he,hr", [
    (30., 20., 0., 0., .04, .039, .95, .988),
    (30., 20., .2, -.1, .06, .057, .93, .981),
    (45., 5., -.8, .3, .10, .095, .89, .973),
    (20., 20., -.9, .7, .08, .077, .91, .975),
])
def test_factorized_42_amplitude_against_reference(m1, m2, s1, s2, w, w0, he, hr):
    expected = factorized_42_amplitude(m1, m2, s1, s2, w, w0, he, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, 3.]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(4, 2)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (45., 5., -.8, .3, .10, .095, 2.8, .89, .973),
    (20., 20., -.9, .7, .08, .077, 3., .91, .975),
])
def test_factorized_43_amplitude_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_43_amplitude(m1, m2, s1, s2, w, w0, j, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="tabulated").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(4, 3)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., 0., 0., .04, .039, 3., .95, .988),
    (30., 20., .2, -.1, .06, .057, 3.2, .93, .981),
    (20., 20., .7, -.4, .08, .077, 2.8, .91, .975),
    (45., 5., -.8, .3, .10, .095, 2.6, .89, .973),
])
def test_factorized_32_amplitude_against_reference(m1, m2, s1, s2, w, w0, j, he, hr):
    expected = factorized_32_amplitude(m1, m2, s1, s2, w, w0, j, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(3, 2)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,w,w0,j,he,hr", [
    (30., 20., .04, .039, 3., .95, .988),
    (45., 5., .10, .095, 2.6, .89, .973),
    (20., 20., .08, .077, 2.8, .91, .975),
])
def test_factorized_21_nonspin_amplitude_against_reference(m1, m2, w, w0, j, he, hr):
    expected = factorized_21_nonspin_amplitude(m1, m2, w, w0, j, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(0.), chi2=scalar(0.))[(2, 1)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (30., 20., .2, -.1, .04, .039, 3., .95, .988),
    (45., 5., -.8, .3, .10, .095, 2.6, .89, .973),
    (20., 20., -.9, .7, .08, .077, 2.8, .91, .975),
    (20., 20., .4, .4, .08, .077, 2.8, .91, .975),
])
def test_factorized_21_spin_amplitude_against_reference(m1, m2, s1, s2,
                                                         w, w0, j, he, hr):
    expected = factorized_21_spin_amplitude(m1, m2, s1, s2, w, w0, j, hr)
    actual = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").amplitudes(
        scalar([9., 0., -.01, j]), omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(2, 1)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


@pytest.mark.parametrize("m1,m2,s1,s2,w,w0,j,he,hr", [
    (40., 10., .7, -.3, .01, .0098, 4.7, .97, .99),
    (40., 10., .7, -.3, .04, .039, 3., .91, .98),
    (40., 10., 0., 0., .02, .019, 3.6, .95, .985),
    (20., 20., .7, -.3, .02, .019, 3.6, .95, .985),
])
def test_factorized_21_rho_hybrid_against_reference(m1, m2, s1, s2,
                                                     w, w0, j, he, hr):
    expected = factorized_21_rho_hybrid_amplitude(m1, m2, s1, s2, w, w0, j, hr)
    model = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8",
                                  mode_21_higher_spin="rho_hybrid")
    state = scalar([9., 0., -.01, j])
    actual = model.amplitudes(
        state, omega=scalar(w), omega_zero_radial=scalar(w0),
        heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
        x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
        chi1=scalar(s1), chi2=scalar(s2))[(2, 1)]
    assert float(actual) == pytest.approx(float(expected), rel=3e-12, abs=2e-14)


def test_factorized_21_rho_hybrid_batch_and_gradient():
    model = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                  mode_21_higher_spin="rho_hybrid")
    spins = scalar([.3, -.2]).requires_grad_()
    state = scalar([[9., 0., -.01, 3.], [9., 0., -.01, 3.]])
    args = dict(omega=scalar([.02, .03]), omega_zero_radial=scalar([.019, .029]),
                heff_over_mu=scalar([.95, .94]), hreal_over_M=scalar([.985, .98]),
                x1=scalar([.8, .8]), x2=scalar([.2, .2]),
                chi1=spins, chi2=scalar([-.1, -.1]))
    amplitudes = model.amplitudes(state, **args)[(2, 1)]
    flux = model.energy_flux(state, **args)
    assert amplitudes.shape == (2,) and flux.shape == (2,)
    gradient = torch.autograd.grad(flux.sum(), spins)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient != 0).all())

    with pytest.raises(ValueError, match="mode_21_higher_spin"):
        SEOBNRv4PublishedFlux(spin_truncation="tabulated", mode_21_higher_spin="unknown")


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


@pytest.mark.parametrize("m1,m2,s1,s2,omega", [
    (30., 20., 0., 0., .04),
    (30., 20., .2, -.1, .04),
    (45., 5., -.8, .3, .025),
])
def test_leading_adiabatic_initial_state_against_scalar_reference(m1, m2, s1, s2, omega):
    torque = -.0007
    class ConstantTorque:
        def radiation_reaction_forces(self, frequency, pr_star, pphi, *, r, hamiltonian):
            return torch.zeros_like(pphi), torch.full_like(pphi, torque)

    n = m1*m2/(m1+m2)**2
    ham = SEOBNRv4Hamiltonian(scalar(n), scalar(s1), scalar(s2),
                              delta=scalar((m1-m2)/(m1+m2)))
    actual = find_quasicircular_initial_conditions(ham, ConstantTorque(), scalar(omega))
    expected = leading_adiabatic_state(m1, m2, s1, s2, omega, torque)
    for value, reference in zip(actual, expected):
        assert float(value) == pytest.approx(float(reference), rel=2e-11, abs=2e-13)


def test_35_mode_flux_composition_against_reference():
    ham = SEOBNRv4Hamiltonian(scalar(.24), scalar(.3), scalar(-.2))
    state = scalar([9., 0., -.01, 3.])
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
    args = flux.from_hamiltonian(state, ham)
    modes = flux.amplitudes(state, **args)
    assert len(modes) == 35
    expected = emitted_flux(float(args["omega"]), {key: float(value) for key, value in modes.items()})
    assert float(flux.energy_flux(state, **args)) == pytest.approx(float(expected), rel=2e-13)


def test_35_mode_flux_from_independent_scalar_amplitudes():
    """Compose all declared paper mode oracles, including the weak modes."""
    m1, m2, s1, s2 = 30., 20., .2, -.1
    w, w0, j, he, hr = .06, .057, 3.2, .93, .981
    base = (m1, m2, s1, s2, w, w0)
    h_args, j_args = (*base, he, hr), (*base, j, hr)
    ref = {
        (2, 2): scalar_flux.factorized_22_amplitude(*h_args),
        (2, 1): scalar_flux.factorized_21_spin_amplitude(*j_args),
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
         scalar_flux.factorized_8_even_amplitudes(*h_args, spin_truncation="tabulated")),
        (((8, 7), (8, 5), (8, 3), (8, 1)), scalar_flux.factorized_8_odd_amplitudes(*j_args)),
    )
    for keys, values in grouped:
        ref.update(zip(keys, values))
    assert len(ref) == 35
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
    state = scalar([9., 0., -.01, j])
    args = dict(omega=scalar(w), omega_zero_radial=scalar(w0),
                heff_over_mu=scalar(he), hreal_over_M=scalar(hr),
                x1=scalar(m1/(m1+m2)), x2=scalar(m2/(m1+m2)),
                chi1=scalar(s1), chi2=scalar(s2))
    actual_modes = flux.amplitudes(state, **args)
    assert set(actual_modes) == set(ref)
    expected_flux = scalar_flux.emitted_flux(w, ref)
    assert float(flux.energy_flux(state, **args)) == pytest.approx(
        float(expected_flux), rel=3e-12, abs=2e-14)


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


def test_ringdown_against_reference():
    n, c = 0.24, 0.15
    times = scalar([0., .01, .5, 2., 10.])
    sigma = torch.complex(scalar(.55), scalar(-.08))
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
    actual_frequency, actual_damping = dimensionless_qnm(2, 2, 0, scalar(spin))
    assert float(actual_frequency) == pytest.approx(frequency, rel=0, abs=1e-9)
    assert float(actual_damping) == pytest.approx(damping, rel=0, abs=1e-9)


@pytest.mark.parametrize("spin,frequency,damping", COOK_AUTHOR_220_SAMPLES + (COOK_AUTHOR_220_NEAR_EXTREMAL,))
def test_qnm_against_cook_independent_author_data(spin, frequency, damping):
    # Cook KerrQNM_00.h5, n=0, (ell,m)=(2,2); the raw solver's full domain.
    actual_frequency, actual_damping = dimensionless_qnm(2, 2, 0, scalar(spin))
    assert float(actual_frequency) == pytest.approx(frequency, rel=0, abs=1e-10)
    assert float(actual_damping) == pytest.approx(damping, rel=0, abs=1e-10)
