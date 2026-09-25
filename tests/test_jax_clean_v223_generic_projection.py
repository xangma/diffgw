"""Generic source-frame v223 JAX composition against Torch clean and LAL."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.precession_msa_v223_generic_projection import (
    generic_msa_v223_waveform,
    generic_source_frame_geometry_v223,
)
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry


CASES = (
    (40.0, 20.0, (0.3, 0.2, 0.4), (-0.2, 0.1, -0.2), 0.4, 40.0,
     (25.0, 40.0, 80.0)),
    (48.0, 17.0, (-0.18, 0.27, 0.52), (0.15, -0.21, 0.09), 0.7, 30.0,
     (20.0, 30.0, 70.0)),
)


def _t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _relative_l2(actual, target):
    return np.linalg.norm(np.asarray(actual)-np.asarray(target))/np.linalg.norm(target)


def _torch_waveform(case, *, phic, convention, spins=None):
    m1, m2, s1, s2, inc, fr, frequencies = case
    if spins is not None:
        s1, s2 = spins[:3], spins[3:]
    return waveform_v223(
        _t(frequencies), m1, m2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
        spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
        distance=500.0, inclination=inc, phic=phic, f_ref=fr,
        convention=convention,
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("phic", (0.0, 0.3))
def test_generic_source_geometry_matches_torch_clean(case, phic):
    m1, m2, s1, s2, inc, fr, _ = case
    actual = generic_source_frame_geometry_v223(m1, m2, s1, s2, inc, phic, fr)
    expected = source_frame_geometry(
        m1, m2, *s1, *s2, inc, phic, fr, compatibility="v223",
    )
    for name, value in zip(("alpha0", "theta_jn", "zeta", "phi_j"), expected):
        np.testing.assert_allclose(np.asarray(actual[name]), value.detach().numpy(),
                                   rtol=0.0, atol=3e-14, err_msg=name)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", (0, 1))
def test_generic_waveform_matches_torch_clean(case, convention):
    m1, m2, s1, s2, inc, fr, frequencies = case
    hp, hc, state = generic_msa_v223_waveform(
        frequencies, m1, m2, jnp.asarray(s1), jnp.asarray(s2),
        distance=500.0, inclination=inc, phic=0.3, f_ref=fr,
        convention=convention, return_state=True,
    )
    thp, thc = _torch_waveform(case, phic=0.3, convention=convention)
    assert _relative_l2(hp, thp.detach().numpy()) < 5e-12
    assert _relative_l2(hc, thc.detach().numpy()) < 5e-12
    assert hp.shape == hc.shape == (3,)
    assert np.isfinite(np.asarray(state["final_spin"]))


@pytest.mark.parametrize("case", CASES)
def test_generic_waveform_spin_gradients_match_torch_clean(case):
    m1, m2, s1, s2, inc, fr, frequencies = case
    spins0 = np.array((*s1, *s2))

    def jax_loss(spins):
        hp, hc = generic_msa_v223_waveform(
            frequencies, m1, m2, spins[:3], spins[3:],
            distance=500.0, inclination=inc, phic=0.3, f_ref=fr,
            convention=1,
        )
        return jnp.real(hp[0]*1e22)+jnp.imag(hc[2]*1e22)

    spins = _t(spins0).requires_grad_()
    thp, thc = _torch_waveform(case, phic=_t(0.3), convention=1, spins=spins)
    torch_loss = torch.real(thp[0]*1e22)+torch.imag(thc[2]*1e22)
    expected = torch.autograd.grad(torch_loss, spins)[0].detach().numpy()
    actual = np.asarray(jax.grad(jax_loss)(jnp.asarray(spins0)))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_generic_waveform_jit_batch_and_label_exchange():
    frequency = jnp.asarray([25.0, 40.0, 80.0])
    s1 = jnp.asarray([CASES[0][2], CASES[1][2]])
    s2 = jnp.asarray([CASES[0][3], CASES[1][3]])
    evaluate = jax.jit(generic_msa_v223_waveform, static_argnames=("convention",))
    batch = evaluate(
        frequency, jnp.asarray([40.0, 48.0]), jnp.asarray([20.0, 17.0]), s1, s2,
        distance=500.0, inclination=jnp.asarray([0.4, 0.7]), phic=0.3,
        f_ref=jnp.asarray([40.0, 30.0]), convention=1,
    )
    assert batch[0].shape == batch[1].shape == (2, 3)
    for row, case in enumerate(CASES):
        m1, m2, chi1, chi2, inc, fr, _ = case
        single = evaluate(frequency, m1, m2, jnp.asarray(chi1), jnp.asarray(chi2),
                          distance=500.0, inclination=inc, phic=0.3,
                          f_ref=fr, convention=1)
        exchanged = evaluate(frequency, m2, m1, jnp.asarray(chi2), jnp.asarray(chi1),
                             distance=500.0, inclination=inc, phic=0.3,
                             f_ref=fr, convention=1)
        for output, one, swapped in zip(batch, single, exchanged):
            np.testing.assert_allclose(np.asarray(output[row]), np.asarray(one),
                                       rtol=3e-12, atol=0.0)
            np.testing.assert_allclose(np.asarray(swapped), np.asarray(one),
                                       rtol=3e-12, atol=0.0)


def _lal_v223(case, convention, *, phic=0.3):
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    m1, m2, s1, s2, inc, fr, _ = case
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 223)
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(options, 5)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(options, convention)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 3)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, *s1, *s2,
        500e6*lal.PC_SI, inc, phic, 0.0, 0.0, 0.0,
        0.25, 20.0, 512.0, fr, options, ls.IMRPhenomXP,
    )
    return np.asarray(hp.data.data[80:2048]), np.asarray(hc.data.data[80:2048])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", (0, 1))
def test_generic_dense_waveform_matches_public_v223_compatibility_oracle(case, convention):
    m1, m2, s1, s2, inc, fr, _ = case
    frequency = jnp.arange(20.0, 512.0, 0.25)
    hp, hc = generic_msa_v223_waveform(
        frequency, m1, m2, jnp.asarray(s1), jnp.asarray(s2),
        distance=500.0, inclination=inc, phic=0.3, f_ref=fr,
        convention=convention,
    )
    target_hp, target_hc = _lal_v223(case, convention)
    assert _relative_l2(hp, target_hp) < 1e-8
    assert _relative_l2(hc, target_hc) < 1e-8


@pytest.mark.parametrize("case", CASES)
def test_public_convention6_phase_diagnostic(case):
    """Public LAL convention 6 is a phase diagnostic, not JAX dispatch."""
    one = _lal_v223(case, 1)
    six = _lal_v223(case, 6)
    for source, target in zip(one, six):
        assert _relative_l2(np.exp(0.6j)*source, target) < 1e-12
