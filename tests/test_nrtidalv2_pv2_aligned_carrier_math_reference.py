"""Published Pv2 aligned BNS carrier, conditional on a supplied BBH carrier."""

import math

import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from diffgw_reference import nrtidalv2_amplitude, nrtidalv2_pv2_terms
from diffgw_reference.nrtidalv2_pv2_projection import (
    mode_from_face_on_strain, project_positive_22,
)
from test_nrtidalv2_paper_d_composition_math_reference import CASES
from test_nrtidalv2_phase_math_reference import require_module


FRACTIONS = (0.75, 0.99, 1.0, 1.025, 1.05, 1.10, 1.15, 1.25)
DISTANCE_MPC = 100.0
LIMIT = 1e-10
ANGLES = (0.47, -0.32, 0.79, 1.08, 0.16)


def _native(backend, case, frequencies):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    options = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                   distance=DISTANCE_MPC, inclination=0.0,
                   phic=0.0, tc=0.0, phasing_convention="coalescence")
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import (
            assemble_nrtidalv2_pv2_aligned_carrier,
        )
        from torchwave.approximants.phenompv2_waveform_clean import (
            face_on_to_positive_22,
        )
        from torchwave.approximants.precession_twisting_clean import (
            project_twisted_positive_modes,
        )
        f = torch.as_tensor(frequencies, dtype=torch.float64)
        bbh, _ = IMRPhenomD()(f=f, **options)
        result = assemble_nrtidalv2_pv2_aligned_carrier(
            f, bbh, m1, m2, chi1, chi2, lam1, lam2, DISTANCE_MPC)
        mode = face_on_to_positive_22(result)
        angles = [torch.full_like(f, value) for value in ANGLES]
        projected = project_twisted_positive_modes({(2, 2): mode}, *angles)
        return (np.asarray(bbh), np.asarray(result),
                tuple(np.asarray(value) for value in projected))

    jax = require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenomd import IMRPhenomD
    from jaxwave.approximants.imrphenomd_nrtidal import (
        assemble_nrtidalv2_pv2_aligned_carrier,
    )
    from jaxwave.approximants.phenompv2_waveform import (
        _project, face_on_to_positive_22,
    )
    f = jnp.asarray(frequencies, dtype=jnp.float64)
    bbh, _ = IMRPhenomD.evaluate(f=f, **options)
    result = assemble_nrtidalv2_pv2_aligned_carrier(
        f, bbh, m1, m2, chi1, chi2, lam1, lam2, DISTANCE_MPC)
    mode = face_on_to_positive_22(result)
    angles = [jnp.full_like(f, value) for value in ANGLES]
    projected = _project(mode, *angles)
    return (np.asarray(bbh), np.asarray(result),
            tuple(np.asarray(value) for value in projected))


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_pv2_bns_carrier_is_assembled_before_rotation(backend, case):
    _, m1, m2, chi1, chi2, lam1, lam2 = case
    m_sec = (m1 + m2) * MSUN_SEC
    d_sec = DISTANCE_MPC * MPC_SI / C_SI
    delta = (m1 - m2) / (m1 + m2)
    merger = float(nrtidalv2_amplitude.merger_frequency(
        m_sec, delta, lam1, lam2))
    frequencies = np.array((100.0, 800.0,
                            *(fraction * merger for fraction in FRACTIONS)))
    bbh, actual, projected = _native(backend, case, frequencies)
    assert np.all(np.abs(bbh) > 0)
    expected = np.asarray([
        nrtidalv2_pv2_terms.assembled_carrier_from_bbh(
            f, baseline, m1, m2, chi1, chi2, lam1, lam2,
            m_sec, d_sec, merger)
        for f, baseline in zip(frequencies, bbh)
    ])
    windows = np.asarray([
        float(nrtidalv2_amplitude.taper(f, merger)) for f in frequencies
    ])
    tidal_amplitudes = np.asarray([
        float(nrtidalv2_amplitude.amplitude(
            f, m_sec, d_sec, delta, lam1, lam2)) for f in frequencies
    ])
    scale = windows * (np.abs(bbh) + np.abs(tidal_amplitudes))
    live = scale > 0
    assert np.max(np.abs(actual[live] - expected[live]) / scale[live]) <= LIMIT
    assert actual[-1] == 0j

    expected_projected = tuple(np.asarray([
        project_positive_22(mode_from_face_on_strain(value), *ANGLES)[component]
        for value in expected
    ]) for component in (0, 1))
    for observed, target in zip(projected, expected_projected):
        assert np.max(np.abs(observed[live] - target[live]) / scale[live]) <= LIMIT
        assert observed[-1] == 0j

    # The correction changes the complete BBH carrier, including its phase;
    # the gate would fail if the native path returned the bare BBH input.
    assert np.max(np.abs(actual[:2] - bbh[:2]) / np.abs(bbh[:2])) > 1e-4


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_supplied_carrier_frequency_gradient_and_jax_jit(backend):
    _, m1, m2, chi1, chi2, lam1, lam2 = CASES[1]
    m_sec = (m1 + m2) * MSUN_SEC
    d_sec = DISTANCE_MPC * MPC_SI / C_SI
    merger = float(nrtidalv2_amplitude.merger_frequency(
        m_sec, (m1 - m2) / (m1 + m2), lam1, lam2))
    bbh = 1e-22 * (1 + 0.3j)
    scale = abs(bbh)

    def reference(f):
        return nrtidalv2_pv2_terms.assembled_carrier_from_bbh(
            f, bbh, m1, m2, chi1, chi2, lam1, lam2,
            m_sec, d_sec, merger).real / scale

    frequency = 300.0
    step = 0.01
    expected_grad = (-reference(frequency + 2*step)
                     + 8*reference(frequency + step)
                     - 8*reference(frequency - step)
                     + reference(frequency - 2*step)) / (12*step)
    grid = np.array([100.0, frequency, 800.0])
    expected_values = np.array([nrtidalv2_pv2_terms.assembled_carrier_from_bbh(
        f, bbh, m1, m2, chi1, chi2, lam1, lam2,
        m_sec, d_sec, merger) for f in grid])

    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import (
            assemble_nrtidalv2_pv2_aligned_carrier,
        )
        bbh_native = torch.tensor(bbh, dtype=torch.complex128)
        f = torch.tensor(frequency, dtype=torch.float64, requires_grad=True)
        actual = assemble_nrtidalv2_pv2_aligned_carrier(
            f, bbh_native, m1, m2, chi1, chi2, lam1, lam2, DISTANCE_MPC)
        actual_grad = float(torch.autograd.grad(actual.real / scale, f)[0])
        values = np.asarray(assemble_nrtidalv2_pv2_aligned_carrier(
            torch.as_tensor(grid, dtype=torch.float64), bbh_native,
            m1, m2, chi1, chi2, lam1, lam2, DISTANCE_MPC))
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import (
            assemble_nrtidalv2_pv2_aligned_carrier,
        )

        def carrier(f):
            return assemble_nrtidalv2_pv2_aligned_carrier(
                f, bbh, m1, m2, chi1, chi2, lam1, lam2, DISTANCE_MPC)

        actual_grad = float(jax.jit(jax.grad(
            lambda f: jnp.real(carrier(f)) / scale))(jnp.float64(frequency)))
        values = np.asarray(jax.jit(carrier)(jnp.asarray(grid)))

    assert np.max(np.abs(values - expected_values)) / scale <= LIMIT
    assert math.isclose(actual_grad, expected_grad, rel_tol=1e-6, abs_tol=1e-8)
