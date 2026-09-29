"""Conditional NSBH complex composition against independent scalar sources."""

import numpy as np
import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomd_waveform as phenomd
from diffgw_reference import nsbh_phase
from diffgw_reference.nsbh_polarization_composition import compose


CASES = ((7., 1.4, 0., 0., 600.), (12.6, 1.4, .4, 0., 0.))
MF = (.010, .018, .035, .070)
CENTERS_MF = (.025, .035, .045)
WIDTH_MF = .04
CARRIERS = (.8, .3, .5)
SCALE = .73
INCLINATIONS = (0., .7, np.pi / 2, np.pi)
LIMIT = 2e-13


def _conditioned_spec(case):
    """QNM and PhenomD rows are supplied, not qualified by this check."""
    import torch

    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    from torchwave.approximants.imrphenom_nsbh import zappa_remnant_properties
    from torchwave.approximants.nsbh_phase_baseline import NSBHPhaseBaseline

    m1, m2, chi1, chi2, lam = (torch.tensor(x, dtype=torch.float64) for x in case)
    _, chi_final = zappa_remnant_properties(m1, m2, chi1, chi2, lam)
    _, _, f_rd, f_damp = NSBHPhaseBaseline(chi_final).remnant_properties(
        m1, m2, chi1, chi2)
    names = set(phenomd.amp.FIT_NAMES) | set(phenomd.phase.MODEL_FIT_NAMES)
    rows = {name: tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name])
            for name in names}
    return phenomd.compose(*case[:4], 400., f_rd.item(), f_damp.item(),
                           coefficient_rows=rows)


def _native(backend, frequencies, mass_sec, centers, phases, inclination):
    if backend == "torch":
        import torch

        from torchwave.approximants.imrphenom_nsbh import (
            nsbh_amplitude_window_sum, nsbh_polarizations_from_amplitude,
        )

        f = torch.tensor(frequencies, dtype=torch.float64)
        amp = nsbh_amplitude_window_sum(
            f, torch.tensor(mass_sec, dtype=torch.float64),
            tuple(torch.tensor(x, dtype=torch.float64) for x in centers),
            torch.tensor(WIDTH_MF, dtype=torch.float64),
            tuple(torch.full_like(f, x) for x in CARRIERS),
        )
        hp, hc = nsbh_polarizations_from_amplitude(
            SCALE * amp, torch.tensor(phases, dtype=torch.float64),
            torch.tensor(inclination, dtype=torch.float64),
        )
        return hp.detach().numpy(), hc.detach().numpy()

    import jax
    import jax.numpy as jnp

    from jaxwave.approximants.imrphenom_nsbh import (
        nsbh_amplitude_window_sum, nsbh_polarizations_from_amplitude,
    )

    with jax.default_device(jax.devices("cpu")[0]):
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        amp = nsbh_amplitude_window_sum(
            f, jnp.asarray(mass_sec, dtype=jnp.float64),
            tuple(jnp.asarray(x, dtype=jnp.float64) for x in centers),
            jnp.asarray(WIDTH_MF, dtype=jnp.float64),
            tuple(jnp.full_like(f, x) for x in CARRIERS),
        )
        hp, hc = nsbh_polarizations_from_amplitude(
            SCALE * amp, jnp.asarray(phases, dtype=jnp.float64),
            jnp.asarray(inclination, dtype=jnp.float64),
        )
        return np.asarray(hp), np.asarray(hc)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", ("coalescence", "reference_frequency"))
def test_conditioned_complex_polarizations(backend, case, convention, reference_device):
    if reference_device != "cpu":
        pytest.fail("this gate is frozen for CPU FP64")
    spec = _conditioned_spec(case)
    mass_sec = float(spec.m_sec)
    frequencies = np.asarray(MF) / mass_sec
    centers = np.asarray(CENTERS_MF) / mass_sec
    f_ref = .020 / mass_sec
    phases = [float(nsbh_phase.evaluate(
        spec, f, case[0], case[1], case[4], tc=.003, phic=.23,
        convention=convention,
        f_ref_hz=f_ref if convention == "reference_frequency" else None,
        dps=70,
    )) for f in frequencies]

    maximum_error = 0.
    for inclination in INCLINATIONS:
        expected = [compose(
            f, mass_sec, centers, WIDTH_MF, CARRIERS, phase,
            SCALE, inclination,
        ) for f, phase in zip(frequencies, phases, strict=True)]
        hp, hc = _native(backend, frequencies, mass_sec, centers, phases, inclination)
        plus_error = float(np.max(np.abs(hp - np.asarray([complex(x.plus) for x in expected]))))
        cross_error = float(np.max(np.abs(hc - np.asarray([complex(x.cross) for x in expected]))))
        maximum_error = max(maximum_error, plus_error, cross_error)
        assert maximum_error <= LIMIT, (backend, case, convention, inclination,
                                        plus_error, cross_error)
    print(f"{backend} {case} {convention}: max_complex_error={maximum_error:.3e}")


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_public_polarization_ratio(backend, inclination):
    """The public path must call the checked projection with its actual inputs."""
    frequencies = np.asarray((50., 100., 250., 500., 900.))
    kwargs = dict(mass1=7., mass2=1.4, chi1=0., chi2=0., lambda2=600.,
                  distance=400., inclination=inclination)
    if backend == "torch":
        import torch

        from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH

        hp, hc = IMRPhenomNSBH()(torch.tensor(frequencies, dtype=torch.float64), **kwargs)
        hp, hc = hp.detach().numpy(), hc.detach().numpy()
    else:
        import jax
        import jax.numpy as jnp

        from jaxwave.approximants.imrphenom_nsbh import IMRPhenomNSBH

        with jax.default_device(jax.devices("cpu")[0]):
            hp, hc = IMRPhenomNSBH.evaluate(jnp.asarray(frequencies, dtype=jnp.float64),
                                            **kwargs)
            hp, hc = np.asarray(hp), np.asarray(hc)
    cosine = np.cos(inclination)
    expected = -2j * cosine / (1 + cosine * cosine)
    assert np.all(np.abs(hp) > 0)
    assert np.max(np.abs(hc / hp - expected)) <= 1e-12
