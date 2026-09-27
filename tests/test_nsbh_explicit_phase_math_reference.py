"""Conditional NSBH explicit-gauge phase against a paper-composed scalar source."""

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomd_waveform as phenomd
from diffgw_reference import nsbh_phase as ref

CASES = ((7., 1.4, 0., 0., 600.), (12.6, 1.4, .4, 0., 0.))
US = (.010, .018, .035, .070)


def _conditioned_spec(case):
    """Supply QNM/fit data explicitly; neither is certified by this gate."""
    torch = pytest.importorskip("torch")
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    from torchwave.approximants.imrphenom_nsbh import zappa_remnant_properties
    from torchwave.approximants.nsbh_phase_baseline import NSBHPhaseBaseline

    m1, m2, chi1, chi2, lam = (torch.tensor(x, dtype=torch.float64) for x in case)
    _, chi_final = zappa_remnant_properties(m1, m2, chi1, chi2, lam)
    _, _, f_rd, f_damp = NSBHPhaseBaseline(chi_final).remnant_properties(
        m1, m2, chi1, chi2)
    rows = {name: tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name])
            for name in set(phenomd.amp.FIT_NAMES) | set(phenomd.phase.MODEL_FIT_NAMES)}
    return phenomd.compose(*case[:4], 400., f_rd.item(), f_damp.item(),
                           coefficient_rows=rows)


@pytest.mark.parametrize("case", CASES)
def test_scalar_precision(case):
    from diffgw_core.constants import MSUN_SEC

    spec = _conditioned_spec(case)
    f = .018 / ((case[0] + case[1]) * MSUN_SEC)
    args = (spec, f, case[0], case[1], case[4])
    p70 = ref.evaluate(*args, dps=70)
    p90 = ref.evaluate(*args, dps=90)
    assert abs(p70 - p90) <= sp.Float("1e-30")


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("convention", ("coalescence", "reference_frequency"))
def test_native_unit_phase(backend, case, convention, reference_device):
    from diffgw_core.constants import MSUN_SEC

    if reference_device != "cpu":
        pytest.fail("this gate is frozen for CPU FP64")
    m1, m2, chi1, chi2, lam = case
    m_sec = (m1 + m2) * MSUN_SEC
    frequencies = np.asarray(US) / m_sec
    f_ref = .020 / m_sec
    spec = _conditioned_spec(case)
    expected = np.array([
        -np.exp(-1j * float(ref.evaluate(
            spec, f, m1, m2, lam, tc=.003, phic=.23,
            convention=convention,
            f_ref_hz=f_ref if convention == "reference_frequency" else None,
            dps=70)))
        for f in frequencies
    ])

    kwargs = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                  lambda2=lam, distance=400., inclination=.7,
                  phic=.23, tc=.003, f_ref=f_ref,
                  phasing_convention=convention)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
        hp, _ = IMRPhenomNSBH()(torch.tensor(frequencies, dtype=torch.float64),
                               **kwargs)
        hp = hp.detach().cpu().numpy()
    else:
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            from jaxwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
            with jax.default_device(jax.devices("cpu")[0]):
                hp, _ = IMRPhenomNSBH.evaluate(
                    jnp.asarray(frequencies, dtype=jnp.float64), **kwargs)
                hp = np.asarray(hp)
        finally:
            jax.config.update("jax_enable_x64", previous)
    assert np.all(np.isfinite(hp))
    assert np.all(np.abs(hp) > 0)
    assert np.max(np.abs(hp / np.abs(hp) - expected)) <= 3e-7
