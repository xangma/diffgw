"""Optional binary-oracle checks; LALSuite is never a runtime dependency."""

import importlib

import numpy as np
import pytest


CASES = (
    (40.0, 15.0, (0.2, 0.1, 0.3), (0.0, 0.1, -0.2), 0.8, 0.3),
    (30.0, 25.0, (0.0, 0.0, 0.2), (0.0, 0.0, -0.1), 0.5, 0.2),
    (60.0, 6.0, (0.2, 0.1, 0.8), (0.1, 0.0, 0.1), 1.2, 0.1),
)


@pytest.mark.parametrize("name", ["IMRPhenomTHM", "IMRPhenomTP", "IMRPhenomTPHM"])
@pytest.mark.parametrize("case", CASES)
def test_lal_binary_reference(name, case):
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    import torch

    m1, m2, s1, s2, inclination, phase = case
    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 10210)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 0)
    spins = (s1[2], s2[2]) if name == "IMRPhenomTHM" else (*s1, *s2)
    # The binary oracle anchors its NNLO ringdown two samples before tc.
    # Resolve that offset for the rapidly precessing q=10 case; the native
    # physical-time anchor is separately tested for grid independence.
    rate = 32768 if name != "IMRPhenomTHM" and m1 / m2 >= 10 else 2048
    dt = 1 / rate
    hp, hc = getattr(lalsim, "Sim" + name)(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *spins,
        100e6 * lal.PC_SI, inclination, dt, 20.0, 20.0, phase, options,
    )
    times = torch.from_numpy(float(hp.epoch) + dt * np.arange(hp.data.length))
    kwargs = dict(chi1=s1[2], chi2=s2[2], distance=100.0,
                  inclination=inclination, phic=phase, f_ref=20.0)
    if name != "IMRPhenomTHM":
        kwargs.update(spin1x=s1[0], spin1y=s1[1], spin2x=s2[0], spin2y=s2[1],
                      precession_method="nnlo")
    plus, cross = getattr(importlib.import_module("torchwave"), name)().time_domain(
        times, m1, m2, **kwargs,
    )
    actual = (plus - 1j * cross).detach().numpy()
    reference = hp.data.data - 1j * hc.data.data
    relative_l2 = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    overlap = abs(np.vdot(actual, reference)) / (np.linalg.norm(actual) * np.linalg.norm(reference))
    # Direct complex strain: no shift, scaling, or phase alignment. NNLO uses
    # the published orbital angular momentum; LAL 102 omits its pi^2 term.
    tolerance = 1e-3 if name == "IMRPhenomTHM" else 1e-2
    assert relative_l2 < tolerance
    assert 1 - overlap < tolerance**2


@pytest.mark.parametrize("name", ["IMRPhenomTP", "IMRPhenomTPHM"])
@pytest.mark.parametrize("method,version,spin_mode,l2_limit,mismatch_limit", [
    ("msa", 22310, 3, 0.01, 1e-4),
    ("numerical", 300, 4, 0.03, 5e-4),
])
@pytest.mark.parametrize("case,fref", [
    (CASES[0], 20.0),
    ((35.0, 20.0, (0.3, 0.0, 0.4), (0.0, 0.0, 0.0), 1.1, 0.2), 30.0),
    ((30.0, 30.0, (0.2, -0.1, 0.4), (-0.12, 0.08, -0.3), 0.8, 0.3), 20.0),
])
def test_additional_precession_binary_reference(name, method, version, spin_mode,
                                               l2_limit, mismatch_limit, case, fref):
    """Measured clean-room residual bounds, not a claim of exact LAL parity."""
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    import torch

    # The extra holdouts exercise numerical precession. MSA has its separate
    # supported-domain checks; its original binary comparison is retained.
    if method == "msa" and case != CASES[0]:
        pytest.skip("additional holdouts cover numerical precession")
    m1, m2, s1, s2, inclination, phase = case
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, version)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, spin_mode)
    dt = 1/2048
    hp, hc = getattr(ls, "Sim" + name)(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, *s1, *s2, 100e6*lal.PC_SI,
        inclination, dt, 20.0, fref, phase, options,
    )
    times = torch.from_numpy(float(hp.epoch) + dt*np.arange(hp.data.length))
    a, b = getattr(importlib.import_module("torchwave"), name)().time_domain(
        times, m1, m2, chi1=s1[2], chi2=s2[2],
        spin1x=s1[0], spin1y=s1[1], spin2x=s2[0], spin2y=s2[1],
        distance=100.0, inclination=inclination, phic=phase, f_ref=fref,
        precession_method=method, precession_version=version,
    )
    actual, reference = (a-1j*b).detach().numpy(), hp.data.data-1j*hc.data.data
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual-reference)/np.linalg.norm(reference) < l2_limit
    overlap = abs(np.vdot(actual, reference))/(np.linalg.norm(actual)*np.linalg.norm(reference))
    assert 1-overlap < mismatch_limit


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("only22", [1, 0])
def test_project_modes_rotation_against_public_lal(backend, only22):
    """Check only the native mode-rotation convention against public LAL.

    The binary oracle supplies both complete coprecessing modes and its
    already-rotated J-frame modes.  This deliberately says nothing about the
    native spin evolution, carrier, or waveform accuracy.
    """
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    pytest.importorskip(backend)
    from scripts.diagnose_phenomt_rotation import measure

    result = measure(backend, "baseline", 2048, only22, samples=257)
    assert max(view["relative_l2"] for view in result["views"]) < 5e-12
    assert result["inverse_l0"]["status"] in ("measured", "skipped")
    if only22:
        assert max(view["relative_l2"] for view in result["inverse_l0"]["views"]) < 5e-12
