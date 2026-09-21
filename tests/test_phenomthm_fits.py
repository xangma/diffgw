"""Deterministic checks for the shared IMRPhenomT calibration fits.

Expected values were generated from MIT-licensed phenomxpy/phenomt/fits.py
at commit 4121fafa0484f3db2a7352f4eea697faaaf29f65 (Cecilio Garcia Quiros,
2023--25). The pinned source is an update oracle and is not imported at test
time.
"""
from __future__ import annotations

import numpy as np
import pytest
from diffgw_core import phenomt_fits as fits

_FREQ_CASE = (0.23, 0.17, -0.61, np.sqrt(1 - 4 * 0.23))
_FREQ_EXPECTED = {
    21: (0.11421508326783598, 0.23770279294337263, 0.18049172318626458, 3.4400414284178087),
    22: (0.19372006028251604, 0.36326084273676423, 0.1615631295578558, 1.9758715479241662),
    33: (0.3239214482114925, 0.5660244551214806, 0.1608858960768311, 1.8915481432813328),
    44: (0.4252436995038217, 0.7227134593969075, 0.13880123615273204, 2.053363280533313),
    55: (0.5338814407771567, 0.9165700373454179, 0.1417299186901563, 2.0150840218168447),
}
_AMP_CASE = (0.11, -0.42, 1.05, np.sqrt(1 - 4 * 0.11))
_AMP_EXPECTED = {
    21: ((0.021157954642726913, 0.05100319531292715, -0.008299399714814693), (0.0049313881148377444, 0.009046953316201043, 0.01059234274694524)),
    22: ((0.12846127433772306, 0.15950525443725702, -0.3241165340152553), (0.054741097072051756, 0.07968435253278183, 0.08781678084142519)),
    33: ((0.031032501145898197, 0.04734649594548226, -0.18557913543441668), (0.00875893111708246, 0.015213340890835921, 0.017522999398980395)),
    44: ((0.011411976870921608, 0.020204173766385152, -0.12991795560759625), (0.002104052888012396, 0.004438235421942508, 0.005367746182052959)),
    55: ((0.004495033081371099, 0.009595634326886083, -0.047864627779901206), (0.0005470941460682107, 0.001380963146129552, 0.0017442716138999972)),
}
_QNM_EXPECTED = {
    20: (0.06283593129066349, 0.013411604119117753, 0.0410437541730534),
    21: (0.07278492736473564, 0.01301340133526961, 0.039472284910601826),
    22: (0.08548816282066439, 0.012796928813834112, 0.03869803415010866),
    33: (0.13533630094406976, 0.013115624261802801, 0.03949619401222492),
    44: (0.18328297249278125, 0.013332781187457506, 0.04009404869394438),
    55: (0.23027085939635378, 0.013465972462727576, 0.040448951407232495),
}
_SHIFT_CASE = (0.19, -0.36, 0.83)
_SHIFT_EXPECTED = {20: 0.0, 21: 6.874113294168357, 22: 0.0, 33: 3.4706003835617945, 44: 1.7469865959560404, 55: 3.170969655894681}
_REMNANT_EXPECTED = {
    (0.23, 0.17, -0.61): (0.9592565636359295, 0.6470512253489278),
    (0.11, -0.42, 0.83): (0.9898828042110625, 0.05774832573189734),
    (0.25, 0.2, -0.1): (0.950172271766202, 0.7015071693512464),
}


@pytest.mark.parametrize("mode", [21, 22, 33, 44, 55])
def test_frequency_fit_branches_match_pinned_mit_fixture(mode):
    args = _FREQ_CASE + (mode,)
    names = ("_IMRPhenomT_Intermediate_Freq_CP1", "_IMRPhenomT_PeakFrequency", "_IMRPhenomT_RD_Freq_C2", "_IMRPhenomT_RD_Freq_C3")
    for i, name in enumerate(names):
        np.testing.assert_allclose(getattr(fits, name)(*args), _FREQ_EXPECTED[mode][i], rtol=0, atol=2e-15)


@pytest.mark.parametrize("mode", [21, 22, 33, 44, 55])
def test_amplitude_fit_branches_match_pinned_mit_fixture(mode):
    eta, spin, dchi, delta = _AMP_CASE
    main, inspiral = _AMP_EXPECTED[mode]
    for name, expected in zip(("_IMRPhenomT_Intermediate_Amp_CP1", "_IMRPhenomT_PeakAmp", "_IMRPhenomT_Ringdown_Amp_C3"), main):
        actual = (getattr(fits, name)(eta, spin, dchi, delta, mode) if name != "_IMRPhenomT_Ringdown_Amp_C3" else getattr(fits, name)(eta, spin, dchi, mode))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    for idx, expected in enumerate(inspiral, 1):
        np.testing.assert_allclose(fits._IMRPhenomT_Inspiral_Amp_CP(eta, spin, dchi, delta, mode, idx), expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize("name,index", [("IMRPhenomT_fringfit", 0), ("IMRPhenomT_fdampfit", 1), ("IMRPhenomT_fdampn2fit", 2)])
@pytest.mark.parametrize("mode", [20, 21, 22, 33, 44, 55])
def test_qnm_fit_branches_match_pinned_mit_fixture(name, index, mode):
    np.testing.assert_allclose(getattr(fits, name)(0.71, mode), _QNM_EXPECTED[mode][index], rtol=0, atol=2e-15)


@pytest.mark.parametrize("mode", [20, 21, 22, 33, 44, 55])
def test_peak_time_shift_matches_pinned_mit_fixture(mode):
    np.testing.assert_allclose(fits._IMRPhenomT_tshift(*_SHIFT_CASE, mode), _SHIFT_EXPECTED[mode], rtol=0, atol=2e-15)


def test_aligned_remnant_fits_match_pinned_mit_fixture():
    for (eta, s1, s2), expected in _REMNANT_EXPECTED.items():
        delta = np.sqrt(1 - 4 * eta)
        actual = (fits.IMRPhenomX_FinalMass2017(eta, s1, s2, delta), fits.IMRPhenomX_FinalSpin2017(eta, s1, s2, delta))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_torch_and_jax_array_algebra_matches_scalar():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    eta, spin, dchi, delta = _FREQ_CASE
    expected = fits._IMRPhenomT_PeakFrequency(eta, spin, dchi, delta, 33)
    torch_value = fits._IMRPhenomT_PeakFrequency(*(torch.tensor(x, dtype=torch.float64) for x in (eta, spin, dchi)), torch.tensor(delta, dtype=torch.float64), 33)
    jax_value = fits._IMRPhenomT_PeakFrequency(*(jnp.asarray(x, dtype=jnp.float64) for x in (eta, spin, dchi)), jnp.asarray(delta, dtype=jnp.float64), 33)
    np.testing.assert_allclose(torch_value.detach().numpy(), expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jax_value), expected, rtol=1e-13, atol=1e-14)


def test_equal_mass_remnant_gradients_are_finite_when_delta_is_supplied():
    torch = pytest.importorskip("torch")
    eta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor(0.2, dtype=torch.float64)
    s2 = torch.tensor(-0.1, dtype=torch.float64)
    delta = torch.tensor(0.0, dtype=torch.float64)
    value = fits.IMRPhenomX_FinalMass2017(eta, s1, s2, delta) + fits.IMRPhenomX_FinalSpin2017(eta, s1, s2, delta)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(eta.grad)
