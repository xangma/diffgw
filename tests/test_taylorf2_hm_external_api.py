"""Black-box checks for the public TaylorF2 external-reference boundary.

These tests inspect only the installed public ``lal``/``lalsimulation`` API.
They deliberately do not read implementation sources.  A TaylorF2 frequency
series is a single carrier output; a mode array does not turn it into a
mode-resolved reference, and the public frequency-domain-mode dispatcher does
not provide a TaylorF2 generator in the installed API.
"""

import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

from diffgw_reference import taylorf2_hm as ref


def _taylorf2_series(mode_string=None):
    params = lal.CreateDict()
    if mode_string is not None:
        lalsim.SimInspiralWaveformParamsInsertModeArrayFromModeString(
            params, mode_string
        )
    series = lalsim.SimInspiralTaylorF2(
        0.2,
        0.25,
        1.4 * lal.MSUN_SI,
        1.4 * lal.MSUN_SI,
        0.0,
        0.0,
        30.0,
        500.0,
        30.0,
        100.0 * lal.PC_SI * 1.0e6,
        params,
    )
    return np.asarray(series.data.data)


def test_public_taylorf2_mode_array_does_not_provide_mode_resolved_output():
    """Mode-array metadata leaves the scalar TaylorF2 carrier unchanged."""
    carrier = _taylorf2_series()
    for mode_string in ("[(2,2)]", "[(3,2)]", "[(2,2),(3,2)]", "[(2,1),(3,3),(4,4)]"):
        np.testing.assert_array_equal(_taylorf2_series(mode_string), carrier)


def test_public_fd_modes_dispatcher_rejects_taylorf2():
    """The public FD-mode dispatcher has no TaylorF2 mode generator."""
    params = lal.CreateDict()
    approximant = lalsim.GetApproximantFromString("TaylorF2")
    with pytest.raises((RuntimeError, ValueError), match="(?i)(invalid|generator|argument)"):
        lalsim.SimInspiralChooseFDModes(
            1.4 * lal.MSUN_SI,
            1.4 * lal.MSUN_SI,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            30.0,
            500.0,
            30.0,
            0.0,
            100.0 * lal.PC_SI * 1.0e6,
            0.0,
            params,
            approximant,
        )


@pytest.mark.parametrize("inclination", (0.2, 0.91, 1.4))
@pytest.mark.parametrize("phi0", (-0.37, 0.0, 0.51))
def test_public_lal_harmonics_match_all_eight_mode_pairs(inclination, phi0):
    """LAL independently checks the angular basis, not a TaylorF2_HM waveform."""
    values = {ref.IOTA: inclination, ref.PHI0: phi0}
    for l, m in ref.MODES:
        for signed_m in (-m, m):
            expected = complex(
                ref.evaluate(ref.spin_weighted_harmonic(l, signed_m), values)
            )
            actual = lal.SpinWeightedSphericalHarmonic(
                inclination, phi0, -2, l, signed_m
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=5e-16)
