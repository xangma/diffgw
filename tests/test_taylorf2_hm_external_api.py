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
