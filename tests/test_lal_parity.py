"""Tests for numerical parity against LALSimulation (skipped if lalsuite is not installed)."""

import pytest
import torch
import numpy as np

from torchwave import get_fd_waveform


def _has_lal():
    try:
        import lal
        import lalsimulation
        from pycbc.filter.matchedfilter import match
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
def test_taylorf2_lal_parity():
    """Verify TaylorF2 matches LALSimulation to near machine precision."""
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import match
    from pycbc.types import FrequencySeries

    m1_msun = 1.4
    m2_msun = 1.4
    f_min = 30.0
    f_max = 500.0
    df = 0.25
    dist_mpc = 100.0

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_msun * lal.MSUN_SI,
        m2_msun * lal.MSUN_SI,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        dist_mpc * 1e6 * lal.PC_SI,
        0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_min,
        None,
        lalsim.TaylorF2,
    )

    lal_series = FrequencySeries(hp_lal.data.data, delta_f=hp_lal.deltaF)
    freqs = np.arange(len(lal_series)) * lal_series.delta_f
    mask = (freqs >= f_min) & (freqs <= f_max)

    hp_tw, _ = get_fd_waveform(
        "TaylorF2",
        mass1=m1_msun,
        mass2=m2_msun,
        sample_frequencies=torch.tensor(freqs[mask], dtype=torch.float64),
        distance=dist_mpc,
        dtype=torch.float64,
    )

    tw_data = np.zeros(len(lal_series), dtype=np.complex128)
    tw_data[mask] = hp_tw.numpy()
    tw_series = FrequencySeries(tw_data, delta_f=df)

    overlap, _ = match(lal_series, tw_series, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    mismatch = 1.0 - overlap

    assert overlap > 0.999999, f"Expected TaylorF2 overlap > 0.999999, got {overlap} (mismatch {mismatch})"


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
def test_taylorf2_tidal_lal_parity():
    """Verify TaylorF2 with tidal deformability matches LALSimulation (overlap > 0.99999)."""
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import match
    from pycbc.types import FrequencySeries

    m1_msun = 1.4
    m2_msun = 1.4
    f_min = 30.0
    f_max = 1000.0
    df = 0.25
    dist_mpc = 100.0

    extra = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, 400.0)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, 400.0)

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_msun * lal.MSUN_SI,
        m2_msun * lal.MSUN_SI,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        dist_mpc * 1e6 * lal.PC_SI,
        0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_min,
        extra,
        lalsim.TaylorF2,
    )

    lal_series = FrequencySeries(hp_lal.data.data, delta_f=hp_lal.deltaF)
    freqs = np.arange(len(lal_series)) * lal_series.delta_f
    mask = (freqs >= f_min) & (freqs <= f_max)

    hp_tw, _ = get_fd_waveform(
        "TaylorF2",
        mass1=m1_msun,
        mass2=m2_msun,
        lambda1=400.0,
        lambda2=400.0,
        sample_frequencies=torch.tensor(freqs[mask], dtype=torch.float64),
        distance=dist_mpc,
        dtype=torch.float64,
    )

    tw_data = np.zeros(len(lal_series), dtype=np.complex128)
    tw_data[mask] = hp_tw.numpy()
    tw_series = FrequencySeries(tw_data, delta_f=df)

    overlap, _ = match(lal_series, tw_series, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    mismatch = 1.0 - overlap

    assert overlap > 0.99999, f"Expected TaylorF2 tidal overlap > 0.99999, got {overlap} (mismatch {mismatch})"


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
@pytest.mark.parametrize(
    "m1,m2,chi1,chi2,min_overlap",
    [
        (10.0, 10.0, 0.0, 0.0, 0.999),
        (30.0, 30.0, 0.0, 0.0, 0.998),
        (30.0, 30.0, 0.8, 0.8, 0.995),
        (36.0, 29.0, 0.3, -0.2, 0.998),
        (30.0, 10.0, 0.5, -0.3, 0.995),
        (20.0, 10.0, 0.4, 0.0, 0.997),
        (50.0, 10.0, 0.3, 0.2, 0.998),
        (40.0, 10.0, 0.6, -0.4, 0.995),
        (45.0, 5.0, 0.5, 0.0, 0.995),
        (80.0, 10.0, 0.7, 0.1, 0.995),
    ],
)
def test_imrphenomd_lal_parity(m1, m2, chi1, chi2, min_overlap):
    """Verify IMRPhenomD matches LALSimulation across symmetric, asymmetric, and spinning regimes."""
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import match
    from pycbc.types import FrequencySeries

    f_min = 20.0
    f_max = 1024.0
    df = 0.25
    dist_mpc = 410.0

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0, 0.0, chi1,
        0.0, 0.0, chi2,
        dist_mpc * 1e6 * lal.PC_SI,
        0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_min,
        None,
        lalsim.IMRPhenomD,
    )

    lal_series = FrequencySeries(hp_lal.data.data, delta_f=hp_lal.deltaF)
    freqs = np.arange(len(lal_series)) * lal_series.delta_f
    mask = (freqs >= f_min) & (freqs <= f_max)

    hp_tw, _ = get_fd_waveform(
        "IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        sample_frequencies=torch.tensor(freqs[mask], dtype=torch.float64),
        distance=dist_mpc,
        dtype=torch.float64,
    )

    tw_data = np.zeros(len(lal_series), dtype=np.complex128)
    tw_data[mask] = hp_tw.numpy()
    tw_series = FrequencySeries(tw_data, delta_f=df)

    overlap, _ = match(lal_series, tw_series, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    mismatch = 1.0 - overlap

    assert overlap >= min_overlap, (
        f"For m=({m1},{m2}), chi=({chi1},{chi2}): expected overlap >= {min_overlap}, got {overlap} (mismatch {mismatch})"
    )


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
@pytest.mark.parametrize(
    "name,m1,m2,c1,c2,l1,l2,f_min,f_max,min_match",
    [
        ("NSBH high spin", 5.0, 1.4, 0.9, -0.3, None, None, 30.0, 1000.0, 0.99999),
        ("High mass ratio q=10", 10.0, 1.0, 0.2, -0.1, None, None, 30.0, 500.0, 0.99999),
        ("BNS low spin tidal", 1.4, 1.4, 0.01, -0.01, 400.0, 400.0, 30.0, 500.0, 0.99999),
    ],
)
def test_taylorf2_high_spin_nsbh_lal_parity(name, m1, m2, c1, c2, l1, l2, f_min, f_max, min_match):
    """Verify TaylorF2 NSBH high-spin, high-mass-ratio, and BNS low-spin match > 0.99999 against LAL."""
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import match
    from pycbc.types import FrequencySeries

    df = 0.25
    extra = lal.CreateDict()
    if l1 is not None:
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, l1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, l2)

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0.0, 0.0, c1, 0.0, 0.0, c2,
        100.0 * 1e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_min, extra, lalsim.TaylorF2,
    )
    s_lal = FrequencySeries(hp_lal.data.data, delta_f=df)
    freqs = torch.arange(len(s_lal), dtype=torch.float64) * df

    hp_tw, _ = get_fd_waveform(
        "TaylorF2",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        lambda1=l1,
        lambda2=l2,
        sample_frequencies=freqs,
        distance=100.0,
        f_isco_cutoff=False,
    )
    s_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
    overlap, _ = match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap >= min_match, f"{name}: overlap = {overlap:.8f} below expected {min_match}"


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
@pytest.mark.parametrize(
    "name,tw_app,lal_model_str,m1,m2,s1x,s1y,s1z,s2x,s2y,s2z,inc,min_match",
    [
        ("Pv2 m=(20,15) inc=0.2", "IMRPhenomP", "IMRPhenomPv2", 20.0, 15.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.2, 0.995),
        ("Pv2 m=(15,12) inc=0.3", "IMRPhenomP", "IMRPhenomPv2", 15.0, 12.0, 0.2, 0.1, -0.1, 0.1, 0.0, 0.1, 0.3, 0.995),
        ("XP m=(20,15) inc=0.2", "IMRPhenomXP", "IMRPhenomXP", 20.0, 15.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.2, 0.995),
        ("XP m=(15,12) inc=0.3", "IMRPhenomXP", "IMRPhenomXP", 15.0, 12.0, 0.2, 0.1, -0.1, 0.1, 0.0, 0.1, 0.3, 0.995),
    ],
)
def test_precessing_phenomp_xp_lal_parity(name, tw_app, lal_model_str, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match):
    """Verify IMRPhenomP/Pv2 and IMRPhenomXP achieve match >= 0.995 against LAL precessing models."""
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import match
    from pycbc.types import FrequencySeries

    df = 0.25
    f_min, f_max, f_ref = 20.0, 300.0, 20.0
    lal_approx = lalsim.GetApproximantFromString(lal_model_str)

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,
        100.0 * 1e6 * lal.PC_SI, inc, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_ref, lal.CreateDict(), lal_approx,
    )
    s_lal = FrequencySeries(hp_lal.data.data, delta_f=df)
    freqs = torch.arange(len(s_lal), dtype=torch.float64) * df

    hp_tw, _ = get_fd_waveform(
        tw_app,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        distance=100.0,
        inclination=inc,
        sample_frequencies=freqs,
        f_ref=f_ref,
    )
    s_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
    overlap, _ = match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap >= min_match, f"{name}: overlap = {overlap:.8f} below expected {min_match}"


