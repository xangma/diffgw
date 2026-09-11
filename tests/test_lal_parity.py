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
    from pycbc.filter.matchedfilter import optimized_match
    from pycbc.types import FrequencySeries

    df = 0.25
    f_min, f_max, f_ref = 20.0, 300.0, 20.0
    lal_approx = lalsim.GetApproximantFromString(lal_model_str)

    d_lal = lal.CreateDict()
    tw_kwargs = {}
    if "xp" in tw_app.lower():
        # Pin NNLO comparator explicitly for PhenomX precession parity check
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d_lal, 102)
        tw_kwargs["precession_method"] = "nnlo"
        tw_kwargs["prec_version"] = 102

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,
        100.0 * 1e6 * lal.PC_SI, inc, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_ref, d_lal, lal_approx,
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
        **tw_kwargs,
    )
    s_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
    overlap, _ = optimized_match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap >= min_match, f"{name}: overlap = {overlap:.8f} below expected {min_match}"


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation or PyCBC not installed in environment")
@pytest.mark.parametrize(
    "label,tw_name,lal_name,kwargs",
    [
        ("TF2 point particle", "TaylorF2", "TaylorF2", dict(mass1=1.4, mass2=1.4, chi1=0.0, chi2=0.0, f_ref=30.0, flow=30.0, fhigh=500.0, spin_order=7, tidal_order=0)),
        ("TF2 6PN tides", "TaylorF2", "TaylorF2", dict(mass1=1.4, mass2=1.4, chi1=0.0, chi2=0.0, lambda1=400.0, lambda2=400.0, f_ref=30.0, flow=30.0, fhigh=1000.0, spin_order=7, tidal_order=12)),
        ("TF2 6.5PN tides", "TaylorF2", "TaylorF2", dict(mass1=1.4, mass2=1.4, chi1=0.0, chi2=0.0, lambda1=400.0, lambda2=400.0, f_ref=30.0, flow=30.0, fhigh=1000.0, spin_order=7, tidal_order=13)),
        ("TF2 unequal spinning", "TaylorF2", "TaylorF2", dict(mass1=1.7, mass2=1.2, chi1=0.15, chi2=-0.08, f_ref=50.0, flow=30.0, fhigh=600.0, spin_order=7, tidal_order=0)),
        ("Ecc explicit references", "TaylorF2_Eccentric", "TaylorF2Ecc", dict(mass1=1.4, mass2=1.4, chi1=0.0, chi2=0.0, e0=0.05, f0_ecc=30.0, f_ref=50.0, flow=30.0, fhigh=500.0, spin_order=7, ecc_order=6)),
        ("Ecc unequal spinning", "TaylorF2_Eccentric", "TaylorF2Ecc", dict(mass1=1.7, mass2=1.2, chi1=0.15, chi2=-0.08, e0=0.03, f0_ecc=30.0, f_ref=50.0, flow=30.0, fhigh=500.0, spin_order=7, ecc_order=6)),
    ],
)
def test_explicit_option_oracle_cases(label, tw_name, lal_name, kwargs):
    """Verify complex relative L2 <= 1e-11 against LALSimulation for matched physical options."""
    import lal
    import lalsimulation as lalsim

    p = dict(kwargs)
    flow = float(p.pop("flow"))
    fhigh = float(p.pop("fhigh"))
    df = 0.25
    fref = float(p.get("f_ref", flow))
    m1 = float(p["mass1"])
    m2 = float(p["mass2"])
    s1 = float(p.get("chi1", 0.0))
    s2 = float(p.get("chi2", 0.0))
    e0 = float(p.get("e0", 0.0))

    extra = lal.CreateDict()
    spin_order = p.get("spin_order", 7)
    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(extra, spin_order)
    if "lambda1" in p:
        tidal_order = p.get("tidal_order", 12)
        lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(extra, tidal_order)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, float(p["lambda1"]))
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, float(p["lambda2"]))
    if e0 > 0.0:
        lalsim.SimInspiralWaveformParamsInsertEccentricityFreq(extra, float(p.get("f0_ecc", flow)))
        lalsim.SimInspiralWaveformParamsInsertPNEccentricityOrder(extra, p.get("ecc_order", 6))

    lal_approx = lalsim.GetApproximantFromString(lal_name)
    hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, s1, 0.0, 0.0, s2,
        100.0 * 1e6 * lal.PC_SI, 0.0, 0.0, 0.0, e0, 0.0,
        df, flow, fhigh, fref, extra, lal_approx,
    )

    n_pts = len(hp_lal.data.data)
    freqs = torch.arange(n_pts, dtype=torch.float64) * df

    hp_tw, hc_tw = get_fd_waveform(
        tw_name,
        sample_frequencies=freqs,
        distance=100.0,
        dtype=torch.float64,
        **p,
    )

    sel = (freqs >= flow) & (freqs < fhigh)
    h_ref_p = torch.tensor(hp_lal.data.data, dtype=torch.complex128)[sel]
    h_ref_c = torch.tensor(hc_lal.data.data, dtype=torch.complex128)[sel]

    rel_l2_p = float(torch.linalg.norm(hp_tw[sel] - h_ref_p) / torch.linalg.norm(h_ref_p))
    rel_l2_c = float(torch.linalg.norm(hc_tw[sel] - h_ref_c) / torch.linalg.norm(h_ref_c))

    assert rel_l2_p <= 1e-11, f"{label} plus relative L2 {rel_l2_p:.3e} exceeds 1e-11"
    assert rel_l2_c <= 1e-11, f"{label} cross relative L2 {rel_l2_c:.3e} exceeds 1e-11"
    assert hp_tw[0] == 0.0 and hc_tw[0] == 0.0, f"{label} DC bin must be zero"


def test_antenna_factors_geometry():
    """Verify distinct detector antenna responses and rejection of invalid detectors."""
    from tests.parity_harness import compute_antenna_factors
    ra, dec, psi = 1.2, 0.4, 0.3
    h1 = compute_antenna_factors(ra, dec, psi, detector="H1")
    l1 = compute_antenna_factors(ra, dec, psi, detector="L1")
    v1 = compute_antenna_factors(ra, dec, psi, detector="V1")

    # Responses must be distinct
    assert h1 != l1, "H1 and L1 responses must not be identical"
    assert h1 != v1, "H1 and V1 responses must not be identical"
    assert l1 != v1, "L1 and V1 responses must not be identical"

    # Invalid detector must raise ValueError
    with pytest.raises(ValueError, match="not recognized"):
        compute_antenna_factors(ra, dec, psi, detector="not-a-detector")


@pytest.mark.skipif(not _has_lal(), reason="LALSimulation / PyCBC required for negative control tests")
def test_parity_harness_negative_controls():
    """Verify strict verification harness rejects corruptions (DC, PSD, projection, missing oracle)."""
    import copy
    import builtins
    from unittest.mock import patch
    from tests import parity_harness

    case = copy.deepcopy(parity_harness.DETERMINISTIC_HELD_OUT_CASES[0])

    # 1. Baseline passes strict gates
    baseline = parity_harness.run_oracle_comparison(case, strict=True)
    assert baseline["status"] == "PASSED", f"Baseline failed: {baseline}"

    # 2. Corrupt DC bin -> must fail
    orig_wf = parity_harness.get_fd_waveform
    def corrupt_dc(*args, **kwargs):
        hp, hc = orig_wf(*args, **kwargs)
        hp, hc = hp.clone(), hc.clone()
        hp[0], hc[0] = 1.0 + 1.0j, 1.0 - 1.0j
        return hp, hc

    with patch.object(parity_harness, "get_fd_waveform", corrupt_dc):
        res = parity_harness.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Non-zero DC failed to trigger verification failure"

    # 3. Corrupt PSD mismatch -> must fail
    orig_metrics = parity_harness.evaluate_waveform_metrics
    def corrupt_psd(*args, **kwargs):
        metrics = orig_metrics(*args, **kwargs)
        metrics["psd_mismatch"] = 0.5
        return metrics

    with patch.object(parity_harness, "evaluate_waveform_metrics", corrupt_psd):
        res = parity_harness.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Corrupted PSD mismatch failed to trigger verification failure"

    # 4. Corrupt projection strain -> must fail
    orig_proj = parity_harness.project_strain
    proj_calls = [0]
    def corrupt_proj(*args, **kwargs):
        proj_calls[0] += 1
        s = orig_proj(*args, **kwargs)
        return s * 2.0 if proj_calls[0] == 1 else s

    with patch.object(parity_harness, "project_strain", corrupt_proj):
        res = parity_harness.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Corrupted detector projection failed to trigger verification failure"

    # 5. Missing PyCBC in strict mode -> must fail
    orig_import = builtins.__import__
    def without_pycbc(name, *args, **kwargs):
        if name == "pycbc" or name.startswith("pycbc."):
            raise ImportError("simulated absent pycbc")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", without_pycbc):
        res = parity_harness.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Missing PyCBC in strict mode failed to trigger verification failure"



