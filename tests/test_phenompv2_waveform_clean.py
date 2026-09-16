import pytest
import torch
from torchwave.approximants.phenompv2_waveform_clean import (
    PHASE_SPLINE_CUTOFF_MF,
    _L2,
    _natural_cubic_derivative,
    _phase_sample_bounds,
    _phase_time_shift,
    waveform,
)
from torchwave.waveforms import get_fd_waveform


def test_clean_pv2_shapes_and_finite():
    f = torch.linspace(20, 256, 64, dtype=torch.float64)
    hp, hc = waveform(f, 35, 25, .2, .1, .2, -.1, .15, -.1,
                      inclination=.7, phic=.3, f_ref=20)
    assert hp.shape == hc.shape == f.shape
    assert torch.isfinite(hp.real).all() and torch.isfinite(hc.imag).all()


def test_clean_pv2_broadcasts_parameter_axes():
    f = torch.linspace(20, 100, 12)
    hp, hc = waveform(f, torch.tensor([30., 40.]), 20.,
                      spin1x=torch.tensor([.1, .2]), inclination=torch.tensor([.2, .8]))
    assert hp.shape == hc.shape == (2, 12)


def test_clean_pv2_relabeling_is_invariant():
    f = torch.linspace(20, 100, 16, dtype=torch.float64)
    a = waveform(f, 35, 25, .2, .1, .2, -.1, .15, -.1, inclination=.7, f_ref=20)
    b = waveform(f, 25, 35, -.1, .15, -.1, .2, .1, .2, inclination=.7, f_ref=20)
    torch.testing.assert_close(a[0], b[0], rtol=2e-12, atol=1e-29)
    torch.testing.assert_close(a[1], b[1], rtol=2e-12, atol=1e-29)


def test_clean_pv2_rejects_nonpositive_grid():
    try:
        waveform(torch.tensor([0., -1.]), 30., 20.)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_clean_pv2_orbital_angular_momentum_evolves_beta():
    f = torch.tensor([20., 80., 320.], dtype=torch.float64)
    l = _L2(torch.tensor(35.), torch.tensor(25.), f)
    # L decreases monotonically with frequency in the PN regime, hence the
    # published atan2 cone angle changes when in-plane spin is nonzero.
    assert l[1] < l[0] and not torch.allclose(l[0], l[-1])


def test_clean_pv2_frequency_gradient_is_finite():
    f = torch.tensor([20., 80., 160.], dtype=torch.float64, requires_grad=True)
    hp, hc = waveform(f, 35., 25., .2, .1, .2, -.1, .15, -.1,
                      inclination=.7, phic=.3, f_ref=20.)
    loss = (hp.real.square() + hc.imag.square()).sum()
    loss.backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()


def test_clean_pv2_mass_gradients_are_finite():
    m1 = torch.tensor(35., dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25., dtype=torch.float64, requires_grad=True)
    f = torch.tensor([20., 80., 160.], dtype=torch.float64)
    hp, hc = waveform(f, m1, m2, .2, .1, .2, -.1, .15, -.1,
                      inclination=.7, phic=.3, f_ref=20.)
    (hp.real.square() + hc.imag.square()).sum().backward()
    assert m1.grad is not None and m2.grad is not None
    assert torch.isfinite(m1.grad) and torch.isfinite(m2.grad)


def test_clean_pv2_mass_spin_gradients_match_scaled_finite_difference():
    f = torch.tensor([80.], dtype=torch.float64)
    m1 = torch.tensor(35., dtype=torch.float64, requires_grad=True)
    sx = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    def value(a, x):
        hp, hc = waveform(f, a, 25., x, .1, .2, -.1, .15, -.1,
                         inclination=.7, phic=.3, f_ref=20.)
        return hp[0], hc[0]
    hp, hc = value(m1, sx)
    grads = {}
    for name, y in (('hp', hp), ('hc', hc)):
        grads[name, 'real', 'm1'] = torch.autograd.grad(y.real, m1, retain_graph=True)[0]
        grads[name, 'real', 'spin1x'] = torch.autograd.grad(y.real, sx, retain_graph=True)[0]
        grads[name, 'imag', 'm1'] = torch.autograd.grad(y.imag, m1, retain_graph=True)[0]
        grads[name, 'imag', 'spin1x'] = torch.autograd.grad(y.imag, sx, retain_graph=True)[0]
    # Scale by each component magnitude so this checks derivatives rather
    # than absolute tiny Fourier-domain amplitudes.
    for eps, tol in ((1.e-3, 3.e-3), (1.e-4, 5.e-4), (1.e-5, 5.e-4)):
        plus_m = value(35.+eps, .2); minus_m = value(35.-eps, .2)
        plus_s = value(35., .2+eps); minus_s = value(35., .2-eps)
        for name in ('hp', 'hc'):
            fm = (plus_m[0 if name == 'hp' else 1]-minus_m[0 if name == 'hp' else 1])/(2*eps)
            fs = (plus_s[0 if name == 'hp' else 1]-minus_s[0 if name == 'hp' else 1])/(2*eps)
            for part, finite_m, finite_s in (('real', fm.real, fs.real), ('imag', fm.imag, fs.imag)):
                for parameter, autod, finite in (('m1', grads[name, part, 'm1'], finite_m),
                                                 ('spin1x', grads[name, part, 'spin1x'], finite_s)):
                    scale = max(float(abs(finite.detach())), 1.e-30)
                    assert abs(float((autod-finite).detach()))/scale < tol, f'{name} {part} {parameter}'


def test_clean_pv2_no_grad_control():
    f = torch.tensor([20., 80.], dtype=torch.float64)
    m1 = torch.tensor(35., dtype=torch.float64, requires_grad=True)
    enabled = waveform(f, m1, 25., .2, .1, .2, -.1, .15, -.1,
                        inclination=.7, phic=.3, f_ref=20.)
    assert enabled[0].requires_grad and enabled[1].requires_grad
    with torch.no_grad():
        hp, hc = waveform(f,
                           m1, 25., .2, .1, .2, -.1, .15, -.1,
                           inclination=.7, phic=.3, f_ref=20.)
    torch.testing.assert_close(hp * 1e22, enabled[0] * 1e22, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(hc * 1e22, enabled[1] * 1e22, rtol=1e-11, atol=1e-11)
    assert not hp.requires_grad and not hc.requires_grad
    assert torch.isfinite(hp.real).all() and torch.isfinite(hc.imag).all()


def test_natural_cubic_derivative_matches_scipy_when_available():
    scipy = pytest.importorskip("scipy.interpolate")
    import numpy as np

    nodes = torch.linspace(0.8, 1.2, 10, dtype=torch.float64)
    values = torch.sin(2.3 * nodes) + 0.07 * nodes.square()
    query = torch.tensor([0.8, 0.913, 1.0, 1.199, 1.2], dtype=torch.float64)
    got = _natural_cubic_derivative(nodes, values, query).detach().numpy()
    spline = scipy.CubicSpline(nodes.numpy(), values.numpy(), bc_type="natural")
    expected = np.asarray(spline(query.numpy(), 1))
    np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_phase_time_shift_uses_actual_ten_point_spline():
    # A fake carrier makes the interpolation convention observable without
    # relying on an external waveform implementation.
    frd = torch.tensor(1.0, dtype=torch.float64)
    total_mass_seconds = torch.tensor(0.1, dtype=torch.float64)

    def d_call(frequencies, carrier_phase=0.0):
        phase = torch.sin(1.7 * frequencies) + 0.03 * frequencies.square()
        return torch.zeros_like(phase, dtype=torch.complex128), torch.zeros_like(phase, dtype=torch.complex128), phase

    got = _phase_time_shift(torch.tensor(0.0, dtype=torch.float64), frd, total_mass_seconds, d_call)
    lo, hi = _phase_sample_bounds(frd, total_mass_seconds)
    nodes = torch.linspace(float(lo), float(hi), 10, dtype=torch.float64)
    scipy = pytest.importorskip("scipy.interpolate")
    expected = scipy.CubicSpline(
        nodes.numpy(), (torch.sin(1.7 * nodes) + 0.03 * nodes.square()).numpy(), bc_type="natural"
    )(1.0, 1)
    assert abs(float(got) - float(expected)) < 2.0e-13


def test_phase_sample_bounds_exposes_cutoff_branch_and_units():
    frd = torch.tensor(1.0, dtype=torch.float64)
    low, high = _phase_sample_bounds(frd, torch.tensor(0.1, dtype=torch.float64))
    assert float(low) == 0.8
    assert float(high) == 1.2
    # Here .2/Msec is below 1.2*fRD but above .8*fRD, so the physical cutoff
    # is the selected upper endpoint.
    clipped_mass = torch.tensor(PHASE_SPLINE_CUTOFF_MF / 0.9, dtype=torch.float64)
    low, high = _phase_sample_bounds(frd, clipped_mass)
    assert float(low) == 0.8
    assert abs(float(high) - 0.9) < 1.0e-14


def test_arbitrary_sequence_retains_requested_endpoint_and_matches_uniform_index():
    uniform = torch.arange(20.0, 512.0001, 0.25, dtype=torch.float64)
    indices = torch.tensor([0, 1, uniform.numel() - 2, uniform.numel() - 1])
    sequence = uniform[indices]
    kwargs = dict(
        mass1=35.0,
        mass2=25.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.2,
        spin2x=-0.1,
        spin2y=0.15,
        spin2z=-0.1,
        inclination=0.7,
        phic=0.3,
        f_ref=20.0,
        distance=500.0,
    )
    uniform_hp, uniform_hc = waveform(uniform, **kwargs)
    sequence_hp, sequence_hc = waveform(sequence, **kwargs)
    torch.testing.assert_close(sequence_hp, uniform_hp[indices], rtol=2.0e-12, atol=1.0e-35)
    torch.testing.assert_close(sequence_hc, uniform_hc[indices], rtol=2.0e-12, atol=1.0e-35)
    assert abs(sequence_hp[-1]) > 0.0 and abs(sequence_hc[-1]) > 0.0


def _routed_kwargs():
    return dict(
        mass1=35.0,
        mass2=25.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.2,
        spin2x=-0.1,
        spin2y=0.15,
        spin2z=-0.1,
        distance=500.0,
        inclination=0.7,
        phic=0.3,
        tc=0.0,
        f_ref=20.0,
    )


def test_routed_pv2_matches_clean_driver_and_preserves_sequence_endpoint():
    f = torch.tensor([20.0, 31.0, 50.0], dtype=torch.float64)
    direct = waveform(f, **_routed_kwargs())
    routed = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=50.0,
        **_routed_kwargs()
    )
    torch.testing.assert_close(routed[0], direct[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(routed[1], direct[1], rtol=0.0, atol=0.0)
    assert abs(routed[0][-1]) > 0.0 and abs(routed[1][-1]) > 0.0

    # A supplied sequence remains authoritative even when its last requested
    # frequency lies above the metadata f_final value.
    above = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=49.0,
        **_routed_kwargs()
    )
    explicit_sequence = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=49.0,
        frequency_support="sequence", **_routed_kwargs()
    )
    torch.testing.assert_close(above[0], direct[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(above[1], direct[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit_sequence[0], direct[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit_sequence[1], direct[1], rtol=0.0, atol=0.0)


def test_routed_pv2_uniform_contract_has_exclusive_endpoint():
    f = torch.arange(20.0, 50.0001, 10.0, dtype=torch.float64)
    seq = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=50.0,
        **_routed_kwargs()
    )
    uniform = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=50.0,
        frequency_support="uniform", **_routed_kwargs()
    )
    assert abs(seq[0][-1]) > 0.0
    assert uniform[0][-1] == 0.0 and uniform[1][-1] == 0.0
    torch.testing.assert_close(uniform[0][:-1], seq[0][:-1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(uniform[1][:-1], seq[1][:-1], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="equally spaced"):
        get_fd_waveform(
            "IMRPhenomPv2", sample_frequencies=torch.tensor([20.0, 31.0, 50.0]),
            f_lower=20.0, f_final=50.0, frequency_support="uniform",
            **_routed_kwargs()
        )
    with pytest.raises(ValueError, match="cover f_final"):
        get_fd_waveform(
            "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=51.0,
            frequency_support="uniform", **_routed_kwargs()
        )
    with pytest.raises(ValueError, match="requires f_final"):
        get_fd_waveform(
            "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0,
            frequency_support="uniform", **_routed_kwargs()
        )


def test_routed_pv2_uniform_contract_accepts_padded_non_power_of_two_edge():
    # Public FD adapters can pad a grid beyond the requested support edge.
    # 47.5 Hz is deliberately non-power-of-two and the 64 Hz terminal sample
    # represents that padding; all samples at and beyond the edge are zeroed.
    f = torch.arange(20.0, 64.0001, 0.25, dtype=torch.float64)
    routed = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=47.5,
        frequency_support="uniform", **_routed_kwargs()
    )
    clean = waveform(f, **_routed_kwargs())
    interior = f < 47.5
    torch.testing.assert_close(routed[0][interior], clean[0][interior], rtol=0.0, atol=0.0)
    torch.testing.assert_close(routed[1][interior], clean[1][interior], rtol=0.0, atol=0.0)
    assert torch.all(routed[0][~interior] == 0) and torch.all(routed[1][~interior] == 0)


def test_parity_harness_uses_oracle_fhigh_for_padded_pv2_grid():
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    pytest.importorskip("pycbc")
    import copy
    from tests import parity_harness as ph

    case = next(
        c for c in ph.get_canonical_catalog_cases()
        if c["torch_name"] == "IMRPhenomPv2"
    )
    case = copy.deepcopy(case)
    case["grid"]["fhigh"] = 47.5
    case["oracle_fhigh"] = 63.5
    result = ph.run_oracle_comparison(case, strict=True)
    assert result["status"] == "PASSED", result
    assert result["oracle_fhigh"] > result["grid"]["fhigh"]
    assert result["plus"]["out_of_band_zero"]
    assert result["cross"]["out_of_band_zero"]
    assert all(metric["out_of_band_zero"] for metric in result["projections"].values())


def test_routed_pv2_public_batch_dtype_phasing_and_gradients():
    f = torch.tensor([20.0, 40.0, 80.0], dtype=torch.float32)
    mass1 = torch.tensor([35.0, 42.0], dtype=torch.float32, requires_grad=True)
    tc = torch.tensor(0.002, dtype=torch.float32, requires_grad=True)
    hp, hc = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=80.0,
        mass1=mass1, mass2=25.0, spin1x=torch.tensor([0.2, 0.1]),
        spin1y=0.1, spin1z=torch.tensor([0.2, -0.2]), spin2x=-0.1,
        spin2y=0.15, spin2z=-0.1, distance=500.0, inclination=0.7,
        phic=0.3, tc=tc, f_ref=20.0,
        phasing_convention="reference_frequency",
        dtype=torch.float32,
    )
    assert hp.shape == hc.shape == (2, 3)
    assert hp.dtype is torch.complex64 and hc.dtype is torch.complex64
    loss = 1.0e30 * (hp.real + 0.7 * hc.imag).sum()
    loss.backward()
    assert mass1.grad is not None and torch.isfinite(mass1.grad).all()
    assert tc.grad is not None and torch.isfinite(tc.grad)

    alias_hp, alias_hc = get_fd_waveform(
        "IMRPhenomPv2", sample_frequencies=f, f_lower=20.0, f_final=80.0,
        mass1=35.0, mass2=25.0, spin1x=0.2, spin1y=0.1, spin1z=0.2,
        spin2x=-0.1, spin2y=0.15, spin2z=-0.1, distance=500.0,
        inclination=0.7, phic=0.3, tc=0.002, f_ref=20.0,
        reference_frame="reference_frequency",
        dtype=torch.float32,
    )
    torch.testing.assert_close(alias_hp, hp.detach()[0], rtol=2e-6, atol=0.0)
    torch.testing.assert_close(alias_hc, hc.detach()[0], rtol=2e-6, atol=0.0)


def test_routed_legacy_imrphenomp_remains_available():
    f = torch.tensor([20.0, 40.0], dtype=torch.float64)
    hp, hc = get_fd_waveform(
        "IMRPhenomP", sample_frequencies=f, f_ref=20.0,
        mass1=35.0, mass2=25.0, spin1x=0.2, spin1y=0.1, spin1z=0.2,
        spin2x=-0.1, spin2y=0.15, spin2z=-0.1, distance=500.0,
        inclination=0.7, phic=0.3, tc=0.0,
    )
    assert hp.shape == hc.shape == f.shape
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
