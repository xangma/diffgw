"""Independent published-equation checks for the bounded HM/XHM corrections.

Sources and explicit limits are in docs/reviews/2026-09-12-clean-room/higher-modes.md.
These tests do not use a waveform oracle.
"""

import math

import pytest
import torch

import torchwave.approximants.imrphenom_hm as hm_module
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenom_xhm import (
    IMRPhenomXHM,
    _pn_amplitude_ratio,
    _published_parameter_amplitude_veto,
)
from torchwave.approximants.xhm_fits import XHMParameterFits, _nonnegative_sqrt
from torchwave.constants import MSUN_SEC


def _matched_ringdown(mf, eta, alpha2, alpha3, alpha4, alpha5, rd22, ratio, width):
    """Full London Eq9, solving its two affine coefficients from a C1 join.

    Synthetic join phase/slope are sufficient because they cancel when the
    two independently constructed phases are subtracted.
    """
    join = (0.5 * rd22).detach().requires_grad_(True)

    def nonaffine(x):
        return (-alpha2 / x + (4.0 / 3.0) * alpha3 * x**0.75
                + alpha4 * ratio * torch.atan((x - alpha5 * rd22) / width)) / eta

    value = nonaffine(join)
    slope, = torch.autograd.grad(value, join, create_graph=True)
    c1 = -0.7 - slope
    c0 = 1.3 - value - c1 * join
    return c0 + c1 * mf + nonaffine(mf)


def _independent_eq9_change(mf, eta, a4, a5, rd22, damp22, rdlm, damplm):
    a2, a3 = eta.new_tensor(0.04), eta.new_tensor(-0.3)
    original = _matched_ringdown(mf, eta, a2, a3, a4, a5, rd22, 1.0, damp22)
    ratio = rd22 / rdlm
    modified = _matched_ringdown(mf, eta, a2, a3, a4, a5, rd22, ratio, ratio * damplm)
    return torch.where(mf > 0.5 * rd22, modified - original, torch.zeros_like(mf))


def test_hm_eq9_full_ansatz_and_parameter_gradients():
    """The correction equals independently C1-matched full Eq9 minus D."""
    values = ([0.03, 0.045, 0.07, 0.11], 0.21, 0.18, 0.98,
              0.085, 0.012, 0.14, 0.014)
    args = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in values)
    actual = hm_module._d_ringdown_phase_correction(*args)
    expected = _independent_eq9_change(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=2e-14)
    assert torch.autograd.gradcheck(hm_module._d_ringdown_phase_correction, args,
                                    eps=1e-6, atol=1e-5, rtol=1e-4)
    # Eq9 reduces exactly to the original carrier for its own QNM.
    same = hm_module._d_ringdown_phase_correction(*args[:6], *args[4:6])
    torch.testing.assert_close(same, torch.zeros_like(same), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("f_ref", [None, 350.0])
def test_hm_eq9_reaches_all_modes_and_phase_boundary_offsets(monkeypatch, f_ref):
    """Eq3/Eq7 map the corrected phase, including reference anchors in MR."""
    model = IMRPhenomHM()
    m1, m2, s1, s2 = [torch.tensor(v, dtype=torch.float64) for v in (35.0, 20.0, 0.4, -0.2)]
    msec = (m1 + m2) * MSUN_SEC
    mf = torch.tensor([0.01, 0.03, 0.06, 0.095, 0.12, 0.15], dtype=torch.float64)
    f = mf / msec
    params = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                  f_ref=f_ref, return_unprojected_modes=True)
    _, _, corrected = model._legacy_forward(f, **params)
    monkeypatch.setattr(hm_module, "_d_ringdown_phase_correction", lambda mf, *args: torch.zeros_like(mf))
    _, _, unchanged = model._legacy_forward(f, **params)
    eta = m1 * m2 / (m1 + m2)**2
    chi_pn = (s1 + s2) / 2.0 * (1.0 - 76.0 / 113.0 * eta) + (m1 - m2) / (m1 + m2) * (s1 - s2) / 2.0
    a4 = model.base_22.eval_table_v("alpha4", eta, chi_pn)
    a5 = model.base_22.eval_table_v("alpha5", eta, chi_pn)
    rd22, damp22 = model.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
    torch.testing.assert_close(corrected[(2, 2)], unchanged[(2, 2)], rtol=0.0, atol=0.0)
    for l, m in model.SUPPORTED_MODES[1:]:
        rdlm, damplm = model.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
        f0 = 0.018 * rdlm / rd22
        s_mid = (rd22 - 2.0 * f0 / m) / (rdlm - f0)
        k1, k2, k3 = m / 2.0, 1.0 / s_mid, rdlm / rd22
        mapped = torch.where(mf <= f0, 2.0 * mf / m,
                             torch.where(mf <= rdlm, 2.0 * f0 / m + s_mid * (mf - f0), mf / k3))

        def change(x):
            result = _independent_eq9_change(x, eta, a4, a5, rd22, damp22, rdlm, damplm)
            if f_ref is not None:
                result -= _independent_eq9_change(msec * f_ref, eta, a4, a5, rd22, damp22, rdlm, damplm)
            return result

        offset2 = (k1 - k2) * change(2.0 * f0 / m)
        offset3 = offset2 + (k2 - k3) * change(rd22)
        phase_change = torch.where(mf <= f0, k1 * change(mapped),
                                   torch.where(mf <= rdlm, k2 * change(mapped) + offset2,
                                               k3 * change(mapped) + offset3))
        expected = unchanged[(l, m)] * torch.exp(-1j * phase_change)
        scale = expected.abs().max()
        torch.testing.assert_close(corrected[(l, m)] / scale, expected / scale,
                                   rtol=2e-12, atol=2e-12)


def test_hm_eq9_phase_and_slope_continuity_and_batched_backward():
    """C1 holds at both map joins and the mapped D merger-ringdown join."""
    model = IMRPhenomHM()
    params = [torch.tensor(v, dtype=torch.float64) for v in (35.0, 20.0, 0.4, -0.2)]
    msec = (params[0] + params[1]) * MSUN_SEC
    rd22, _ = model.mode_qnm_frequencies(2, 2, *params)
    for l, m in model.SUPPORTED_MODES[1:]:
        rdlm, _ = model.mode_qnm_frequencies(l, m, *params)
        f0 = 0.018 * rdlm / rd22
        slope = (rd22 - 2.0 * f0 / m) / (rdlm - f0)
        carrier_join = f0 + (0.5 * rd22 - 2.0 * f0 / m) / slope
        for boundary in (f0, carrier_join, rdlm):
            f = (boundary / msec + torch.tensor([-1e-6, 0.0, 1e-6], dtype=torch.float64)).requires_grad_(True)
            _, _, modes = model(f, mass1=params[0], mass2=params[1], chi1=params[2], chi2=params[3],
                                 f_ref=350.0, modes=((l, m),), return_unprojected_modes=True)
            h = modes[(l, m)] * 1e22
            real_derivative, = torch.autograd.grad(h.real.sum(), f, retain_graph=True)
            imag_derivative, = torch.autograd.grad(h.imag.sum(), f)
            phase_derivative = (h.imag * real_derivative - h.real * imag_derivative) / h.abs().square()
            torch.testing.assert_close(phase_derivative[[0, 2]], phase_derivative[1].expand(2),
                                       rtol=2e-5, atol=1e-7)
            assert torch.angle(h[2] * h[0].conj()).abs() < 1e-5

    parameters = torch.tensor([[35.0, 20.0, 0.4, -0.2], [30.0, 30.0, 0.6, -0.6]],
                              dtype=torch.float64, requires_grad=True)
    m1, m2, s1, s2 = parameters.unbind(-1)
    f = torch.linspace(30.0, 900.0, 40, dtype=torch.float64)
    _, _, modes = model(f, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                       f_ref=350.0, return_unprojected_modes=True)
    gradient, = torch.autograd.grad(sum(h.real.sum() for h in modes.values()) * 1e22, parameters)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("lm", [(2, 1), (3, 3), (3, 2), (4, 4)])
def test_xhm_leading_spa_amplitudes(lm):
    """E8 and E10-E12/E14: retain the same-frequency SPA envelope ratio."""
    l, m = lm
    eta = torch.tensor(0.2, dtype=torch.float64)
    delta = torch.sqrt(1.0 - 4.0 * eta)
    zero = torch.zeros_like(eta)
    v = torch.tensor(3e-4, dtype=torch.float64)
    f = m * v**3 / (2.0 * math.pi)
    leading = {
        (2, 1): math.sqrt(2.0) / 3.0 * delta * v,
        (3, 3): 3.0 / 4.0 * math.sqrt(5.0 / 7.0) * delta * v,
        (3, 2): math.sqrt(5.0 / 7.0) / 3.0 * (1.0 - 3.0 * eta) * v**2,
        (4, 4): 4.0 / 9.0 * math.sqrt(10.0 / 7.0) * (1.0 - 3.0 * eta) * v**2,
    }[lm]
    expected = (m / 2.0) ** (7.0 / 6.0) * leading
    actual = _pn_amplitude_ratio(l, m, f, eta, delta, zero, zero)
    # Neglected nonspinning terms enter at relative O(v^2).
    torch.testing.assert_close(actual, expected, rtol=5e-7, atol=0.0)


class _FlatCarrier(torch.nn.Module):
    def forward(self, f, return_phase=False, **kwargs):
        hp = torch.ones_like(f, dtype=torch.complex128)
        if return_phase:
            return hp, -1j * hp, torch.zeros_like(f)
        return hp, -1j * hp


def test_hm_runtime_transitions_and_spa_rescaling(monkeypatch):
    """Check runtime carrier joins and the alternative beta stated after Eq4.

    London et al.'s higher-mode mapping paper states f_0^A=0.018*ratio and
    f_0^phi=0.014*ratio.  IMRPhenomHM currently uses the underlying PhenomD
    carrier joins, f_0^A=0.014*ratio and f_0^phi=0.018*ratio; the source
    distinction is audited in pass-30-phenomhm-transition-audit.md.
    """
    seen = {}
    for kind in ("amplitude", "phase"):
        original = getattr(hm_module, f"frequency_mapping_{kind}")

        def capture(f, f0, rd_lm, rd_22, m, *, kind=kind, original=original):
            seen[kind] = (f0, rd_lm, rd_22)
            result = original(f, f0, rd_lm, rd_22, m)
            seen[f"mapped_{kind}"] = result
            return result

        monkeypatch.setattr(hm_module, f"frequency_mapping_{kind}", capture)

    f = torch.tensor([1.0, 100.0, 300.0, 600.0], dtype=torch.float64)
    model = IMRPhenomHM(modes=((3, 3),), base_22=_FlatCarrier())
    _, _, modes = model(
        f, mass1=35.0, mass2=20.0, chi1=0.2, chi2=-0.1,
        f_ref=None, return_unprojected_modes=True,
    )
    m_sec = 55.0 * MSUN_SEC
    for kind, transition in (("amplitude", 0.014), ("phase", 0.018)):
        f0, rd_lm, rd_22 = seen[kind]
        torch.testing.assert_close(f0, transition / m_sec * rd_lm / rd_22)
    v = (2.0 * math.pi * m_sec * f / 3.0) ** (1.0 / 3.0)
    leading = 3.0 / 4.0 * math.sqrt(5.0 / 7.0) * (15.0 / 55.0) * v
    envelope = ((2.0 * f / 3.0) / seen["mapped_amplitude"]) ** (-7.0 / 6.0)
    expected = math.sqrt(16.0 * math.pi / 5.0) * leading * envelope
    torch.testing.assert_close(modes[(3, 3)].abs(), expected, rtol=1e-12, atol=0.0)


def test_xhm_time_and_orbital_phase_covariance_across_all_regions():
    """Fourier shift theorem and Eq2.13 hold after all joins and mode mixing."""
    f = torch.linspace(15.0, 1400.0, 150, dtype=torch.float64)
    model = IMRPhenomXHM()
    params = dict(mass1=35.0, mass2=20.0, chi1=0.4, chi2=-0.2,
                  f_ref=30.0, return_unprojected_modes=True)
    _, _, reference = model(f, **params)
    tc = torch.tensor(0.013, dtype=torch.float64, requires_grad=True)
    phic = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    _, _, shifted = model(f, tc=tc, phic=phic, **params)
    for l, m in model.SUPPORTED_MODES:
        factor = torch.exp(-1j * (2.0 * math.pi * f * tc - m * phic))
        scale = reference[(l, m)].abs().max()
        torch.testing.assert_close(shifted[(l, m)] / scale,
                                   reference[(l, m)] * factor / scale,
                                   rtol=1e-11, atol=1e-12)
    h = shifted[(3, 2)] / shifted[(3, 2)].abs().max().detach()
    grad_t, grad_phi = torch.autograd.grad(h.real.sum(), (tc, phic))
    torch.testing.assert_close(grad_t, (2.0 * math.pi * f * h.imag).sum())
    torch.testing.assert_close(grad_phi, (-2.0 * h.imag).sum())


def test_xhm_sqrt_preserves_small_modes_and_has_finite_boundary_gradient():
    value = torch.tensor([0.0, 1e-30, 0.25], dtype=torch.float64, requires_grad=True)
    result = _nonnegative_sqrt(value)
    torch.testing.assert_close(result, torch.tensor([0.0, 1e-15, 0.5], dtype=torch.float64),
                               rtol=1e-15, atol=0.0)
    gradient, = torch.autograd.grad(result.sum(), value)
    torch.testing.assert_close(gradient, torch.tensor([0.0, 5e14, 1.0], dtype=torch.float64))


def test_xhm_fit_composition_equal_mass_gradients():
    """Finite subgradients at the eta=1/4 boundary; no coefficient calibration."""
    eta = torch.tensor([0.2, 0.25], dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    s2 = torch.tensor([-0.2, -0.6], dtype=torch.float64, requires_grad=True)
    _, _, _, spin = XHMParameterFits.compute_spin_variables(eta, s1, s2)
    terms = []
    for mode in ("21", "33", "32", "44"):
        for prefix in ("amp_insp1", "amp_int1", "amp_alambda"):
            terms.append(getattr(XHMParameterFits, f"{prefix}_{mode}")(eta, s1, s2, spin))
    result = torch.stack(terms)
    assert torch.isfinite(result).all()
    gradients = torch.autograd.grad(result.sum(), (eta, s1, s2))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_xhm_body_exchange_including_equal_mass_unequal_spin_batch():
    """Eq2.13: relabeling the bodies multiplies each mode by (-1)^m."""
    f = torch.linspace(20.0, 1000.0, 50, dtype=torch.float64)
    parameters = torch.tensor([[35.0, 20.0, 0.4, -0.2], [30.0, 30.0, 0.6, -0.6]],
                              dtype=torch.float64, requires_grad=True)
    model = IMRPhenomXHM()
    m1, m2, s1, s2 = parameters.unbind(-1)
    _, _, first = model(f, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                       return_unprojected_modes=True)
    _, _, exchanged = model(f, mass1=m2, mass2=m1, chi1=s2, chi2=s1,
                           return_unprojected_modes=True)
    for l, m in model.SUPPORTED_MODES:
        scale = first[(l, m)].abs().max()
        torch.testing.assert_close(exchanged[(l, m)] / scale,
                                   (-1.0)**m * first[(l, m)] / scale,
                                   rtol=1e-12, atol=1e-12)
    scalar = sum(mode.real.sum() for lm, mode in first.items() if lm[1] > 0) * 1e22
    gradient, = torch.autograd.grad(scalar, parameters)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('lm', [(2, 1), (3, 3), (3, 2), (4, 4)])
def test_xhm_pn_phase_linear_expansion(lm):
    """Expand the complex coefficients of Appendix E independently of joins."""
    from torchwave.approximants.imrphenom_xhm import _pn_phase_lambda
    l, m = lm
    eta = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    delta = torch.sqrt(1 - 4 * eta)
    v = eta.new_tensor(1e-3)
    if lm == (2, 1):
        leading = 1j * delta * v
        correction = delta * (1 + math.log(16)) / 2 * v**4
    elif lm == (3, 3):
        leading = -1j * delta * v
        correction = delta * (-21 / 5 + 6 * math.log(1.5)) * v**4
    elif lm == (3, 2):
        leading = (1 - 3 * eta) * v**2
        correction = 1j * (3 - 66 * eta / 5) * v**5
    else:
        leading = (3 * eta - 1) * v**2
        correction = 1j * (eta * (3579 - 2880 * math.log(2)) - 1008 + 960 * math.log(2)) / 120 * v**5
    mf = m * v**3 / (2 * math.pi)
    expected = torch.angle((leading + correction) / leading)
    actual = _pn_phase_lambda(l, m, mf, eta, delta)
    torch.testing.assert_close(actual, expected, atol=1e-19, rtol=1e-9)
    assert torch.autograd.gradcheck(lambda e: _pn_phase_lambda(l, m, mf, e, torch.sqrt(1-4*e)), (eta,))


def test_xhm_spherical_basis_elimination_and_gradients():
    from torchwave.approximants.imrphenom_xhm import _spherical_32_from_spheroidal
    matrix = torch.tensor([[0.97+0.02j, 0.08-0.01j], [-0.05+0.03j, 0.96+0.04j]], dtype=torch.complex128, requires_grad=True)
    spheroidal = torch.tensor([1.2-0.4j, -0.3+0.7j], dtype=torch.complex128, requires_grad=True)
    spherical = matrix @ spheroidal
    actual = _spherical_32_from_spheroidal(spherical[0], spheroidal[1], *matrix.flatten())
    torch.testing.assert_close(actual, spherical[1], rtol=1e-14, atol=1e-14)
    inputs = (spherical[0].detach().requires_grad_(), spheroidal[1], *matrix.flatten())
    assert torch.autograd.gradcheck(_spherical_32_from_spheroidal, inputs)
    identity = _spherical_32_from_spheroidal(*inputs[:2], 1.0, 0.0, 0.0, 1.0)
    torch.testing.assert_close(identity, spheroidal[1])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_xhm_derivative_connections_work_in_inference_mode(dtype):
    model = IMRPhenomXHM()
    parameters = dict(mass1=torch.tensor([35., 30.], dtype=dtype),
                      mass2=torch.tensor([20., 30.], dtype=dtype),
                      chi1=torch.tensor([.4, .6], dtype=dtype),
                      chi2=torch.tensor([-.3, -.6], dtype=dtype))
    with torch.no_grad():
        expected = model(torch.linspace(20., 1000., 24, dtype=dtype), **parameters)
    with torch.inference_mode():
        # The frequency tensor is created inside inference mode deliberately.
        actual = model(torch.linspace(20., 1000., 24, dtype=dtype), **parameters)
    for result, reference in zip(actual, expected):
        assert not result.requires_grad
        assert torch.isfinite(result).all()
        scale = reference.abs().max()
        torch.testing.assert_close(result / scale, reference / scale)


@pytest.mark.parametrize('lm', [(2, 1), (3, 3), (3, 2), (4, 4)])
def test_xhm_exact_phase_joins_and_mixed_amplitude_boundaries(lm, monkeypatch):
    """Check one-sided derivatives of complete complex modes at paper cuts."""
    from torchwave.approximants.imrphenom_xas import XASParameterFits
    from torchwave.approximants.imrphenom_xhm import _eval_f_isco
    m1, m2, s1, s2 = [torch.tensor(v, dtype=torch.float64) for v in (35., 20., .4, -.3)]
    eta = m1*m2/(m1+m2)**2
    spin = XASParameterFits.fit_final_spin(eta, s1, s2)
    remnant = 1-XASParameterFits.fit_erad(eta, s1, s2)
    ring = {k: getattr(XHMParameterFits, 'qnm_fring_'+k)(spin)/remnant for k in ('21', '22', '32', '33', '44')}
    damp = {k: getattr(XHMParameterFits, 'qnm_fdamp_'+k)(spin)/remnant for k in ring}
    meco = XASParameterFits.fit_f_meco(eta, s1, s2)
    l, m = lm
    k = f'{l}{m}'
    cuts = {'phase_inspiral': m/2*meco,
            'phase_ringdown': ring[k]-damp[k] if lm != (3, 2) else ring['22']-.5*damp['22']}
    if lm == (3, 2):
        eff = (m1*s1+m2*s2)/(m1+m2)
        cuts.update(amplitude_inspiral=(meco+(_eval_f_isco(spin)-meco)*(.75-.235*abs(eff)))*ring['32']/ring['22'], amplitude_ringdown=ring['22'])
    model = IMRPhenomXHM(modes=(lm,))
    pole_fallback = False
    if lm == (3, 2):
        # Prove the default native case takes the Table VI pole branch. The
        # branch is intrinsic-row state, so one ordinary frequency sample is
        # enough to capture its production six-condition solve.
        solves = []
        original_solve = torch.linalg.solve

        def capture_amplitude_solve(matrix, rhs, *args, **kwargs):
            if matrix.shape[-2:] == (6, 6) and not solves:
                solves.append((matrix.detach().clone(), rhs.detach().clone()))
            return original_solve(matrix, rhs, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, 'solve', capture_amplitude_solve)
        model(torch.tensor([0.05], dtype=torch.float64), mass1=m1, mass2=m2,
              chi1=s1, chi2=s2, return_unprojected_modes=True)
        assert len(solves) == 1
        matrix, rhs = solves[0]
        coefficients = original_solve(matrix, rhs)
        nodes = torch.linspace(matrix[0, 1], matrix[4, 1], 11,
                               dtype=matrix.dtype, device=matrix.device)
        denominator = sum(coefficients[i] * nodes**i for i in range(6))
        threshold = 0.05 * torch.minimum(rhs[0], rhs[4])
        has_zero_crossing = (denominator <= threshold).any()
        parameter_veto = _published_parameter_amplitude_veto(
            lm, m1 / m2, eta, s1, s2,
        )
        assert not bool(parameter_veto)
        assert bool(has_zero_crossing)
        pole_fallback = True
    for name, cut in cuts.items():
        step = 1e-10
        mf = torch.stack([cut-step, cut+step]).requires_grad_()
        h = model(mf/((m1+m2)*MSUN_SEC), mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                  f_ref=20., return_unprojected_modes=True)[2][lm]*1e22
        deriv = torch.complex(torch.autograd.grad(h.real.sum(), mf, create_graph=True, retain_graph=True)[0],
                              torch.autograd.grad(h.imag.sum(), mf, create_graph=True, retain_graph=True)[0])
        phase_slope = -(deriv/h).imag
        phase_curvature = torch.autograd.grad(phase_slope.sum(), mf, retain_graph=True)[0]
        # Extrapolate each value and slope to the join, removing the known
        # O(step) change; a C1 phase may have different one-sided curvatures.
        value_error = abs((h[0]+step*deriv[0]-h[1]+step*deriv[1])/h[0])
        assert value_error < 1e-8, (lm, name)
        if name != 'phase_inspiral':
            torch.testing.assert_close(phase_slope[0]+step*phase_curvature[0],
                                       phase_slope[1]-step*phase_curvature[1], rtol=1e-9, atol=5e-6)
        if lm == (3, 2) and (name not in ('amplitude_inspiral', 'amplitude_ringdown') or not pole_fallback):
            torch.testing.assert_close((deriv/h).real[0], (deriv/h).real[1], rtol=1e-6, atol=1e-4)
            if name == 'phase_ringdown':
                torch.testing.assert_close(phase_curvature[0], phase_curvature[1], rtol=1e-6, atol=.05)


def test_xhm_quintic_amplitude_control_is_c1_at_both_joins(monkeypatch):
    """The source six-condition fit remains C1 when its pole detector is clear.

    The default (3,2) row above intentionally exercises the published pole
    fallback. Here inner fit values come from the actual endpoint Hermite
    polynomial, retaining physical boundary values and derivatives. The
    complete waveform must be C1 with this positive six-condition control.
    """
    m1, m2, s1, s2 = [torch.tensor(v, dtype=torch.float64) for v in (35., 20., .4, -.3)]
    model = IMRPhenomXHM(modes=((3, 2),))
    solves = []
    original_solve = torch.linalg.solve

    def capture_amplitude_solve(matrix, rhs, *args, **kwargs):
        if matrix.shape[-2:] == (6, 6) and matrix[1, 0] == 0:
            solves.append((matrix.detach().clone(), rhs.detach().clone()))
        return original_solve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'solve', capture_amplitude_solve)
    options = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                   return_unprojected_modes=True)
    model(torch.tensor([50.0], dtype=torch.float64), **options)
    assert len(solves) == 1
    matrix, rhs = solves[0]
    f_ins, f_rd = matrix[0, 1], matrix[4, 1]
    hermite = original_solve(matrix[[0, 1, 4, 5], :4], rhs[[0, 1, 4, 5]])
    for index, row in ((1, 2), (2, 3)):
        value = 1.0 / sum(hermite[i] * matrix[row, 1]**i for i in range(4))
        monkeypatch.setattr(XHMParameterFits, f'amp_int{index}_32',
                            staticmethod(lambda *args, _value=value: _value))

    eta = m1 * m2 / (m1 + m2)**2
    assert not bool(_published_parameter_amplitude_veto((3, 2), m1 / m2, eta, s1, s2))
    for cut in (f_ins, f_rd):
        step = 1e-10
        mf = torch.stack((cut - step, cut + step)).requires_grad_()
        h = model(mf / ((m1 + m2) * MSUN_SEC), **options)[2][(3, 2)] * 1e22
        amplitude = h.abs()
        derivative = torch.autograd.grad(amplitude.sum(), mf)[0]
        torch.testing.assert_close(amplitude[0] + step * derivative[0],
                                   amplitude[1] - step * derivative[1], rtol=1e-8, atol=0.)
        torch.testing.assert_close((derivative / amplitude)[0],
                                   (derivative / amplitude)[1], rtol=1e-6, atol=1e-4)
        # Verify the actual production solve clears the detector, rather than
        # inferring branch selection from continuity alone.
        active_matrix, active_rhs = solves[-1]
        coefficients = original_solve(active_matrix, active_rhs)
        check_nodes = torch.linspace(f_ins, f_rd, 11, dtype=matrix.dtype)
        denominator = sum(coefficients[i] * check_nodes**i for i in range(6))
        assert (denominator > 0.05 * active_rhs[[0, 4]].min()).all()


def test_xhm_default_inspiral_does_not_import_unverified_intermediate_fit_gauge(monkeypatch):
    """Unknown derivative-fit offsets must not set the default inspiral time."""
    frequency = torch.tensor([15.0, 18.0, 21.0], dtype=torch.float64)
    parameters = dict(mass1=35.0, mass2=20.0, chi1=0.4, chi2=-0.3,
                      f_ref=20.0, return_unprojected_modes=True)
    models = [IMRPhenomXHM()]
    reference = [model(frequency, **parameters)[2] for model in models]
    for name in ('21', '33', '32', '44'):
        attribute = f'phase_p1_{name}'
        original = getattr(XHMParameterFits, attribute)
        monkeypatch.setattr(XHMParameterFits, attribute,
                            staticmethod(lambda *args, original=original: original(*args) + 12345.0))
    changed = [model(frequency, **parameters)[2] for model in models]
    for mode in ((2, 1), (3, 3), (3, 2), (4, 4)):
        torch.testing.assert_close(changed[0][mode], reference[0][mode], rtol=0.0, atol=0.0)
