"""Pass 15 source checks for ordered XHM intermediate amplitude vetoes."""

import torch

from torchwave.approximants import imrphenom_xhm as xhm
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.xhm_fits import XHMParameterFits


DTYPE = torch.float64


def _veto(mode, q, eta=0.2, chi1=0.0, chi2=0.0, h_ins=None, h_int1=None,
          h_int2=None, h_rd=None):
    q = torch.as_tensor(q, dtype=DTYPE)
    eta = torch.as_tensor(eta, dtype=DTYPE)
    chi1 = torch.as_tensor(chi1, dtype=DTYPE)
    chi2 = torch.as_tensor(chi2, dtype=DTYPE)
    h_ins = torch.as_tensor(1.0 if h_ins is None else h_ins, dtype=DTYPE)
    h_int1 = torch.as_tensor(1.0 if h_int1 is None else h_int1, dtype=DTYPE)
    h_int2 = torch.as_tensor(1.0 if h_int2 is None else h_int2, dtype=DTYPE)
    h_rd = torch.as_tensor(1.0 if h_rd is None else h_rd, dtype=DTYPE)
    return xhm._ordered_amplitude_vetoes(
        mode, q, eta, chi1, chi2, h_ins, h_int1, h_int2, h_rd,
    )


def test_monotonic_order_removes_only_middle_node_and_supports_batches():
    alive1, alive2, derivatives = _veto(
        (2, 1), [2.0, 2.0, 2.0], h_ins=[1.0, 3.0, 3.0],
        h_int1=[2.0, 2.0, 1.0], h_int2=[3.0, 1.0, 2.0], h_rd=[4.0, 0.0, 3.0],
    )
    torch.testing.assert_close(alive1, torch.tensor([True, True, True]))
    torch.testing.assert_close(alive2, torch.tensor([True, True, False]))
    assert not derivatives.any()


def test_high_q_veto_applies_to_each_node_still_alive():
    # The first row has a nonmonotonic sequence and loses fInt2 first; the
    # later q>40 row then removes the remaining live fInt1 as well.
    alive1, alive2, _ = _veto(
        (2, 1), [41.0, 41.0], chi1=[0.91, 0.91],
        h_ins=[1.0, 1.0], h_int1=[3.0, 2.0], h_int2=[2.0, 3.0], h_rd=[4.0, 4.0],
    )
    torch.testing.assert_close(alive1, torch.tensor([False, False]))
    torch.testing.assert_close(alive2, torch.tensor([False, False]))


def test_numeric_thresholds_are_deferred_without_normalization_claim():
    # The primary paper does not say whether 0.2/0.01 are A/A0 or H_lm.
    # Consequently values on either side of those constants do not silently
    # activate a guessed unit convention in this pass.
    alive1, alive2, derivatives = _veto(
        (2, 1), 2.0, h_ins=0.1, h_int1=0.199, h_int2=0.201, h_rd=0.3,
    )
    assert bool(alive1) and bool(alive2) and not bool(derivatives)


def test_production_branch_retains_only_selected_inner_node(monkeypatch):
    model = IMRPhenomXHM(modes=((2, 1),))
    frequency = torch.tensor([100.0, 120.0], dtype=DTYPE)
    options = dict(
        mass1=torch.tensor([60.0, 60.0], dtype=DTYPE),
        mass2=torch.tensor([20.0, 10.0], dtype=DTYPE),
        chi1=torch.tensor([0.2, 0.2], dtype=DTYPE),
        chi2=torch.tensor([-0.1, -0.1], dtype=DTYPE),
        return_unprojected_modes=True,
    )

    def forced_selection(mode, q, eta, chi1, chi2, h_ins, h_int1, h_int2, h_rd):
        # q=3 retains fInt1 only (quartic branch); q=6 removes both.
        alive1 = q < 4.0
        alive2 = torch.zeros_like(alive1)
        return alive1, alive2, torch.zeros_like(alive1)

    monkeypatch.setattr(xhm, "_ordered_amplitude_vetoes", forced_selection)
    before = model(frequency, **options)[2][(2, 1)]
    original = XHMParameterFits.amp_int1_21
    monkeypatch.setattr(
        XHMParameterFits, "amp_int1_21",
        staticmethod(lambda *args: 1.2 * original(*args)),
    )
    after = model(frequency, **options)[2][(2, 1)]
    assert torch.isfinite(after).all()
    assert torch.abs(after[0] - before[0]).max() > 0.0
    torch.testing.assert_close(after[1], before[1], rtol=0.0, atol=0.0)


def test_production_branch_retains_either_single_inner_node(monkeypatch):
    """Both one-node quartics preserve the corresponding fit value."""
    model = IMRPhenomXHM(modes=((2, 1),))
    frequency = torch.tensor([100.0, 120.0], dtype=DTYPE)
    options = dict(
        mass1=torch.tensor([60.0, 60.0], dtype=DTYPE),
        mass2=torch.tensor([20.0, 10.0], dtype=DTYPE),
        chi1=torch.tensor([0.2, 0.2], dtype=DTYPE),
        chi2=torch.tensor([-0.1, -0.1], dtype=DTYPE),
        return_unprojected_modes=True,
    )
    original1 = XHMParameterFits.amp_int1_21
    original2 = XHMParameterFits.amp_int2_21
    for retain_first in (True, False):
        def forced_selection(mode, q, eta, chi1, chi2, h_ins, h_int1, h_int2, h_rd,
                             retain_first=retain_first):
            live = q < 4.0
            zeros = torch.zeros_like(live)
            return (live, zeros, zeros) if retain_first else (zeros, live, zeros)

        monkeypatch.setattr(xhm, "_ordered_amplitude_vetoes", forced_selection)
        monkeypatch.setattr(
            XHMParameterFits, "amp_int1_21", staticmethod(original1),
        )
        monkeypatch.setattr(
            XHMParameterFits, "amp_int2_21", staticmethod(original2),
        )
        before = model(frequency, **options)[2][(2, 1)]
        monkeypatch.setattr(
            XHMParameterFits, "amp_int1_21", staticmethod(
                (lambda *args: 1.2 * original1(*args)) if retain_first else original1,
            ),
        )
        monkeypatch.setattr(
            XHMParameterFits, "amp_int2_21", staticmethod(
                (lambda *args: 1.2 * original2(*args)) if not retain_first else original2,
            ),
        )
        after = model(frequency, **options)[2][(2, 1)]
        assert torch.isfinite(after).all()
        assert torch.abs(after[0] - before[0]).max() > 0.0
        torch.testing.assert_close(after[1], before[1], rtol=0.0, atol=0.0)


def test_production_veto_receives_common_a_over_a0_values(monkeypatch):
    captured = []
    solves = []
    original_veto = xhm._ordered_amplitude_vetoes
    original_solve = torch.linalg.solve

    def capture_veto(mode, q, eta, chi1, chi2, h_ins, h_int1, h_int2, h_rd):
        captured.append((h_ins.detach(), h_int1.detach(), h_int2.detach(), h_rd.detach()))
        return original_veto(mode, q, eta, chi1, chi2, h_ins, h_int1, h_int2, h_rd)

    def capture_solve(matrix, rhs, *args, **kwargs):
        if tuple(matrix.shape[-2:]) == (6, 6) and not solves:
            solves.append(rhs.detach().clone())
        return original_solve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(xhm, "_ordered_amplitude_vetoes", capture_veto)
    monkeypatch.setattr(torch.linalg, "solve", capture_solve)
    IMRPhenomXHM(modes=((2, 1),))(
        torch.tensor([100.0], dtype=DTYPE), mass1=torch.tensor(60.0, dtype=DTYPE),
        mass2=torch.tensor(20.0, dtype=DTYPE), chi1=torch.tensor(0.2, dtype=DTYPE),
        chi2=torch.tensor(-0.1, dtype=DTYPE), return_unprojected_modes=True,
    )
    assert len(captured) == len(solves) == 1
    endpoint1, inner1, inner2, endpoint2 = captured[0]
    rhs = solves[0]
    torch.testing.assert_close(endpoint1, 1.0 / rhs[..., 0])
    torch.testing.assert_close(inner1, 1.0 / rhs[..., 2])
    torch.testing.assert_close(inner2, 1.0 / rhs[..., 3])
    torch.testing.assert_close(endpoint2, 1.0 / rhs[..., 4])


def test_production_heterogeneous_batch_has_finite_parameter_gradients():
    model = IMRPhenomXHM(modes=((3, 2),))
    frequency = torch.tensor([100.0, 120.0], dtype=DTYPE, requires_grad=True)
    mass1 = torch.tensor([60.0, 60.0], dtype=DTYPE, requires_grad=True)
    chi1 = torch.tensor([0.2, 0.91], dtype=DTYPE, requires_grad=True)
    output = model(
        frequency, mass1=mass1, mass2=torch.tensor([20.0, 1.0], dtype=DTYPE),
        chi1=chi1, chi2=torch.tensor([-0.1, 0.0], dtype=DTYPE),
        return_unprojected_modes=True,
    )[2][(3, 2)]
    gradients = torch.autograd.grad((output.abs() * 1.0e22).sum(), (frequency, mass1, chi1))
    assert torch.isfinite(output).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
