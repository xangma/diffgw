"""Focused checks for the Section V.B nonmixing phase-node correction."""

import math

import torch

import torchwave.approximants.imrphenom_xhm as xhm
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM


NONMIXING = ((2, 1), (3, 3), (4, 4))


def _phase_node_values(b, e):
    """Reconstruct Table VII's six points from its two endpoints."""
    return torch.stack((
        b,
        0.25 * ((math.sqrt(3.0) + 2.0) * b - (math.sqrt(3.0) - 2.0) * e),
        0.25 * (e + 3.0 * b),
        0.50 * (e + b),
        0.25 * (3.0 * e + b),
        0.125 * (7.0 * e + b),
    ))


def test_nonmixing_phase_solve_captures_default_table_vii_subset(monkeypatch):
    """The production 5x5 solve uses f1,f2,f3,f4,f6 for every nonmixing mode."""
    original_solve = torch.linalg.solve

    for lm in NONMIXING:
        captured = []
        captured_rhs = []
        captured_p6 = []
        p6_name = f"phase_p6_{lm[0]}{lm[1]}"
        original_p6 = getattr(xhm.XHMParameterFits, p6_name)

        def capture_solve(a, b, *args, **kwargs):
            if a.shape[-2:] == (5, 5) and torch.allclose(
                a[..., 1], torch.ones_like(a[..., 1])
            ):
                captured.append(a.detach().clone())
                captured_rhs.append(b.detach().clone())
            return original_solve(a, b, *args, **kwargs)

        def capture_p6(*args, _original=original_p6, **kwargs):
            value = _original(*args, **kwargs)
            captured_p6.append(value.detach().clone())
            return value

        monkeypatch.setattr(torch.linalg, "solve", capture_solve)
        monkeypatch.setattr(xhm.XHMParameterFits, p6_name, staticmethod(capture_p6))
        model = IMRPhenomXHM(modes=(lm,))
        model(
            torch.linspace(20.0, 500.0, 48, dtype=torch.float64),
            mass1=torch.tensor(35.0, dtype=torch.float64),
            mass2=torch.tensor(20.0, dtype=torch.float64),
            chi1=torch.tensor(0.4, dtype=torch.float64),
            chi2=torch.tensor(-0.2, dtype=torch.float64),
            return_unprojected_modes=True,
        )

        assert len(captured) == 1, f"expected one nonmixing phase solve for {lm}"
        assert len(captured_rhs) == 1 and len(captured_p6) == 1
        matrix = captured[0]
        nodes = 1.0 / matrix[..., 2]
        b = nodes[0]
        e = 4.0 * nodes[2] - 3.0 * b
        expected = _phase_node_values(b, e)
        torch.testing.assert_close(nodes, expected[[0, 1, 2, 3, 5]], atol=0.0, rtol=2e-12)
        assert not torch.isclose(nodes[-1], expected[4], atol=0.0, rtol=1e-12)
        torch.testing.assert_close(captured_rhs[0][-1], captured_p6[0], atol=0.0, rtol=0.0)


def test_nonmixing_phase_p6_changes_normalized_strain_and_p5_is_unused(monkeypatch):
    """Changing only p6 changes each mode while a p5 call would fail the test."""
    frequency = torch.linspace(20.0, 600.0, 96, dtype=torch.float64)
    baseline = {}
    calls = {lm: 0 for lm in NONMIXING}

    for lm in NONMIXING:
        suffix = f"{lm[0]}{lm[1]}"
        p5_name = f"phase_p5_{suffix}"
        p6_name = f"phase_p6_{suffix}"
        original_p6 = getattr(xhm.XHMParameterFits, p6_name)

        def forbidden(*args, _name=p5_name, **kwargs):
            raise AssertionError(f"excluded {_name} was evaluated")

        def tracked_p6(*args, _original=original_p6, _lm=lm, **kwargs):
            calls[_lm] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(xhm.XHMParameterFits, p5_name, staticmethod(forbidden))
        monkeypatch.setattr(xhm.XHMParameterFits, p6_name, staticmethod(tracked_p6))
        out = IMRPhenomXHM(modes=(lm,))(
            frequency,
            mass1=35.0,
            mass2=20.0,
            chi1=0.4,
            chi2=-0.2,
            return_unprojected_modes=True,
        )[2][lm]
        baseline[lm] = out / out.abs().max()

        def shifted_p6(*args, _original=original_p6, **kwargs):
            value = _original(*args, **kwargs)
            return value + torch.ones_like(value)

        monkeypatch.setattr(xhm.XHMParameterFits, p6_name, staticmethod(shifted_p6))
        changed = IMRPhenomXHM(modes=(lm,))(
            frequency,
            mass1=35.0,
            mass2=20.0,
            chi1=0.4,
            chi2=-0.2,
            return_unprojected_modes=True,
        )[2][lm]
        changed = changed / changed.abs().max()
        assert torch.max(torch.abs(changed - baseline[lm])).item() > 1e-8
        assert calls[lm] == 1


def test_nonmixing_phase_gradients_are_finite(monkeypatch):
    """The corrected solve remains differentiable for all three nonmixing modes."""
    for lm in NONMIXING:
        suffix = f"{lm[0]}{lm[1]}"
        p5_name = f"phase_p5_{suffix}"

        def forbidden(*args, _name=p5_name, **kwargs):
            raise AssertionError(f"excluded {_name} was evaluated")

        monkeypatch.setattr(xhm.XHMParameterFits, p5_name, staticmethod(forbidden))
        m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
        m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
        chi1 = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        chi2 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
        h = IMRPhenomXHM(modes=(lm,))(
            torch.linspace(20.0, 500.0, 64, dtype=torch.float64),
            mass1=m1,
            mass2=m2,
            chi1=chi1,
            chi2=chi2,
            return_unprojected_modes=True,
        )[2][lm]
        (1e22 * (h.real + 0.37 * h.imag).sum()).backward()
        for parameter in (m1, m2, chi1, chi2):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs() > 0.0
