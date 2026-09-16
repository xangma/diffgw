"""Parent integration regression: an inactive fallback cannot change the strain."""
import torch
from torchwave.approximants import imrphenom_xhm as xhm


def test_bad_unused_value_polynomial_does_not_override_valid_selected_polynomial(monkeypatch):
    model = xhm.IMRPhenomXHM(modes=((2, 1),))
    frequency = torch.tensor([100.0, 120.0], dtype=torch.float64)
    options = dict(mass1=50.0, mass2=20.0, chi1=0.6, chi2=-0.8,
                   return_unprojected_modes=True)
    decisions = []

    def detector(coefficients, lower, upper):
        decision = decisions.pop(0)
        return torch.full_like(lower, decision, dtype=torch.bool)

    monkeypatch.setattr(xhm, "polynomial_has_nonpositive", detector)
    decisions[:] = [False, False]
    reference = model(frequency, **options)[2][(2, 1)]
    assert not decisions
    decisions[:] = [False, True]
    actual = model(frequency, **options)[2][(2, 1)]
    assert not decisions
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
