"""Source-defined parameter vetoes exercised through the XHM waveform."""

import pytest
import torch

from torchwave.approximants import imrphenom_xhm as xhm
from torchwave.approximants.xhm_fits import XHMParameterFits

DTYPE = torch.float64


@pytest.mark.parametrize("mode,q,eta,c1,c2,expected", [
    ((2, 1), 3., .20, .8, -.6, True),
    ((2, 1), 3., .23, .8, -.6, False),
    ((2, 1), 3., .20, .7, -.6, False),
    ((2, 1), 3., .20, .8, -.5, False),
    ((3, 2), 3., .20, -.7, .1, True),
    ((3, 2), 2.5, .20, -.7, .1, False),
    ((3, 2), 3., .20, -.6, .1, False),
    ((3, 2), 3., .20, -.7, 0., False),
    ((3, 2), 2., .22, -.95, -.95, True),
    ((3, 2), 2., .22, -.9, -.95, False),
    ((3, 2), 2., .22, -.95, -.9, False),
    ((3, 3), 8., .10, .96, 0., True),
    ((4, 4), 8., .10, .96, 0., True),
    ((3, 3), 7., .10, .96, 0., False),
    ((4, 4), 8., .10, .95, 0., False),
    ((2, 2), 8., .10, .96, 0., False),
    # The conditional q > 40 veto is deliberately outside this subset.
    ((2, 1), 41., .02, .91, 0., False),
])
def test_published_regions_and_strict_boundaries(mode, q, eta, c1, c2, expected):
    args = [torch.tensor(x, dtype=DTYPE) for x in (q, eta, c1, c2)]
    assert bool(xhm._published_parameter_amplitude_veto(mode, *args)) == expected


def test_vetoed_waveform_ignores_inner_fit_nodes_in_mixed_batch(monkeypatch):
    """Changing an inner node affects only the non-vetoed native-input row."""
    f = torch.linspace(20., 500., 151, dtype=DTYPE)
    # The first physical system is in the published (2,1) veto region:
    # eta < .23, chi1 > .7, chi2 < -.5. The control differs only in chi1.
    spin = torch.tensor([.8, .6], dtype=DTYPE, requires_grad=True)
    options = dict(mass1=torch.tensor([50., 50.], dtype=DTYPE), mass2=20.,
                   chi1=spin, chi2=-.8, return_unprojected_modes=True)
    model = xhm.IMRPhenomXHM(modes=((2, 1),))
    before = model(f, **options)[2][(2, 1)]
    gradient = torch.autograd.grad((before.abs() * 1e22).square().sum(), spin)[0]
    assert torch.isfinite(before).all()
    assert torch.isfinite(gradient).all()
    assert (gradient != 0).all()

    original = XHMParameterFits.amp_int1_21
    monkeypatch.setattr(XHMParameterFits, "amp_int1_21",
                        staticmethod(lambda *args: 1.01 * original(*args)))
    after = model(f, **options)[2][(2, 1)]
    torch.testing.assert_close(after[0], before[0], rtol=0., atol=0.)
    relative_change = torch.linalg.vector_norm(after[1] - before[1]) / torch.linalg.vector_norm(before[1])
    assert relative_change > 1e-4

    # Disable just the published selection: the vetoed physical system now
    # depends on the inner nodes, so the check cannot pass via a fallback.
    monkeypatch.setattr(xhm, "_published_parameter_amplitude_veto",
                        lambda mode, q, eta, c1, c2: torch.zeros_like(eta, dtype=torch.bool))
    without_veto = model(f, **options)[2][(2, 1)]
    relative_change = torch.linalg.vector_norm(without_veto[0] - after[0]) / torch.linalg.vector_norm(after[0])
    assert relative_change > 1e-4
