"""Focused contracts for the native NNLO PhenomT precession wrappers."""

import numpy as np
import pytest
import torch

from torchwave.approximants.imrphenomt import IMRPhenomT
from torchwave.approximants.imrphenomtp import IMRPhenomTP
from torchwave.approximants.imrphenomtphm import IMRPhenomTPHM


def _complex(h):
    return h[0] - 1j * h[1]


def test_aligned_tp_is_exactly_the_aligned_model():
    t = torch.linspace(-0.5, 0.05, 64, dtype=torch.float64)
    kw = dict(
        mass1=30.0, mass2=20.0, chi1=0.4, chi2=-0.2, distance=100.0, inclination=1.1, phic=0.37, tc=0.02, f_ref=20.0
    )
    reference = IMRPhenomT().time_domain(t, **kw)
    actual = IMRPhenomTP().time_domain(t, **kw, spin1x=0.0, spin1y=0.0, spin2x=0.0, spin2y=0.0)
    torch.testing.assert_close(actual[0], reference[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], reference[1], rtol=0.0, atol=0.0)


def test_nonaligned_tp_is_finite_and_responds_to_inplane_spin():
    t = torch.linspace(-0.5, 0.05, 64, dtype=torch.float64)
    model = IMRPhenomTP()
    common = dict(mass1=30.0, mass2=20.0, chi1=0.4, chi2=-0.2, distance=100.0, inclination=1.1, phic=0.37)
    aligned = _complex(model.time_domain(t, **common))
    precessing = _complex(
        model.time_domain(
            t,
            **common,
            spin1x=0.2,
            spin1y=-0.1,
            spin2x=0.05,
            spin2y=0.08,
        )
    )
    assert torch.isfinite(precessing.real).all()
    assert torch.isfinite(precessing.imag).all()
    assert torch.linalg.vector_norm(precessing - aligned) > 1e-8 * torch.linalg.vector_norm(aligned)


def test_tp_rejects_unimplemented_precession_prescriptions():
    with pytest.raises(ValueError, match="precession_method='nnlo'"):
        IMRPhenomTP().time_domain(torch.tensor([-0.1], dtype=torch.float64), 30.0, 20.0, precession_method="numerical")


def test_tp_intrinsic_gradients_are_finite():
    t = torch.linspace(-0.3, 0.02, 20, dtype=torch.float64)
    x = torch.tensor([30.0, 20.0, 0.3, -0.2, 0.1, 0.05, -0.1, 0.08], dtype=torch.float64, requires_grad=True)
    m1, m2, c1, c2, x1, y1, x2, y2 = x
    hp, hc = IMRPhenomTP().time_domain(
        t,
        m1,
        m2,
        c1,
        c2,
        100.0,
        0.8,
        0.3,
        0.0,
        20.0,
        x1,
        y1,
        x2,
        y2,
    )
    weight = torch.linspace(0.3, 1.2, t.numel(), dtype=t.dtype)
    objective = (hp * (1.0 + weight) + hc * (1.0 - 0.2 * weight)).sum() * 1e21
    grad = torch.autograd.grad(objective, x)[0]
    assert torch.isfinite(grad).all()
    assert np.linalg.norm(grad.detach().numpy()) > 0.0


@pytest.mark.parametrize("model_cls", (IMRPhenomTP, IMRPhenomTPHM))
def test_precession_scalar_batch_and_mass_swap_equivalence(model_cls):
    t = torch.linspace(-0.35, 0.02, 24, dtype=torch.float64)
    model = model_cls()
    common = dict(
        chi1=torch.tensor([0.3, -0.4], dtype=torch.float64),
        chi2=torch.tensor([-0.2, 0.1], dtype=torch.float64),
        distance=100.0,
        inclination=0.8,
        phic=0.21,
        tc=0.0,
        f_ref=20.0,
        spin1x=torch.tensor([0.12, -0.08], dtype=torch.float64),
        spin1y=torch.tensor([-0.04, 0.06], dtype=torch.float64),
        spin2x=torch.tensor([0.03, 0.05], dtype=torch.float64),
        spin2y=torch.tensor([0.07, -0.02], dtype=torch.float64),
    )
    batched = model.time_domain(t, mass1=torch.tensor([40.0, 15.0]), mass2=torch.tensor([15.0, 40.0]), **common)
    for i, (m1, m2) in enumerate(((40.0, 15.0), (15.0, 40.0))):
        scalar_kw = {
            key: value[i] if isinstance(value, torch.Tensor) and value.ndim else value for key, value in common.items()
        }
        scalar = model.time_domain(t, mass1=m1, mass2=m2, **scalar_kw)
        batch_complex = batched[0][i] - 1j * batched[1][i]
        scalar_complex = scalar[0] - 1j * scalar[1]
        relative = torch.linalg.vector_norm(batch_complex - scalar_complex) / torch.linalg.vector_norm(scalar_complex)
        assert relative < 1e-12

    base = dict(
        mass1=40.0,
        mass2=15.0,
        chi1=0.3,
        chi2=-0.2,
        spin1x=0.12,
        spin1y=-0.04,
        spin2x=0.03,
        spin2y=0.07,
        distance=100.0,
        inclination=0.8,
        phic=0.21,
        f_ref=20.0,
    )
    swapped = dict(
        base,
        mass1=15.0,
        mass2=40.0,
        chi1=base["chi2"],
        chi2=base["chi1"],
        spin1x=base["spin2x"],
        spin1y=base["spin2y"],
        spin2x=base["spin1x"],
        spin2y=base["spin1y"],
    )
    torch.testing.assert_close(
        _complex(model.time_domain(t, **base)), _complex(model.time_domain(t, **swapped)), rtol=1e-12, atol=1e-30
    )


@pytest.mark.parametrize("model_cls", (IMRPhenomTP, IMRPhenomTPHM))
def test_negative_mode_selection_is_the_positive_mode_pair(model_cls):
    t = torch.linspace(-0.3, 0.01, 16, dtype=torch.float64)
    kw = dict(
        mass1=30.0,
        mass2=20.0,
        chi1=0.3,
        chi2=-0.2,
        distance=100.0,
        inclination=0.8,
        phic=0.21,
        f_ref=20.0,
        spin1x=0.1,
        spin1y=-0.04,
        spin2x=0.03,
        spin2y=0.07,
    )
    model = model_cls()
    positive = _complex(model.time_domain(t, **kw, modes=[(2, 2)]))
    negative = _complex(model.time_domain(t, **kw, modes=[(2, -2)]))
    pair = _complex(model.time_domain(t, **kw, modes=[(2, 2), (2, -2)]))
    assert torch.linalg.vector_norm(positive - negative) / torch.linalg.vector_norm(positive) < 1e-12
    assert torch.linalg.vector_norm(positive - pair) / torch.linalg.vector_norm(positive) < 1e-12


@pytest.mark.parametrize("model_cls", (IMRPhenomTP, IMRPhenomTPHM))
def test_precession_tiny_transverse_spin_has_aligned_limit(model_cls):
    t = torch.linspace(-0.35, 0.02, 24, dtype=torch.float64)
    model = model_cls()
    kw = dict(mass1=30.0, mass2=20.0, chi1=0.3, chi2=-0.2, distance=100.0, inclination=0.8, phic=0.21, f_ref=20.0)
    aligned = _complex(model.time_domain(t, **kw))
    tiny = _complex(model.time_domain(t, **kw, spin1x=1e-8, spin1y=-0.5e-8, spin2x=0.3e-8, spin2y=-0.2e-8))
    assert torch.linalg.vector_norm(tiny - aligned) / torch.linalg.vector_norm(aligned) < 1e-5


@pytest.mark.parametrize("model_cls", (IMRPhenomTP, IMRPhenomTPHM))
def test_exact_alignment_intrinsic_gradients_are_finite(model_cls):
    """The aligned limit must remain differentiable through NNLO geometry."""
    t = torch.linspace(-0.25, 0.015, 18, dtype=torch.float64)
    x = torch.tensor([30.0, 20.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    m1, m2, c1, c2, x1, y1, x2, y2 = x
    hp, hc = model_cls().time_domain(
        t,
        m1,
        m2,
        c1,
        c2,
        100.0,
        0.8,
        0.3,
        0.0,
        20.0,
        x1,
        y1,
        x2,
        y2,
    )
    weight = torch.linspace(0.3, 1.2, t.numel(), dtype=t.dtype)
    objective = (hp * (1.0 + weight) + hc * (1.0 - 0.2 * weight)).sum() * 1e21
    grad = torch.autograd.grad(objective, x)[0]
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad[:4]) > 0.0
