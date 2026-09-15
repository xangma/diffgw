import math
import torch

from torchwave.approximants.precession_twisting_clean import (
    project_twisted_positive_modes, wigner_d,
)


def test_positive_projection_face_on_22_normalization():
    h = torch.tensor([1.2 + .4j, -.2 + .7j], dtype=torch.complex128)
    hp, hc = project_twisted_positive_modes({(2, 2): h}, 0., 0., 0., 0., 0.)
    # At theta=beta=0 only m=2 survives and Y_222=sqrt(5/(4 pi)).
    expected = math.sqrt(5 / (4 * math.pi)) * h
    assert torch.allclose(hp + 1j * hc, expected, atol=1e-14, rtol=1e-14)


def test_positive_projection_is_finite_and_rotation_identity():
    f = torch.linspace(.01, .3, 9, dtype=torch.float64)
    h = torch.exp(1j * f)
    hp0, hc0 = project_twisted_positive_modes({(2, 2): h}, 0., 0., 0., .7, 0.)
    hp1, hc1 = project_twisted_positive_modes({(2, 2): h}, 0., 0., 0., .7, .23)
    assert torch.isfinite(hp0).all() and torch.isfinite(hc0).all()
    assert torch.isfinite(hp1).all() and torch.isfinite(hc1).all()
    c, s = math.cos(2 * .23), math.sin(2 * .23)
    assert torch.allclose(hp1, c * hp0 + s * hc0, atol=1e-14, rtol=1e-14)
    assert torch.allclose(hc1, c * hc0 - s * hp0, atol=1e-14, rtol=1e-14)


def test_wigner_identity_at_zero_tilt():
    for ell in range(2, 5):
        for m in range(-ell, ell + 1):
            for mp in range(-ell, ell + 1):
                got = wigner_d(ell, m, mp, torch.tensor(0., dtype=torch.float64))
                expected = 1. if m == mp else 0.
                assert abs(float(got) - expected) < 1e-14
