import torch

from torchwave.approximants.precession_twisting_clean import twist_mode, wigner_d


def test_wigner_identity_at_zero_tilt():
    b = torch.tensor([0., 0.3], dtype=torch.float64)
    assert torch.allclose(wigner_d(2, 2, 2, b[:1]), torch.ones(1, dtype=torch.float64))
    assert torch.allclose(wigner_d(2, 2, 1, b[:1]), torch.zeros(1, dtype=torch.float64))


def test_twist_identity_and_phase_rotation():
    h22 = torch.tensor([1+2j, 2-1j], dtype=torch.complex128)
    h = twist_mode(2, 2, torch.zeros(2), torch.zeros(2), torch.zeros(2), {2: h22})
    torch.testing.assert_close(h, h22)
    rotated = twist_mode(2, 2, torch.tensor([.2]), torch.zeros(1), torch.zeros(1), {2: h22[:1]})
    torch.testing.assert_close(rotated, torch.exp(-2j*torch.tensor([.2]))*h22[:1])


def test_wigner_d_is_real_and_finite():
    b = torch.linspace(0., 3.14, 17, dtype=torch.float64)
    for m in range(-2, 3):
        for mp in range(-2, 3):
            d = wigner_d(2, m, mp, b)
            assert d.dtype == torch.float64 and torch.isfinite(d).all()

