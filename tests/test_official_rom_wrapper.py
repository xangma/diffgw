from pathlib import Path
import pytest
import torch
from torchwave.waveforms import get_fd_waveform

from torchwave.approximants.rom import OfficialSEOBNRv4ROM

ROM = Path('/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5')

@pytest.fixture()
def wrapper():
    if not ROM.is_file():
        pytest.skip('external public ROM unavailable')
    return OfficialSEOBNRv4ROM(ROM, flow=20., fhigh=256.)

def test_external_path_and_aliases(wrapper):
    f = torch.tensor([0., 20., 30., 256.], dtype=torch.float64)
    a = wrapper(f, 30., 20., 0., 0., distance=100., f_ref=20.)
    b = wrapper(f, 30., 20., 0., 0., distance_mpc=100., f_ref=20.)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[0][0] == 0 and a[0][-1] == 0

def test_missing_external_path():
    with pytest.raises(FileNotFoundError):
        OfficialSEOBNRv4ROM('/tmp/does-not-exist-seobnr.hdf5')

def test_mass_swap_and_intrinsic_gradient(wrapper):
    f = torch.tensor([30., 50., 100.], dtype=torch.float64)
    a = wrapper(f, 30., 20., .2, -.1, distance=100., inclination=.6, phic=.2, f_ref=20.)
    b = wrapper(f, 20., 30., -.1, .2, distance=100., inclination=.6, phic=.2, f_ref=20.)
    assert torch.allclose(a[0], b[0], rtol=1e-12, atol=0.)
    assert torch.allclose(a[1], b[1], rtol=1e-12, atol=0.)
    m = torch.tensor(30., dtype=torch.float64, requires_grad=True)
    hp, hc = wrapper(f, m, 20., .2, -.1, distance=100., f_ref=20.)
    (hp.abs().sum() / 1e-21).backward()
    assert torch.isfinite(m.grad) and abs(float(m.grad)) > 0

def test_public_waveform_dispatch_and_alias(wrapper):
    f = torch.tensor([0., 20., 30., 80., 120.], dtype=torch.float64)
    direct = wrapper(f, 30., 20., .2, -.1, distance=100., inclination=.6,
                     phic=.17, tc=1.2e-3, f_ref=20., flow=20., fhigh=100.)
    routed = get_fd_waveform(
        "SEOBNRv4_ROM", 30., 20., chi1=.2, chi2=-.1, distance=100.,
        inclination=.6, phic=.17, tc=1.2e-3, f_lower=20., f_final=100.,
        delta_f=10., sample_frequencies=f, model_path=str(ROM), f_ref=None,
    )
    assert torch.allclose(direct[0], routed[0], rtol=1e-12, atol=0.)
    assert torch.allclose(direct[1], routed[1], rtol=1e-12, atol=0.)

def test_public_waveform_dispatch_requires_external_data():
    with pytest.raises(FileNotFoundError):
        get_fd_waveform("SEOBNR_ROM", 30., 20., chi1=0., chi2=0.,
                        sample_frequencies=torch.tensor([20., 30.], dtype=torch.float64),
                        model_path="/tmp/no-official-rom.hdf5")

def test_canonical_and_official_aliases_match(wrapper):
    f = torch.tensor([20., 30., 80.], dtype=torch.float64)
    kwargs = dict(mass1=30., mass2=20., chi1=.2, chi2=-.1,
                  sample_frequencies=f, f_lower=20., f_final=100.,
                  model_path=str(ROM), distance=100.)
    official = get_fd_waveform("official_seobnr_v4_rom", **kwargs)
    canonical = get_fd_waveform("SEOBNR_ROM", **kwargs)
    assert torch.allclose(official[0], canonical[0], rtol=1e-12, atol=0.)
    assert torch.allclose(official[1], canonical[1], rtol=1e-12, atol=0.)

def test_batched_support_bounds(wrapper):
    f = torch.tensor([0., 20., 30., 80., 100., 120.], dtype=torch.float64)
    hp, hc = wrapper(
        f, torch.tensor([30., 31.], dtype=torch.float64),
        torch.tensor([20., 19.], dtype=torch.float64),
        torch.tensor([.2, .1], dtype=torch.float64),
        torch.tensor([-.1, 0.], dtype=torch.float64), distance=100.,
        flow=torch.tensor([20., 30.], dtype=torch.float64),
        fhigh=torch.tensor([100., 90.], dtype=torch.float64),
    )
    assert hp.shape == hc.shape == (2, f.numel())
    assert torch.equal(hp[:, 0], torch.zeros(2, dtype=hp.dtype))
    assert torch.equal(hp[0, -1:], torch.zeros(1, dtype=hp.dtype))
    assert torch.equal(hp[1, 1], torch.zeros((), dtype=hp.dtype))
    assert torch.equal(hp[1, 4:], torch.zeros(2, dtype=hp.dtype))

def test_scalar_intrinsics_vector_bounds_and_constructor_bounds():
    if not ROM.is_file():
        pytest.skip('external public ROM unavailable')
    w = OfficialSEOBNRv4ROM(ROM)
    f = torch.tensor([0., 20., 30., 80., 100.], dtype=torch.float64)
    hp, _ = w(f, 30., 20., .2, -.1, distance=100.,
              flow=torch.tensor([20., 30.]), fhigh=torch.tensor([90., 100.]))
    assert hp.shape == (2, f.numel())
    w2 = OfficialSEOBNRv4ROM(ROM, flow=torch.tensor([20., 30.]),
                             fhigh=torch.tensor([90., 100.]))
    hp2, _ = w2(f, 30., 20., .2, -.1, distance=100.)
    assert torch.allclose(hp, hp2, rtol=1e-12, atol=0.)

def test_dtype_alias_conflict_and_nonfinite_extrinsics(wrapper):
    for dtype, expected in ((torch.float32, torch.complex64),
                            (torch.float64, torch.complex128)):
        hp, hc = wrapper(torch.tensor([20., 30.], dtype=dtype), 30., 20.,
                         distance_mpc=torch.tensor(100., dtype=dtype))
        assert hp.dtype == hc.dtype == expected
    with pytest.raises(TypeError):
        wrapper(torch.tensor([20., 30.]), 30., 20., distance=100., distance_mpc=100.)
    for name in ("inclination", "phic", "tc"):
        with pytest.raises(ValueError):
            wrapper(torch.tensor([20., 30.]), 30., 20., **{name: float("nan")})
    with pytest.raises(ValueError):
        wrapper(torch.tensor([20., 30.]), 30., 20., f_ref=float("inf"))

def test_finite_intrinsic_gradients_and_ringdown_validation(wrapper):
    f = torch.tensor([20., 40., 80.], dtype=torch.float64, requires_grad=True)
    mass = torch.tensor(30., dtype=torch.float64, requires_grad=True)
    hp, hc = wrapper(f, mass, 20., .2, -.1, distance=100.)
    (hp.real.square().sum() + hc.imag.square().sum()).backward()
    assert torch.isfinite(mass.grad) and torch.isfinite(f.grad).all()
    for value in (float("nan"), 0., -1.):
        bad = OfficialSEOBNRv4ROM(ROM, ringdown_mf=lambda *args, v=value: torch.tensor(v))
        with pytest.raises(ValueError):
            bad(torch.tensor([20., 30.]), 30., 20., distance=100.)
