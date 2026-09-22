"""Clean-room contract tests for the external ROM glue.

The external HDF5 is intentionally not packaged with TorchWave.  These tests
are skipped when that separately obtained public data file is unavailable.
"""

import pytest
import torch

from torchwave.approximants.rom.bspline import load_official_rom
from torchwave.approximants.rom.glue import OfficialROMGlue


@pytest.fixture(scope="module")
def glue(official_rom_path):
    low = load_official_rom(official_rom_path, "sub1")
    high = load_official_rom(official_rom_path, "sub2")
    return OfficialROMGlue(low, high, ringdown_mf=0.012)


def test_extrinsic_phase_and_distance_contract(glue):
    f = torch.tensor([30.0, 41.25, 83.5], dtype=torch.float64)
    hp0, hc0 = glue(f, 50.0, 25.0, .2, .1, distance_mpc=400.0,
                    inclination=.7, phic=.2, fref_hz=30.0, tc=.003)
    hp1, hc1 = glue(f, 50.0, 25.0, .2, .1, distance_mpc=800.0,
                    inclination=.7, phic=.2, fref_hz=30.0, tc=.003)
    assert torch.allclose(hp0 / hp1, torch.tensor(2.0, dtype=torch.complex128))
    assert torch.allclose(hc0 / hc1, torch.tensor(2.0, dtype=torch.complex128))
    hp_base, hc_base = glue(f, 50.0, 25.0, .2, .1, distance_mpc=400.0,
                            inclination=.7, phic=0., fref_hz=30., tc=0.)
    expected = torch.exp(1j * (2 * torch.pi * f * .003 + 2 * .2))
    assert torch.allclose(hp0 / hp_base, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(hc0 / hc_base, expected, rtol=1e-12, atol=1e-12)
    # Changing fref changes only the documented phase reference gauge.
    hp2, hc2 = glue(f, 50.0, 25.0, .2, .1, distance_mpc=400.0,
                    inclination=.7, phic=.2, fref_hz=35.0, tc=.003)
    assert torch.all(torch.isfinite(torch.stack((hp2.real, hp2.imag, hc2.real, hc2.imag))))
    assert torch.max(torch.abs(hp0 - hp2) / torch.abs(hp0)) > 1e-3


def test_parameter_batch_matches_scalar(glue):
    f = torch.tensor([30.0, 55.0, 120.0], dtype=torch.float64)
    args = [torch.tensor([50.0, 35.0]), torch.tensor([25.0, 35.0]),
            torch.tensor([.2, -.4]), torch.tensor([.1, .3])]
    hp, hc = glue(f, *args, distance_mpc=500.0, fref_hz=30.0)
    assert hp.shape == (2, 3)
    for i in range(2):
        hs = glue(f, *(x[i] for x in args), distance_mpc=500.0, fref_hz=30.0)
        assert torch.allclose(hp[i], hs[0])
        assert torch.allclose(hc[i], hs[1])


def test_support_bounds_and_gradient(glue):
    f = torch.tensor([30.0, 50.0], dtype=torch.float64)
    m1 = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    hp, hc = glue(f, m1, m2, torch.tensor(.2, dtype=torch.float64, requires_grad=True),
                  torch.tensor(.1, dtype=torch.float64), distance_mpc=500.0,
                  fref_hz=30.0, f_lower_hz=30.0, f_final_hz=100.0)
    (hp.real.square() + hp.imag.square() + hc.real.square() + hc.imag.square()).sum().backward()
    assert torch.isfinite(m1.grad) and torch.isfinite(m2.grad)
    padded_hp, padded_hc = glue(torch.tensor([29.0, 30.0, 100.0]), 50.0, 25.0, .2, .1,
                                distance_mpc=500.0, fref_hz=30.0,
                                f_lower_hz=30.0, f_final_hz=100.0)
    assert padded_hp[0] == 0 and padded_hc[0] == 0
    assert padded_hp[-1] == 0 and padded_hc[-1] == 0
    hp0, hc0 = glue(torch.tensor([0.0, 30.0]), 50.0, 25.0, .2, .1, distance_mpc=500.0)
    assert hp0[0] == 0 and hc0[0] == 0
    with pytest.raises(ValueError, match="non-negative"):
        glue(torch.tensor([-1.0]), 50.0, 25.0, .2, .1, distance_mpc=500.0)


def test_component_validation_and_spin_swap(glue):
    f = torch.tensor([30.0, 50.0], dtype=torch.float64)
    kwargs = dict(distance_mpc=500.0, fref_hz=30.0)
    for masses in ((0., 25.), (-1., 25.), (float("nan"), 25.)):
        with pytest.raises(ValueError, match="component masses"):
            glue(f, *masses, .2, .1, **kwargs)
    with pytest.raises(ValueError, match="spins"):
        glue(f, 50., 25., 1.1, .1, **kwargs)
    a = glue(f, 50., 25., .2, .1, **kwargs)
    b = glue(f, 25., 50., .1, .2, **kwargs)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
