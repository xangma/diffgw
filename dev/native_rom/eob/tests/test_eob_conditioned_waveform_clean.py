"""Focused contract checks for the clean SEOBNRv4 TD-to-FD adapter.

The generator is injected in every end-to-end test.  These checks exercise
the adapter's public wiring (sampling, support start, projection, phase and
metadata) without requiring a long physical EOB integration or any external
reference implementation.
"""

# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

from torchwave.constants import MSUN_SEC, PI
from dev.native_rom.eob.torch.conditioned_waveform_clean import SEOBNRV4_ADAPTIVE_CHOICES, ConditionedSEOBNRv4FD, _extract_generated_modes, conditioned_seobnrv4_fd
from dev.native_rom.eob.torch.published_waveform import PublishedEOB22
from torchwave.approximants.eob.td_fd_conditioning_clean import (
    fft_length_and_sampling,
    plan_td_support,
)


class SyntheticTDGenerator:
    """Small differentiable mode source with the native adapter signature."""

    def __init__(self, n=256):
        self.n = n
        self.calls = []

    def __call__(self, omega_start, mass1, mass2, chi1, chi2, **kwargs):
        self.calls.append((omega_start, mass1, mass2, chi1, chi2, kwargs))
        dt_geom = kwargs["dt_geom"]
        time = torch.arange(
            self.n, dtype=mass1.dtype, device=mass1.device
        ) * dt_geom
        amplitude = 0.01 * (mass1 + 2.0 * mass2) + 0.1
        mode22 = amplitude * torch.exp(-0.001 * time) * torch.exp(0.02j * time)
        mode2m2 = 0.75 * torch.conj(mode22)
        peak = SimpleNamespace(time=time[self.n // 2])
        return SimpleNamespace(
            time=time,
            mode=mode22,
            mode_2m2=mode2m2,
            trajectory=SimpleNamespace(peak=peak),
        )


def _generate(generator, **kwargs):
    return conditioned_seobnrv4_fd(
        30.0,
        20.0,
        f_low=800.0,
        f_max=1024.0,
        delta_f=1.0,
        generator=generator,
        num_steps=256,
        return_metadata=True,
        **kwargs,
    )


def test_native_published_eob_namedtuple_uses_mode_field_for_extraction():
    time = torch.arange(4, dtype=torch.float64)
    mode = torch.tensor([1.0 + 0.5j, 0.8 + 0.4j, 0.6 + 0.3j, 0.4 + 0.2j], dtype=torch.complex128)
    native = PublishedEOB22(
        time,
        mode,
        SimpleNamespace(),
        SimpleNamespace(),
        torch.tensor(0.95, dtype=torch.float64),
        torch.tensor(0.7, dtype=torch.float64),
    )
    extracted_time, extracted_mode, extracted_minus, extracted_result = _extract_generated_modes(
        native, dtype=torch.float64, device=torch.device("cpu")
    )
    torch.testing.assert_close(extracted_time, time)
    torch.testing.assert_close(extracted_mode, mode)
    torch.testing.assert_close(extracted_minus, torch.conj(mode))
    assert extracted_result is native


def test_adapter_derives_fixed_grid_start_frequency_dt_and_reviewed_choices():
    generator = SyntheticTDGenerator()
    result = _generate(generator, distance=120.0, inclination=0.4, phic=0.2)

    assert isinstance(result, ConditionedSEOBNRv4FD)
    assert result.sampling.n_time == 2048
    assert result.sampling.n_frequency == 1025
    assert result.frequencies.shape == (1025,)
    torch.testing.assert_close(
        result.frequencies,
        torch.arange(1025, dtype=torch.float64),
    )
    torch.testing.assert_close(result.sampling.dt, torch.tensor(1.0 / 2048.0, dtype=torch.float64))
    torch.testing.assert_close(result.dt_seconds, result.sampling.dt)
    torch.testing.assert_close(
        result.dt_geom,
        result.dt_seconds / result.support.total_mass_seconds,
    )

    omega, m1, m2, chi1, chi2, call_kwargs = generator.calls[0]
    expected_omega = (
        torch.tensor(PI, dtype=torch.float64)
        * result.support.total_mass_seconds
        * result.support.start_frequency
    )
    torch.testing.assert_close(omega, expected_omega)
    torch.testing.assert_close(omega, torch.tensor(PI) * (m1 + m2) * MSUN_SEC * result.support.start_frequency)
    torch.testing.assert_close(call_kwargs["dt_geom"], result.dt_geom)
    assert call_kwargs["spin_truncation"] == "generic-leading-ell8"
    assert call_kwargs["spin_v6"] == "bohe_total_energy"
    assert call_kwargs["include_pan_v8"] is False
    assert call_kwargs["include_positive_chi_540"] is False
    assert call_kwargs["final_spin_fit_order"] == (3, 4)
    assert call_kwargs["ringdown_duration"] == 150.0
    assert result.choices["spin_truncation"] == SEOBNRV4_ADAPTIVE_CHOICES["spin_truncation"]
    assert result.choices["spin_v6"] == SEOBNRV4_ADAPTIVE_CHOICES["spin_v6"]
    assert result.choices["include_pan_v8"] is False
    assert result.choices["include_positive_chi_540"] is False
    assert result.choices["final_spin_fit_order"] == (3, 4)


def test_tc_is_the_only_frequency_domain_phase_and_epoch_is_metadata():
    zero = _generate(SyntheticTDGenerator(), tc=0.0)
    delayed_generator = SyntheticTDGenerator()
    delayed = _generate(delayed_generator, tc=0.00125)

    phase = torch.exp(-2j * torch.tensor(PI) * delayed.frequencies * 0.00125)
    torch.testing.assert_close(delayed.h_plus, zero.h_plus * phase, rtol=2e-6, atol=2e-28)
    torch.testing.assert_close(delayed.h_cross, zero.h_cross * phase, rtol=2e-6, atol=2e-28)
    expected_epoch = (
        zero.td_time[0] - zero.generator_result.trajectory.peak.time
    ) * zero.support.total_mass_seconds
    torch.testing.assert_close(zero.physical_epoch, expected_epoch)
    assert zero.physical_epoch.item() < 0.0


def test_scalar_gradients_survive_projection_conditioning_and_fft():
    generator = SyntheticTDGenerator()
    mass1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    distance = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
    phic = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    result = conditioned_seobnrv4_fd(
        mass1,
        20.0,
        chi1z=torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
        chi2z=-0.2,
        f_low=800.0,
        f_max=1024.0,
        delta_f=1.0,
        distance=distance,
        phic=phic,
        tc=torch.tensor(0.001, dtype=torch.float64, requires_grad=True),
        generator=generator,
        num_steps=256,
        return_metadata=True,
    )
    loss = result.h_plus.real.sum() + 0.25 * result.h_cross.imag.sum()
    loss.backward()
    for value in (mass1.grad, distance.grad, phic.grad):
        assert value is not None
        assert torch.isfinite(value).all()
    assert result.h_plus.dtype == torch.complex128
    assert result.h_cross.dtype == torch.complex128
    assert result.frequencies.dtype == torch.float64


def test_batch_rows_integrate_independently_on_one_frequency_grid():
    generator = SyntheticTDGenerator()
    result = conditioned_seobnrv4_fd(
        torch.tensor([30.0, 32.0], dtype=torch.float64),
        torch.tensor([20.0, 18.0], dtype=torch.float64),
        chi1z=torch.tensor([0.1, -0.2], dtype=torch.float64),
        chi2z=0.0,
        f_low=800.0,
        f_max=1024.0,
        delta_f=1.0,
        generator=generator,
        num_steps=256,
        return_metadata=True,
    )
    assert len(generator.calls) == 2
    assert result.h_plus.shape == (2, 1025)
    assert result.h_cross.shape == (2, 1025)
    assert result.frequencies.shape == (1025,)
    assert len(result.support) == 2
    torch.testing.assert_close(result.dt_seconds, torch.tensor(1.0 / 2048.0, dtype=torch.float64))


def test_float32_dtype_and_device_are_preserved():
    generator = SyntheticTDGenerator()
    result = conditioned_seobnrv4_fd(
        torch.tensor(30.0, dtype=torch.float32),
        torch.tensor(20.0, dtype=torch.float32),
        f_low=torch.tensor(800.0, dtype=torch.float32),
        f_max=torch.tensor(1024.0, dtype=torch.float32),
        delta_f=torch.tensor(1.0, dtype=torch.float32),
        generator=generator,
        num_steps=256,
        return_metadata=True,
    )
    assert result.h_plus.dtype == torch.complex64
    assert result.h_cross.dtype == torch.complex64
    assert result.frequencies.dtype == torch.float32
    assert result.dt_geom.dtype == torch.float32
    assert result.h_plus.device.type == "cpu"


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"f_low": 800.0, "f_max": 800.0, "delta_f": 1.0}, ValueError, "greater"),
        ({"f_low": torch.tensor([800.0, 801.0]), "f_max": 1024.0, "delta_f": 1.0}, ValueError, "scalar"),
        ({"f_low": 800.0, "f_max": 1024.0, "delta_f": 1.0, "fref": 20.0}, ValueError, "fref"),
        ({"f_low": 800.0, "f_max": 1024.0, "delta_f": 1.0, "distance": 0.0}, ValueError, "distance"),
    ],
)
def test_invalid_requests_are_rejected_explicitly(kwargs, error, match):
    with pytest.raises(error, match=match):
        conditioned_seobnrv4_fd(
            30.0,
            20.0,
            generator=SyntheticTDGenerator(),
            **kwargs,
        )


def test_grid_plan_matches_adapter_metadata():
    generator = SyntheticTDGenerator()
    result = _generate(generator)
    expected = fft_length_and_sampling(1024.0, 1.0)
    assert result.sampling.n_time == expected.n_time
    assert result.sampling.n_frequency == expected.n_frequency
    torch.testing.assert_close(result.sampling.dt, expected.dt)
    torch.testing.assert_close(result.sampling.delta_f, expected.delta_f)
    support = plan_td_support(30.0, 20.0, 800.0)
    torch.testing.assert_close(result.support.start_frequency, support.start_frequency)
