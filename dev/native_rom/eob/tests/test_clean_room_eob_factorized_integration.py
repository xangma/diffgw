"""Independent energy-balance and covariance checks for factorized EOB wiring."""

import pytest
import torch

from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform


DTYPE = torch.float64


@pytest.mark.parametrize("model,spin", [("din2009_nonspinning", 0.), ("pan2011_leading_spin", .3)])
def test_factorized_waveform_preserves_total_mass_distance_and_time_scaling(model, spin):
    f = torch.tensor([30., 50., 70., 90.], dtype=DTYPE)
    args = dict(approximant="SEOBNR_ODE", conservative_model="seobnrv4_published",
                radiation_model=model, chi1=spin, chi2=-spin/2, inclination=.8,
                num_steps=3, dt_geom=1., phic=.2)
    first = get_fd_waveform(mass1=30., mass2=20., distance=200., tc=.002,
                            sample_frequencies=f, f_lower=30., **args)
    second = get_fd_waveform(mass1=60., mass2=40., distance=600., tc=.004,
                             sample_frequencies=f/2, f_lower=15., **args)
    for a, b in zip(first, second):
        torch.testing.assert_close(b, 4*a/3, rtol=3e-11, atol=1e-33)


def test_factorized_waveform_batched_spins_and_inference_match_scalar_calls():
    f = torch.tensor([30., 45., 60.], dtype=DTYPE)
    m1 = torch.tensor([30., 35.], dtype=DTYPE)
    spin = torch.tensor([.3, -.2], dtype=DTYPE)
    args = dict(approximant="SEOBNR_ODE", mass2=20., sample_frequencies=f,
                f_lower=30., conservative_model="seobnrv4_published",
                radiation_model="pan2011_leading_spin", num_steps=2, dt_geom=1.)
    batch = get_fd_waveform(mass1=m1, chi1=spin, **args)
    with torch.inference_mode():
        inference = get_fd_waveform(mass1=m1.clone(), chi1=spin.clone(), **args)
    for b, value in zip(batch, inference):
        torch.testing.assert_close(b, value, rtol=3e-12, atol=1e-34)
    for i in range(2):
        single = get_fd_waveform(mass1=m1[i], chi1=spin[i], **args)
        for b, value in zip(batch, single):
            torch.testing.assert_close(b[i], value, rtol=3e-11, atol=1e-33)


def test_nonspinning_radiation_rejects_spins_instead_of_silently_dropping_them():
    with pytest.raises(NotImplementedError, match="Nonzero spins"):
        get_fd_waveform("SEOBNR_ODE", mass1=30., mass2=20., chi1=.1,
                        sample_frequencies=torch.tensor([30., 40.], dtype=DTYPE),
                        f_lower=30., radiation_model="din2009_nonspinning", num_steps=2)
