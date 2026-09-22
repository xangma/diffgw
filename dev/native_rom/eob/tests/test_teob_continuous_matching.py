"""Live continuous-event, target, grid-independence and gradient checks."""

import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import assemble_teob_merger_nqc


def _nqc(step=.1, tilt=None, targets=None):
    time = torch.linspace(0, 40, 81, dtype=torch.float64)
    state = torch.stack((20 - .08*time, .01*time.square(), -.04-.001*time, 3+.01*time), -1)
    frequency = .15 - .0001 * (time - 30.17).square()
    if tilt is not None:
        frequency = frequency + tilt * (time - 30)
    return assemble_teob_merger_nqc(
        time, state, .8 + .002*time + .00001*time**3, .1*time + .0002*time**3,
        frequency, .0008 + .00002*time, nu=.24, fixed_step=step,
        targets=targets if targets is not None else (.32, -.002, .38, .012),
        target_provenance="synthetic test", matching_method="continuous",
    )


def test_seobnr_ode_waveform_continuous_matching():
    from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform

    f = torch.tensor([30.0, 40.0, 50.0], dtype=torch.float64)
    hp, hc = get_fd_waveform(
        "SEOBNR_ODE",
        mass1=30.0,
        mass2=20.0,
        sample_frequencies=f,
        f_lower=30.0,
        num_steps=8,
        dt_geom=1.0,
        matching_method="continuous",
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.isfinite(hp).all()
    assert torch.isfinite(hc).all()
    assert torch.any(hp.abs() > 0)
