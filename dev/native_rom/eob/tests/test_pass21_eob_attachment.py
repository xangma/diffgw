"""Default SEOBNR_ODE dispatch through the native NQC/C1 attachment."""

import torch
import pytest

import dev.native_rom.eob.torch.waveform as eob_waveform
import dev.native_rom.eob.torch.published_waveform as published_waveform
from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform


PUBLISHED_CHOICES = dict(
    spin_truncation="tabulated",
    spin_v6="bohe_total_energy",
    include_pan_v8=False,
    include_positive_chi_540=False,
    final_spin_fit_order=(1, 2),
)


def _attached_args(**overrides):
    args = dict(
        approximant="SEOBNR_ODE",
        mass1=30.0,
        mass2=20.0,
        sample_frequencies=torch.tensor([30.0, 50.0, 70.0], dtype=torch.float64),
        f_lower=30.0,
        dt_geom=1.0,
        num_steps=2500,
    )
    args.update(overrides)
    return args


def test_default_prototype_dispatches_to_peak_nqc_c1_attachment(monkeypatch):
    calls = {}
    original = eob_waveform.attach_seobnrv4_22

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["result"] = result
        return result

    monkeypatch.setattr(eob_waveform, "attach_seobnrv4_22", wrapped)
    hp, hc = get_fd_waveform(**_attached_args())

    attached = calls["result"]
    assert attached.mode.ndim == 1 and attached.mode.shape[0] > 100
    assert attached.calibration.amplitude_condition.isfinite()
    assert attached.calibration.phase_condition.isfinite()
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()


def test_attached_prototype_keeps_batch_and_mass_gradient():
    mass1 = torch.tensor([30.0, 35.0], dtype=torch.float64, requires_grad=True)
    hp, hc = get_fd_waveform(**_attached_args(mass1=mass1))

    assert hp.shape == (2, 3) and hc.shape == (2, 3)
    gradient = torch.autograd.grad(hp.real.sum(), mass1)[0]
    assert torch.isfinite(gradient).all()


def test_published_assembly_dispatches_all_explicit_choices(monkeypatch):
    calls = {}

    def fake_published(self, f, **kwargs):
        calls.update(kwargs)
        return torch.ones_like(f, dtype=torch.complex128), -torch.ones_like(
            f, dtype=torch.complex128)

    monkeypatch.setattr(published_waveform.SEOBNRv4Published, "__call__", fake_published)
    hp, hc = get_fd_waveform(
        **_attached_args(assembly_model="seobnrv4_published", **PUBLISHED_CHOICES)
    )

    assert torch.equal(hp, torch.ones_like(hp))
    assert torch.equal(hc, -torch.ones_like(hc))
    assert {name: calls[name] for name in PUBLISHED_CHOICES} == PUBLISHED_CHOICES


def test_published_assembly_keeps_its_long_horizon_by_default(monkeypatch):
    calls = {}

    def fake_published(self, f, **kwargs):
        calls.update(kwargs)
        return torch.ones_like(f, dtype=torch.complex128), -torch.ones_like(
            f, dtype=torch.complex128)

    monkeypatch.setattr(published_waveform.SEOBNRv4Published, "__call__", fake_published)
    get_fd_waveform(
        approximant="SEOBNR_ODE",
        mass1=30.0,
        mass2=20.0,
        sample_frequencies=torch.tensor([30.0, 50.0, 70.0], dtype=torch.float64),
        f_lower=30.0,
        assembly_model="seobnrv4_published",
        **PUBLISHED_CHOICES,
    )

    assert calls["num_steps"] == 20000


@pytest.mark.parametrize("missing", tuple(PUBLISHED_CHOICES))
def test_published_assembly_requires_each_choice(missing):
    choices = dict(PUBLISHED_CHOICES)
    choices.pop(missing)
    with pytest.raises(ValueError, match="requires explicit"):
        get_fd_waveform(
            **_attached_args(assembly_model="seobnrv4_published", **choices)
        )


def test_prototype_rejects_published_only_choices():
    with pytest.raises(ValueError, match="published SEOBNRv4 choices"):
        get_fd_waveform(**_attached_args(spin_v6=PUBLISHED_CHOICES["spin_v6"]))
