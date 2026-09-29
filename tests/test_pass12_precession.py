"""Pass 12 checks for the versioned precession prescription boundary."""

import pytest
import torch

from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.precession_msa import (
    evaluate_msa_precession,
    resolve_precession_version,
)


DTYPE = torch.float64


@pytest.mark.parametrize(
    ("method", "expected"),
    (("nnlo", 102), ("msa", 223)),
)
def test_wrapper_method_defaults_are_versioned(method, expected):
    assert resolve_precession_version(method, None) == expected


def test_low_level_msa_default_is_explicitly_223():
    common = dict(
        v=torch.tensor([0.16, 0.22], dtype=DTYPE),
        v0=torch.tensor(0.14, dtype=DTYPE),
        m1=torch.tensor(35.0, dtype=DTYPE),
        m2=torch.tensor(25.0, dtype=DTYPE),
        s1x=torch.tensor(0.2, dtype=DTYPE),
        s1y=torch.tensor(0.1, dtype=DTYPE),
        s1z=torch.tensor(0.3, dtype=DTYPE),
        s2x=torch.tensor(-0.1, dtype=DTYPE),
        s2y=torch.tensor(0.05, dtype=DTYPE),
        s2z=torch.tensor(-0.2, dtype=DTYPE),
        iota=torch.tensor(0.7, dtype=DTYPE),
        phi_ref=torch.tensor(0.3, dtype=DTYPE),
    )
    implicit = evaluate_msa_precession(**common)
    explicit = evaluate_msa_precession(**common, prec_version=223)
    for got, want in zip(implicit, explicit):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("model", [IMRPhenomXP(), IMRPhenomXPHM(modes=[(2, 2)])])
def test_unknown_remnant_mode_is_rejected(model):
    with pytest.raises(ValueError, match="final_spin_mod"):
        model(torch.tensor([40.0], dtype=DTYPE), final_spin_mod=99)


@pytest.mark.parametrize("model", [IMRPhenomXP(), IMRPhenomXPHM(modes=[(2, 2)])])
def test_explicit_msa_remnant_mode_is_finite_and_aligned_continuous(model):
    f = torch.tensor([40.0, 80.0], dtype=DTYPE)
    kwargs = dict(mass1=35., mass2=25., spin1x=.2, spin1y=.1, spin1z=.3,
                  spin2x=-.1, spin2y=.05, spin2z=-.2, f_ref=30.)
    out = model(f, **kwargs, prec_version=223, final_spin_mod=3)
    assert all(torch.isfinite(value).all() for value in out[:2])


@pytest.mark.parametrize("model_class,carrier_name,default_msa_remnant", [
    (IMRPhenomXP, "xas_model", True), (IMRPhenomXPHM, "xhm_model", True),
])
def test_explicit_nnlo_remnant_reaches_carrier_with_msa_angles(
    monkeypatch, model_class, carrier_name, default_msa_remnant
):
    """Both public v223 routes use their independent MSA remnant by default."""
    from torchwave.approximants.imrphenom_xas import XASParameterFits
    from torchwave.approximants.precession_msa import compute_effective_chip

    model = model_class()
    carrier = getattr(model, carrier_name)
    original = carrier.forward
    received = []

    def capture(*args, **kwargs):
        received.append(kwargs.get("_final_spin_override"))
        return original(*args, **kwargs)

    monkeypatch.setattr(carrier, "forward", capture)
    f = torch.tensor([25., 80., 180.], dtype=DTYPE)
    options = dict(mass1=40., mass2=20., spin1x=.3, spin1y=.2, spin1z=.1,
                   spin2x=-.2, spin2y=.3, spin2z=-.1, inclination=.7, f_ref=40.)
    values = [torch.tensor(x, dtype=DTYPE) for x in (40., 20., .3, .2, -.2, .3)]
    chip = compute_effective_chip(*values)
    expected = XASParameterFits.fit_final_spin_nnlo(
        torch.tensor(2./9., dtype=DTYPE), torch.tensor(.1, dtype=DTYPE),
        torch.tensor(-.1, dtype=DTYPE), chip,
    )
    for version in (102, 223):
        received.clear()
        default = model(f, **options, prec_version=version)
        assert all(torch.isfinite(value).all() for value in default)
        assert received
        if version == 223:
            if default_msa_remnant:
                for value in received:
                    assert value is not None
                    assert torch.isfinite(value).all()
                    assert not torch.allclose(value, expected, rtol=1e-6, atol=1e-8)
            else:
                assert all(value is None for value in received)
        received.clear()
        explicit = model(f, **options, prec_version=version, final_spin_mod=0)
        assert received
        for value in received:
            torch.testing.assert_close(value, expected, rtol=1e-14, atol=0.)
        if version == 102:
            for actual, prior in zip(explicit, default):
                torch.testing.assert_close(actual, prior, rtol=0., atol=0.)
        else:
            relative_change = torch.linalg.vector_norm(explicit[0] - default[0]) / torch.linalg.vector_norm(default[0])
            assert relative_change > 1e-6
