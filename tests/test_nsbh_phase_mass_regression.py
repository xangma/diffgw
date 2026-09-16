"""Strict NSBH remnant-mass composition regressions against public LAL."""
import copy
import importlib.util
import pytest
import torch

from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from torchwave.approximants.imrphenomd import IMRPhenomD
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases, run_oracle_comparison


def _base(holdout=False):
    cases = get_canonical_holdout_cases() if holdout else get_canonical_catalog_cases()
    return next(c for c in cases if c.get("torch_name") == "IMRPhenomNSBH")


def _case(label, params, holdout=False):
    c = copy.deepcopy(_base(holdout)); c["id"] = "NSBH_mass_regression_" + label
    c["params"].update(params); return c


CASES = []
for label, params in (
    ("catalog", dict(mass1=5., mass2=1.4, chi1=.2, lambda2=400., f_ref=30.)),
    ("holdout", dict(mass1=7., mass2=1.4, chi1=.4, lambda2=2600., f_ref=40.)),
    ("independent_a", dict(mass1=6., mass2=1.3, chi1=.1, lambda2=700., f_ref=35.)),
    ("independent_b", dict(mass1=8., mass2=1.5, chi1=.25, lambda2=1200., f_ref=45.)),
    ("independent_c", dict(mass1=3.2, mass2=1.2, chi1=-.25, lambda2=800., f_ref=32.)),
    ("independent_d", dict(mass1=10., mass2=1.6, chi1=.5, lambda2=500., f_ref=50.)),
    ("independent_e", dict(mass1=4.5, mass2=1.3, chi1=0., lambda2=1800., f_ref=38.)),
    ("independent_f", dict(mass1=9., mass2=1.1, chi1=-.4, lambda2=2000., f_ref=42.)),
    ("neighbor_a", dict(mass1=4., mass2=1.3, chi1=-.2, lambda2=1000., f_ref=35.)),
    ("neighbor_b", dict(mass1=5., mass2=1.4, chi1=-.1, lambda2=1200., f_ref=40.)),
    ("independent_g", dict(mass1=4.2, mass2=1.25, chi1=-.15, lambda2=900., f_ref=34.)),
):
    for tidal in (0., params["lambda2"]):
        p = dict(params, lambda1=0., lambda2=tidal)
        CASES.append(_case(label + ("_lambda0" if tidal == 0 else "_lambda"), p,
                           holdout=(label == "holdout")))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_nsbh_strict_public_binary_parity(case):
    if importlib.util.find_spec("lalsimulation") is None:
        pytest.skip("LALSimulation unavailable")
    try:
        result = run_oracle_comparison(case, strict=True)
    except ImportError as exc:
        pytest.fail(f"unexpected import failure with LALSimulation installed: {exc}")
    assert result["status"] == "PASSED", result


def test_nsbh_complete_waveform_batch_matches_scalars_and_gradients():
    model = IMRPhenomNSBH()
    f = torch.linspace(30., 900., 64, dtype=torch.float64)
    m1 = torch.tensor([5., 7., 4.], dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor([1.4, 1.4, 1.3], dtype=torch.float64)
    chi = torch.tensor([.2, .4, -.2], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor([400., 2600., 1000.], dtype=torch.float64, requires_grad=True)
    hp, hc = model(f, m1, m2, chi, lambda2=lam, inclination=.7, f_ref=40.)
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    for i in range(3):
        sp = model(f, m1[i], m2[i], chi[i], lambda2=lam[i], inclination=.7, f_ref=40.)
        torch.testing.assert_close(hp[i], sp[0], rtol=2e-9, atol=1e-35)
        torch.testing.assert_close(hc[i], sp[1], rtol=2e-9, atol=1e-35)
    hp_scale = hp.abs().detach().clamp_min(1e-30)
    hc_scale = hc.abs().detach().clamp_min(1e-30)
    phase_sensitive = ((hp.real + hp.imag) / hp_scale +
                       (hc.real - hc.imag) / hc_scale).sum()
    grads = torch.autograd.grad(phase_sensitive, (m1, chi, lam))
    assert all(torch.isfinite(g).all() for g in grads)


def test_phenomd_default_and_identity_provider_are_exactly_equivalent():
    f = torch.linspace(30., 900., 32, dtype=torch.float64)
    args = (f, torch.tensor(8., dtype=torch.float64), torch.tensor(1.5, dtype=torch.float64),
            torch.tensor(.25, dtype=torch.float64), torch.tensor(0., dtype=torch.float64))
    model = IMRPhenomD()
    default = model(*args, f_ref=45., return_phase=True)
    explicit = model(*args, f_ref=45., return_phase=True,
                     remnant_provider=IMRPhenomD.remnant_properties)
    for a, b in zip(default, explicit):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    slope_default = IMRPhenomD.compute_slope_peak(*args[1:])
    slope_explicit = IMRPhenomD.compute_slope_peak(
        *args[1:], remnant_provider=IMRPhenomD.remnant_properties)
    torch.testing.assert_close(slope_default, slope_explicit, rtol=0, atol=0)
