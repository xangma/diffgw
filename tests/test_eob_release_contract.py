"""Unqualified SEOBNR assemblers must not be part of the release API."""
from importlib import import_module

import pytest


PROTOTYPE_ALIASES = (
    "seobnr", "SEOBNR_ODE", "SEOBNRv5", "SEOBNRv4Published",
    "SEOBNRv4PublishedNonspinning",
)


@pytest.mark.parametrize("frontend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("approximant", PROTOTYPE_ALIASES)
def test_unqualified_seobnr_alias_is_rejected(frontend, approximant):
    pytest.importorskip("torch" if frontend == "torchwave" else "jax")
    module = import_module(frontend)
    with pytest.raises(ValueError, match="not recognized"):
        module.get_fd_waveform(approximant, mass1=30., mass2=20.,
                               sample_frequencies=[20., 40.])


@pytest.mark.parametrize("frontend", ["torchwave", "jaxwave"])
def test_complete_seobnr_assemblers_are_not_exported(frontend):
    pytest.importorskip("torch" if frontend == "torchwave" else "jax")
    for suffix in ("approximants", "approximants.eob",
                   "approximants.eob.waveform", "approximants.eob.published_waveform"):
        module = import_module(f"{frontend}.{suffix}")
        for name in ("SEOBNR_ODE", "SEOBNRv4Published", "SEOBNRv4PublishedNonspinning",
                     "reconstruct_seobnrv4_22"):
            assert not hasattr(module, name), f"{module.__name__}.{name} is development-only"


def test_torch_complete_adaptive_and_conditioned_routes_are_not_exported():
    pytest.importorskip("torch")
    adaptive = import_module("torchwave.approximants.eob.adaptive_driver_clean")
    conditioned = import_module("torchwave.approximants.eob.conditioned_waveform_clean")
    for name in ("reconstruct_seobnrv4_22_adaptive", "reconstruct_seobnrv4_22_numerical_compatibility"):
        assert not hasattr(adaptive, name)
    for name in ("conditioned_seobnrv4_fd", "seobnrv4_conditioned_fd",
                 "conditioned_waveform", "generate_conditioned_seobnrv4_fd"):
        assert not hasattr(conditioned, name)


@pytest.mark.parametrize("frontend", ["torchwave", "jaxwave"])
def test_prototype_physics_classes_are_not_exposed(frontend):
    pytest.importorskip("torch" if frontend == "torchwave" else "jax")
    for suffix in ("approximants.eob", "approximants.eob.hamiltonian",
                   "approximants.eob.radiation_reaction"):
        module = import_module(f"{frontend}.{suffix}")
        assert not hasattr(module, "EOBHamiltonian")
        assert not hasattr(module, "EOBRadiationReaction")
    component = import_module(f"{frontend}.approximants.eob.hamiltonian")
    with pytest.raises(TypeError):
        component.Nonspinning3PNHamiltonian(.2, chi1=.1, chi2=0.)
