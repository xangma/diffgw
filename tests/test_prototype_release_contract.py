"""Release APIs reject quarantined prototype waveforms and component options."""

from importlib import import_module, util

import pytest

import diffgw


@pytest.fixture(params=["torch", "jax"])
def frontend(request):
    library = pytest.importorskip(request.param)
    package = {"torch": "torchwave", "jax": "jaxwave"}[request.param]
    if request.param == "torch":
        array = lambda value: library.tensor(value, dtype=library.float64)
    else:
        library.config.update("jax_enable_x64", True)
        array = import_module("jax.numpy").asarray
    return request.param, package, array


@pytest.mark.parametrize("approximant", ["TEOBResumS", "teobresum", "teob"])
def test_teob_waveform_aliases_are_unavailable(frontend, approximant):
    backend, _, _ = frontend
    with pytest.raises(ValueError, match="Approximant .* not recognized"):
        diffgw.get_fd_waveform(approximant, backend=backend, mass1=30., mass2=20.,
                              f_lower=20., f_final=40., delta_f=10.)


def test_prototype_exports_and_modules_are_absent(frontend):
    _, package, _ = frontend
    for module in (package, f"{package}.approximants"):
        public = import_module(module)
        for name in ("TEOBResumS", "SpinPrecessionODE", "compute_tidal_merger_frequency"):
            assert not hasattr(public, name)
            assert name not in getattr(public, "__all__", ())
    assert util.find_spec(f"{package}.approximants.precession_ode") is None
    if package == "torchwave":
        assert util.find_spec(f"{package}.approximants.eob.teob_conditioned_waveform_clean") is None
    else:
        teob = import_module(f"{package}.approximants.teobresums")
        assert not hasattr(teob, "TEOBResumSRadiationReaction")


@pytest.mark.parametrize("approximant, option", [
    ("IMRPhenomXPHM", {"precession_method": "ode"}),
    ("IMRPhenomXHM", {"experimental_phase_connections": True}),
])
def test_dispatch_rejects_prototype_options(frontend, approximant, option):
    backend, _, _ = frontend
    with pytest.raises(ValueError, match="unvalidated prototype"):
        diffgw.get_fd_waveform(approximant, backend=backend, mass1=30., mass2=20.,
                              f_lower=20., f_final=40., delta_f=10., **option)


def test_direct_xhm_constructor_and_evaluation_reject_experiment(frontend):
    _, package, array = frontend
    model_type = import_module(f"{package}.approximants.imrphenom_xhm").IMRPhenomXHM
    with pytest.raises(ValueError, match="unvalidated prototype"):
        model_type(experimental_phase_connections=True)
    with pytest.raises(ValueError, match="unvalidated prototype"):
        model_type()(array([20., 30.]), mass1=30., mass2=20.,
                     experimental_phase_connections=True)


@pytest.mark.parametrize("option", [
    {"chi1": .1}, {"chi2": -.1}, {"lambda1": 100.}, {"lambda2": 100.},
    {"lambda1_oct": 10.}, {"lambda2_oct": 10.}, {"cq1": 2.}, {"r_lr": 2.},
])
def test_teob_hamiltonian_rejects_unvalidated_sectors(frontend, option):
    _, package, array = frontend
    model_type = import_module(f"{package}.approximants.teobresums").TEOBResumSHamiltonian
    with pytest.raises(ValueError, match="nonspinning vacuum"):
        model_type(array(.24), **{name: array(value) for name, value in option.items()})


def test_teob_hamiltonian_has_no_heuristic_implementations(frontend):
    _, package, array = frontend
    model = import_module(f"{package}.approximants.teobresums").TEOBResumSHamiltonian(array(.24))
    for name in ("gyro_factors", "pade_tidal_factor", "tidal_metric_potential"):
        assert not hasattr(model, name)
    assert float(model.effective_hamiltonian(array(10.), array(-.01), array(3.5))) > 0


def test_torch_teob_radiation_requires_validated_state_context():
    torch = pytest.importorskip("torch")
    from torchwave.approximants.teobresums import TEOBResumSRadiationReaction
    nu, zero = torch.tensor(.24), torch.tensor(0.)
    model = TEOBResumSRadiationReaction(nu, zero, zero)
    assert not hasattr(model, "energy_flux")
    with pytest.raises(ValueError, match="prototype fallbacks are unavailable"):
        model.radiation_reaction_forces(torch.tensor(.02), zero, torch.tensor(3.))
    with pytest.raises(ValueError, match="nonspinning vacuum"):
        TEOBResumSRadiationReaction(nu, torch.tensor(.1), zero)
