"""Paper phase, continuity, and unwrapped Torch/JAX IMRPhenomD phase."""

import importlib

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomd_phase as ref
from diffgw.waveforms import get_waveform


CASES = ((30., 30., 0., 0.), (45., 15., .45, -.25))


def test_phase_component_bindings_are_complete_and_resolve():
    assert set(ref.COMPONENT_IDS.values()) == set(ref.COMPONENT_SOURCES)
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    components = {component.id: component for component in get_waveform("IMRPhenomD").components}
    for component_id, binding in ref.COMPONENT_BINDINGS.items():
        assert callable(getattr(ref, binding["reference"]))
        if component_id == ref.COMPONENT_IDS["beta1"]:
            assert components[component_id].implementation == "reference_only"
            assert not {"torch", "jax"}.intersection(binding)
        else:
            assert components[component_id].implementation == "native_and_reference"
        for frontend in ("torch", "jax"):
            if frontend not in binding:
                continue
            module_name, name = binding[frontend]
            target = importlib.import_module(module_name)
            for part in name.split("."):
                target = getattr(target, part)
            assert callable(target)
        assert binding["tests"]
        assert all(callable(globals()[name]) for name in binding["tests"])


@pytest.mark.parametrize("option", ("phasing_convention", "reference_frame"))
def test_public_dispatch_rejects_unknown_phase_option(backend, option):
    name, _, _, array = backend
    waveform = importlib.import_module(f"{name}wave.waveforms")
    with pytest.raises(ValueError, match="unsupported IMRPhenomD phasing convention"):
        waveform.get_fd_waveform(
            approximant="IMRPhenomD", sample_frequencies=array([20., 30.]),
            mass1=array(30.), mass2=array(30.), chi1=array(0.), chi2=array(0.),
            distance=array(400.), inclination=array(.7), **{option: "unknown"},
        )


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The phase reference requires FP64")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.taylorf2 import TaylorF2
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        yield "torch", IMRPhenomD, TaylorF2, lambda v: torch.as_tensor(
            v, dtype=torch.float64, device=reference_device)
        return
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.taylorf2 import TaylorF2
        platform = "gpu" if reference_device == "cuda" else "cpu"
        devices = jax.devices(platform)
        with jax.default_device(devices[0]):
            yield "jax", IMRPhenomD, TaylorF2, lambda v: jax.device_put(
                jnp.asarray(v, dtype=jnp.float64), devices[0])
    finally:
        jax.config.update("jax_enable_x64", previous)


def _coordinates(case):
    return ref.spin_coordinates(*(ref.exact(x) for x in case))


def _pieces(case):
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    eta, delta, chi_s, chi_a, chi_pn = _coordinates(case)
    fits = {name: ref.fit_expr(name, eta, chi_pn,
            coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name]))
            for name in ref.MODEL_FIT_NAMES}
    f_rd, f_damp = ref.exact(.09), ref.exact(.014)
    pieces = ref.phase_pieces(eta=eta,delta=delta,chi_s=chi_s,chi_a=chi_a,
                              f_rd=f_rd,f_damp=f_damp,
                              **{name: fits[name] for name in
                                 ("sigma1","sigma2","sigma3","sigma4","beta2","beta3",
                                  "alpha2","alpha3","alpha4","alpha5")})
    return pieces, fits, (eta, f_rd, f_damp)


def test_published_phase_rows_are_pinned_to_table_5():
    from torchwave.phenomd_published_fits import PARAMETER_NAMES, TABLE_V_ROWS
    rows = dict(zip((x.replace("_", "") for x in PARAMETER_NAMES), TABLE_V_ROWS))
    assert len(ref.FIT_NAMES) == 12
    assert all(len(ref.PUBLISHED_ROWS[name]) == 11 for name in ref.FIT_NAMES)
    assert all(ref.PUBLISHED_ROWS[name] == rows[name] for name in ref.FIT_NAMES)
    assert "beta1" not in ref.MODEL_FIT_NAMES  # continuity fixes it


@pytest.mark.parametrize("case", CASES)
def test_paper_pn_coefficients_against_both_frontends(backend, case):
    _, _, TaylorF2, array = backend
    eta, delta, chi_s, chi_a, _ = _coordinates(case)
    expected = [float(ref.evaluate(x)) for x in
                ref.pn_phase_coefficients(eta,delta,chi_s,chi_a)]
    actual = TaylorF2.phase_coefficients(*(array(x) for x in case),
                                          phenom_d_calibration=True)[:9]
    np.testing.assert_allclose([float(x) for x in actual], expected,
                               rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("case", CASES)
def test_phase_fits_match_both_frontends_with_explicit_inherited_rows(backend, case):
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    _, model, _, array = backend
    eta, _, _, _, chi_pn = _coordinates(case)
    for name in ref.MODEL_FIT_NAMES:
        expected = float(ref.evaluate(ref.fit_expr(name,eta,chi_pn,
            coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS[name]))))
        actual = float(model.eval_table_v(name,array(float(eta)),array(float(chi_pn))))
        np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-11,
                                   err_msg=name)


def test_integrated_branches_and_c1_matching():
    eta, f_rd, f_damp = sp.Rational(1,4), sp.Rational(9,100), sp.Rational(14,1000)
    middle_shape = ref.intermediate_shape(beta2=sp.Rational(-3,2),
                                           beta3=sp.Rational(1,100000))
    assert sp.simplify(sp.diff(middle_shape,ref.U)
                       +sp.Rational(3,2)/ref.U-sp.Rational(1,100000)/ref.U**4) == 0
    mr_shape = ref.merger_ringdown_shape(f_rd=f_rd,f_damp=f_damp,
        alpha2=sp.Rational(-1,10),alpha3=sp.Rational(2),
        alpha4=sp.Rational(1,2),alpha5=sp.Rational(1))
    expected_mr_derivative = (sp.Rational(-1,10)/ref.U**2
        +2*ref.U**sp.Rational(-1,4)
        +sp.Rational(1,2)*f_damp/(f_damp**2+(ref.U-f_rd)**2))
    assert sp.simplify(sp.diff(mr_shape,ref.U)-expected_mr_derivative) == 0
    pieces = ref.phase_pieces(eta=eta,delta=0,chi_s=0,chi_a=0,
        f_rd=f_rd,f_damp=f_damp,sigma1=10,sigma2=-3,sigma3=2,sigma4=-1,
        beta2=sp.Rational(-3,2),beta3=sp.Rational(1,100000),
        alpha2=sp.Rational(-1,10),alpha3=2,alpha4=sp.Rational(1,2),alpha5=1)
    for frequency,left,right in ((pieces["f1"],"inspiral","intermediate"),
                                 (pieces["f2"],"intermediate","merger_ringdown")):
        for order in (0,1):
            difference = sp.diff(pieces[left]-pieces[right],ref.U,order).subs(ref.U,frequency)
            assert abs(float(ref.evaluate(difference))) < 1e-35


@pytest.mark.parametrize("case", CASES)
def test_unwrapped_coalescence_phase_across_regions(backend, case):
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    name, model, _, array = backend
    pieces, fits, (eta, f_rd, f_damp) = _pieces(case)
    f1, f2 = float(pieces["f1"]), float(pieces["f2"])
    us = np.array([.01, f1-1e-7, f1, f1+1e-7,
                   (f1+f2)/2, f2-1e-7, f2, f2+1e-7, .07, .10])
    m_sec = sum(case[:2])*MSUN_SEC
    def remnant_provider(*_):
        return array(sum(case[:2])),array(0),array(float(f_rd)),array(float(f_damp))
    kwargs = dict(f=array(us/m_sec),mass1=array(case[0]),mass2=array(case[1]),
                  chi1=array(case[2]),chi2=array(case[3]),distance=array(400),
                  inclination=array(.7),remnant_provider=remnant_provider,
                  phasing_convention="coalescence",return_phase=True)
    phase = model()(**kwargs)[2] if name == "torch" else model.evaluate(**kwargs)[2]
    actual = np.asarray(phase.detach().cpu() if name == "torch" else phase)
    from diffgw_reference import imrphenomd_amplitude as amp
    gamma2 = amp.fit_expr("gamma2",eta,_coordinates(case)[-1],
                         coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma2"]))
    gamma3 = amp.fit_expr("gamma3",eta,_coordinates(case)[-1],
                         coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma3"]))
    peak = amp.peak_compatibility(f_rd,f_damp,gamma2,gamma3)
    gauge = ref.peak_gauge_slope(eta=eta,alpha1_raw=fits["alpha1"],
        alpha2=fits["alpha2"],alpha3=fits["alpha3"],alpha4=fits["alpha4"],
        alpha5=fits["alpha5"],f_peak=peak,f_rd=f_rd,f_damp=f_damp)
    expected = []
    for u in us:
        branch = ("inspiral" if u <= f1 else
                  "intermediate" if u <= f2 else "merger_ringdown")
        expression = -sp.pi/4+pieces[branch]-gauge*ref.U
        expected.append(float(ref.evaluate(expression,{ref.U:u})))
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-8)


@pytest.mark.parametrize("case", CASES)
def test_reference_frequency_phase_anchor(backend, case):
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    from diffgw_reference import imrphenomd_amplitude as amp

    name, model, _, array = backend
    pieces, fits, (eta, f_rd, f_damp) = _pieces(case)
    f1, f2 = float(pieces["f1"]), float(pieces["f2"])
    us = np.array([.01, .024, .045, .08])
    u_ref = .024
    m_sec = sum(case[:2])*MSUN_SEC
    def remnant_provider(*_):
        return array(sum(case[:2])),array(0),array(float(f_rd)),array(float(f_damp))
    kwargs = dict(f=array(us/m_sec),mass1=array(case[0]),mass2=array(case[1]),
                  chi1=array(case[2]),chi2=array(case[3]),distance=array(400),
                  inclination=array(.7),remnant_provider=remnant_provider,
                  f_ref=array(u_ref/m_sec),return_phase=True)
    phase = model()(**kwargs)[2] if name == "torch" else model.evaluate(**kwargs)[2]
    actual = np.asarray(phase.detach().cpu() if name == "torch" else phase)
    chi_pn = _coordinates(case)[-1]
    gamma2 = amp.fit_expr("gamma2",eta,chi_pn,
        coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma2"]))
    gamma3 = amp.fit_expr("gamma3",eta,chi_pn,
        coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma3"]))
    peak = amp.peak_compatibility(f_rd,f_damp,gamma2,gamma3)
    gauge = ref.peak_gauge_slope(eta=eta,alpha1_raw=fits["alpha1"],
        alpha2=fits["alpha2"],alpha3=fits["alpha3"],alpha4=fits["alpha4"],
        alpha5=fits["alpha5"],f_peak=peak,f_rd=f_rd,f_damp=f_damp)
    ref_branch = ("inspiral" if u_ref <= f1 else
                  "intermediate" if u_ref <= f2 else "merger_ringdown")
    expected = []
    for u in us:
        branch = ("inspiral" if u <= f1 else
                  "intermediate" if u <= f2 else "merger_ringdown")
        expression = (pieces[branch].subs(ref.U,u)
                      -pieces[ref_branch].subs(ref.U,u_ref)-gauge*(u-u_ref))
        expected.append(float(ref.evaluate(expression)))
    np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-8)
    assert abs(actual[1]) < 2e-8


@pytest.mark.parametrize("case", CASES)
def test_phase_frequency_gradient_across_regions(backend, case):
    from diffgw_core.constants import MSUN_SEC
    from diffgw_core.fits import TABLE_V_UNROUNDED_COEFFS
    from diffgw_reference import imrphenomd_amplitude as amp

    name, model, _, array = backend
    pieces, fits, (eta, f_rd, f_damp) = _pieces(case)
    f1, f2 = float(pieces["f1"]), float(pieces["f2"])
    chi_pn = _coordinates(case)[-1]
    gamma2 = amp.fit_expr("gamma2",eta,chi_pn,
        coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma2"]))
    gamma3 = amp.fit_expr("gamma3",eta,chi_pn,
        coefficients=tuple(str(x) for x in TABLE_V_UNROUNDED_COEFFS["gamma3"]))
    peak = amp.peak_compatibility(f_rd,f_damp,gamma2,gamma3)
    gauge = ref.peak_gauge_slope(eta=eta,alpha1_raw=fits["alpha1"],
        alpha2=fits["alpha2"],alpha3=fits["alpha3"],alpha4=fits["alpha4"],
        alpha5=fits["alpha5"],f_peak=peak,f_rd=f_rd,f_damp=f_damp)
    m_sec = sum(case[:2])*MSUN_SEC
    def remnant_provider(*_):
        return array(sum(case[:2])),array(0),array(float(f_rd)),array(float(f_damp))
    kwargs = dict(mass1=array(case[0]),mass2=array(case[1]),
                  chi1=array(case[2]),chi2=array(case[3]),distance=array(400),
                  inclination=array(.7),remnant_provider=remnant_provider,
                  phasing_convention="coalescence",return_phase=True)
    for u, branch in ((.012,"inspiral"),(.028,"intermediate"),(.08,"merger_ringdown")):
        assert (u <= f1 if branch == "inspiral" else
                f1 < u <= f2 if branch == "intermediate" else u > f2)
        if name == "torch":
            import torch
            frequency = array([u/m_sec]).requires_grad_(True)
            phase = model()(f=frequency,**kwargs)[2]
            actual = float(torch.autograd.grad(phase.sum(),frequency)[0][0])
        else:
            import jax
            import jax.numpy as jnp
            actual = float(jax.grad(lambda frequency: model.evaluate(
                f=jnp.expand_dims(frequency,0),**kwargs)[2].sum())(array(u/m_sec)))
        expected = float(ref.evaluate(
            m_sec*(sp.diff(pieces[branch],ref.U)-gauge),{ref.U:u}))
        np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=2e-7)
