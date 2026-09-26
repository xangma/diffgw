"""All-mode component gates from the Mishra et al. author supplement.

The frozen values are generated from the checksum-pinned Mathematica file
published with arXiv:1601.05588v1.  The original leading-amplitude magnitude
gate is retained.  A second SymPy gate combines the supplement's signed
leading H[l,m] terms and nonspinning TF28Phasing with the absolute mode
normalization in the accompanying paper.  The paper's polarization map then
checks both projected polarizations and their eight-mode sum.  This is a
symbolic author-source composition gate, not an external numerical series.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2_hm as ref


DATA = Path(__file__).parent / "data" / "taylorf2_hm_mishra2016_samples.json"
MODES = ((2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4))
INCLINATION = 0.91
PHI0 = 0.37


@pytest.fixture(scope="module")
def fixture():
    return json.loads(DATA.read_text())


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The author-supplement gate requires FP64, unsupported on MPS.")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2_hm import TaylorF2_HM

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Requested CUDA qualification, but Torch CUDA is unavailable.")
        yield (
            "torch",
            TaylorF2_HM,
            lambda value: torch.as_tensor(value, dtype=torch.float64, device=reference_device),
        )
        return

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jaxwave.approximants.taylorf2_hm import TaylorF2_HM

    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        pytest.fail(f"Requested {reference_device} qualification is unavailable: {exc}")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.default_device(devices[0]):
            yield "jax", TaylorF2_HM, lambda value: jnp.asarray(value, dtype=jnp.float64)
    finally:
        jax.config.update("jax_enable_x64", previous)


@lru_cache(maxsize=None)
def _paper_projection(mode, inclination=INCLINATION, phi0=PHI0):
    values = {
        ref.IOTA: sp.Rational(str(inclination)),
        ref.PHI0: sp.Rational(str(phi0)),
    }
    return tuple(complex(ref.evaluate(expression, values)) for expression in ref.projection(*mode))


def _carrier(mode_data, mode):
    """Invert the published polarization map without native private helpers."""
    d_plus, d_cross = _paper_projection(mode)
    h_plus, h_cross = (_to_numpy(value) for value in mode_data[mode])
    norm = abs(d_plus) ** 2 + abs(d_cross) ** 2
    return (d_plus.conjugate() * h_plus + d_cross.conjugate() * h_cross) / norm


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _native_waveform(backend, sample, *, f_ref=43.0,
                     inclination=INCLINATION, phi0=PHI0):
    name, model, array = backend
    options = dict(
        f=array([float(value) for value in sample["frequencies_hz"]]),
        mass1=array(float(sample["mass1_msun"])),
        mass2=array(float(sample["mass2_msun"])),
        chi1=array(0.0),
        chi2=array(0.0),
        distance=array(410.0),
        inclination=array(inclination),
        phic=array(0.23),
        tc=array(0.0041),
        f_ref=None if f_ref is None else array(f_ref),
        phi0=array(phi0),
        modes=MODES,
        f_isco_cutoff=False,
        return_modes=True,
    )
    return model()(**options) if name == "torch" else model.evaluate(**options)


def _native_modes(backend, sample, *, f_ref=43.0):
    return _native_waveform(backend, sample, f_ref=f_ref)[2]


@lru_cache(maxsize=None)
def _author_complex_modes(fixture_text, sample_name):
    """Evaluate the checksum-derived author formulas with exact SymPy inputs."""
    fixture = json.loads(fixture_text)
    sample = next(row for row in fixture["samples"] if row["name"] == sample_name)
    constants = fixture["constants"]
    component = fixture["complex_component"]
    r = sp.Rational
    m1, m2 = r(sample["mass1_msun"]), r(sample["mass2_msun"])
    total_mass = m1 + m2
    eta = m1 * m2 / total_mass**2
    delta = (m1 - m2) / total_mass
    mass_seconds = total_mass * r(constants["G_SI"]) * r(constants["MSUN_SI"]) / r(constants["C_SI"]) ** 3
    distance_seconds = r(component["inputs"]["distance_mpc"]) * r(constants["MPC_SI"]) / r(constants["C_SI"])
    phic = r(component["inputs"]["phic"])
    tc = r(component["inputs"]["tc_seconds"])
    eta_symbol, delta_symbol, velocity = sp.symbols("eta delta v")
    tf28 = sp.sympify(
        component["tf28_phasing"],
        locals={"eta": eta_symbol, "v": velocity, "EulerGamma": sp.EulerGamma},
    )
    result = {}
    for mode in MODES:
        l, m = mode
        term = fixture["leading_terms"][f"{l}{m}"]
        power = int(term["velocity_power"])
        coefficient = sp.sympify(
            term["coefficient"],
            locals={"eta": eta_symbol, "delta": delta_symbol},
        ).subs({eta_symbol: eta, delta_symbol: delta})
        values = []
        for frequency_text in sample["frequencies_hz"]:
            frequency = r(frequency_text)
            vm = (2 * sp.pi * mass_seconds * frequency / m) ** r(1, 3)
            normalization = mass_seconds**2 / distance_seconds * sp.pi * sp.sqrt(2 * eta / 3) * vm ** (-r(7, 2))
            orbital_phase = 2 * sp.pi * (frequency / m) * tc - phic + tf28.subs({eta_symbol: eta, velocity: vm})
            author_mode = normalization * coefficient * vm**power * sp.exp(-sp.I * (m * orbital_phase + sp.pi / 4))
            adapter = sp.exp(sp.I * sp.pi * (m - 3) / 2)
            values.append(complex(sp.N(adapter * author_mode, 60)))
        result[mode] = np.asarray(values, dtype=np.complex128)
    return result


def test_author_fixture_provenance_and_completeness(fixture):
    assert fixture["format"] == "taylorf2-hm-mishra2016-leading-amplitude-v1"
    assert fixture["source"]["bundle_sha256"] == ("f0b6c6810da95f45141dbb9f0c5cccec0ba0742f60b836dba7698c90a25d5154")
    assert fixture["source"]["member_sha256"] == ("501ea53c29b77e061585540b9ccfd69a4cd0283e805ea1e9c24c38800e627a70")
    assert fixture["source"]["redistribution_status"] == (
        "terms unresolved: the normalized TF28Phasing is a transformed "
        "author-supplement expression; no downstream MIT grant was found"
    )
    assert tuple(fixture["leading_terms"]) == tuple(f"{l}{m}" for l, m in MODES)
    assert len(fixture["samples"]) == 3
    assert fixture["complex_component"]["format"] == ("mishra2016-leading-nonspinning-complex-modes-v1")
    assert fixture["complex_component"]["evidence_class"] == (
        "symbolic author-source component gate, not external numerical parity"
    )


def test_all_modes_match_author_supplement_leading_amplitudes(backend, fixture):
    for sample in fixture["samples"]:
        mode_data = _native_modes(backend, sample)
        quadrupole = np.abs(_carrier(mode_data, (2, 2)))
        for mode in MODES:
            actual = np.abs(_carrier(mode_data, mode)) / quadrupole
            expected = np.asarray(
                sample["expected_abs_hlm_over_h22"][f"{mode[0]}{mode[1]}"],
                dtype=np.float64,
            )
            np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=2e-15)


def test_all_modes_match_author_supplement_complex_component(backend, fixture):
    """Eq. (11) raw modes: signed LO H_lm, TF28 phase, and normalization."""
    expected_cases = {
        sample["name"]: _author_complex_modes(json.dumps(fixture), sample["name"]) for sample in fixture["samples"]
    }
    for sample in fixture["samples"]:
        mode_data = _native_modes(backend, sample, f_ref=None)
        for mode in MODES:
            actual = _carrier(mode_data, mode)
            expected = expected_cases[sample["name"]][mode]
            relative_error = np.abs(actual - expected) / np.abs(expected)
            assert np.max(relative_error) < 8e-10, (sample["name"], mode, relative_error)


@pytest.mark.parametrize("inclination,phi0", ((0.91, 0.37), (1.32, -0.44)))
def test_author_modes_compose_to_complex_polarizations(
    backend, fixture, inclination, phi0
):
    """Author Eq. (11) modes through García-Quirós Eqs. (8)--(9), nonspinning.

    Compare the complex projected modes and final sum directly: recovering
    the raw modes by inverting the native polarizations would miss an error
    common to the mode and its projection.
    """
    fixture_text = json.dumps(fixture)
    for sample in fixture["samples"]:
        author_modes = _author_complex_modes(fixture_text, sample["name"])
        plus, cross, native_modes = _native_waveform(
            backend, sample, f_ref=None, inclination=inclination, phi0=phi0
        )
        expected_plus = np.zeros(len(sample["frequencies_hz"]), dtype=np.complex128)
        expected_cross = np.zeros_like(expected_plus)
        plus_budget = np.zeros(len(expected_plus))
        cross_budget = np.zeros_like(plus_budget)
        for mode in MODES:
            d_plus, d_cross = _paper_projection(mode, inclination, phi0)
            carrier = author_modes[mode]
            for native, projection, total, budget, label in (
                (native_modes[mode][0], d_plus, expected_plus, plus_budget, "plus"),
                (native_modes[mode][1], d_cross, expected_cross, cross_budget, "cross"),
            ):
                expected = projection * carrier
                error = np.abs(_to_numpy(native) - expected) / np.abs(carrier)
                assert np.max(error) < 8e-10, (sample["name"], mode, label, error)
                total += expected
                budget += np.abs(expected)
        for native, expected, budget, label in (
            (plus, expected_plus, plus_budget, "plus"),
            (cross, expected_cross, cross_budget, "cross"),
        ):
            error = np.abs(_to_numpy(native) - expected) / budget
            assert np.max(error) < 8e-10, (sample["name"], label, error)
