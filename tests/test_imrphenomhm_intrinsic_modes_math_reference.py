"""London Eq. (3) complex intrinsic modes against both public frontends."""

import importlib
import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_frequency_map as fmap
from diffgw_reference import imrphenomhm_intrinsic_modes as ref


CASE = (35.0, 20.0, 0.2, -0.1)
MODES = ref.SUPPORTED_MODES


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("This component's frozen FP64 qualification is CPU-only")
    name = request.param
    module = pytest.importorskip(f"{name}wave.approximants.imrphenom_hm")
    if name == "torch":
        torch = pytest.importorskip("torch")

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)

        yield name, module, array
        return

    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        device = jax.devices("cpu")[0]
        with jax.default_device(device):
            def array(value):
                return jax.device_put(jnp.asarray(value, dtype=jnp.float64), device)

            yield name, module, array
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_source_binding_and_symbolic_eq3():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    binding = ref.COMPONENT_BINDINGS[ref.COMPONENT]
    assert callable(getattr(ref, binding["reference"]))
    assert callable(getattr(ref, binding["public_adapter"]))
    for name in ("torch", "jax"):
        module, cls = binding[name]
        assert callable(getattr(importlib.import_module(module), cls))
    assert all(callable(globals()[name]) for name in binding["tests"])
    assert sp.simplify(ref.symbolic_eq3() -
                       sp.Abs(ref.BETA) * ref.A22 *
                       sp.exp(sp.I * (ref.KAPPA * ref.PHI22 + ref.OFFSET))) == 0


def test_scalar_paper_assembly_and_zero():
    with mp.workdps(60):
        kwargs = dict(fr_lm=".08", fr22=".06", transition="london_2018")
        def carrier_amplitude(_):
            return mp.mpf(3)

        def carrier_phase(_):
            return mp.mpf(".7")
        result22 = ref.paper_intrinsic_mode(
            ".01", 2, 2, 3, 2, 0, 0, carrier_amplitude, carrier_phase, **kwargs)
        assert abs(result22 - 3 * mp.exp(mp.j * mp.mpf(".7"))) < mp.mpf("1e-55")

        result33 = ref.paper_intrinsic_mode(
            ".01", 3, 3, 3, 2, 0, 0, carrier_amplitude, carrier_phase, **kwargs)
        from diffgw_reference.imrphenomhm_amplitude_ratio import beta33_leading
        expected_amplitude = 3 * beta33_leading(mp.mpf(".02") / 3, mp.mpf(1) / 5)
        expected_phase = mp.mpf(3) / 2 * mp.mpf(".7") + mp.mpf(7) / 2 * mp.pi
        assert abs(result33 - expected_amplitude * mp.exp(mp.j * expected_phase)) < mp.mpf("1e-55")

        # Eq. (4)'s exact symmetric odd-mode quotient is undefined, while
        # the public waveform applies the exchange-symmetry zero.
        for ell, m in ((2, 1), (3, 3), (4, 3)):
            with pytest.raises(ValueError):
                ref.paper_intrinsic_mode(
                    ".01", ell, m, 2, 2, ".2", ".2",
                    carrier_amplitude, carrier_phase, **kwargs)
            assert ref.public_intrinsic_mode(
                ".01", ell, m, 2, 2, ".2", ".2",
                carrier_amplitude, carrier_phase, **kwargs,
                fd_lm=".01", fd22=".008", base_fr22=".061",
                base_fd22=".0081", alpha4=0, alpha5=1) == 0
        with pytest.raises(ValueError, match="frequency_hz"):
            ref.public_intrinsic_mode(
                ".01", 2, 2, 3, 2, 0, 0,
                carrier_amplitude, carrier_phase, tc=".1", **kwargs,
                fd_lm=".01", fd22=".008", base_fr22=".061",
                base_fd22=".0081", alpha4=0, alpha5=1)


def _d_carrier(name, module, array, geometric_frequencies, public_args, mass_seconds):
    hz = array(np.asarray([float(g) for g in geometric_frequencies]) / mass_seconds)
    if name == "torch":
        hp, _, phase = module.IMRPhenomD()(
            f=hz, return_phase=True, **public_args)
    else:
        hp, _, phase = module.IMRPhenomD.evaluate(
            f=hz, return_phase=True, **public_args)
    normalization = math.sqrt(5.0 / (16.0 * math.pi))
    return np.abs(np.asarray(hp)) / normalization, np.asarray(phase)


def _decimal_query_lookup(samples, query):
    """Match only mpmath decimal roundtrips in the supplied D sample grid."""
    if query in samples:
        return samples[query]
    nearest = min(samples, key=lambda key: abs(key - query))
    if abs(nearest - query) > mp.mpf("1e-12") * abs(query):
        raise KeyError(query)
    return samples[nearest]


@pytest.mark.parametrize("transition", ("london_2018", "phenomd_runtime"))
def test_public_complex_modes(backend, transition):
    name, module, array = backend
    m1, m2, s1, s2 = CASE
    mass_seconds = (m1 + m2) * float(module.MSUN_SEC)
    masses_spins = tuple(array(value) for value in CASE)
    hm_fr22, hm_fd22 = (
        float(value) for value in
        module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *masses_spins)
    )
    _, _, base_fr22, base_fd22 = module.IMRPhenomD.remnant_properties(*masses_spins)
    eta = m1 * m2 / (m1 + m2) ** 2
    chi_pn = module.spin_combinations(*masses_spins)[2]
    alpha4 = float(module.IMRPhenomD.eval_table_v("alpha4", array(eta), chi_pn))
    alpha5 = float(module.IMRPhenomD.eval_table_v("alpha5", array(eta), chi_pn))

    phic, tc = 0.37, 0.008
    public_args = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                       distance=100.0, inclination=0.0, f_ref=None)
    carrier_args = dict(public_args, phic=0.0, tc=0.0)
    hm_args = dict(public_args, phic=phic, tc=tc)
    model = module.IMRPhenomHM(modes=MODES)

    for ell, m in MODES:
        fr_lm, fd_lm = (
            float(value) for value in
            module.IMRPhenomHM.mode_qnm_frequencies(ell, m, *masses_spins)
        )
        f0_amp, f0_phi = (
            float(value) for value in
            fmap.outer_map_cutoffs(transition, fr_lm, hm_fr22)
        )
        step = 1e-7
        x = np.array((0.6 * min(f0_amp, f0_phi),
                      f0_amp - step, f0_amp, f0_amp + step,
                      f0_phi - step, f0_phi, f0_phi + step,
                      0.5 * (max(f0_amp, f0_phi) + fr_lm),
                      fr_lm - step, fr_lm, fr_lm + step, 1.1 * fr_lm))
        assert np.all(x > 0)
        frequencies = array(x / mass_seconds)
        native = np.asarray(model(
            f=frequencies, modes=((ell, m),),
            hm_mapping_transitions=transition,
            hm_21_pn_variant="mishra_eq12b",
            hm_32_pn_variant="mishra_eq12d",
            hm_43_pn_variant="mishra_eq12g",
            return_unprojected_modes=True, **hm_args)[2][(ell, m)])
        assert np.all(np.isfinite(native))

        with mp.workdps(50):
            X = tuple(mp.mpf(str(z)) for z in x)
            FR, Q = mp.mpf(str(fr_lm)), mp.mpf(str(hm_fr22))
            F0A, F0P = fmap.outer_map_cutoffs(transition, FR, Q)
            amp_queries = tuple(
                X if (ell, m) == (2, 2) else
                (fmap.eq5_frequency_map(z, F0A, FR, Q, m) for z in X)
            )
            phase_queries = tuple(
                X if (ell, m) == (2, 2) else
                (2 * F0P / m, Q) + tuple(
                    fmap.eq8_phase_frequency_map(z, F0P, FR, Q, m) for z in X)
            )
            carrier_amp, _ = _d_carrier(
                name, module, array, amp_queries, carrier_args, mass_seconds)
            _, carrier_phase = _d_carrier(
                name, module, array, phase_queries, carrier_args, mass_seconds)
            amp_by_frequency = dict(zip(amp_queries,
                                        (mp.mpf(str(z)) for z in carrier_amp)))
            phase_by_frequency = dict(zip(phase_queries,
                                          (mp.mpf(str(z)) for z in carrier_phase)))
            expected = np.array([complex(ref.public_intrinsic_mode(
                z, ell, m, m1, m2, s1, s2,
                lambda g: _decimal_query_lookup(amp_by_frequency, g),
                lambda g: _decimal_query_lookup(phase_by_frequency, g),
                fr_lm=fr_lm, fr22=hm_fr22, fd_lm=fd_lm, fd22=hm_fd22,
                base_fr22=float(base_fr22), base_fd22=float(base_fd22),
                alpha4=alpha4, alpha5=alpha5, transition=transition,
                phic=phic, tc=tc, frequency_hz=float(z) / mass_seconds,
            )) for z in X])

        # Every complex sample is checked relative to its own full strain
        # scale, so the gate includes the carrier amplitude as well as phase.
        assert np.all(np.abs(expected) > 0)
        assert np.max(np.abs(native - expected) / np.abs(expected)) < 3e-8
