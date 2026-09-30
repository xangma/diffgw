"""Public HM intrinsic phase against composed London/Khan scalar components."""

import importlib

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from diffgw_reference import imrphenomhm_frequency_map as fmap
from diffgw_reference import imrphenomhm_phase_composition as ref


MODES = ((2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
CASES = ((35.0, 20.0, 0.2, -0.1), (50.0, 10.0, 0.7, -0.3))


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


def test_source_bindings_resolve():
    assert set(ref.COMPONENT_BINDINGS) == set(ref.COMPONENT_SOURCES)
    binding = ref.COMPONENT_BINDINGS[ref.COMPONENT]
    assert callable(getattr(ref, binding["reference"]))
    for name in ("torch", "jax"):
        module, cls = binding[name]
        assert callable(getattr(importlib.import_module(module), cls))
    assert all(callable(globals()[name]) for name in binding["tests"])


def _public_d_phase(name, module, array, geom_frequencies, case, public_args):
    frequencies = array(np.asarray(geom_frequencies, dtype=np.float64) /
                        ((case[0] + case[1]) * float(module.MSUN_SEC)))
    if name == "torch":
        return np.asarray(module.IMRPhenomD()(
            f=frequencies, return_phase=True, **public_args)[2])
    return np.asarray(module.IMRPhenomD.evaluate(
        f=frequencies, return_phase=True, **public_args)[2])


@pytest.mark.parametrize("variant", ("phenomd_runtime", "london_2018"))
@pytest.mark.parametrize("case", CASES)
def test_public_phase_values_and_one_sided_slopes(backend, variant, case):
    name, module, array = backend
    m1, m2, chi1, chi2 = case
    masses_spins = tuple(array(value) for value in case)
    fr22, fd22 = (float(value) for value in
                  module.IMRPhenomHM.mode_qnm_frequencies(2, 2, *masses_spins))
    _, _, base_fr22, base_fd22 = module.IMRPhenomD.remnant_properties(*masses_spins)
    base_fr22, base_fd22 = float(base_fr22), float(base_fd22)
    eta = float(module.component_masses_to_eta(masses_spins[0], masses_spins[1]))
    chi_pn = module.spin_combinations(*masses_spins)[2]
    alpha4 = float(module.IMRPhenomD.eval_table_v("alpha4", array(eta), chi_pn))
    alpha5 = float(module.IMRPhenomD.eval_table_v("alpha5", array(eta), chi_pn))
    m_seconds = (m1 + m2) * float(module.MSUN_SEC)
    public_args = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                       distance=100.0, inclination=0.0, phic=0.0,
                       tc=0.0, f_ref=None)
    h = 1e-7
    model = module.IMRPhenomHM(modes=MODES)

    for ell, m in MODES:
        fr_lm, fd_lm = (float(value) for value in
                        module.IMRPhenomHM.mode_qnm_frequencies(
                            ell, m, *masses_spins))
        _, f0 = (float(value) for value in
                 fmap.outer_map_cutoffs(variant, fr_lm, fr22))
        g0 = 2 * f0 / m
        d_join = base_fr22 / 2
        # Invert London's middle Eq. (8) map to locate the carrier Eq. (14)
        # attachment in the public mode's frequency coordinate.
        assert g0 < d_join < fr22
        mapped_d_join = f0 + (d_join - g0) * (fr_lm - f0) / (fr22 - g0)
        joins = (f0, mapped_d_join, fr_lm)
        x = np.array((0.7 * f0, f0 - h, f0, f0 + h,
                      (f0 + mapped_d_join) / 2,
                      mapped_d_join - h, mapped_d_join,
                      mapped_d_join + h,
                      (mapped_d_join + fr_lm) / 2,
                      fr_lm - h, fr_lm, fr_lm + h, 1.08 * fr_lm))
        assert np.all(x > 0) and f0 + h < mapped_d_join - h
        assert mapped_d_join + h < fr_lm - h

        mode = np.asarray(model(
            f=array(x / m_seconds), hm_mapping_transitions=variant,
            modes=((ell, m),), return_unprojected_modes=True,
            **public_args)[2][(ell, m)])
        assert np.all(np.isfinite(mode)) and np.all(np.abs(mode) > 0)

        with mp.workdps(50):
            X = tuple(mp.mpf(str(value)) for value in x)
            F0, FR, Q = (mp.mpf(str(value)) for value in (f0, fr_lm, fr22))
            queries = (2 * F0 / m, Q) + tuple(
                fmap.eq8_phase_frequency_map(value, F0, FR, Q, m)
                for value in X)
            d_phase = _public_d_phase(name, module, array, queries,
                                      case, public_args)
            carrier_values = dict(zip(queries, (mp.mpf(str(value))
                                                for value in d_phase)))

            def supplied_d_carrier(g):
                return carrier_values[g]

            expected = np.array([float(ref.composed_phase(
                value, F0, FR, Q, ell, m, supplied_d_carrier,
                eta=eta, alpha4=alpha4, alpha5=alpha5,
                fd22=fd22, fd_lm=fd_lm,
                base_fr22=base_fr22, base_fd22=base_fd22))
                for value in X])

        # The public D-carrier contract emits exp(-i phase). For these
        # m1>m2 fixtures, the signed odd-m amplitude factor is positive.
        np.testing.assert_allclose(mode / np.abs(mode), np.exp(-1j * expected),
                                   rtol=2e-9, atol=2e-9)
        for join, lower, center, upper in ((joins[0], 1, 2, 3),
                                           (joins[1], 5, 6, 7),
                                           (joins[2], 9, 10, 11)):
            assert abs(x[center] - join) < 1e-14
            observed_slopes = np.array((
                -np.angle(mode[center] * mode[lower].conjugate()) / h,
                -np.angle(mode[upper] * mode[center].conjugate()) / h,
            ))
            expected_slopes = np.array(((expected[center] - expected[lower]) / h,
                                        (expected[upper] - expected[center]) / h))
            np.testing.assert_allclose(observed_slopes, expected_slopes,
                                       rtol=3e-5, atol=3e-4)
            np.testing.assert_allclose(observed_slopes[0], observed_slopes[1],
                                       rtol=3e-5, atol=3e-4)
