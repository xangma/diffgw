"""Published last-digit rounding intervals versus the native NSBH remnant."""

import ast
import inspect
from fractions import Fraction

import numpy as np
import pytest

from diffgw_reference.nsbh_remnant_rounding import (
    CASES,
    FP64_ABSOLUTE_MARGIN,
    MASS_CONSTRAINT,
    MASS_SCALES,
    MASS_TABLE,
    SPIN_CONSTRAINT,
    SPIN_SCALES,
    SPIN_TABLE,
    ZAPPA_MASS_TABLE,
    ZAPPA_SPIN_TABLE,
    _constraint,
    _coeffs,
    bbh_remnant_interval,
    nsbh_remnant_interval,
    printed,
    sqrt_integer,
    zappa_factor_interval,
)


def _native_literals(module):
    tree = ast.parse(inspect.getsource(module.bbh_remnant_uib2016))
    return {
        target.id: ast.literal_eval(statement.value)
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and isinstance(statement.value, (ast.Constant, ast.UnaryOp))
    }


def test_exact_scalar_bounds_and_guard_inactivity():
    for n in (2, 3):
        root = sqrt_integer(n)
        assert root.lower**2 <= n <= root.upper**2
        assert root.width <= Fraction(1, 10**60)
    assert printed("3.833").lower == Fraction(7665, 2000)
    assert printed("-1.83e-3").upper == Fraction(-1825, 10**6)
    for q, chi, lam in CASES:
        mass_bbh, spin_bbh = bbh_remnant_interval(q, chi)
        mass_factor = zappa_factor_interval(q, chi, lam, ZAPPA_MASS_TABLE)
        spin_factor = zappa_factor_interval(q, chi, lam, ZAPPA_SPIN_TABLE)
        mass, spin = nsbh_remnant_interval(q, chi, lam)
        assert mass_bbh.lower > 0 and spin_bbh.lower > 0
        assert 0 < mass_factor.lower <= mass_factor.upper < 1
        assert 0 < spin_factor.lower <= spin_factor.upper < 1
        assert mass.lower > 0 and spin.lower > 0


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_coefficients_round_to_published_tables(backend):
    module = __import__(f"{backend}wave.approximants.imrphenom_nsbh", fromlist=["dummy"])
    literals = _native_literals(module)
    for table, suffix in ((MASS_TABLE, ""), (MASS_SCALES, ""),
                          (SPIN_TABLE, "_s"), (SPIN_SCALES, "_s")):
        for name, paper_decimal in table.items():
            assert printed(paper_decimal).contains(literals[name + suffix]), (backend, name)
    for name, constraint, table in (("f10", MASS_CONSTRAINT, MASS_TABLE),
                                    ("f11_s", SPIN_CONSTRAINT, SPIN_TABLE)):
        assert _constraint(constraint, _coeffs(table)).contains(literals[name]), (backend, name)
    for native, paper in ((module.ZAPPA_MASS_COEFFS, ZAPPA_MASS_TABLE),
                          (module.ZAPPA_SPIN_COEFFS, ZAPPA_SPIN_TABLE)):
        assert len(native) == 12
        for native_value, paper_decimal in zip(native, sum(paper, ()), strict=True):
            assert printed(paper_decimal).contains(native_value), (backend, paper_decimal)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_remnant_within_published_rounding_intervals(backend, reference_device):
    if reference_device != "cpu":
        pytest.fail("NSBH remnant rounding gate is frozen for CPU FP64")
    m_ns = np.full(len(CASES), 1.4, dtype=np.float64)
    m_bh = np.asarray([q * 1.4 for q, _, _ in CASES], dtype=np.float64)
    chi_bh = np.asarray([float(chi) for _, chi, _ in CASES], dtype=np.float64)
    chi_ns = np.zeros_like(chi_bh)
    lam = np.asarray([float(l) for _, _, l in CASES], dtype=np.float64)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_nsbh import bbh_remnant_uib2016, zappa_remnant_properties

        tensors = [torch.as_tensor(x, dtype=torch.float64, device="cpu")
                   for x in (m_bh, m_ns, chi_bh, chi_ns, lam)]
        bbh = tuple(x.detach().cpu().numpy() for x in bbh_remnant_uib2016(*tensors[:4]))
        nsbh = tuple(x.detach().cpu().numpy() for x in zappa_remnant_properties(*tensors))
    else:
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp
            from jaxwave.approximants.imrphenom_nsbh import bbh_remnant_uib2016, zappa_remnant_properties

            with jax.default_device(jax.devices("cpu")[0]):
                arrays = [jnp.asarray(x, dtype=jnp.float64)
                          for x in (m_bh, m_ns, chi_bh, chi_ns, lam)]
                bbh = tuple(np.asarray(x) for x in bbh_remnant_uib2016(*arrays[:4]))
                nsbh = tuple(np.asarray(x) for x in zappa_remnant_properties(*arrays))
        finally:
            jax.config.update("jax_enable_x64", previous)
    for i, (q, chi, tidal) in enumerate(CASES):
        bbh_intervals = bbh_remnant_interval(q, chi)
        nsbh_intervals = nsbh_remnant_interval(q, chi, tidal)
        for actual, bound in zip((bbh[0][i], bbh[1][i]), bbh_intervals, strict=True):
            assert np.isfinite(actual) and bound.contains(float(actual), FP64_ABSOLUTE_MARGIN), (
                backend, (q, chi, tidal), "BBH", actual, float(bound.lower), float(bound.upper))
        for actual, bound in zip((nsbh[0][i], nsbh[1][i]), nsbh_intervals, strict=True):
            assert np.isfinite(actual) and bound.contains(float(actual), FP64_ABSOLUTE_MARGIN), (
                backend, (q, chi, tidal), "NSBH", actual, float(bound.lower), float(bound.upper))
