"""Frozen discriminator for Pan's published higher-spin rho43 placement."""

import json
import math
from pathlib import Path

import mpmath as mp
import pytest

from dev.native_rom.reference.flux import factorized_43_amplitude


_PLAN = json.loads(
    (
        Path(__file__).parents[2]
        / "evidence/2026-09-27/flux_43_pan_rho_plan.json"
    ).read_text()
)
_ROWS = [pytest.param(row, id=row["name"]) for row in _PLAN["rows"]]


def _reference(row, branch, digits):
    return factorized_43_amplitude(
        row["m1"],
        row["m2"],
        row["chi1"],
        row["chi2"],
        row["omega"],
        row["omega_zero_radial"],
        row["p_phi"],
        row["hreal_over_M"],
        higher_spin=branch,
        digits=digits,
    )


@pytest.mark.parametrize("row", _ROWS)
@pytest.mark.parametrize("branch", ["taylor_f", "pan_rho"])
@pytest.mark.parametrize("frontend", ["torch", "jax"])
def test_frontends_match_frozen_scalar_branches(frontend, branch, row):
    config = _PLAN["configuration"]
    gates = _PLAN["gates"]
    expected = _reference(row, branch, config["reference_digits"])
    mass = row["m1"] + row["m2"]
    values = [9.0, 0.0, -0.01, row["p_phi"]]
    quantities = {
        "omega": row["omega"],
        "omega_zero_radial": row["omega_zero_radial"],
        "heff_over_mu": row["heff_over_mu"],
        "hreal_over_M": row["hreal_over_M"],
        "x1": row["m1"] / mass,
        "x2": row["m2"] / mass,
        "chi1": row["chi1"],
        "chi2": row["chi2"],
    }
    frontend_branch = (
        config["frontend_pan_rho_branch"] if branch == "pan_rho" else branch
    )
    options = {
        "lmax": config["lmax"],
        "spin_truncation": config["spin_truncation"],
        "retain_pan_cubic_spin": config["retain_pan_cubic_spin"],
        "mode_21_higher_spin": config["mode_21_higher_spin"],
        "mode_31_higher_spin": config["mode_31_higher_spin"],
        "mode_32_spin_square": config["mode_32_spin_square"],
        "mode_33_higher_spin": config["mode_33_higher_spin"],
        "mode_43_higher_spin": frontend_branch,
    }

    if frontend == "torch":
        import torch
        from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux

        state = torch.tensor(values, dtype=torch.float64)
        args = {key: torch.tensor(value, dtype=torch.float64)
                for key, value in quantities.items()}
    else:
        import jax
        import jax.numpy as jnp
        from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux

        jax.config.update("jax_enable_x64", True)
        state = jnp.asarray(values, dtype=jnp.float64)
        args = {key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in quantities.items()}

    actual = float(SEOBNRv4PublishedFlux(**options).amplitudes(state, **args)[4, 3])
    assert math.isfinite(actual) and actual >= 0
    assert actual == pytest.approx(
        float(expected),
        rel=gates["frontend_relative_error_at_most"],
        abs=gates["frontend_absolute_error_at_most"],
    )


@pytest.mark.parametrize("row", _ROWS)
def test_frozen_branch_discriminator_and_precision(row):
    config = _PLAN["configuration"]
    gates = _PLAN["gates"]
    low = {
        branch: _reference(row, branch, config["reference_digits"])
        for branch in config["mode_43_branches"]
    }
    high = {
        branch: _reference(row, branch, config["precision_check_digits"])
        for branch in config["mode_43_branches"]
    }
    precision_gate = mp.mpf(
        str(gates["scalar_50_to_70_digit_relative_error_at_most"])
    )
    for branch in low:
        scale = max(abs(high[branch]), mp.mpf("1e-80"))
        assert abs(low[branch] - high[branch]) / scale <= precision_gate
        assert mp.isfinite(low[branch]) and low[branch] >= 0

    difference = abs(low["pan_rho"] - low["taylor_f"])
    if row["role"] == "discriminator":
        assert difference / abs(low["taylor_f"]) >= mp.mpf(
            str(gates["strong_row_relative_branch_separation_at_least"])
        )
    else:
        assert difference <= mp.mpf(
            str(gates["control_row_absolute_branch_difference_at_most"])
        )


def test_scalar_branch_must_be_explicitly_valid():
    row = _PLAN["rows"][0]
    with pytest.raises(ValueError, match="taylor_f or pan_rho"):
        _reference(row, "historical", _PLAN["configuration"]["reference_digits"])
