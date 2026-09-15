"""Independent diagnostics for the clean-room nonspinning TEOB radiation.

The checks exercise algebraic contracts from 1212.4357 and 1806.01772 without
using a waveform oracle: odd-l=5 mass exchange, the finite-mass residual-phase
truncation, horizon leading powers/source factors, flux additivity, and Torch
gradient/dtype behavior.  Results are emitted as JSON for review artifacts.
"""

from __future__ import annotations

import json
import math

import torch

from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    _residual_phase,
    _residual_phase_taylor,
)


def _state(*, x: float, m1: float = 0.6, m2: float = 0.4,
           dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    radius = torch.tensor(1.0, dtype=dtype)
    omega = torch.sqrt(torch.tensor(x, dtype=dtype))
    return {
        "r": radius,
        "Omega": omega,
        "phi": torch.tensor(0.17, dtype=dtype),
        "prstar": torch.tensor(-0.03, dtype=dtype),
        "pphi": omega.clone(),
        "Heff": torch.tensor(1.0, dtype=dtype),
        "H": torch.tensor(1.0, dtype=dtype),
        "Hcirc": torch.tensor(1.0, dtype=dtype),
        "Aprime": torch.tensor(2.0, dtype=dtype),
        "m1": torch.tensor(m1, dtype=dtype),
        "m2": torch.tensor(m2, dtype=dtype),
    }


def _max_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    magnitude = b.abs()
    scale = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    return float(((a - b).abs() / scale).max())


def main() -> None:
    radiation = TEOBRadiationClean(8)
    unequal = _state(x=0.02)
    equal = _state(x=0.02, m1=0.5, m2=0.5)
    l5 = radiation.mode(5, 1, **unequal)
    l5_equal = radiation.mode(5, 1, **equal)
    exchanged = radiation.mode(5, 1, **dict(unequal, m1=unequal["m2"], m2=unequal["m1"]))

    d1, d2 = 1e-4, 2e-4
    near1 = _state(x=0.02, m1=0.5 + d1 / 2, m2=0.5 - d1 / 2)
    near2 = _state(x=0.02, m1=0.5 + d2 / 2, m2=0.5 - d2 / 2)
    linear_ratio = (
        radiation.mode(5, 1, **near2).abs() / d2
    ) / (radiation.mode(5, 1, **near1).abs() / d1)

    y = torch.tensor(0.04, dtype=torch.float64)
    nu = torch.tensor(0.24, dtype=torch.float64)
    phase_v9_residuals = {}
    coefficients = {
        (3, 2): 208 / 63 * math.pi**2 - 9112 / 405,
        (4, 2): 25136 / 10395 * math.pi**2 - 6893 / 375,
        (4, 4): 201088 / 10395 * math.pi**2 - 55144 / 375,
    }
    for mode, coefficient in coefficients.items():
        expected = _residual_phase_taylor(mode, y, nu) - coefficient * torch.sqrt(y)**9
        phase_v9_residuals[str(mode)] = float((_residual_phase(mode, y, nu) - expected).abs())

    horizon = TEOBRadiationClean(2)
    leading_state = _state(x=1e-4)
    leading_nu = leading_state["m1"] * leading_state["m2"]
    scale = (32 / 5) * leading_nu**2 * leading_state["Omega"]**5
    leading = (1 - 4 * leading_nu + 2 * leading_nu**2) * leading_state["Omega"]**8
    leading_ratio = horizon.horizon_flux(**leading_state) / scale / leading

    source_state = _state(x=0.01)
    baseline = horizon.horizon_flux(**dict(source_state, pphi=torch.tensor(0.0, dtype=torch.float64)))
    source_one = horizon.horizon_flux(**dict(source_state, pphi=torch.tensor(1.0, dtype=torch.float64)))
    source_two = horizon.horizon_flux(**dict(source_state, pphi=torch.tensor(2.0, dtype=torch.float64)))
    source_residual = (source_two - baseline) - 4 * (source_one - baseline)

    infinity = horizon.flux(**source_state)
    absorbed = horizon.horizon_flux(**source_state)
    total = horizon.total_flux(**source_state)
    torque = horizon.radiation_reaction_torque(**source_state)
    total_residual = total - infinity - absorbed
    torque_residual = torque + total / (source_state["m1"] * source_state["m2"])

    differentiable_omega = source_state["Omega"].clone().requires_grad_(True)
    differentiated = horizon.horizon_flux(**dict(source_state, Omega=differentiable_omega))
    gradient = torch.autograd.grad(differentiated, differentiable_omega)[0]
    source_state32 = _state(x=0.01, dtype=torch.float32)
    float32_relative = _max_relative(
        horizon.horizon_flux(**source_state32).double(), horizon.horizon_flux(**source_state)
    )

    result = {
        "l5_equal_mass_abs": float(l5_equal.abs()),
        "l5_linear_ratio_d2_over_d1": float(linear_ratio),
        "l5_exchange_relative": _max_relative(exchanged, -l5),
        "phase_v9_residuals": phase_v9_residuals,
        "horizon_leading_ratio": float(leading_ratio),
        "horizon_source_quadratic_residual": float(source_residual.abs()),
        "horizon_exchange_relative": _max_relative(
            horizon.horizon_flux(**unequal), horizon.horizon_flux(
                **dict(unequal, m1=unequal["m2"], m2=unequal["m1"])
            )
        ),
        "horizon_total_additivity_residual": float(total_residual.abs()),
        "horizon_torque_contract_residual": float(torque_residual.abs()),
        "horizon_gradient": float(gradient),
        "horizon_float32_relative": float(float32_relative),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
