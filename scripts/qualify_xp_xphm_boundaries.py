#!/usr/bin/env python3
"""Emit strict XP/XPHM boundary-qualification residuals as JSON.

This is a clean-room verification driver.  LALSimulation is used only through
its installed public binary API as an independent numerical oracle.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP
from tests.parity_harness import (
    get_canonical_holdout_cases,
    get_canonical_model_cases,
    run_oracle_comparison,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP


GATE = {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8}


def metrics(actual, reference):
    """Return the direct, unshifted full-complex-waveform metrics."""
    actual = np.asarray(actual, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    reference_norm = np.linalg.norm(reference)
    actual_norm = np.linalg.norm(actual)
    relative_l2 = np.linalg.norm(actual - reference) / reference_norm
    overlap = np.real(np.vdot(actual, reference)) / (actual_norm * reference_norm)
    mismatch = 1.0 - min(1.0, float(overlap))
    finite = bool(np.isfinite(relative_l2) and np.isfinite(mismatch))
    return {
        "complex_relative_l2": float(relative_l2) if finite else None,
        "mismatch": float(mismatch) if finite else None,
        "finite": finite,
        "pass": bool(
            finite
            and relative_l2 <= GATE["complex_relative_l2"]
            and 0.0 <= mismatch <= GATE["mismatch"]
        ),
    }


def pair_metrics(actual, reference):
    return {
        name: metrics(left, right)
        for name, left, right in zip(("plus", "cross"), actual, reference)
    }


def torch_waveform(frequencies, params, *, force_legacy=False):
    kwargs = dict(params)
    if force_legacy:
        kwargs["_force_legacy"] = True
        if kwargs["mass1"] == kwargs["mass2"]:
            kwargs["final_spin_mod"] = 0
    result = TorchXP()(torch.as_tensor(frequencies, dtype=torch.float64), **kwargs)
    return tuple(value.detach().cpu().numpy() for value in result)


def jax_waveform(frequencies, params, *, force_legacy=False):
    kwargs = dict(params)
    if force_legacy:
        kwargs["_force_legacy"] = True
        if kwargs["mass1"] == kwargs["mass2"]:
            kwargs["final_spin_mod"] = 0
    result = JAXXP.evaluate(jnp.asarray(frequencies), **kwargs)
    return tuple(np.asarray(value) for value in result)


def native_boundary(name, frequencies, params):
    def attempt(call):
        try:
            return call(), None
        except Exception as error:  # qualification must preserve baseline failures
            return None, f"{type(error).__name__}: {error}"

    torch_actual, torch_error = attempt(
        lambda: torch_waveform(frequencies, params)
    )
    torch_legacy, torch_legacy_error = attempt(
        lambda: torch_waveform(frequencies, params, force_legacy=True)
    )
    jax_actual, jax_error = attempt(lambda: jax_waveform(frequencies, params))
    jax_legacy, jax_legacy_error = attempt(
        lambda: jax_waveform(frequencies, params, force_legacy=True)
    )

    def compare(left, right, left_error, right_error):
        if left is None or right is None:
            return {"left_error": left_error, "right_error": right_error}
        return pair_metrics(left, right)

    return {
        "name": name,
        "torch_vs_jax": compare(
            torch_actual, jax_actual, torch_error, jax_error
        ),
        "torch_vs_forced_legacy": compare(
            torch_actual, torch_legacy, torch_error, torch_legacy_error
        ),
        "jax_vs_forced_legacy": compare(
            jax_actual, jax_legacy, jax_error, jax_legacy_error
        ),
    }


def compact_oracle(result):
    compact = {
        "case": result["case_id"],
        "status": result["status"],
        "reference_kind": result.get("reference_kind"),
        "gate": result["gate"],
    }
    if "plus" not in result:
        compact["reason"] = result.get("reason")
        return compact
    for polarization in ("plus", "cross"):
        source = result[polarization]
        values = {
            "complex_relative_l2": source["complex_relative_l2"],
            "mismatch": source["flat_mismatch"],
        }
        values["pass"] = bool(
            values["complex_relative_l2"] <= result["gate"]["complex_relative_l2"]
            and 0.0 <= values["mismatch"] <= result["gate"]["mismatch"]
        )
        compact[polarization] = values
    return compact


def oracle_case(case_id, params, flow, fhigh):
    template = next(
        case for case in get_canonical_model_cases()
        if case["torch_name"] == "IMRPhenomXP"
    )
    case = copy.deepcopy(template)
    case["id"] = case_id
    case["params"].pop("chi1", None)
    case["params"].pop("chi2", None)
    case["params"].update(params)
    case["grid"] = {"flow": flow, "fhigh": fhigh, "df": 0.25}
    return compact_oracle(run_oracle_comparison(case, strict=True))


def xphm_oracles():
    cases = get_canonical_model_cases() + get_canonical_holdout_cases()
    return [
        compact_oracle(run_oracle_comparison(case, strict=True))
        for case in cases
        if case["torch_name"] == "IMRPhenomXPHM"
    ]


def main():
    jax.config.update("jax_enable_x64", True)
    common = {
        "distance": 150.0,
        "inclination": 0.7,
        "phic": 0.3,
        "tc": 0.0,
        "f_ref": 20.0,
        "prec_version": 223,
        "convention": 1,
        "final_spin_mod": 3,
    }
    spins = {
        "spin1x": 0.2,
        "spin1y": 0.1,
        "spin1z": 0.3,
        "spin2x": -0.1,
        "spin2y": 0.05,
        "spin2z": -0.2,
    }
    full_grid = np.arange(20.0, 512.0, 0.25)
    equal = dict(common, **spins, mass1=25.0, mass2=25.0, phi0=0.0)
    azimuth = dict(common, **spins, mass1=35.0, mass2=25.0, phi0=0.4)

    native = [
        native_boundary("generic_equal_mass", full_grid, equal),
        native_boundary("generic_nonzero_phi0", full_grid, azimuth),
    ]

    near_aligned = []
    oracle = [oracle_case("XP_generic_equal_mass", equal, 20.0, 512.0)]
    for label, fhigh in (("before_instability", 55.0), ("crosses_instability", 300.0)):
        frequencies = np.linspace(25.0, fhigh, 128)
        base = {
            "mass1": 36.0,
            "mass2": 22.0,
            "distance": 150.0,
            "inclination": 0.65,
            "phic": 0.3,
            "tc": 0.0,
            "f_ref": 40.0,
            "phi0": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.35,
            "spin2y": 0.0,
            "spin2z": -0.25,
            "prec_version": 223,
            "convention": 1,
            "final_spin_mod": 3,
        }
        tiny_params = dict(base, spin1x=1.0e-10, spin2x=-5.0e-11)
        aligned_params = dict(base, spin1x=0.0, spin2x=0.0)
        torch_tiny = torch_waveform(frequencies, tiny_params)
        torch_aligned = torch_waveform(frequencies, aligned_params)
        jax_tiny = jax_waveform(frequencies, tiny_params)
        near_aligned.append({
            "name": label,
            "frequency_band_hz": [25.0, fhigh],
            "torch_vs_jax": pair_metrics(torch_tiny, jax_tiny),
            "tiny_vs_exact_aligned": pair_metrics(torch_tiny, torch_aligned),
        })
        oracle.append(oracle_case(f"XP_near_aligned_{label}", tiny_params, 25.0, fhigh))

    print(json.dumps({
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "gate": GATE,
        "native": native,
        "near_aligned": near_aligned,
        "lal_binary_oracle": oracle + [{
            "case": "XP_generic_nonzero_phi0",
            "status": "NOT_EVALUABLE",
            "reference_kind": "lal_binary",
            "reason": (
                "No independently derived mapping from diffGW observer phi0 "
                "to the public LAL binary parameters is available."
            ),
        }],
        "xphm_lal_binary_oracle": xphm_oracles(),
    }, indent=2))


if __name__ == "__main__":
    main()
