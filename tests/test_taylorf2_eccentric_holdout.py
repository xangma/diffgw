"""Frozen nonspinning TaylorF2_Eccentric binary-oracle holdouts for both backends."""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests import parity_harness as ph
from diffgw_core.constants import MSUN_SEC


_IDS = (
    "TF2Ecc_nonspin_offcatalog_20260927",
    "TF2Ecc_nonspin_offcatalog_second_20260927",
)


@pytest.mark.parametrize("case_id", _IDS)
@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_nonspinning_amplitude_off_strict_lal_holdout(case_id, backend, monkeypatch):
    missing = [name for name in ("lal", "lalsimulation", "pycbc", "jax")
               if importlib.util.find_spec(name) is None]
    if missing:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required binary-oracle dependencies missing: {missing}")
        pytest.skip(f"binary-oracle dependencies missing: {missing}")

    import jax
    import jax.numpy as jnp
    from jaxwave.waveforms import get_fd_waveform as jax_get_fd_waveform

    assert jax.default_backend() == "cpu"
    assert jax.config.read("jax_enable_x64")
    assert ph.validate_case_inventory(ph.get_canonical_holdout_cases(), scope="holdout")[0]
    case = next(case for case in ph.get_canonical_holdout_cases() if case["id"] == case_id)
    receipt = json.loads((Path(__file__).resolve().parents[1] / "docs/validation"
                          / "taylorf2_eccentric_nonspinning_holdout_20260927.json").read_text())
    recorded = next(row for row in receipt["cases"] if row["case"]["id"] == case_id)
    assert recorded["case"] == case
    assert recorded["results"][backend]["status"] == "PASSED"
    assert receipt["gate"] == ph.PARITY_GATE
    assert case["expected_gate"] == ph.PARITY_GATE
    assert case["params"]["chi1"] == case["params"]["chi2"] == 0
    assert 0 < case["params"]["e0"] <= .1
    assert case["params"]["include_amplitude_correction"] is False
    assert case["params"]["ecc_order"] == 6
    assert case["params"]["f0_ecc"] > 0 and case["params"]["f_ref"] > 0
    assert 1 <= case["params"]["mass1"] / case["params"]["mass2"] <= 2
    assert 0 < case["grid"]["flow"] < case["grid"]["fhigh"] < (
        1 / (6**1.5 * math.pi *
             (case["params"]["mass1"] + case["params"]["mass2"]) * MSUN_SEC)
    )
    projection_names = {
        f"{detector}_sky{index}" for detector in ("H1", "L1", "V1")
        for index in (0, 1)
    }
    frozen_projections = recorded["results"][backend]["projections"]
    assert set(frozen_projections) == projection_names
    assert recorded["results"][backend]["projection_count"] == 6
    assert recorded["results"][backend]["plus_l2"] < ph.PARITY_GATE["complex_relative_l2"]
    assert recorded["results"][backend]["cross_l2"] < ph.PARITY_GATE["complex_relative_l2"]
    for projection in frozen_projections.values():
        assert projection["complex_relative_l2"] < ph.PARITY_GATE["complex_relative_l2"]
        assert projection["flat_mismatch"] < ph.PARITY_GATE["mismatch"]
        assert all(value < ph.PARITY_GATE["mismatch"]
                   for value in projection["psd_mismatches"].values())

    if backend == "jax":
        def jax_adapter(approximant, **kwargs):
            frequencies = kwargs.pop("sample_frequencies")
            kwargs.pop("dtype", None)
            hp, hc = jax_get_fd_waveform(
                approximant,
                sample_frequencies=jnp.asarray(frequencies.numpy(), dtype=jnp.float64),
                dtype=jnp.float64, **kwargs,
            )
            return (torch.from_numpy(np.array(hp, dtype=np.complex128, copy=True)),
                    torch.from_numpy(np.array(hc, dtype=np.complex128, copy=True)))

        monkeypatch.setattr(ph, "get_fd_waveform", jax_adapter)

    result = ph.run_oracle_comparison(case, strict=True)
    assert result["status"] == "PASSED", result.get("reason", result)
    assert result["gate"] == ph.PARITY_GATE
    assert set(result["projections"]) == projection_names
