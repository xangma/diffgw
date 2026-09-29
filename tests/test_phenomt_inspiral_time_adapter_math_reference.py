"""Source-backed scalar time inversion and public adapter gate for PhenomT 22."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from diffgw_reference.phenomt_inspiral_time_adapter import (
    InspiralCarrier, fixed_grid, lower_frequency_mask, reference_frequency,
)


ROWS = json.loads(
    (Path(__file__).parent / "data/phenomt_author_inspiral_fits_4121.json").read_text()
)["cases"]
CASES = [row for row in ROWS if tuple(row["inputs"]) in (
    (30, 30, 0, 0), (35, 25, 0.7, -0.5), (60, 6, 0.9, 0.1),
)]
assert len(CASES) == 3


def _native(backend):
    if backend == "torchwave":
        import torch

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)

    package = importlib.import_module(backend)
    return package, package.IMRPhenomT(), array


def _carrier(row):
    return InspiralCarrier(*row["inputs"], row["t0"], row["frequencies"])


def _unit_complex(hp, hc):
    h = np.asarray(hp) + 1j * np.asarray(hc)
    assert np.all(np.abs(h) > 0)
    return h / np.abs(h)


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("row", CASES, ids=lambda row: "-".join(map(str, row["inputs"])))
def test_inspiral_time_root_and_forward_frequency(backend, row):
    _, model, array = _native(backend)
    carrier = _carrier(row)
    m1, m2, chi1, chi2 = row["inputs"]
    for theta in ("0.45", "0.65", "0.75"):
        frequency = float(carrier.frequency_hz(theta))
        expected_time = float(carrier.time_at_frequency(frequency))
        actual_time = float(np.asarray(model.time_at_frequency(
            array(frequency), *map(array, (m1, m2, chi1, chi2))
        )))
        np.testing.assert_allclose(actual_time, expected_time, rtol=2e-9, atol=2e-9)
        actual_frequency = np.asarray(model.frequency(
            array([expected_time]), *map(array, (m1, m2, chi1, chi2))
        )).item()
        np.testing.assert_allclose(actual_frequency, frequency, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("row", CASES, ids=lambda row: "-".join(map(str, row["inputs"])))
def test_inspiral_reference_phase_from_integrated_frequency(backend, row):
    _, model, array = _native(backend)
    carrier = _carrier(row)
    m1, m2, chi1, chi2 = row["inputs"]
    tc, phic = 0.017, 0.23
    tref1 = float(carrier.time_seconds("0.55"))
    tref2 = float(carrier.time_seconds("0.65"))
    teval = float(carrier.time_seconds("0.75"))
    fref1 = float(carrier.frequency_hz("0.55"))
    fref2 = float(carrier.frequency_hz("0.65"))
    times = array([tref1 + tc, tref2 + tc, teval + tc])

    def wave(fref, phase=phic):
        hp, hc = model.time_domain(
            times, *map(array, (m1, m2, chi1, chi2)),
            inclination=array(0.0), phic=array(phase), tc=array(tc), f_ref=array(fref),
        )
        return _unit_complex(hp, hc)

    unit1, unit2 = wave(fref1), wave(fref2)
    delta = float(carrier.phase("0.75") - carrier.phase("0.55"))
    np.testing.assert_allclose(unit1[2] / unit1[0], np.exp(1j * delta), atol=3e-7, rtol=0)
    expected_rotation = np.exp(1j * float(carrier.phase("0.55") - carrier.phase("0.65")))
    np.testing.assert_allclose(unit2[2] / unit1[2], expected_rotation, atol=3e-7, rtol=0)
    np.testing.assert_allclose(unit1[0] / unit2[1], 1.0, atol=3e-7, rtol=0)
    np.testing.assert_allclose(unit1[0] / wave(fref1, 0.0)[0], np.exp(2j * phic),
                               atol=3e-7, rtol=0)


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("centered", (True, False))
def test_public_grid_mask_and_reference_zero(backend, centered):
    package, _, _ = _native(backend)
    row = CASES[0]
    carrier = _carrier(row)
    f_lower = float(carrier.frequency_hz("0.65"))
    start = float(carrier.time_at_frequency(f_lower))
    step, duration, pad, tc = 1 / 1024, 0.5, 0.1, 0.017
    kwargs = dict(
        mass1=30.0, mass2=30.0, duration=duration, delta_t=step,
        t_pad=pad, center_coalescence=centered, f_lower=f_lower,
        tc=tc, phic=0.23, return_times=True,
    )
    hp, hc, times = package.get_td_waveform("IMRPhenomT", f_ref=0, **kwargs)
    grid = np.asarray(fixed_grid(step, duration, pad, centered=centered))
    np.testing.assert_allclose(np.asarray(times), grid, rtol=0, atol=2e-14)
    active = np.asarray(lower_frequency_mask(grid, start, tc))
    assert active.any()
    if centered:
        assert (~active).any()
        assert np.array_equal(np.asarray(hp)[~active], np.zeros((~active).sum()))
        assert np.array_equal(np.asarray(hc)[~active], np.zeros((~active).sum()))
    assert np.any(np.abs(np.asarray(hp)[active]) > 0)
    hp_ref, hc_ref, _ = package.get_td_waveform(
        "IMRPhenomT", f_ref=reference_frequency(f_lower, 0), **kwargs
    )
    np.testing.assert_allclose(np.asarray(hp), np.asarray(hp_ref), rtol=0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc), np.asarray(hc_ref), rtol=0, atol=1e-30)
    if centered:
        hp_default, hc_default, _ = package.get_td_waveform("IMRPhenomT", **kwargs)
        np.testing.assert_allclose(np.asarray(hp), np.asarray(hp_default), rtol=0, atol=1e-30)
        np.testing.assert_allclose(np.asarray(hc), np.asarray(hc_default), rtol=0, atol=1e-30)
