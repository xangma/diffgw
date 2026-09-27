"""Frozen independent Kerr roots versus the six-mode N61 Torch/JAX input."""

import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "hm_qnm_n61_offgrid_roots.json"
SOLVER = ROOT / "src" / "torchwave" / "qnm" / "solver.py"
DATA = json.loads(FIXTURE.read_text())
MODES = ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
SPINS = (-0.71, 0.37, 0.79)
MAX_COMPLEX_RELATIVE_ERROR = 0.01
MAX_FRONTEND_RELATIVE_ERROR = 1e-12
N61_PAYLOAD_SHA256 = "afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525"


@pytest.fixture(params=("torch", "jax"))
def frontend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("the frozen N61 off-grid gate qualifies CPU FP64 only")
    if request.param == "torch":
        import torch
        from torchwave.approximants import imrphenom_hm as hm
        from torchwave.approximants import imrphenom_hm_qnm_n61 as qnm

        yield "torch", torch, hm, qnm, lambda x: torch.tensor(x, dtype=torch.float64)
        return

    import jax
    import jax.numpy as jnp
    from jaxwave.approximants import imrphenom_hm as hm
    from jaxwave.approximants import imrphenom_hm_qnm_n61 as qnm

    old_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            yield "jax", jnp, hm, qnm, lambda x: jnp.asarray(x, dtype=jnp.float64)
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _root(row):
    return complex(*row["omega_exp_minus_iwt"])


def _fit(frontend, mode, spin):
    _, _, _, qnm, array = frontend
    real, damping = qnm.evaluate_n61_qnm(mode, array(spin))
    assert float(damping) > 0
    return complex(float(real), -float(damping))


def test_reference_fixture_is_complete_and_source_pinned():
    assert DATA["specification_id"] == "hm.qnm.n61.offgrid-six-mode/v1"
    assert DATA["variant"] == "n61_main; hypothesized 61-node a=0.995 sin(theta) training grid"
    assert DATA["reference_solver_sha256"] == hashlib.sha256(SOLVER.read_bytes()).hexdigest()
    assert DATA["spins"] == list(SPINS)
    assert DATA["modes"] == [list(mode) for mode in MODES]
    assert len(DATA["rows"]) == len(SPINS) * len(MODES)
    assert DATA["minimum_training_grid_distance"] == 1e-3
    assert [(row["spin"], tuple(row["mode"])) for row in DATA["rows"]] == [
        (spin, mode) for spin in SPINS for mode in MODES
    ]
    grid = [0.995 * math.sin(-math.pi / 2 + j * math.pi / 60) for j in range(61)]
    for row in DATA["rows"]:
        assert min(abs(row["spin"] - node) for node in grid) > 1e-3
        assert row["grid_distance"] > 1e-3
        assert row["branch_origin"] == "large-ell WKB then separation/spin continuation"
        assert 0 <= row["estimated_frequency_error"] < 1e-9
        assert 0 <= row["cf_error"] < 1e-10
        assert abs(complex(*row["radial_residual"])) < 1e-10
        assert 0 <= row["angular_error"] < 3e-11
        assert row["cf_depth"] >= 512
        root = _root(row)
        assert root.real > 0 and root.imag < 0


@pytest.mark.parametrize("row", DATA["rows"], ids=lambda row: f"a{row['spin']:+.2f}-{row['mode'][0]}{row['mode'][1]}0")
def test_native_n61_complex_frequency_against_independent_root(frontend, row):
    _, _, _, qnm, _ = frontend
    assert qnm.N61_COEFFICIENT_PAYLOAD_SHA256 == N61_PAYLOAD_SHA256
    fit = _fit(frontend, tuple(row["mode"][:2]), row["spin"])
    relative_error = abs(fit - _root(row)) / abs(_root(row))
    assert relative_error <= MAX_COMPLEX_RELATIVE_ERROR, (
        f"{row['mode']} at a={row['spin']}: complex QNM error {relative_error:.6g}"
    )


@pytest.mark.parametrize("row", DATA["rows"], ids=lambda row: f"a{row['spin']:+.2f}-{row['mode'][0]}{row['mode'][1]}0")
def test_torch_jax_n61_parity(row, reference_device):
    if reference_device != "cpu":
        pytest.fail("the frozen N61 off-grid gate qualifies CPU FP64 only")
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm as jax_qnm
    from torchwave.approximants.imrphenom_hm_qnm_n61 import evaluate_n61_qnm as torch_qnm

    old_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        mode, spin = tuple(row["mode"][:2]), row["spin"]
        tr, td = torch_qnm(mode, torch.tensor(spin, dtype=torch.float64))
        with jax.default_device(jax.devices("cpu")[0]):
            jr, jd = jax_qnm(mode, jnp.asarray(spin, dtype=jnp.float64))
        tz, jz = complex(float(tr), -float(td)), complex(float(jr), -float(jd))
        assert abs(tz - jz) / abs(tz) <= MAX_FRONTEND_RELATIVE_ERROR
    finally:
        jax.config.update("jax_enable_x64", old_x64)


@pytest.mark.parametrize("mode", MODES)
def test_public_hm_qnm_total_mass_conversion(frontend, monkeypatch, mode):
    name, _, hm, _, array = frontend
    spin = 0.37
    final_mass = 57.6
    total_mass = 60.0
    row = next(row for row in DATA["rows"] if row["spin"] == spin and tuple(row["mode"]) == mode)
    with monkeypatch.context() as patch:
        patch.setattr(
            hm.IMRPhenomD,
            "remnant_properties",
            staticmethod(lambda *_: (array(final_mass), array(spin), None, None)),
        )
        fr, fd = hm.IMRPhenomHM.mode_qnm_frequencies(
            mode[0], mode[1], array(36.0), array(24.0), array(0.0), array(0.0)
        )
    got = complex(float(fr), -float(fd))
    scale = total_mass / final_mass / (2 * math.pi)
    expected = scale * _root(row)
    assert abs(got - expected) / abs(expected) <= MAX_COMPLEX_RELATIVE_ERROR
    fit = _fit(frontend, mode[:2], spin)
    assert abs(got - scale * fit) / abs(scale * fit) <= 2e-12, name
