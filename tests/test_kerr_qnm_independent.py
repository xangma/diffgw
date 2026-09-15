"""Check the runtime interpolant against fresh, stricter equation solves."""

import json
from pathlib import Path

import pytest
import torch

from torchwave.approximants.kerr_qnm_generated import (
    SPIN_MAX, SUPPORTED_MODES, _spline_tensors, dimensionless_qnm,
)

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def independent_roots():
    return json.loads((DATA / "kerr_qnm_independent_holdouts.json").read_text())


@pytest.mark.parametrize("mode", [(l, m, n) for n in range(3)
    for l, m in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))] + [(2, -2, 0)])
def test_all_modes_match_stricter_equation_solves(mode, independent_roots):
    fixture = independent_roots["modes"][",".join(map(str, mode))]
    rows = fixture["rows"] + fixture["endpoints"]
    spin = torch.tensor([r["spin"] for r in rows], dtype=torch.float64)
    expected = torch.tensor([r["omega"] for r in rows], dtype=torch.float64)
    real, decay = dimensionless_qnm(*mode, spin)
    actual = torch.stack((real, -decay), dim=-1)
    error = abs(actual-expected)
    # Both gates apply; an absolute allowance cannot hide relative damping error.
    assert torch.all(error <= 1e-12)
    assert torch.all(error/abs(expected) <= 1e-10)
    assert all(r["converged"] and r["metadata"]["strict"] for r in rows)
    assert independent_roots["external_numerical_inputs"] == []


@pytest.mark.parametrize("mode", [(2, 1, 2), (3, 2, 0), (3, 2, 2), (4, 4, 1), (2, -2, 0)])
def test_multimode_spin_derivatives(mode):
    spin = torch.tensor([.17, .731, .962, .99125], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: dimensionless_qnm(*mode, a),
                                    (spin,), eps=1e-8, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_dtype_endpoint_and_inference_cache(dtype):
    _spline_tensors.cache_clear()
    spin = torch.tensor([[0., .3], [.8, SPIN_MAX]], dtype=dtype)
    with torch.inference_mode():
        expected = dimensionless_qnm(2, 2, 0, spin)
    spin.requires_grad_()
    real, decay = dimensionless_qnm(2, 2, 0, spin)
    assert real.shape == decay.shape == spin.shape
    assert real.dtype == decay.dtype == dtype
    assert real.device == decay.device == spin.device
    assert torch.equal(real, expected[0]) and torch.equal(decay, expected[1])
    gradient, = torch.autograd.grad((real+decay).sum(), spin)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("spin", [-.1, 1., float("nan"), float("inf")])
def test_rejects_unvalidated_spin(spin, dtype):
    with pytest.raises(ValueError, match="spin domain"):
        dimensionless_qnm(2, 2, 0, torch.tensor(spin, dtype=dtype))


def test_complete_mode_coverage_and_type_contract():
    assert len(SUPPORTED_MODES) == 16
    with pytest.raises(ValueError, match="Unsupported"):
        dimensionless_qnm(2, 2, 3, .7)
    with pytest.raises(TypeError, match="floating"):
        dimensionless_qnm(2, 2, 0, torch.tensor(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_results_and_spin_gradients(independent_roots):
    rows = independent_roots["modes"]["2,2,0"]["rows"]
    spin = torch.tensor([r["spin"] for r in rows], dtype=torch.float64,
                        device="cuda", requires_grad=True)
    expected = torch.tensor([r["omega"] for r in rows], dtype=torch.float64, device="cuda")
    real, decay = dimensionless_qnm(2, 2, 0, spin)
    error = abs(torch.stack((real, -decay), dim=-1)-expected)
    assert torch.all(error <= 1e-12) and torch.all(error/abs(expected) <= 1e-10)
    gradient, = torch.autograd.grad((real+decay).sum(), spin)
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_mps_float32_values_and_gradients(mode):
    # Compare at identical representable spins: rounding the spin itself near
    # extremality can exceed the interpolation error by orders of magnitude.
    spin_cpu = torch.tensor([0., .17, .3, .7, .962, .99125, SPIN_MAX],
                            dtype=torch.float32)
    spin_mps = spin_cpu.to("mps").requires_grad_()
    spin_ref = spin_cpu.double().requires_grad_()
    _spline_tensors.cache_clear()
    with torch.inference_mode():
        dimensionless_qnm(*mode, spin_mps)
    actual = dimensionless_qnm(*mode, spin_mps)
    expected = dimensionless_qnm(*mode, spin_ref)
    for a, e in zip(actual, expected):
        assert a.device.type == "mps" and a.dtype == torch.float32
        torch.testing.assert_close(a.cpu().double(), e, rtol=2e-6, atol=0.)
        ga, = torch.autograd.grad(a.sum(), spin_mps, retain_graph=True)
        ge, = torch.autograd.grad(e.sum(), spin_ref, retain_graph=True)
        torch.testing.assert_close(ga.cpu().double(), ge, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mps_rejects_spin_one_before_log(dtype):
    with pytest.raises(ValueError, match="spin domain"):
        dimensionless_qnm(2, 2, 0, torch.tensor(1., dtype=dtype, device="mps"))


def test_ringdown_default_and_legacy_alias_use_independent_frequencies():
    from torchwave.approximants.kerr_ringdown import evaluate_qnm
    from torchwave.constants import MSUN_SEC

    spin = torch.tensor([0., .41, .99], dtype=torch.float64)
    real, decay = dimensionless_qnm(3, 2, 1, spin)
    default = evaluate_qnm(3, 2, 1, 60., spin)
    explicit = evaluate_qnm(3, 2, 1, 60., spin, qnm_backend="independent")
    legacy = evaluate_qnm(3, 2, 1, 60., spin, qnm_backend="author")
    expected = (real/(60*MSUN_SEC), 60*MSUN_SEC/decay, real/(2*decay))
    for a, b, c, target in zip(default, explicit, legacy, expected):
        assert torch.equal(a, b) and torch.equal(a, c)
        torch.testing.assert_close(a, target, rtol=2e-15, atol=0.)
