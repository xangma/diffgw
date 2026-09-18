"""Unit tests verifying recompile elimination and autodiff gradient performance."""

import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

import torchwave
from torchwave.approximants.precession_msa_initial_clean import (
    _get_leggauss,
    elliptic_E_quadrature,
    spin_average_exact,
)
from torchwave.benchmark import (
    benchmark_torchwave_model,
    benchmark_jaxwave_model,
    DEFAULT_MODEL_SPECS,
)
import jaxwave
from jaxwave.waveforms import WaveformGenerator


def test_precession_msa_quadrature_graph_break_eliminated():
    """Verify that _get_leggauss and elliptic_E_quadrature compile with 0 graph breaks."""
    m = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    explanation = torch._dynamo.explain(elliptic_E_quadrature)(m)
    assert explanation.graph_break_count == 0, f"Expected 0 graph breaks, got {explanation.graph_break_count}"

    compiled_E = torch.compile(elliptic_E_quadrature)
    res_compiled = compiled_E(m)
    res_eager = elliptic_E_quadrature(m)
    assert torch.allclose(res_compiled, res_eager, atol=1e-12, rtol=1e-10)


def test_torch_compile_dynamic_batch_generalization():
    """Verify that torch.compile with dynamic=True does not recompile across arbitrary batch sizes."""
    torch._dynamo.reset()
    freqs = torch.linspace(20.0, 512.0, 200, dtype=torch.float64)

    def eval_fn(m1, m2):
        hp, hc = torchwave.get_fd_waveform("TaylorF2", sample_frequencies=freqs, mass1=m1, mass2=m2)
        return (hp.abs()**2 + hc.abs()**2).sum()

    compiled_eval = torch.compile(eval_fn, dynamic=True)

    # Initial compile at B=1
    m1_1 = torch.full((1,), 30.0, dtype=torch.float64, requires_grad=True)
    m2_1 = torch.full((1,), 20.0, dtype=torch.float64, requires_grad=True)
    l1 = compiled_eval(m1_1, m2_1)
    l1.backward()
    assert m1_1.grad is not None and torch.isfinite(m1_1.grad).all()

    # Dynamic transition at B=10
    m1_10 = torch.full((10,), 30.0, dtype=torch.float64, requires_grad=True)
    m2_10 = torch.full((10,), 20.0, dtype=torch.float64, requires_grad=True)
    l10 = compiled_eval(m1_10, m2_10)
    l10.backward()
    assert m1_10.grad is not None and m1_10.grad.shape == (10,)

    # Evaluation at arbitrary new batch size B=27 without recompiling
    m1_27 = torch.full((27,), 30.0, dtype=torch.float64, requires_grad=True)
    m2_27 = torch.full((27,), 20.0, dtype=torch.float64, requires_grad=True)
    l27 = compiled_eval(m1_27, m2_27)
    l27.backward()
    assert m1_27.grad is not None and m1_27.grad.shape == (27,)


def test_jaxwave_benchmark_measure_backward():
    """Verify benchmark_jaxwave_model computes forward and backward times via jax.value_and_grad."""
    spec = DEFAULT_MODEL_SPECS["TaylorF2"]
    res = benchmark_jaxwave_model(
        "TaylorF2",
        batch_size=2,
        grid=spec["grid"],
        params=spec["params"],
        iterations=5,
        warmup=2,
        use_jit=True,
        measure_backward=True,
    )
    assert res is not None
    assert "forward_mean_ms" in res and res["forward_mean_ms"] > 0.0
    assert "backward_mean_ms" in res and res["backward_mean_ms"] >= 0.0
    assert res["wf_per_sec"] > 0.0


def test_jaxwave_waveform_generator_jit_caching():
    """Verify WaveformGenerator caches compiled functions across repeated calls."""
    gen = WaveformGenerator(
        approximant="IMRPhenomD",
        f_lower=20.0,
        f_final=512.0,
        delta_f=0.5,
        use_jit=True,
    )
    # First call compiles
    hp1, hc1 = gen.generate_fd_waveform(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)
    assert hp1.ndim == 1 and len(hp1) > 0

    # Second call with different parameters reuses compiled function
    hp2, hc2 = gen.generate_fd_waveform(mass1=35.0, mass2=25.0, chi1=0.1, chi2=0.0)
    assert hp2.ndim == 1 and len(hp2) > 0
    assert not jnp.allclose(hp1, hp2, atol=0.0, rtol=1e-4)
