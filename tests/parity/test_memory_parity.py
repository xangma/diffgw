"""Parity tests: torchwave vs jaxwave for Non-Linear Gravitational Wave Memory."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.memory import (
    compute_memory_h20 as torch_compute_memory,
    memory_polarizations as torch_memory_polarizations,
    add_memory_to_waveform as torch_add_memory,
    MemoryWaveform as TorchMemoryWaveform,
)
from jaxwave.approximants.memory import (
    compute_memory_h20 as jax_compute_memory,
    memory_polarizations as jax_memory_polarizations,
    add_memory_to_waveform as jax_add_memory,
    MemoryWaveform as JaxMemoryWaveform,
)


def test_memory_h20_parity():
    """Verify exact parity of compute_memory_h20."""
    t_np = np.linspace(-0.1, 0.1, 1000)
    amp = 1e-21 * np.exp(-((t_np / 0.01) ** 2))
    phase = 2.0 * math.pi * 150.0 * t_np + 500.0 * (t_np ** 2)
    hp_np = amp * np.cos(phase)
    hc_np = amp * np.sin(phase)

    t_t = torch.tensor(t_np, dtype=torch.float64)
    hp_t = torch.tensor(hp_np, dtype=torch.float64)
    hc_t = torch.tensor(hc_np, dtype=torch.float64)

    t_j = jnp.array(t_np, dtype=jnp.float64)
    hp_j = jnp.array(hp_np, dtype=jnp.float64)
    hc_j = jnp.array(hc_np, dtype=jnp.float64)

    h20_t = torch_compute_memory(times=t_t, modes_or_hp=hp_t, hc=hc_t).numpy()
    h20_j = np.array(jax_compute_memory(times=t_j, modes_or_hp=hp_j, hc=hc_j))

    np.testing.assert_allclose(h20_j, h20_t, rtol=1e-12, atol=1e-35)


def test_memory_polarizations_parity():
    """Verify exact parity of memory_polarizations across inclinations."""
    h20_np = np.linspace(0.0, 1e-22, 100)
    h20_t = torch.tensor(h20_np, dtype=torch.float64)
    h20_j = jnp.array(h20_np, dtype=jnp.float64)

    for inc in [0.0, 0.3, 0.5 * math.pi, 2.5]:
        hp_mem_t, hc_mem_t = torch_memory_polarizations(h20_t, inclination=inc)
        hp_mem_j, hc_mem_j = jax_memory_polarizations(h20_j, inclination=inc)

        np.testing.assert_allclose(np.array(hp_mem_j), hp_mem_t.numpy(), rtol=1e-12, atol=1e-35)
        np.testing.assert_allclose(np.array(hc_mem_j), hc_mem_t.numpy(), rtol=1e-12, atol=1e-35)


def test_add_memory_to_waveform_parity():
    """Verify add_memory_to_waveform parity."""
    t_np = np.linspace(-0.05, 0.05, 500)
    amp = 1e-21 * np.exp(-((t_np / 0.01) ** 2))
    phase = 2.0 * math.pi * 100.0 * t_np
    hp_np = amp * np.cos(phase)
    hc_np = amp * np.sin(phase)

    t_t = torch.tensor(t_np, dtype=torch.float64)
    hp_t = torch.tensor(hp_np, dtype=torch.float64)
    hc_t = torch.tensor(hc_np, dtype=torch.float64)

    t_j = jnp.array(t_np, dtype=jnp.float64)
    hp_j = jnp.array(hp_np, dtype=jnp.float64)
    hc_j = jnp.array(hc_np, dtype=jnp.float64)

    hp_tot_t, hc_tot_t, h20_t = torch_add_memory(times=t_t, hp=hp_t, hc=hc_t, inclination=0.7)
    hp_tot_j, hc_tot_j, h20_j = jax_add_memory(times=t_j, hp=hp_j, hc=hc_j, inclination=0.7)

    np.testing.assert_allclose(np.array(hp_tot_j), hp_tot_t.numpy(), rtol=1e-12, atol=1e-35)
    np.testing.assert_allclose(np.array(hc_tot_j), hc_tot_t.numpy(), rtol=1e-12, atol=1e-35)
    np.testing.assert_allclose(np.array(h20_j), h20_t.numpy(), rtol=1e-12, atol=1e-35)


def test_memory_jax_jit():
    """Verify JIT compatibility of JaxMemoryWaveform."""
    t_j = jnp.linspace(-0.05, 0.05, 100)
    hp_j = 1e-21 * jnp.cos(2.0 * math.pi * 100.0 * t_j)
    hc_j = 1e-21 * jnp.sin(2.0 * math.pi * 100.0 * t_j)

    model = JaxMemoryWaveform()
    jitted_fn = jax.jit(lambda t, hp, hc, inc: model(t, hp, hc, inc))

    hp_tot, hc_tot, h20 = jitted_fn(t_j, hp_j, hc_j, 0.5)
    assert hp_tot.shape == hp_j.shape
    assert hc_tot.shape == hc_j.shape
    assert h20.shape == t_j.shape
