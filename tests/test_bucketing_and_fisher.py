"""Unit tests for MCMC shape bucketing utilities and Fisher Information Matrix parity."""

import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

import torchwave
import jaxwave
from jaxwave.inference.bucketing import (
    get_bucketed_batch_size,
    pad_parameters_to_bucket,
    unpad_waveform_from_bucket,
    BucketedWaveformEvaluator,
)
from torchwave.inference.fisher import fisher_information_matrix as torch_fim
from torchwave.detectors import aLIGOZeroDetHighPower as torch_psd
from jaxwave.inference.fisher import fisher_information_matrix as jax_fim
from jaxwave.detectors import aLIGOZeroDetHighPower as jax_psd


def test_get_bucketed_batch_size():
    buckets = (1, 8, 16, 32, 64)
    assert get_bucketed_batch_size(1, buckets) == 1
    assert get_bucketed_batch_size(3, buckets) == 8
    assert get_bucketed_batch_size(8, buckets) == 8
    assert get_bucketed_batch_size(9, buckets) == 16
    assert get_bucketed_batch_size(33, buckets) == 64
    assert get_bucketed_batch_size(65, buckets) == 128


def test_pad_parameters_to_bucket():
    params = {
        "mass1": jnp.array([30.0, 32.0, 35.0]),
        "mass2": jnp.array([20.0, 22.0, 25.0]),
    }
    padded, mask = pad_parameters_to_bucket(params, bucket_size=8)
    assert padded["mass1"].shape == (8,)
    assert padded["mass2"].shape == (8,)
    assert mask.shape == (8,)
    assert mask.sum() == 3
    assert jnp.all(mask[:3]) and not jnp.any(mask[3:])
    assert padded["mass1"][3] == 35.0  # last element edge padding


def test_bucketed_waveform_evaluator():
    evaluator = BucketedWaveformEvaluator(
        approximant="IMRPhenomD",
        f_lower=20.0,
        f_final=256.0,
        delta_f=0.5,
        buckets=(1, 8, 16),
    )

    # Call with batch size 3
    p3 = {
        "mass1": jnp.array([30.0, 32.0, 35.0]),
        "mass2": jnp.array([20.0, 22.0, 25.0]),
        "chi1": jnp.array([0.1, 0.2, 0.3]),
        "chi2": jnp.array([-0.1, 0.0, 0.1]),
    }
    hp3, hc3 = evaluator(p3)
    assert hp3.shape[0] == 3 and hc3.shape[0] == 3

    # Call with batch size 5 (routed to bucket 8 without compiling a new bucket)
    p5 = {
        "mass1": jnp.array([30.0, 31.0, 32.0, 33.0, 34.0]),
        "mass2": jnp.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        "chi1": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        "chi2": jnp.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
    }
    hp5, hc5 = evaluator(p5)
    assert hp5.shape[0] == 5 and hc5.shape[0] == 5

    # Evaluator should have compiled exactly 1 bucket (bucket 8)
    assert len(evaluator._compiled_evaluators) == 1
    assert 8 in evaluator._compiled_evaluators


def test_fisher_matrix_torch_jax_parity():
    freqs_t = torch.linspace(25.0, 400.0, 256, dtype=torch.float64)
    df_t = float(freqs_t[1] - freqs_t[0])
    psd_t = torch_psd(freqs_t)

    freqs_j = jnp.linspace(25.0, 400.0, 256, dtype=jnp.float64)
    psd_j = jax_psd(freqs_j)

    params = {"mass1": 30.0, "mass2": 20.0}
    param_order = ["mass1", "mass2"]

    def wf_t(p):
        return torchwave.get_fd_waveform("TaylorF2", mass1=p["mass1"], mass2=p["mass2"], sample_frequencies=freqs_t)

    def wf_j(p):
        return jaxwave.get_fd_waveform("TaylorF2", mass1=p["mass1"], mass2=p["mass2"], sample_frequencies=freqs_j)

    fim_t, _ = torch_fim(params, wf_t, psd_t, df=df_t, parameter_order=param_order)
    fim_j, _ = jax_fim(params, wf_j, psd_j, df=df_t, parameter_order=param_order, method="autodiff")

    t_mat = fim_t.detach().cpu().numpy()
    j_mat = np.array(fim_j)
    rel_diff = np.max(np.abs(t_mat - j_mat) / np.maximum(np.abs(t_mat), 1e-30))
    assert rel_diff < 1e-10, f"FIM relative difference too large: {rel_diff}"
