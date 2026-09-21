# `jaxwave` API Reference

This section provides the auto-generated reference documentation for JAX-based waveform generation, JIT compilation, shape-bucketing, and inference modules in **`jaxwave`**.

---

## Core Waveform Functions

::: jaxwave.get_fd_waveform
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.WaveformGenerator
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.get_td_waveform
    options:
      show_root_heading: true
      heading_level: 3

## Native Time-Domain Models

See the [IMRPhenomT guide](../imrphenomt.md) for examples and conventions.

::: jaxwave.IMRPhenomT
    options:
      members: [time_domain, frequency, time_at_frequency]
      heading_level: 3

::: jaxwave.IMRPhenomTHM
    options:
      members: [time_domain, modes]
      heading_level: 3

::: jaxwave.IMRPhenomTP
    options:
      members: [time_domain]
      heading_level: 3

::: jaxwave.IMRPhenomTPHM
    options:
      members: [time_domain]
      heading_level: 3

---

## Shape Bucketing & MCMC Evaluators

::: jaxwave.inference.bucketing.BucketedWaveformEvaluator
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.inference.bucketing.get_bucketed_batch_size
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.inference.bucketing.pad_parameters_to_bucket
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.inference.bucketing.unpad_waveform_from_bucket
    options:
      show_root_heading: true
      heading_level: 3

---

## Bayesian Inference & Fisher Matrix

::: jaxwave.inference.fisher.fisher_information_matrix
    options:
      show_root_heading: true
      heading_level: 3

::: jaxwave.inference.relative_binning.RelativeBinningLikelihood
    options:
      show_root_heading: true
      heading_level: 3
