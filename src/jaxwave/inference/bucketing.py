"""Shape bucketing and padding utilities for JAX MCMC and sampling.

Eliminates XLA recompilations during variable-batch or adaptive proposal sampling
(e.g. parallel tempering, ensemble MCMC, NumPyro, BlackJAX) by padding arbitrary
batch sizes to fixed bucket thresholds.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import math
import jax
import jax.numpy as jnp

from jaxwave.waveforms import get_fd_waveform

DEFAULT_BATCH_BUCKETS: Tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def get_bucketed_batch_size(batch_size: int, buckets: Sequence[int] = DEFAULT_BATCH_BUCKETS) -> int:
    """Find the smallest bucket capacity greater than or equal to batch_size.

    If batch_size exceeds all predefined buckets, rounds up to the next power of 2.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for b in buckets:
        if b >= batch_size:
            return b
    # Round up to next power of 2
    return 1 << (batch_size - 1).bit_length()


def pad_parameters_to_bucket(
    parameters: Dict[str, jnp.ndarray],
    bucket_size: int,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Pad parameter arrays along axis 0 up to bucket_size.

    Pads using the edge (last) value of each parameter array to guarantee physical validity.

    Returns:
        padded_parameters: Dict with arrays of length bucket_size.
        valid_mask: Boolean array of shape (bucket_size,) indicating valid (non-padded) entries.
    """
    padded = {}
    valid_mask = None

    for k, v in parameters.items():
        arr = jnp.asarray(v)
        if arr.ndim == 0:
            arr = arr[None]
        n_orig = arr.shape[0]
        if n_orig > bucket_size:
            raise ValueError(f"Parameter '{k}' length ({n_orig}) exceeds bucket_size ({bucket_size})")

        pad_len = bucket_size - n_orig
        if pad_len > 0:
            # Pad with the last value along axis 0
            last_val = arr[-1:]
            pad_shape = (pad_len,) + arr.shape[1:]
            fill_arr = jnp.broadcast_to(last_val, pad_shape)
            padded[k] = jnp.concatenate([arr, fill_arr], axis=0)
        else:
            padded[k] = arr

        if valid_mask is None:
            valid_mask = jnp.arange(bucket_size) < n_orig

    if valid_mask is None:
        valid_mask = jnp.ones((bucket_size,), dtype=bool)

    return padded, valid_mask


def unpad_waveform_from_bucket(
    hp: jnp.ndarray,
    hc: jnp.ndarray,
    original_batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Slice evaluated batched waveforms back to the original proposal batch size."""
    if hp.ndim >= 2:
        return hp[:original_batch_size], hc[:original_batch_size]
    return hp, hc


class BucketedWaveformEvaluator:
    """Manages JIT-compiled waveform evaluators across fixed shape buckets.

    Guarantees zero XLA recompilations across variable proposal batch sizes by
    routing evaluations to the nearest bucket size and caching the compiled kernel.
    """

    def __init__(
        self,
        approximant: str = "IMRPhenomD",
        f_lower: Optional[float] = None,
        f_final: Optional[float] = None,
        delta_f: Optional[float] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        buckets: Sequence[int] = DEFAULT_BATCH_BUCKETS,
        parameter_order: Tuple[str, ...] = ("mass1", "mass2", "spin1z", "spin2z", "distance", "inclination", "phic", "tc"),
        **default_kwargs,
    ):
        self.approximant = approximant
        self.f_lower = f_lower
        self.f_final = f_final
        self.delta_f = delta_f
        self.sample_frequencies = sample_frequencies
        self.buckets = sorted(buckets)
        self.parameter_order = parameter_order
        self.default_kwargs = default_kwargs
        self._compiled_evaluators: Dict[int, Callable] = {}

    def _get_or_compile_evaluator(self, bucket_size: int) -> Callable:
        """Compile or retrieve the JIT evaluator for a specific bucket size."""
        if bucket_size in self._compiled_evaluators:
            return self._compiled_evaluators[bucket_size]

        approx = self.approximant
        f_low = self.f_lower
        f_fin = self.f_final
        df = self.delta_f
        s_freqs = self.sample_frequencies
        param_keys = self.parameter_order
        d_kwargs = dict(self.default_kwargs)

        def _raw_eval(*param_arrays):
            kw = dict(d_kwargs)
            for k, arr in zip(param_keys, param_arrays):
                kw[k] = arr
            return get_fd_waveform(
                approximant=approx,
                f_lower=f_low,
                f_final=f_fin,
                delta_f=df,
                sample_frequencies=s_freqs,
                **kw,
            )

        jitted_eval = jax.jit(_raw_eval)
        self._compiled_evaluators[bucket_size] = jitted_eval
        return jitted_eval

    def __call__(
        self,
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate waveforms for an arbitrary batch size with automatic bucketing."""
        norm_params = {}
        name_map = {
            "mass_1": "mass1",
            "mass_2": "mass2",
            "spin_1z": "spin1z",
            "spin_2z": "spin2z",
            "luminosity_distance": "distance",
            "theta_jn": "inclination",
            "phase": "phic",
            "geocent_time": "tc",
        }
        for k, v in parameters.items():
            norm_params[name_map.get(k, k)] = v
        norm_params.update(kwargs)

        # Determine original batch size
        first_val = next(iter(norm_params.values()))
        first_arr = jnp.asarray(first_val)
        orig_batch = first_arr.shape[0] if first_arr.ndim > 0 else 1

        bucket_size = get_bucketed_batch_size(orig_batch, self.buckets)
        evaluator = self._get_or_compile_evaluator(bucket_size)

        # Pad parameters to bucket
        padded_params, _ = pad_parameters_to_bucket(norm_params, bucket_size)

        # Build positional arguments according to parameter_order
        args = []
        for k in self.parameter_order:
            if k in padded_params:
                args.append(jnp.asarray(padded_params[k], dtype=jnp.float64))
            else:
                default_v = self.default_kwargs.get(k, 0.0)
                args.append(jnp.full((bucket_size,), float(default_v), dtype=jnp.float64))

        hp_padded, hc_padded = evaluator(*args)
        return unpad_waveform_from_bucket(hp_padded, hc_padded, orig_batch)
