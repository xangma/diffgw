"""Development-only adapters for exercising quarantined SEOBNR prototypes.

These adapters never register models in the release dispatcher. New scientific
qualification should call the explicit reconstruction functions directly.
"""
from importlib import import_module


def _get_fd_waveform(backend, approximant, *, mass1, mass2, sample_frequencies=None,
                     f_lower=None, f_final=None, delta_f=None, chi1=0., chi2=0.,
                     spin1z=None, spin2z=None, spin1x=0., spin1y=0., spin2x=0.,
                     spin2y=0., dtype=None, device=None, **kwargs):
    if backend == "torch":
        import torch as xp
        dtype = dtype or xp.float64
        def array(value):
            return xp.as_tensor(value, dtype=dtype, device=device)
        def grid():
            return xp.arange(f_lower, f_final + delta_f / 2, delta_f,
                             dtype=dtype, device=device)
    else:
        import jax.numpy as xp
        dtype = dtype or xp.float64
        def array(value):
            return xp.asarray(value, dtype=dtype)
        def grid():
            return xp.arange(f_lower, f_final + delta_f / 2, delta_f, dtype=dtype)
    if any(bool(xp.any(array(spin) != 0)) for spin in (spin1x, spin1y, spin2x, spin2y)):
        raise ValueError("SEOBNR prototypes support aligned spins only")
    if sample_frequencies is None:
        if any(value is None for value in (f_lower, f_final, delta_f)):
            raise ValueError("provide sample_frequencies or f_lower, f_final and delta_f")
        frequencies = grid()
    else:
        frequencies = array(sample_frequencies)
    key = approximant.lower().replace("_", "").replace("-", "")
    classes = {
        "seobnr": ("waveform", "SEOBNR_ODE"),
        "seobnrode": ("waveform", "SEOBNR_ODE"),
        "seobnrv5": ("waveform", "SEOBNR_ODE"),
        "seobnrv4published": ("published_waveform", "SEOBNRv4Published"),
        "seobnrv4publishednonspinning": ("published_waveform", "SEOBNRv4PublishedNonspinning"),
    }
    if key not in classes:
        raise ValueError(f"Unknown development prototype: {approximant}")
    module, name = classes[key]
    cls = getattr(import_module(f"dev.native_rom.eob.{backend}.{module}"), name)
    result = cls()(f=frequencies, mass1=mass1, mass2=mass2,
                   chi1=chi1 if spin1z is None else spin1z,
                   chi2=chi2 if spin2z is None else spin2z,
                   f_lower=f_lower, **kwargs)
    mask = frequencies <= 0
    if f_lower is not None:
        mask = mask | (frequencies < f_lower)
    if f_final is not None:
        mask = mask | (frequencies >= f_final)
    return tuple(xp.where(mask, xp.zeros_like(value), value) for value in result)


def torch_get_fd_waveform(approximant, **kwargs):
    return _get_fd_waveform("torch", approximant, **kwargs)


def jax_get_fd_waveform(approximant, **kwargs):
    return _get_fd_waveform("jax", approximant, **kwargs)
