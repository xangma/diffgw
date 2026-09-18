"""Differentiable gravitational waves with independently optional backends.

``diffgw.torch`` and ``diffgw.jax`` lazily expose the existing frontend modules.
The convenience API selects PyTorch by default; use ``backend="jax"`` for JAX.
"""

from importlib import import_module

import diffgw_core as core

__version__ = "0.1.0"
__all__ = ["core", "torch", "jax", "get_fd_waveform", "get_td_waveform", "WaveformGenerator", "list_approximants"]


def _backend(name):
    if name not in ("torch", "jax"):
        raise ValueError("backend must be 'torch' or 'jax'")
    pkg = {"torch": "torchwave", "jax": "jaxwave"}[name]
    try:
        return import_module(pkg)
    except ModuleNotFoundError as exc:
        if exc.name in (name, pkg):
            raise ModuleNotFoundError(
                f"The {name} backend requires its optional dependencies; "
                f"install 'diffgw[{name}]'.", name=exc.name
            ) from exc
        raise


def __getattr__(name):
    if name in ("torch", "jax"):
        module = _backend(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_fd_waveform(*args, backend="torch", **kwargs):
    """Generate polarizations using the selected frontend's public API."""
    return _backend(backend).get_fd_waveform(*args, **kwargs)


def get_td_waveform(*args, backend="torch", **kwargs):
    """Generate time-domain polarizations using the selected frontend's public API."""
    return _backend(backend).get_td_waveform(*args, **kwargs)


def list_approximants(*args, backend="torch", **kwargs):
    """List available approximants using the selected frontend's public API."""
    return _backend(backend).list_approximants(*args, **kwargs)


class WaveformGenerator:
    """Construct the selected frontend's generator with its native interface."""

    def __new__(cls, *args, backend="torch", **kwargs):
        return _backend(backend).WaveformGenerator(*args, **kwargs)
