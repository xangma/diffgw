"""Load independent, optional mathematical references for waveform definitions.

The reference modules use SymPy and mpmath for scalar, high-precision checks.
They are intentionally separate from the Torch and JAX waveform evaluators.
Inspect :mod:`diffgw.waveforms` first to see each module's exact scope.
"""

from importlib import import_module
from types import ModuleType

from .waveforms import get_waveform


def get_reference_module(model_id: str, *, component_id: str | None = None) -> ModuleType:
    """Load the declared reference module for a model or one of its components.

    The ``reference`` extra is required only when calling this function. A
    component ID must belong to the requested model. The returned module
    exposes its own expressions and evaluators; the catalog states which
    parts of the production waveform it covers.
    """
    definition = get_waveform(model_id)
    if component_id is None:
        module_name = definition.reference_module
    else:
        component = next((part for part in definition.components
                          if part.id == component_id), None)
        if component is None:
            raise KeyError(f"unknown component {component_id!r} for {definition.id}")
        module_name = component.reference_module

    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {"sympy", "mpmath", "scipy"}:
            raise ModuleNotFoundError(
                f"Mathematical references require optional dependencies; "
                f"install 'diffgw[reference]' to load {module_name}.",
                name=exc.name,
            ) from exc
        raise
