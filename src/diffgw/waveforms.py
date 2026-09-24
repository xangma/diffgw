"""Lightweight catalog of the currently scoped mathematical waveforms.

This catalog describes native model availability and explicitly bounded scalar
reference compositions. It does not import a backend or evaluate a waveform.
For tensor evaluation use :func:`diffgw.get_fd_waveform`.
"""

from __future__ import annotations

from dataclasses import asdict
import json

from diffgw_core.waveform_definitions import WAVEFORMS, WaveformDefinition
from diffgw_core.waveform_definitions import validate_catalog as _validate_catalog

__all__ = ["list_waveforms", "get_waveform", "describe_waveform", "validate_catalog"]


def _key(model_id: str) -> str:
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    return model_id.lower().replace("_", "").replace("-", "")


_BY_ID = {
    _key(alias): definition
    for definition in WAVEFORMS
    for alias in (definition.id, *definition.aliases)
}


def list_waveforms() -> tuple[WaveformDefinition, ...]:
    """Return immutable, versioned definitions for the cataloged model scopes.

    This list is deliberately narrower than the native approximant registries.
    A record's ``claims`` field is empty until a bounded verification result is
    attached with its component, settings, precision, device, and evidence.
    """
    return WAVEFORMS


def get_waveform(model_id: str) -> WaveformDefinition:
    """Get a definition by canonical ID or registered alias.

    Lookup is case-insensitive and ignores underscores and hyphens, matching
    the native FD dispatcher. Unknown IDs raise ``KeyError``.
    """
    try:
        return _BY_ID[_key(model_id)]
    except KeyError:
        raise KeyError(f"waveform definition not found: {model_id!r}") from None


def describe_waveform(model_id: str) -> dict[str, object]:
    """Return a detached, JSON-serializable description of one definition.

    The returned dictionary can be changed without mutating the catalog.
    Metadata tags are descriptive; sampled agreement and physical validation
    have distinct claim categories when a scoped claim is present.
    """
    return json.loads(json.dumps(asdict(get_waveform(model_id))))


def validate_catalog(definitions: tuple[WaveformDefinition, ...] | None = None) -> None:
    """Check model/component identity, versions, options, and source links."""
    _validate_catalog(WAVEFORMS if definitions is None else definitions)
