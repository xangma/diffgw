"""Small, immutable vocabulary for published waveform definitions.

These records describe the current mathematical-reference scope. They are not
evaluators and do not imply that a sampled comparison validates a full model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class SourceDefinition:
    id: str
    title: str
    url: str
    version: str
    kind: str = "publication"
    status: str = "published"
    sha256: str | None = None
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    location: str


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    meaning: str
    unit: str
    domain: str
    role: str = "public_input"


@dataclass(frozen=True)
class UnitDefinition:
    quantity: str
    unit: str
    relation: str


@dataclass(frozen=True)
class ConventionDefinition:
    id: str
    description: str


@dataclass(frozen=True)
class OptionDefinition:
    name: str
    choices: tuple[str, ...]
    default: str
    semantics: str
    native_choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ComponentDefinition:
    id: str
    name: str
    spec_version: str
    scope: str
    variant: str
    implementation: str
    evidence: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]
    reference_module: str
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationClaim:
    """A bounded verification result; never a whole-model status badge."""

    component_ids: tuple[str, ...]
    scope: str
    variant: str
    backends: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    domain: str
    precision: str
    device: str
    category: str
    status: str
    evidence_path: str


@dataclass(frozen=True)
class WaveformDefinition:
    id: str
    name: str
    aliases: tuple[str, ...]
    domain: str
    modes: tuple[tuple[int, int], ...]
    scope: str
    reference_version: str
    reference_module: str
    spec_page: str
    components: tuple[ComponentDefinition, ...]
    sources: tuple[SourceDefinition, ...]
    parameters: tuple[ParameterDefinition, ...]
    units: tuple[UnitDefinition, ...]
    conventions: tuple[ConventionDefinition, ...]
    options: tuple[OptionDefinition, ...]
    carrier_parameters: tuple[str, ...]
    capabilities: tuple[str, ...]
    evidence: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    claims: tuple[VerificationClaim, ...] = ()
    fit_asset_ids: tuple[str, ...] = ()


_VERSION = re.compile(r"^[a-z0-9][a-z0-9.-]*/v[1-9][0-9]*$")
_COMPONENT_ID = re.compile(r"^[a-z][a-zA-Z0-9_.]*\.v[1-9][0-9]*$")


def validate_catalog(definitions: Iterable[WaveformDefinition]) -> None:
    """Raise ``ValueError`` for broken catalog identity and provenance links."""
    model_ids: set[str] = set()
    spec_pages: set[str] = set()
    aliases: set[str] = set()
    components_by_id: dict[str, ComponentDefinition] = {}
    for model in definitions:
        if not model.id or model.id in model_ids:
            raise ValueError(f"duplicate or empty waveform ID: {model.id!r}")
        model_ids.add(model.id)
        if not _VERSION.fullmatch(model.reference_version):
            raise ValueError(f"invalid reference version: {model.reference_version!r}")
        if not model.scope or not model.reference_module:
            raise ValueError(f"missing waveform scope or reference module: {model.id}")
        if not re.fullmatch(r"math/[a-z0-9][a-z0-9-]*\.md", model.spec_page):
            raise ValueError(f"invalid canonical specification page: {model.spec_page!r}")
        if model.spec_page in spec_pages:
            raise ValueError(f"duplicate canonical specification page: {model.spec_page!r}")
        spec_pages.add(model.spec_page)
        if not model.reference_module.startswith("diffgw_reference."):
            raise ValueError(f"invalid reference module: {model.reference_module!r}")
        for alias in (model.id, *model.aliases):
            key = alias.lower().replace("_", "").replace("-", "")
            if not key or key in aliases:
                raise ValueError(f"duplicate or empty waveform alias: {alias!r}")
            aliases.add(key)
        source_ids = {source.id for source in model.sources}
        if len(source_ids) != len(model.sources):
            raise ValueError(f"duplicate source ID in {model.id}")
        source_by_id = {source.id: source for source in model.sources}
        if any(asset_id not in source_by_id or source_by_id[asset_id].kind != "external_asset"
               for asset_id in model.fit_asset_ids):
            raise ValueError(f"unresolved fit asset in {model.id}")
        if len({parameter.name for parameter in model.parameters}) != len(model.parameters):
            raise ValueError(f"duplicate parameter name in {model.id}")
        if len({name for name in model.carrier_parameters}) != len(model.carrier_parameters):
            raise ValueError(f"duplicate carrier parameter name in {model.id}")
        for option in model.options:
            if not option.choices or option.default not in option.choices:
                raise ValueError(f"invalid option {option.name!r} in {model.id}")
            if option.native_choices is not None and (
                option.default not in option.native_choices
                or not set(option.native_choices) <= set(option.choices)
                or len(set(option.native_choices)) != len(option.native_choices)
            ):
                raise ValueError(f"invalid native choices for {option.name!r} in {model.id}")
        if len({option.name for option in model.options}) != len(model.options):
            raise ValueError(f"duplicate option name in {model.id}")
        known_options = {option.name: option for option in model.options}
        local_components: set[str] = set()
        for component in model.components:
            if not _COMPONENT_ID.fullmatch(component.id) or component.id in local_components:
                raise ValueError(f"duplicate or invalid component ID: {component.id!r}")
            previous = components_by_id.setdefault(component.id, component)
            if previous != component:
                raise ValueError(f"conflicting component ID: {component.id!r}")
            local_components.add(component.id)
            if not _VERSION.fullmatch(component.spec_version):
                raise ValueError(f"invalid component version: {component.spec_version!r}")
            if not component.scope or not component.variant or not component.implementation:
                raise ValueError(f"missing scope, variant, or implementation: {component.id}")
            if not component.reference_module.startswith("diffgw_reference."):
                raise ValueError(f"invalid reference module: {component.reference_module!r}")
            if not component.source_refs:
                raise ValueError(f"missing source references: {component.id}")
            for ref in component.source_refs:
                if ref.source_id not in source_ids or not ref.location:
                    raise ValueError(f"unresolved source reference in {component.id}: {ref!r}")
        for claim in model.claims:
            if not claim.component_ids or not set(claim.component_ids) <= local_components:
                raise ValueError(f"unresolved claim component IDs in {model.id}")
            if not all((claim.scope, claim.variant, claim.domain, claim.precision,
                        claim.device, claim.category, claim.status, claim.evidence_path)):
                raise ValueError(f"incomplete verification claim in {model.id}")
            if not claim.backends or not set(claim.backends) <= {"torch", "jax", "scalar_reference", "lal", "nr"}:
                raise ValueError(f"invalid verification backends in {model.id}")
            for name, value in claim.options:
                if name not in known_options or value not in known_options[name].choices:
                    raise ValueError(f"unresolved claim option in {model.id}: {name}={value}")
