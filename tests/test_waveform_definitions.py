"""Public waveform definitions and their relationship to native frontends."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


MODEL_IDS = {"TaylorF2", "TaylorF2_HM", "TaylorF2_Eccentric", "IMRPhenomD",
             "IMRPhenomXAS", "KerrRingdown220Direct", "ContinuousWaveH1Direct",
             "MemoryWaveform22Direct"}


def test_catalog_import_needs_no_optional_math_or_tensor_packages():
    """Documentation and inspection must work with the base install alone."""
    root = Path(__file__).resolve().parents[1]
    code = r'''
import importlib.abc
import sys

blocked = {"sympy", "mpmath", "scipy", "torch", "jax", "torchwave", "jaxwave", "diffgw_reference"}

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockOptional())
import diffgw.waveforms as waveforms
import diffgw.reference as reference

assert {definition.id for definition in waveforms.list_waveforms()} == {
    "TaylorF2", "TaylorF2_HM", "TaylorF2_Eccentric", "IMRPhenomD",
    "IMRPhenomXAS", "KerrRingdown220Direct", "ContinuousWaveH1Direct",
    "MemoryWaveform22Direct"
}
assert waveforms.get_waveform("TaylorF2").id == "TaylorF2"
assert callable(reference.get_reference_module)
assert not blocked.intersection(sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_public_catalog_is_versioned_immutable_and_json_safe():
    from diffgw_core.waveform_definitions.schema import validate_catalog
    from diffgw.waveforms import describe_waveform, get_waveform, list_waveforms

    definitions = list_waveforms()
    validate_catalog(definitions)
    assert isinstance(definitions, tuple)
    assert {definition.id for definition in definitions} == MODEL_IDS
    assert len({definition.id for definition in definitions}) == len(definitions)

    for definition in definitions:
        assert get_waveform(definition.id) is definition
        assert definition.reference_version.endswith("/v1")
        assert definition.reference_module.startswith("diffgw_reference.")
        assert (Path(__file__).resolve().parents[1] / "docs" / definition.spec_page).is_file()
        assert definition.components
        assert definition.sources
        if definition.id == "MemoryWaveform22Direct":
            assert {"times", "h22", "distance_m"}.issubset(
                {parameter.name for parameter in definition.parameters})
        elif definition.id == "KerrRingdown220Direct":
            assert {"final_mass", "final_spin", "distance", "energy_fraction"}.issubset(
                {parameter.name for parameter in definition.parameters})
        elif definition.id == "ContinuousWaveH1Direct":
            assert {"t", "t0", "f0", "ra", "dec", "h0"}.issubset(
                {parameter.name for parameter in definition.parameters})
        else:
            assert {"mass1", "mass2"}.issubset({parameter.name for parameter in definition.parameters})
        assert definition.units
        assert definition.conventions
        assert definition.options
        assert len({component.id for component in definition.components}) == len(definition.components)
        for component in definition.components:
            assert component.id
            assert component.spec_version.endswith("/v1")
            assert component.scope
            assert component.variant
            assert component.source_refs

        with pytest.raises(FrozenInstanceError):
            definition.id = "replaced"

        description = describe_waveform(definition.id)
        assert description["id"] == definition.id
        assert json.loads(json.dumps(description)) == description


def test_catalog_validation_rejects_broken_identity_version_and_provenance():
    from diffgw_core.waveform_definitions.schema import SourceRef, validate_catalog
    from diffgw.waveforms import get_waveform

    model = get_waveform("TaylorF2")
    component = model.components[0]
    with pytest.raises(ValueError, match="waveform ID"):
        validate_catalog((model, model))
    with pytest.raises(ValueError, match="reference version"):
        validate_catalog((replace(model, reference_version="unversioned"),))
    with pytest.raises(ValueError, match="canonical specification page"):
        validate_catalog((replace(model, spec_page="../other.md"),))
    with pytest.raises(ValueError, match="duplicate canonical specification page"):
        validate_catalog((model, replace(model, id="TaylorF2Other")))
    with pytest.raises(ValueError, match="component ID"):
        validate_catalog((replace(model, components=(replace(component, id="bad-id"),)),))
    with pytest.raises(ValueError, match="component version"):
        validate_catalog((replace(model, components=(replace(component, spec_version="unversioned"),)),))
    with pytest.raises(ValueError, match="variant"):
        validate_catalog((replace(model, components=(replace(component, variant=""),)),))
    with pytest.raises(ValueError, match="native choices"):
        validate_catalog((replace(model, options=(replace(
            model.options[0], native_choices=("not-a-choice",)
        ),)),))
    with pytest.raises(ValueError, match="source reference"):
        validate_catalog((replace(model, components=(replace(
            component, source_refs=(SourceRef("unknown-source", "Eq. 1"),)
        ),)),))


def test_catalog_can_reuse_unchanged_components_across_compositions():
    from diffgw_core.waveform_definitions.schema import validate_catalog
    from diffgw.waveforms import get_waveform

    model = get_waveform("TaylorF2")
    component = model.components[0]
    extension = replace(model, id="TaylorF2Extended",
                        spec_page="math/taylorf2-extended.md", components=(component,))
    validate_catalog((model, extension))
    with pytest.raises(ValueError, match="conflicting component ID"):
        validate_catalog((model, replace(extension, components=(
            replace(component, scope="different mathematics"),
        ))))


def test_verification_claims_require_bound_components_and_supported_options():
    from diffgw_core.waveform_definitions.schema import VerificationClaim, validate_catalog
    from diffgw.waveforms import get_waveform

    model = get_waveform("TaylorF2")
    component = model.components[0]
    option = model.options[0]
    claim = VerificationClaim(
        component_ids=(component.id,), scope=component.scope, variant=component.variant,
        backends=("torch",), options=((option.name, option.default),),
        domain="positive frequency", precision="float64", device="cpu",
        category="implementation_verification", status="sampled_pass",
        evidence_path="tests/test_taylorf2_math_reference.py",
    )
    validate_catalog((replace(model, claims=(claim,)),))
    with pytest.raises(ValueError, match="claim component IDs"):
        validate_catalog((replace(model, claims=(replace(claim, component_ids=("unknown-component",)),)),))
    with pytest.raises(ValueError, match="claim option"):
        validate_catalog((replace(model, claims=(replace(claim, options=((option.name, "unsupported"),)),)),))


def test_lookup_accepts_canonical_case_and_rejects_unknown_model():
    from diffgw.waveforms import get_waveform

    for model_id in MODEL_IDS:
        assert get_waveform(model_id.lower()).id == model_id
    with pytest.raises(KeyError):
        get_waveform("not-a-waveform")


def test_reference_only_xas_choices_are_not_advertised_as_native():
    from diffgw.waveforms import describe_waveform

    options = {item["name"]: item for item in describe_waveform("IMRPhenomXAS")["options"]}
    assert options["intermediate_amplitude"]["native_choices"] == ["carrier_quartic"]
    assert options["intermediate_phase"]["native_choices"] == ["carrier_sign"]


def test_xas_spec_component_rows_match_catalog():
    """Every XAS component row on the canonical page has a public binding."""
    from diffgw.waveforms import get_waveform

    model = get_waveform("IMRPhenomXAS")
    page = Path(__file__).resolve().parents[1] / "docs" / model.spec_page
    rows = re.findall(r"^\| `(phenomxas\.[^`]+\.v\d+)` \|", page.read_text(), re.MULTILINE)
    assert len(rows) == len(set(rows))
    assert set(rows) == {component.id for component in model.components}


def test_optional_reference_facade_loads_declared_modules():
    for dependency in ("sympy", "mpmath", "scipy"):
        pytest.importorskip(dependency)
    from diffgw.reference import get_reference_module
    from diffgw.waveforms import list_waveforms

    for model in list_waveforms():
        assert get_reference_module(model.id) is import_module(model.reference_module)
        component = model.components[0]
        assert get_reference_module(model.id, component_id=component.id) is import_module(component.reference_module)
    with pytest.raises(KeyError):
        get_reference_module("TaylorF2", component_id="unknown-component")


def test_catalog_component_ids_and_versions_match_reference_modules():
    """The handwritten catalog must track the independent executable specs."""
    for dependency in ("sympy", "mpmath", "scipy"):
        pytest.importorskip(dependency)
    from diffgw.waveforms import list_waveforms

    special_versions = {
        ("diffgw_reference.imrphenomxas_waveform", "phenomxas.waveform.mass_scaling.v1"):
            "MASS_SCALING_VERSION",
    }
    for model in list_waveforms():
        reference = import_module(model.reference_module)
        assert model.reference_version == reference.SPEC_VERSION, model.id
        for component in model.components:
            reference = import_module(component.reference_module)
            assert component.id in reference.COMPONENT_IDS.values(), component.id
            version_attribute = special_versions.get((component.reference_module, component.id), "SPEC_VERSION")
            assert component.spec_version == getattr(reference, version_attribute), component.id


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_catalog_models_are_bound_to_public_backend_dispatch(backend):
    if find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    root = Path(__file__).resolve().parents[1]
    code = f'''
import diffgw
model_ids = {{"TaylorF2", "TaylorF2_HM", "TaylorF2_Eccentric", "IMRPhenomD", "IMRPhenomXAS"}}
frontend = getattr(diffgw, {backend!r})
assert model_ids.issubset(set(diffgw.list_approximants(backend={backend!r})))
for model_id in model_ids:
    assert getattr(frontend, model_id).definition_id == model_id
'''
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
