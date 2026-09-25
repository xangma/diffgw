"""The unified model inventory works with either optional frontend."""

from importlib import import_module
import json
from pathlib import Path

import pytest

import diffgw


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_unified_callable_inventory_lists_only_routable_models(backend):
    pytest.importorskip({"torch": "torchwave", "jax": "jaxwave"}[backend])
    models = diffgw.list_approximants(backend=backend, inventory="callable")
    dispatch = import_module(f"{backend}wave.waveforms")._APPROXIMANTS
    assert all(name.lower().replace("_", "") in dispatch for name in models)
    assert "SEOBNRv4_ROM" in models
    assert "SEOBNR_ODE" not in models
    categories = diffgw.list_approximants(backend=backend, by_category=True, inventory="callable")
    assert categories["time_domain"] == diffgw.list_approximants(
        "time_domain", backend=backend, inventory="callable"
    )
    assert set(models) == set().union(*map(set, categories.values()))


def test_frontend_inventories_agree():
    pytest.importorskip("torchwave")
    pytest.importorskip("jaxwave")
    assert diffgw.list_approximants(backend="torch") == diffgw.list_approximants(
        backend="jax"
    )
    assert diffgw.list_approximants(backend="torch", inventory="callable") == diffgw.list_approximants(
        backend="jax", inventory="callable"
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_default_historical_reference_inventory_tracks_manifest(backend):
    pytest.importorskip({"torch": "torchwave", "jax": "jaxwave"}[backend])
    manifest_path = Path(__file__).resolve().parents[1] / "docs" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "SEOBNRv4_ROM" if name == "SEOBNR_ROM" else name
        for name, record in manifest["models"].items()
        if record["status"].startswith("verified_declared_cases")
        and record["validation"]["catalog"]["cases_total"] > 0
        and record["validation"]["catalog"]["cases_passed"] == record["validation"]["catalog"]["cases_total"]
        and "full_polarizations" in record["validation"]["catalog"]["verification_scopes"]
        and record["validation"]["holdout"]["cases_passed"] == record["validation"]["holdout"]["cases_total"]
    }
    callable_models = diffgw.list_approximants(backend=backend, inventory="callable")
    reference_models = diffgw.list_approximants(backend=backend)
    assert set(reference_models) == expected
    assert set(reference_models) <= set(callable_models)
    assert "IMRPhenomP" in callable_models and "IMRPhenomP" not in reference_models
    assert "IMRPhenomPv2" in reference_models
    assert not {"IMRPhenomXHM", "IMRPhenomXPHM", "TaylorF2_HM", "IMRPhenomTP", "IMRPhenomTPHM"} & set(reference_models)
    categories = diffgw.list_approximants(backend=backend, by_category=True)
    assert categories["time_domain"] == []
    assert set().union(*map(set, categories.values())) == set(reference_models)
    with pytest.raises(ValueError, match="inventory"):
        diffgw.list_approximants(backend=backend, inventory="release")
