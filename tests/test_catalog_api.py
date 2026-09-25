"""The unified model inventory works with either optional frontend."""

from importlib import import_module

import pytest

import diffgw


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_unified_inventory_lists_only_routable_models(backend):
    pytest.importorskip({"torch": "torchwave", "jax": "jaxwave"}[backend])
    models = diffgw.list_approximants(backend=backend)
    dispatch = import_module(f"{backend}wave.waveforms")._APPROXIMANTS
    assert all(name.lower().replace("_", "") in dispatch for name in models)
    assert "SEOBNRv4_ROM" in models
    assert "SEOBNR_ODE" not in models
    categories = diffgw.list_approximants(backend=backend, by_category=True)
    assert categories["time_domain"] == diffgw.list_approximants(
        "time_domain", backend=backend
    )
    assert set(models) == set().union(*map(set, categories.values()))


def test_frontend_inventories_agree():
    pytest.importorskip("torchwave")
    pytest.importorskip("jaxwave")
    assert diffgw.list_approximants(backend="torch") == diffgw.list_approximants(
        backend="jax"
    )
