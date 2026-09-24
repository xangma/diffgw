"""Release APIs expose only the official SEOBNRv4 ROM evaluator."""

from importlib import import_module, resources, util

import pytest

import diffgw


@pytest.fixture(params=["torch", "jax"])
def frontend(request):
    pytest.importorskip(request.param)
    package = {"torch": "torchwave", "jax": "jaxwave"}[request.param]
    return request.param, package, import_module(f"{package}.approximants.rom")


def test_rom_class_alias_matches_official_dispatch(frontend):
    backend, package, rom = frontend
    assert rom.SEOBNR_ROM is rom.OfficialSEOBNRv4ROM
    public = import_module("torchwave.approximants" if backend == "torch" else package)
    assert public.SEOBNR_ROM is rom.OfficialSEOBNRv4ROM


@pytest.mark.parametrize("approximant", ["SEOBNR_ROM", "SEOBNRv4_ROM", "official_seobnr_v4_rom"])
def test_public_rom_names_require_official_external_data(frontend, approximant, tmp_path):
    backend, _, rom = frontend
    missing = tmp_path / rom.OFFICIAL_ROM_FILENAME
    with pytest.raises(FileNotFoundError):
        diffgw.get_fd_waveform(
            approximant, backend=backend, mass1=30.0, mass2=20.0,
            f_lower=20.0, f_final=40.0, delta_f=10.0, model_path=str(missing),
        )


@pytest.mark.parametrize("approximant", ["seobnrv5rom", "SEOBNRv5_ROM"])
def test_unvalidated_v5_rom_alias_is_rejected(frontend, approximant):
    backend, _, _ = frontend
    with pytest.raises(ValueError, match="Approximant .* not recognized"):
        diffgw.get_fd_waveform(
            approximant, backend=backend, mass1=30.0, mass2=20.0,
            f_lower=20.0, f_final=40.0, delta_f=10.0,
        )


def test_prototype_modules_and_training_exports_are_absent(frontend):
    _, package, rom = frontend
    for module in ("surrogate", "generator"):
        assert util.find_spec(f"{package}.approximants.rom.{module}") is None
    for name in (
        "generate_eob_training_grid", "compute_svd_reduced_basis",
        "fit_parameter_mapping", "train_and_export_surrogate",
    ):
        assert not hasattr(rom, name)
        assert name not in rom.__all__


@pytest.mark.parametrize("package, filename", [
    ("torchwave", "seobnr_surrogate.pt"),
    ("diffgw_core", "seobnr_surrogate.npz"),
])
def test_legacy_rom_tables_are_not_packaged(package, filename):
    if package == "torchwave":
        pytest.importorskip("torch")
    assert not resources.files(package).joinpath("data", filename).is_file()
