"""Offline contract tests for external SEOBNRv4 ROM resolution."""

from pathlib import Path

import pytest
import torch

from torchwave.approximants.rom import official, official_wrapper
from torchwave.external_data import resolve_dataset_path


def _asset(directory: Path) -> Path:
    directory.mkdir()
    asset = directory / official.OFFICIAL_ROM_FILENAME
    asset.write_bytes(b"fixture")
    return asset


def test_external_data_precedence_and_explicit_path(tmp_path, monkeypatch):
    explicit = _asset(tmp_path / "explicit")
    torchwave_dir = tmp_path / "torchwave"
    lal_dir = tmp_path / "lal"
    torchwave_asset = _asset(torchwave_dir)
    lal_asset = _asset(lal_dir)
    monkeypatch.setenv("TORCHWAVE_DATA_PATH", str(torchwave_dir))
    monkeypatch.setenv("LAL_DATA_PATH", str(lal_dir))

    assert resolve_dataset_path("seobnrv4-rom-v3", verify=False) == torchwave_asset.resolve()
    assert resolve_dataset_path("seobnrv4-rom-v3", path=explicit, verify=False) == explicit.resolve()
    assert resolve_dataset_path("seobnrv4-rom-v3", path=explicit.parent, verify=False) == explicit.resolve()
    assert lal_asset.exists()  # documents that the lower-priority source remains untouched


def test_schema_resolution_does_not_verify_by_default(monkeypatch, tmp_path):
    path = tmp_path / official.OFFICIAL_ROM_FILENAME
    path.write_bytes(b"fixture")
    calls = []

    def resolve(dataset, *, path=None, verify=True):
        calls.append((dataset, path, verify))
        return path

    monkeypatch.setattr("torchwave.external_data.resolve_dataset_path", resolve)
    assert official.resolve_official_rom_path(path) == path
    assert calls == [("seobnrv4-rom-v3", path, False)]


def test_wrapper_rejects_corrupt_rom_before_loading(monkeypatch, tmp_path):
    path = tmp_path / official.OFFICIAL_ROM_FILENAME
    path.write_bytes(b"corrupt")
    loads = []
    hashes = []

    monkeypatch.setattr(official_wrapper, "resolve_official_rom_path", lambda *a, **k: path)

    def digest(candidate):
        hashes.append(candidate)
        return "0" * 64

    monkeypatch.setattr(official_wrapper, "sha256_file", digest)
    monkeypatch.setattr(official_wrapper, "load_official_rom", lambda *a, **k: loads.append(a))
    with pytest.raises(ValueError, match="Unexpected SHA-256"):
        official_wrapper.OfficialSEOBNRv4ROM(path)
    assert len(hashes) == 1
    assert loads == []


def test_wrapper_hashes_once_before_loading_all_groups(monkeypatch, tmp_path):
    path = tmp_path / official.OFFICIAL_ROM_FILENAME
    path.write_bytes(b"valid fixture for mocked loader")
    events = []

    class Table:
        amp_coeff = torch.zeros(1, dtype=torch.float64)

    table = Table()

    def digest(candidate):
        events.append(("hash", candidate))
        return official.OFFICIAL_ROM_SHA256

    def load(candidate, group):
        events.append(("load", group))
        return table

    class Glue:
        def __init__(self, low, high, **kwargs):
            self.low = low

    monkeypatch.setattr(official_wrapper, "resolve_official_rom_path", lambda *a, **k: path)
    monkeypatch.setattr(official_wrapper, "sha256_file", digest)
    monkeypatch.setattr(official_wrapper, "load_official_rom", load)
    monkeypatch.setattr(official_wrapper, "OfficialROMGlue", Glue)

    official_wrapper.OfficialSEOBNRv4ROM(path)
    assert events == [("hash", path), ("load", "sub1"), ("load", "sub2"), ("load", "sub3")]
