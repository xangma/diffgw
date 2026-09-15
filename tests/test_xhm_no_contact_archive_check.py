from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile

def _load_checker():
    path = "scripts/xhm_no_contact_archive_check.py"
    spec = importlib.util.spec_from_file_location("xhm_no_contact_archive_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _add_file(archive, name):
    payload = b"metadata only"
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _make_archive(path, *, complete=False):
    with tarfile.open(path, "w:gz") as archive:
        _add_file(archive, "anc/ParspaceFits/README")
        if complete:
            for prefix, count in CHECKER.EXPECTED_PREFIX_COUNTS.items():
                for index in range(count):
                    _add_file(archive, f"{prefix}synthetic-{index}")
            _add_file(archive, "anc/ParspaceFits/IMRPhenomXHM_AmplitudeFits.nb")
            _add_file(archive, "anc/ParspaceFits/IMRPhenomXHM_PhaseFits.nb")


def test_inventory_checker_reports_incomplete_metadata_only_archive(tmp_path):
    archive = tmp_path / "paper.tar.gz"
    _make_archive(archive)
    report = CHECKER.inspect_archive(archive)
    assert report["file_count"] == 1
    assert report["metadata_members"] == ["anc/ParspaceFits/README"]
    assert report["inventory_verified"] is False
    assert report["identity_verified"] is False
    assert report["sha256_status"] == "unchecked"
    assert report["verified"] is False


def test_inventory_checker_checks_hash_without_extracting_members(tmp_path):
    archive = tmp_path / "paper.tar.gz"
    _make_archive(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = CHECKER.inspect_archive(archive, expected_sha256=digest)
    assert report["sha256_status"] == "match"
    assert report["identity_verified"] is True
    assert CHECKER.inspect_archive(archive, expected_sha256="0" * 64)["verified"] is False


def test_complete_inventory_without_expected_hash_is_not_verified(tmp_path):
    archive = tmp_path / "paper.tar.gz"
    _make_archive(archive, complete=True)
    report = CHECKER.inspect_archive(archive)
    assert report["inventory_verified"] is True
    assert report["sha256_status"] == "unchecked"
    assert report["identity_verified"] is False
    assert report["verified"] is False


def test_cli_returns_failure_for_unverified_inventory(tmp_path):
    archive = tmp_path / "paper.tar.gz"
    _make_archive(archive)
    assert CHECKER.main([str(archive)]) == 1
