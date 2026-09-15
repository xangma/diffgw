import json
import hashlib
import sys
import types

import numpy as np
import pytest

import scripts.prepare_nr33_calibration_v2 as v2
from scripts.prepare_nr33_calibration_v2 import (
    _required_metadata,
    inventory_files,
    load_selection,
    validate_exact_metadata,
    validate_record,
    verify_download,
)


def _file(key, checksum="md5:" + "a" * 32, size=3):
    return {"key": key, "checksum": checksum, "size": size,
            "links": {"content": "https://example.invalid/" + key}}


def _record(sxs="SXS:BBH:0030", lev="Lev3", keywords=None):
    return {"id": 123, "metadata": {"simulation_name": f"d14.0_q3.0/" + lev,
            "alternative_names": sxs, "keywords": keywords or [],
            "creators": [{"name": "A. Author"}], "license": {"id": "cc-by-4.0"}},
            "files": [_file(f"{lev}:Strain_N3.h5"), _file(f"{lev}:Strain_N3.json"),
                      _file(f"{lev}:metadata.json")]}


def test_swapped_record_and_resolution_are_rejected():
    row = {"sxs_id": "SXS:BBH:0030", "record_id": "123", "resolution": "Lev3",
           "extrapolation_order": 3, "split": "holdout"}
    with pytest.raises(ValueError, match="record ID mismatch"):
        validate_record(row, "999", _record())
    with pytest.raises(ValueError, match="does not identify"):
        validate_record({**row, "sxs_id": "SXS:BBH:0166"}, "123", _record())
    with pytest.raises(ValueError, match="resolution mismatch"):
        validate_exact_metadata(_record()["metadata"], {**row, "resolution": "Lev4"})
    with pytest.raises(ValueError, match="missing exact selected files"):
        inventory_files(_record(), {**row, "resolution": "Lev4"})


def test_caltech_entries_shape_uses_exact_lev_identity():
    row = {"sxs_id": "SXS:BBH:0166", "record_id": "28pv5-4wj69", "resolution": "Lev3",
           "extrapolation_order": 3, "split": "holdout"}
    record = _record(sxs="SXS:BBH:3630")
    record["id"] = "28pv5-4wj69"
    record["metadata"] = {"title": "Binary black-hole simulation SXS:BBH:3630"}
    record["files"] = {"entries": {item["key"]: item for item in record["files"]}}
    validate_record(row, row["record_id"], record)
    assert set(inventory_files(record, row)) == {"h5", "sidecar", "metadata", "source_format"}
    exact = {"alternative_names": ["SXS:BBH:3630"], "simulation_name": "0166/Lev3"}
    validate_exact_metadata(exact, row)
    embedded = _record()
    embedded["files"] = {"entries": {
        "Lev3:ExtraWaveforms.h5": _file("Lev3:ExtraWaveforms.h5"),
        "Lev3:ExtraWaveforms.json": _file("Lev3:ExtraWaveforms.json"),
        "Lev3:metadata.json": _file("Lev3:metadata.json")}}
    with pytest.raises(ValueError, match="explicit embedded"):
        inventory_files(embedded, {**row, "h5_key": "Lev4:ExtraWaveforms.h5",
                                   "sidecar_key": "Lev4:ExtraWaveforms.json",
                                   "h5_group": "/Strain_N3.dir"})


def test_remnant_normalization_and_time_conversion_are_derived():
    fields = _required_metadata({"reference_mass1": 0.75, "reference_mass2": 0.25,
        "remnant_mass": 0.971, "remnant_dimensionless_spin": [0.0, 0.0, 0.5],
        "reference_time": 640.0, "relaxation_time": 640.0})
    assert fields["q"] == pytest.approx(3.0)
    assert fields["remnant_mass_fraction"] == pytest.approx(0.971)
    assert fields["remnant_spin_magnitude"] == pytest.approx(0.5)
    assert fields["reference_time_M"] == pytest.approx(640.0)
    assert fields["relaxation_time_M"] == pytest.approx(640.0)
    assert fields["reference_time_M"] != pytest.approx(0.0)


def test_memory_time_is_not_required_for_relaxation():
    fields = _required_metadata({"reference_mass1": 0.8, "reference_mass2": 0.2,
        "remnant_mass": 0.9, "remnant_spin": [0.1, 0.2, 0.3],
        "reference_time": 100.0, "relaxation_time": 200.0})
    assert fields["relaxation_time_code"] == 200.0
    with pytest.raises(ValueError, match="numeric"):
        _required_metadata({"reference_mass1": 0.8, "reference_mass2": 0.2,
            "remnant_mass": 0.9, "remnant_spin": [0.1, 0.2, 0.3],
            "reference_time": None, "relaxation_time": 200.0})


def test_deprecated_record_is_rejected():
    row = {"sxs_id": "SXS:BBH:0030", "record_id": "123", "resolution": "Lev3",
           "extrapolation_order": 3, "split": "holdout"}
    with pytest.raises(ValueError, match="deprecated"):
        validate_record(row, "123", _record(keywords=["Non-Spinning", "deprecated"]))
    superseded = _record()
    superseded["metadata"]["superseded_by"] = ["SXS:BBH:9999"]
    with pytest.raises(ValueError, match="superseded"):
        validate_record(row, "123", superseded)


def test_checksum_and_size_are_verified(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_download("unused", path, "md5:" + "0" * 32, 3)
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_download("unused", path, "md5:" + "0" * 32, 3)


def test_selection_requires_policy_and_candidate_provenance(tmp_path):
    row = {"sxs_id": "SXS:BBH:0030", "record_id": "123", "resolution": "Lev3",
           "extrapolation_order": 3, "split": "holdout", "target_q": 3.0,
           "candidate_rank": 1}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema_version": 1,
        "policy_sha256": "a" * 64, "inventory_sha256": "b" * 64, "cases": [row]}))
    assert load_selection(path)["cases"][0]["candidate_rank"] == 1
    path.write_text(json.dumps({"schema_version": 1, "cases": [row]}))
    with pytest.raises(ValueError, match="policy_sha256"):
        load_selection(path)


def test_prepare_keeps_sidecar_memory_separate_from_lev_relaxation(tmp_path, monkeypatch):
    lev_metadata = {"reference_mass1": 0.75, "reference_mass2": 0.25,
                    "remnant_mass": 0.971, "remnant_dimensionless_spin": [0, 0, 0.5],
                    "reference_time": 640, "relaxation_time": 640}
    payloads = {
        "h5": b"h5",
        "sidecar": json.dumps({"data_info": {"data_type": "h", "spin_weight": -2,
            "m_is_scaled_out": True, "r_is_scaled_out": True, "ell_min": 2, "ell_max": 8},
            "modifications": {"add_memory": {"integration_start_time": 261}}}).encode(),
        "metadata": json.dumps(lev_metadata).encode(),
    }
    files = []
    for name, content in payloads.items():
        files.append({"key": {"h5": "Lev3:Strain_N3.h5", "sidecar": "Lev3:Strain_N3.json",
                               "metadata": "Lev3:metadata.json"}[name],
                      "checksum": "md5:" + hashlib.md5(content).hexdigest(), "size": len(content),
                      "links": {"content": name}})
    record = {"id": 123, "metadata": {"alternative_names": "SXS:BBH:0030",
              "simulation_name": "d14/Lev3"}, "files": files}
    selection = {"schema_version": 1, "policy_sha256": "a" * 64, "cases": [{
        "sxs_id": "SXS:BBH:0030", "record_id": "123", "resolution": "Lev3",
        "extrapolation_order": 3, "split": "holdout", "target_q": 3, "candidate_rank": 1}]}
    selection_path = tmp_path / "selection.json"
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"records": {"123": record}}))
    selection["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    selection_path.write_text(json.dumps(selection))

    def fake_download(url, path, expected_md5, expected_size):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[{"h5": "h5", "sidecar": "sidecar", "metadata": "metadata"}[url]])

    monkeypatch.setattr(v2, "verify_download", fake_download)

    def fake_extract(row, raw_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (row["raw_stem"] + ".npz")
        path.write_bytes(b"npz")
        return {"npz": str(path), "peak_time": 0.0, "npz_sha256": hashlib.sha256(b"npz").hexdigest(),
                "sxs_package_version": "2026.0.3", "source_frame_type": "inertial",
                "extraction_frame_type": "inertial", "frame_transform": "none"}

    report = v2.prepare(selection_path, inventory_path, tmp_path / "raw", tmp_path / "npz",
                        extractor=fake_extract)
    metadata = report["cases"][0]["metadata"]
    assert metadata["relaxation_time"] == pytest.approx(640)
    assert metadata["memory_integration_start_time_waveform_M"] == 261


def test_embedded_decoder_passes_group_keyword_to_pinned_sxs(tmp_path, monkeypatch):
    calls = []

    class FakeWaveform:
        frame_type = "inertial"
        t = np.array([0.0, 1.0, 2.0])
        data = np.ones((3, 2), dtype=complex)

        @staticmethod
        def index(l, m):
            return 0 if (l, m) == (3, 3) else 1

    fake_sxs = types.SimpleNamespace(
        __version__ = "2026.0.3",
        load=lambda *args, **kwargs: (calls.append((args, kwargs)) or FakeWaveform()),
    )
    monkeypatch.setitem(sys.modules, "sxs", fake_sxs)
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "npz"
    raw_dir.mkdir()
    (raw_dir / "case.h5").write_bytes(b"placeholder")
    (raw_dir / "case.json").write_text(json.dumps({"/Strain_N3.dir": {}}))
    result = v2.extract_embedded_case(
        {"sxs_id": "SXS:BBH:0030", "raw_stem": "case", "h5_group": "/Strain_N3.dir"},
        raw_dir, out_dir)
    assert calls == [((str(raw_dir / "case.h5"),), {"group": "/Strain_N3.dir", "download": False})]
    assert result["sxs_package_version"] == "2026.0.3"
