#!/usr/bin/env python3
"""Prepare an auditable NR33 manifest from an explicit record selection.

The selection and Zenodo inventory are inputs, never inferred from catalog
searches.  Raw files live outside the repository.  Extraction delegates to
the frozen ``extract_case`` decoder in ``prepare_nr33_calibration_data.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import numpy as np

try:
    from scripts.prepare_nr33_calibration_data import extract_case as frozen_extract_case
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from prepare_nr33_calibration_data import extract_case as frozen_extract_case


SCHEMA = "torchwave.nr33-calibration.v2"
DEFAULT_RAW = Path("/tmp/torchwave-nr33-v2/raw")
DEFAULT_NPZ = Path("/tmp/torchwave-nr33-v2/npz")
DEFAULT_SXS_VERSION = "2026.0.3"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _records(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = inventory.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError("inventory.records must be a non-empty record_id mapping")
    return {str(k): v for k, v in records.items() if isinstance(v, dict)}


def load_selection(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("selection must be an object with a cases list")
    if value.get("schema_version") != 1:
        raise ValueError("selection schema_version must be 1")
    policy_sha = value.get("policy_sha256")
    if not isinstance(policy_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", policy_sha):
        raise ValueError("selection policy_sha256 must be a 64-character SHA-256 hex digest")
    inventory_sha = value.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", inventory_sha):
        raise ValueError("selection inventory_sha256 must be a 64-character SHA-256 hex digest")
    required = {"sxs_id", "record_id", "resolution", "extrapolation_order", "split",
                "target_q", "candidate_rank"}
    seen = set()
    for row in value["cases"]:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("each selection row must contain " + ", ".join(sorted(required)))
        key = (str(row["sxs_id"]), str(row["record_id"]), str(row["resolution"]), int(row["extrapolation_order"]))
        if key in seen:
            raise ValueError("duplicate selection row: " + repr(key))
        seen.add(key)
        if row["split"] not in {"training", "holdout", "validation_only"}:
            raise ValueError("invalid split in selection: " + str(row["split"]))
        if int(row["extrapolation_order"]) < 1:
            raise ValueError("extrapolation_order must be positive")
        try:
            target_q = float(row["target_q"])
            candidate_rank = int(row["candidate_rank"])
            if not np.isfinite(target_q) or target_q <= 1:
                raise ValueError("target_q must be finite and greater than one")
            if candidate_rank < 1:
                raise ValueError("candidate_rank must be positive")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid target_q/candidate_rank in selection row: {row!r}") from exc
    if not value["cases"]:
        raise ValueError("selection cases cannot be empty")
    return value


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def validate_record(row: dict[str, Any], record_id: str, record: dict[str, Any]) -> None:
    if str(record.get("id")) != record_id:
        raise ValueError(f"record ID mismatch: selection {record_id!r}, response {record.get('id')!r}")
    selected_id = str(row["sxs_id"]).casefold()
    meta = _metadata(record)
    aliases = meta.get("alternative_names", meta.get("alternative_name", []))
    if isinstance(aliases, str):
        aliases = [aliases]
    identifiers = meta.get("identifiers", [])
    if isinstance(identifiers, str):
        identifiers = [identifiers]
    explicit_ids = {str(x).casefold() for x in [*aliases, *identifiers]
                    if isinstance(x, (str, int, float))}
    # Require an explicit simulation identifier when the provider supplies one.
    if explicit_ids:
        if selected_id not in explicit_ids:
            raise ValueError(f"record {record_id} does not identify selected simulation {row['sxs_id']}")
    # Some provider records identify the simulation only inside the downloaded
    # Lev metadata (their landing-page title may contain the provider record
    # number).  That exact metadata is checked after download below.
    keywords = meta.get("keywords", [])
    if any("deprecated" in str(k).casefold() for k in (keywords if isinstance(keywords, list) else [keywords])):
        raise ValueError(f"record {record_id} is marked deprecated")
    if any(bool(meta.get(k)) for k in ("deprecated", "is_deprecated")):
        raise ValueError(f"record {record_id} is marked deprecated")
    if meta.get("superseded_by"):
        raise ValueError(f"record {record_id} is superseded")


def inventory_files(record: dict[str, Any], row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = record.get("files")
    if isinstance(files, dict) and isinstance(files.get("entries"), dict):
        # Data Caltech's API wraps entries by filename; normalize that shape
        # while retaining the exact filename and inventory checks below.
        files = [{**item, "key": key} for key, item in files["entries"].items()
                 if isinstance(item, dict)]
    if not isinstance(files, list):
        raise ValueError("record files must be a list or an entries mapping")
    by_key = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError("each inventory file requires key")
        key = item["key"]
        checksum = str(item.get("checksum", ""))
        size = item.get("size")
        links = item.get("links")
        if not checksum.startswith("md5:") or len(checksum) != 36:
            raise ValueError(f"inventory file {key} lacks an exact MD5 checksum")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"inventory file {key} lacks an exact size")
        if not isinstance(links, dict) or not isinstance(links.get("content"), str):
            raise ValueError(f"inventory file {key} lacks links.content")
        if key in by_key:
            raise ValueError("duplicate inventory key: " + key)
        by_key[key] = item
    lev = str(row["resolution"])
    order = int(row["extrapolation_order"])
    direct = [f"{lev}:Strain_N{order}.h5", f"{lev}:Strain_N{order}.json"]
    metadata = f"{lev}:metadata.json"
    requested_h5 = row.get("h5_key")
    requested_sidecar = row.get("sidecar_key")
    requested_group = row.get("h5_group")
    if requested_h5 is not None or requested_sidecar is not None or requested_group is not None:
        expected_group = f"/Strain_N{order}.dir"
        expected_h5, expected_sidecar = f"{lev}:ExtraWaveforms.h5", f"{lev}:ExtraWaveforms.json"
        if (requested_h5 != expected_h5 or requested_sidecar != expected_sidecar or
                requested_group != expected_group or
                requested_h5 not in by_key or requested_sidecar not in by_key or metadata not in by_key):
            raise ValueError("explicit embedded source keys/group do not match selected extrapolation order")
        return {"h5": by_key[requested_h5], "sidecar": by_key[requested_sidecar],
                "metadata": by_key[metadata], "source_format": "extra", "group": expected_group}
    if all(key in by_key for key in [*direct, metadata]):
        return {"h5": by_key[direct[0]], "sidecar": by_key[direct[1]],
                "metadata": by_key[metadata], "source_format": "strain"}
    extra = [f"{lev}:ExtraWaveforms.h5", f"{lev}:ExtraWaveforms.json"]
    if all(key in by_key for key in [*extra, metadata]):
        return {"h5": by_key[extra[0]], "sidecar": by_key[extra[1]],
                "metadata": by_key[metadata], "source_format": "extra",
                "group": f"/Strain_N{order}.dir"}
    required = [*direct, metadata]
    missing = [key for key in required if key not in by_key]
    raise ValueError("record is missing exact selected files: " + ", ".join(missing))


def verify_download(url: str, path: Path, expected_md5: str, expected_size: int,
                    opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    """Download once, then require exact inventory size and MD5."""
    expected = expected_md5.removeprefix("md5:").lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        parsed = urllib.parse.urlsplit(url)
        encoded_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%")
        encoded_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path,
                                               parsed.query, parsed.fragment))
        # Data-Caltech inventories have historically emitted literal colons in
        # filename path components.  Keep the inventory URL in provenance, but
        # retry its standards-compliant percent-encoded transport form on 403.
        urls = [url] if encoded_url == url else [url, encoded_url]
        try:
            for attempt, candidate in enumerate(urls):
                try:
                    if opener is urllib.request.urlopen:
                        request = urllib.request.Request(
                            candidate,
                            headers={"User-Agent": "torchwave NR33 calibration downloader/1.0"},
                        )
                        source_context = opener(request, timeout=60)
                    else:
                        # Injectable openers keep the small one-argument test
                        # seam used for offline fixtures.
                        source_context = opener(candidate)
                    with source_context as source, partial.open("wb") as target:
                        total = 0
                        while block := source.read(1024 * 1024):
                            total += len(block)
                            if total > expected_size:
                                raise ValueError(f"download exceeded inventory size for {path}")
                            target.write(block)
                    partial.replace(path)
                    break
                except urllib.error.HTTPError as exc:
                    partial.unlink(missing_ok=True)
                    if attempt == 0 and exc.code == 403 and len(urls) == 2:
                        continue
                    raise
        finally:
            partial.unlink(missing_ok=True)
    if path.stat().st_size != expected_size:
        raise ValueError(f"size mismatch for {path}: {path.stat().st_size} != {expected_size}")
    actual = digest(path, "md5")
    if actual != expected:
        raise ValueError(f"MD5 mismatch for {path}: {actual} != {expected}")


def _required_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    aliases = {"spin": ("remnant_dimensionless_spin", "remnant_spin")}
    for key in ("reference_mass1", "reference_mass2", "remnant_mass", "reference_time", "relaxation_time"):
        if key not in meta:
            raise ValueError("exact Lev metadata missing " + key)
    spin_key = next((k for k in aliases["spin"] if k in meta), None)
    if spin_key is None:
        raise ValueError("exact Lev metadata missing remnant dimensionless spin")
    try:
        m1, m2 = float(meta["reference_mass1"]), float(meta["reference_mass2"])
        remnant = float(meta["remnant_mass"])
        reference_time = float(meta["reference_time"])
        relaxation_time = float(meta["relaxation_time"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid exact Lev metadata numeric value") from exc
    spin = np.asarray(meta[spin_key], dtype=float)
    if not np.isfinite([m1, m2, remnant]).all() or m1 <= 0 or m2 <= 0 or remnant <= 0:
        raise ValueError("invalid exact Lev masses")
    if spin.shape != (3,) or not np.isfinite(spin).all() or np.linalg.norm(spin) >= 1:
        raise ValueError("invalid exact Lev remnant spin vector")
    if not np.isfinite([reference_time, relaxation_time]).all():
        raise ValueError("invalid exact Lev metadata time")
    mref = m1 + m2
    q = m1 / m2
    if q <= 1:
        raise ValueError("derived q must exceed one for fitter-compatible manifest")
    return {"reference_mass1": m1, "reference_mass2": m2, "remnant_mass": remnant,
            "remnant_spin": spin.tolist(), "remnant_spin_magnitude": float(np.linalg.norm(spin)),
            "q": q, "remnant_mass_fraction": remnant / mref,
            "reference_time_code": reference_time,
            "relaxation_time_code": relaxation_time,
            "reference_time_M": float(reference_time / mref),
            "relaxation_time_M": float(relaxation_time / mref),
            "reference_mass_sum": mref}


def validate_exact_metadata(meta: dict[str, Any], row: dict[str, Any]) -> None:
    """Check that downloaded Lev metadata belongs to this selection row."""
    selected_id = str(row["sxs_id"]).casefold()
    aliases = meta.get("alternative_names", meta.get("alternative_name", []))
    if isinstance(aliases, str):
        aliases = [aliases]
    explicit_ids = {str(x).casefold() for x in aliases if isinstance(x, (str, int, float))}
    simulation_name = meta.get("simulation_name")
    sim_token = str(simulation_name).split("/", 1)[0].casefold() if simulation_name else ""
    selected_token = selected_id.rsplit(":", 1)[-1]
    if explicit_ids and selected_id not in explicit_ids and sim_token != selected_token:
        raise ValueError("exact Lev metadata simulation ID mismatch")
    if simulation_name and not str(simulation_name).rstrip().endswith("/" + str(row["resolution"])):
        raise ValueError("exact Lev metadata resolution mismatch")
    keywords = meta.get("keywords", [])
    if any("deprecated" in str(k).casefold() for k in (keywords if isinstance(keywords, list) else [keywords])):
        raise ValueError("exact Lev metadata is marked deprecated")
    if meta.get("superseded_by"):
        raise ValueError("exact Lev metadata is superseded")


def validate_waveform_sidecar(sidecar: dict[str, Any], group: str | None = None) -> dict[str, Any]:
    """Require a strain sidecar whose declared data matches the decoder input."""
    selected = sidecar.get(group) if group is not None else sidecar
    if not isinstance(selected, dict):
        raise ValueError("selected waveform sidecar group metadata is not an object")
    data_info = selected.get("data_info")
    if not isinstance(data_info, dict):
        raise ValueError("selected waveform sidecar lacks data_info")
    if data_info.get("data_type") != "h" or data_info.get("spin_weight") != -2:
        raise ValueError("selected waveform sidecar is not spin-weight -2 strain")
    if data_info.get("m_is_scaled_out") is not True or data_info.get("r_is_scaled_out") is not True:
        raise ValueError("selected waveform sidecar lacks scaled-out strain convention")
    try:
        ell_min, ell_max = int(data_info["ell_min"]), int(data_info["ell_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected waveform sidecar lacks ell range") from exc
    if ell_min > 3 or ell_max < 3:
        raise ValueError("selected waveform sidecar ell range does not include h33")
    return selected


def extract_embedded_case(case: dict[str, Any], raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Decode an ExtraWaveforms group with the pinned SXS package.

    Standalone Strain_N#.h5 files continue through the frozen helper.  This
    small adapter is only for the provider's multi-group ExtraWaveforms file.
    """
    try:
        import sxs
    except ImportError as exc:  # pragma: no cover - depends on decoder env
        raise RuntimeError("embedded extraction requires the pinned sxs package") from exc
    h5_path = raw_dir / f"{case['raw_stem']}.h5"
    sidecar_path = raw_dir / f"{case['raw_stem']}.json"
    group = str(case["h5_group"])
    # ``group`` is an HDF5 loader keyword.  Appending it to the filesystem
    # path makes the path nonexistent and silently bypasses the SXS loader's
    # group dispatch in pinned sxs releases.
    waveform = sxs.load(str(h5_path), group=group, download=False)
    source_frame_type = getattr(waveform, "frame_type", "unknown")
    frame_transform = "none"
    if source_frame_type != "inertial":
        converted = waveform.to_inertial_frame()
        waveform = converted if converted is not None else waveform
        frame_transform = "to_inertial_frame"
    extraction_frame_type = getattr(waveform, "frame_type", "unknown")
    time = np.asarray(waveform.t, dtype=np.float64)
    h33 = np.asarray(waveform.data[:, waveform.index(3, 3)], dtype=np.complex128)
    h22 = np.asarray(waveform.data[:, waveform.index(2, 2)], dtype=np.complex128)
    if (time.ndim != 1 or h33.shape != time.shape or h22.shape != time.shape
            or not np.iscomplexobj(h33) or not np.iscomplexobj(h22)
            or not np.isfinite(time).all() or not np.isfinite(h33).all()
            or not np.isfinite(h22).all() or not np.all(np.diff(time) > 0)):
        raise ValueError(f"unexpected embedded mode shapes for {case['sxs_id']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['raw_stem']}.npz"
    np.savez_compressed(out_path, time=time, h33=h33, h22=h22)
    wave_meta = json.loads(sidecar_path.read_text())
    peak_index = int(np.argmax(np.abs(h22)))
    return {"npz": str(out_path), "n_times": int(time.size), "time_start": float(time[0]),
            "time_end": float(time[-1]), "dt_min": float(np.min(np.diff(time))),
            "dt_max": float(np.max(np.diff(time))), "peak_time_h22": float(time[peak_index]),
            "peak_time": float(time[peak_index]), "npz_sha256": digest(out_path),
            "sxs_package_version": getattr(sxs, "__version__", "unknown"),
            "source_frame_type": source_frame_type, "extraction_frame_type": extraction_frame_type,
            "frame_transform": frame_transform,
            "mode_convention": "complex spin-weight -2 strain h_lm; sxs WaveformModes indexing",
            "waveform_json": wave_meta}


def prepare(selection_path: Path, inventory_path: Path, raw_dir: Path, npz_dir: Path,
            expected_sxs_version: str = DEFAULT_SXS_VERSION,
            expected_policy_sha256: str | None = None,
            extractor: Callable[..., dict[str, Any]] = frozen_extract_case) -> dict[str, Any]:
    selection = load_selection(selection_path)
    if expected_policy_sha256 is not None and selection["policy_sha256"].casefold() != expected_policy_sha256.casefold():
        raise ValueError("selection policy_sha256 does not match expected policy")
    if digest(inventory_path) != selection["inventory_sha256"].casefold():
        raise ValueError("inventory_sha256 does not match selection")
    inventory = json.loads(inventory_path.read_text())
    records = _records(inventory)
    cases = []
    identities = []
    # Exactly one highest-order/highest-resolution primary per simulation.
    for row in selection["cases"]:
        sxs_id, record_id, lev = str(row["sxs_id"]), str(row["record_id"]), str(row["resolution"])
        if record_id not in records:
            raise ValueError(f"selection record {record_id} absent from inventory")
        record = records[record_id]
        validate_record(row, record_id, record)
        selected = inventory_files(record, row)
        # Include Lev so two records for the same SXS ID and N cannot share a
        # local filename and accidentally pair one resolution with another.
        record_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)
        raw_stem = sxs_id.replace(":", "_") + f"_{record_token}_{lev}_N{int(row['extrapolation_order'])}"
        h5_path, sidecar_path = raw_dir / (raw_stem + ".h5"), raw_dir / (raw_stem + ".json")
        for key, path in (("h5", h5_path), ("sidecar", sidecar_path),
                          ("metadata", raw_dir / (raw_stem + "_metadata.json"))):
            item = selected[key]
            verify_download(item["links"]["content"], path, item["checksum"], item["size"])
        exact_meta_path = raw_dir / (raw_stem + "_metadata.json")
        exact_meta = json.loads(exact_meta_path.read_text())
        sidecar_meta = json.loads(sidecar_path.read_text())
        if selected["source_format"] == "extra" and selected["group"] not in sidecar_meta:
            raise ValueError(f"embedded source sidecar lacks exact group {selected['group']}")
        validate_exact_metadata(exact_meta, row)
        fields = _required_metadata(exact_meta)
        extract_row = {"sxs_id": sxs_id, "raw_stem": raw_stem,
                       "h5_group": selected.get("group")}
        if selected["source_format"] == "extra":
            extracted = extract_embedded_case(extract_row, raw_dir, npz_dir)
        else:
            extracted = extractor(extract_row, raw_dir, npz_dir)
        if extracted.get("sxs_package_version") != expected_sxs_version:
            raise RuntimeError(f"expected sxs {expected_sxs_version}, got {extracted.get('sxs_package_version')}")
        if not Path(extracted["npz"]).exists():
            raise RuntimeError("extractor did not produce NPZ")
        group_meta = validate_waveform_sidecar(
            sidecar_meta, selected.get("group") if selected["source_format"] == "extra" else None)
        memory_modifications = group_meta.get("modifications", {}).get("add_memory", {})
        if not isinstance(memory_modifications, dict):
            memory_modifications = {}
        meta = {"sxs_id": sxs_id, "record_id": record_id, "resolution": lev,
                "extrapolation_order": int(row["extrapolation_order"]), "reference_mass1": fields["reference_mass1"],
                "reference_mass2": fields["reference_mass2"], "remnant_mass": fields["remnant_mass"],
                "remnant_spin": fields["remnant_spin"], "final_spin": fields["remnant_spin"],
                "final_spin_magnitude": fields["remnant_spin_magnitude"],
                "final_mass_fraction": fields["remnant_mass_fraction"],
                "remnant_mass_fraction": fields["remnant_mass_fraction"],
                "reference_time": fields["reference_time_M"], "relaxation_time": fields["relaxation_time_M"],
                "reference_time_code": fields["reference_time_code"], "relaxation_time_code": fields["relaxation_time_code"],
                "reference_time_M": fields["reference_time_M"], "relaxation_time_M": fields["relaxation_time_M"],
                "initial_spin1": exact_meta.get("initial_dimensionless_spin1"),
                "initial_spin2": exact_meta.get("initial_dimensionless_spin2"),
                "reference_spin1": exact_meta.get("reference_dimensionless_spin1"),
                "reference_spin2": exact_meta.get("reference_dimensionless_spin2"),
                "eccentricity": exact_meta.get("reference_eccentricity"),
                "time_end": extracted.get("time_end"),
                "peak_time_h22": extracted.get("peak_time_h22", extracted.get("peak_time")),
                "time_conversion": {"source_unit": "metadata code units", "target_unit": "waveform M",
                                     "Mref": fields["reference_mass_sum"], "formula": "time_M=time_code/(reference_mass1+reference_mass2)"},
                "memory_integration_start_time_waveform_M": memory_modifications.get("integration_start_time"),
                "memory_time_note": "selected sidecar group add_memory integration start retained separately; never used as relaxation_time"}
        record_meta = _metadata(record)
        record_links = record.get("links", {}) if isinstance(record.get("links"), dict) else {}
        pids = record.get("pids", {}) if isinstance(record.get("pids"), dict) else {}
        doi = (record_meta.get("doi") or record.get("doi") or
               (pids.get("doi", {}).get("identifier") if isinstance(pids.get("doi"), dict) else None))
        source_paths = {"h5": h5_path, "sidecar": sidecar_path, "metadata": exact_meta_path}
        source_files = {}
        for logical_name, item in selected.items():
            if logical_name not in source_paths:
                continue
            source_files[logical_name] = {"key": item["key"], "checksum": item["checksum"],
                                          "size": item["size"], "content": item["links"]["content"],
                                          "sha256": digest(source_paths[logical_name])}
        source = {"record_id": record_id, "api_id": record.get("id"),
                  "source_format": selected["source_format"], "h5_group": selected.get("group"),
                  "selected_files": source_files,
                  "title": record_meta.get("title"), "alternative_names": record_meta.get("alternative_names"),
                  "simulation_name": record_meta.get("simulation_name"),
                  "doi": doi,
                  "record_url": (record_links.get("html") or record_links.get("self_html") or
                                 record_links.get("self")),
                  "creators": record_meta.get("creators", record.get("creators")),
                  "license": record_meta.get("license", record.get("license")),
                  "metadata_sha256": digest(exact_meta_path),
                  "waveform_sidecar_sha256": digest(sidecar_path)}
        meta["source_identity"] = source
        split = str(row["split"])
        cases.append({"id": f"{sxs_id}/{lev}/N{int(row['extrapolation_order'])}", "path": extracted["npz"],
                      "npz_path": extracted["npz"], "q": fields["q"], "split": split,
                      "sxs_id": sxs_id, "resolution": lev,
                      "extrapolation_order": int(row["extrapolation_order"]),
                      "target_q": row["target_q"], "candidate_rank": row["candidate_rank"],
                      "source_split": split, "npz_sha256": extracted["npz_sha256"],
                      "selection": {"target_q": row.get("target_q"), "candidate_rank": row.get("candidate_rank")},
                      "metadata": {**meta, "peak_time": extracted["peak_time"],
                                   "source_frame_type": extracted["source_frame_type"],
                                   "extraction_frame_type": extracted["extraction_frame_type"],
                                   "frame_transform": extracted["frame_transform"],
                                   "npz_sha256": extracted["npz_sha256"]}})
        identities.append(source)
    # Detect accidental duplicate IDs and ensure the highest selected N/order is
    # the sole primary. All lower-order or lower-resolution additions are
    # explicitly validation-only.
    ids = [(c["metadata"]["sxs_id"], c["metadata"]["resolution"], c["metadata"]["extrapolation_order"]) for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate simulation/resolution/order in selection")
    for sxs_id in sorted({c["metadata"]["sxs_id"] for c in cases}):
        group = [c for c in cases if c["metadata"]["sxs_id"] == sxs_id]
        primaries = [c for c in group if c["split"] in {"training", "holdout"}]
        if len(primaries) != 1:
            raise ValueError(f"{sxs_id} must have exactly one training/holdout primary")
        def rank(c: dict[str, Any]) -> tuple[int, int]:
            match = re.fullmatch(r"Lev(\d+)", str(c["metadata"]["resolution"]))
            return (int(c["metadata"]["extrapolation_order"]), int(match.group(1)) if match else -1)
        n3 = [c for c in group if c["metadata"]["extrapolation_order"] == 3]
        if not n3:
            raise ValueError(f"{sxs_id} requires a selected N3 primary")
        best = max(n3, key=rank)
        if primaries[0] is not best:
            raise ValueError(f"{sxs_id} primary must be highest selected extrapolation order/resolution")
    return {"schema": SCHEMA, "schema_version": 2,
            "selection_manifest": str(selection_path), "selection_manifest_sha256": digest(selection_path),
            "policy_sha256": selection.get("policy_sha256"), "inventory": str(inventory_path),
            "inventory_sha256": digest(inventory_path), "raw_dir": str(raw_dir), "npz_dir": str(npz_dir),
            "policy": {"primary": "one highest selected N3/resolution per simulation; additional rows validation_only",
                       "q": "derived from exact Lev reference_mass1/reference_mass2",
                       "times": "reference/relaxation read from exact Lev metadata and converted code units to waveform M",
                       "metadata": "remnant mass fraction derived from remnant_mass/(reference_mass1+reference_mass2); spin is vector norm",
                       "download": "every selected H5, Strain_N#.json sidecar, and Lev#:metadata.json verified by inventory MD5 and size"},
            "source_identities": identities, "software": {"sxs": expected_sxs_version,
             "decoder": "scripts.prepare_nr33_calibration_data.extract_case",
             "decoder_script": str(Path(__file__).with_name("prepare_nr33_calibration_data.py")),
             "decoder_script_sha256": digest(Path(__file__).with_name("prepare_nr33_calibration_data.py"))}, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--expected-sxs-version", default=DEFAULT_SXS_VERSION)
    ap.add_argument("--expected-policy-sha256")
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    report = prepare(args.selection, args.inventory, args.raw_dir, args.npz_dir,
                     expected_sxs_version=args.expected_sxs_version,
                     expected_policy_sha256=args.expected_policy_sha256)
    report["run"] = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                      "host": platform.node(), "cwd": str(Path.cwd()), "python": sys.executable,
                      "command": sys.argv, "script_sha256": digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
