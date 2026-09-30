"""Qualify the published-calibration HM option under a frozen JSON manifest.

The manifest selects case identity and its two resolutions. Numerical settings
are the reviewed 3144 protocol, reused without changing that historical runner.
Require a separately recorded manifest SHA-256 before retrieval or scoring.
Raw files, derived arrays, and JSON execution receipts stay outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = Path(__file__).with_name("run_imrphenomhm_sxs_3144.py")
OPTIONS = {
    "calibration": "published",
    "hm_21_pn_variant": "mishra_eq12b",
    "hm_32_pn_variant": "mishra_eq12d",
    "hm_43_pn_variant": "mishra_eq12g",
    "hm_mapping_transitions": "london_2018",
}
SETTINGS = {
    "total_mass_msun": 100.0, "distance_mpc": 500.0, "f_ref_hz": 30.0,
    "frequency_hz": [30, 1024, 1], "sample_rate_hz": 8192.0,
    "start_ramp_m": 500.0, "end_ramp_m": 100.0,
    "angles_rad": [[0.7, 0.0], [1.2, 0.4]], "nr_ell_range": [2, 5],
    "nr_phase_grid": 2048, "optimizer_seeds": [17, 23, 41, 67],
    "time_indices": [-819, 819], "fft_length": 8192,
    "resolution_match_floor": 0.995, "resolution_norm_bounds": [0.95, 1.05],
    "model_match_floor": 0.980, "model_norm_bounds": [0.85, 1.15],
    "optimizer_match_spread_max": 1e-5,
}
PREFIX = "hm_published_nr"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    tree = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        tree.update(len(relative).to_bytes(8, "big") + relative)
        tree.update(len(content).to_bytes(8, "big") + content)
    return {"runner": sha256(Path(__file__)), "base_runner": sha256(BASE_RUNNER),
            "src_python_tree": tree.hexdigest()}


def load_manifest(path: Path, expected_sha256: str) -> dict:
    """Fail before data access if the prospectively committed inputs changed."""
    if sha256(path) != expected_sha256:
        raise ValueError("manifest SHA-256 differs from frozen identity")
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != "imrphenomhm-published-nr/v1":
        raise ValueError("unsupported manifest schema")
    if manifest["options"] != OPTIONS or manifest["settings"] != SETTINGS:
        raise ValueError("manifest does not specify the supported published protocol")
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("runner source differs from frozen manifest")
    protocol = manifest["protocol"]
    if sha256(ROOT / protocol["path"]) != protocol["sha256"]:
        raise ValueError("protocol SHA-256 differs from frozen manifest")
    levels = manifest["levels"]
    if (len(levels) != 2 or any(re.fullmatch(r"Lev\d+", item) is None for item in levels)
            or int(levels[0][3:]) <= int(levels[1][3:])):
        raise ValueError("levels must name the higher and lower resolutions in order")
    expected_files = {f"{level}:{name}" for level in levels
                      for name in ("Strain_N2.h5", "Strain_N2.json", "metadata.json")}
    if set(manifest["files"]) != expected_files:
        raise ValueError("manifest must contain exactly the six selected official files")
    if not re.fullmatch(r"[a-z0-9-]+", manifest["record"]):
        raise ValueError("invalid Caltech record ID")
    for entry in manifest["files"].values():
        if entry["size"] <= 0 or not re.fullmatch(r"[a-f0-9]{32}", entry["md5"]):
            raise ValueError("invalid official file identity")
    for name in ("reference_spin1", "reference_spin2"):
        value = np.asarray(manifest[name], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {name}")
        if np.linalg.norm(value[:2]) >= 0.01:
            raise ValueError("aligned protocol requires transverse spin norms below 0.01")
    for number in (1, 2):
        value = np.asarray(manifest[f"reference_spin{number}"], dtype=float)
        model_spin = manifest[f"model_chi{number}"]
        nonspinning = model_spin == 0.0 and np.linalg.norm(value) < 0.001
        if not nonspinning and (not np.isfinite(model_spin)
                                or abs(model_spin) > 1 or model_spin != value[2]):
            raise ValueError("model spin must equal catalog z-spin except for nonspinning residuals")
    if not (np.isfinite(manifest["reference_q"]) and manifest["reference_q"] >= 1
            and np.isfinite(manifest["reference_time_m"])):
        raise ValueError("invalid case reference mass ratio/time")
    return manifest


def configured_base(manifest: dict) -> ModuleType:
    """Isolate inherited module globals; adapt real file labels at the boundary.

    Lev3/Lev2 are internal high/low slot names in the unchanged conditioning
    implementation. They never change the source paths or reported identities.
    """
    spec = importlib.util.spec_from_file_location("_hm_nr_conditioning", BASE_RUNNER)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    slots = dict(zip(("Lev3", "Lev2"), manifest["levels"]))
    base.RECORD = manifest["record"]
    base.BASE = f"https://data.caltech.edu/api/records/{base.RECORD}/files"
    base.FILES = {key: (entry["size"], entry["md5"])
                  for key, entry in manifest["files"].items()}
    base.REFERENCE_Q = manifest["reference_q"]
    base.REFERENCE_TIME_M = manifest["reference_time_m"]

    # Official verification/download uses actual level names. The conditioning
    # calls this adapted function only after verification, for its two slots.
    original_source_path = base.source_path
    verify_sources = base.verify_sources
    download_sources = base.download_sources

    def actual_source_path(data_dir: Path, key: str) -> Path:
        level, name = key.split(":", 1)
        return data_dir / slots[level] / name

    def with_actual_labels(function, data_dir):
        base.source_path = original_source_path
        try:
            return function(data_dir)
        finally:
            base.source_path = actual_source_path

    base.verify_sources = lambda directory: with_actual_labels(verify_sources, directory)
    base.download_sources = lambda directory: with_actual_labels(download_sources, directory)
    base.source_path = actual_source_path

    def check_sidecars(data_dir: Path) -> dict:
        modifications = {}
        nonspinning = all(manifest[f"model_chi{number}"] == 0.0
                          and np.linalg.norm(manifest[f"reference_spin{number}"]) < 0.001
                          for number in (1, 2))
        for slot, level in slots.items():
            sidecar = json.loads(actual_source_path(data_dir, f"{slot}:Strain_N2.json").read_text())
            for key, value in dict(data_type="h", spin_weight=-2, m_is_scaled_out=True,
                                   r_is_scaled_out=True, ell_min=2, ell_max=8).items():
                if sidecar["data_info"].get(key) != value:
                    raise RuntimeError(f"{level} sidecar {key} disagrees with protocol")
            metadata = json.loads(actual_source_path(data_dir, f"{slot}:metadata.json").read_text())
            q = metadata["reference_mass1"] / metadata["reference_mass2"]
            if abs(q - manifest["reference_q"]) > 0.01:
                raise RuntimeError(f"{level} mass ratio disagrees with case")
            for number in (1, 2):
                spin = np.asarray(metadata[f"reference_dimensionless_spin{number}"])
                if (spin.shape != (3,) or not np.all(np.isfinite(spin))
                        or np.linalg.norm(spin[:2]) >= 0.01
                        or abs(spin[2] - manifest[f"reference_spin{number}"][2]) > 5e-4):
                    raise RuntimeError(f"{level} spin disagrees with aligned protocol")
                if nonspinning and np.linalg.norm(spin) >= 0.001:
                    raise RuntimeError(f"{level} spin disagrees with nonspinning protocol")
                if slot == "Lev3" and not np.allclose(
                        spin, manifest[f"reference_spin{number}"], rtol=0, atol=1e-10):
                    raise RuntimeError(f"{level} spin differs from frozen catalog metadata")
            if slot == "Lev3" and not np.isclose(
                    metadata["reference_time"], manifest["reference_time_m"], rtol=0, atol=1e-8):
                raise RuntimeError(f"{level} reference time differs from frozen catalog")
            modifications[level] = sidecar.get("modifications")
        return modifications

    base._check_sidecars = check_sidecars
    return base


def assert_local_frontend(backend: str) -> None:
    module = importlib.import_module(f"{backend}wave.approximants.imrphenom_hm")
    expected = ROOT / "src" / f"{backend}wave/approximants/imrphenom_hm.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError("native frontend did not import from the frozen source tree")


def native_modes(base: ModuleType, manifest: dict, backend: str) -> dict:
    """Pass every named physics option explicitly to both native implementations."""
    q = manifest["reference_q"]
    kwargs = dict(mass1=base.TOTAL_MASS * q / (1 + q), mass2=base.TOTAL_MASS / (1 + q),
                  chi1=manifest["model_chi1"], chi2=manifest["model_chi2"],
                  distance=base.DISTANCE_MPC, inclination=0.0, phi0=0.0,
                  phic=0.0, tc=0.0, f_ref=30.0, return_unprojected_modes=True,
                  modes=base.TEMPLATE_MODES, **manifest["options"])
    if {key: kwargs[key] for key in OPTIONS} != OPTIONS:
        raise ValueError("native options differ from frozen published variant")
    if backend == "torch":
        import torch
        from torchwave.approximants.imrphenom_hm import IMRPhenomHM

        assert_local_frontend(backend)
        values = IMRPhenomHM()(f=torch.as_tensor(base.FREQUENCIES, dtype=torch.float64,
                                               device="cpu"), **kwargs)[2]
        modes = {mode: np.asarray(value.detach().cpu().numpy()) for mode, value in values.items()}
    elif backend == "jax":
        import jax
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenom_hm import IMRPhenomHM

        assert_local_frontend(backend)
        jax.config.update("jax_enable_x64", True)
        with jax.default_device(jax.devices("cpu")[0]):
            values = IMRPhenomHM.evaluate(
                f=jnp.asarray(base.FREQUENCIES, dtype=jnp.float64), **kwargs)[2]
        modes = {mode: np.asarray(value) for mode, value in values.items()}
    else:
        raise ValueError(f"unsupported backend: {backend}")
    if tuple(modes) != base.TEMPLATE_MODES:
        raise RuntimeError("native mode set differs from frozen six-mode protocol")
    for mode, value in modes.items():
        if value.shape != base.FREQUENCIES.shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite or wrong-shaped mode: {mode}")
    return modes


def prepare(base, manifest, manifest_hash, data_dir):
    source_identities = base.verify_sources(data_dir)
    reference = base.build_reference(data_dir)
    # Persist actual levels, not inherited implementation slot labels.
    renamed = {}
    for key, value in reference.items():
        for slot, level in zip(("Lev3", "Lev2"), manifest["levels"]):
            if key.startswith(slot + "_"):
                key = f"{level}_{key[len(slot) + 1:]}"
                break
        renamed[key] = value
    reference = renamed
    fixture_path = data_dir / f"{PREFIX}_reference.npz"
    np.savez_compressed(fixture_path, **reference)
    preparation = {
        "manifest_sha256": manifest_hash, "source_hashes": source_hashes(),
        "raw_file_hashes": source_identities, "fixture_sha256": sha256(fixture_path),
        "sidecar_modifications": base._check_sidecars(data_dir),
        "versions": {"python": sys.version.split()[0],
                     **{name: importlib.metadata.version(name)
                        for name in ("numpy", "scipy", "sxs", "sympy")}},
        "levels": manifest["levels"],
    }
    (data_dir / f"{PREFIX}_preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")


def evaluate(base, manifest, manifest_hash, data_dir, source_revision):
    preparation = json.loads((data_dir / f"{PREFIX}_preparation.json").read_text())
    fixture_path = data_dir / f"{PREFIX}_reference.npz"
    if (preparation["manifest_sha256"] != manifest_hash
            or preparation["source_hashes"] != source_hashes()
            or preparation["raw_file_hashes"] != base.verify_sources(data_dir)
            or preparation["fixture_sha256"] != sha256(fixture_path)):
        raise RuntimeError("manifest, raw files, fixture, or preparation source changed")
    with np.load(fixture_path) as fixture:
        if (not np.array_equal(fixture["frequencies"], base.FREQUENCIES)
                or not np.array_equal(fixture["resolution_basis_modes"], base.NR_MODES)
                or float(np.max(fixture["basis_relative_discrepancies"])) > 1e-10):
            raise RuntimeError("fixture violates frozen frequency/mode/reconstruction contract")
        nr = np.stack([np.stack([fixture[f"{level}_{name}_{i}"] for name in ("plus", "cross")])
                       for level in manifest["levels"] for i in range(len(base.ANGLES))])
        nr = nr.reshape(2, len(base.ANGLES), 2, base.FREQUENCIES.size)
        resolution_basis = fixture["resolution_basis"].copy()
        high_basis = fixture["lev3_mode_basis"].copy()
        conditioning = {name: float(fixture[name]) for name in (
            "time_shift_origin_m", "common_start_m", "common_end_m",
            "uniform_samples", "m_seconds", "amplitude_scale")}
        reconstruction_errors = fixture["basis_relative_discrepancies"].tolist()
    psd = base.psd_comparison_bins()
    synthetic = base.synthetic_fft_check()
    resolution = base.align_resolutions(nr[0], resolution_basis, psd)
    results, backend_modes = {}, {}
    for backend in ("torch", "jax"):
        modes = native_modes(base, manifest, backend)
        backend_modes[backend] = modes
        rows = []
        for i, (inclination, psi) in enumerate(base.ANGLES):
            reference = np.cos(2 * psi) * nr[0, i, 0] + np.sin(2 * psi) * nr[0, i, 1]
            optimization = base.optimize_model(reference, base.template_basis(modes, inclination), psd)
            best = optimization["best"]
            rows.append({"inclination_rad": inclination, "source_psi_rad": psi,
                         "optimization": optimization,
                         "passed": bool(optimization["feasible"]
                                        and best["match"] >= base.MODEL_MATCH_FLOOR
                                        and base.MODEL_NORM_BOUNDS[0] <= best["norm_ratio"]
                                        <= base.MODEL_NORM_BOUNDS[1])})
        results[backend] = rows
    parity = {f"{ell},{m}": float(np.max(np.abs(backend_modes["torch"][ell, m]
                                               - backend_modes["jax"][ell, m]))
                                / max(np.max(np.abs(backend_modes["torch"][ell, m])), 1e-300))
              for ell, m in base.TEMPLATE_MODES}
    resolution_pass = all(row["passed"] for row in resolution["views"])
    model_pass = all(row["passed"] for rows in results.values() for row in rows)
    result = {
        "case": manifest["case"], "levels": manifest["levels"],
        "protocol": manifest["protocol"], "manifest_sha256": manifest_hash,
        "manifest": manifest, "source_revision": source_revision,
        "preparation": preparation, "evaluation_versions": base.version_record(),
        "conditioning": conditioning, "basis_relative_discrepancies": reconstruction_errors,
        "psd_sha256": hashlib.sha256(np.ascontiguousarray(psd, dtype="<f8").tobytes()).hexdigest(),
        "synthetic_fft_check": synthetic, "resolution": resolution,
        "omitted_mode_diagnostics": base.omitted_mode_diagnostics(nr[0], high_basis, psd),
        "model": results, "frontend_mode_relative_errors": parity,
        "resolution_pass": resolution_pass, "model_pass": model_pass,
        "qualified": bool(resolution_pass and model_pass),
    }
    (data_dir / f"{PREFIX}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    base = configured_base(manifest)
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only or args.verify_only:
        base.verify_sources(data_dir)
    else:
        base.download_sources(data_dir)
    if args.verify_only:
        print("Frozen manifest and official files verified.", flush=True)
        return
    if not args.evaluate_only:
        prepare(base, manifest, args.manifest_sha256, data_dir)
        print("Prepared independently conditioned NR fixture.", flush=True)
    if not args.prepare_only:
        result = evaluate(base, manifest, args.manifest_sha256, data_dir, args.source_revision)
        print(json.dumps({key: result[key] for key in
                          ("case", "qualified", "resolution_pass", "model_pass")}), flush=True)


if __name__ == "__main__":
    main()
