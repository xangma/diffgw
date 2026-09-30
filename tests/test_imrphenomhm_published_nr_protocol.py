"""Data-free integrity and native-option contracts for prospective NR runs."""
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_imrphenomhm_published_nr.py"
SPEC = importlib.util.spec_from_file_location("hm_published_nr", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "protocol.md").write_text("Prospectively frozen protocol.\n")
    return dict(
        schema_version="imrphenomhm-published-nr/v1", case="SXS:BBH:0000v3.0",
        record="example-record", catalog_sha256="0" * 64,
        reference_q=3.0, reference_time_m=1000.0,
        reference_spin1=[0., 0., 0.], reference_spin2=[0., 0., 0.],
        model_chi1=0., model_chi2=0., levels=["Lev4", "Lev3"],
        files={f"{level}:{name}": {"size": 1, "md5": "0" * 32}
               for level in ("Lev4", "Lev3")
               for name in ("Strain_N2.h5", "Strain_N2.json", "metadata.json")},
        protocol={"path": "protocol.md", "sha256": runner.sha256(tmp_path / "protocol.md")},
        source_hashes=runner.source_hashes(), options=copy.deepcopy(runner.OPTIONS),
        settings=copy.deepcopy(runner.SETTINGS),
    )


def test_accepts_frozen_manifest_and_rejects_changed_bytes(manifest, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    frozen = runner.sha256(path)
    assert runner.load_manifest(path, frozen) == manifest
    path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        runner.load_manifest(path, frozen)


@pytest.mark.parametrize("change,message", [
    ("source", "runner source"), ("protocol", "protocol SHA-256"),
    ("options", "supported published protocol"), ("levels", "higher and lower"),
])
def test_rejects_rebound_source_protocol_or_options(manifest, tmp_path, change, message):
    if change == "source":
        manifest["source_hashes"]["base_runner"] = "0" * 64
    elif change == "protocol":
        (tmp_path / "protocol.md").write_text("Modified protocol")
    elif change == "options":
        manifest["options"]["calibration"] = "inherited"
    else:
        manifest["levels"].reverse()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        runner.load_manifest(path, runner.sha256(path))


def test_level_adaptation_keeps_actual_verification_paths(manifest, tmp_path):
    for key, identity in manifest["files"].items():
        level, name = key.split(":")
        path = tmp_path / level / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(key.encode())
        identity.update(size=path.stat().st_size, md5=hashlib.md5(path.read_bytes()).hexdigest())
    base = runner.configured_base(manifest)
    assert base.source_path(tmp_path, "Lev3:metadata.json") == tmp_path / "Lev4/metadata.json"
    assert base.source_path(tmp_path, "Lev2:metadata.json") == tmp_path / "Lev3/metadata.json"
    assert set(base.verify_sources(tmp_path)) == set(manifest["files"])
    # Verification temporarily accesses actual labels and then restores slot mapping.
    assert base.source_path(tmp_path, "Lev3:metadata.json") == tmp_path / "Lev4/metadata.json"
    (tmp_path / "Lev4/Strain_N2.h5").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="official SXS file"):
        base.verify_sources(tmp_path)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_forwards_exact_published_options_to_each_backend(manifest, monkeypatch, backend):
    base = runner.configured_base(manifest)
    captured = {}

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones_like(base.FREQUENCIES, dtype=complex)

    def evaluate(**kwargs):
        captured.update(kwargs)
        values = {mode: FakeTensor() if backend == "torch" else np.ones_like(base.FREQUENCIES)
                  for mode in base.TEMPLATE_MODES}
        return None, None, values

    class FakeHM:
        def __call__(self, **kwargs):
            return evaluate(**kwargs)

    FakeHM.evaluate = staticmethod(evaluate)
    module_name = f"{backend}wave.approximants.imrphenom_hm"
    module = ModuleType(module_name)
    module.IMRPhenomHM = FakeHM
    module.__file__ = str(runner.ROOT / "src" / f"{backend}wave/approximants/imrphenom_hm.py")
    monkeypatch.setitem(sys.modules, module_name, module)
    if backend == "torch":
        module = ModuleType("torch")
        module.as_tensor = lambda value, **kwargs: value
        module.float64 = np.float64
        monkeypatch.setitem(sys.modules, "torch", module)
    else:
        module = ModuleType("jax")
        module.config = SimpleNamespace(update=lambda *args: None)
        module.devices = lambda name: ["cpu"]
        module.default_device = lambda device: nullcontext()
        numpy_module = ModuleType("jax.numpy")
        numpy_module.asarray, numpy_module.float64 = np.asarray, np.float64
        module.numpy = numpy_module
        monkeypatch.setitem(sys.modules, "jax", module)
        monkeypatch.setitem(sys.modules, "jax.numpy", numpy_module)
    assert tuple(runner.native_modes(base, manifest, backend)) == base.TEMPLATE_MODES
    assert {key: captured[key] for key in runner.OPTIONS} == runner.OPTIONS
    assert captured["modes"] == base.TEMPLATE_MODES
    assert captured["return_unprojected_modes"] is True
    assert captured["f_ref"] == 30.0
    assert captured["mass1"] / captured["mass2"] == manifest["reference_q"]


@pytest.mark.parametrize("levels", [("Lev3", "Lev2"), ("Lev4", "Lev3")])
def test_preparation_persists_actual_high_low_labels(manifest, tmp_path, monkeypatch, levels):
    manifest["levels"] = list(levels)
    base = SimpleNamespace(
        verify_sources=lambda _: {}, _check_sidecars=lambda _: {},
        build_reference=lambda _: {"Lev3_plus_0": np.array([3]), "Lev2_plus_0": np.array([2])},
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test")
    runner.prepare(base, manifest, "test-manifest-hash", tmp_path)
    with np.load(tmp_path / f"{runner.PREFIX}_reference.npz") as fixture:
        assert fixture[f"{levels[0]}_plus_0"].item() == 3
        assert fixture[f"{levels[1]}_plus_0"].item() == 2


def test_accepts_catalog_aligned_spins_and_rejects_changed_model_spin(manifest, tmp_path):
    manifest["reference_spin1"] = [1e-5, 2e-5, 0.6]
    manifest["model_chi1"] = 0.6
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    runner.load_manifest(path, runner.sha256(path))
    manifest["model_chi1"] = 0.5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model spin must equal"):
        runner.load_manifest(path, runner.sha256(path))


def test_nonspinning_lower_resolution_retains_stricter_spin_screen(manifest, tmp_path):
    sidecar = {"data_info": dict(data_type="h", spin_weight=-2, m_is_scaled_out=True,
                                r_is_scaled_out=True, ell_min=2, ell_max=8)}
    metadata = dict(reference_mass1=3., reference_mass2=1., reference_time=1000.,
                    reference_dimensionless_spin1=[0., 0., 0.],
                    reference_dimensionless_spin2=[0., 0., 0.])
    for level in manifest["levels"]:
        directory = tmp_path / level
        directory.mkdir()
        (directory / "Strain_N2.json").write_text(json.dumps(sidecar))
        (directory / "metadata.json").write_text(json.dumps(metadata))
    base = runner.configured_base(manifest)
    assert set(base._check_sidecars(tmp_path)) == set(manifest["levels"])
    # Passes the aligned-spin transverse/z checks but violates nonspinning scope.
    metadata["reference_dimensionless_spin1"] = [0.002, 0., 0.]
    (tmp_path / manifest["levels"][1] / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="nonspinning protocol"):
        base._check_sidecars(tmp_path)
