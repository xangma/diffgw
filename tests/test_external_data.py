import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import torchwave.external_data as external_data


def _spec(data=b"numerical data"):
    return external_data.DatasetSpec(
        "tiny", "tiny.bin", hashlib.sha256(data).hexdigest(), "https://example.invalid/tiny.bin", "test", "https://example.invalid", "upstream terms"
    )


def test_resolver_precedence_and_verification(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setitem(external_data.DATASETS, "tiny", spec)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    (second / spec.filename).write_bytes(b"numerical data")
    monkeypatch.setenv("TORCHWAVE_DATA_PATH", str(first) + os.pathsep + str(second))
    assert external_data.resolve_dataset_path("tiny") == second / spec.filename
    (first / spec.filename).write_bytes(b"bad")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        external_data.resolve_dataset_path("tiny")


def test_explicit_missing_does_not_fallback(tmp_path, monkeypatch):
    data = b"numerical data"
    monkeypatch.setitem(external_data.DATASETS, "tiny", _spec(data))
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "tiny.bin").write_bytes(data)
    monkeypatch.setenv("TORCHWAVE_DATA_PATH", str(fallback))
    with pytest.raises(FileNotFoundError):
        external_data.resolve_dataset_path("tiny", tmp_path / "missing.bin")


def test_default_cache_is_dynamic_and_empty_xdg_uses_home(tmp_path, monkeypatch):
    data = b"numerical data"
    monkeypatch.setitem(external_data.DATASETS, "tiny", _spec(data))
    monkeypatch.delenv("TORCHWAVE_DATA_PATH", raising=False)
    monkeypatch.delenv("LAL_DATA_PATH", raising=False)
    first = tmp_path / "first-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(first))
    (first / "torchwave" / "waveform-data").mkdir(parents=True)
    target = first / "torchwave" / "waveform-data" / "tiny.bin"
    target.write_bytes(data)
    assert external_data.resolve_dataset_path("tiny") == target.resolve()
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    target = home / ".cache" / "torchwave" / "waveform-data" / "tiny.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    assert external_data.resolve_dataset_path("tiny") == target.resolve()


def test_download_checksum_and_partial_cleanup(tmp_path, monkeypatch):
    data = b"numerical data"
    spec = _spec(data)
    monkeypatch.setitem(external_data.DATASETS, "tiny", spec)

    class Response:
        def __init__(self): self.done = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n=-1):
            if self.done: return b""
            self.done = True
            return data

    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: Response())
    result = external_data.download_dataset("tiny", tmp_path)
    assert result.read_bytes() == data
    assert not list(tmp_path.glob("*.part"))

    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: Response())
    monkeypatch.setitem(external_data.DATASETS, "tiny", _spec(b"different"))
    with pytest.raises(ValueError, match="wrong checksum"):
        external_data.download_dataset("tiny", tmp_path)


@pytest.mark.parametrize("failure", ["checksum", "size", "transfer"])
def test_failed_new_download_cleans_partial(tmp_path, monkeypatch, failure):
    expected = b"good"
    spec = _spec(expected)
    if failure == "size":
        spec = external_data.DatasetSpec(spec.dataset_id, spec.filename, spec.sha256, spec.url, spec.release, spec.upstream_url, spec.terms, size_bytes=3)
    monkeypatch.setitem(external_data.DATASETS, "tiny", spec)

    class Response:
        def __init__(self): self.done = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n=-1):
            if self.done: return b""
            self.done = True
            if failure == "transfer": raise OSError("connection lost")
            return expected if failure != "checksum" else b"bad"

    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: Response())
    with pytest.raises((ValueError, OSError)):
        external_data.download_dataset("tiny", tmp_path)
    assert not list(tmp_path.glob("*.part"))
    assert not (tmp_path / spec.filename).exists()


def test_download_does_not_overwrite_concurrent_destination(tmp_path, monkeypatch):
    data = b"numerical data"
    spec = _spec(data)
    monkeypatch.setitem(external_data.DATASETS, "tiny", spec)

    class Response:
        def __init__(self): self.done = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n=-1):
            if self.done: return b""
            self.done = True
            return data

    def link(source, destination):
        destination.write_bytes(b"someone else's file")
        raise FileExistsError(destination)

    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: Response())
    monkeypatch.setattr(external_data.os, "link", link)
    with pytest.raises(ValueError, match="concurrent"):
        external_data.download_dataset("tiny", tmp_path)
    assert (tmp_path / spec.filename).read_bytes() == b"someone else's file"
    assert not list(tmp_path.glob("*.part"))


def test_download_reuses_valid_existing_without_network(tmp_path, monkeypatch):
    data = b"numerical data"
    spec = _spec(data)
    monkeypatch.setitem(external_data.DATASETS, "tiny", spec)
    destination = tmp_path / spec.filename
    destination.write_bytes(data)
    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: pytest.fail("network should not be used"))
    assert external_data.download_dataset("tiny", tmp_path) == destination.resolve()


def test_download_rejects_package_destination_before_mkdir_or_network(monkeypatch):
    package_dir = Path(external_data.__file__).resolve().parent
    destination_dir = package_dir / "should-not-be-created"
    monkeypatch.setattr(external_data, "urlopen", lambda *a, **k: pytest.fail("network should not be used"))
    with pytest.raises(ValueError, match="package or source checkout"):
        external_data.download_dataset("seobnrv4-rom-v3", destination_dir)
    assert not destination_dir.exists()


def test_cli_list_and_info(capsys):
    assert external_data.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "seobnrv4-rom-v3" in output and "xhm122022" in output
    assert external_data.main(["info", "seobnrv4-rom-v3"]) == 0
    output = capsys.readouterr().out
    assert "Zenodo" in output or "zenodo" in output


def test_cli_verify_path_and_unavailable_error(tmp_path, capsys):
    data = b"numerical data"
    path = tmp_path / "tiny.bin"
    path.write_bytes(data)
    monkey = pytest.MonkeyPatch()
    monkey.setitem(external_data.DATASETS, "tiny", _spec(data))
    try:
        assert external_data.main(["verify", "tiny", "--path", str(path)]) == 0
        assert str(path) in capsys.readouterr().out
        assert external_data.main(["info", "xhm122022"]) == 0
        assert "unavailable" in capsys.readouterr().out
        assert external_data.main(["verify", "missing"]) == 2
        assert "error:" in capsys.readouterr().err
        assert external_data.main(["fetch", "tiny", "--path", str(path)]) == 2
        assert "only valid with verify" in capsys.readouterr().err
    finally:
        monkey.undo()


def test_import_external_data():
    from torchwave import external_data as ed
    assert ed is external_data
    assert hasattr(ed, "resolve_dataset_path")
    assert hasattr(ed, "main")


def test_module_cli_list():
    res = subprocess.run(
        [sys.executable, "-m", "torchwave.external_data", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "seobnrv4-rom-v3" in res.stdout
    assert "xhm122022" in res.stdout


def test_module_cli_info():
    res = subprocess.run(
        [sys.executable, "-m", "torchwave.external_data", "info", "seobnrv4-rom-v3"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "seobnrv4-rom-v3" in res.stdout
    assert "Zenodo" in res.stdout or "zenodo" in res.stdout


def test_console_script_cli():
    exe = shutil.which("torchwave-data")
    if exe is None:
        pytest.skip("torchwave-data is not installed in PATH")
    res = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "torchwave-data" in res.stdout
    assert "Locate and fetch optional torchwave numerical data" in res.stdout

