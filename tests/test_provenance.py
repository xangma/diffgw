"""Compatibility and invalidation checks for the relocated source identity."""

import hashlib

import pytest

from torchwave import provenance


FILES = ('__init__.py', 'waveforms.py', 'constants.py',
         'conversions.py', 'approximants/taylorf2.py')


@pytest.fixture
def sources(tmp_path, monkeypatch):
    for relative in FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode() + b'\x00\xffsource')
    monkeypatch.setattr(provenance, '__file__',
                        str(tmp_path / 'provenance.py'))
    yield tmp_path


def test_legacy_digest_compatibility(sources):
    expected = hashlib.sha256(
        b''.join(name.encode() + name.encode() + b'\x00\xffsource'
                 for name in FILES))
    assert provenance.taylorf2_source_identity() == expected.hexdigest()


@pytest.mark.parametrize('relative', FILES)
def test_each_source_changes_identity(sources, relative):
    before = provenance.taylorf2_source_identity()
    (sources / relative).write_bytes(b'changed source')
    assert provenance.taylorf2_source_identity() != before


def test_identity_does_not_depend_on_installation_path(sources, monkeypatch,
                                                     tmp_path_factory):
    before = provenance.taylorf2_source_identity()
    other = tmp_path_factory.mktemp('other-install')
    for relative in FILES:
        target = other / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((sources / relative).read_bytes())
    monkeypatch.setattr(provenance, '__file__', str(other / 'provenance.py'))
    assert provenance.taylorf2_source_identity() == before


def test_unrelated_sources_do_not_change_legacy_identity(sources):
    before = provenance.taylorf2_source_identity()
    (sources / 'provenance.py').write_bytes(b'helper implementation')
    (sources / 'unrelated.py').write_bytes(b'other approximant')
    assert provenance.taylorf2_source_identity() == before


def test_missing_source_is_an_error(sources):
    (sources / 'approximants/taylorf2.py').unlink()
    with pytest.raises(FileNotFoundError):
        provenance.taylorf2_source_identity()
