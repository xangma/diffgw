"""Regression checks for the MIT-only shipped source tree."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_external_cook_fixtures_are_not_in_repository():
    names = {
        "kerr_qnm_author_holdout.json",
        "kerr_qnm_author_validation.json",
        "kerr_angular_author_samples.json",
    }
    assert not any(path.name in names for path in (ROOT / "tests" / "data").glob("*.json"))


def test_provenance_helper_is_functional_and_source_assets_are_external():
    from torchwave.provenance import taylorf2_source_identity

    identity = taylorf2_source_identity()
    assert len(identity) == 64
    assert all(character in "0123456789abcdef" for character in identity)
    assert not list((ROOT / "src").rglob("*.m"))
    assert not list((ROOT / "docs" / "reviews" / "2026-09-12-clean-room" / "source-assets").rglob("*.m"))


def test_external_waveform_data_is_not_shipped():
    forbidden = {".hdf5", ".h5", ".m"}
    shipped_roots = (ROOT / "src",)
    assert not [p for root in shipped_roots for p in root.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]

