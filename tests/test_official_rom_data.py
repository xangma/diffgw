"""Clean-room checks for the published SEOBNRv4 ROM data contract."""

from torchwave.approximants.rom.official import (
    OFFICIAL_ROM_SHA256,
    inspect_official_rom,
    sha256_dataset,
)


def test_official_rom_schema_and_file_hash(official_rom_path):
    path = official_rom_path
    schema = inspect_official_rom(path, verify_hash=True)

    assert schema.sha256 == OFFICIAL_ROM_SHA256
    assert [group.name for group in schema.groups] == ["sub1", "sub2", "sub3"]
    assert [(group.n_frequency, group.n_eta, group.n_chi1, group.n_chi2) for group in schema.groups] == [
        (259, 70, 12, 12),
        (325, 57, 33, 21),
        (325, 13, 11, 21),
    ]
    for group in schema.groups:
        assert group.amplitude_coefficients == group.phase_coefficients
        assert group.amplitude_coefficients[1:] == group.padded_parameter_shape


def test_official_rom_representative_table_hash(official_rom_path):
    path = official_rom_path
    assert sha256_dataset(path, "sub1", "Bamp") == (
        "3e5c9d3252ca901c8515be22d18e6f8bfa4933d612cd2f17d0a28fcf4b1fa0b1"
    )
