import os

import pytest

# Prevent macOS crashes when conda-forge OpenBLAS and PyTorch bundle separate libomp runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@pytest.fixture(scope="module")
def official_rom_path():
    """Use the supported explicit/environment/cache lookup for external tests."""
    pytest.importorskip("h5py")
    from torchwave.approximants.rom import resolve_official_rom_path

    try:
        return resolve_official_rom_path(os.environ.get("TORCHWAVE_EXTERNAL_ROM"))
    except FileNotFoundError:
        pytest.skip("official SEOBNRv4 ROM HDF5 is unavailable")
