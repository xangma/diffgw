"""Stable identity for the original TorchWave TaylorF2 source set.

The identity is defined over package-relative names and bytes so installing
the package at a different filesystem location does not change the result.
"""

from hashlib import sha256
from pathlib import Path


_SOURCE_NAMES = (
    "__init__.py",
    "waveforms.py",
    "constants.py",
    "conversions.py",
    "approximants/taylorf2.py",
)


def taylorf2_source_identity() -> str:
    """Return the SHA-256 identity of the required original source files."""
    package_dir = Path(__file__).resolve().parent
    digest = sha256()
    for relative_name in _SOURCE_NAMES:
        digest.update(relative_name.encode("utf-8"))
        digest.update((package_dir / relative_name).read_bytes())
    return digest.hexdigest()
