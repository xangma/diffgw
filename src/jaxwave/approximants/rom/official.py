"""Schema and provenance checks for the published SEOBNRv4 ROM tables in JAX.

Port of torchwave.approximants.rom.official for jaxwave.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

OFFICIAL_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
OFFICIAL_ROM_SHA256 = "4ff5317967d757fcad9323a74b75f4f63894ed87f4b1852d46cbe47cde488b9c"

_DATASETS = (
    "Amp_ciall",
    "Bamp",
    "Bphase",
    "Mf_grid_Amp",
    "Mf_grid_Phi",
    "Phase_ciall",
    "chi1vec",
    "chi2vec",
    "etavec",
)
_GROUPS = ("sub1", "sub2", "sub3")


@dataclass(frozen=True)
class ROMSubmodelSchema:
    """Shape and grid metadata for one published ROM submodel."""

    name: str
    description: str
    n_frequency: int
    n_eta: int
    n_chi1: int
    n_chi2: int
    amplitude_coefficients: Tuple[int, ...]
    phase_coefficients: Tuple[int, ...]

    @property
    def padded_parameter_shape(self) -> Tuple[int, int, int]:
        """Coefficient-table parameter shape, including the two extra planes."""
        return (self.n_eta + 2, self.n_chi1 + 2, self.n_chi2 + 2)


@dataclass(frozen=True)
class ROMFileSchema:
    """Validated metadata for the published SEOBNRv4 ROM HDF5 file."""

    path: Path
    sha256: Optional[str]
    canonical_basename: str
    groups: Tuple[ROMSubmodelSchema, ...]


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading the published SEOBNRv4 ROM requires the optional h5py package"
        ) from exc
    return h5py


def resolve_official_rom_path(
    path: Optional[Union[str, os.PathLike]] = None,
    *,
    verify: bool = False,
) -> Path:
    """Resolve the externally owned official ROM asset."""
    from torchwave.external_data import resolve_dataset_path

    return resolve_dataset_path("seobnrv4-rom-v3", path=path, verify=verify)


def sha256_file(path: Union[str, os.PathLike], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a ROM data file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataset(path: Union[str, os.PathLike], group: str, dataset: str) -> str:
    """Hash one HDF5 dataset in canonical C-order for provenance tests."""
    import numpy as np

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        if group not in handle or dataset not in handle[group]:
            raise KeyError(f"Missing ROM dataset {group}/{dataset}")
        values = np.asarray(handle[group][dataset])
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def inspect_official_rom(
    path: Optional[Union[str, os.PathLike]] = None,
    *,
    verify_hash: bool = False,
) -> ROMFileSchema:
    """Validate the published HDF5 layout and return compact metadata."""
    resolved = resolve_official_rom_path(path)
    digest = sha256_file(resolved) if verify_hash else None
    if verify_hash and digest != OFFICIAL_ROM_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {resolved.name}: {digest}; "
            f"expected {OFFICIAL_ROM_SHA256}"
        )

    import numpy as np

    h5py = _h5py()
    schemas = []
    with h5py.File(resolved, "r") as handle:
        missing_groups = [name for name in _GROUPS if name not in handle]
        if missing_groups:
            raise ValueError(f"Missing ROM groups: {missing_groups}")
        canonical = handle.attrs.get("CANONICAL_FILE_BASENAME", b"")
        if isinstance(canonical, bytes):
            canonical = canonical.decode("utf-8")
        canonical = str(canonical)
        if canonical != OFFICIAL_ROM_FILENAME:
            raise ValueError(
                f"Unexpected CANONICAL_FILE_BASENAME attribute: {canonical!r}; "
                f"expected {OFFICIAL_ROM_FILENAME!r}"
            )
        for name in _GROUPS:
            group = handle[name]
            missing_datasets = [ds for ds in _DATASETS if ds not in group]
            if missing_datasets:
                raise ValueError(f"Group {name} is missing datasets: {missing_datasets}")

            description = group.attrs.get("Submodel_description", b"")
            if isinstance(description, bytes):
                description = description.decode("utf-8")
            description = str(description)

            schemas.append(
                ROMSubmodelSchema(
                    name=name,
                    description=description,
                    n_frequency=int(group["Bamp"].shape[1]),
                    n_eta=int(group["etavec"].shape[0]),
                    n_chi1=int(group["chi1vec"].shape[0]),
                    n_chi2=int(group["chi2vec"].shape[0]),
                    amplitude_coefficients=tuple(int(d) for d in group["Amp_ciall"].shape),
                    phase_coefficients=tuple(int(d) for d in group["Phase_ciall"].shape),
                )
            )

    return ROMFileSchema(
        path=resolved,
        sha256=digest,
        canonical_basename=OFFICIAL_ROM_FILENAME,
        groups=tuple(schemas),
    )


__all__ = [
    "OFFICIAL_ROM_FILENAME",
    "OFFICIAL_ROM_SHA256",
    "ROMFileSchema",
    "ROMSubmodelSchema",
    "inspect_official_rom",
    "resolve_official_rom_path",
    "sha256_dataset",
    "sha256_file",
]
