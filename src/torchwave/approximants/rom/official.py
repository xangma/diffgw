"""Schema and provenance checks for the published SEOBNRv4 ROM tables.

The public SEOBNRv4 ROM release stores the reduced bases and parameter
interpolation tables in an HDF5 file.  Pürrer (2014, 2016) specifies the
tensor-product cubic-spline construction, but the papers do not specify the
packing of the two extra coefficient planes in ``*_ciall``.  This module
therefore validates the public data contract without silently inventing a
boundary convention.

References:
    Pürrer, CQG 31, 195010 (2014), arXiv:1402.4146.
    Pürrer, PRD 93, 064041 (2016), arXiv:1512.02248.
    The official data release is distributed as
    ``SEOBNRv4ROM_v3.0.hdf5`` (Zenodo record 14999310).
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
    except ImportError as exc:  # pragma: no cover - exercised only without optional IO
        raise ImportError(
            "Reading the published SEOBNRv4 ROM requires the optional h5py package"
        ) from exc
    return h5py


def resolve_official_rom_path(
    path: Optional[Union[str, os.PathLike]] = None,
    *,
    verify: bool = False,
) -> Path:
    """Resolve the externally owned official ROM asset.

    Resolution is centralized in :mod:`torchwave.external_data`, which keeps
    explicit paths, ``TORCHWAVE_DATA_PATH``, ``LAL_DATA_PATH``, and the shared
    external cache in a single policy.  ``verify`` is intentionally false by
    default: schema inspection and other metadata-only callers must continue
    to work with a local fixture.  Production loading performs the canonical
    checksum check immediately before opening any tables.
    """

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
    """Validate the published HDF5 layout and return compact metadata.

    ``verify_hash=True`` is intended for the canonical v3.0 release and
    rejects a modified file before any table is used.  The checks are limited
    to facts exposed by the public HDF5 asset and the published ROM papers:
    required groups/datasets, monotone physical grids, shifted spline-grid
    metadata, orthonormal basis dimensions, and coefficient-table sizes.
    """

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
            raise ValueError(f"Unexpected ROM canonical basename: {canonical!r}")

        for name in _GROUPS:
            group = handle[name]
            missing = [dataset for dataset in _DATASETS if dataset not in group]
            if missing:
                raise ValueError(f"Missing datasets in {name}: {missing}")
            grids = {key: np.asarray(group[key], dtype=np.float64) for key in ("etavec", "chi1vec", "chi2vec")}
            if any(values.ndim != 1 or len(values) < 2 for values in grids.values()):
                raise ValueError(f"ROM parameter grids must be one-dimensional in {name}")
            if any(np.any(np.diff(values) <= 0.0) for values in grids.values()):
                raise ValueError(f"ROM parameter grids must be strictly increasing in {name}")

            # The public file exposes shifted spline coordinates as attributes.
            for attr, key in (("ncx", "etavec"), ("ncy", "chi1vec"), ("ncz", "chi2vec")):
                shifted = np.asarray(group.attrs.get(attr, []), dtype=np.float64)
                if shifted.shape != grids[key].shape or not np.array_equal(shifted, grids[key] + 2.0):
                    raise ValueError(f"Unexpected {name}/{attr} spline-grid metadata")

            n_frequency = int(group["Bamp"].shape[0])
            if group["Bamp"].shape != (n_frequency, n_frequency):
                raise ValueError(f"Amplitude basis is not square in {name}")
            if group["Bphase"].shape != (n_frequency, n_frequency):
                raise ValueError(f"Phase basis is not square in {name}")
            if group["Mf_grid_Amp"].shape != (n_frequency,) or group["Mf_grid_Phi"].shape != (n_frequency,):
                raise ValueError(f"ROM frequency grids do not match the basis in {name}")
            padded = tuple(len(grids[key]) + 2 for key in ("etavec", "chi1vec", "chi2vec"))
            expected_coefficients = (n_frequency,) + padded
            for dataset in ("Amp_ciall", "Phase_ciall"):
                if int(np.prod(group[dataset].shape)) != int(np.prod(expected_coefficients)):
                    raise ValueError(f"Unexpected {name}/{dataset} coefficient-table size")
            schemas.append(
                ROMSubmodelSchema(
                    name=name,
                    description=str(group.attrs.get("Description", b""), "utf-8")
                    if isinstance(group.attrs.get("Description", b""), bytes)
                    else str(group.attrs.get("Description", "")),
                    n_frequency=n_frequency,
                    n_eta=len(grids["etavec"]),
                    n_chi1=len(grids["chi1vec"]),
                    n_chi2=len(grids["chi2vec"]),
                    # The public datasets are flattened; this is the logical
                    # frequency-first shape implied by their size and grids.
                    amplitude_coefficients=expected_coefficients,
                    phase_coefficients=expected_coefficients,
                )
            )
    return ROMFileSchema(
        path=resolved,
        sha256=digest,
        canonical_basename=canonical,
        groups=tuple(schemas),
    )
