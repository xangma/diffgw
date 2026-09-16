"""Locate and download optional, user-managed waveform data files."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Optional, Union
from urllib.request import urlopen


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    filename: str
    sha256: str
    url: str
    release: str
    upstream_url: str
    terms: str
    status: str = "available"
    size_bytes: Optional[int] = None


DATASETS = {
    "seobnrv4-rom-v3": DatasetSpec(
        dataset_id="seobnrv4-rom-v3",
        filename="SEOBNRv4ROM_v3.0.hdf5",
        sha256="4ff5317967d757fcad9323a74b75f4f63894ed87f4b1852d46cbe47cde488b9c",
        url="https://zenodo.org/records/14999310/files/SEOBNRv4ROM_v3.0.hdf5?download=1",
        release="3.0",
        upstream_url="https://zenodo.org/records/14999310",
        terms="See the upstream record and its attached data terms; those terms govern this file.",
        size_bytes=333279176,
    ),
}

def _default_cache() -> Path:
    value = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(value).expanduser() / "torchwave" / "waveform-data"


def get_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASETS[dataset_id]
    except KeyError:
        raise KeyError(f"Unknown dataset {dataset_id!r}; available: {', '.join(DATASETS)}") from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset_path(dataset_id: str, path: Optional[Union[str, os.PathLike]] = None, *, verify: bool = True) -> Path:
    spec = get_dataset(dataset_id)
    candidates = []
    if path is not None:
        chosen = Path(path).expanduser()
        candidates = [chosen / spec.filename if chosen.is_dir() else chosen]
    else:
        for value in (os.environ.get("TORCHWAVE_DATA_PATH"), os.environ.get("LAL_DATA_PATH")):
            if value:
                candidates.extend(Path(item).expanduser() / spec.filename for item in value.split(os.pathsep) if item)
        candidates.append(_default_cache() / spec.filename)
    for candidate in candidates:
        if candidate.is_file():
            if verify and _sha256(candidate) != spec.sha256:
                raise ValueError(f"Checksum mismatch for {candidate} (expected {spec.sha256})")
            return candidate.resolve()
    if path is not None:
        raise FileNotFoundError(f"Dataset {dataset_id} not found at {candidates[0]}")
    raise FileNotFoundError(f"Dataset {dataset_id} is unavailable locally; download it with `python -m torchwave.external_data fetch {dataset_id}`")


def _inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _forbidden_roots() -> tuple[Path, ...]:
    here = Path(__file__).resolve()
    roots = [here.parent]
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            roots.append(parent)
    return tuple(roots)


def download_dataset(dataset_id: str, directory: Optional[Union[str, os.PathLike]] = None) -> Path:
    spec = get_dataset(dataset_id)
    destination_dir = Path(directory).expanduser() if directory is not None else _default_cache()
    destination_dir = destination_dir.resolve()
    destination = destination_dir / spec.filename
    if any(_inside(destination, root) for root in _forbidden_roots()):
        raise ValueError("Refusing to download data inside the installed package or source checkout; choose an external cache directory")
    destination_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and _sha256(destination) == spec.sha256:
            return destination
        raise ValueError(f"Existing dataset {destination} has the wrong checksum; remove it manually before retrying")
    temp_name = None
    try:
        with urlopen(spec.url, timeout=30) as response:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination_dir, prefix=f".{spec.filename}.", suffix=".part", delete=False) as output:
                temp_name = Path(output.name)
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if spec.size_bytes is not None and total > spec.size_bytes:
                        raise ValueError(f"Downloaded file exceeds expected size ({spec.size_bytes} bytes)")
                    output.write(block)
        if spec.size_bytes is not None and total != spec.size_bytes:
            raise ValueError(f"Downloaded file has size {total}, expected {spec.size_bytes}")
        if _sha256(temp_name) != spec.sha256:
            raise ValueError(f"Downloaded checksum mismatch (expected {spec.sha256}); partial file was removed")
        try:
            # Hard-link publication is atomic and refuses to replace a file
            # created by another downloader after our initial existence check.
            os.link(temp_name, destination)
        except FileExistsError:
            if destination.is_file() and _sha256(destination) == spec.sha256:
                return destination.resolve()
            raise ValueError(f"A concurrent dataset appeared at {destination} with the wrong checksum") from None
        temp_name.unlink()
        return destination.resolve()
    finally:
        if temp_name is not None and temp_name.exists():
            temp_name.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Locate and fetch optional torchwave numerical data")
    parser.add_argument("command", choices=("list", "info", "verify", "fetch"))
    parser.add_argument("dataset_id", nargs="?")
    parser.add_argument("--directory", "-d", help="destination directory for fetch or directory to verify")
    parser.add_argument("--path", help="explicit file or directory to verify")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("seobnrv4-rom-v3\tavailable")
        print("xhm122022\tunavailable: no complete standalone public asset is registered")
        return 0
    if args.path and args.command != "verify":
        print("error: --path is only valid with verify", file=sys.stderr)
        return 2
    if not args.dataset_id:
        parser.error("dataset_id is required for this command")
    if args.dataset_id.lower() == "xhm122022":
        if args.command == "info":
            print("xhm122022: unavailable (no complete standalone public asset is registered)")
            return 0
        print("xhm122022 is unavailable: no complete standalone public asset is registered", file=sys.stderr)
        return 2
    try:
        spec = get_dataset(args.dataset_id)
        if args.command == "info":
            print(f"{spec.dataset_id}: {spec.filename} (release {spec.release})")
            print(f"Source: {spec.upstream_url}\nTerms: {spec.terms}\nSHA256: {spec.sha256}")
        elif args.command == "verify":
            if args.path and args.directory:
                raise ValueError("use only one of --path and --directory")
            print(resolve_dataset_path(args.dataset_id, args.path or args.directory))
        else:
            print(f"Source: {spec.upstream_url}\nTerms: {spec.terms}", file=sys.stderr)
            print(download_dataset(args.dataset_id, args.directory))
        return 0
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
