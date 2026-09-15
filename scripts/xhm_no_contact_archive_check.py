#!/usr/bin/env python3
"""Check the public paper-era XHM ancillary archive without reading bodies.

The arXiv v2 source archive is a possible external input for the *paper-era*
target only.  This checker hashes the downloaded archive and checks its member
inventory; it never extracts or evaluates calibration files.  It therefore
cannot turn the archive into a release-122022 provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable


EXPECTED_PREFIX_COUNTS = {
    "anc/MixingCoefficients/": 8,
    "anc/QNMs/": 10,
    "anc/ParspaceFits/AmpFits/": 37,
    "anc/ParspaceFits/PhaseFits/": 33,
}
EXPECTED_METADATA_MEMBERS = {
    "anc/ParspaceFits/README",
    "anc/ParspaceFits/IMRPhenomXHM_AmplitudeFits.nb",
    "anc/ParspaceFits/IMRPhenomXHM_PhaseFits.nb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_names(path: Path) -> list[str]:
    # Listing tar members is metadata inspection; no member body is opened.
    with tarfile.open(path, mode="r:*") as archive:
        return sorted(member.name for member in archive.getmembers() if member.isfile())


def inspect_archive(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    """Return a metadata-only integrity and inventory report for ``path``."""

    archive_path = Path(path)
    actual_sha256 = _sha256(archive_path)
    names = _file_names(archive_path)
    prefix_counts = {
        prefix: sum(name.startswith(prefix) for name in names)
        for prefix in EXPECTED_PREFIX_COUNTS
    }
    metadata_members = sorted(EXPECTED_METADATA_MEMBERS.intersection(names))
    inventory_verified = (
        prefix_counts == EXPECTED_PREFIX_COUNTS
        and set(metadata_members) == EXPECTED_METADATA_MEMBERS
    )
    if expected_sha256 is None:
        sha256_status = "unchecked"
        identity_verified = False
    elif actual_sha256 == expected_sha256:
        sha256_status = "match"
        identity_verified = True
    else:
        sha256_status = "mismatch"
        identity_verified = False
    report = {
        "archive": str(archive_path),
        "sha256": actual_sha256,
        "expected_sha256_supplied": expected_sha256 is not None,
        "sha256_status": sha256_status,
        "file_count": len(names),
        "prefix_counts": prefix_counts,
        "metadata_members": metadata_members,
        "inventory_verified": inventory_verified,
        "identity_verified": identity_verified,
    }
    # A complete-looking inventory is insufficient to identify a release.
    report["verified"] = bool(report["inventory_verified"] and identity_verified)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="downloaded arXiv source archive")
    parser.add_argument("--sha256", help="expected archive SHA-256")
    args = parser.parse_args(argv)
    report = inspect_archive(args.archive, expected_sha256=args.sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
