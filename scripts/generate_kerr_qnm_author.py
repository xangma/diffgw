#!/usr/bin/env python3
"""Reproduce historical Cook reference fixtures for external verification.

Requires numpy, scipy, h5py. Example, from the repository root::

    python scripts/generate_kerr_qnm_author.py --data-dir /tmp/cook-qnm --download
    python scripts/generate_kerr_qnm_author.py --data-dir /tmp/cook-qnm --check

The large upstream HDF5 files are not committed. Pinning each complete file's
SHA256 prevents another data version from silently changing the representation.
This script does not generate or modify runtime QNM coefficients. TorchWave's
runtime now uses scripts/generate_kerr_qnm_independent.py. The historical
author-spline errors are retained to reproduce the earlier validation record.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import h5py
import numpy as np
from scipy.interpolate import make_interp_spline

DEFAULT_OUTPUT = Path.home() / ".cache" / "torchwave" / "external-validation" / "cook"
SOURCE = "https://zenodo.org/records/16801267"
SHA256 = {
    0: "9797479ed07ebe8f5172b5c591b5cf0bf3196cb3a843c0bcd987df68265971cc",
    1: "e3a52a852dbc3687a415da4047c4e0aaccecf0a0578f2d55086af964d80b830f",
    2: "8834f486ad99592f6f23765aa4a39f0a67727a3c7cb7c7377d18bde74706bb40",
}
PUBLISHED_MD5 = {
    0: "62a1086b7e59da567c624f331c39d119",
    1: "e767ac934083c364790d250689ee009c",
    2: "32a8b90d276f778e161a37479f5c9b40",
}
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
DEGREE = 5
DOMAIN = (0.0, 0.99999)


def source_file(directory, n, download):
    filename = f"KerrQNM_{n:02}.h5"
    path = directory / filename
    if not path.exists() and download:
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".h5.part")
        url = f"https://zenodo.org/api/records/16801267/files/{filename}/content"
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
        partial.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256[n]:
        raise ValueError(f"Unexpected SHA256 for {path}: {digest}")
    return path


def split_rows(spin):
    """Predefined disjoint training, development and final-test row indices.

    Row indices refer to the author's original ascending-spin sequence. The
    final-test rows (index mod 10 == 1) were not inspected during development.
    Data near extremality are retained more densely because x=sqrt(1-a) is
    unevenly sampled there. Source endpoints are always training rows.
    """
    index = np.arange(len(spin))
    allowed = ~np.isin(index % 10, [1, 3])
    train = allowed & (((spin < .9) & (index % 4 == 0))
                       | ((spin >= .9) & (spin < .97) & (index % 2 == 0))
                       | (spin >= .97))
    train[[0, -1]] = True
    in_domain = (spin >= DOMAIN[0]) & (spin <= DOMAIN[1])
    development = (index % 10 == 3) & ~train & in_domain
    test = (index % 10 == 1) & ~train & in_domain
    assert not np.any(train & development | train & test | development & test)
    return train, development, test


def errors(spline, x, values, mask):
    delta = np.abs(spline(x[mask]) - values[mask])
    return {"count": int(mask.sum()),
            "max_abs": delta.max(axis=0).tolist(),
            "max_relative": (delta / np.abs(values[mask])).max(axis=0).tolist()}


def write_or_check(path, contents, check):
    if check:
        if path.read_text() != contents:
            raise SystemExit(f"Generated output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("TORCHWAVE_EXTERNAL_VALIDATION_DIR", DEFAULT_OUTPUT)))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    modes, fixtures = {}, {}
    for n in range(3):
        with h5py.File(source_file(args.data_dir, n, args.download), "r") as source:
            for l, m in MODES:
                key = f"n{n:02}/m+{m:02}/{{{l},{m},{n}}}"
                # Only the author's spin and frequency columns enter the fit.
                rows = source[key][:, :3]
                spin, values = rows[:, 0], rows[:, 1:3]
                if not (np.all(np.diff(spin) > 0) and np.all(values[:, 1] < 0)):
                    raise ValueError(f"Unexpected author sequence: {key}")
                train, development, test = split_rows(spin)
                x = np.sqrt(1 - spin)
                spline = make_interp_spline(x[train][::-1], values[train][::-1], k=DEGREE)
                label = f"{l},{m},{n}"
                modes[label] = {"author_dataset": key, "source_rows": len(rows),
                                "training_rows": int(train.sum()),
                                "training_row_indices": np.flatnonzero(train).tolist(),
                                "development": errors(spline, x, values, development),
                                "holdout": errors(spline, x, values, test),
                                "holdout_spin_le_0.99": errors(spline, x, values, test & (spin <= .99))}
                fixtures[label] = {"row_indices": np.flatnonzero(test).tolist(),
                                   "rows": rows[test].tolist()}
    provenance = {"source": SOURCE, "doi": "10.5281/zenodo.16801267", "version": "v4",
                  "author": "Gregory B. Cook", "license": "ODC-By-1.0",
                  "license_url": "https://opendatacommons.org/licenses/by/1-0/",
                  "sha256": {f"KerrQNM_{n:02}.h5": sha for n, sha in SHA256.items()},
                  "source_files": {f"KerrQNM_{n:02}.h5": {
                      "url": f"https://zenodo.org/api/records/16801267/files/KerrQNM_{n:02}.h5/content",
                      "published_md5": PUBLISHED_MD5[n], "sha256": SHA256[n]}
                      for n in range(3)},
                  "spin_domain": DOMAIN, "coordinate": "sqrt(1-a)", "spline_degree": DEGREE,
                  "frequency_columns": ["Re(M omega)", "Im(M omega)"],
                  "modes": modes}
    fixture = {"source": SOURCE, "license": "ODC-By-1.0",
               "license_url": "https://opendatacommons.org/licenses/by/1-0/",
               "attribution": "Gregory B. Cook, Kerr Modes: Phase fixed gravitational QNMs and TTMs, v4 (2025)",
               "description": "Unmodified author rows withheld from spline fitting; columns a, Re(M omega), Im(M omega).",
               "modes": fixtures}
    for filename, data in [("kerr_qnm_author_validation.json", provenance),
                           ("kerr_qnm_author_holdout.json", fixture)]:
        path = args.output_dir / filename
        write_or_check(path, json.dumps(data, indent=2) + "\n", args.check)
    worst = np.max([v["holdout"]["max_abs"] for v in modes.values()], axis=0)
    lower = np.max([v["holdout_spin_le_0.99"]["max_abs"] for v in modes.values()], axis=0)
    print(f"{'Verified' if args.check else 'Generated'} {len(modes)} modes; final holdout max |delta(Re,Im)|={worst}; a<=.99: {lower}")


if __name__ == "__main__":
    main()
