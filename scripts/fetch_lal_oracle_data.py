"""Stage checksum-pinned LAL runtime data for external verification only.

This file is not a TorchWave runtime dependency or a source of native fits.
Dataset: https://zenodo.org/records/14999310 (published version v1).
"""

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen


FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
URL = f"https://zenodo.org/records/14999310/files/{FILENAME}?download=1"
MD5 = "bb0eab9ae721af6abd06c4872b13e477"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    destination = args.directory / FILENAME
    if destination.exists():
        if checksum(destination) != MD5:
            raise ValueError(f"Existing oracle data checksum mismatch: {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urlopen(URL, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if checksum(temporary) != MD5:
            raise ValueError(f"Downloaded oracle data checksum mismatch: {temporary}")
        temporary.replace(destination)
    print(f"Verified {destination} md5={MD5}")


if __name__ == "__main__":
    main()
