"""Decode pinned SXS:BBH:2123 strain into a development-only (2,2) input.

Run with an isolated Python environment containing sxs==2026.0.5. Download the
six files named in sxs_2123_plan.json into --archive-dir/Lev{3,4}/ first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sxs


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "diffgw.seobnrv4.sxs-2123-plan.v1":
        raise ValueError("unexpected SXS comparison plan")
    if sxs.__version__ != "2026.0.5":
        raise ValueError(f"expected sxs 2026.0.5, got {sxs.__version__}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for archive_name, details in plan["catalog"]["waveforms"].items():
        lev, name = archive_name.split(":", 1)
        path = args.archive_dir / lev / name
        actual = digest(path, "md5")
        if actual != details["md5"]:
            raise ValueError(f"archive checksum mismatch: {archive_name}")
        files[archive_name] = dict(md5=actual, sha256=digest(path, "sha256"))
    extracted = {}
    for lev in (3, 4):
        waveform = sxs.load(args.archive_dir / f"Lev{lev}" / "Strain_N2.h5")
        if waveform.spin_weight != -2 or waveform.data_type != "h":
            raise ValueError("expected spin -2 gravitational-wave strain")
        time = np.asarray(waveform.t, dtype=np.float64)
        mode = np.asarray(waveform.data[:, waveform.index(2, 2)], dtype=np.complex128)
        if (time.ndim != 1 or mode.shape != time.shape or time.size < 100
                or not np.all(np.diff(time) > 0)
                or not np.isfinite(time).all() or not np.isfinite(mode).all()):
            raise ValueError("invalid decoded SXS mode")
        path = args.output_dir / f"Lev{lev}_22.npz"
        np.savez(path, time_M=time, mode=mode)
        extracted[f"Lev{lev}"] = dict(path=path.name, sha256=digest(path, "sha256"),
                                     samples=int(time.size), peak_amplitude=float(max(abs(mode))))
    manifest = dict(schema="diffgw.seobnrv4.sxs-2123-extraction.v1",
                    plan_sha256=digest(args.plan, "sha256"), sxs_version=sxs.__version__,
                    archive_files=files, extracted=extracted)
    (args.output_dir / "extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(extracted=extracted, manifest=str(args.output_dir / "extraction.json"))))


if __name__ == "__main__":
    main()
