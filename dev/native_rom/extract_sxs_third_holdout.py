"""Fetch and decode the prospective third SXS holdout for dev-only comparisons.

Run in the isolated environment containing sxs==2026.0.5. Archive files stay
outside the repository; every download is checked against the pinned catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sxs
from sxscatalog.utilities.downloads import download_file

SCHEMAS = {
    "diffgw.seobnrv4.sxs-third-holdout-plan.v1":
        "diffgw.seobnrv4.sxs-third-holdout-extraction.v1",
    "diffgw.seobnrv4.sxs-paper-catalog-plan.v1":
        "diffgw.seobnrv4.sxs-paper-catalog-extraction.v1",
}


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def fetch(path: Path, details: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or digest(path, "md5") != details["md5"]:
        temporary = path.with_suffix(path.suffix + ".partial")
        download_file(details["url"], temporary, if_newer=False)
        if temporary.stat().st_size != details["size"]:
            temporary.unlink()
            raise ValueError(f"archive size mismatch: {path}")
        if digest(temporary, "md5") != details["md5"]:
            temporary.unlink()
            raise ValueError(f"archive checksum mismatch: {path}")
        temporary.replace(path)
    if path.stat().st_size != details["size"] or digest(path, "md5") != details["md5"]:
        raise ValueError(f"archive identity mismatch: {path}")
    return dict(md5=details["md5"], sha256=digest(path, "sha256"), size=details["size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan.get("schema") not in SCHEMAS or plan.get("qualification_claim") is not False
            or len(plan.get("cases", {})) != 5):
        raise ValueError("unexpected SXS comparison plan")
    if sxs.__version__ != "2026.0.5":
        raise ValueError(f"expected sxs 2026.0.5, got {sxs.__version__}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(schema=SCHEMAS[plan["schema"]],
                  plan_sha256=digest(args.plan, "sha256"),
                  script_sha256=digest(Path(__file__), "sha256"),
                  sxs_version=sxs.__version__, cases={})
    manifest = args.output_dir / "extraction.json"
    for simulation, case in plan["cases"].items():
        case_dir = simulation.replace(":", "_")
        entry = dict(archive_files={}, extracted={})
        for archive_name, details in case["files"].items():
            lev, name = archive_name.split(":", 1)
            path = args.archive_dir / case_dir / lev / name
            entry["archive_files"][archive_name] = fetch(path, details)
        for lev in case["levels"]:
            waveform = sxs.load(args.archive_dir / case_dir / f"Lev{lev}" / "Strain_N2.h5")
            if waveform.spin_weight != -2 or waveform.data_type != "h":
                raise ValueError(f"expected spin -2 strain for {simulation} Lev{lev}")
            time = np.asarray(waveform.t, dtype=np.float64)
            mode = np.asarray(waveform.data[:, waveform.index(2, 2)], dtype=np.complex128)
            if (time.ndim != 1 or mode.shape != time.shape or time.size < 100
                    or not np.all(np.diff(time) > 0)
                    or not np.isfinite(time).all() or not np.isfinite(mode).all()):
                raise ValueError(f"invalid decoded mode: {simulation} Lev{lev}")
            output = args.output_dir / f"{case_dir}_Lev{lev}_22.npz"
            np.savez(output, time_M=time, mode=mode)
            entry["extracted"][f"Lev{lev}"] = dict(
                path=output.name, sha256=digest(output, "sha256"),
                samples=int(time.size), peak_amplitude=float(max(abs(mode))))
        result["cases"][simulation] = entry
        manifest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{simulation} decoded", flush=True)
    print(json.dumps(dict(manifest=str(manifest), cases=list(result["cases"]))))


if __name__ == "__main__":
    main()
