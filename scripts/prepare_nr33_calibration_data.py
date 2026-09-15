#!/usr/bin/env python3
"""Download and extract full SXS NR (3,3) strain modes for the calibration pilot.

The raw files are kept outside the repository.  The extractor writes one NPZ
per case with the complete available time series (no ringdown truncation),
``time``, ``h33`` and ``h22``, and updates a provenance manifest.

The SXS ``rotating_paired_diff_multishuffle_bzip2`` format is decoded by the
public ``sxs`` package.  Install that package in an external environment, for
example ``python -m venv /tmp/torchwave-nr33-pilot/venv && .../pip install
sxs``; this script does not add it to TorchWave's dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request
from typing import Any


CATALOG_DOI = "10.5281/zenodo.15415231"
CATALOG_URL = "https://zenodo.org/records/15415231"
LICENSE_ID = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

# One case per q, using the v2.0-era public SXS files.  The split is frozen by
# simulation identity before any numerical fitting: q=1.5,2,4,5,8 training;
# q=3,6 holdout.  N=2 copies of the holdouts are validation-only.
CASES: tuple[dict[str, Any], ...] = (
    dict(sxs_id="SXS:BBH:0008", q=1.4999989750008755, resolution="Lev4",
         split="training", initial_mass_ratio=1.4999999994664486,
         initial_spin1=[-3.06985301001e-08, 2.1650715e-09, -1.06963082e-08], initial_spin2=[-7.56632000002e-11, -4.36637e-11, 1.2565429e-09],
         reference_spin1=[-1.71655005639e-09, -7.40668572557e-10, 1.47347113534e-07],
         reference_spin2=[-4.20259055639e-09, -7.11391027445e-09, 2.36765414661e-07],
         reference_time=270.0, reference_mass1=0.600000102526, reference_mass2=0.400000341684,
         remnant_mass=0.955267129967, remnant_spin=[5.37191132331e-08, 2.73061848107e-08, 0.664090718508],
         number_of_orbits=21.2800498763, dataset_doi="10.5281/zenodo.13166243",
         record_url="https://zenodo.org/records/13166243",
         waveform_url="https://zenodo.org/api/records/13166243/files/Lev4:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13166243/files/Lev4:Strain_N3.json/content",
         md5="a0a1d88aabb4c7beb893038543bb8c88", size=2637265,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0169", q=1.9999969252812553, resolution="Lev3",
         split="training", initial_mass_ratio=2.000000008168811,
         initial_spin1=[1.90887421316e-09, 2.05426148542e-09, 3.13694116327e-08], initial_spin2=[-9.45869499772e-12, -3.47835634777e-11, 6.97373888336e-10],
         reference_spin1=[-2.43781551504e-10, -2.43781551504e-10, 2.49131120301e-07],
         reference_spin2=[3.81100904886e-10, 3.81100904886e-10, 5.29196403758e-07],
         reference_time=260.0, reference_mass1=0.666666781212, reference_mass2=0.33333390306,
         remnant_mass=0.961246130167, remnant_spin=[-9.66648544771e-05, -9.66648544771e-05, 0.623439030411],
         number_of_orbits=15.6805252633, dataset_doi="10.5281/zenodo.13167391",
         record_url="https://zenodo.org/records/13167391",
         waveform_url="https://zenodo.org/api/records/13167391/files/Lev3:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13167391/files/Lev3:Strain_N3.json/content",
         md5="e4cff549176b1b9b8aa6df85b46bc734", size=2133351,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0030", q=2.999993377707023, resolution="Lev3",
         split="holdout", initial_mass_ratio=3.0000000003832885,
         initial_spin1=[1.44101e-11, -7.58119999999e-12, 1.1691109e-09], initial_spin2=[-2.85409999999e-12, -1.76411000001e-11, 4.363743e-10],
         reference_spin1=[5.98204669549e-09, 3.57374208645e-09, 2.89245176314e-07],
         reference_spin2=[-2.85524183459e-09, 5.17844953009e-09, 5.03495783085e-07],
         reference_time=260.0, reference_mass1=0.75000004581, reference_mass2=0.250000567129,
         remnant_mass=0.971212971306, remnant_spin=[-9.84792248965e-09, -2.4910996981e-09, 0.5406441997],
         number_of_orbits=18.2227985811, dataset_doi="10.5281/zenodo.13167328",
         record_url="https://zenodo.org/records/13167328",
         waveform_url="https://zenodo.org/api/records/13167328/files/Lev3:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13167328/files/Lev3:Strain_N3.json/content",
         md5="d6e1a33a3ce9e0143bc14da352898141", size=2639738,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0182", q=3.9999765924197983, resolution="Lev2",
         split="training", initial_mass_ratio=3.9999999999480234,
         initial_spin1=[2.528364e-09, -9.92451e-11, 1.54966285e-08], initial_spin2=[7.360544e-10, -9.99069e-10, -1.36496993e-08],
         reference_spin1=[1.02500072124e-07, -3.04088010689e-08, 1.19525990708e-05],
         reference_spin2=[3.99465024349e-08, 7.37704067693e-09, -1.50504134191e-05],
         reference_time=310.0, reference_mass1=0.799999412804, reference_mass2=0.200001023586,
         remnant_mass=0.977899284822, remnant_spin=[2.92851075763e-08, -2.72228568822e-08, 0.471647259902],
         number_of_orbits=15.5562869105, dataset_doi="10.5281/zenodo.13159588",
         record_url="https://zenodo.org/records/13159588",
         waveform_url="https://zenodo.org/api/records/13159588/files/Lev2:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13159588/files/Lev2:Strain_N3.json/content",
         md5="a0a00febb0126bf61babaa5da5f29372", size=2337541,
         citation_dois=["10.1103/PhysRevLett.115.121102"]),
    dict(sxs_id="SXS:BBH:0056", q=4.9999883276805726, resolution="Lev3",
         split="training", initial_mass_ratio=4.999999997989988,
         initial_spin1=[-5.6013e-12, 3.31699999999e-13, 6.81223089999e-09], initial_spin2=[-2.7e-15, 1.1e-15, 4.14860000001e-12],
         reference_spin1=[1.02351360187e-09, 2.3204167726e-11, 2.32159011654e-07],
         reference_spin2=[-1.60370401128e-11, -1.97399520676e-11, 2.69647768421e-07],
         reference_time=320.0, reference_mass1=0.833333285079, reference_mass2=0.166667046094,
         remnant_mass=0.982368274666, remnant_spin=[-1.14499013362e-09, 7.71560515973e-10, 0.416614481622],
         number_of_orbits=28.8102224144, dataset_doi="10.5281/zenodo.13165791",
         record_url="https://zenodo.org/records/13165791",
         waveform_url="https://zenodo.org/api/records/13165791/files/Lev3:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13165791/files/Lev3:Strain_N3.json/content",
         md5="7c6b70f7a027e45204918f9fb762f771", size=3837769,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0166", q=5.99997866468524, resolution="Lev5",
         split="holdout", initial_mass_ratio=6.000000135581912,
         initial_spin1=[-1.62141968958e-10, -1.59599118445e-10, 9.18943070825e-09], initial_spin2=[3.18171978598e-11, -2.05397996693e-11, 9.15205298629e-09],
         reference_spin1=[-2.17890253383e-10, -2.17890253383e-10, 4.5272937406e-07],
         reference_spin2=[1.41955616166e-10, 1.41955616166e-10, 4.67194783085e-07],
         reference_time=200.0, reference_mass1=0.85714315334, reference_mass2=0.142857700209,
         remnant_mass=0.985471054121, remnant_spin=[-3.04602721315e-05, -3.04602721315e-05, 0.372452050361],
         number_of_orbits=21.5588781341, dataset_doi="10.5281/zenodo.13167344",
         record_url="https://zenodo.org/records/13167344",
         waveform_url="https://zenodo.org/api/records/13167344/files/Lev5:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13167344/files/Lev5:Strain_N3.json/content",
         md5="1efca43cdc182bd247b478d7ab891887", size=2693916,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0063", q=7.9999692834078795, resolution="Lev3",
         split="training", initial_mass_ratio=7.999999995504958,
         initial_spin1=[-3.33206e-11, 1.92658e-11, 1.22804514e-08], initial_spin2=[-2.5e-14, -2.328e-13, -2.27426e-11],
         reference_spin1=[1.26635047766e-09, -1.48897378e-10, 2.0682688609e-07],
         reference_spin2=[-9.27878134584e-11, 6.08100712406e-11, 4.29282215789e-07],
         reference_time=280.0, reference_mass1=0.888889205519, reference_mass2=0.111111577311,
         remnant_mass=0.989425258975, remnant_spin=[6.0703725516e-09, -1.51640715846e-08, 0.306743698866],
         number_of_orbits=25.8255298679, dataset_doi="10.5281/zenodo.13167443",
         record_url="https://zenodo.org/records/13167443",
         waveform_url="https://zenodo.org/api/records/13167443/files/Lev3:Strain_N3.h5/content",
         metadata_url="https://zenodo.org/api/records/13167443/files/Lev3:Strain_N3.json/content",
         md5="540b88e03928c8a4b401134d0f50ced8", size=3559285,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    # Same simulations and resolutions as the baselines, with N=2 instead of
    # N=3.  These are validation-only records and never enter the split.
    dict(sxs_id="SXS:BBH:0030", raw_stem="SXS_BBH_0030_N2", q=2.999993377707023,
         resolution="Lev3", extrapolation_order=2, split="validation_only",
         initial_mass_ratio=3.0000000003832885,
         initial_spin1=[1.44101e-11, -7.58119999999e-12, 1.1691109e-09], initial_spin2=[-2.85409999999e-12, -1.76411000001e-11, 4.363743e-10],
         reference_spin1=[5.98204669549e-09, 3.57374208645e-09, 2.89245176314e-07],
         reference_spin2=[-2.85524183459e-09, 5.17844953009e-09, 5.03495783085e-07],
         reference_time=260.0, reference_mass1=0.75000004581, reference_mass2=0.250000567129,
         remnant_mass=0.971212971306, remnant_spin=[-9.84792248965e-09, -2.4910996981e-09, 0.5406441997],
         number_of_orbits=18.2227985811, dataset_doi="10.5281/zenodo.13167328",
         record_url="https://zenodo.org/records/13167328",
         waveform_url="https://zenodo.org/api/records/13167328/files/Lev3:Strain_N2.h5/content",
         metadata_url="https://zenodo.org/api/records/13167328/files/Lev3:Strain_N2.json/content",
         md5="3b19ac53a4feacfb7f4ce8d8fb18e492", size=2295638,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
    dict(sxs_id="SXS:BBH:0166", raw_stem="SXS_BBH_0166_N2", q=5.99997866468524,
         resolution="Lev5", extrapolation_order=2, split="validation_only",
         initial_mass_ratio=6.000000135581912,
         initial_spin1=[-1.62141968958e-10, -1.59599118445e-10, 9.18943070825e-09], initial_spin2=[3.18171978598e-11, -2.05397996693e-11, 9.15205298629e-09],
         reference_spin1=[-2.17890253383e-10, -2.17890253383e-10, 4.5272937406e-07],
         reference_spin2=[1.41955616166e-10, 1.41955616166e-10, 4.67194783085e-07],
         reference_time=200.0, reference_mass1=0.85714315334, reference_mass2=0.142857700209,
         remnant_mass=0.985471054121, remnant_spin=[-3.04602721315e-05, -3.04602721315e-05, 0.372452050361],
         number_of_orbits=21.5588781341, dataset_doi="10.5281/zenodo.13167344",
         record_url="https://zenodo.org/records/13167344",
         waveform_url="https://zenodo.org/api/records/13167344/files/Lev5:Strain_N2.h5/content",
         metadata_url="https://zenodo.org/api/records/13167344/files/Lev5:Strain_N2.json/content",
         md5="afa815578ebd015c937c17c39d77b747", size=2420124,
         citation_dois=["10.1103/PhysRevLett.111.241104"]),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, path: pathlib.Path, expected_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch for existing {path}")
        return
    with urllib.request.urlopen(url) as source, path.open("wb") as target:
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > expected_size:
                raise ValueError(f"download exceeded expected size for {path}")
            target.write(block)
    if total != expected_size:
        raise ValueError(f"download size {total} != expected {expected_size} for {path}")


def extract_case(case: dict[str, Any], raw_dir: pathlib.Path, out_dir: pathlib.Path) -> dict[str, Any]:
    try:
        import numpy as np
        import sxs
    except ImportError as exc:
        raise SystemExit("extracting this format requires numpy and the public sxs package") from exc
    stem = case["sxs_id"].replace(":", "_")
    raw_stem = case.get("raw_stem", stem)
    h5_path = raw_dir / f"{raw_stem}.h5"
    json_path = raw_dir / f"{raw_stem}.json"
    out_path = out_dir / f"{raw_stem}.npz"
    waveform = sxs.load(str(h5_path))
    source_frame_type = getattr(waveform, "frame_type", "unknown")
    frame_transform = "none"
    if source_frame_type != "inertial":
        converted = waveform.to_inertial_frame()
        waveform = converted if converted is not None else waveform
        frame_transform = "to_inertial_frame"
    extraction_frame_type = getattr(waveform, "frame_type", "unknown")
    time = np.asarray(waveform.t, dtype=np.float64)
    h33 = np.asarray(waveform.data[:, waveform.index(3, 3)], dtype=np.complex128)
    h22 = np.asarray(waveform.data[:, waveform.index(2, 2)], dtype=np.complex128)
    if (time.ndim != 1 or h33.shape != time.shape or h22.shape != time.shape
            or not np.iscomplexobj(h33) or not np.iscomplexobj(h22)
            or not np.isfinite(time).all() or not np.isfinite(h33).all()
            or not np.isfinite(h22).all()):
        raise ValueError(f"unexpected mode shapes for {case['sxs_id']}")
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"time is not strictly increasing for {case['sxs_id']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, time=time, h33=h33, h22=h22)
    wave_meta = json.loads(json_path.read_text())
    peak_index = int(np.argmax(np.abs(h22)))
    try:
        import sxs
        sxs_version = getattr(sxs, "__version__", "unknown")
    except ImportError:  # pragma: no cover - import was checked above
        sxs_version = "unknown"
    return {"npz": str(out_path), "n_times": int(time.size),
            "time_start": float(time[0]), "time_end": float(time[-1]),
            "dt_min": float(np.min(np.diff(time))), "dt_max": float(np.max(np.diff(time))),
            "peak_time_h22": float(time[peak_index]), "peak_time": float(time[peak_index]),
            "h33_dtype": str(h33.dtype), "h22_dtype": str(h22.dtype),
            "npz_sha256": sha256(out_path),
            "sxs_package_version": sxs_version,
            "source_frame_type": source_frame_type,
            "extraction_frame_type": extraction_frame_type,
            "frame_transform": frame_transform,
            "mode_convention": "complex spin-weight -2 strain h_lm; sxs WaveformModes indexing",
            "waveform_json": wave_meta}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("/tmp/torchwave-nr33-pilot/raw"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/tmp/torchwave-nr33-pilot/npz"))
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json"))
    parser.add_argument("--download", action="store_true", help="download missing raw h5/json files")
    args = parser.parse_args()
    records = []
    for case in CASES:
        stem = case["sxs_id"].replace(":", "_")
        raw_stem = case.get("raw_stem", stem)
        h5 = args.raw_dir / f"{raw_stem}.h5"
        metadata = args.raw_dir / f"{raw_stem}.json"
        if args.download:
            fetch(case["waveform_url"], h5, case["size"])
            # Metadata files are tiny; obtain the server response directly.
            metadata.parent.mkdir(parents=True, exist_ok=True)
            if not metadata.exists():
                urllib.request.urlretrieve(case["metadata_url"], metadata)
        if not h5.exists() or not metadata.exists():
            raise SystemExit(f"missing {h5} or {metadata}; pass --download or stage raw files")
        actual_sha = sha256(h5)
        actual_md5 = md5(h5)
        if actual_md5 != case["md5"]:
            raise ValueError(f"published MD5 mismatch for {case['sxs_id']}: {actual_md5} != {case['md5']}")
        item = dict(case)
        item.update({"license": LICENSE_ID, "license_url": LICENSE_URL,
                     "catalog_doi": CATALOG_DOI, "catalog_url": CATALOG_URL,
                     "dataset_version": "v2.0",
                     "raw_h5": str(h5), "raw_h5_sha256": actual_sha,
                     "raw_h5_md5": actual_md5,
                     "raw_h5_md5_published": case["md5"],
                     "raw_h5_size": h5.stat().st_size,
                     "raw_metadata_json": str(metadata),
                     "raw_metadata_sha256": sha256(metadata),
                     "full_time_series": True,
                     "extrapolation_order": case.get("extrapolation_order", 3),
                     "final_mass_fraction": case["remnant_mass"],
                     "final_spin": case["remnant_spin"],
                     "relaxation_time": case["reference_time"],
                     "strain_time_convention": f"dimensionless geometric units (G=c=1), time in total initial mass M; future-null-infinity extrapolated strain N={case.get('extrapolation_order', 3)}",
                     "remnant_metadata_source": "SXS catalog simulations.json; values copied into this manifest"})
        item.update(extract_case(case, args.raw_dir, args.output_dir))
        item["id"] = f"{case['sxs_id']}/{case['resolution']}/N{item['extrapolation_order']}"
        item["path"] = item["npz"]
        item["final_spin_magnitude"] = float(sum(v * v for v in item["final_spin"]) ** 0.5)
        item["metadata"] = {key: item[key] for key in (
            "sxs_id", "q", "split", "resolution", "extrapolation_order",
            "initial_mass_ratio", "initial_spin1", "initial_spin2",
            "reference_spin1", "reference_spin2", "reference_time",
            "reference_mass1", "reference_mass2", "remnant_mass",
            "remnant_spin", "final_mass_fraction", "final_spin",
            "final_spin_magnitude", "relaxation_time", "peak_time",
            "source_frame_type", "extraction_frame_type", "frame_transform",
        )}
        records.append(item)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    sxs_version = records[0]["sxs_package_version"] if records else "unknown"
    args.manifest.write_text(json.dumps({
        "schema": "torchwave.nr33-pilot.v1",
        "catalog": {"version": "v3.0.0", "doi": CATALOG_DOI, "url": CATALOG_URL},
        "software": {"sxs": sxs_version, "sxs_license": "MIT",
                      "sxs_license_url": "https://github.com/sxs-collaboration/sxs/blob/main/LICENSE"},
        "split_policy": "Frozen before numerical fitting: training={SXS:BBH:0008,0169,0182,0056,0063}; holdout={SXS:BBH:0030,0166}; N=2 records are validation_only.",
        "cases": records,
    }, indent=2, sort_keys=True) + "\n")
    print(f"prepared {len(records)} cases; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
