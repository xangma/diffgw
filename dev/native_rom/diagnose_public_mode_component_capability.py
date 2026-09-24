"""Audit the pinned public wheel for an explicit-state pre-NQC mode oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import lal
import lalsimulation


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/public_mode_component_capability_plan.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-mode-component-capability-plan.v1"
            or plan["release_qualified"] is not False):
        parser.error("frozen capability plan differs")
    source = plan["pinned_source"]
    source_repo = args.source_repo.resolve()
    tag_object = source["tag_object"]
    if (checked(["git", "-C", str(source_repo), "rev-parse", source["tag"]]).strip()
            != tag_object
            or checked(["git", "-C", str(source_repo), "cat-file", "-t",
                        tag_object]).strip() != "tag"
            or checked(["git", "-C", str(source_repo), "rev-parse",
                        f"{tag_object}^{{commit}}" ]).strip() != source["peeled_commit"]):
        raise ValueError("pinned source tag identity changed")
    source_text = {}
    for path, expected in source["files_sha256"].items():
        content = subprocess.run(
            ["git", "-C", str(source_repo), "show", f"{tag_object}:{path}"],
            check=True, capture_output=True).stdout
        if digest(content) != expected:
            raise ValueError(f"pinned source hash differs: {path}")
        source_text[path] = content.decode()
    wheel = plan["pinned_wheel"]
    if (lal.__version__ != wheel["lal_version"]
            or lalsimulation.__version__ != wheel["lalsimulation_version"]
            or importlib.metadata.version("lalsuite") != wheel[
                "lalsuite_distribution"]):
        raise ValueError("pinned public wheel version differs")
    library = (Path(lal.__file__).resolve().parent /
               ".dylibs/liblalsimulation.37.dylib")
    extensions = list(Path(lalsimulation.__file__).resolve().parent.glob(
        "_lalsimulation*.so"))
    if len(extensions) != 1:
        raise ValueError("pinned Python extension is ambiguous")
    extension = extensions[0]
    if (digest(library.read_bytes()) != wheel["liblalsimulation_sha256"]
            or digest(extension.read_bytes()) != wheel["python_extension_sha256"]):
        raise ValueError("pinned public wheel binary differs")
    prior_path = ROOT / plan["prior_diagnostic"]
    if digest(prior_path.read_bytes()) != plan["prior_diagnostic_sha256"]:
        raise ValueError("frozen frequency-trace result differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete" or prior["release_qualified"]:
        raise ValueError("frozen frequency-trace status differs")
    symbols = sorted(line.rsplit(" ", 1)[-1].removeprefix("_") for line in
                     checked(["nm", "-gU", str(library)]).splitlines()
                     if line.strip())
    python_names = set(dir(lalsimulation))
    factorized = source_text[
        "lalsimulation/lib/LALSimIMRSpinEOBFactorizedWaveform.c"]
    factorized_pa = source_text[
        "lalsimulation/lib/LALSimIMRSpinEOBFactorizedWaveform_PA.c"]
    header = source_text[
        "lalsimulation/lib/LALSimIMRSpinEOBFactorizedWaveform.h"]
    candidates = []
    for name in plan["source_component_names"]:
        code = factorized_pa if name.endswith("_PA") else factorized
        static_definition = bool(re.search(r"\bstatic\s+INT4\s+" + re.escape(name)
                                           + r"\s*\(", code))
        if not static_definition:
            raise ValueError(f"pinned source static definition changed: {name}")
        candidates.append(dict(
            name=name, source_static_definition=True,
            header_declaration=name in header, exported_from_pinned_wheel=name in symbols,
            python_binding=name.removeprefix("XLAL") in python_names))
    if any(item["exported_from_pinned_wheel"] or item["python_binding"]
           for item in candidates):
        raise ValueError("unexpected direct mode component exposure")
    waveform_api = plan["available_waveform_api"]
    if waveform_api not in python_names or "XLAL"+waveform_api not in symbols:
        raise ValueError("pinned full-waveform API is unexpectedly absent")
    waveform_signature = (getattr(lalsimulation, waveform_api).__doc__ or "").splitlines()[0]
    if (not all(token in waveform_signature for token in
                ("deltaT", "m1SI", "m2SI", "fMin", "nqcFlag"))
            or "REAL8Vector values" in waveform_signature):
        raise ValueError("pinned full-waveform API signature changed")
    result = dict(
        schema="diffgw.seobnrv4.public-mode-component-capability.v1",
        status="capability_limited", release_qualified=False,
        same_state_pre_nqc_mode_parity_performed=False,
        direct_public_mode_component_api_found=False,
        local_compiled_generator_used=False,
        plan_sha256=digest(PLAN.read_bytes()), script_sha256=digest(Path(__file__).read_bytes()),
        prior_diagnostic_sha256=digest(prior_path.read_bytes()),
        prior_raw_mode_gap=dict(
            max_abs_amplitude_relative=prior["summary"][
                "max_abs_pre_nqc_raw_amplitude_relative"],
            max_abs_frequency_relative=prior["summary"][
                "max_abs_pre_nqc_raw_gravitational_wave_frequency_relative"]),
        source=dict(tag=source["tag"], tag_object=tag_object,
                    peeled_commit=source["peeled_commit"],
                    files_sha256=source["files_sha256"], repo=str(source_repo)),
        wheel=dict(**wheel, library=str(library), python_extension=str(extension)),
        source_component_candidates=candidates,
        public_waveform_api=dict(name=waveform_api, signature=waveform_signature,
                                 python_binding=True,
                                 exported_from_pinned_wheel=True,
                                 accepts_explicit_phase_space_state=False),
        relevant_exported_symbols=[name for name in symbols if
                                   ("Factorized" in name or
                                    "AmplitudeResidual" in name or
                                    name in ("XLALSimInspiralEOBPAFluxWrapper",
                                             "XLALSimIMRSpinAlignedEOBModes"))],
        finding=("The pinned source defines the tested factorized mode and amplitude "
                 "routines as static functions, and the pinned wheel exports none "
                 "of them. The public Python mode API generates a full trajectory "
                 "from binary inputs and does not accept an explicit phase-space "
                 "state. Its NQC controls do not resolve the existing one-sample "
                 "mode-to-dynamics mapping uncertainty."),
        limit=("No direct same-state raw (2,2) mode comparison is justified by "
               "this wheel. The previous 87 ppm amplitude and 672 ppm frequency "
               "gaps remain reconstruction-based. A separately built adapter "
               "would require numerical reproduction of the pinned wheel before "
               "supporting a public-binary parity claim."),
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
