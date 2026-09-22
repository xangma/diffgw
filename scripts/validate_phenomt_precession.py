"""Record direct PhenomTP/TPHM comparisons with the installed LAL binary.

Run from the repository root:
    python scripts/validate_phenomt_precession.py --output receipt.json

This diagnostic reads no LAL implementation source and applies no waveform
alignment or fitted correction. It requires the optional LALSuite package.
"""

import argparse
import datetime
import importlib
import json
from pathlib import Path
import platform
import sys

import lal
import lalsimulation as ls
import numpy as np

CASES = {
    "baseline": (40.0, 15.0, (0.2, 0.1, 0.3), (0.0, 0.1, -0.2), 0.8, 0.3, 20.0),
    "single_spin": (35.0, 20.0, (0.3, 0.0, 0.4), (0.0, 0.0, 0.0), 1.1, 0.2, 30.0),
    "equal_mass": (30.0, 30.0, (0.2, -0.1, 0.4), (-0.12, 0.08, -0.3), 0.8, 0.3, 20.0),
    "high_mass_ratio": (60.0, 6.0, (0.2, 0.1, 0.8), (0.1, 0.0, 0.1), 1.2, 0.1, 20.0),
}
PRESCRIPTIONS = {"msa": (22310, 3), "numerical": (300, 4)}


def parameters(case):
    m1, m2, s1, s2, inc, phase, fref = CASES[case]
    return dict(mass1=m1, mass2=m2, spin1=list(s1), spin2=list(s2),
                distance_mpc=100.0, inclination=inc, phic=phase,
                f_lower_hz=20.0, f_ref_hz=fref)


def metrics(actual, reference):
    norm = np.linalg.norm(reference)
    overlap = abs(np.vdot(actual, reference)) / (np.linalg.norm(actual) * norm)
    return dict(relative_l2=float(np.linalg.norm(actual - reference) / norm),
                mismatch=float(1 - overlap))


def measure(name, method, backend, case, rate):
    version, spin_mode = PRESCRIPTIONS[method]
    m1, m2, s1, s2, inc, phase, fref = CASES[case]
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, version)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, spin_mode)
    dt = 1 / rate
    hp, hc = getattr(ls, "Sim" + name)(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
        100e6 * lal.PC_SI, inc, dt, 20.0, fref, phase, options,
    )
    times = float(hp.epoch) + dt * np.arange(hp.data.length)
    if backend == "torch":
        import torch
        grid = torch.from_numpy(times)
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        grid = jnp.asarray(times)
    frontend = importlib.import_module("torchwave" if backend == "torch" else "jaxwave")
    a, b = getattr(frontend, name)().time_domain(
        grid, m1, m2, chi1=s1[2], chi2=s2[2],
        spin1x=s1[0], spin1y=s1[1], spin2x=s2[0], spin2y=s2[1],
        distance=100.0, inclination=inc, phic=phase, f_ref=fref,
        precession_method=method, precession_version=version,
    )
    strain = a - 1j * b
    actual = strain.detach().numpy() if backend == "torch" else np.asarray(strain)
    reference = hp.data.data - 1j * hc.data.data
    if not np.isfinite(actual).all():
        raise RuntimeError(f"Nonfinite output for {name}/{method}")
    return dict(
        approximant=name, method=method, backend=backend, case=case,
        sample_rate_hz=rate, precession_version=version,
        final_spin_mod=spin_mode, samples=len(times), epoch_seconds=float(hp.epoch),
        **metrics(actual, reference),
        inspiral=metrics(actual[times < 0], reference[times < 0]),
        ringdown=metrics(actual[times >= 0], reference[times >= 0]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("torch", "jax"), nargs="+", default=["torch", "jax"])
    parser.add_argument("--case", choices=tuple(CASES), nargs="+", default=["baseline"])
    parser.add_argument("--method", choices=tuple(PRESCRIPTIONS), nargs="+", default=list(PRESCRIPTIONS))
    parser.add_argument("--sample-rate", type=int, nargs="+", default=[2048])
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.sample_rate):
        parser.error("sample rates must be positive")
    versions = dict(python=platform.python_version(), numpy=np.__version__, lalsimulation=ls.__version__)
    for backend in args.backend:
        versions[backend] = importlib.import_module(backend).__version__
    results = []
    for case in args.case:
        for rate in args.sample_rate:
            for method in args.method:
                for name in ("IMRPhenomTP", "IMRPhenomTPHM"):
                    for backend in args.backend:
                        result = measure(name, method, backend, case, rate)
                        results.append(result)
                        print(f"{case}/{rate} Hz/{method}/{name}/{backend}: "
                              f"relative L2={result['relative_l2']:.8g}, "
                              f"mismatch={result['mismatch']:.8g}", file=sys.stderr, flush=True)
    reproduction = ("python scripts/validate_phenomt_precession.py --backend " + " ".join(args.backend)
                    + " --case " + " ".join(args.case) + " --method " + " ".join(args.method)
                    + " --sample-rate " + " ".join(map(str, args.sample_rate)) + " --output receipt.json")
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction=reproduction, versions=versions, dtype="float64", device="cpu",
        parameters={case: parameters(case) for case in args.case},
        metrics=dict(
            relative_l2="norm(actual-reference)/norm(reference); no alignment",
            mismatch="1-abs(vdot(actual,reference))/(norm(actual)*norm(reference)); "
                     "unweighted, constant-phase maximization only, no time shift",
        ),
        results=results,
    )
    encoded = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
