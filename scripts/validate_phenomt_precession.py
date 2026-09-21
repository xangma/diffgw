"""Record direct PhenomTP/TPHM comparisons with the installed LAL binary.

Run from the repository root:
    python scripts/validate_phenomt_precession.py --output receipt.json

This diagnostic reads no LAL implementation source and applies no waveform
alignment or fitted correction. It requires the optional LALSuite package.
"""

import argparse
import datetime
import json
from pathlib import Path
import platform

import lal
import lalsimulation as ls
import numpy as np
import torch
import torchwave


def measure(name, method, version, spin_mode):
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, version)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, spin_mode)
    dt = 1 / 2048
    hp, hc = getattr(ls, "Sim" + name)(
        40 * lal.MSUN_SI, 15 * lal.MSUN_SI,
        0.2, 0.1, 0.3, 0.0, 0.1, -0.2,
        100e6 * lal.PC_SI, 0.8, dt, 20.0, 20.0, 0.3, options,
    )
    times = torch.from_numpy(float(hp.epoch) + dt * np.arange(hp.data.length))
    a, b = getattr(torchwave, name)().time_domain(
        times, 40.0, 15.0, chi1=0.3, chi2=-0.2,
        spin1x=0.2, spin1y=0.1, spin2y=0.1,
        distance=100.0, inclination=0.8, phic=0.3, f_ref=20.0,
        precession_method=method, precession_version=version,
    )
    actual = (a - 1j * b).detach().numpy()
    reference = hp.data.data - 1j * hc.data.data
    if not np.isfinite(actual).all():
        raise RuntimeError(f"Nonfinite output for {name}/{method}")
    overlap = abs(np.vdot(actual, reference)) / (
        np.linalg.norm(actual) * np.linalg.norm(reference)
    )
    return dict(
        approximant=name, method=method, precession_version=version,
        final_spin_mod=spin_mode, samples=len(times), epoch_seconds=float(hp.epoch),
        relative_l2=float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
        mismatch=float(1 - overlap),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reproduction="python scripts/validate_phenomt_precession.py --output receipt.json",
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, lalsimulation=ls.__version__),
        parameters=dict(mass1=40.0, mass2=15.0, spin1=[0.2, 0.1, 0.3],
                        spin2=[0.0, 0.1, -0.2], distance_mpc=100.0,
                        inclination=0.8, phic=0.3, f_lower_hz=20.0,
                        f_ref_hz=20.0, sample_rate_hz=2048),
        metrics=dict(
            relative_l2="norm(actual-reference)/norm(reference); no alignment",
            mismatch="1-abs(vdot(actual,reference))/(norm(actual)*norm(reference)); "
                     "unweighted, constant-phase maximization only, no time shift",
        ),
        results=[measure(name, method, version, spin_mode)
                 for method, version, spin_mode in (("msa", 22310, 3), ("numerical", 300, 4))
                 for name in ("IMRPhenomTP", "IMRPhenomTPHM")],
    )
    encoded = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
