"""Pinned public-binary component check; run explicitly for qualification.

The paper equations, not this binary, define the scalar remnant candidates.
The binary discriminates their historical selection and checks the independently
generated native QNM solutions. It is never used by waveform evaluation.
"""

import argparse
import json
from pathlib import Path

import jax
import lal
import lalsimulation as lalsim
import torch

from dev.native_rom.qualify_generator import lal_binary_identity
from dev.native_rom.reference.remnant import final_mass_fraction, final_spin_projection
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


jax.config.update("jax_enable_x64", True)

REMNANT_ROWS = (
    (30., 20., 0., 0.),
    (30., 20., .5, -.2),
    (20., 10., .9, .9),
    (45., 5., -.8, .3),
)
QNM_SPINS = (-.9, -.5, 0., .8, .80005, .98, .98995, .999, .9996)
SPIN_ORDERS = ((1, 2), (3, 3), (3, 4))
REMNANT_TOL = 2e-12
QNM_TOL = 6e-7


def run(expected_lal, expected_lalsimulation):
    identity = lal_binary_identity(expected_lal, expected_lalsimulation)
    remnant = []
    for m1, m2, s1, s2 in REMNANT_ROWS:
        status, mass, spin = lalsim.SimIMREOBFinalMassSpin(
            m1*lal.MSUN_SI, m2*lal.MSUN_SI,
            [0., 0., s1], [0., 0., s2], lalsim.SEOBNRv4,
        )
        if status != 0:
            raise RuntimeError(f"LAL remnant status {status}")
        candidates = {f"{i},{j}": float(final_spin_projection(
            m1, m2, s1, s2, fit_order=(i, j))) for i, j in SPIN_ORDERS}
        scalar_mass = float(final_mass_fraction(m1, m2, s1, s2))
        remnant.append(dict(input=[m1, m2, s1, s2], binary_mass=mass,
                            scalar_mass=scalar_mass, binary_spin=spin,
                            scalar_spin_candidates=candidates,
                            mass_abs_error=abs(mass-scalar_mass),
                            selected_spin_abs_error=abs(spin-candidates["3,4"])))

    qnm = []
    for spin in QNM_SPINS:
        vector = lal.CreateCOMPLEX16Vector(1)
        status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
            vector, 1., spin, 2, 2, 1,
        )
        if status != 0:
            raise RuntimeError(f"LAL QNM status {status} at spin {spin}")
        binary = vector.data[0]*lal.MTSUN_SI
        torch_values = signed_qnm_220(torch.tensor(spin, dtype=torch.float64))
        jax_values = evaluate_kerr_qnm_220(spin)
        torch_pair = [float(value) for value in torch_values]
        jax_pair = [float(value) for value in jax_values]
        binary_pair = [float(binary.real), float(binary.imag)]
        qnm.append(dict(final_spin=spin, binary_dimensionless=binary_pair,
                        torch_dimensionless=torch_pair, jax_dimensionless=jax_pair,
                        torch_abs_error=[abs(a-b) for a, b in zip(torch_pair, binary_pair)],
                        jax_abs_error=[abs(a-b) for a, b in zip(jax_pair, binary_pair)]))

    selected = all(row["mass_abs_error"] < REMNANT_TOL and
                   row["selected_spin_abs_error"] < REMNANT_TOL for row in remnant)
    discriminated = all(
        abs(row["binary_spin"]-row["scalar_spin_candidates"][name]) > 1e-5
        for row in remnant[1:] for name in ("1,2", "3,3")
    )
    qnm_near = all(max(row["torch_abs_error"]+row["jax_abs_error"]) < QNM_TOL
                   for row in qnm)
    return dict(binary=identity, model="SEOBNRv4", remnant_rows=remnant,
                qnm_rows=qnm, thresholds=dict(remnant_abs=REMNANT_TOL,
                                              qnm_dimensionless_abs=QNM_TOL),
                gates=dict(fit_3_4_matches=selected, alternatives_discriminated=discriminated,
                           native_qnm_near_binary=qnm_near))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lal-version", required=True)
    parser.add_argument("--lalsimulation-version", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.lal_version, args.lalsimulation_version)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["gates"], sort_keys=True))
    if not all(result["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
