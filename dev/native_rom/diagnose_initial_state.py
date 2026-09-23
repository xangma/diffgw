"""Compare the provisional Torch v4 initial state with the scalar paper check.

The native 35-mode torque is passed as an input to the scalar calculation;
this diagnostic tests the circular solution and adiabatic conversion, not the
torque's provenance or historical LAL initial data.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp
import torch

from .reference import SPEC_VERSION
from .reference.hamiltonian import aligned_effective_hamiltonian
from .reference.initial_conditions import leading_adiabatic_state
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROWS = ((30., 20., 0., 0., .04), (30., 20., .2, -.1, .04))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    output = []
    for m1, m2, s1, s2, omega in ROWS:
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64)

        n = m1*m2/(m1+m2)**2
        ham = SEOBNRv4Hamiltonian(tensor(n), tensor(s1), tensor(s2),
                                  delta=tensor((m1-m2)/(m1+m2)))
        flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8",
                                     retain_pan_cubic_spin=True)
        r, pr, j = find_quasicircular_initial_conditions(ham, flux, tensor(omega))
        _, torque = flux.radiation_reaction_forces(tensor(omega), tensor(0.), j,
                                                   r=r, hamiltonian=ham)
        reference = leading_adiabatic_state(m1, m2, s1, s2, omega, float(torque))
        actual = (float(r), float(pr), float(j))
        with mp.workdps(50):
            def h(momentum):
                return aligned_effective_hamiltonian(
                    m1, m2, s1, s2, reference[0], momentum, reference[2],
                    digits=mp.mp.dps)

            exact = mp.diff(h, 0, 2)
            step = mp.diff(h, mp.mpf("0.001"))/mp.mpf("0.001")
            step_relative = float(step/exact - 1)
        output.append({
            "row": dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2, omega_start=omega),
            "native_35_mode_torque": float(torque),
            "native_initial_state": dict(zip(("r", "pr_star", "pphi"), actual)),
            "scalar_initial_state": dict(zip(("r", "pr_star", "pphi"), map(float, reference))),
            "absolute_difference": dict(zip(("r", "pr_star", "pphi"),
                                            (abs(a-float(b)) for a, b in zip(actual, reference)))),
            "scalar_hamiltonian_radial_response_step_0p001_relative": step_relative,
        })
    result = {"schema": "diffgw.seobnrv4.initial-state-scalar-check.v1",
              "specification": SPEC_VERSION,
              "status": "conditional_component_check", "rows": output,
              "limitation": "Scalar check consumes the native torque; it does not validate 35-mode flux or LAL initial-data parity."}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
