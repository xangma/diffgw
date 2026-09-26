"""Strict full-waveform binary gate for the derived XP phi0 frame map."""

import pytest

pytest.importorskip("lal")
pytest.importorskip("lalsimulation")

from scripts.qualify_xp_xphm_boundaries import GATE, mapped_phi0_oracle_case


@pytest.mark.parametrize("convention", (0, 1, 6))
def test_generic_nonzero_phi0_public_binary_full_band(convention):
    params = dict(
        mass1=35.0, mass2=25.0,
        spin1x=0.2, spin1y=0.1, spin1z=0.3,
        spin2x=-0.1, spin2y=0.05, spin2z=-0.2,
        distance=150.0, inclination=0.7, phic=0.3, tc=0.0,
        phi0=0.4, f_ref=20.0, prec_version=223,
        convention=convention, final_spin_mod=3,
    )
    result = mapped_phi0_oracle_case(params)
    assert result["gate"] == GATE
    assert result["status"] == "PASSED", result
    assert result["plus"]["pass"] and result["cross"]["pass"]
