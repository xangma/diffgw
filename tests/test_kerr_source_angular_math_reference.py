"""Selected source-spectral Kerr 220 direct-call composition gate."""

import json

from scripts.audit_kerr220_public_angular_gate import main


def test_source_spectral_angular_and_carrier_against_independent_scalar(capsys):
    main("source_spectral")
    result = json.loads(capsys.readouterr().out)
    assert result["angular_convention"] == "source_spectral"
    assert len(result["rows"]) == 4
    assert {(row["backend"], row["spin"]) for row in result["rows"]} == {
        (backend, spin) for backend in ("torch", "jax") for spin in (0.35, 0.7)
    }
    for row in result["rows"]:
        assert row["angular_ell10_vs_ell12_absolute_error"] < 1e-11, row
        assert row["angular_plus_absolute_error"] < 3e-10, row
        assert row["angular_mirror_absolute_error"] < 3e-10, row
        assert row["td_polarization_relative_l2"] < 3e-9, row
        assert row["fd_polarization_relative_l2"] < 3e-9, row
