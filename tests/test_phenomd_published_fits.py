"""Independent contract tests for the arXiv Table-V polynomial transcription."""

import hashlib

import torch

from torchwave.phenomd_published_fits import PARAMETER_NAMES, TABLE_V_ROWS, eval_table_v_fit


def _paper_formula(row, eta, chi):
    c = [float(v) for v in row]
    x = chi - 1.0
    return c[0] + c[1] * eta + x * (c[2] + c[3] * eta + c[4] * eta**2) + x**2 * (c[5] + c[6] * eta + c[7] * eta**2) + x**3 * (c[8] + c[9] * eta + c[10] * eta**2)


def test_table_v_has_paper_order_and_precision():
    assert len(PARAMETER_NAMES) == len(TABLE_V_ROWS) == 19
    assert PARAMETER_NAMES[:7] == ("rho_1", "rho_2", "rho_3", "v_2", "gamma_1", "gamma_2", "gamma_3")
    assert PARAMETER_NAMES[-5:] == ("alpha_1", "alpha_2", "alpha_3", "alpha_4", "alpha_5")
    assert TABLE_V_ROWS[0][4] == "-1.21626e6"
    assert TABLE_V_ROWS[13][6] == "-5.578e-5"
    assert hashlib.sha256("\n".join(",".join(r) for r in TABLE_V_ROWS).encode()).hexdigest() == "42f569284dccfba622fc331df6d81fdf5a23d4fd27beec6aa19071530041c169"


def test_eq31_origin_and_eta_dependence():
    eta = torch.tensor([0.05, 0.16, 0.25], dtype=torch.float64)
    chi = torch.ones(3, dtype=torch.float64)
    for name, row in zip(PARAMETER_NAMES, TABLE_V_ROWS):
        got = eval_table_v_fit(name, eta, chi)
        assert torch.allclose(got, torch.tensor([float(row[0]) + float(row[1]) * e for e in eta], dtype=torch.float64))
    # A coefficient with a quadratic eta term exercises the complete basis.
    eta0, chi0 = torch.tensor(0.17, dtype=torch.float64), torch.tensor(0.63, dtype=torch.float64)
    assert torch.allclose(eval_table_v_fit("rho_1", eta0, chi0), _paper_formula(TABLE_V_ROWS[0], eta0, chi0))


def test_broadcast_dtype_device_and_gradients():
    eta = torch.tensor([[0.1], [0.2]], dtype=torch.float64, requires_grad=True)
    chi = torch.tensor([0.2, 0.7, 1.1], dtype=torch.float64, requires_grad=True)
    out = eval_table_v_fit("alpha_5", eta, chi)
    assert out.shape == (2, 3) and out.dtype == torch.float64 and out.device == eta.device
    out.sum().backward()
    assert torch.isfinite(eta.grad).all() and torch.isfinite(chi.grad).all()
