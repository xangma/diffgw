"""Equation and author-data checks with no runtime waveform-oracle dependency.

Khan et al., PRD 93, 044007 (arXiv:1508.07253v2), Eqs. (19), (20), (21).
QNM rows: Gregory B. Cook, Zenodo 2650358 v1, KerrQNM_00.h5,
https://zenodo.org/records/2650358/files/KerrQNM_00.h5?download=1
SHA256: d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923.
Dataset columns 0:3 are a, Re(M omega), Im(M omega); use the n00, l=2 rows.
See docs/reviews/2026-09-12-clean-room/baseline.md for extraction details.
"""

from decimal import Decimal, localcontext

import pytest
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.constants import MSUN_SEC
from torchwave.conversions import component_masses_to_eta, spin_combinations


def test_peak_frequency_matches_high_precision_published_equation():
    """Near zero and near one expose cancellation and artificial root cutoffs."""
    gamma2 = torch.tensor(
        [0.0, 1e-10, 1e-8, 0.3, 0.9, 0.99999999999, 0.999999999999],
        dtype=torch.float64,
    )
    with localcontext() as ctx:
        ctx.prec = 70
        expected = []
        for value in gamma2.tolist():
            g = Decimal.from_float(value)
            expected.append(0.0 if g == 0 else float(abs(((1 - g * g).sqrt() - 1) / g)))
    actual = IMRPhenomD.amplitude_peak_frequency(
        torch.zeros_like(gamma2), torch.ones_like(gamma2), gamma2, torch.ones_like(gamma2)
    )
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-15, atol=0)


def test_peak_frequency_batch_gradients_and_branch_point():
    args = tuple(torch.tensor(values, dtype=torch.float64, requires_grad=True) for values in (
        [0.08, 0.10, 0.12], [0.02, 0.01, 0.005], [0.0, 0.7, 1.02], [1.0, 1.2, 1.1]
    ))
    assert torch.autograd.gradcheck(IMRPhenomD.amplitude_peak_frequency, args)
    at_branch = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    result = IMRPhenomD.amplitude_peak_frequency(
        torch.tensor(0.1), torch.tensor(0.02), at_branch, torch.tensor(1.2)
    )
    assert torch.isfinite(torch.autograd.grad(result.sum(), at_branch)[0]).all()


def test_slope_peak_preserves_promoted_dtype_for_mixed_inputs():
    slope32 = IMRPhenomD.compute_slope_peak(
        torch.tensor([5.0], dtype=torch.float32),
        torch.tensor([1.4], dtype=torch.float32),
        0.35,
        torch.tensor([0], dtype=torch.int64),
    )
    slope64 = IMRPhenomD.compute_slope_peak(
        torch.tensor([5.0], dtype=torch.float32),
        torch.tensor([1.4], dtype=torch.float64),
        0.35,
        torch.tensor([0], dtype=torch.int64),
    )
    integer = IMRPhenomD.compute_slope_peak(
        torch.tensor([5]), torch.tensor([1]), torch.tensor([0]), torch.tensor([0])
    )
    assert slope32.dtype == torch.float32
    assert slope64.dtype == torch.float64
    assert integer.dtype == torch.float64
    assert torch.isfinite(torch.cat((slope32.double(), slope64, integer))).all()


@pytest.mark.parametrize("chi", [0.5, 0.99])
def test_scaled_amplitude_is_c1_at_merger_join(chi):
    """Eq. (21) requires the derivative of Eq. (19), including its no-root join."""
    parameters = tuple(torch.tensor(x, dtype=torch.float64) for x in (60.0, 15.0, chi, chi))
    m1, m2, _, _ = parameters
    eta = component_masses_to_eta(m1, m2)
    chi_pn = spin_combinations(*parameters)[2]
    gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
    gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
    _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(*parameters)
    peak = IMRPhenomD.amplitude_peak_frequency(f_rd, f_damp, gamma2, gamma3).detach()
    mf = (peak + torch.tensor([-1e-10, 0.0, 1e-10], dtype=torch.float64)).requires_grad_()
    f = mf / ((m1 + m2) * MSUN_SEC)
    hp, _ = IMRPhenomD()(f, *parameters, window_type=None)
    amplitude = hp.abs() * f ** (7.0 / 6.0)
    normalized = amplitude / amplitude[1].detach()
    slopes = torch.autograd.grad(normalized.sum(), mf)[0]
    torch.testing.assert_close(normalized, torch.ones_like(normalized), rtol=0, atol=1e-8)
    torch.testing.assert_close(slopes[0], slopes[2], rtol=0, atol=1e-5)
    if chi == 0.99:
        assert gamma2 > 1.0
        assert slopes[2].abs() > 0.1  # This join is not a stationary point.


def test_qnm_against_independently_published_cook_nodes():
    # n00/m+02/{2,2,0}: rows 0, 200, 500, 900, 1013.
    # n00/m-02/{2,-2,0}: rows 200, 500, 900, 1011, with a -> -a.
    author_rows = torch.tensor([
        [0.0, 0.3736716844180418, 0.0889623156889357],
        [0.2, 0.40214532424765553, 0.08831116620760301],
        [0.5, 0.4641230259759384, 0.08563883498806402],
        [0.9, 0.6716142721321627, 0.06486923587579592],
        [0.999875, 0.9840054480904776, 0.0038694275183756115],
        [-0.2, 0.35105255048081213, 0.08918254197837706],
        [-0.5, 0.32430731434882343, 0.08903154514941844],
        [-0.9, 0.29724421359058384, 0.0882806593150116],
        [-0.999875, 0.2915603588595766, 0.08802616905628735],
    ], dtype=torch.float64)
    actual = torch.stack(evaluate_kerr_qnm_220(author_rows[:, 0]), dim=-1)
    torch.testing.assert_close(actual, author_rows[:, 1:], rtol=0, atol=3e-12)


def test_qnm_batch_gradients():
    spins = torch.tensor([[-0.9, -0.2], [0.0, 0.9]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate_kerr_qnm_220, (spins,))


def test_default_merger_amplitude_retains_published_analytic_tail():
    parameters = tuple(torch.tensor(value, dtype=torch.float64) for value in (40.0, 30.0, 0.2, -0.1))
    m1, m2, _, _ = parameters
    eta = component_masses_to_eta(m1, m2)
    chi_pn = spin_combinations(*parameters)[2]
    gamma2 = IMRPhenomD.eval_table_v("gamma2", eta, chi_pn)
    gamma3 = IMRPhenomD.eval_table_v("gamma3", eta, chi_pn)
    _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(*parameters)
    mf = torch.tensor([0.15, 0.175, 0.199, 0.201], dtype=torch.float64)
    frequencies = mf / ((m1 + m2) * MSUN_SEC)
    hp, _ = IMRPhenomD()(frequencies, *parameters)
    assert torch.all(hp[:3].abs() > 0)
    assert hp[3] == 0  # Preserve the inherited Mf=0.2 implementation cutoff.

    # Khan Eq. (19), including A0 proportional to f^(-7/6): normalize to
    # remove distance and the common gamma1*gamma3*f_damp prefactor.
    offset = mf[:3] - f_rd
    width = gamma3 * f_damp
    shape = mf[:3].pow(-7.0 / 6.0) * torch.exp(-gamma2 * offset / width) / (offset.square() + width.square())
    torch.testing.assert_close(hp[:3].abs() / hp[0].abs(), shape / shape[0], rtol=3e-14, atol=0)

    assert mf[2] > f_rd + 5.0 * f_damp
    tapered, _ = IMRPhenomD()(frequencies, *parameters, window_type="planck")
    assert tapered[2] == 0  # The optional taper remains available.
