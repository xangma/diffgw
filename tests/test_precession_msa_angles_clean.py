import torch

from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_initial_clean import spin_oscillation, jacobi_sn


def _args():
    return (torch.tensor([20., 30., 50., 100., 200.], dtype=torch.float64),
            35., 25., [.2, .1, .2], [-.1, .15, -.1])


def test_reference_is_anchored_and_intermediates_are_present():
    out = assemble_precession_msa_angles(*_args())
    for name in ("alpha", "gamma", "cos_beta", "psi", "roots_reference", "secular_phi", "oscillatory_phi"):
        assert name in out and torch.isfinite(out[name]).all()
    assert torch.allclose(out["alpha"][0], torch.zeros((), dtype=torch.float64), atol=2e-14)
    assert torch.allclose(out["gamma"][0], torch.zeros((), dtype=torch.float64), atol=2e-14)


def test_mass_swap_and_frequency_batch_are_invariant():
    f, m1, m2, s1, s2 = _args()
    a = assemble_precession_msa_angles(f, m1, m2, s1, s2)
    b = assemble_precession_msa_angles(f, m2, m1, s2, s1)
    for key in ("alpha", "gamma", "cos_beta"):
        torch.testing.assert_close(a[key], b[key], rtol=2e-12, atol=2e-13)
    assert a["alpha"].shape == f.shape


def test_intrinsic_gradient_is_finite():
    f, _, _, s1, s2 = _args()
    masses = torch.tensor([35., 25.], dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(s1 + s2, dtype=torch.float64, requires_grad=True)
    out = assemble_precession_msa_angles(f, masses[0], masses[1], spin[:3], spin[3:])
    grad = torch.autograd.grad(sum(out[k].sum() for k in ("alpha", "gamma", "cos_beta")), (masses, spin), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grad)


def test_pn_beta_is_explicit_diagnostic_branch():
    out = assemble_precession_msa_angles(*_args(), beta_pn=True)
    assert torch.isfinite(out["cos_beta"]).all()
    assert torch.isfinite(out["cos_beta_pn"]).all()


def test_nutation_uses_current_cubic_roots():
    out = assemble_precession_msa_angles(*_args())
    s3, sm, sp = out["roots"].unbind(-1)
    current = spin_oscillation(out["psi"], sp, sm, s3)
    r3, rm, rp = out["roots_reference"]
    fixed = spin_oscillation(out["psi"], rp, rm, r3)
    # The current-root Jacobi identity is the one used by the angle assembly.
    mod = (sp - sm) / (sp - s3)
    assert torch.allclose(current, sp + (sm - sp) * jacobi_sn(out["psi"], mod).square(), atol=1e-12)
    assert torch.max(torch.abs(current[1:] - fixed[1:])) > 1e-8
