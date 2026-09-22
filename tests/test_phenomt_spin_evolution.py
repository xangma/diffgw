import numpy as np
import pytest
import torch

from torchwave.approximants._phenomt_spin_evolution import spin_derivatives as torch_spin_derivatives


def _state(dtype=torch.float64):
    return (torch.tensor(.22,dtype=dtype), torch.tensor([.15,-.27,.95],dtype=dtype),
            torch.tensor([.21,.08,.13],dtype=dtype), torch.tensor([-.12,.17,.09],dtype=dtype),
            torch.tensor(.62,dtype=dtype), torch.tensor(.38,dtype=dtype))


def test_n4lo_spin_and_orbital_derivatives_are_tangent():
    v,l,s1,s2,m1,m2=_state(); dl,d1,d2=torch_spin_derivatives(v,l,s1,s2,m1,m2)
    assert torch.allclose((d1*s1).sum(),torch.zeros(() ,dtype=torch.float64),atol=2e-18)
    assert torch.allclose((d2*s2).sum(),torch.zeros(() ,dtype=torch.float64),atol=2e-18)
    assert torch.allclose((dl*l).sum(),torch.zeros(() ,dtype=torch.float64),atol=2e-18)


def test_mass_exchange_exchanges_spin_derivatives():
    v,l,s1,s2,m1,m2=_state(); a=torch_spin_derivatives(v,l,s1,s2,m1,m2)
    b=torch_spin_derivatives(v,l,s2,s1,m2,m1)
    assert torch.allclose(a[0],b[0],rtol=1e-13,atol=1e-15)
    assert torch.allclose(a[1],b[2],rtol=1e-13,atol=1e-15)
    assert torch.allclose(a[2],b[1],rtol=1e-13,atol=1e-15)


def test_torch_jax_parity():
    pytest.importorskip("jax")
    from jaxwave.approximants._phenomt_spin_evolution import spin_derivatives as jax_spin_derivatives
    v,l,s1,s2,m1,m2=_state(); got=torch_spin_derivatives(v,l,s1,s2,m1,m2)
    ref=jax_spin_derivatives(float(v),np.asarray(l),np.asarray(s1),np.asarray(s2),float(m1),float(m2))
    for x,y in zip(got,ref):
        np.testing.assert_allclose(x.detach().numpy(),np.asarray(y),rtol=2e-12,atol=2e-14)


def test_torch_jax_parity_scalar_v_batched_masses():
    pytest.importorskip("jax")
    from jaxwave.approximants._phenomt_spin_evolution import spin_derivatives as jax_spin_derivatives

    v=0.22
    l=torch.tensor([[.15,-.27,.95],[.21,.08,.13]],dtype=torch.float64)
    s1=torch.tensor([[.21,.08,.13],[-.12,.17,.09]],dtype=torch.float64)
    s2=torch.tensor([[-.12,.17,.09],[.11,-.04,.19]],dtype=torch.float64)
    m1=torch.tensor([.62,.57],dtype=torch.float64); m2=1.0-m1
    got=torch_spin_derivatives(v,l,s1,s2,m1,m2)
    ref=jax_spin_derivatives(v,np.asarray(l),np.asarray(s1),np.asarray(s2),np.asarray(m1),np.asarray(m2))
    for x,y in zip(got,ref):
        np.testing.assert_allclose(x.detach().numpy(),np.asarray(y),rtol=2e-12,atol=2e-14)


def test_n3lo_switch_removes_only_v8_term():
    v, l, s1, s2, m1, m2 = _state()
    full = torch_spin_derivatives(v, l, s1, s2, m1, m2)
    without = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                      include_n3lo_ss=False)
    # The published N3LO quadratic-spin contribution scales exactly as v**8;
    # lower and higher spin orders therefore cancel from this difference.
    v2 = v * 0.8
    full2 = torch_spin_derivatives(v2, l, s1, s2, m1, m2)
    without2 = torch_spin_derivatives(v2, l, s1, s2, m1, m2,
                                      include_n3lo_ss=False)
    scale = 0.8**8
    # The orbital derivative also contains the v-dependent PN angular-momentum
    # denominator, so apply the exact v**8 check to the spin rates themselves.
    for a, b, c, d in zip(full[1:], without[1:], full2[1:], without2[1:]):
        torch.testing.assert_close((c - d), scale * (a - b), rtol=2e-14, atol=2e-16)


def test_quadrupole_coupling_controls_self_terms():
    v, l, s1, s2, m1, m2 = _state()
    bh = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                include_n3lo_ss=False)
    k0 = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                quadrupole_kappa1=0.0,
                                quadrupole_kappa2=0.0,
                                include_n3lo_ss=False)
    # Companion terms remain present at kappa=0, while the self terms change.
    assert not torch.allclose(bh[1], k0[1], rtol=1e-8, atol=1e-12)
    assert not torch.allclose(bh[2], k0[2], rtol=1e-8, atol=1e-12)
    for x in k0:
        assert torch.isfinite(x).all()


def test_published_v8_and_quadrupole_coefficients():
    v, l, s1, s2, m1, m2 = _state()
    full = torch_spin_derivatives(v, l, s1, s2, m1, m2)
    no_v8 = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                   include_n3lo_ss=False)
    l1 = (l * s1).sum()
    l2 = (l * s2).sum()
    a1 = -13 / 4 + 9 * m1 / 4 + m1**2
    a2 = -13 / 4 + 9 * m2 / 4 + m2**2
    b1 = m1 / 2 + m1**2 / 2
    b2 = m2 / 2 + m2**2 / 2
    d1 = -3 / 2 - 3 * m1 / 2 - m1**2 / 2
    d2 = -3 / 2 - 3 * m2 / 2 - m2**2 / 2
    expected1 = v**8 * torch.linalg.cross(a1 * l1 * l + b1 * s2 + d1 * l2 * l, s1)
    expected2 = v**8 * torch.linalg.cross(a2 * l2 * l + b2 * s1 + d2 * l1 * l, s2)
    torch.testing.assert_close(full[1] - no_v8[1], expected1, rtol=2e-14, atol=2e-16)
    torch.testing.assert_close(full[2] - no_v8[2], expected2, rtol=2e-14, atol=2e-16)

    k0 = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                quadrupole_kappa1=0.0,
                                quadrupole_kappa2=0.0,
                                include_n3lo_ss=False)
    full_no_v8 = no_v8
    expected_self1 = (1.5 * (1 - 1 / m1) * l1 * torch.linalg.cross(l, s1) * v**6)
    expected_self2 = (1.5 * (1 - 1 / m2) * l2 * torch.linalg.cross(l, s2) * v**6)
    torch.testing.assert_close(full_no_v8[1] - k0[1], expected_self1, rtol=2e-14, atol=2e-16)
    torch.testing.assert_close(full_no_v8[2] - k0[2], expected_self2, rtol=2e-14, atol=2e-16)


def test_torch_jax_parity_for_v8_switch_and_batched_kappa():
    jax = pytest.importorskip("jax")
    from jaxwave.approximants._phenomt_spin_evolution import spin_derivatives as jax_spin_derivatives

    v = 0.22
    l = torch.tensor([[.15, -.27, .95], [.21, .08, .13]], dtype=torch.float64)
    s1 = torch.tensor([[.21, .08, .13], [-.12, .17, .09]], dtype=torch.float64)
    s2 = torch.tensor([[-.12, .17, .09], [.11, -.04, .19]], dtype=torch.float64)
    m1 = torch.tensor([.62, .57], dtype=torch.float64)
    m2 = 1.0 - m1
    k1 = torch.tensor([0.0, 1.3], dtype=torch.float64)
    k2 = torch.tensor([1.2, 0.4], dtype=torch.float64)
    for include in (True, False):
        got = torch_spin_derivatives(v, l, s1, s2, m1, m2,
                                      quadrupole_kappa1=k1,
                                      quadrupole_kappa2=k2,
                                      include_n3lo_ss=include)
        ref = jax_spin_derivatives(v, np.asarray(l), np.asarray(s1), np.asarray(s2),
                                    np.asarray(m1), np.asarray(m2),
                                    quadrupole_kappa1=np.asarray(k1),
                                    quadrupole_kappa2=np.asarray(k2),
                                    include_n3lo_ss=include)
        for x, y in zip(got, ref):
            np.testing.assert_allclose(x.detach().numpy(), np.asarray(y), rtol=2e-12, atol=2e-14)
        compiled = jax.jit(lambda ll, aa, bb, mm, nn: jax_spin_derivatives(
            v, ll, aa, bb, mm, nn, quadrupole_kappa1=np.asarray(k1),
            quadrupole_kappa2=np.asarray(k2), include_n3lo_ss=include))
        compiled_ref = compiled(np.asarray(l), np.asarray(s1), np.asarray(s2),
                                np.asarray(m1), np.asarray(m2))
        for x, y in zip(ref, compiled_ref):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-12, atol=2e-14)


def test_single_spin_hand_reduced_coefficients():
    # For l=z, s2=0, and m1=m2=1/2, every nonzero s1 rate is proportional
    # to x=l cross s1.  This independently checks the reduced Eq. 21/24
    # coefficients for both the full and documented TP v8 selections.
    v = torch.tensor(0.21, dtype=torch.float64)
    l = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    s1 = torch.tensor([0.23, -0.11, 0.31], dtype=torch.float64)
    s2 = torch.zeros(3, dtype=torch.float64)
    x = torch.linalg.cross(l, s1)
    sz = s1[2]
    for kappa in (0.0, 1.3):
        for include in (True, False):
            got = torch_spin_derivatives(
                v, l, s1, s2, torch.tensor(.5, dtype=torch.float64),
                torch.tensor(.5, dtype=torch.float64),
                quadrupole_kappa1=kappa, quadrupole_kappa2=kappa,
                include_n3lo_ss=include,
            )
            coefficient = (torch.tensor(7 / 8, dtype=torch.float64) * v**5
                           - 1.5 * kappa * sz * v**6
                           + torch.tensor(335 / 384, dtype=torch.float64) * v**7
                           + (torch.tensor((27 - 57 * kappa) / 16, dtype=torch.float64)
                              * sz * v**8 if include else 0.0)
                           + torch.tensor(701 / 1024, dtype=torch.float64) * v**9)
            torch.testing.assert_close(got[1], coefficient * x, rtol=2e-14, atol=2e-16)
            torch.testing.assert_close(got[2], torch.zeros_like(s2), rtol=0, atol=2e-16)
