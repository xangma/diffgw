import math
import pytest
import torch
import numpy as np

from torchwave.approximants.kerr_angular_table_convention import (
    _G10_INDICES,
    _G10_WEIGHTS,
    _K21_NODES,
    _K21_WEIGHTS,
    _raw,
)
from torchwave.approximants.kerr_angular_table_convention import spin_weighted_spheroidal_harmonic_table_compat

def test_spin_zero_spherical_shape_and_norm():
    th=torch.linspace(0.0, math.pi, 101, dtype=torch.float64)
    got=spin_weighted_spheroidal_harmonic_table_compat(0.0,th)
    expected=math.sqrt(5/(64*math.pi))*(1+torch.cos(th))**2
    assert torch.allclose(got.real,expected,atol=2e-10)
    assert torch.max(abs(got.imag))<1e-12

def test_batch_angles_phi_and_gradient():
    s=torch.tensor([0.,.3,.7],dtype=torch.float64,requires_grad=True)
    th=torch.tensor([.2,.8,1.3],dtype=torch.float64)
    z=spin_weighted_spheroidal_harmonic_table_compat(s,th,phi=.17)
    assert z.shape==(3,) and torch.isfinite(z.real).all()
    torch.autograd.grad(z.real.sum(),s)

def test_public_compatibility_domain():
    z=spin_weighted_spheroidal_harmonic_table_compat(torch.tensor([-.9996,.9996]),torch.tensor(.8))
    assert torch.isfinite(z.real).all()
    with pytest.raises(ValueError): spin_weighted_spheroidal_harmonic_table_compat(.9997,.8)

def test_public_angular_grid_regression_optional():
    pytest.importorskip("lal"); ls=pytest.importorskip("lalsimulation")
    mu=torch.cos(torch.tensor([.2,.8,1.3,2.5],dtype=torch.float64))
    spins=[0.,.1,.3,.5,.7,.8,.9,.92,.95,.97]
    for spin in spins:
        A,w=ls.SimBlackHoleRingdownModeEigenvaluesLeaver(spin/2,2,2,-2)
        pub=torch.tensor([complex(ls.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(float(x),spin/2,2,2,-2,A,w)) for x in mu.numpy()], dtype=torch.complex128)
        got=spin_weighted_spheroidal_harmonic_table_compat(spin,torch.acos(mu))
        # LAL's public callable is normalized over dmu; the clean callable
        # is normalized over dOmega.  Fix only the documented real-positive
        # at mu=0 phase before comparing absolute values.
        clean_anchor = spin_weighted_spheroidal_harmonic_table_compat(
            spin, torch.tensor(math.pi / 2, dtype=torch.float64)
        )
        public_anchor = complex(ls.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(
            0.0, spin / 2, 2, 2, -2, A, w
        ))
        public_anchor = torch.tensor(public_anchor, dtype=torch.complex128)
        got = got * torch.conj(clean_anchor) / clean_anchor.abs()
        pub = pub / math.sqrt(2 * math.pi)
        pub = pub * torch.conj(public_anchor) / public_anchor.abs()
        assert torch.linalg.vector_norm(got-pub).item() / torch.linalg.vector_norm(pub).item() < 1e-6


def _independent_kronrod10_21():
    """Generate the two rules from Legendre orthogonality and moments."""
    # Solve for the odd coefficients of the monic degree-11 polynomial
    # orthogonal to P_10, then solve the degree-20 moment system.
    p10 = np.polynomial.legendre.Legendre.basis(10).convert(kind=np.polynomial.Polynomial)
    powers = (9, 7, 5, 3, 1)
    def inner(poly):
        coeff = np.asarray(poly.coef)
        return sum(v * (2.0 / (k + 1) if k % 2 == 0 else 0.0)
                   for k, v in enumerate(coeff))
    base = np.polynomial.Polynomial([0]*11 + [1])
    mat = np.empty((5, 5)); rhs = np.empty(5)
    for row, k in enumerate(powers):
        rhs[row] = -inner(p10 * base * np.polynomial.Polynomial([0]*k + [1]))
        for col, power in enumerate(powers):
            q = np.polynomial.Polynomial([0]*power + [1])
            mat[row, col] = inner(p10 * q * np.polynomial.Polynomial([0]*k + [1]))
    odd = np.linalg.solve(mat, rhs)
    e = np.zeros(12); e[11] = 1.0
    for power, value in zip(powers, odd): e[power] = value
    roots = np.sort(np.concatenate((np.polynomial.legendre.leggauss(10)[0], np.roots(e[::-1]))))
    roots = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
    vand = np.array([[x**k for x in roots] for k in range(21)])
    moments = np.array([2.0/(k+1) if k % 2 == 0 else 0.0 for k in range(21)])
    weights = np.linalg.solve(vand, moments)
    return roots, weights


def test_embedded_rule_polynomial_moments_and_independent_generation():
    kx, kw = np.asarray(_K21_NODES), np.asarray(_K21_WEIGHTS)
    gx = np.polynomial.legendre.leggauss(10)[0]
    gw = np.polynomial.legendre.leggauss(10)[1]
    for degree in range(32):
        exact = 2.0/(degree+1) if degree % 2 == 0 else 0.0
        assert abs(np.dot(kw, kx**degree)-exact) < 2e-12
    for degree in range(20):
        exact = 2.0/(degree+1) if degree % 2 == 0 else 0.0
        assert abs(np.dot(gw, gx**degree)-exact) < 1e-13
    ix, iw = _independent_kronrod10_21()
    # The independent roots/Vandermonde solve is ordinary double precision;
    # its conditioning limits this implementation check to a few 1e-9.
    assert np.max(np.abs(kx-ix)) < 1e-8
    assert np.max(np.abs(kw-iw)) < 1e-8
    assert tuple(_G10_INDICES) == tuple(range(1, 20, 2))
    assert np.max(np.abs(np.asarray(_G10_WEIGHTS)-gw)) < 2e-15
