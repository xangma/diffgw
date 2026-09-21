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
