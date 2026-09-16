import torch
from torchwave.approximants.eob.nqc_analytic_clean import directional_derivatives, solve_nqc


def test_directional_second_derivative_includes_flow_jacobian():
    x = torch.tensor([1.2, -0.3], dtype=torch.float64)
    def flow(y): return torch.stack((y[1], -y[0] + y[1] ** 2))
    def vals(y):
        g = y[0] ** 2 + y[0] * y[1]
        return torch.stack((g, y[0], y[1], y[0] * y[1], y[0] ** 2, y[1] ** 2, g))
    v, d, dd = directional_derivatives(vals, x, flow)
    def dg(z):
        z = z.detach().requires_grad_(True)
        gg = vals(z)[0]
        return (torch.autograd.grad(gg, z, create_graph=True)[0] * flow(z)).sum()
    h = 1e-5
    xp = x + h * flow(x); xm = x - h * flow(x)
    finite = (dg(xp) - dg(xm)) / (2*h)
    assert torch.allclose(dd[0], finite, rtol=2e-4, atol=2e-4)
    assert torch.isfinite(v).all() and torch.isfinite(d).all()


def test_solve_nqc_reproduces_targets():
    x = torch.tensor([1.2, -0.3], dtype=torch.float64)
    flow = lambda y: torch.stack((y[1], -y[0] + y[1] ** 2))
    vals = lambda y: torch.stack((1+y[0]**2, y[0], y[1], y[0]*y[1], y[0]**2, y[1]**2, 2+y[0]))
    targets = torch.tensor([1.1, 0.2, -0.3, 2.1, 0.4], dtype=torch.float64)
    a, b, (v, d, dd) = solve_nqc(vals, x, flow, targets)
    assert torch.allclose(torch.stack((v[0]+a@v[1:4], d[0]+a@d[1:4], dd[0]+a@dd[1:4])), targets[:3], atol=1e-10)
    assert torch.allclose(torch.stack((v[6]-b@d[4:6], d[6]-b@dd[4:6])), targets[3:], atol=1e-10)


def test_directional_derivatives_closed_form_constant_and_linear():
    x = torch.tensor([2.0, -1.0], dtype=torch.float64)
    flow = lambda y: torch.tensor([3.0, -2.0], dtype=y.dtype)
    vals = lambda y: torch.stack((torch.tensor(7.0, dtype=y.dtype), y[0], y[1], y[0] + 2*y[1],
                                  torch.tensor(1.0, dtype=y.dtype), torch.tensor(0.0, dtype=y.dtype), y[0]))
    value, first, second = directional_derivatives(vals, x, flow)
    assert value[0] == 7 and first[0] == 0 and second[0] == 0
    assert torch.allclose(first, torch.tensor([0., 3., -2., -1., 0., 0., 3.], dtype=torch.float64))
    assert torch.allclose(second, torch.zeros(7, dtype=torch.float64))
    constants = lambda y: torch.ones(7, dtype=y.dtype)
    cv, cd, cdd = directional_derivatives(constants, x, flow)
    assert torch.equal(cv, torch.ones(7, dtype=torch.float64))
    assert torch.equal(cd, torch.zeros(7, dtype=torch.float64))
    assert torch.equal(cdd, torch.zeros(7, dtype=torch.float64))
