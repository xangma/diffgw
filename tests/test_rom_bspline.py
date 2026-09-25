import pytest
import torch
from torchwave.constants import MSUN_SEC

from torchwave.approximants.rom.bspline import (
    OfficialROMSurrogate,
    bspline_basis,
    cubic_spline_interpolate,
    load_official_rom,
    linear_interpolate,
    reconstruct_reduced_basis,
    tensor_product_basis,
    evaluate_stitched_frequency,
)


def test_not_a_knot_cubic_matches_scipy():
    scipy = pytest.importorskip("scipy.interpolate")
    grid = torch.tensor([0.0, 0.15, 0.4, 0.7, 1.0], dtype=torch.float64)
    values = torch.sin(3 * grid)
    x = torch.linspace(0, 1, 31, dtype=torch.float64)
    actual = cubic_spline_interpolate(x, grid, values)
    expected = torch.as_tensor(scipy.CubicSpline(grid.numpy(), values.numpy(), bc_type="not-a-knot")(x.numpy()))
    torch.testing.assert_close(actual, expected)


def test_cubic_basis_partition_unity_and_right_endpoint():
    breaks = torch.tensor([0.0, 1.0, 2.0, 4.0])
    x = torch.tensor([0.0, 0.2, 1.0, 3.9, 4.0])
    values = bspline_basis(x, breaks)
    assert values.shape == (5, 6)  # nbreak + order - 2
    assert torch.allclose(values.sum(-1), torch.ones(5), atol=1e-12)
    assert values[-1, -1] == 1
    assert torch.all(values >= 0)


def test_basis_domain_and_gradient():
    x = torch.tensor([0.25, 1.5], dtype=torch.float64, requires_grad=True)
    values = bspline_basis(x, [0.0, 1.0, 2.0])
    values.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        bspline_basis(torch.tensor([-1.0]), [0.0, 1.0])


def test_basis_matches_scipy_values_and_gradients():
    scipy = pytest.importorskip("scipy.interpolate")
    import numpy as np

    breaks = np.array([0.0, 1.0, 2.0, 4.0])
    knots = np.r_[np.repeat(breaks[0], 4), breaks[1:-1], np.repeat(breaks[-1], 4)]
    x = torch.tensor([0.0, 0.2, 1.0, 3.9, 4.0], dtype=torch.float64, requires_grad=True)
    values = bspline_basis(x, breaks)
    for i in range(values.shape[-1]):
        expected = scipy.BSpline(knots, np.eye(values.shape[-1])[i], 3, extrapolate=False)
        assert np.allclose(values.detach().numpy()[:, i], expected(x.detach().numpy()), atol=1e-12)
        grad = torch.autograd.grad(values[:, i].sum(), x, retain_graph=True)[0]
        assert np.allclose(grad.detach().numpy(), expected.derivative()(x.detach().numpy()), atol=1e-10)


def test_tensor_product_batch_and_reconstruction():
    grids = ([0.0, 1.0], [-1.0, 1.0], [0.0, 2.0])
    p = tensor_product_basis(torch.tensor([0.2, 0.8]), torch.tensor(0.0), torch.tensor([0.5, 1.5]), grids)
    assert p.shape == (2, 4 * 4 * 4)
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-12)
    basis = torch.eye(3)
    c = torch.arange(6.0).reshape(2, 3)
    assert torch.equal(reconstruct_reduced_basis(basis, c), c)


def test_synthetic_official_submodel_reconstruction():
    # Two-point grids produce four clamped basis functions per axis.  A
    # constant coefficient plane should reconstruct the same value everywhere.
    shape = (2, 4, 4, 4)
    group = {
        "etavec": torch.tensor([0.0, 1.0]),
        "chi1vec": torch.tensor([-1.0, 1.0]),
        "chi2vec": torch.tensor([0.0, 2.0]),
        "Bamp": torch.eye(2),
        "Bphase": 2.0 * torch.eye(2),
        "Mf_grid_Amp": torch.tensor([0.1, 0.2]),
        "Mf_grid_Phi": torch.tensor([0.1, 0.2]),
        "Amp_ciall": torch.ones(shape),
        "Phase_ciall": torch.ones(shape),
    }
    model = OfficialROMSurrogate(group)
    amp, phase = model.evaluate(torch.tensor([0.2, 0.8]), torch.tensor(0.0), torch.tensor([0.5, 1.5]))
    assert amp.shape == (2, 2)
    assert phase.shape == (2, 2)
    assert torch.allclose(amp, torch.ones_like(amp))
    assert torch.allclose(phase, torch.ones_like(phase) * 2.0)


def test_physical_polarization_scaling_and_projection():
    group = {
        "etavec": torch.tensor([0.0, 1.0]), "chi1vec": torch.tensor([-1.0, 1.0]),
        "chi2vec": torch.tensor([0.0, 2.0]), "Bamp": torch.eye(2),
        "Bphase": torch.zeros((2, 2)), "Mf_grid_Amp": torch.tensor([0.1, 0.2]),
        "Mf_grid_Phi": torch.tensor([0.1, 0.2]), "Amp_ciall": torch.ones((2, 4, 4, 4)),
        "Phase_ciall": torch.zeros((2, 4, 4, 4)),
    }
    model = OfficialROMSurrogate(group)
    hp, hx = model.evaluate_polarizations(torch.tensor([0.15 / MSUN_SEC]), 1., 500., .5, 0., .5, inclination=0.)
    assert torch.allclose(hp.abs(), hx.abs())
    assert torch.isfinite(hp).all() and torch.isfinite(hx).all()


def test_external_official_reconstruction_when_available(official_rom_path):
    path = official_rom_path
    from torchwave.approximants.rom.official import inspect_official_rom
    schema = inspect_official_rom(path, verify_hash=True)
    for item in schema.groups:
        model = load_official_rom(path, item.name)
        args = (float(model.eta_grid[0]), float(model.chi1_grid[-1]), float(model.chi2_grid[0]))
        amp, phase = model.evaluate(*args)
        assert amp.shape == (item.n_frequency,)
        assert phase.shape == (item.n_frequency,)
        assert amp.dtype == torch.float64 and phase.dtype == torch.float64
        assert torch.isfinite(amp).all() and torch.isfinite(phase).all()
        mf = torch.stack((model.mf_grid_amp[0], model.mf_grid_amp[-1]))
        amp_f, phase_f = model.evaluate_frequency(mf, *args)
        assert amp_f.shape == phase_f.shape == (2,)
        assert torch.isfinite(model.evaluate_complex(mf, *args)).all()


def test_linear_interpolation_endpoints_and_gradient():
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    y = linear_interpolate(x, torch.tensor([0.0, 1.0], dtype=torch.float64), torch.tensor([2.0, 4.0], dtype=torch.float64))
    y.sum().backward()
    assert torch.allclose(y, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    assert torch.allclose(x.grad, torch.ones(3, dtype=torch.float64) * 2.0)
    with pytest.raises(ValueError):
        linear_interpolate(torch.tensor(-0.1), [0.0, 1.0], torch.tensor([2.0, 4.0]))


def test_stitch_cubic_overlap_is_c1_and_ringdown_gauged():
    grid = torch.linspace(0.1, 0.4, 11, dtype=torch.float64)
    pshape = (grid.numel(), 4, 4, 4)

    def make_model(coeff):
        vals = sum(a * grid.pow(k) for k, a in enumerate(coeff))
        return OfficialROMSurrogate({
            "etavec": torch.tensor([0., 1.]), "chi1vec": torch.tensor([-1., 1.]),
            "chi2vec": torch.tensor([0., 1.]), "Bamp": torch.eye(grid.numel()),
            "Bphase": torch.eye(grid.numel()), "Mf_grid_Amp": grid,
            "Mf_grid_Phi": grid, "Amp_ciall": torch.ones(pshape),
            "Phase_ciall": vals[:, None, None, None].expand(pshape),
        })

    low = make_model((1., 2., 3., 4.))
    high = make_model((2., -1., 0.5, -2.))
    eps = torch.tensor(1.e-6, dtype=torch.float64)
    x = torch.stack((torch.tensor(.2, dtype=torch.float64) - eps,
                     torch.tensor(.2, dtype=torch.float64) + eps)).requires_grad_()
    _, phase = evaluate_stitched_frequency(low, high, x, .3, .2, .5,
                                           transition=.2, overlap=(.15, .25),
                                           ringdown_frequency=.3, reference_mf=.2)
    slope = torch.autograd.grad(phase.sum(), x)[0]
    assert abs((phase[1] - phase[0]).item()) < 1.e-5
    assert abs(phase.mean().item()) < 1.e-5
    assert abs((slope[1] - slope[0]).item()) < 1.e-4

    xr = torch.tensor([.3 - eps.item(), .3 + eps.item()], dtype=torch.float64, requires_grad=True)
    _, pr = evaluate_stitched_frequency(low, high, xr, .3, .2, .5,
                                        transition=.2, overlap=(.15, .25),
                                        ringdown_frequency=.3)
    sr = torch.autograd.grad(pr.sum(), xr)[0]
    assert abs(sr.mean().item()) < 1.e-4


def test_polarization_scaling_is_m2_over_distance():
    group = {
        "etavec": torch.tensor([0., 1.]), "chi1vec": torch.tensor([-1., 1.]),
        "chi2vec": torch.tensor([0., 2.]), "Bamp": torch.eye(2),
        "Bphase": torch.zeros((2, 2)), "Mf_grid_Amp": torch.tensor([0.1, 0.2]),
        "Mf_grid_Phi": torch.tensor([0.1, 0.2]), "Amp_ciall": torch.ones((2, 4, 4, 4)),
        "Phase_ciall": torch.zeros((2, 4, 4, 4)),
    }
    model = OfficialROMSurrogate(group)
    f = torch.tensor([0.15 / MSUN_SEC], dtype=torch.float64)
    h1, _ = model.evaluate_polarizations(f, 1., 500., .5, 0., .5)
    # Keep dimensionless Mf fixed while changing the physical mass.
    h2, _ = model.evaluate_polarizations(f / 2., 2., 1000., .5, 0., .5)
    torch.testing.assert_close(h2 / h1, torch.tensor([2.], dtype=torch.complex128))


def test_stitch_ringdown_uses_native_spline_derivative():
    grid = torch.linspace(0.1, 0.4, 11, dtype=torch.float64)
    shape = (grid.numel(), 4, 4, 4)
    def model(values):
        return OfficialROMSurrogate({"etavec": torch.tensor([0., 1.]),
            "chi1vec": torch.tensor([-1., 1.]), "chi2vec": torch.tensor([0., 1.]),
            "Bamp": torch.eye(11), "Bphase": torch.eye(11),
            "Mf_grid_Amp": grid, "Mf_grid_Phi": grid,
            "Amp_ciall": torch.ones(shape),
            "Phase_ciall": values[:, None, None, None].expand(shape)})
    low = model(torch.sin(8 * grid))
    high = model(torch.sin(13 * grid) + .2)
    rd = torch.tensor(.35, dtype=torch.float64)
    x = torch.stack((rd - 1.e-7, rd + 1.e-7)).requires_grad_()
    _, phase = evaluate_stitched_frequency(low, high, x, .3, .2, .5,
                                           transition=.2, ringdown_frequency=rd,
                                           overlap=(.15, .25))
    deriv = torch.autograd.grad(phase.sum(), x)[0]
    assert abs(deriv.mean().item()) < 2.e-5


def test_stitch_rejects_overlap_without_four_nodes():
    grid = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    shape = (4, 4, 4, 4)
    model = OfficialROMSurrogate({"etavec": torch.tensor([0., 1.]),
        "chi1vec": torch.tensor([-1., 1.]), "chi2vec": torch.tensor([0., 1.]),
        "Bamp": torch.eye(4), "Bphase": torch.eye(4), "Mf_grid_Amp": grid,
        "Mf_grid_Phi": grid, "Amp_ciall": torch.ones(shape),
        "Phase_ciall": torch.zeros(shape)})
    with pytest.raises(ValueError, match="four tabulated"):
        evaluate_stitched_frequency(model, model, torch.tensor([.2]), .3, .2, .5,
                                    transition=.2, overlap=(.19, .21),
                                    ringdown_frequency=.3)


def test_ringdown_parameter_gradient_and_no_grad_value():
    grid = torch.linspace(.1, .4, 11, dtype=torch.float64)
    shape = (11, 4, 4, 4)
    def model(values):
        return OfficialROMSurrogate({"etavec": torch.tensor([0., 1.]), "chi1vec": torch.tensor([-1., 1.]),
            "chi2vec": torch.tensor([0., 1.]), "Bamp": torch.eye(11), "Bphase": torch.eye(11),
            "Mf_grid_Amp": grid, "Mf_grid_Phi": grid, "Amp_ciall": torch.ones(shape),
            "Phase_ciall": values[:, None, None, None].expand(shape)})
    low, high = model(grid.square()), model(2 * grid.square() + grid)
    rd = torch.tensor(.35, dtype=torch.float64, requires_grad=True)
    _, phase = evaluate_stitched_frequency(low, high, torch.tensor([.25]), .3, .2, .5,
                                           transition=.2, overlap=(.13, .25), ringdown_frequency=rd)
    grad = torch.autograd.grad(phase.sum(), rd)[0]
    h = 1.e-5
    def val(r):
        return evaluate_stitched_frequency(low, high, torch.tensor([.25]), .3, .2, .5,
            transition=.2, overlap=(.13, .25), ringdown_frequency=r)[1].item()
    finite = (val(.35 + h) - val(.35 - h)) / (2*h)
    assert abs(grad.item() - finite) < 1.e-5
    with torch.no_grad():
        _, phase_ng = evaluate_stitched_frequency(low, high, torch.tensor([.25]), .3, .2, .5,
            transition=.2, overlap=(.13, .25), ringdown_frequency=rd)
    assert not phase_ng.requires_grad
    torch.testing.assert_close(phase_ng, phase.detach())


def test_stitch_matches_scipy_reference_on_distinct_nonuniform_grids():
    scipy = pytest.importorskip("scipy.interpolate")
    import numpy as np
    lg = np.array([.08, .13, .17, .21, .28, .40]); hg = np.array([.10, .145, .19, .23, .31, .40])
    def model(g, amp, ph):
        g = torch.as_tensor(g, dtype=torch.float64); n = len(g); shape = (n, 4, 4, 4)
        return OfficialROMSurrogate({"etavec":torch.tensor([0.,1.]), "chi1vec":torch.tensor([-1.,1.]),
            "chi2vec":torch.tensor([0.,1.]), "Bamp":torch.eye(n), "Bphase":torch.eye(n),
            "Mf_grid_Amp":g, "Mf_grid_Phi":g, "Amp_ciall":torch.as_tensor(amp)[:,None,None,None].expand(shape),
            "Phase_ciall":torch.as_tensor(ph)[:,None,None,None].expand(shape)})
    avl, avh = np.sin(3*lg)+2, np.sin(3*hg)+2.1
    pvl, pvh = np.sin(7*lg)+lg**2, np.sin(5*hg)+.4*hg**2
    low, high = model(lg,avl,pvl), model(hg,avh,pvh)
    nodes = hg[(hg >= .10) & (hg <= .23)]
    z = (nodes-.165)/.065; cl = np.polynomial.polynomial.polyfit(z, scipy.CubicSpline(lg,pvl)(nodes), 3); ch = np.polynomial.polynomial.polyfit(z, pvh[(hg>=.10)&(hg<=.23)], 3)
    tr=.2; zt=(tr-.165)/.065
    pl=np.polynomial.polynomial.polyval(zt,cl); ph=np.polynomial.polynomial.polyval(zt,ch); dl=np.polynomial.polynomial.polyval(zt,[cl[1],2*cl[2],3*cl[3]])/.065; dh=np.polynomial.polynomial.polyval(zt,[ch[1],2*ch[2],3*ch[3]])/.065
    slope=dl-dh; intercept=pl-ph-slope*tr
    pg=np.r_[lg[lg<=tr],hg[hg>tr]]; pv=np.r_[pvl[lg<=tr],pvh[hg>tr]+intercept+slope*hg[hg>tr]]
    ag=np.r_[lg[lg<=tr],hg[hg>tr]]; av=np.r_[avl[lg<=tr],avh[hg>tr]]
    rd=.35; ps=scipy.CubicSpline(pg,pv); as_=scipy.CubicSpline(ag,av); gauge=ps(rd,1)
    x=np.array([.11,.2,.25,.35]); _, actual=evaluate_stitched_frequency(low,high,torch.tensor(x),.3,.2,.5,transition=tr,overlap=(.10,.23),ringdown_frequency=rd,reference_mf=.11)
    expected=ps(x)-gauge*(x-rd); expected-=ps(.11)-gauge*(.11-rd)
    torch.testing.assert_close(actual,torch.as_tensor(expected,dtype=torch.float64),atol=2e-10,rtol=2e-10)
    amp,_=evaluate_stitched_frequency(low,high,torch.tensor(x),.3,.2,.5,transition=tr,overlap=(.10,.23),ringdown_frequency=rd)
    torch.testing.assert_close(amp,torch.as_tensor(as_(x),dtype=torch.float64),atol=2e-10,rtol=2e-10)
