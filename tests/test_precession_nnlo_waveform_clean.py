import pytest
import torch
import os

from torchwave.approximants.precession_nnlo_waveform_clean import waveform
from torchwave.waveforms import get_fd_waveform


def _kwargs():
    return dict(mass1=35., mass2=25., spin1x=.2, spin1y=.1, spin1z=.2,
                spin2x=-.1, spin2y=.15, spin2z=-.1, distance=100.,
                inclination=.7, phic=.3, tc=.02, f_ref=20.)


def test_clean_interface_dc_support_and_gradient():
    f = torch.tensor([0., 25., 40., 80.], dtype=torch.float64, requires_grad=True)
    hp, hc = waveform(f, **_kwargs())
    assert hp[0] == 0 and hc[0] == 0
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    (hp.real.square().sum() * 1e40).backward()
    assert torch.isfinite(f.grad).all()


def test_clean_broadcast_matches_stacked_and_float32_zero_reference():
    f = torch.tensor([0., 25., 40.], dtype=torch.float32)
    m1 = torch.tensor([35., 40.], dtype=torch.float32)
    batched = waveform(f, m1, 20., spin1z=torch.tensor([.2, .3]), f_ref=0.)
    stacked = [torch.stack([waveform(f, float(m), 20., spin1z=float(s), f_ref=0.)[j]
                            for m, s in zip(m1, [.2, .3])]) for j in (0, 1)]
    for got, expected in zip(batched, stacked):
        torch.testing.assert_close(got, expected, rtol=2e-6, atol=0.)
        assert got.dtype is torch.complex64


def test_clean_rejects_invalid_reference_grid_and_spin_norm():
    f = torch.tensor([20., 40.], dtype=torch.float64)
    kw = _kwargs()
    kw["f_ref"] = -1.
    with pytest.raises(ValueError, match="f_ref"):
        waveform(f, **kw)
    with pytest.raises(ValueError, match="finite"):
        waveform(torch.tensor([20., float("nan")]), **_kwargs())
    kw = _kwargs()
    kw.update(spin1x=.9, spin1y=.9)
    with pytest.raises(ValueError, match="spin-vector"):
        waveform(f, **kw)


def test_clean_generic_parameter_gradients_are_finite():
    f = torch.tensor([20., 35., 70.], dtype=torch.float64)
    args = dict(mass1=torch.tensor(35., requires_grad=True), mass2=25.,
                spin1x=torch.tensor(.2, requires_grad=True), spin1y=.1,
                spin1z=.2, spin2x=-.1, spin2y=.15, spin2z=-.1,
                distance=100., inclination=torch.tensor(.7, requires_grad=True),
                phic=torch.tensor(.3, requires_grad=True),
                tc=torch.tensor(.002, requires_grad=True),
                f_ref=torch.tensor(20., requires_grad=True))
    hp, hc = waveform(f, **args)
    loss = (hp.abs().square() + hc.abs().square()).sum()
    loss.backward()
    for key in ("mass1", "spin1x", "inclination", "phic", "tc", "f_ref"):
        assert torch.isfinite(args[key].grad)


@pytest.mark.parametrize("key", ["mass1", "spin1x", "inclination", "phic", "tc", "f_ref"])
def test_clean_central_difference_matches_autograd(key):
    f = torch.tensor([27., 43., 87.], dtype=torch.float64)
    kw = _kwargs()
    value = torch.tensor(float(kw[key]), dtype=torch.float64, requires_grad=True)
    kw[key] = value

    def objective(call_kw):
        hp, hc = waveform(f, **call_kw)
        return 1e22 * (hp.real + .7 * hc.imag).sum()

    loss = objective(kw)
    (grad,) = torch.autograd.grad(loss, value)
    step = 1e-5 * max(1., abs(float(value.detach())))
    plus = dict(kw); minus = dict(kw)
    plus[key] = value.detach() + step
    minus[key] = value.detach() - step
    finite = (objective(plus) - objective(minus)) / (2 * step)
    assert torch.isfinite(loss) and torch.isfinite(grad) and torch.isfinite(finite)
    scale = max(1., abs(float(grad)), abs(float(finite)))
    assert abs(float(grad - finite)) / scale < 3e-3


@pytest.mark.parametrize("inclination", [0., 1.1, 3.141592653589793])
@pytest.mark.parametrize("convention", [0, 1])
@pytest.mark.parametrize("reverse_masses", [False, True])
def test_clean_aligned_face_on_autograd_is_finite(inclination, convention, reverse_masses):
    f = torch.tensor([20., 40.], dtype=torch.float64)
    iota = torch.tensor(inclination, dtype=torch.float64, requires_grad=True)
    kw = _kwargs()
    for name in ("spin1x", "spin1y", "spin2x", "spin2y"):
        kw[name] = torch.tensor(0., dtype=torch.float64)
    if reverse_masses:
        kw.update(mass1=kw["mass2"], mass2=kw["mass1"],
                  spin1z=kw["spin2z"], spin2z=kw["spin1z"])
    kw["inclination"] = iota
    hp, hc = waveform(f, **kw, convention=convention)
    (hp.abs().square().sum() + hc.abs().square().sum()).backward()
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    assert torch.isfinite(iota.grad)


@pytest.mark.parametrize("convention", [0, 1])
def test_clean_reversed_mass_spin_labels_are_equivalent(convention):
    f = torch.tensor([20., 35., 70.], dtype=torch.float64)
    kw = _kwargs()
    a = waveform(f, **kw, convention=convention)
    swapped = dict(kw)
    swapped.update(mass1=kw["mass2"], mass2=kw["mass1"],
                   spin1x=kw["spin2x"], spin1y=kw["spin2y"], spin1z=kw["spin2z"],
                   spin2x=kw["spin1x"], spin2y=kw["spin1y"], spin2z=kw["spin1z"])
    b = waveform(f, **swapped, convention=convention)
    for got, expected in zip(b, a):
        rel = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected)
        assert rel < 1e-12


def test_clean_dispatch_is_explicit_and_matches_callable():
    f = torch.tensor([0., 25., 40., 80.], dtype=torch.float64)
    direct = waveform(f, **_kwargs())
    routed = get_fd_waveform('IMRPhenomXP', sample_frequencies=f,
                             prec_version=102, convention=0, final_spin_mod=0,
                             **_kwargs())
    for got, expected in zip(routed, direct):
        torch.testing.assert_close(got, expected, rtol=0., atol=0.)


def test_clean_convention_one_is_finite_and_routed():
    f = torch.tensor([0., 25., 40., 80.], dtype=torch.float64)
    direct = waveform(f, **_kwargs(), convention=1)
    routed = get_fd_waveform('IMRPhenomXP', sample_frequencies=f,
                             prec_version=102, convention=1, final_spin_mod=0,
                             **_kwargs())
    for got, expected in zip(routed, direct):
        torch.testing.assert_close(got, expected, rtol=0., atol=0.)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not os.environ.get("TORCHWAVE_RUN_PUBLIC_PRECESSION"),
                    reason="optional public LAL regression")
def test_clean_public_xp_nine_cases_both_conventions():
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    from torchwave.approximants.precession_nnlo_waveform_clean import waveform as clean
    cases = [
        (35.,25.,.2,.1,.2,-.1,.15,-.1,.7,.3,.02,20.),
        (42.,17.,.3,-.15,-.3,.12,.08,.2,1.1,.71,.01,35.),
        (40.,20.,.3,.2,.4,-.2,.1,-.2,.7,0.,0.,40.),
        (30.,10.,.1,-.25,-.4,.15,.05,.2,.5,0.,0.,40.),
        (50.,15.,-.2,.3,.1,.1,-.2,.25,1.,0.,0.,30.),
        (28.,12.,.05,-.1,.6,-.3,.2,-.4,.4,0.,0.,25.),
        (35.,25.,0.,0.,.2,0.,0.,-.1,0.,.3,.02,20.),
        (35.,25.,0.,0.,.2,0.,0.,-.1,.7,.3,.02,20.),
        (35.,25.,0.,0.,.2,0.,0.,-.1,1.1,.3,.02,20.),
    ]
    approx = ls.GetApproximantFromString("IMRPhenomXP")
    for convention in (0, 1):
        for m1,m2,x1,y1,z1,x2,y2,z2,inc,phic,tc,ref in cases:
            d = lal.CreateDict()
            ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d, 102)
            ls.SimInspiralWaveformParamsInsertPhenomXPConvention(d, convention)
            ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(d, 0)
            assert ls.SimInspiralWaveformParamsLookupPhenomXPrecVersion(d) == 102
            assert ls.SimInspiralWaveformParamsLookupPhenomXPConvention(d) == convention
            assert ls.SimInspiralWaveformParamsLookupPhenomXPFinalSpinMod(d) == 0
            hp, hc = ls.SimInspiralChooseFDWaveform(
                m1*lal.MSUN_SI,m2*lal.MSUN_SI,x1,y1,z1,x2,y2,z2,
                100.*1e6*lal.PC_SI,inc,phic,0.,0.,0.,.25,20.,512.,ref,d,approx)
            f = torch.arange(len(hp.data.data), dtype=torch.float64)*.25
            # The public FD constructor has no t_c argument; apply the
            # documented Fourier time translation to its returned series.
            tc_shift = torch.exp(-2j * torch.pi * f * tc)
            pred = clean(f,m1,m2,x1,y1,z1,x2,y2,z2,100.,inc,phic,tc,ref,
                         prec_version=102,convention=convention,final_spin_mod=0)
            sel=(f>=20.)&(f<512.)
            for got,target in zip(pred,(hp.data.data,hc.data.data)):
                target=torch.as_tensor(target,dtype=torch.complex128) * tc_shift
                assert torch.linalg.vector_norm(got[sel]-target[sel])/torch.linalg.vector_norm(target[sel]) < 2e-10


@pytest.mark.parametrize('bad', [dict(prec_version=101, convention=0, final_spin_mod=0),
                                 dict(prec_version=102, convention=0, final_spin_mod=1)])
def test_clean_rejects_unreviewed_options(bad):
    with pytest.raises(ValueError, match='requires prec_version=102'):
        waveform(torch.tensor([25.], dtype=torch.float64), **_kwargs(), **bad)
