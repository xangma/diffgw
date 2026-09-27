"""Selected whole-class monochromatic CW implementation gate."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

from diffgw_reference import continuous_wave_direct_class as ref
from jaxwave.approximants.continuous_wave import ContinuousWave as JaxCW
from torchwave.approximants.continuous_wave import ContinuousWave as TorchCW
from torchwave.detectors.network import Detector as TorchDetector

jax.config.update("jax_enable_x64", True)
T0 = 1126259462.0
CASES = (
    # Separated sky positions, frequency, inclination, polarization, and sidereal times.
    (1.1, -0.3, 30.0, 0.4, 0.27, 0.13, (0.0, 0.125, 1234.0)),
    (2.7, 0.6, 71.0, -0.6, -0.4, 1.2, (43200.0, 86400.0)),
)


def _scalar(t, ra, dec, f0, ci, psi, phi0, *, ellipticity=None):
    det = TorchDetector("H1")
    # The survey digits are conditioning inputs, not claimed paper-verified data.
    kw = dict(vertex_ecef=det.vertex.tolist(), x_arm_ecef=det.x_arm.tolist(),
              y_arm_ecef=det.y_arm.tolist())
    if ellipticity is None:
        kw["h0"] = 2e-25
    else:
        kw.update(ellipticity=ellipticity, distance=1e20, i_zz=1.2e38)
    with mp.workdps(70):
        return float(ref.monochromatic_h1_sample(
            t, t0=T0, ra=ra, dec=dec, f0=f0, cos_iota=ci,
            psi=psi, phi0=phi0, **kw))


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("ra,dec,f0,ci,psi,phi0,offsets", CASES)
@pytest.mark.parametrize("amplitude", ("h0", "ellipticity"))
def test_direct_class_whole_strain(backend, ra, dec, f0, ci, psi, phi0, offsets, amplitude):
    times = [T0 + x for x in offsets]
    kw = dict(cos_iota=ci, psi=psi, phi0=phi0, t0=T0, f1=0.0, f2=0.0)
    if amplitude == "h0":
        kw["h0"] = 2e-25
        inertia = 1e38
        ellipticity = None
    else:
        kw.update(ellipticity=1e-5, distance=1e20)
        inertia = 1.2e38
        ellipticity = 1e-5
    if backend == "torch":
        actual = TorchCW(detector="H1", i_zz=inertia)(
            torch.tensor(times, dtype=torch.float64), ra, dec, f0, **kw)
    else:
        actual = JaxCW(detector="H1", i_zz=inertia)(
            jnp.asarray(times, dtype=jnp.float64), ra, dec, f0, **kw)
    expected = np.array([_scalar(t, ra, dec, f0, ci, psi, phi0,
                                 ellipticity=ellipticity) for t in times])
    scale = max(np.max(np.abs(expected)), 1e-30)
    np.testing.assert_allclose(np.asarray(actual) / scale, expected / scale,
                               rtol=0, atol=2e-8)


def test_scalar_clock_is_bounded_to_fixed_leap_epoch():
    for outside in (1119744016.0, 1167264018.0):
        with pytest.raises(ValueError, match="fixed 17-second"):
            ref.gmst_utc_2015(outside)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("parameter", ("f0", "phi0"))
def test_direct_class_selected_gradients(backend, parameter):
    t, ra, dec, ci, psi = T0 + 1234.0, 1.1, -0.3, 0.4, 0.27
    f0, phi0, h0 = 30.0, 0.13, 2e-25
    det = TorchDetector("H1")
    site = dict(vertex_ecef=det.vertex.tolist(), x_arm_ecef=det.x_arm.tolist(),
                y_arm_ecef=det.y_arm.tolist())
    with mp.workdps(70):
        def scalar(x):
            frequency, phase_origin = (x, phi0) if parameter == "f0" else (f0, x)
            return ref.monochromatic_h1_sample(
                t, t0=T0, ra=ra, dec=dec, f0=frequency, h0=h0,
                cos_iota=ci, psi=psi, phi0=phase_origin, **site)
        expected = float(mp.diff(scalar, mp.mpf(str(f0 if parameter == "f0" else phi0))))
    if backend == "torch":
        x = torch.tensor(f0 if parameter == "f0" else phi0,
                         dtype=torch.float64, requires_grad=True)
        model = TorchCW(detector="H1")
        y = model(torch.tensor([t], dtype=torch.float64), ra, dec,
                  x if parameter == "f0" else f0, h0=h0, cos_iota=ci,
                  psi=psi, phi0=x if parameter == "phi0" else phi0, t0=T0)[0]
        actual = torch.autograd.grad(y, x)[0].item()
    else:
        model = JaxCW(detector="H1")
        def native(x):
            return model(jnp.asarray([t]), ra, dec,
                         x if parameter == "f0" else f0, h0=h0,
                         cos_iota=ci, psi=psi,
                         phi0=x if parameter == "phi0" else phi0, t0=T0)[0]
        actual = float(jax.grad(native)(jnp.asarray(f0 if parameter == "f0" else phi0)))
    assert abs(actual - expected) <= 2e-8 * max(abs(expected), h0)


def test_direct_class_definition_binding_is_scoped():
    from diffgw.waveforms import get_waveform

    definition = get_waveform("ContinuousWaveH1Direct")
    assert TorchCW.definition_id == JaxCW.definition_id == definition.id
    assert definition.claims[0].options == (
        ("call_route", "direct_class"), ("detector", "H1"),
        ("spindown", "f1=f2=0"),
    )
