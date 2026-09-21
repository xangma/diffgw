"""Torch/JAX parity gates for the aligned-spin IMRPhenomTHM modes.

The checks deliberately consume the public ``modes`` method and compare the
complete signed-mode complex waveform.  This catches projection/conjugation
errors that can cancel in a single real polarization.
"""

import numpy as np
import pytest


torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.numpy  # keep the import explicit for environments with lazy JAX loading
import jax.numpy as jnp

from torchwave.approximants.imrphenomthm import IMRPhenomTHM as TorchTHM
from jaxwave.approximants.imrphenomthm import IMRPhenomTHM as JaxTHM


jax.config.update("jax_enable_x64", True)

# Include every supported signed mode.  The negative partners are essential:
# their equatorial-symmetry conjugation is part of the physical complex signal.
_MODES = tuple((l, s * m) for l, m in ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5)) for s in (1, -1))


_MSUN_SEC = 4.925490947641267e-6


def _times(mass1, mass2):
    # The public API takes seconds while the fit is expressed in total-mass
    # units.  Scale the compact grid per case so it spans inspiral, matching,
    # and ringdown without making the test depend on a particular binary mass.
    return np.linspace(-200.0, 20.0, 17, dtype=np.float64) * (mass1 + mass2) * _MSUN_SEC


def _flatten(modes):
    return np.concatenate([np.asarray(modes[key]).reshape(-1) for key in _MODES])


def _metrics(torch_modes, jax_modes):
    a = _flatten(torch_modes)
    b = _flatten(jax_modes)
    relative_l2 = np.linalg.norm(a - b) / np.linalg.norm(b)
    inner = np.real(np.vdot(a, b))
    norm_product = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    mismatch = max(0.0, 1.0 - inner / norm_product)
    return float(relative_l2), float(mismatch)


_EDGE_CASES = (
    # Near equal mass with unequal aligned spins exercises delta/S/dchi.
    (30.0, 25.0, 0.2, -0.1),
    # Large mass ratio and near-extremal primary spin exercise remnant/QNM fits.
    (60.0, 6.0, 0.9, 0.1),
    # Strong anti-aligned contrast tests the low-final-spin branch.
    (50.0, 10.0, 0.5, -0.4),
)


@pytest.mark.parametrize("mass1,mass2,chi1,chi2", _EDGE_CASES)
def test_phenomthm_signed_complex_mode_parity(mass1, mass2, chi1, chi2):
    """All signed complex modes agree through merger and ringdown."""
    times = _times(mass1, mass2)
    kwargs = dict(mass1=mass1, mass2=mass2, chi1=chi1, chi2=chi2,
                  distance=140.0, phic=0.23, tc=0.0, f_ref=20.0,
                  mode_array=_MODES)
    torch_modes = TorchTHM().modes(torch.as_tensor(times), **kwargs)
    jax_modes = JaxTHM().modes(jnp.asarray(times), **kwargs)
    relative_l2, mismatch = _metrics(torch_modes, jax_modes)
    assert relative_l2 <= 1.0e-11, f"complex relative L2 {relative_l2:.3e}"
    assert mismatch <= 1.0e-14, f"complex mismatch {mismatch:.3e}"


def _objective_torch(values, times):
    model = TorchTHM()
    kwargs = dict(mass1=values[0], mass2=values[1], chi1=values[2], chi2=values[3],
                  distance=180.0, phic=0.17, f_ref=20.0, mode_array=_MODES)
    modes = model.modes(torch.as_tensor(times), **kwargs)
    # Scale physical strain to order unity and retain phase sensitivity.
    weight = torch.linspace(0.3, 1.2, len(times), dtype=torch.float64)
    return sum(((value.real * weight + value.imag * (1.3 - weight)) * 1e21).sum()
               for value in modes.values())


def _objective_jax(values, times):
    model = JaxTHM()
    modes = model.modes(
        jnp.asarray(times), values[0], values[1], chi1=values[2], chi2=values[3],
        distance=180.0, phic=0.17, f_ref=20.0, mode_array=_MODES,
    )
    weight = jnp.linspace(0.3, 1.2, len(times))
    return sum(jnp.sum((jnp.real(value) * weight + jnp.imag(value) * (1.3 - weight)) * 1e21)
               for value in modes.values())


def test_phenomthm_intrinsic_mass_spin_gradient_parity():
    """Mass and aligned-spin derivatives retain Torch/JAX parity."""
    times = _times(32.0, 24.0)
    values = torch.tensor((32.0, 24.0, 0.35, -0.15), dtype=torch.float64, requires_grad=True)
    _objective_torch(values, times).backward()
    torch_gradient = values.grad.detach().numpy()
    jax_gradient = np.asarray(jax.grad(_objective_jax)(jnp.asarray((32.0, 24.0, 0.35, -0.15), dtype=jnp.float64), jnp.asarray(times)))
    assert np.isfinite(torch_gradient).all() and np.isfinite(jax_gradient).all()
    assert np.linalg.norm(torch_gradient) > 1e-3
    np.testing.assert_allclose(jax_gradient, torch_gradient, rtol=2.0e-9, atol=1.0e-10)


def test_phenomthm_jax_jit_full_signed_modes():
    """JIT compilation preserves the eager full-complex signed mode result."""
    times = _times(60.0, 6.0)
    model = JaxTHM()

    def evaluate(mass1, mass2, chi1, chi2):
        return model.modes(
            jnp.asarray(times), mass1, mass2, chi1=chi1, chi2=chi2,
            distance=140.0, phic=0.23, f_ref=20.0, mode_array=_MODES,
        )

    eager = evaluate(60.0, 6.0, 0.9, 0.1)
    compiled = jax.jit(evaluate)(60.0, 6.0, 0.9, 0.1)
    relative_l2, mismatch = _metrics(compiled, eager)
    assert relative_l2 <= 1.0e-11, f"JIT complex relative L2 {relative_l2:.3e}"
    assert mismatch <= 1.0e-14, f"JIT complex mismatch {mismatch:.3e}"
