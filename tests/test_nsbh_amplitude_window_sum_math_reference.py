"""Frozen conditional scalar gate for Thompson's NSBH amplitude window sum."""

import mpmath
import numpy as np
import pytest

from diffgw_reference.nsbh_amplitude_window_sum import amplitude_window_sum


MASS_SEC = 5e-5
ROWS = (
    ((800.0, 800.0, 800.0), 0.015, (0.8, 0.3, 0.5)),
    ((700.0, 700.0, 850.0), 0.040, (0.8, 0.3, 0.5)),
    ((650.0, 750.0, 850.0), 0.040, (0.8, 0.3, 0.0)),
    ((500.0, 600.0, 850.0), 0.080, (0.8, 0.3, 0.0)),
)
SCALAR_LIMIT = mpmath.mpf("1e-30")
VALUE_LIMIT = 2e-13
DERIVATIVE_LIMIT = 2e-12


def _grid(centers, width):
    full = sorted({50.0, 1500.0, *(center + k * width / (8 * MASS_SEC) for center in centers for k in (-2, -1, 0, 1, 2))})
    selected = {
        center + k * width / (8 * MASS_SEC)
        for center in centers
        for k in (-1, 0, 1)
    }
    return np.asarray([f for f in full if f > 0.0], dtype=np.float64), selected


@pytest.mark.parametrize("centers,width,carriers", ROWS)
def test_scalar_stability_and_window_centers(centers, width, carriers):
    frequencies, _ = _grid(centers, width)
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    for f in frequencies:
        low = amplitude_window_sum(f, MASS_SEC, centers, width, carriers, dps=70)
        high = amplitude_window_sum(f, MASS_SEC, centers, width, carriers, dps=90)
        for a, b in (
            (low.value, high.value),
            (low.frequency_derivative, high.frequency_derivative),
            *zip(low.carrier_partials, high.carrier_partials, strict=True),
        ):
            assert abs(ctx.mpf(a) - ctx.mpf(b)) <= SCALAR_LIMIT
        if f in centers:
            assert abs(low.carrier_partials[centers.index(f)] - ctx.mpf("0.5")) <= SCALAR_LIMIT


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("centers,width,carriers", ROWS)
def test_native_window_sum_value_and_partials(backend, centers, width, carriers):
    frequencies, selected = _grid(centers, width)
    reference = [amplitude_window_sum(f, MASS_SEC, centers, width, carriers) for f in frequencies]

    if backend == "torch":
        import torch

        from torchwave.approximants.imrphenom_nsbh import nsbh_amplitude_window_sum

        f = torch.tensor(frequencies, dtype=torch.float64, requires_grad=True)
        amp = [torch.full_like(f, value, requires_grad=True) for value in carriers]
        center_t = tuple(torch.tensor(value, dtype=torch.float64) for value in centers)
        value = nsbh_amplitude_window_sum(
            f, torch.tensor(MASS_SEC, dtype=torch.float64), center_t,
            torch.tensor(width, dtype=torch.float64), tuple(amp),
        )
        gradients = torch.autograd.grad(value.sum(), (f, *amp))
        actual_values = value.detach().numpy()
        actual_gradients = [gradient.detach().numpy() for gradient in gradients]
    else:
        import jax
        import jax.numpy as jnp

        from jaxwave.approximants.imrphenom_nsbh import nsbh_amplitude_window_sum

        center_j = tuple(jnp.asarray(value, dtype=jnp.float64) for value in centers)
        mass_j = jnp.asarray(MASS_SEC, dtype=jnp.float64)
        width_j = jnp.asarray(width, dtype=jnp.float64)
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        amp = tuple(jnp.full_like(f, value) for value in carriers)

        def scalar_value(freq, pn, pm, rd):
            return nsbh_amplitude_window_sum(freq, mass_j, center_j, width_j, (pn, pm, rd))

        value = jax.vmap(scalar_value)(f, *amp)
        gradients = jax.vmap(jax.grad(scalar_value, argnums=(0, 1, 2, 3)))(f, *amp)
        actual_values = np.asarray(value)
        actual_gradients = [np.asarray(gradient) for gradient in gradients]

    expected_values = np.asarray([float(item.value) for item in reference])
    value_error = np.max(np.abs(actual_values - expected_values))
    assert value_error <= VALUE_LIMIT, (backend, centers, value_error)
    selected_indices = [i for i, f in enumerate(frequencies) if f in selected]
    expected_gradients = [
        np.asarray([float(item.frequency_derivative) for item in reference]),
        *(np.asarray([float(item.carrier_partials[j]) for item in reference]) for j in range(3)),
    ]
    gradient_error = max(
        float(np.max(np.abs(actual[selected_indices] - expected[selected_indices])))
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True)
    )
    assert gradient_error <= DERIVATIVE_LIMIT, (backend, centers, gradient_error)
    print(f"{backend}: max_absolute_value={value_error:.3e} max_absolute_selected_partial={gradient_error:.3e}")


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_public_amplitude_magnitude_unchanged(backend):
    """Fixed output of the pre-refactor native forward path on two regimes."""
    frequency = np.asarray((50.0, 100.0, 250.0, 500.0, 900.0))
    rows = (
        (
            dict(mass1=8.4, mass2=1.4, chi1=0.0, lambda2=500.0),
            (0.18117323295912793, 0.07764675311334253, 0.023949869157876436,
             0.009305104030401082, 0.004365855539859555),
        ),
        (
            dict(mass1=2.1, mass2=1.4, chi1=0.5, lambda2=800.0),
            (0.11122610930973484, 0.04906737035109688, 0.016431574198706656,
             0.00702955209335676, 0.003247422574474279),
        ),
    )
    if backend == "torch":
        import torch

        from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH

        def generate(kwargs):
            h_plus, _ = IMRPhenomNSBH()(torch.tensor(frequency, dtype=torch.float64), **kwargs)
            return np.abs(h_plus.detach().numpy()) * 1e22

    else:
        import jax.numpy as jnp

        from jaxwave.approximants.imrphenom_nsbh import IMRPhenomNSBH

        def generate(kwargs):
            h_plus, _ = IMRPhenomNSBH.evaluate(jnp.asarray(frequency, dtype=jnp.float64), **kwargs)
            return np.abs(np.asarray(h_plus)) * 1e22

    for kwargs, expected in rows:
        np.testing.assert_allclose(generate(kwargs), expected, atol=2e-13, rtol=0.0)
