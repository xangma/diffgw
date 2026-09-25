# SPDX-License-Identifier: MIT
"""Independent scalar and frontend checks for printed XHM PN amplitudes."""

import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_inspiral_amplitude as ref


ETA, DELTA, CHI_S, CHI_A = 0.21, 0.4, 0.13, 0.04
MODES = ((2, 1), (3, 2), (3, 3), (4, 4))

# Frozen complex checkpoints at v=(0.20, 0.35) for the independent scalar
# transcription of Appendix E. They are literal values, not native outputs.
HHAT_CHECKPOINTS = {
    (2, 1): (
        complex(0.0006197166793402352, 0.03576723748980353),
        complex(0.006791530317940104, 0.05904174000283082),
    ),
    (3, 2): (
        complex(0.004352901803241471, 2.0554151474997547e-05),
        complex(0.013713347190095496, 0.00033735705453152696),
    ),
    (3, 3): (
        complex(-0.0007646691395972388, -0.047204743577775383),
        complex(-0.00809559409758057, -0.06874494045763525),
    ),
    (4, 4): (
        complex(-0.006496046442471783, -1.4454824187519459e-05),
        complex(-0.00949821748473127, -0.00023724827160121037),
    ),
}


@pytest.mark.parametrize("mode", MODES)
def test_scalar_appendix_e_complex_checkpoints(mode):
    got = [ref.printed_hhat(mode, v, ETA, DELTA, CHI_S, CHI_A) for v in (0.2, 0.35)]
    np.testing.assert_allclose(got, HHAT_CHECKPOINTS[mode], rtol=3e-14, atol=3e-16)


@pytest.mark.parametrize("mode", MODES)
def test_torch_clean_complex_polynomial_matches_scalar_reference(mode):
    torch = pytest.importorskip("torch")
    from torchwave.approximants.xhm_amplitude_clean import hhat_lm_printed

    v = torch.tensor((0.2, 0.35), dtype=torch.float64)
    got = hhat_lm_printed(mode, v, ETA, DELTA, CHI_S, CHI_A)
    torch.testing.assert_close(
        got,
        torch.tensor(HHAT_CHECKPOINTS[mode], dtype=torch.complex128),
        rtol=3e-14,
        atol=3e-16,
    )


@pytest.mark.parametrize("mode", MODES)
def test_production_torch_and_jax_pn_amplitudes_match_scalar_reference(mode):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jaxwave.approximants.imrphenom_xhm import _pn_amplitude_ratio as jax_ratio
    from torchwave.approximants.imrphenom_xhm import _pn_amplitude_ratio as torch_ratio

    frequencies = np.asarray((0.004, 0.017), dtype=np.float64)
    expected = np.asarray(
        [ref.normalized_pn_amplitude(mode, frequency, ETA, DELTA, CHI_S, CHI_A) for frequency in frequencies]
    )
    torch_frequency = torch.tensor(frequencies, dtype=torch.float64, requires_grad=True)
    got_torch = torch_ratio(
        *mode,
        torch_frequency,
        torch.tensor(ETA, dtype=torch.float64),
        torch.tensor(DELTA, dtype=torch.float64),
        torch.tensor(CHI_S, dtype=torch.float64),
        torch.tensor(CHI_A, dtype=torch.float64),
    )
    got_jax = jax.jit(
        lambda frequency: jax_ratio(
            *mode,
            frequency,
            jnp.asarray(ETA),
            jnp.asarray(DELTA),
            jnp.asarray(CHI_S),
            jnp.asarray(CHI_A),
        )
    )(jnp.asarray(frequencies))

    np.testing.assert_allclose(got_torch.detach().numpy(), expected, rtol=3e-13, atol=3e-15)
    np.testing.assert_allclose(np.asarray(got_jax), expected, rtol=3e-13, atol=3e-15)
    gradient = torch.autograd.grad(got_torch.sum(), torch_frequency)[0]
    assert torch.isfinite(gradient).all()


def test_scalar_reference_rejects_unpublished_scope_and_nonpositive_frequency():
    with pytest.raises(ValueError, match="unsupported printed Appendix-E mode"):
        ref.printed_hhat((2, 2), 0.2, ETA, DELTA, CHI_S, CHI_A)
    with pytest.raises(ValueError, match="must be positive"):
        ref.normalized_pn_amplitude((3, 2), 0.0, ETA, DELTA, CHI_S, CHI_A)
