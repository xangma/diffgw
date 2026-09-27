"""Eq. (24) face-on tidal strain through the positive-22 Pv2 boundary."""

import numpy as np
import pytest

from diffgw_reference import nrtidalv2_amplitude
from diffgw_reference.nrtidalv2_pv2_projection import (
    mode_from_face_on_strain,
    project_positive_22,
)
from diffgw_core.constants import MSUN_SEC
from test_nrtidalv2_paper_d_composition_math_reference import CASES
from test_nrtidalv2_phase_math_reference import require_module
from test_nrtidalv2_pv2_aligned_additive_terms_math_reference import (
    _native as native_aligned_term,
    _reference as reference_aligned_term,
)


LIMIT = 1e-10
ZERO_THETA = (0.0, 0.4, 1.1, 0.6)
ANGLE_SETS = (
    ((0.0,) * 4, (0.0,) * 4, (0.0,) * 4, ZERO_THETA, (0.0,) * 4),
    ((0.2, 0.5, -0.7, 0.9), (-0.4, 0.8, 0.3, -0.2),
     (0.35, 0.9, 1.2, 0.5), (0.6, 1.1, 2.2, 0.8),
     (0.15, -0.25, 0.45, 0.1)),
)


def _native_projection(backend, face_on, angles):
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.phenompv2_waveform_clean import (
            face_on_to_positive_22,
        )
        from torchwave.approximants.precession_twisting_clean import (
            project_twisted_positive_modes,
        )
        strain = torch.as_tensor(face_on, dtype=torch.complex128)
        mode = face_on_to_positive_22(strain)
        args = [torch.as_tensor(angle, dtype=torch.float64) for angle in angles]
        projected = project_twisted_positive_modes({(2, 2): mode}, *args)
        return tuple(np.asarray(value.detach().cpu()) for value in projected), np.asarray(mode.detach().cpu())

    jax = require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.phenompv2_waveform import (
        _project,
        face_on_to_positive_22,
    )
    mode = face_on_to_positive_22(jnp.asarray(face_on, dtype=jnp.complex128))
    args = [jnp.asarray(angle, dtype=jnp.float64) for angle in angles]
    projected = _project(mode, *args)
    return tuple(np.asarray(value) for value in projected), np.asarray(mode)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_signed_tidal_term_normalizes_to_pv2_mode_and_projects(backend, case):
    _, m1, m2, _, _, lam1, lam2 = case
    merger = float(nrtidalv2_amplitude.merger_frequency(
        (m1 + m2) * MSUN_SEC, (m1 - m2) / (m1 + m2), lam1, lam2))
    frequencies = np.array((100.0, 800.0, 1.05 * merger, 1.25 * merger))
    face_on, scales, _ = reference_aligned_term(case, frequencies)
    native_plus, native_cross = native_aligned_term(case, frequencies, backend)
    live = scales > 0

    for angle_index, angles in enumerate(ANGLE_SETS):
        expected_mode = np.asarray([mode_from_face_on_strain(value) for value in face_on])
        expected = tuple(np.asarray([pair[index] for pair in (
            project_positive_22(value, *(angle[i] for angle in angles))
            for i, value in enumerate(expected_mode))]) for index in (0, 1))
        actual, native_mode = _native_projection(backend, native_plus, angles)

        assert np.max(np.abs(native_mode[live] - expected_mode[live]) / scales[live]) <= LIMIT
        for observed, target in zip(actual, expected):
            assert np.max(np.abs(observed[live] - target[live]) / scales[live]) <= LIMIT
            assert observed[-1] == 0j
        assert native_mode[-1] == 0j
        assert np.max(np.abs(native_plus[live] - native_cross[live]) / scales[live]) <= LIMIT

        if angle_index == 0:
            theta = np.asarray(ZERO_THETA)
            closed_plus = face_on * (1 + np.cos(theta) ** 2) / 2
            closed_cross = -1j * face_on * np.cos(theta)
            assert np.max(np.abs(actual[0][live] - closed_plus[live]) / scales[live]) <= LIMIT
            assert np.max(np.abs(actual[1][live] - closed_cross[live]) / scales[live]) <= LIMIT
