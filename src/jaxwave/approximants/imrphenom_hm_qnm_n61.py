"""Independent N=61 London higher mode QNM coefficient artifact in JAX.

Port of torchwave.approximants.imrphenom_hm_qnm_n61 for JAX / jaxwave.
Evaluates the Kerr QNM polynomial fits using JAX arrays and operations.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Tuple, Union

import jax.numpy as jnp
from jaxwave.conversions import as_array


N61_COUNT = 61
N61_SPIN_LIMIT = 0.995
N61_GRID_DEFINITION = (
    "a=0.995*sin(theta), theta=linspace(-pi/2, pi/2, 61); "
    "sampling hypothesis inferred from published intervals"
)
N61_FIT_STRATEGY = "main"
N61_ARTIFACT_FORMAT = "rectangular-complex-polynomial-v1"

N61_COEFFICIENTS: Mapping[Tuple[int, int], Tuple[Tuple[float, float], ...]] = {
    (2, 2): (
        (-1.5137592356077287, 0.36798564422900365),
        (1.8243639809368366, -0.6913165107600538),
        (-1.9491019671889762, 0.7807819246225719),
        (1.3160005627471976, -0.5086885200100241),
        (-0.3862329415256309, 0.13939638563161322),
    ),
    (2, 1): (
        (0.5885554892596092, 0.02563335346659733),
        (-0.12446791209339214, 0.14217928426441337),
        (1.0237860603050764, -0.5240817953957255),
        (-5.569485148742999, 2.352279429576638),
        (10.070993355955238, -4.729609575684578),
        (-8.324174192430714, 4.251671113144883),
        (2.6784785675009144, -1.434408482636016),
    ),
    (3, 3): (
        (-2.0630526007097023, 0.3682129194385027),
        (2.3689903457083252, -0.6977933297868538),
        (-2.5267737462779536, 0.8485933993243995),
        (1.651391823096703, -0.6037405141322733),
        (-0.4670490066606495, 0.1757695652734098),
    ),
    (3, 2): (
        (1.0224522788354553, 0.004979262341947852),
        (0.1945626674672721, 0.1526602147127759),
        (-1.704633640843033, 0.005659980087435735),
        (0.933358394113722, 0.15370006474868098),
        (1.2818695224619372, -0.838937276838682),
        (-2.04726775480385, 1.0049315192185433),
        (0.8331866494462653, -0.3937205753177846),
    ),
    (4, 4): (
        (-2.6333312060241103, 0.36788767700452407),
        (2.898278419245905, -0.6854363442149691),
        (-3.106773353597755, 0.8400608550544463),
        (2.03656558955733, -0.6155581849437892),
        (-0.5742966915917768, 0.1849590836153018),
    ),
    (4, 3): (
        (0.16977139484575113, 0.11498537962071931),
        (-3.078971526335092, 0.3881207859200101),
        (4.11030038777349, -1.0248059820254125),
        (-2.8795425634478558, 0.9394300060363092),
        (0.8476129416325867, -0.3268320899788283),
    ),
}

N61_COEFFICIENT_PAYLOAD_SHA256 = (
    "afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525"
)
_N61_COMPUTED_PAYLOAD_SHA256 = hashlib.sha256(
    repr(dict(N61_COEFFICIENTS)).encode("utf-8")
).hexdigest()
if _N61_COMPUTED_PAYLOAD_SHA256 != N61_COEFFICIENT_PAYLOAD_SHA256:
    raise RuntimeError(
        "N=61 coefficient payload does not match its expected SHA256 digest"
    )

N61_GENERATION_ARTIFACT = {
    "count": N61_COUNT,
    "spin_limit": N61_SPIN_LIMIT,
    "grid": N61_GRID_DEFINITION,
    "fit_strategy": N61_FIT_STRATEGY,
    "format": N61_ARTIFACT_FORMAT,
    "coefficient_payload_sha256": N61_COEFFICIENT_PAYLOAD_SHA256,
}

_ZERO_DAMPED = frozenset(((2, 2), (3, 3), (4, 4), (4, 3)))


def _horner(coeffs: Tuple[float, ...] | list[float], x: jnp.ndarray) -> jnp.ndarray:
    result = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        result = result * x + coeff
    return result


def evaluate_n61_qnm(
    mode: Tuple[int, int] | Tuple[int, int, int],
    final_spin: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate one frozen N=61 fit in final-mass units in JAX.

    Returns ``(M_f omega_RD, M_f/tau)``.  The fitted complex frequency follows
    London's ``omega=omega_r+i/tau`` convention; the second component is the
    positive angular damping magnitude ``M_f/tau``.  The caller performs the
    ``1/(2*pi)`` conversion to cyclic frequency and the total-to-final-mass
    conversion.
    """
    lm = (int(mode[0]), int(mode[1]))
    if lm not in N61_COEFFICIENTS:
        raise ValueError(f"unsupported N=61 HM QNM mode {mode!r}")
    spin = as_array(final_spin, dtype=jnp.float64)

    # Equation (21) of London et al.
    kappa = jnp.log(2.0 - spin) / math.log(3.0)
    exponent = 1.0 / float(2 + lm[0] - abs(lm[1]))
    kappa = jnp.power(kappa, exponent)

    raw = N61_COEFFICIENTS[lm]
    real_coeffs = [c[0] for c in raw]
    imag_coeffs = [c[1] for c in raw]

    real = _horner(real_coeffs, kappa)
    imag = _horner(imag_coeffs, kappa)

    if lm in _ZERO_DAMPED:
        omega_real = float(lm[1]) / 2.0 + kappa * real
        omega_imag = kappa * imag
    else:
        omega_real = real
        omega_imag = imag
    return omega_real, jnp.abs(omega_imag)


__all__ = [
    "N61_COUNT",
    "N61_SPIN_LIMIT",
    "N61_GRID_DEFINITION",
    "N61_FIT_STRATEGY",
    "N61_ARTIFACT_FORMAT",
    "N61_COEFFICIENTS",
    "N61_COEFFICIENT_PAYLOAD_SHA256",
    "N61_GENERATION_ARTIFACT",
    "evaluate_n61_qnm",
]
