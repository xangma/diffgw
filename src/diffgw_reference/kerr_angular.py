"""Scalar angular projection oracle for spin weight -2.

Cook and Zalutskiy, arXiv:1410.7698v2, Eqs. (4), (48), (49)--(56),
specify the angular equation, spherical expansion and mirror symmetry.  The
spherical basis here comes from SymPy's Wigner-d matrix.  Spheroidal matrix
elements are integrated on the sphere with mpmath Gauss--Legendre quadrature;
no native angular matrix, coefficients or waveform helper is imported.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_d_small


_THETA = sp.Symbol("theta", real=True)


@lru_cache(None)
def _wigner_function(ell: int, m: int):
    if ell < 2 or abs(m) > ell:
        raise ValueError("Spin -2 harmonics require ell >= 2 and |m| <= ell")
    # SymPy's row is m'=2 and column is m for this convention.  In particular
    # d[2,1] is positive and d[3,3] is negative, fixing the mode phase.
    element = wigner_d_small(ell, _THETA)[ell - 2, ell - m]
    return sp.lambdify(_THETA, element, "mpmath")


def spherical_harmonic(ell: int, m: int, theta, phi=0):
    """Unit-norm {}_-2 Y_lm with exp(+im phi) azimuthal dependence."""
    return (mp.sqrt((2 * ell + 1) / (4 * mp.pi))
            * _wigner_function(ell, m)(theta) * mp.exp(1j * m * phi))


def spherical_projection(ell: int, m: int, theta, phi):
    """Complex plus/cross coefficients of a positive-m and mirror pair."""
    positive = spherical_harmonic(ell, m, theta, phi)
    mirror = (-1) ** ell * mp.conj(spherical_harmonic(ell, -m, theta, phi))
    return (positive + mirror) / 2, -1j * (positive - mirror) / 2


def spheroidal_coefficients(ell: int, m: int, c, *, ell_max=10, order=32):
    """Integrate the angular operator and follow its c=0 eigenvalue branch.

    Coefficients have unit Hermitian norm and a real-positive ell anchor.  The
    finite basis and quadrature order are explicit convergence controls.
    """
    ell_min = max(2, abs(m))
    if ell < ell_min or ell_max < ell + 3:
        raise ValueError("Require ell >= max(2, |m|) and ell_max >= ell+3")
    degrees = range(ell_min, ell_max + 1)
    size = len(degrees)
    nodes, weights = mp.gauss_quadrature(order, "legendre")
    basis = [[spherical_harmonic(j, m, mp.acos(x)) for j in degrees]
             for x in nodes]
    x1, x2 = mp.matrix(size), mp.matrix(size)
    for a in range(size):
        for b in range(size):
            x1[a, b] = 2 * mp.pi * sum(
                weights[k] * nodes[k] * mp.conj(basis[k][a]) * basis[k][b]
                for k in range(order))
            x2[a, b] = 2 * mp.pi * sum(
                weights[k] * nodes[k] ** 2 * mp.conj(basis[k][a]) * basis[k][b]
                for k in range(order))

    def operator(oblat):
        matrix = mp.matrix(size)
        for a, j in enumerate(degrees):
            for b in range(size):
                matrix[a, b] = ((j * (j + 1) - 2 if a == b else 0)
                                - 4 * oblat * x1[a, b]
                                - oblat ** 2 * x2[a, b])
        return matrix

    previous = mp.mpc(ell * (ell + 1) - 2)
    for step in range(1, 9):
        values = mp.eig(operator(c * step / 8), left=False, right=False)
        previous = min(values, key=lambda value: abs(value - previous))
    values, vectors = mp.eig(operator(c), left=False, right=True)
    index = min(range(size), key=lambda k: abs(values[k] - previous))
    coefficients = [vectors[k, index] for k in range(size)]
    norm = mp.sqrt(sum(abs(value) ** 2 for value in coefficients))
    coefficients = [value / norm for value in coefficients]
    anchor = coefficients[ell - ell_min]
    coefficients = [value * mp.conj(anchor) / abs(anchor)
                    for value in coefficients]
    return values[index], coefficients


def spheroidal_harmonic(ell: int, m: int, c, theta, phi=0, **kwargs):
    """Evaluate the quadrature Galerkin eigenfunction at one sky angle."""
    _, coefficients = spheroidal_coefficients(ell, m, c, **kwargs)
    return sum(coef * spherical_harmonic(j, m, theta, phi)
               for j, coef in zip(range(max(2, abs(m)),
                                        kwargs.get("ell_max", 10) + 1), coefficients))


def spheroidal_projection(ell: int, m: int, c, theta, phi, **kwargs):
    """Mirror projection using the Cook--Zalutskiy symmetry at -c* ."""
    _, coefficients = spheroidal_coefficients(ell, m, c, **kwargs)
    return projection_from_coefficients(ell, m, coefficients, theta, phi)


def projection_from_coefficients(ell: int, m: int, coefficients, theta, phi):
    """Project a fixed, already normalized spherical expansion."""
    degrees = range(max(2, abs(m)), max(2, abs(m)) + len(coefficients))
    positive = sum(coef * spherical_harmonic(j, m, theta, phi)
                   for j, coef in zip(degrees, coefficients))
    mirror = sum(coef * spherical_harmonic(j, m, mp.pi - theta, phi)
                 for j, coef in zip(degrees, coefficients))
    return (positive + mirror) / 2, -1j * (positive - mirror) / 2
