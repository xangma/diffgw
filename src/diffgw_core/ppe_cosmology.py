"""Quadrature inputs for the flat matter/Lambda ppE dispersion distance.

The independent scalar oracle integrates the published equation with mpmath;
these Gauss--Legendre nodes belong only to the numerical carrier path.
"""

from numpy.polynomial.legendre import leggauss

OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7
D0_QUADRATURE_NODES, D0_QUADRATURE_WEIGHTS = leggauss(24)
