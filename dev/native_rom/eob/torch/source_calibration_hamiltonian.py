"""Development-only SEOBNRv4 calibration decimals from pinned LALSuite v6.2.1.

The rows are powers of nu, columns are powers of chi. Their source is the
pinned LALSimIMRSpinEOBHamiltonian.h header, not the published Bohé decimals.
"""

from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

from .historical_hamiltonian import HistoricalLogDerivative


SOURCE_V4_COEFFICIENTS = {
    "K": (
        (1.7336, -1.62045, -1.38086, 1.43659),
        (10.2573, 2.26831, 0.0, -0.426958),
        (-126.687, 17.3736, 6.16466, 0.0),
        (267.788, -27.5201, 31.1746, -59.1658),
    ),
    "dSO": (
        (-44.5324, 0.0, 0.0, 66.1987),
        (0.0, 0.0, -343.313, -568.651),
        (0.0, 2495.29, 0.0, 147.481),
        (0.0, 0.0, 0.0, 0.0),
    ),
    "dSS": (
        (6.06807, 0.0, 0.0, 0.0),
        (-36.0272, 37.1964, 0.0, -41.0003),
        (0.0, 0.0, -326.325, 528.511),
        (706.958, 0.0, 1161.78, 0.0),
    ),
}


class SourceCalibration:
    """Use exactly the historical v4 header's shorter decimal constants."""

    def calibration_parameters(self, r):
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        chi = a / (1.0 - 2.0*nu)
        return tuple(
            sum(value*nu**i*chi**j
                for i, row in enumerate(SOURCE_V4_COEFFICIENTS[name])
                for j, value in enumerate(row))
            for name in ("K", "dSO", "dSS")
        )


class SourceCalibrationHamiltonian(SourceCalibration, SEOBNRv4Hamiltonian):
    """Historical calibration with the mathematical potential derivative."""


class SourceCalibratedHistoricalHamiltonian(SourceCalibration, HistoricalLogDerivative):
    """Historical calibration and historical potential derivative."""
