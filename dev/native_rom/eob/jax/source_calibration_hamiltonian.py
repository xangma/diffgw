"""Development-only pinned LAL SEOBNRv4 Hamiltonian calibration decimals."""

from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

from .historical_hamiltonian import HistoricalLogDerivative


class SourceCalibration:
    """Use the historical v4 header constants instead of published decimals."""

    def calibration_parameters(self, r):
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        chi = a / (1.0 - 2.0*nu)
        nu2, nu3 = nu*nu, nu*nu*nu
        chi2, chi3 = chi*chi, chi*chi*chi
        k = (1.7336 - 1.62045*chi - 1.38086*chi2 + 1.43659*chi3
             + nu*(10.2573 + 2.26831*chi - 0.426958*chi3)
             + nu2*(-126.687 + 17.3736*chi + 6.16466*chi2)
             + nu3*(267.788 - 27.5201*chi + 31.1746*chi2 - 59.1658*chi3))
        dso = (-44.5324 + 66.1987*chi3
               + nu*(-343.313*chi2 - 568.651*chi3)
               + nu2*(2495.29*chi + 147.481*chi3))
        dss = (6.06807
               + nu*(-36.0272 + 37.1964*chi - 41.0003*chi3)
               + nu2*(-326.325*chi2 + 528.511*chi3)
               + nu3*(706.958 + 1161.78*chi2))
        return k, dso, dss


class SourceCalibrationHamiltonian(SourceCalibration, SEOBNRv4Hamiltonian):
    """Source calibration with the mathematical potential derivative."""


class SourceCalibratedHistoricalHamiltonian(SourceCalibration, HistoricalLogDerivative):
    """Source calibration and its historical potential derivative."""
