/* Development-only wrapper around the pinned LALSuite V4 NQC source. */
#include <lal/AVFactories.h>
#include <lal/XLALGSL.h>
#include "LALSimIMREOBNQCCorrection.c"

int diffgw_v4_nqc_same_input(const double *amp, const double *phase,
                             const double *radius, const double *prstar,
                             const double *omega, unsigned int n,
                             double time_peak, double dt, double m1, double m2,
                             double chi_a, double chi_s, double *out) {
  REAL8Vector a = {n, (double *)amp};
  REAL8Vector p = {n, (double *)phase};
  REAL8Vector r = {n, (double *)radius};
  REAL8Vector pr = {n, (double *)prstar};
  REAL8Vector om = {n, (double *)omega};
  EOBNonQCCoeffs c;
  int rc = XLALSimIMRSpinEOBCalculateNQCCoefficientsV4(
      &a, &p, &r, &pr, &om, 2, 2, time_peak, dt,
      m1, m2, chi_a, chi_s, &c);
  if (rc == 0) {
    out[0] = c.a1; out[1] = c.a2; out[2] = c.a3;
    out[3] = c.b1; out[4] = c.b2;
  }
  return rc;
}

double diffgw_v4_nqc_delta_t(double m1, double m2, double chi_a,
                              double chi_s) {
  return XLALSimIMREOBGetNRSpinPeakDeltaTv4(
      2, 2, m1, m2, chi_s + chi_a, chi_s - chi_a);
}
