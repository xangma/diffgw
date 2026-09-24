/* Development-only complete-source mode with a controlled (2,2) NQC override. */
#include <math.h>
#include <stddef.h>

static int diffgw_use_nqc_override;
static double diffgw_nqc_override[5];

#include "full_v4_source_mode_adapter.c"
#undef XLALSimIMRSpinEOBCalculateNQCCoefficientsV4

static int diffgw_override_nqc_coefficients(
    REAL8Vector *amplitude, REAL8Vector *phase, REAL8Vector *radius,
    REAL8Vector *radial_momentum, REAL8Vector *orbital_frequency,
    INT4 ell, INT4 m, REAL8 time_peak, REAL8 step,
    REAL8 mass1, REAL8 mass2, REAL8 chi_a, REAL8 chi_s,
    EOBNonQCCoeffs *coefficients) {
  int rc = XLALSimIMRSpinEOBCalculateNQCCoefficientsV4(
      amplitude, phase, radius, radial_momentum, orbital_frequency,
      ell, m, time_peak, step, mass1, mass2, chi_a, chi_s, coefficients);
  if (!rc && diffgw_use_nqc_override && ell == 2 && m == 2) {
    coefficients->a1 = diffgw_nqc_override[0];
    coefficients->a2 = diffgw_nqc_override[1];
    coefficients->a3 = diffgw_nqc_override[2];
    coefficients->b1 = diffgw_nqc_override[3];
    coefficients->b2 = diffgw_nqc_override[4];
  }
  return rc;
}

int diffgw_v4_full_source_mode_with_nqc(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, size_t capacity,
    size_t *count, double *epoch, double *delta_t, double *out,
    const double *active_coefficients) {
  if (!active_coefficients) return 1;
  for (size_t i = 0; i < 5; ++i) {
    if (!isfinite(active_coefficients[i])) return 2;
    diffgw_nqc_override[i] = active_coefficients[i];
  }
  diffgw_use_nqc_override = 1;
  int rc = diffgw_v4_full_source_mode(
      mass1, mass2, chi1, chi2, requested_omega, step_M,
      capacity, count, epoch, delta_t, out);
  diffgw_use_nqc_override = 0;
  return rc;
}
