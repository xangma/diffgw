/* Development-only capture of the pinned source's pre-fit NQC arrays. */
#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <lal/LALConstants.h>

static double *diffgw_capture_data;
static size_t diffgw_capture_capacity;
static size_t diffgw_capture_count;
static double diffgw_capture_time_peak;
static double diffgw_capture_match_time;
static double diffgw_capture_step;
static double diffgw_capture_targets[5];

/* The diagnostic builds a verified private source snapshot, then inserts a
 * prototype and a macro after its NQCCorrection.c include. Only the v4 call
 * site is redirected here; the original matching function remains intact. */
#include "LALSimIMRSpinAlignedEOB.c"
#undef XLALSimIMRSpinEOBCalculateNQCCoefficientsV4

static int diffgw_capture_nqc_inputs(
    REAL8Vector *amplitude, REAL8Vector *phase,
    REAL8Vector *radius, REAL8Vector *radial_momentum,
    REAL8Vector *orbital_frequency, INT4 ell, INT4 m,
    REAL8 time_peak, REAL8 step, REAL8 mass1, REAL8 mass2,
    REAL8 chi_a, REAL8 chi_s, EOBNonQCCoeffs *coefficients) {
  size_t count = radius->length;
  if (ell != 2 || m != 2)
    return XLALSimIMRSpinEOBCalculateNQCCoefficientsV4(
        amplitude, phase, radius, radial_momentum, orbital_frequency,
        ell, m, time_peak, step, mass1, mass2, chi_a, chi_s, coefficients);
  if (!diffgw_capture_data || count < 10
      || count > diffgw_capture_capacity || amplitude->length != count
      || phase->length != count || radial_momentum->length != count
      || orbital_frequency->length < count) return XLAL_FAILURE;
  for (size_t i = 0; i < count; ++i) {
    diffgw_capture_data[5*i] = amplitude->data[i];
    diffgw_capture_data[5*i+1] = phase->data[i];
    diffgw_capture_data[5*i+2] = radius->data[i];
    diffgw_capture_data[5*i+3] = radial_momentum->data[i];
    diffgw_capture_data[5*i+4] = orbital_frequency->data[i];
  }
  diffgw_capture_count = count;
  diffgw_capture_time_peak = time_peak;
  diffgw_capture_match_time = time_peak - XLALSimIMREOBGetNRSpinPeakDeltaTv4(
      ell, m, mass1, mass2, chi_s + chi_a, chi_s - chi_a);
  diffgw_capture_step = step;
  double eta = mass1*mass2/((mass1+mass2)*(mass1+mass2));
  double chi = chi_s + chi_a*(mass1-mass2)/(mass1+mass2)/(1.-2.*eta);
  diffgw_capture_targets[0] = fabs(XLALSimIMREOBGetNRSpinPeakAmplitudeV4(
      ell, m, mass1, mass2, chi_s, chi_a));
  diffgw_capture_targets[1] = XLALSimIMREOBGetNRSpinPeakADotV4(
      ell, m, mass1, mass2, chi_s, chi_a);
  diffgw_capture_targets[2] = XLALSimIMREOBGetNRSpinPeakADDotV4(
      ell, m, mass1, mass2, chi_s, chi_a);
  diffgw_capture_targets[3] = XLALSimIMREOBGetNRSpinPeakOmegaV4(ell, m, eta, chi);
  diffgw_capture_targets[4] = XLALSimIMREOBGetNRSpinPeakOmegaDotV4(ell, m, eta, chi);
  return XLALSimIMRSpinEOBCalculateNQCCoefficientsV4(
      amplitude, phase, radius, radial_momentum, orbital_frequency,
      ell, m, time_peak, step, mass1, mass2, chi_a, chi_s, coefficients);
}

int diffgw_v4_full_source_nqc_inputs(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, size_t capacity,
    size_t *count, double *meta, double *data, double *coefficients) {
  if (!count || !meta || !data || !coefficients || capacity < 10
      || capacity > 50000 || !isfinite(mass1) || !isfinite(mass2)
      || mass1 <= 0. || mass2 <= 0. || !isfinite(chi1) || !isfinite(chi2)
      || !isfinite(requested_omega) || requested_omega <= 0.
      || !isfinite(step_M) || step_M <= 0.) return 1;
  diffgw_capture_data = data;
  diffgw_capture_capacity = capacity;
  diffgw_capture_count = 0;
  double scaled = (mass1+mass2)*LAL_MTSUN_SI;
  double fmin = requested_omega/(LAL_PI*scaled);
  REAL8Vector *nqc = XLALCreateREAL8Vector(10);
  LALDict *pa = XLALCreateDict(), *tgr = XLALCreateDict();
  SphHarmTimeSeries *modes = NULL;
  REAL8Vector *dynamics = NULL, *high = NULL;
  if (!nqc || !pa || !tgr) return 2;
  int rc = XLALSimIMRSpinAlignedEOBModes(
      &modes, &dynamics, &high, step_M*scaled,
      mass1*LAL_MSUN_SI, mass2*LAL_MSUN_SI, fmin, 100.e6*LAL_PC_SI,
      chi1, chi2, 4,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
      nqc, 1, pa, tgr);
  int result = rc || diffgw_capture_count < 10 ? 3 : 0;
  if (!result) {
    *count = diffgw_capture_count;
    meta[0] = diffgw_capture_time_peak;
    meta[1] = diffgw_capture_step;
    meta[2] = diffgw_capture_match_time;
    for (size_t i = 0; i < 5; ++i) meta[3+i] = diffgw_capture_targets[i];
    for (size_t i = 0; i < 10; ++i) coefficients[i] = nqc->data[i];
  }
  diffgw_capture_data = NULL;
  XLALDestroySphHarmTimeSeries(modes);
  XLALDestroyREAL8Vector(dynamics);
  XLALDestroyREAL8Vector(high);
  XLALDestroyREAL8Vector(nqc);
  XLALDestroyDict(pa);
  XLALDestroyDict(tgr);
  return result;
}
