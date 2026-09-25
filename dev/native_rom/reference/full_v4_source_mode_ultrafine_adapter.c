/* Development-only ultrafine pinned-source generator probe; source supplied separately. */
#include <complex.h>
#include <math.h>
#include <lal/LALConstants.h>
#include "LALSimIMRSpinAlignedEOB.c"

int diffgw_v4_full_source_mode(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, size_t capacity,
    size_t *count, double *epoch, double *delta_t, double *out) {
  if (!count || !epoch || !delta_t || !out || capacity == 0 || capacity > 600000
      || !isfinite(mass1) || !isfinite(mass2) || mass1 <= 0. || mass2 <= 0.
      || !isfinite(chi1) || !isfinite(chi2)
      || !isfinite(requested_omega) || requested_omega <= 0.
      || !isfinite(step_M) || step_M <= 0.) return 1;
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
      nqc, 0, pa, tgr);
  COMPLEX16TimeSeries *mode = modes ? XLALSphHarmTimeSeriesGetMode(modes, 2, 2) : NULL;
  int result = 0;
  if (rc || !mode || !mode->data || mode->data->length < 10) {
    result = 3;
  } else if (mode->data->length > capacity) {
    result = 4;
  } else {
    *count = mode->data->length;
    *epoch = mode->epoch.gpsSeconds + mode->epoch.gpsNanoSeconds*1.e-9;
    *delta_t = mode->deltaT;
    for (size_t i = 0; i < *count; ++i) {
      out[2*i] = creal(mode->data->data[i]);
      out[2*i+1] = cimag(mode->data->data[i]);
    }
  }
  XLALDestroySphHarmTimeSeries(modes);
  XLALDestroyREAL8Vector(dynamics);
  XLALDestroyREAL8Vector(high);
  XLALDestroyREAL8Vector(nqc);
  XLALDestroyDict(pa);
  XLALDestroyDict(tgr);
  return result;
}

int diffgw_v4_full_source_nqc(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, double *out) {
  if (!out || !isfinite(mass1) || !isfinite(mass2) || mass1 <= 0. || mass2 <= 0.
      || !isfinite(chi1) || !isfinite(chi2)
      || !isfinite(requested_omega) || requested_omega <= 0.
      || !isfinite(step_M) || step_M <= 0.) return 1;
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
  int result = rc ? 3 : 0;
  if (!result) for (size_t i = 0; i < 10; ++i) out[i] = nqc->data[i];
  XLALDestroySphHarmTimeSeries(modes);
  XLALDestroyREAL8Vector(dynamics);
  XLALDestroyREAL8Vector(high);
  XLALDestroyREAL8Vector(nqc);
  XLALDestroyDict(pa);
  XLALDestroyDict(tgr);
  return result;
}
