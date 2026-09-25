/* Development-only larger-buffer pinned-source probe; source supplied separately. */
#include <math.h>
#include <string.h>
#include <lal/LALConstants.h>
#include "LALSimIMRSpinAlignedEOB.c"

int diffgw_v4_full_source_dynamics(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, size_t capacity,
    size_t *count, double *out) {
  if (!count || !out || capacity == 0 || capacity > 150000
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
  int result = 0;
  if (rc || !dynamics || dynamics->length < 50 || dynamics->length%5) {
    result = 3;
  } else if (dynamics->length/5 > capacity) {
    result = 4;
  } else {
    *count = dynamics->length/5;
    memcpy(out, dynamics->data, dynamics->length*sizeof(double));
  }
  XLALDestroySphHarmTimeSeries(modes);
  XLALDestroyREAL8Vector(dynamics);
  XLALDestroyREAL8Vector(high);
  XLALDestroyREAL8Vector(nqc);
  XLALDestroyDict(pa);
  XLALDestroyDict(tgr);
  return result;
}
