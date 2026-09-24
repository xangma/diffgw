/* Development-only diagnostic; the LAL source is supplied separately. */
#include <lal/LALConstants.h>
#include "LALSimIMRSpinAlignedEOB.c"

int diffgw_v4_full_source_first(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double *out) {
  if (!out) return 1;
  double scaled = (mass1+mass2)*LAL_MTSUN_SI;
  double fmin = requested_omega/(LAL_PI*scaled);
  REAL8Vector *nqc = XLALCreateREAL8Vector(10);
  LALDict *pa = XLALCreateDict(), *tgr = XLALCreateDict();
  SphHarmTimeSeries *modes = NULL;
  REAL8Vector *dynamics = NULL, *high = NULL;
  if (!nqc || !pa || !tgr) return 2;
  int rc = XLALSimIMRSpinAlignedEOBModes(
      &modes, &dynamics, &high, 0.125*scaled,
      mass1*LAL_MSUN_SI, mass2*LAL_MSUN_SI, fmin, 100.e6*LAL_PC_SI,
      chi1, chi2, 4,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
      nqc, 0, pa, tgr);
  if (rc || !dynamics || dynamics->length < 20 || dynamics->length%5) return 3;
  size_t n = dynamics->length/5;
  out[0] = dynamics->data[n];
  out[1] = dynamics->data[3*n];
  out[2] = dynamics->data[4*n];
  XLALDestroySphHarmTimeSeries(modes);
  XLALDestroyREAL8Vector(dynamics);
  XLALDestroyREAL8Vector(high);
  XLALDestroyREAL8Vector(nqc);
  XLALDestroyDict(pa);
  XLALDestroyDict(tgr);
  return 0;
}
