/* Development-only component oracle. Compile against the pinned LALSuite
 * source/header snapshot and wheel; no LAL implementation is redistributed. */
#include <math.h>
#include <string.h>

#include <lal/LALDict.h>
#include <lal/XLALGSL.h>
#include <gsl/gsl_deriv.h>
#include "LALSimIMRSpinEOBHamiltonian.c"

extern REAL8 XLALSimInspiralEOBPAHamiltonianWrapper(
    REAL8, REAL8, REAL8, SpinEOBHCoeffs *, LALDict *);

/* out = {source H_real/nu, wheel H_real/nu, K, dSO, dSS}. */
int diffgw_v4_hamiltonian_same_state(
    double x1, double x2, double chi1, double chi2,
    double r, double prstar, double pphi, double *out) {
  if (!out || x1 <= 0. || x2 <= 0. || fabs(x1 + x2 - 1.) > 1.e-12
      || !isfinite(chi1) || !isfinite(chi2) || r <= 0.
      || !isfinite(prstar) || !isfinite(pphi)) return 1;

  const double nu = x1*x2;
  const double a1 = x1*x1*chi1;
  const double a2 = x2*x2*chi2;
  const double aK = a1 + a2;
  const double Sstar = nu*(chi1 + chi2);
  TidalEOBParams tidal1 = {0}, tidal2 = {0};
  tidal1.quadparam = tidal2.quadparam = 1.;
  SpinEOBHCoeffs coeffs = {0};
  coeffs.tidal1 = &tidal1;
  coeffs.tidal2 = &tidal2;
  if (XLALSimIMRCalculateSpinEOBHCoeffs(&coeffs, nu, aK, 4) != 0) return 2;

  LALDict *params = XLALCreateDict();
  if (!params) return 3;
  int rc = XLALDictInsertREAL8Value(params, "nu", nu)
      || XLALDictInsertREAL8Value(params, "a1", a1)
      || XLALDictInsertREAL8Value(params, "a2", a2)
      || XLALDictInsertREAL8Value(params, "aK", aK)
      || XLALDictInsertREAL8Value(params, "Sstar", Sstar)
      || XLALDictInsertUINT2Value(params, "analyticFlag", 0);
  if (rc) {
    XLALDestroyDict(params);
    return 4;
  }

  double x_data[3] = {r, 0., 0.};
  double p_data[3] = {prstar, pphi/r, 0.};
  double s1_data[3] = {0., 0., a1};
  double s2_data[3] = {0., 0., a2};
  double sk_data[3] = {0., 0., aK};
  double ss_data[3] = {0., 0., Sstar};
  REAL8Vector x = {3, x_data}, p = {3, p_data};
  REAL8Vector s1 = {3, s1_data}, s2 = {3, s2_data};
  REAL8Vector sk = {3, sk_data}, ss = {3, ss_data};
  out[0] = XLALSimIMRSpinEOBHamiltonian(
      nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs)/nu;
  out[1] = XLALSimInspiralEOBPAHamiltonianWrapper(
      r, prstar, pphi, &coeffs, params);
  out[2] = coeffs.KK;
  out[3] = coeffs.d1v2;
  out[4] = coeffs.dheffSSv2;
  XLALDestroyDict(params);
  if (!isfinite(out[0]) || !isfinite(out[1])) return 5;
  return 0;
}
