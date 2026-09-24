/* Development-only explicit-state oracle. The LAL implementation is supplied
 * by a separately pinned source snapshot at compile time, not redistributed. */
#include <math.h>
#include <complex.h>
#include <string.h>

#include <lal/LALDict.h>
#include <lal/XLALGSL.h>
#include <gsl/gsl_deriv.h>
#include "AVFactories.h"
#include "LALSimIMRSpinEOBFactorizedFlux.c"
#include "LALSimIMRSpinEOBHcapExactDerivative.c"
#include "LALSimIMRSpinEOBInitialConditions.c"

extern REAL8 XLALSimInspiralEOBPAFluxWrapper(
    REAL8, REAL8, REAL8, REAL8, SpinEOBParams *, EOBNonQCCoeffs *, LALDict *);

typedef struct {
  double pphi;
  SpinEOBParams *params;
  int derivative;
} RadialDerivativeContext;

static double radial_derivative_term(double r, void *context) {
  RadialDerivativeContext *ctx = context;
  double values[6] = {r, 0., 0., 0., ctx->pphi/r, 0.};
  double dHdpy = XLALSpinHcapHybDerivWRTParam(4, values, ctx->params);
  if (ctx->derivative == 5) return dHdpy/r;
  double dHdx = XLALSpinHcapHybDerivWRTParam(0, values, ctx->params);
  return dHdx-dHdpy*ctx->pphi/(r*r);
}

/* out = {source flux/nu, wheel PA flux/nu, H_real/nu,
 * source dE/dr, source dH/dpr at pr=.001, source initial pr*,
 * source dH/dpphi, source d2H/dr2, source d2H/drdpphi}. */
int diffgw_v4_flux_same_state(
    double x1, double x2, double chi1, double chi2,
    double r, double pphi, double omega, double *out) {
  if (!out || x1 <= 0. || x2 <= 0. || fabs(x1 + x2 - 1.) > 1.e-12
      || !isfinite(chi1) || !isfinite(chi2) || r <= 0.
      || !isfinite(pphi) || pphi <= 0. || !isfinite(omega) || omega <= 0.) return 1;

  const double nu = x1*x2;
  const double a1 = x1*x1*chi1;
  const double a2 = x2*x2*chi2;
  const double aK = a1+a2;
  const double Sstar = nu*(chi1+chi2);
  const double chiS = (chi1+chi2)/2.;
  const double chiA = (chi1-chi2)/2.;
  TidalEOBParams tidal1 = {0}, tidal2 = {0};
  tidal1.mByM = x1;
  tidal2.mByM = x2;
  tidal1.quadparam = tidal2.quadparam = 1.;
  SpinEOBHCoeffs coeffs = {0};
  coeffs.tidal1 = &tidal1;
  coeffs.tidal2 = &tidal2;
  FacWaveformCoeffs hcoeffs = {0};
  hcoeffs.tidal1 = &tidal1;
  hcoeffs.tidal2 = &tidal2;
  NewtonMultipolePrefixes prefixes = {0};
  EOBNonQCCoeffs nqc = {0};
  EOBParams eob = {0};
  eob.eta = nu;
  eob.m1 = x1;
  eob.m2 = x2;
  eob.hCoeffs = &hcoeffs;
  eob.nqcCoeffs = &nqc;
  eob.prefixes = &prefixes;
  double s1_data[3] = {0., 0., a1};
  double s2_data[3] = {0., 0., a2};
  double sk_data[3] = {0., 0., aK};
  double ss_data[3] = {0., 0., Sstar};
  REAL8Vector s1 = {3, s1_data}, s2 = {3, s2_data};
  REAL8Vector sk = {3, sk_data}, ss = {3, ss_data};
  SpinEOBParams params = {0};
  params.eobParams = &eob;
  params.seobCoeffs = &coeffs;
  params.nqcCoeffs = &nqc;
  params.s1Vec = &s1;
  params.s2Vec = &s2;
  params.sigmaKerr = &sk;
  params.sigmaStar = &ss;
  params.a = aK;
  params.chi1 = chi1;
  params.chi2 = chi2;
  params.alignedSpins = 1;

  if (XLALSimIMRCalculateSpinEOBHCoeffs(&coeffs, nu, aK, 4) != 0
      || XLALSimIMREOBCalcSpinFacWaveformCoefficients(
          &hcoeffs, &params, x1, x2, nu, aK, chiS, chiA, 4) != 0
      || XLALSimIMREOBComputeNewtonMultipolePrefixes(&prefixes, x1, x2) != 0)
    return 2;

  LALDict *dict = XLALCreateDict();
  if (!dict) return 3;
  int rc = XLALDictInsertREAL8Value(dict, "nu", nu)
      || XLALDictInsertREAL8Value(dict, "a1", a1)
      || XLALDictInsertREAL8Value(dict, "a2", a2)
      || XLALDictInsertREAL8Value(dict, "aK", aK)
      || XLALDictInsertREAL8Value(dict, "Sstar", Sstar)
      || XLALDictInsertUINT2Value(dict, "analyticFlag", 0);
  if (rc) {
    XLALDestroyDict(dict);
    return 4;
  }

  double x_data[3] = {r, 0., 0.};
  double p_data[3] = {0., pphi/r, 0.};
  REAL8Vector x = {3, x_data}, p = {3, p_data};
  double H = XLALSimIMRSpinEOBHamiltonian(
      nu, &x, &p, &s1, &s2, &sk, &ss, 0, &coeffs);
  double polar_data[4] = {r, 0., 0., pphi};
  REAL8Vector polar = {4, polar_data};
  params.tortoise = 0;
  out[0] = XLALInspiralSpinFactorizedFlux(
      &polar, &nqc, omega, &params, H, 8, 4)/nu;
  params.tortoise = 1;
  out[1] = -XLALSimInspiralEOBPAFluxWrapper(
      r, 0., pphi, omega, &params, &nqc, dict);
  out[2] = H/nu;
  params.tortoise = 0;
  RadialDerivativeContext ctx = {pphi, &params, 0};
  gsl_function F = {radial_derivative_term, &ctx};
  double abs_err, d2Hdr2, d2Hdrdpphi;
  if (gsl_deriv_central(&F, r, 1.e-4, &d2Hdr2, &abs_err) != GSL_SUCCESS) {
    XLALDestroyDict(dict);
    return 6;
  }
  ctx.derivative = 5;
  if (gsl_deriv_central(&F, r, 1.e-4, &d2Hdrdpphi, &abs_err) != GSL_SUCCESS) {
    XLALDestroyDict(dict);
    return 7;
  }
  double circular_values[6] = {r, 0., 0., 0., pphi/r, 0.};
  double dHdpphi = XLALSpinHcapExactDerivWRTParam(4, circular_values, &params)/r;
  double finite_values[6] = {r, 0., 0., 1.e-3, pphi/r, 0.};
  double dHdpr = XLALSpinHcapExactDerivWRTParam(3, finite_values, &params);
  double dEdr = -dHdpphi*d2Hdr2/d2Hdrdpphi;
  double deltaR = XLALSimIMRSpinEOBHamiltonianDeltaR(&coeffs, r, nu, fabs(aK));
  double deltaT = XLALSimIMRSpinEOBHamiltonianDeltaT(&coeffs, r, nu, fabs(aK));
  double csi = sqrt(deltaT*deltaR)/(r*r+aK*aK);
  out[3] = dEdr;
  out[4] = dHdpr;
  out[5] = (-out[0]/dEdr)/(dHdpr/1.e-3)*csi;
  out[6] = dHdpphi;
  out[7] = d2Hdr2;
  out[8] = d2Hdrdpphi;
  XLALDestroyDict(dict);
  for (int i = 0; i < 9; i++) if (!isfinite(out[i])) return 5;
  return 0;
}

/* Run one fresh source initial-condition path, without the same-state flux
 * wrapper's mutable EOB parameters leaking into this calculation. */
int diffgw_v4_initial_conditions(
    double m1, double m2, double chi1, double chi2,
    double omega, int optimized, double *out) {
  if (!out || m1 <= 0. || m2 <= 0. || !isfinite(chi1) || !isfinite(chi2)
      || !isfinite(omega) || omega <= 0. || (optimized != 0 && optimized != 1))
    return 1;
  double total = m1+m2;
  double x1 = m1/total, x2 = m2/total, nu = x1*x2;
  double a1 = x1*x1*chi1, a2 = x2*x2*chi2, aK = a1+a2;
  TidalEOBParams tidal1 = {0}, tidal2 = {0};
  tidal1.mByM = x1;
  tidal2.mByM = x2;
  tidal1.quadparam = tidal2.quadparam = 1.;
  SpinEOBHCoeffs coeffs = {0};
  coeffs.tidal1 = &tidal1;
  coeffs.tidal2 = &tidal2;
  FacWaveformCoeffs hcoeffs = {0};
  hcoeffs.tidal1 = &tidal1;
  hcoeffs.tidal2 = &tidal2;
  NewtonMultipolePrefixes prefixes = {0};
  EOBNonQCCoeffs nqc = {0};
  EOBParams eob = {0};
  eob.eta = nu;
  eob.m1 = m1;
  eob.m2 = m2;
  eob.hCoeffs = &hcoeffs;
  eob.nqcCoeffs = &nqc;
  eob.prefixes = &prefixes;
  double s1_data[3] = {0., 0., a1};
  double s2_data[3] = {0., 0., a2};
  double sk_data[3] = {0., 0., aK};
  double ss_data[3] = {0., 0., nu*(chi1+chi2)};
  REAL8Vector s1 = {3, s1_data}, s2 = {3, s2_data};
  REAL8Vector sk = {3, sk_data}, ss = {3, ss_data};
  SpinEOBParams params = {0};
  params.eobParams = &eob;
  params.seobCoeffs = &coeffs;
  params.nqcCoeffs = &nqc;
  params.s1Vec = &s1;
  params.s2Vec = &s2;
  params.sigmaKerr = &sk;
  params.sigmaStar = &ss;
  params.a = aK;
  params.chi1 = chi1;
  params.chi2 = chi2;
  params.alignedSpins = 1;
  params.tortoise = 1;
  if (XLALSimIMRCalculateSpinEOBHCoeffs(&coeffs, nu, aK, 4) != 0
      || XLALSimIMREOBCalcSpinFacWaveformCoefficients(
          &hcoeffs, &params, m1, m2, nu, aK, (chi1+chi2)/2.,
          (chi1-chi2)/2., 4) != 0
      || XLALSimIMREOBComputeNewtonMultipolePrefixes(
          &prefixes, m1, m2) != 0)
    return 2;
  double spin1[3] = {0., 0., m1*m1*chi1};
  double spin2[3] = {0., 0., m2*m2*chi2};
  double ic_data[14] = {0.};
  REAL8Vector ic = {14, ic_data};
  double start_hz = omega/(LAL_PI*LAL_MTSUN_SI*total);
  if (XLALSimIMRSpinEOBInitialConditions(
          &ic, m1, m2, start_hz, 0., spin1, spin2,
          &params, optimized) != XLAL_SUCCESS)
    return 3;
  out[0] = ic_data[0];
  out[1] = ic_data[3];
  out[2] = ic_data[0]*ic_data[4];
  for (int i = 0; i < 3; i++) if (!isfinite(out[i])) return 4;
  return 0;
}

/* Compare the source's private pre-NQC (2,2) mode and evolution flux
 * at an explicit state.  These are source-component oracles, not exported
 * wheel APIs.  out = {Re(h22), Im(h22), Hreal/M, flux/nu,
 *                    35 per-mode fluxes/nu in increasing (ell,m) order}. */
int diffgw_v4_mode22_same_state(
    double x1, double x2, double chi1, double chi2,
    double r, double phi, double pr_star, double pphi,
    double omega, double *out) {
  if (!out || x1 <= 0. || x2 <= 0. || fabs(x1+x2-1.) > 1.e-12
      || !isfinite(chi1) || !isfinite(chi2) || r <= 0.
      || !isfinite(phi) || !isfinite(pr_star)
      || !isfinite(pphi) || pphi <= 0.
      || !isfinite(omega) || omega <= 0.) return 1;
  double nu = x1*x2, a1 = x1*x1*chi1, a2 = x2*x2*chi2;
  double aK = a1+a2, Sstar = nu*(chi1+chi2);
  TidalEOBParams tidal1 = {0}, tidal2 = {0};
  tidal1.mByM = x1;
  tidal2.mByM = x2;
  tidal1.quadparam = tidal2.quadparam = 1.;
  SpinEOBHCoeffs coeffs = {0};
  coeffs.tidal1 = &tidal1;
  coeffs.tidal2 = &tidal2;
  FacWaveformCoeffs hcoeffs = {0};
  hcoeffs.tidal1 = &tidal1;
  hcoeffs.tidal2 = &tidal2;
  NewtonMultipolePrefixes prefixes = {0};
  EOBNonQCCoeffs nqc = {0};
  EOBParams eob = {0};
  eob.eta = nu;
  eob.m1 = x1;
  eob.m2 = x2;
  eob.hCoeffs = &hcoeffs;
  eob.nqcCoeffs = &nqc;
  eob.prefixes = &prefixes;
  double s1_data[3] = {0., 0., a1};
  double s2_data[3] = {0., 0., a2};
  double sk_data[3] = {0., 0., aK};
  double ss_data[3] = {0., 0., Sstar};
  REAL8Vector s1 = {3, s1_data}, s2 = {3, s2_data};
  REAL8Vector sk = {3, sk_data}, ss = {3, ss_data};
  SpinEOBParams params = {0};
  params.eobParams = &eob;
  params.seobCoeffs = &coeffs;
  params.nqcCoeffs = &nqc;
  params.s1Vec = &s1;
  params.s2Vec = &s2;
  params.sigmaKerr = &sk;
  params.sigmaStar = &ss;
  params.a = aK;
  params.chi1 = chi1;
  params.chi2 = chi2;
  params.alignedSpins = 1;
  params.tortoise = 1;
  if (XLALSimIMRCalculateSpinEOBHCoeffs(&coeffs, nu, aK, 4) != 0
      || XLALSimIMREOBCalcSpinFacWaveformCoefficients(
          &hcoeffs, &params, x1, x2, nu, aK,
          (chi1+chi2)/2., (chi1-chi2)/2., 4) != 0
      || XLALSimIMREOBComputeNewtonMultipolePrefixes(
          &prefixes, x1, x2) != 0) return 2;
  double x_data[3] = {r, 0., 0.};
  double p_data[3] = {pr_star, pphi/r, 0.};
  REAL8Vector x = {3, x_data}, p = {3, p_data};
  double Hreal = XLALSimIMRSpinEOBHamiltonian(
      nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs);
  double values_data[4] = {r, phi, pr_star, pphi};
  REAL8Vector values = {4, values_data};
  COMPLEX16 mode;
  if (XLALSimIMRSpinEOBGetSpinFactorizedWaveform(
          &mode, &values, cbrt(omega), Hreal, 2, 2, &params, 0)
      != XLAL_SUCCESS) return 3;
  out[0] = creal(mode);
  out[1] = cimag(mode);
  out[2] = Hreal;
  out[3] = XLALInspiralSpinFactorizedFlux(
      &values, &nqc, omega, &params, Hreal, 8, 4)/nu;
  int index = 4;
  for (int ell = 2; ell <= 8; ell++) {
    for (int m = 1; m <= ell; m++) {
      COMPLEX16 h;
      if (XLALSimIMRSpinEOBFluxGetSpinFactorizedWaveform(
              &h, &values, cbrt(omega), Hreal, ell, m, &params, 0, NULL)
          != XLAL_SUCCESS) return 5;
      if (ell == 2 && m == 2) {
        COMPLEX16 correction;
        if (XLALSimIMREOBNonQCCorrection(&correction, &values, omega, &nqc)
            != XLAL_SUCCESS) return 6;
        h *= correction;
      }
      out[index++] = m*m*omega*omega*(creal(h)*creal(h)+cimag(h)*cimag(h))
          *LAL_1_PI/(8.*nu);
    }
  }
  for (int i = 0; i < index; i++) if (!isfinite(out[i])) return 4;
  return 0;
}
