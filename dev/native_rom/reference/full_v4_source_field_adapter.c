/* Development-only explicit-state regular v4 vector field. Source is
 * supplied from a separately pinned LAL snapshot at compile time. */
#include <math.h>
#include <lal/LALConstants.h>
#include "LALSimIMRSpinAlignedEOB.c"

/* out = {dr/dt, dphi/dt, d(pr_star)/dt, dpphi/dt, regular flux/nu,
 * torque-implied flux/nu}. The latter follows from -omega*dpphi/dt. */
int diffgw_v4_source_field(
    double m1, double m2, double chi1, double chi2,
    double r, double phi, double pr_star, double pphi, double *out) {
  if (!out || !isfinite(m1) || !isfinite(m2) || m1 <= 0. || m2 <= 0.
      || !isfinite(chi1) || !isfinite(chi2) || !isfinite(r) || r <= 0.
      || !isfinite(phi) || !isfinite(pr_star) || !isfinite(pphi) || pphi <= 0.)
    return 1;
  const double total = m1+m2, x1 = m1/total, x2 = m2/total;
  const double nu = x1*x2, a1 = x1*x1*chi1, a2 = x2*x2*chi2;
  const double aK = a1+a2, Sstar = nu*(chi1+chi2);
  const double chiS = 0.5*(chi1+chi2), chiA = 0.5*(chi1-chi2);
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
  double s1_data[3] = {0., 0., a1}, s2_data[3] = {0., 0., a2};
  double sk_data[3] = {0., 0., aK}, ss_data[3] = {0., 0., Sstar};
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
          &hcoeffs, &params, m1, m2, nu, aK, chiS, chiA, 4) != 0
      || XLALSimIMREOBComputeNewtonMultipolePrefixes(
          &prefixes, m1, m2) != 0) return 2;
  const double state[4] = {r, phi, pr_star, pphi};
  if (XLALSpinAlignedHcapDerivative(0., state, out, &params)
      != XLAL_SUCCESS) return 3;
  double x_data[3] = {r, 0., 0.};
  double p_data[3] = {pr_star, pphi/r, 0.};
  REAL8Vector x = {3, x_data}, p = {3, p_data};
  double H = XLALSimIMRSpinEOBHamiltonianOptimized(
      nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs);
  double polar_data[4] = {r, phi, pr_star, pphi};
  REAL8Vector polar = {4, polar_data};
  out[4] = XLALInspiralSpinFactorizedFlux(
      &polar, &nqc, out[1], &params, H, 8, 4)/nu;
  out[5] = -out[1]*out[3];
  for (int i = 0; i < 6; i++) if (!isfinite(out[i])) return 4;
  return 0;
}
