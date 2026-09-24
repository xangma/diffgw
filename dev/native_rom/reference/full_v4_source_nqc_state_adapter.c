/* Development-only explicit-state regular v4 NQC input probe. Source is
 * supplied from a separately pinned LAL snapshot at compile time. */
#include <complex.h>
#include <math.h>
#include <lal/LALConstants.h>
#include "LALSimIMRSpinAlignedEOB.c"

/* out = {omega, Re(raw h22), Im(raw h22)} using precisely the source
 * generator's regular-v4 orbital-frequency and raw-mode calls. */
int diffgw_v4_source_nqc_state(
    double m1, double m2, double chi1, double chi2,
    double r, double phi, double pr_star, double pphi, double *out) {
  if (!out || !isfinite(m1) || !isfinite(m2) || m1 <= 0. || m2 <= 0.
      || !isfinite(chi1) || !isfinite(chi2) || !isfinite(r) || r <= 0.
      || !isfinite(phi) || !isfinite(pr_star) || !isfinite(pphi) || pphi <= 0.)
    return 1;
  const double m1_si = m1*LAL_MSUN_SI, m2_si = m2*LAL_MSUN_SI;
  m1 = m1_si/LAL_MSUN_SI;
  m2 = m2_si/LAL_MSUN_SI;
  const double total = m1+m2;
  const double nu = m1*m2/(total*total);
  const double chiS = 0.5*(chi1+chi2), chiA = 0.5*(chi1-chi2);
  const double tplspin = (1.-2.*nu)*chiS+(m1-m2)/total*chiA;
  TidalEOBParams tidal1 = {0}, tidal2 = {0};
  tidal1.mByM = m1_si/(m1_si+m2_si);
  tidal2.mByM = m2_si/(m1_si+m2_si);
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
  double s1_physical[3] = {0., 0., chi1*m1*m1};
  double s2_physical[3] = {0., 0., chi2*m2*m2};
  double s1_data[3] = {0., 0., s1_physical[2]/total/total};
  double s2_data[3] = {0., 0., s2_physical[2]/total/total};
  double sk_data[3] = {0.}, ss_data[3] = {0.};
  REAL8Vector s1_physical_vec = {3, s1_physical};
  REAL8Vector s2_physical_vec = {3, s2_physical};
  REAL8Vector s1 = {3, s1_data}, s2 = {3, s2_data};
  REAL8Vector sk = {3, sk_data}, ss = {3, ss_data};
  if (XLALSimIMRSpinEOBCalculateSigmaStar(
          &ss, m1, m2, &s1_physical_vec, &s2_physical_vec) != XLAL_SUCCESS
      || XLALSimIMRSpinEOBCalculateSigmaKerr(
          &sk, m1, m2, &s1_physical_vec, &s2_physical_vec) != XLAL_SUCCESS)
    return 2;
  const double aK = sk_data[2];
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
          &hcoeffs, &params, m1, m2, nu, tplspin, chiS, chiA, 4) != 0
      || XLALSimIMREOBComputeNewtonMultipolePrefixes(
          &prefixes, m1, m2) != 0) return 2;
  double state[4] = {r, phi, pr_star, pphi};
  out[0] = XLALSimIMRSpinAlignedEOBCalcOmega(
      state, &params, STEP_SIZE_CALCOMEGA);
  if (!isfinite(out[0]) || out[0] <= 0.) return 3;
  double x_data[3] = {r, 0., 0.};
  double p_data[3] = {pr_star, pphi/r, 0.};
  REAL8Vector x = {3, x_data}, p = {3, p_data};
  double H = XLALSimIMRSpinEOBHamiltonian(
      nu, &x, &p, &s1, &s2, &sk, &ss, 1, &coeffs);
  REAL8Vector values = {4, state};
  COMPLEX16 mode;
  if (XLALSimIMRSpinEOBGetSpinFactorizedWaveform(
          &mode, &values, cbrt(out[0]), H, 2, 2, &params, 0)
      != XLAL_SUCCESS) return 4;
  out[1] = creal(mode);
  out[2] = cimag(mode);
  for (int i = 0; i < 3; i++) if (!isfinite(out[i])) return 5;
  return 0;
}
