/* Inserted only into a private copy of the checked dev mode adapter.
 * Slots 39:47 expose the pinned source's non-Kepler/Newtonian path. */
  double circ_values[4] = {r, phi, 0., pphi};
  double omega_circ = XLALSimIMRSpinAlignedEOBCalcOmega(
      circ_values, &params, 1.e-4);
  double non_kepler = XLALSimIMRSpinAlignedEOBNonKeplerCoeff(
      values_data, &params);
  double v_phi = r*cbrt(non_kepler)*omega;
  COMPLEX16 newton;
  if (XLALSimIMRSpinEOBCalculateNewtonianMultipole(
          &newton, v_phi*v_phi, r, phi, 2, 2, &eob) != XLAL_SUCCESS)
    return 7;
  double cart_values[6] = {r, 0., 0., 0., pphi/r, 0.};
  double exact_omega_circ = XLALSpinHcapExactDerivWRTParam(
      4, cart_values, &params)/r;
  HcapDerivParams deriv = {0};
  deriv.values = cart_values;
  deriv.params = &params;
  deriv.varyParam = 4;
  double hstep = 1.e-4;
  double hplus = GSLSpinAlignedHamiltonianWrapper(cart_values[4]+hstep, &deriv);
  double hminus = GSLSpinAlignedHamiltonianWrapper(cart_values[4]-hstep, &deriv);
  double hhalf_plus = GSLSpinAlignedHamiltonianWrapper(
      cart_values[4]+hstep/2., &deriv);
  double hhalf_minus = GSLSpinAlignedHamiltonianWrapper(
      cart_values[4]-hstep/2., &deriv);
  out[39] = omega_circ;
  out[40] = exact_omega_circ;
  out[41] = non_kepler;
  out[42] = v_phi*v_phi;
  out[43] = cabs(newton);
  out[44] = cabs(mode)/cabs(newton);
  out[45] = hplus-hminus;
  out[46] = hhalf_plus-hhalf_minus;
