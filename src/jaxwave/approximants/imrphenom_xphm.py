"""IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Frequency-Domain Approximant in JAX.

Mathematical references for the inherited model and audited corrections:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
- Hamilton et al. (2021), Phys. Rev. D 104, 124027 (arXiv:2107.08876)
- Colleoni et al. (2025), Phys. Rev. D 111, 104019 (arXiv:2412.16721)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    MSUN_SEC,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
)
from jaxwave.approximants.imrphenom_xhm import IMRPhenomXHM
from jaxwave.approximants.imrphenom_xas import XASParameterFits
from jaxwave.approximants.imrphenom_xp import wigner_small_d_matrix
from jaxwave.approximants.precession_msa import (
    compute_effective_chip,
    compute_msa_final_spin,
    evaluate_msa_precession,
    resolve_precession_version,
)


@jax.tree_util.register_pytree_node_class
class IMRPhenomXPHM:
    """IMRPhenomXPHM Precessing Multipolar Higher-Harmonic Waveform Approximant in JAX.

    Twists up underlying coprecessing multipolar modes from IMRPhenomXHM
    using MSA, NNLO, SpinTaylor, or PNR precession angles and the full
    Wigner D-matrix frame rotation.
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

    def __new__(cls, *args, **kwargs):
        # Allow direct functional evaluation if waveform frequencies are provided
        if "f" in kwargs or "sample_frequencies" in kwargs:
            instance = super().__new__(cls)
            init_kw = {}
            for k in ("modes", "calibration", "precession_method", "prec_version", "convention", "final_spin_mod"):
                if k in kwargs:
                    init_kw[k] = kwargs.pop(k)
            instance.__init__(**init_kw)
            return instance.evaluate(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        calibration: str = "sovereign",
        precession_method: str = "msa",
        prec_version: Optional[int] = None,
        convention: int = 1,
        final_spin_mod: Optional[int] = None,
    ):
        self.calibration = calibration.lower()
        self.default_precession_method = precession_method
        self.default_prec_version = prec_version
        self.default_convention = convention
        self.default_final_spin_mod = final_spin_mod
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.xhm_model = IMRPhenomXHM(modes=self.modes, calibration=self.calibration)

    def tree_flatten(self):
        aux_data = (
            self.modes,
            self.calibration,
            self.default_precession_method,
            self.default_prec_version,
            self.default_convention,
            self.default_final_spin_mod,
        )
        return ((), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        modes, calibration, prec_method, prec_ver, conv, final_spin = aux_data
        return cls(
            modes=modes,
            calibration=calibration,
            precession_method=prec_method,
            prec_version=prec_ver,
            convention=conv,
            final_spin_mod=final_spin,
        )

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("modes", self.modes)
        kwargs.setdefault("calibration", self.calibration)
        kwargs.setdefault("precession_method", self.default_precession_method)
        kwargs.setdefault("prec_version", self.default_prec_version)
        kwargs.setdefault("convention", self.default_convention)
        kwargs.setdefault("final_spin_mod", self.default_final_spin_mod)
        return self.evaluate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    @classmethod
    def evaluate(
        cls,
        f: Optional[jnp.ndarray] = None,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        spin1x: Union[float, jnp.ndarray] = 0.0,
        spin1y: Union[float, jnp.ndarray] = 0.0,
        spin1z: Optional[Union[float, jnp.ndarray]] = None,
        spin2x: Union[float, jnp.ndarray] = 0.0,
        spin2y: Union[float, jnp.ndarray] = 0.0,
        spin2z: Optional[Union[float, jnp.ndarray]] = None,
        chi1: Optional[Union[float, jnp.ndarray]] = None,
        chi2: Optional[Union[float, jnp.ndarray]] = None,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phi0: Union[float, jnp.ndarray] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        precession_method: Optional[str] = None,
        prec_version: Optional[int] = None,
        convention: Optional[int] = None,
        final_spin_mod: Optional[int] = None,
        calibration: str = "sovereign",
        **kwargs,
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray, Dict[Tuple[int, int], Tuple[jnp.ndarray, jnp.ndarray]]],
    ]:
        """Generate frequency-domain precessing multipolar polarizations (h_plus, h_cross) in JAX."""
        if precession_method is None:
            precession_method = "msa"
        if convention is None:
            convention = 1

        p_ver = None if precession_method.lower() == "ode" else resolve_precession_version(precession_method, prec_version)
        if isinstance(final_spin_mod, bool) or final_spin_mod not in (None, 0, 2, 3):
            raise ValueError("final_spin_mod must be None, 0 (NNLO), 2 (Vectorial), or 3 (MSA).")
        if precession_method.lower() == "ode" and prec_version is not None:
            raise ValueError("prec_version does not apply to precession_method='ode'.")
        if isinstance(convention, bool) or convention not in (0, 1, 5, 6, 7):
            raise ValueError(f"Unsupported precession convention: {convention}.")
        if precession_method.lower() == "ode" and convention != 1:
            raise ValueError("Table IV conventions apply to NNLO/MSA analytical angles.")

        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        dtype = jnp.float64
        complex_dtype = jnp.complex128
        freqs = as_array(freqs, dtype=dtype)

        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1x = as_array(spin1x, dtype=dtype)
        s1y = as_array(spin1y, dtype=dtype)
        s2x = as_array(spin2x, dtype=dtype)
        s2y = as_array(spin2y, dtype=dtype)

        s1z_val = spin1z if spin1z is not None else (chi1 if chi1 is not None else 0.0)
        s2z_val = spin2z if spin2z is not None else (chi2 if chi2 is not None else 0.0)
        s1z = as_array(s1z_val, dtype=dtype)
        s2z = as_array(s2z_val, dtype=dtype)

        dist = as_array(distance, dtype=dtype)
        inc = as_array(inclination, dtype=dtype)
        phi_c = as_array(kwargs.pop("phi_c", phic), dtype=dtype)
        t_c = as_array(kwargs.pop("t_c", tc), dtype=dtype)
        phi_0 = as_array(phi0, dtype=dtype)
        fref = as_array(f_ref, dtype=dtype) if f_ref is not None else None
        fref_eval = fref if fref is not None else jnp.asarray(20.0, dtype=dtype)

        (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
         dist, inc, phi_c, t_c, phi_0, fref_eval) = jnp.broadcast_arrays(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z,
            dist, inc, phi_c, t_c, phi_0, fref_eval,
        )
        if fref is not None:
            fref = fref_eval

        # Canonical body ordering: primary mass m1 >= m2
        swap_mask = m1 < m2
        m1_eff = jnp.where(swap_mask, m2, m1)
        m2_eff = jnp.where(swap_mask, m1, m2)
        s1x_eff = jnp.where(swap_mask, s2x, s1x)
        s1y_eff = jnp.where(swap_mask, s2y, s1y)
        s1z_eff = jnp.where(swap_mask, s2z, s1z)
        s2x_eff = jnp.where(swap_mask, s1x, s2x)
        s2y_eff = jnp.where(swap_mask, s1y, s2y)
        s2z_eff = jnp.where(swap_mask, s1z, s2z)
        m1, m2 = m1_eff, m2_eff
        s1x, s1y, s1z = s1x_eff, s1y_eff, s1z_eff
        s2x, s2y, s2z = s2x_eff, s2y_eff, s2z_eff

        active_modes = tuple(modes) if modes is not None else cls.SUPPORTED_MODES
        for lm in active_modes:
            if lm not in cls.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {cls.SUPPORTED_MODES}")

        # In-plane spin check
        in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
        is_aligned = (in_plane_sq == 0.0)

        # Fast path: if purely aligned scalar concrete, directly return IMRPhenomXHM
        if (
            convention == 1
            and not isinstance(is_aligned, jax.core.Tracer)
            and is_aligned.ndim == 0
            and bool(is_aligned)
        ):
            return IMRPhenomXHM.evaluate(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_ref=fref,
                phi0=phi_0,
                modes=active_modes,
                return_modes=return_modes,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                calibration=calibration,
            )

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        m_sec = total_mass * MSUN_SEC

        final_spin_override = None
        if final_spin_mod == 3:
            v0_remnant = (PI * m_sec * fref_eval) ** (1.0 / 3.0)
            final_spin_override = compute_msa_final_spin(
                v0_remnant, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
            )
        elif final_spin_mod == 2 or (
            final_spin_mod is None
            and (
                precession_method.lower() in ("spintaylor", "st", "spintaylort4", "pnr", "xpnr", "imrphenomxpnr")
                or p_ver in (300, 310, 311, 320, 321, 330)
            )
        ):
            from jaxwave.approximants.precession_spintaylor import compute_chi_tot_perp

            chi_tot_perp = compute_chi_tot_perp(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chi_tot_perp)
        elif final_spin_mod == 0 or p_ver in (101, 102, 103, 104):
            chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
            final_spin_override = XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)

        is_batched = total_mass.ndim >= 1
        f_eval = freqs if (not is_batched or freqs.ndim > 1) else freqs[None, :]

        def unsqueeze_param(t: jnp.ndarray) -> jnp.ndarray:
            if is_batched and f_eval.ndim > t.ndim:
                return jnp.expand_dims(t, axis=-1)
            return t

        # Precession angles evaluation
        if precession_method.lower() == "ode":
            from jaxwave.approximants.precession_ode import SpinPrecessionODE

            alpha, beta, epsilon = SpinPrecessionODE.compute_angles_frequency(
                f=jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)),
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                f_ref=fref_eval,
            )
            gamma = -epsilon
            theta_jn = inc
            zeta = jnp.zeros_like(theta_jn)
            carrier_phase = unsqueeze_param(jnp.zeros_like(phi_c))
        elif precession_method.lower() in ("pnr", "xpnr", "imrphenomxpnr") or p_ver in (300, 330):
            from jaxwave.approximants.precession_pnr import evaluate_pnr_precession

            alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_pnr_precession(
                f=jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)),
                f_ref=fref_eval,
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                inclination=inc,
                phi_ref=phi_c,
                convention=convention,
                observer_azimuth=phi_0,
            )
        elif precession_method.lower() in ("spintaylor", "st", "spintaylort4") or p_ver in (310, 311, 320, 321):
            from jaxwave.approximants.precession_spintaylor import evaluate_spintaylor_precession

            alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_spintaylor_precession(
                f=jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)),
                f_ref=fref_eval,
                mass1=m1,
                mass2=m2,
                spin1x=s1x,
                spin1y=s1y,
                spin1z=s1z,
                spin2x=s2x,
                spin2y=s2y,
                spin2z=s2z,
                inclination=inc,
                phi_ref=phi_c,
                convention=convention,
                observer_azimuth=phi_0,
            )
        else:
            f_angles = jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval))
            v = (PI * unsqueeze_param(m_sec) * f_angles) ** (1.0 / 3.0)
            v0 = (PI * unsqueeze_param(m_sec) * unsqueeze_param(fref_eval)) ** (1.0 / 3.0)

            alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_msa_precession(
                v=v,
                v0=v0,
                m1=unsqueeze_param(m1),
                m2=unsqueeze_param(m2),
                s1x=unsqueeze_param(s1x),
                s1y=unsqueeze_param(s1y),
                s1z=unsqueeze_param(s1z),
                s2x=unsqueeze_param(s2x),
                s2y=unsqueeze_param(s2y),
                s2z=unsqueeze_param(s2z),
                iota=unsqueeze_param(inc),
                phi_ref=unsqueeze_param(phi_c),
                prec_version=p_ver,
                convention=convention,
                return_carrier_phase=True,
                observer_azimuth=unsqueeze_param(phi_0),
            )

        # Coprecessing modes receive precisely the Table IV phiRef argument.
        carrier_phi_c = carrier_phase.squeeze(-1) if is_batched else carrier_phase
        _, _, coprec_dict = IMRPhenomXHM.evaluate(
            f=freqs,
            mass1=m1,
            mass2=m2,
            chi1=s1z,
            chi2=s2z,
            distance=dist,
            inclination=jnp.zeros_like(inc),
            phic=carrier_phi_c,
            tc=t_c,
            f_ref=fref,
            phi0=jnp.zeros_like(phi_0),
            modes=active_modes,
            return_unprojected_modes=True,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            calibration=calibration,
            _final_spin_override=final_spin_override,
        )

        h_plus_total = jnp.zeros(f_eval.shape, dtype=complex_dtype)
        h_cross_total = jnp.zeros(f_eval.shape, dtype=complex_dtype)
        mode_dict = {}

        unique_l = sorted(list(set(l for l, _ in active_modes)))
        angles_by_mp = {2: (alpha, beta, gamma)}
        other_mps = {mp for _, mp in active_modes} - {2}

        if precession_method.lower() in ("pnr", "xpnr", "imrphenomxpnr") or p_ver in (300, 330):
            from jaxwave.approximants.precession_pnr import evaluate_pnr_precession

            for mp in other_mps:
                f_mp = jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)) * (2.0 / float(mp))
                a_mp, b_mp, g_mp, _, _, _ = evaluate_pnr_precession(
                    f=f_mp,
                    f_ref=fref_eval,
                    mass1=m1,
                    mass2=m2,
                    spin1x=s1x,
                    spin1y=s1y,
                    spin1z=s1z,
                    spin2x=s2x,
                    spin2y=s2y,
                    spin2z=s2z,
                    inclination=inc,
                    phi_ref=phi_c,
                    convention=convention,
                    observer_azimuth=phi_0,
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)
        elif precession_method.lower() in ("spintaylor", "st", "spintaylort4") or p_ver in (310, 311, 320, 321):
            from jaxwave.approximants.precession_spintaylor import evaluate_spintaylor_precession

            for mp in other_mps:
                f_mp = jnp.where(f_eval > 0, f_eval, unsqueeze_param(fref_eval)) * (2.0 / float(mp))
                a_mp, b_mp, g_mp, _, _, _ = evaluate_spintaylor_precession(
                    f=f_mp,
                    f_ref=fref_eval,
                    mass1=m1,
                    mass2=m2,
                    spin1x=s1x,
                    spin1y=s1y,
                    spin1z=s1z,
                    spin2x=s2x,
                    spin2y=s2y,
                    spin2z=s2z,
                    inclination=inc,
                    phi_ref=phi_c,
                    convention=convention,
                    observer_azimuth=phi_0,
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)
        elif precession_method.lower() != "ode":
            for mp in other_mps:
                a_mp, b_mp, g_mp, _, _ = evaluate_msa_precession(
                    v=v * (2.0 / mp) ** (1.0 / 3.0),
                    v0=v0,
                    m1=unsqueeze_param(m1),
                    m2=unsqueeze_param(m2),
                    s1x=unsqueeze_param(s1x),
                    s1y=unsqueeze_param(s1y),
                    s1z=unsqueeze_param(s1z),
                    s2x=unsqueeze_param(s2x),
                    s2y=unsqueeze_param(s2y),
                    s2z=unsqueeze_param(s2z),
                    iota=unsqueeze_param(inc),
                    phi_ref=unsqueeze_param(phi_c),
                    prec_version=p_ver,
                    convention=convention,
                    observer_azimuth=unsqueeze_param(phi_0),
                )
                angles_by_mp[mp] = (a_mp, b_mp, g_mp)

        d_mat_jn = {l: wigner_small_d_matrix(l, unsqueeze_param(theta_jn)) for l in unique_l}

        norm_mode = math.sqrt(16.0 * PI / 5.0)
        two_zeta = 2.0 * unsqueeze_param(zeta)
        cos_2z = jnp.cos(two_zeta)
        sin_2z = jnp.sin(two_zeta)

        for l, mp in active_modes:
            # C4 uses source azimuth pi/2-phi_ref, while the carrier modes use
            # zero azimuth. The mode-dependent conversion is exp(i*mp*pi/2).
            # A blanket minus accounts only for mp=2.
            h_mode = (1j ** mp) * coprec_dict[(l, mp)]
            alpha, beta, gamma = angles_by_mp.get(mp, angles_by_mp[2])

            scale_l = math.sqrt((2 * l + 1) / 5.0) / norm_mode

            d_beta_l = wigner_small_d_matrix(l, beta)
            d_jn_l = d_mat_jn[l]

            term_plus = jnp.zeros(f_eval.shape, dtype=complex_dtype)
            term_cross = jnp.zeros(f_eval.shape, dtype=complex_dtype)

            for m_val in range(-l, l + 1):
                m_idx = m_val + l
                dm_minus = d_beta_l[..., m_idx, -mp + l]
                dm_plus = d_beta_l[..., m_idx, mp + l]

                y_lm = d_jn_l[..., m_idx, 2 + l]

                m_phase = float(m_val) * (-alpha)
                cos_m = jnp.cos(m_phase)
                sin_m = jnp.sin(m_phase)

                exp_minus_ima = cos_m + 1j * sin_m
                exp_plus_ima = cos_m - 1j * sin_m

                a_m_minus = exp_minus_ima * dm_minus * y_lm
                a_m_plus_star = exp_plus_ima * dm_plus * y_lm

                term_plus = term_plus + (a_m_minus + ((-1) ** l) * a_m_plus_star)
                term_cross = term_cross + (a_m_minus - ((-1) ** l) * a_m_plus_star)

            # Phase factor exp(-i * mp' * gamma) for mp' = mp
            mp_gamma = float(mp) * gamma
            exp_neg_im_gamma = jnp.cos(mp_gamma) + 1j * jnp.sin(mp_gamma)

            hp_raw = scale_l * exp_neg_im_gamma * h_mode * term_plus
            hc_raw = 1j * scale_l * exp_neg_im_gamma * h_mode * term_cross

            # Rotate from the spherical projection basis to the observer basis.
            hp_lm = cos_2z * hp_raw + sin_2z * hc_raw
            hc_lm = -sin_2z * hp_raw + cos_2z * hc_raw

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, mp)] = (hp_lm, hc_lm)

        # If aligned-spin limit, blend with exact IMRPhenomXHM
        if convention == 1:
            out_al = IMRPhenomXHM.evaluate(
                f=freqs,
                mass1=m1,
                mass2=m2,
                chi1=s1z,
                chi2=s2z,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_ref=fref,
                phi0=phi_0,
                modes=active_modes,
                return_modes=return_modes,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                calibration=calibration,
            )
            is_al_u = unsqueeze_param(is_aligned)
            if return_modes:
                hp_al, hc_al, dict_al = out_al
                h_plus_total = jnp.where(is_al_u, hp_al, h_plus_total)
                h_cross_total = jnp.where(is_al_u, hc_al, h_cross_total)
                for k in active_modes:
                    hp_m, hc_m = mode_dict[k]
                    hp_m_al, hc_m_al = dict_al[k]
                    mode_dict[k] = (
                        jnp.where(is_al_u, hp_m_al, hp_m),
                        jnp.where(is_al_u, hc_m_al, hc_m),
                    )
            else:
                hp_al, hc_al = out_al
                h_plus_total = jnp.where(is_al_u, hp_al, h_plus_total)
                h_cross_total = jnp.where(is_al_u, hc_al, h_cross_total)

        # Zero out invalid frequencies (f <= 0)
        h_plus_total = jnp.where(f_eval > 0.0, h_plus_total, jnp.zeros_like(h_plus_total))
        h_cross_total = jnp.where(f_eval > 0.0, h_cross_total, jnp.zeros_like(h_cross_total))

        if not is_batched and h_plus_total.ndim > freqs.ndim:
            h_plus_total = jnp.squeeze(h_plus_total, 0)
            h_cross_total = jnp.squeeze(h_cross_total, 0)
            if return_modes:
                for k in active_modes:
                    hp_m, hc_m = mode_dict[k]
                    mode_dict[k] = (jnp.squeeze(hp_m, 0), jnp.squeeze(hc_m, 0))

        if return_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total


__all__ = [
    "IMRPhenomXPHM",
]
