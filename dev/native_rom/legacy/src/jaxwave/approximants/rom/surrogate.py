"""Frequency-Domain Reduced-Order Model (ROM / Surrogate) Waveform Engine in JAX.

Port of torchwave.approximants.rom.surrogate for jaxwave.
Strict Clean-Room Implementation based on published peer-reviewed literature:
- Pürrer (2014), Class. Quantum Grav. 31, 195010 (arXiv:1402.4146)
- Pürrer (2016), Phys. Rev. D 93, 064041 (arXiv:1512.02248)
- Cotesta et al. (2018), Phys. Rev. D 98, 084028 (arXiv:1803.10701)
- Gadre et al. (2022), Phys. Rev. D 110, 124038 (arXiv:2203.00381)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    spin_combinations,
)


def poly_features(
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: Optional[jnp.ndarray] = None,
    degree: int = 3,
) -> jnp.ndarray:
    """Compute polynomial expansion features up to given degree in JAX."""
    eta_flat = jnp.asarray(eta).reshape(-1)
    chi1_flat = jnp.asarray(chi1).reshape(-1)
    cols = [jnp.ones_like(eta_flat)]
    if chi2 is None:
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                cols.append((eta_flat ** i) * (chi1_flat ** j))
    else:
        chi2_flat = jnp.asarray(chi2).reshape(-1)
        for d in range(1, degree + 1):
            for i in range(d + 1):
                for j in range(d - i + 1):
                    k = d - i - j
                    cols.append((eta_flat ** i) * (chi1_flat ** j) * (chi2_flat ** k))
    return jnp.stack(cols, axis=-1)


def chebyshev_features(
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: Optional[jnp.ndarray] = None,
    degree: int = 3,
) -> jnp.ndarray:
    """Compute tensor-product Chebyshev polynomial features in JAX."""
    eta_flat = jnp.asarray(eta).reshape(-1)
    chi1_flat = jnp.asarray(chi1).reshape(-1)

    eta_norm = jnp.clip(2.0 * (eta_flat - 0.03) / (0.25 - 0.03) - 1.0, -1.0, 1.0)
    chi1_norm = jnp.clip(chi1_flat, -1.0, 1.0)

    def cheb_1d(x: jnp.ndarray, deg: int) -> list:
        polys = [jnp.ones_like(x)]
        if deg >= 1:
            polys.append(x)
        for d in range(2, deg + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])
        return polys

    t_eta = cheb_1d(eta_norm, degree)
    t_chi1 = cheb_1d(chi1_norm, degree)

    cols = []
    if chi2 is None:
        for d in range(degree + 1):
            for i in range(d + 1):
                j = d - i
                cols.append(t_eta[i] * t_chi1[j])
    else:
        chi2_flat = jnp.asarray(chi2).reshape(-1)
        chi2_norm = jnp.clip(chi2_flat, -1.0, 1.0)
        t_chi2 = cheb_1d(chi2_norm, degree)
        for d in range(degree + 1):
            for i in range(d + 1):
                for j in range(d - i + 1):
                    k = d - i - j
                    cols.append(t_eta[i] * t_chi1[j] * t_chi2[k])

    return jnp.stack(cols, axis=-1)


@jax.tree_util.register_pytree_node_class
class SEOBNR_ROM:
    """Frequency-Domain Reduced-Order Model (ROM) Surrogate Engine for SEOBNR Waveforms in JAX."""

    _CACHED_SURROGATE: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        num_basis_modes: int = 8,
        model_path: Optional[str] = None,
    ):
        self.num_basis_modes = num_basis_modes
        self.surrogate_data = self._get_native_surrogate(model_path=model_path)
        if self.surrogate_data is None:
            raise RuntimeError("Native SEOBNR surrogate model not found or failed to load.")

    def tree_flatten(self):
        children = (
            self.surrogate_data["mf_grid"],
            self.surrogate_data["amp_basis"],
            self.surrogate_data["phase_basis"],
            self.surrogate_data["amp_weights"],
            self.surrogate_data["phase_weights"],
        )
        aux_data = (
            self.num_basis_modes,
            self.surrogate_data["degree"],
            self.surrogate_data["feature_type"],
            self.surrogate_data["has_two_spins"],
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.num_basis_modes = aux_data[0]
        obj.surrogate_data = {
            "mf_grid": children[0],
            "amp_basis": children[1],
            "phase_basis": children[2],
            "amp_weights": children[3],
            "phase_weights": children[4],
            "degree": aux_data[1],
            "feature_type": aux_data[2],
            "has_two_spins": aux_data[3],
        }
        return obj

    @classmethod
    def _get_native_surrogate(cls, model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if model_path is None and cls._CACHED_SURROGATE is not None:
            return cls._CACHED_SURROGATE

        if model_path is None:
            # Check diffgw_core/data/seobnr_surrogate.npz
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            npz_path = os.path.join(base_dir, "diffgw_core", "data", "seobnr_surrogate.npz")
            pt_path = os.path.join(base_dir, "torchwave", "data", "seobnr_surrogate.pt")
            model_path = npz_path if os.path.isfile(npz_path) else pt_path

        if os.path.isfile(model_path):
            if model_path.endswith(".npz"):
                data = np.load(model_path)
                res = {
                    "mf_grid": jnp.array(data["mf_grid"], dtype=jnp.float64),
                    "amp_basis": jnp.array(data["amp_basis"], dtype=jnp.float64),
                    "phase_basis": jnp.array(data["phase_basis"], dtype=jnp.float64),
                    "amp_weights": jnp.array(data["amp_weights"], dtype=jnp.float64),
                    "phase_weights": jnp.array(data["phase_weights"], dtype=jnp.float64),
                    "degree": int(data["degree"]),
                    "feature_type": str(data["feature_type"]),
                    "has_two_spins": bool(data["has_two_spins"]),
                }
                cls._CACHED_SURROGATE = res
                return res
            elif model_path.endswith(".pt"):
                import torch
                data = torch.load(model_path, weights_only=False)
                res = {
                    "mf_grid": jnp.array(data["mf_grid"].numpy(), dtype=jnp.float64),
                    "amp_basis": jnp.array(data["amp_basis"].numpy(), dtype=jnp.float64),
                    "phase_basis": jnp.array(data["phase_basis"].numpy(), dtype=jnp.float64),
                    "amp_weights": jnp.array(data["amp_weights"].numpy(), dtype=jnp.float64),
                    "phase_weights": jnp.array(data["phase_weights"].numpy(), dtype=jnp.float64),
                    "degree": int(data["degree"]),
                    "feature_type": str(data.get("feature_type", "poly")),
                    "has_two_spins": bool(data.get("has_two_spins", False)),
                }
                cls._CACHED_SURROGATE = res
                return res
        return None

    def __call__(
        self,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate SEOBNR Reduced-Order Model (ROM) frequency-domain polarizations in JAX."""
        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        s1 = as_array(chi1, dtype=jnp.float64)
        s2 = as_array(chi2, dtype=jnp.float64)
        dist = as_array(distance, dtype=jnp.float64)
        inc = as_array(inclination, dtype=jnp.float64)
        phi_c = as_array(phic, dtype=jnp.float64)
        t_c = as_array(tc, dtype=jnp.float64)
        f_eval = as_array(f, dtype=jnp.float64)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)

        m_sec = total_mass * MSUN_SEC
        is_batched = total_mass.ndim >= 1

        if f_eval.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f_eval, 0)
            m_sec_eval = jnp.expand_dims(m_sec, -1)
            eta_eval = jnp.expand_dims(eta, -1)
            dist_eval = jnp.expand_dims(dist, -1)
            inc_eval = jnp.expand_dims(inc, -1)
            phi_c_eval = jnp.expand_dims(phi_c, -1)
            t_c_eval = jnp.expand_dims(t_c, -1)
        else:
            m_sec_eval = m_sec
            eta_eval = eta
            dist_eval = dist
            inc_eval = inc
            phi_c_eval = phi_c
            t_c_eval = t_c

        mf = m_sec_eval * f_eval

        d_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = jnp.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_eval ** (-7.0 / 6.0))

        v = (PI * mf) ** (1.0 / 3.0)
        v5 = v ** 5.0
        phi_0pn = 3.0 / (128.0 * eta_eval * v5)

        deg = self.surrogate_data["degree"]
        w_a = self.surrogate_data["amp_weights"]
        w_p = self.surrogate_data["phase_weights"]
        v_a = self.surrogate_data["amp_basis"]
        v_p = self.surrogate_data["phase_basis"]
        grid_mf = self.surrogate_data["mf_grid"]

        feat_type = self.surrogate_data.get("feature_type", "poly").lower()
        feat_fn = chebyshev_features if feat_type == "chebyshev" else poly_features

        if self.surrogate_data.get("has_two_spins", False):
            phi = feat_fn(eta, s1, s2, degree=deg)
        else:
            phi = feat_fn(eta, chi_pn, degree=deg)

        c_amp = phi @ w_a
        c_phase = phi @ w_p

        amp_nodes = c_amp @ v_a
        phase_nodes = c_phase @ v_p

        if is_batched:
            amp_rom = jax.vmap(lambda m, a: jnp.interp(m, grid_mf, a, left=0.0, right=0.0))(mf, amp_nodes)
            phase_rom = jax.vmap(lambda m, p: jnp.interp(m, grid_mf, p, left=0.0, right=0.0))(mf, phase_nodes)
        else:
            amp_rom = jnp.interp(mf, grid_mf, amp_nodes.reshape(-1), left=0.0, right=0.0)
            phase_rom = jnp.interp(mf, grid_mf, phase_nodes.reshape(-1), left=0.0, right=0.0)

        full_amplitude = a0_phys * (1.0 + amp_rom)

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.size > 0 else 20.0)
            fref = as_array(ref_f, dtype=jnp.float64)
            fref_eval = jnp.expand_dims(fref, -1) if is_batched and fref.ndim < f_eval.ndim else fref

            v_ref = (PI * m_sec_eval * fref_eval) ** (1.0 / 3.0)
            phi_0pn_ref = 3.0 / (128.0 * eta_eval * (v_ref ** 5.0))
            dphi_0pn_ref = -5.0 * PI * m_sec_eval / (128.0 * eta_eval * (v_ref ** 8.0))

            delta_f = 1e-4 * jnp.maximum(fref_eval, 1.0)
            mf_ref = m_sec_eval * fref_eval
            mf_hi = m_sec_eval * (fref_eval + delta_f)
            mf_lo = m_sec_eval * (fref_eval - delta_f)

            if is_batched:
                p_ref = jax.vmap(lambda m, p: jnp.interp(m, grid_mf, p, left=0.0, right=0.0))(mf_ref, phase_nodes)
                p_hi = jax.vmap(lambda m, p: jnp.interp(m, grid_mf, p, left=0.0, right=0.0))(mf_hi, phase_nodes)
                p_lo = jax.vmap(lambda m, p: jnp.interp(m, grid_mf, p, left=0.0, right=0.0))(mf_lo, phase_nodes)
            else:
                p_ref = jnp.interp(mf_ref, grid_mf, phase_nodes.reshape(-1), left=0.0, right=0.0)
                p_hi = jnp.interp(mf_hi, grid_mf, phase_nodes.reshape(-1), left=0.0, right=0.0)
                p_lo = jnp.interp(mf_lo, grid_mf, phase_nodes.reshape(-1), left=0.0, right=0.0)

            dphase_rom_ref = (p_hi - p_lo) / (2.0 * delta_f)

            psi_raw_ref = -(PI / 4.0) + phi_0pn_ref + p_ref
            dpsi_raw_ref = dphi_0pn_ref + dphase_rom_ref
            t_ref = dpsi_raw_ref / (2.0 * PI)

            psi_raw = -(PI / 4.0) + phi_0pn + phase_rom
            full_phase = (
                psi_raw
                - 2.0 * PI * (f_eval - fref_eval) * t_ref
                - psi_raw_ref
                + 2.0 * PI * f_eval * t_c_eval
                - phi_c_eval
            )
        else:
            full_phase = (
                2.0 * PI * f_eval * t_c_eval
                - phi_c_eval
                - (PI / 4.0)
                + phi_0pn
                + phase_rom
            )

        cos_inc = jnp.cos(inc_eval)
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        exp_minus_i_phase = jnp.cos(full_phase) - 1j * jnp.sin(full_phase)

        hp = (plus_factor * full_amplitude).astype(exp_minus_i_phase.dtype) * exp_minus_i_phase
        hc = (cross_factor * full_amplitude * jnp.sin(full_phase)
              - 1j * (cross_factor * full_amplitude * jnp.cos(full_phase)))

        return hp, hc


__all__ = ["SEOBNR_ROM", "poly_features", "chebyshev_features"]
