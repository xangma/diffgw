"""Frequency-Domain Reduced-Order Model (ROM / Surrogate) Waveform Engine.

Strict Clean-Room Implementation based on published peer-reviewed literature:
- Pürrer (2014), Class. Quantum Grav. 31, 195010 (arXiv:1402.4146)
- Pürrer (2016), Phys. Rev. D 93, 064041 (arXiv:1512.02248)
- Cotesta et al. (2018), Phys. Rev. D 98, 084028 (arXiv:1803.10701)
- Gadre et al. (2022), Phys. Rev. D 110, 124038 (arXiv:2203.00381)
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import torch

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    spin_combinations,
)
from torchwave.approximants.eob.waveform import _interp1d
from torchwave.approximants.rom.generator import poly_features


class SEOBNR_ROM:
    """Frequency-Domain Reduced-Order Model (ROM) Surrogate Engine for SEOBNR Waveforms.

    Provides microsecond-latency evaluation and smooth autograd differentiability
    via empirical interpolation and basis projection onto the clean-room native PyTorch surrogate.
    """

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

    @classmethod
    def _get_native_surrogate(cls, model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load and cache native bundled PyTorch surrogate model."""
        if model_path is None and cls._CACHED_SURROGATE is not None:
            return cls._CACHED_SURROGATE

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "data", "seobnr_surrogate.pt")

        if os.path.isfile(model_path):
            try:
                data = torch.load(model_path, weights_only=False)
                if model_path.endswith("seobnr_surrogate.pt"):
                    cls._CACHED_SURROGATE = data
                return data
            except Exception:
                return None
        return None

    def __call__(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the SEOBNR Reduced-Order Model (ROM) frequency-domain polarizations.

        Args:
            f: Frequency evaluation tensor in Hz
            mass1, mass2: Component masses in solar masses
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Inclination angle in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz for physical phase alignment
            phasing_convention: Phasing convention ('coalescence' or 'reference_frequency')
            reference_frame: Alias for phasing_convention

        Returns:
            (h_plus, h_cross) frequency-domain complex tensors
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)

        m_sec = total_mass * MSUN_SEC
        is_batched = total_mass.dim() >= 1

        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec_eval = m_sec.unsqueeze(-1)
            eta_eval = eta.unsqueeze(-1)
            dist_eval = dist.unsqueeze(-1)
            inc_eval = inc.unsqueeze(-1)
            phi_c_eval = phi_c.unsqueeze(-1)
            t_c_eval = t_c.unsqueeze(-1)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            dist_eval = dist
            inc_eval = inc
            phi_c_eval = phi_c
            t_c_eval = t_c

        mf = m_sec_eval * f_eval  # Dimensionless frequency M*f

        # Baseline Leading 0PN Amplitude and Phase
        d_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = torch.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * d_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_eval ** (-7.0 / 6.0))

        # Base TaylorF2 0PN phase: 3 / (128 * eta * v^5)
        v = (PI * mf) ** (1.0 / 3.0)
        v5 = v ** 5.0
        phi_0pn = 3.0 / (128.0 * eta_eval * v5)

        # Native PyTorch surrogate evaluation
        deg = self.surrogate_data["degree"]
        w_a = self.surrogate_data["amp_weights"].to(device=device, dtype=dtype)
        w_p = self.surrogate_data["phase_weights"].to(device=device, dtype=dtype)
        v_a = self.surrogate_data["amp_basis"].to(device=device, dtype=dtype)
        v_p = self.surrogate_data["phase_basis"].to(device=device, dtype=dtype)
        grid_mf = self.surrogate_data["mf_grid"].to(device=device, dtype=dtype)

        # Features shape: (B, num_features) or (1, num_features)
        if self.surrogate_data.get("has_two_spins", False):
            phi = poly_features(eta, s1, s2, degree=deg).to(device=device, dtype=dtype)
        else:
            phi = poly_features(eta, chi_pn, degree=deg).to(device=device, dtype=dtype)
        c_amp = phi @ w_a
        c_phase = phi @ w_p

        amp_nodes = c_amp @ v_a
        phase_nodes = c_phase @ v_p

        amp_rom = _interp1d(mf, grid_mf, amp_nodes)
        phase_rom = _interp1d(mf, grid_mf, phase_nodes)

        if not is_batched and amp_rom.dim() > 1:
            amp_rom = amp_rom.squeeze(0)
            phase_rom = phase_rom.squeeze(0)

        full_amplitude = a0_phys * (1.0 + amp_rom)

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
        )
        conv = conv.lower()

        # Reference-frequency physical conditioning
        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.numel() > 0 else 20.0)
            fref = as_tensor(ref_f, device=device, dtype=dtype)
            fref_eval = fref.unsqueeze(-1) if is_batched and fref.dim() < f_eval.dim() else fref

            v_ref = (PI * m_sec_eval * fref_eval) ** (1.0 / 3.0)
            phi_0pn_ref = 3.0 / (128.0 * eta_eval * (v_ref ** 5.0))
            dphi_0pn_ref = -5.0 * PI * m_sec_eval / (128.0 * eta_eval * (v_ref ** 8.0))

            delta_f = 1e-4 * torch.clamp(fref_eval, min=1.0)
            mf_ref = m_sec_eval * fref_eval
            mf_hi = m_sec_eval * (fref_eval + delta_f)
            mf_lo = m_sec_eval * (fref_eval - delta_f)

            p_ref = _interp1d(mf_ref, grid_mf, phase_nodes)
            p_hi = _interp1d(mf_hi, grid_mf, phase_nodes)
            p_lo = _interp1d(mf_lo, grid_mf, phase_nodes)

            if not is_batched and p_ref.dim() > 1:
                p_ref = p_ref.squeeze(0)
                p_hi = p_hi.squeeze(0)
                p_lo = p_lo.squeeze(0)

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

        # Inclination projections
        cos_inc = torch.cos(inc_eval)
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        exp_minus_i_phase = torch.complex(torch.cos(full_phase), -torch.sin(full_phase))

        hp = (plus_factor * full_amplitude).to(exp_minus_i_phase.dtype) * exp_minus_i_phase
        hc = torch.complex(
            cross_factor * full_amplitude * torch.sin(full_phase),
            -cross_factor * full_amplitude * torch.cos(full_phase),
        )

        return hp, hc
