"""Clean-room implementation of IMRPhenomXHM calibrated higher-order multipolar approximant.

Literature citations:
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
  "Multimode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries: IMRPhenomXHM"
- Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)
  "Setting the cornerstone for next-generation models of gravitational reactions: IMRPhenomXAS"
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
  "Quasinormal modes of Kerr black holes: from astrophysics to string theory"

Strict Clean-Room Barrier Compliance:
- Mathematical equations from published literature (García-Quirós et al. 2020) and spec docs/reviews/2026-09-11-clean-room-spec/phenomxhm-spec.md.
- Calibration parameter fits from author ancillary release in src/torchwave/approximants/xhm_fits.py.
- Zero copying, inspecting, or porting of LAL/LALSimulation/PyCBC source code or internal constants.
- Pure PyTorch native tensor implementation with autograd differentiability.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
)
from torchwave.approximants.imrphenom_xas import (
    IMRPhenomXAS,
    XASParameterFits,
    safe_sqrt,
)
from torchwave.approximants.xhm_fits import XHMParameterFits
from torchwave.approximants.imrphenom_hm import _swsh_projection_coefficients


# Pre-inverted 3x3 matrix for inspiral collocation points
# Collocation points at 0.5 f_Ins, 0.75 f_Ins, 1.0 f_Ins
# Powers: 7/3, 8/3, 3
_INSP_MAT = torch.tensor([
    [0.5 ** (7.0 / 3.0), 0.5 ** (8.0 / 3.0), 0.5 ** 3.0],
    [0.75 ** (7.0 / 3.0), 0.75 ** (8.0 / 3.0), 0.75 ** 3.0],
    [1.0, 1.0, 1.0],
], dtype=torch.float64)
_INSP_MAT_INV = torch.linalg.inv(_INSP_MAT)


def _unsqueeze_to(tensor: Union[float, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Unsqueeze tensor along trailing dimensions until it matches target's dimensionality."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=target.dtype, device=target.device)
    res = tensor
    while res.dim() < target.dim():
        res = res.unsqueeze(-1)
    return res


def _eval_f_isco(chi: torch.Tensor) -> torch.Tensor:
    """Kerr ISCO frequency for dimensionless spin chi."""
    chi_clamped = torch.clamp(chi, min=-0.9999, max=0.9999)
    one_m_chi2 = torch.clamp(1.0 - chi_clamped**2, min=1e-12)
    z1 = 1.0 + torch.pow(one_m_chi2, 1.0 / 3.0) * (
        torch.pow(1.0 + chi_clamped, 1.0 / 3.0) + torch.pow(1.0 - chi_clamped, 1.0 / 3.0)
    )
    z2 = safe_sqrt(3.0 * chi_clamped**2 + z1**2)
    diff = torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=0.0)
    r_isco = 3.0 + z2 - torch.sign(chi_clamped) * safe_sqrt(diff)
    r_isco_clamped = torch.clamp(r_isco, min=1.0)
    return 1.0 / (PI * (r_isco_clamped**1.5 + chi_clamped))


def _pn_amplitude_h21(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """3PN Fourier-domain amplitude H_21 (Eqs. 1864-1887 of García-Quirós et al. 2020)."""
    c1 = (1.0 / 3.0) * math.sqrt(2.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = torch.zeros_like(v)

    t2_re = v2 * (-1.5 * delta * chi_s - 1.5 * chi_a)
    t2_im = torch.zeros_like(v)

    t3_re = v3 * ((117.0 / 56.0) * delta * eta + (335.0 / 672.0) * delta)
    t3_im = torch.zeros_like(v)

    t4_re = v4 * (
        -(965.0 / 336.0) * delta * eta * chi_s
        + (3427.0 / 1344.0) * delta * chi_s
        - PI * delta
        - (2101.0 / 336.0) * eta * chi_a
        + (3427.0 / 1344.0) * chi_a
    )
    t4_im = v4 * (-0.5 * delta - 0.5 * delta * math.log(16.0))

    t5_re = v5 * (
        (21365.0 / 8064.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (39.0 / 8.0) * delta * eta * (chi_s**2)
        - (36529.0 / 12544.0) * delta * eta
        - (307.0 / 32.0) * delta * (chi_a**2)
        - (307.0 / 32.0) * delta * (chi_s**2)
        + 3.0 * PI * delta * chi_s
        - (964357.0 / 8128512.0) * delta
        + (213.0 / 4.0) * eta * chi_a * chi_s
        - (307.0 / 16.0) * chi_a * chi_s
        + 3.0 * PI * chi_a
    )
    t5_im = torch.zeros_like(v)

    t6_re = v6 * (
        -(547.0 / 768.0) * delta * (eta**2) * chi_s
        - 15.0 * delta * eta * (chi_a**2) * chi_s
        - (3.0 / 16.0) * delta * eta * (chi_s**3)
        - (7049629.0 / 225792.0) * delta * eta * chi_s
        + (417.0 / 112.0) * PI * delta * eta
        + (729.0 / 64.0) * delta * (chi_a**2) * chi_s
        + (243.0 / 64.0) * delta * (chi_s**3)
        + (143063173.0 / 5419008.0) * delta * chi_s
        - (2455.0 / 1344.0) * PI * delta
        + (42617.0 / 1792.0) * (eta**2) * chi_a
        - 15.0 * eta * (chi_a**3)
        - (489.0 / 16.0) * eta * chi_a * (chi_s**2)
        - (22758317.0 / 225792.0) * eta * chi_a
        + (243.0 / 64.0) * (chi_a**3)
        + (729.0 / 64.0) * chi_a * (chi_s**2)
        + (143063173.0 / 5419008.0) * chi_a
    )
    t6_im = v6 * (
        -(1489.0 / 112.0) * delta * eta
        - (89.0 / 28.0) * delta * eta * math.log(2.0)
        - (335.0 / 1344.0) * delta
        - (335.0 / 336.0) * delta * math.log(2.0)
    )

    t_re = t1_re + t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t2_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return safe_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h33(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """3PN Fourier-domain amplitude H_33 (Eqs. 1889-1910 of García-Quirós et al. 2020)."""
    c1 = -(3.0 / 4.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t1_re = v * delta
    t1_im = torch.zeros_like(v)

    t3_re = v3 * delta * ((27.0 / 8.0) * eta - 1945.0 / 672.0)
    t3_im = torch.zeros_like(v)

    t4_re = v4 * (
        -(2.0 / 3.0) * delta * eta * chi_s
        + (65.0 / 24.0) * delta * chi_s
        + PI * delta
        - (28.0 / 3.0) * eta * chi_a
        + (65.0 / 24.0) * chi_a
    )
    t4_im = v4 * (-(21.0 / 5.0) * delta + 6.0 * delta * math.log(1.5))

    t5_re = v5 * (
        (420389.0 / 63360.0) * delta * (eta**2)
        + 10.0 * delta * eta * (chi_a**2)
        + (1.0 / 8.0) * delta * eta * (chi_s**2)
        - (11758073.0 / 887040.0) * delta * eta
        - (81.0 / 32.0) * delta * (chi_a**2)
        - (81.0 / 32.0) * delta * (chi_s**2)
        - (1077664867.0 / 447068160.0) * delta
        + (81.0 / 4.0) * eta * chi_a * chi_s
        - (81.0 / 16.0) * chi_a * chi_s
    )
    t5_im = torch.zeros_like(v)

    t6_re = v6 * (
        -(67.0 / 24.0) * delta * (eta**2) * chi_s
        - (58745.0 / 4032.0) * delta * eta * chi_s
        + (131.0 / 16.0) * PI * delta * eta
        + (163021.0 / 16128.0) * delta * chi_s
        - (5675.0 / 1344.0) * PI * delta
        - (137.0 / 24.0) * (eta**2) * chi_a
        - (148501.0 / 4032.0) * eta * chi_a
        + (163021.0 / 16128.0) * chi_a
    )
    t6_im = v6 * (
        -(440957.0 / 9720.0) * delta * eta
        + (69.0 / 4.0) * delta * eta * math.log(1.5)
        + (389.0 / 32.0) * delta
        - (1945.0 / 112.0) * delta * math.log(1.5)
    )

    t_re = t1_re + t3_re + t4_re + t5_re + t6_re
    t_im = t1_im + t3_im + t4_im + t5_im + t6_im

    h_re = -c1 * t_im
    h_im = c1 * t_re
    return safe_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h32(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """3PN Fourier-domain amplitude H_32 (Eqs. 1912-1923 of García-Quirós et al. 2020)."""
    c1 = (1.0 / 3.0) * math.sqrt(5.0 / 7.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (1.0 - 3.0 * eta)
    t3_re = v3 * (4.0 * eta * chi_s)
    t4_re = v4 * (
        -(589.0 / 72.0) * (eta**2)
        + (12325.0 / 2016.0) * eta
        - (10471.0 / 10080.0)
    )

    t5_re = v5 * (
        eta * ((113.0 / 8.0) * delta * chi_a + (1081.0 / 84.0) * chi_s)
        + (1.0 / 24.0) * (-113.0 * delta * chi_a - 113.0 * chi_s)
        - 15.0 * (eta**2) * chi_s
    )
    t5_im = v5 * (eta * (-66.0 / 5.0) + (72.0 / 24.0))

    t6_re = v6 * (
        eta * (
            -(1633.0 / 48.0) * delta * chi_a * chi_s
            - (563.0 / 32.0) * (chi_a**2)
            - (2549.0 / 96.0) * (chi_s**2)
            + 8.0 * PI * chi_s
            - (8689883.0 / 149022720.0)
        )
        + (81.0 / 16.0) * delta * chi_a * chi_s
        + (837223.0 / 63360.0) * (eta**3)
        + (eta**2) * (30.0 * (chi_a**2) + (313.0 / 24.0) * (chi_s**2) - 78584047.0 / 2661120.0)
        + (81.0 / 32.0) * (chi_a**2)
        + (81.0 / 32.0) * (chi_s**2)
        + (824173699.0 / 447068160.0)
    )
    t6_im = torch.zeros_like(v)

    t_re = t2_re + t3_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return safe_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_h44(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """3PN Fourier-domain amplitude H_44 (Eqs. 1948-1960 of García-Quirós et al. 2020)."""
    c1 = (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
    v2 = v * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v5 * v

    t2_re = v2 * (3.0 * eta - 1.0)
    t4_re = v4 * (
        (1063.0 / 88.0) * (eta**2)
        - (128221.0 / 7392.0) * eta
        + (158383.0 / 36960.0)
    )

    term5_inner_re = (
        -eta * (1695.0 * delta * chi_a + 2075.0 * chi_s)
        + 565.0 * delta * chi_a
        + 1140.0 * (eta**2) * chi_s
        + 565.0 * chi_s
    )
    term5_inner_im = -eta * (-3579.0 + 2880.0 * math.log(2.0)) - 1008.0 + 960.0 * math.log(2.0)
    t5_re = v5 * (PI * (2.0 - 6.0 * eta) + (1.0 / 120.0) * term5_inner_re)
    t5_im = v5 * ((1.0 / 120.0) * term5_inner_im)

    t6_re = v6 * (
        eta * (
            (243.0 / 16.0) * delta * chi_a * chi_s
            + (563.0 / 32.0) * (chi_a**2)
            + (247.0 / 32.0) * (chi_s**2)
            - (22580029007.0 / 880588800.0)
        )
        - (81.0 / 16.0) * delta * chi_a * chi_s
        - (7606537.0 / 274560.0) * (eta**3)
        + (eta**2) * (-30.0 * (chi_a**2) - (3.0 / 8.0) * (chi_s**2) + 901461137.0 / 11531520.0)
        - (81.0 / 32.0) * (chi_a**2)
        - (81.0 / 32.0) * (chi_s**2)
        + (7888301437.0 / 29059430400.0)
    )
    t6_im = torch.zeros_like(v)

    t_re = t2_re + t4_re + t5_re + t6_re
    t_im = t5_im + t6_im

    h_re = c1 * t_re
    h_im = c1 * t_im
    return safe_sqrt(h_re**2 + h_im**2)


def _pn_amplitude_ratio(
    l: int,
    m: int,
    f: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    chi_s: torch.Tensor,
    chi_a: torch.Tensor,
) -> torch.Tensor:
    """Evaluate dimensionless amplitude ratio |A_lm^PN| / A_22^0 = (2/m)^(7/6) * |H_lm(v)|."""
    m_flt = float(m)
    v = torch.pow(torch.clamp((2.0 * PI * f) / m_flt, min=1e-12), 1.0 / 3.0)
    factor = (2.0 / m_flt) ** (7.0 / 6.0)
    eta_u = _unsqueeze_to(eta, v)
    delta_u = _unsqueeze_to(delta, v)
    chi_s_u = _unsqueeze_to(chi_s, v)
    chi_a_u = _unsqueeze_to(chi_a, v)
    if l == 2 and m == 1:
        h_val = _pn_amplitude_h21(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 3:
        h_val = _pn_amplitude_h33(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 3 and m == 2:
        h_val = _pn_amplitude_h32(v, eta_u, delta_u, chi_s_u, chi_a_u)
    elif l == 4 and m == 4:
        h_val = _pn_amplitude_h44(v, eta_u, delta_u, chi_s_u, chi_a_u)
    else:
        raise ValueError(f"Mode ({l}, {m}) not supported.")
    return factor * h_val


class IMRPhenomXHM(nn.Module):
    """IMRPhenomXHM Calibrated Higher-Harmonic IMR Frequency-Domain Approximant.

    Supports harmonics (l, m) in {(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)}.
    Literature:
    - García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
    - Pratten et al. (2020), Phys. Rev. D 101, 024029 (arXiv:2001.10911)
    """

    SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    supported_modes = SUPPORTED_MODES

    @property
    def get_supported_modes(self):
        return self.SUPPORTED_MODES

    def __init__(
        self,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        carrier: str = "XAS",
        base_22: Optional[nn.Module] = None,
    ):
        self.carrier = carrier
        super().__init__()
        self.modes = tuple(modes) if modes is not None else self.SUPPORTED_MODES
        for lm in self.modes:
            if lm not in self.SUPPORTED_MODES:
                raise ValueError(f"Mode {lm} not supported. Supported: {self.SUPPORTED_MODES}")
        self.base_22 = base_22 if base_22 is not None else IMRPhenomXAS()

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        mass1: Union[float, torch.Tensor] = 30.0,
        mass2: Union[float, torch.Tensor] = 30.0,
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phi0: Union[float, torch.Tensor] = 0.0,
        modes: Optional[Sequence[Tuple[int, int]]] = None,
        return_modes: bool = False,
        return_unprojected_modes: bool = False,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]],
    ]:
        """Generate frequency-domain polarizations (h_plus, h_cross) including higher multipoles.

        Args:
            f: Frequency tensor in Hz
            mass1, mass2: Component masses in solar masses (mass1 >= mass2)
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
            phi0: Observer azimuth angle in radians (default 0.0)
            modes: Optional tuple of modes to evaluate
            return_modes: If True, return projected mode polarizations (hp, hc)
            return_unprojected_modes: If True, return unprojected complex mode strains h_lm
        """
        if f is None:
            if sample_frequencies is not None:
                f = sample_frequencies
            else:
                raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1_in = as_tensor(mass1, device=device, dtype=dtype)
        m2_in = as_tensor(mass2, device=device, dtype=dtype)
        s1_in = as_tensor(chi1, device=device, dtype=dtype)
        s2_in = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(kwargs.pop("phic", phic), device=device, dtype=dtype)
        t_c = as_tensor(kwargs.pop("tc", tc), device=device, dtype=dtype)
        phi_0 = as_tensor(kwargs.pop("phi0", phi0), device=device, dtype=dtype)

        active_modes = tuple(modes) if modes is not None else self.modes

        # Body exchange canonical ordering (m1 >= m2)
        # Binary relabeling 1 <-> 2: under 180-deg orbital rotation, odd-m modes pick up a factor (-1)^m = -1.
        swap_mask = m1_in < m2_in
        m1 = torch.where(swap_mask, m2_in, m1_in)
        m2 = torch.where(swap_mask, m1_in, m2_in)
        s1 = torch.where(swap_mask, s2_in, s1_in)
        s2 = torch.where(swap_mask, s1_in, s2_in)

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta, _, _, S = XHMParameterFits.compute_spin_variables(eta, s1, s2)
        chi_s = 0.5 * (s1 + s2)
        chi_a = 0.5 * (s1 - s2)
        chi_eff = (m1 * s1 + m2 * s2) / total_mass
        m_sec = total_mass * MSUN_SEC

        # Final remnant properties from XAS fits
        chi_f = XASParameterFits.fit_final_spin(eta, s1, s2)
        erad = XASParameterFits.fit_erad(eta, s1, s2)
        m_f = 1.0 - erad

        # ISCO and MECO frequencies for (2,2)
        f_isco_22 = _eval_f_isco(chi_f)
        f_meco_22 = XASParameterFits.fit_f_meco(eta, s1, s2)
        diff_isco_meco = torch.abs(f_isco_22 - f_meco_22)

        # Mode QNMs in geometric units (rescaled by remnant mass M_f)
        f_ring: Dict[str, torch.Tensor] = {}
        f_damp: Dict[str, torch.Tensor] = {}
        for lm_str in ("21", "22", "32", "33", "44"):
            f_ring[lm_str] = getattr(XHMParameterFits, f"qnm_fring_{lm_str}")(chi_f) / m_f
            f_damp[lm_str] = getattr(XHMParameterFits, f"qnm_fdamp_{lm_str}")(chi_f) / m_f

        # Cutting frequencies per mode
        eps_amp = {
            "21": torch.tensor(1.0, device=device, dtype=dtype),
            "33": torch.tensor(1.0, device=device, dtype=dtype),
            "32": f_ring["32"] / f_ring["22"],
            "44": torch.tensor(1.0, device=device, dtype=dtype),
        }
        del_amp = {
            "21": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * (chi_eff**2),
            "33": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff,
            "32": (0.75 - 0.235 * torch.abs(chi_eff)) * (f_ring["32"] / f_ring["22"]),
            "44": 0.75 - 0.235 * chi_eff,
        }
        m_vals = {"21": 1, "32": 2, "33": 3, "44": 4}

        f_ins_amp = {
            lm_str: (m_vals[lm_str] / 2.0) * (f_meco_22 * eps_amp[lm_str] + diff_isco_meco * del_amp[lm_str])
            for lm_str in ("21", "32", "33", "44")
        }
        f_rd_amp = {
            "21": 0.75 * f_ring["21"],
            "33": 0.95 * f_ring["33"],
            "32": f_ring["22"],
            "44": 0.90 * f_ring["44"],
        }
        f_ins_phi = {
            lm_str: (m_vals[lm_str] / 2.0) * f_meco_22
            for lm_str in ("21", "32", "33", "44")
        }
        f_rd_phi = {
            "21": f_ring["21"] - f_damp["21"],
            "33": f_ring["33"] - f_damp["33"],
            "32": f_ring["22"] - 0.5 * f_damp["22"],
            "44": f_ring["44"] - f_damp["44"],
        }

        # Amplitude sigma fits
        sigma_amp = {
            "21": XHMParameterFits.amp_sigma_21(eta, s1, s2, S),
            "33": torch.tensor(1.30, device=device, dtype=dtype),
            "32": torch.tensor(1.33, device=device, dtype=dtype),
            "44": torch.tensor(1.33, device=device, dtype=dtype),
        }

        # 22-mode phase parameters for ringdown rescaling (Eq. 6.5)
        alpha_lambda_22 = XHMParameterFits.phase_rd_alpha_lambda(eta, s1, s2, S)
        alpha2_22 = XHMParameterFits.phase_rd_alpha2(eta, s1, s2, S)

        # Batching alignment
        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec_eval = m_sec.unsqueeze(-1)
            eta_eval = eta.unsqueeze(-1)
            delta_eval = delta.unsqueeze(-1)
            chi_s_eval = chi_s.unsqueeze(-1)
            chi_a_eval = chi_a.unsqueeze(-1)
            dist_eval = dist.unsqueeze(-1)
            inc_eval = inc.unsqueeze(-1)
            phi0_eval = phi_0.unsqueeze(-1)
            phic_eval = phi_c.unsqueeze(-1)
            tc_eval = t_c.unsqueeze(-1)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta
            delta_eval = delta
            chi_s_eval = chi_s
            chi_a_eval = chi_a
            dist_eval = dist
            inc_eval = inc
            phi0_eval = phi_0
            phic_eval = phi_c
            tc_eval = t_c

        # Geometric frequency grid: Mf
        f_safe = torch.clamp(f_eval, min=1e-8)
        mf_grid = m_sec_eval * f_safe

        # Evaluate carrier (2, 2) mode at face-on (iota=0)
        hp_22_raw, _, phi_22_carrier = self.base_22(
            f=f_eval,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=dist,
            inclination=0.0,
            phic=phic,
            tc=tc,
            f_ref=f_ref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            return_phase=True,
        )
        amp_22_phys = torch.abs(hp_22_raw)

        # Base physical leading quadrupole factor A0_phys
        dist_sec = (dist_eval * MPC_SI) / C_SI
        prefactor = torch.sqrt(5.0 * eta_eval / 24.0) / (PI ** (2.0 / 3.0) * dist_sec)
        a0_phys = prefactor * (m_sec_eval ** (5.0 / 6.0)) * (f_safe ** (-7.0 / 6.0))
        # Mode normalization factor relates mode amplitude to physical strain
        mode_norm = math.sqrt(16.0 * PI / 5.0)

        # Evaluate carrier phase and derivative at specific geometric frequencies for boundary alignment
        def eval_carrier_phi_and_deriv(mf_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            eps_mf = 1e-5
            f_hz_target = mf_target / m_sec
            f_hz_minus = (mf_target - eps_mf) / m_sec
            f_hz_plus = (mf_target + eps_mf) / m_sec
            f_probe = torch.stack([f_hz_minus, f_hz_target, f_hz_plus], dim=-1)

            _, _, phi_probe = self.base_22(
                f=f_probe,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=0.0,
                phic=phic,
                tc=tc,
                f_ref=f_ref,
                phasing_convention=phasing_convention,
                reference_frame=reference_frame,
                return_phase=True,
            )
            p_minus = phi_probe[..., 0]
            p_val = phi_probe[..., 1]
            p_plus = phi_probe[..., 2]
            dp_dmf = (p_plus - p_minus) / (2.0 * eps_mf)
            return p_val, dp_dmf

        h_plus_total = torch.zeros_like(
            amp_22_phys, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        h_cross_total = torch.zeros_like(h_plus_total)
        mode_dict = {}

        # Store h22 for mode-mixing with (3,2)
        exp_neg_i_phi22 = torch.complex(torch.cos(phi_22_carrier), -torch.sin(phi_22_carrier))
        h22_mode_strain = (mode_norm * amp_22_phys).to(exp_neg_i_phi22.dtype) * exp_neg_i_phi22

        for l, m in active_modes:
            lm_str = f"{l}{m}"
            if (l, m) == (2, 2):
                h_mode = h22_mode_strain
            else:
                m_flt = float(m)
                f_ins_a = f_ins_amp[lm_str]
                f_rd_a = f_rd_amp[lm_str]
                f_ins_p = f_ins_phi[lm_str]
                f_rd_p = f_rd_phi[lm_str]
                f_ring_lm = f_ring[lm_str]
                f_damp_lm = f_damp[lm_str]
                sigma_lm = sigma_amp[lm_str]

                # -------------------------------------------------------------
                # 1. Amplitude Reconstruction
                # -------------------------------------------------------------
                f1_ins = 0.5 * f_ins_a
                f2_ins = 0.75 * f_ins_a
                f3_ins = f_ins_a

                h1_pn = _pn_amplitude_ratio(l, m, f1_ins, eta, delta, chi_s, chi_a)
                h2_pn = _pn_amplitude_ratio(l, m, f2_ins, eta, delta, chi_s, chi_a)
                h3_pn = _pn_amplitude_ratio(l, m, f3_ins, eta, delta, chi_s, chi_a)

                h1_fit = getattr(XHMParameterFits, f"amp_insp1_{lm_str}")(eta, s1, s2, S)
                h2_fit = getattr(XHMParameterFits, f"amp_insp2_{lm_str}")(eta, s1, s2, S)
                h3_fit = getattr(XHMParameterFits, f"amp_insp3_{lm_str}")(eta, s1, s2, S)

                rhs_insp = torch.stack([h1_fit - h1_pn, h2_fit - h2_pn, h3_fit - h3_pn], dim=-1)
                inv_mat = _INSP_MAT_INV.to(device=device, dtype=dtype)
                abc = torch.matmul(rhs_insp, inv_mat.T)
                alpha_insp = abc[..., 0]
                beta_insp = abc[..., 1]
                gamma_insp = abc[..., 2]

                # Ringdown amplitude parameters
                alambda_fit = getattr(XHMParameterFits, f"amp_alambda_{lm_str}")(eta, s1, s2, S)
                lambda_fit = getattr(XHMParameterFits, f"amp_lambda_{lm_str}")(eta, s1, s2, S)

                def eval_h_rd(f_in: torch.Tensor) -> torch.Tensor:
                    w = _unsqueeze_to(f_damp_lm * sigma_lm, f_in)
                    f_ring_u = _unsqueeze_to(f_ring_lm, f_in)
                    lam_u = _unsqueeze_to(lambda_fit, f_in)
                    alam_u = _unsqueeze_to(alambda_fit, f_in)
                    den = (f_in - f_ring_u) ** 2 + w**2
                    exp_term = torch.exp(-torch.clamp((f_in - f_ring_u) * lam_u / w, min=-80.0, max=80.0))
                    f_pow = torch.pow(torch.clamp(f_in, min=1e-8), 1.0 / 12.0)
                    return (1.0 / f_pow) * (torch.abs(alam_u) * w / torch.clamp(den, min=1e-12)) * exp_term

                def eval_dh_rd(f_in: torch.Tensor) -> torch.Tensor:
                    eps_h = 1e-6
                    return (eval_h_rd(f_in + eps_h) - eval_h_rd(f_in - eps_h)) / (2.0 * eps_h)

                def eval_h_ins(f_in: torch.Tensor) -> torch.Tensor:
                    pn_part = _pn_amplitude_ratio(l, m, f_in, eta, delta, chi_s, chi_a)
                    f_ins_u = _unsqueeze_to(f_ins_a, f_in)
                    a_u = _unsqueeze_to(alpha_insp, f_in)
                    b_u = _unsqueeze_to(beta_insp, f_in)
                    g_u = _unsqueeze_to(gamma_insp, f_in)
                    f_ratio = f_in / f_ins_u
                    pseudo = (
                        a_u * torch.pow(f_ratio, 7.0 / 3.0)
                        + b_u * torch.pow(f_ratio, 8.0 / 3.0)
                        + g_u * torch.pow(f_ratio, 3.0)
                    )
                    return pn_part + pseudo

                def eval_dh_ins(f_in: torch.Tensor) -> torch.Tensor:
                    eps_h = 1e-6
                    return (eval_h_ins(f_in + eps_h) - eval_h_ins(f_in - eps_h)) / (2.0 * eps_h)

                # Intermediate amplitude collocation matching
                f_int1 = f_ins_a + (1.0 / 3.0) * (f_rd_a - f_ins_a)
                f_int2 = f_ins_a + (2.0 / 3.0) * (f_rd_a - f_ins_a)

                h_int1_fit = getattr(XHMParameterFits, f"amp_int1_{lm_str}")(eta, s1, s2, S)
                h_int2_fit = getattr(XHMParameterFits, f"amp_int2_{lm_str}")(eta, s1, s2, S)

                h_ins_at_cut = eval_h_ins(f_ins_a)
                dh_ins_at_cut = eval_dh_ins(f_ins_a)
                p_ins = torch.pow(PI * f_ins_a, 7.0 / 6.0) / torch.clamp(h_ins_at_cut, min=1e-12)
                dp_ins = torch.pow(PI * f_ins_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_ins_a * torch.clamp(h_ins_at_cut, min=1e-12))
                    - dh_ins_at_cut / torch.clamp(h_ins_at_cut**2, min=1e-12)
                )

                p_int1 = torch.pow(PI * f_int1, 7.0 / 6.0) / torch.clamp(h_int1_fit, min=1e-12)
                p_int2 = torch.pow(PI * f_int2, 7.0 / 6.0) / torch.clamp(h_int2_fit, min=1e-12)

                h_rd_at_cut = eval_h_rd(f_rd_a)
                dh_rd_at_cut = eval_dh_rd(f_rd_a)
                p_rd = torch.pow(PI * f_rd_a, 7.0 / 6.0) / torch.clamp(h_rd_at_cut, min=1e-12)
                dp_rd = torch.pow(PI * f_rd_a, 7.0 / 6.0) * (
                    (7.0 / 6.0) / (f_rd_a * torch.clamp(h_rd_at_cut, min=1e-12))
                    - dh_rd_at_cut / torch.clamp(h_rd_at_cut**2, min=1e-12)
                )

                def make_vandermonde_rows(f_node: torch.Tensor, is_deriv: bool = False) -> torch.Tensor:
                    if not is_deriv:
                        return torch.stack([
                            torch.ones_like(f_node), f_node, f_node**2, f_node**3, f_node**4, f_node**5
                        ], dim=-1)
                    return torch.stack([
                        torch.zeros_like(f_node), torch.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2), 4.0 * (f_node**3), 5.0 * (f_node**4)
                    ], dim=-1)

                v_mat = torch.stack([
                    make_vandermonde_rows(f_ins_a),
                    make_vandermonde_rows(f_ins_a, is_deriv=True),
                    make_vandermonde_rows(f_int1),
                    make_vandermonde_rows(f_int2),
                    make_vandermonde_rows(f_rd_a),
                    make_vandermonde_rows(f_rd_a, is_deriv=True),
                ], dim=-2)
                rhs_amp = torch.stack([p_ins, dp_ins, p_int1, p_int2, p_rd, dp_rd], dim=-1)
                delta_amp = torch.linalg.solve(v_mat, rhs_amp)

                # 3rd-order Hermite polynomial (García-Quirós et al. 2020 Sec V.1.3 fallback for zero-crossings)
                def make_hermite_rows(f_node: torch.Tensor, is_deriv: bool = False) -> torch.Tensor:
                    if not is_deriv:
                        return torch.stack([
                            torch.ones_like(f_node), f_node, f_node**2, f_node**3
                        ], dim=-1)
                    return torch.stack([
                        torch.zeros_like(f_node), torch.ones_like(f_node), 2.0 * f_node, 3.0 * (f_node**2)
                    ], dim=-1)

                v_mat3 = torch.stack([
                    make_hermite_rows(f_ins_a),
                    make_hermite_rows(f_ins_a, is_deriv=True),
                    make_hermite_rows(f_rd_a),
                    make_hermite_rows(f_rd_a, is_deriv=True),
                ], dim=-2)
                rhs_amp3 = torch.stack([p_ins, dp_ins, p_rd, dp_rd], dim=-1)
                delta_amp3 = torch.linalg.solve(v_mat3, rhs_amp3)

                p3_poly_grid = (
                    _unsqueeze_to(delta_amp3[..., 0], mf_grid)
                    + _unsqueeze_to(delta_amp3[..., 1], mf_grid) * mf_grid
                    + _unsqueeze_to(delta_amp3[..., 2], mf_grid) * (mf_grid**2)
                    + _unsqueeze_to(delta_amp3[..., 3], mf_grid) * (mf_grid**3)
                )

                # Interval-level check for 5th-order zero crossing or severe dip
                s_nodes = torch.linspace(0.0, 1.0, 11, device=device, dtype=dtype)
                f_ins_u1 = f_ins_a.unsqueeze(-1) if f_ins_a.dim() > 0 else f_ins_a
                f_rd_u1 = f_rd_a.unsqueeze(-1) if f_rd_a.dim() > 0 else f_rd_a
                nodes_check = f_ins_u1 + s_nodes * (f_rd_u1 - f_ins_u1)
                
                c0 = delta_amp[..., 0].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[0]
                c1 = delta_amp[..., 1].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[1]
                c2 = delta_amp[..., 2].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[2]
                c3_c = delta_amp[..., 3].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[3]
                c4 = delta_amp[..., 4].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[4]
                c5 = delta_amp[..., 5].unsqueeze(-1) if delta_amp.dim() > 1 else delta_amp[5]
                
                p5_check = c0 + c1*nodes_check + c2*(nodes_check**2) + c3_c*(nodes_check**3) + c4*(nodes_check**4) + c5*(nodes_check**5)
                p_bound_min = torch.minimum(p_ins, p_rd)
                p_bound_min_u = p_bound_min.unsqueeze(-1) if p_bound_min.dim() > 0 else p_bound_min
                has_zero_crossing = torch.any(p5_check <= 0.05 * p_bound_min_u, dim=-1)
                has_zero_crossing_u = _unsqueeze_to(has_zero_crossing, mf_grid)

                # Evaluate amplitude ratio across frequency grid
                h_ins_grid = eval_h_ins(mf_grid)
                p_poly_grid = (
                    _unsqueeze_to(delta_amp[..., 0], mf_grid)
                    + _unsqueeze_to(delta_amp[..., 1], mf_grid) * mf_grid
                    + _unsqueeze_to(delta_amp[..., 2], mf_grid) * (mf_grid**2)
                    + _unsqueeze_to(delta_amp[..., 3], mf_grid) * (mf_grid**3)
                    + _unsqueeze_to(delta_amp[..., 4], mf_grid) * (mf_grid**4)
                    + _unsqueeze_to(delta_amp[..., 5], mf_grid) * (mf_grid**5)
                )
                f_ins_a_u = _unsqueeze_to(f_ins_a, mf_grid)
                f_rd_a_u = _unsqueeze_to(f_rd_a, mf_grid)

                # Order 1 (linear) polynomial (García-Quirós et al. 2020 Sec V.1.3 minimum order)
                p_ins_u = _unsqueeze_to(p_ins, mf_grid)
                p_rd_u = _unsqueeze_to(p_rd, mf_grid)
                p1_poly_grid = p_ins_u + (mf_grid - f_ins_a_u) * (
                    (p_rd_u - p_ins_u) / torch.clamp(f_rd_a_u - f_ins_a_u, min=1e-8)
                )

                p_bound_min = torch.minimum(p_ins, p_rd)
                p_bound_min_u = _unsqueeze_to(p_bound_min, mf_grid)

                p_poly_active = torch.where(has_zero_crossing_u, p3_poly_grid, p_poly_grid)
                # If polynomial still dips or crosses zero, fallback to order 1 linear
                p_poly_active = torch.where(p_poly_active <= 0.05 * p_bound_min_u, p1_poly_grid, p_poly_active)
                p_poly_active = torch.clamp(p_poly_active, min=1e-12)
                h_int_grid = torch.pow(PI * mf_grid, 7.0 / 6.0) / p_poly_active
                h_rd_grid = eval_h_rd(mf_grid)
                h_ratio_grid = torch.where(
                    mf_grid <= f_ins_a_u,
                    h_ins_grid,
                    torch.where(
                        mf_grid <= f_rd_a_u,
                        h_int_grid,
                        h_rd_grid,
                    ),
                )
                if m % 2 != 0:
                    zero_odd_mask = _unsqueeze_to((delta == 0.0) & (chi_a == 0.0), mf_grid)
                    h_ratio_grid = torch.where(zero_odd_mask, torch.zeros_like(h_ratio_grid), h_ratio_grid)

                amp_lm_phys = mode_norm * a0_phys * h_ratio_grid

                # -------------------------------------------------------------
                # 2. Phase Reconstruction
                # -------------------------------------------------------------
                beta_eta = 1.0 + 0.001 * (0.25 / torch.clamp(eta, min=1e-6) - 1.0)
                f_start = beta_eta * f_ins_p
                f_end = f_ring_lm if (l, m) != (3, 2) else (f_ring["22"] - 0.5 * f_damp["22"])

                c_pts = [
                    f_start,
                    0.25 * ((math.sqrt(3.0) + 2.0) * f_start - (math.sqrt(3.0) - 2.0) * f_end),
                    0.25 * (f_end + 3.0 * f_start),
                    0.50 * (f_end + f_start),
                    0.25 * (3.0 * f_end + f_start),
                ]

                if (l, m) != (3, 2):
                    # Modes without mixing: (2,1), (3,3), (4,4)
                    f_p1 = getattr(XHMParameterFits, f"phase_p1_{lm_str}")(eta, s1, s2, S)
                    f_p2 = getattr(XHMParameterFits, f"phase_p2_{lm_str}")(eta, s1, s2, S)
                    f_p3 = getattr(XHMParameterFits, f"phase_p3_{lm_str}")(eta, s1, s2, S)
                    f_p4 = getattr(XHMParameterFits, f"phase_p4_{lm_str}")(eta, s1, s2, S)
                    f_p5 = getattr(XHMParameterFits, f"phase_p5_{lm_str}")(eta, s1, s2, S)

                    f_nodes = torch.stack(c_pts, dim=-1)
                    f_damp_u_nodes = f_damp_lm.unsqueeze(-1) if f_damp_lm.dim() > 0 else f_damp_lm
                    f_ring_u_nodes = f_ring_lm.unsqueeze(-1) if f_ring_lm.dim() > 0 else f_ring_lm
                    lorentz_nodes = f_damp_u_nodes / ((f_damp_u_nodes**2) + (f_nodes - f_ring_u_nodes)**2)
                    m_phase = torch.stack([
                        lorentz_nodes,
                        torch.ones_like(f_nodes),
                        1.0 / f_nodes,
                        1.0 / (f_nodes**2),
                        1.0 / (f_nodes**4),
                    ], dim=-1)
                    rhs_phase = torch.stack([f_p1, f_p2, f_p3, f_p4, f_p5], dim=-1)
                    a_int = torch.linalg.solve(m_phase, rhs_phase)

                    a_lam = a_int[..., 0]
                    a_0 = a_int[..., 1]
                    a_1 = a_int[..., 2]
                    a_2 = a_int[..., 3]
                    a_4 = a_int[..., 4]

                    def eval_dphi_int(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a_lam_u = _unsqueeze_to(a_lam, f_in)
                        a_0_u = _unsqueeze_to(a_0, f_in)
                        a_1_u = _unsqueeze_to(a_1, f_in)
                        a_2_u = _unsqueeze_to(a_2, f_in)
                        a_4_u = _unsqueeze_to(a_4, f_in)
                        lor = damp_u / ((damp_u**2) + (f_in - ring_u)**2)
                        return (
                            a_lam_u * lor
                            + a_0_u
                            + a_1_u / f_in
                            + a_2_u / (f_in**2)
                            + a_4_u / (f_in**4)
                        )

                    def eval_phi_int(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a_lam_u = _unsqueeze_to(a_lam, f_in)
                        a_0_u = _unsqueeze_to(a_0, f_in)
                        a_1_u = _unsqueeze_to(a_1, f_in)
                        a_2_u = _unsqueeze_to(a_2, f_in)
                        a_4_u = _unsqueeze_to(a_4, f_in)
                        return (
                            a_lam_u * torch.atan((f_in - ring_u) / damp_u)
                            + a_0_u * f_in
                            + a_1_u * torch.log(torch.clamp(f_in, min=1e-12))
                            - a_2_u / f_in
                            - (1.0 / 3.0) * a_4_u / (f_in**3)
                        )

                    # Ringdown phase connection at f_rd_p
                    alpha_2_lm = (f_damp["22"] / f_damp_lm) * alpha2_22
                    dphi_rd_val = eval_dphi_int(f_rd_p)
                    lor_rd_at_cut = f_damp_lm / ((f_damp_lm**2) + (f_rd_p - f_ring_lm)**2)
                    dt_rd = dphi_rd_val - (
                        alpha_2_lm * (f_ring_lm**2) / (f_rd_p**2) + alpha_lambda_22 * lor_rd_at_cut
                    )
                    phi_int_at_cut = eval_phi_int(f_rd_p)
                    phi_rd_base = (
                        -alpha_2_lm * (f_ring_lm**2) / f_rd_p
                        + alpha_lambda_22 * torch.atan((f_rd_p - f_ring_lm) / f_damp_lm)
                        + dt_rd * f_rd_p
                    )
                    dphi_rd_shift = phi_int_at_cut - phi_rd_base

                    def eval_phi_rd(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp_lm, f_in)
                        ring_u = _unsqueeze_to(f_ring_lm, f_in)
                        a2_u = _unsqueeze_to(alpha_2_lm, f_in)
                        alam_u = _unsqueeze_to(alpha_lambda_22, f_in)
                        dt_u = _unsqueeze_to(dt_rd, f_in)
                        shift_u = _unsqueeze_to(dphi_rd_shift, f_in)
                        return (
                            -a2_u * (ring_u**2) / f_in
                            + alam_u * torch.atan((f_in - ring_u) / damp_u)
                            + dt_u * f_in
                            + shift_u
                        )

                    # Alignment frequency f_align and target phase per Section 3 of SPEC-TW-2026-09-11-PARITY
                    f_align_geom = torch.where(eta > 0.05, 0.6 * (m_flt / 2.0) * f_meco_22, (m_flt / 2.0) * f_meco_22)
                    f_align_hz = f_align_geom / m_sec

                    delta_target = (l % 2) * PI - (3.0 * PI / 8.0) * (m_flt - 2.0)
                    if l == 2 and m == 1:
                        ff = 0.008
                        a21_pn = (
                            -16.0 * delta * eta * ff * (math.pi**1.5) / (3.0 * math.sqrt(5.0))
                            + 4.0 * (2.0**(1.0/3.0)) * ((s1 - s2) + delta * (s1 + s2)) * eta * (ff**(4.0/3.0)) * (math.pi**(11.0/6.0)) / math.sqrt(5.0)
                            + 2.0 * (2.0**(2.0/3.0)) * eta * (306.0 * delta - 360.0 * delta * eta) * (ff**(5.0/3.0)) * (math.pi**(13.0/6.0)) / (189.0 * math.sqrt(5.0))
                        )
                        delta_target = delta_target + torch.where(a21_pn >= 0, torch.tensor(PI, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype))

                    # Evaluate carrier phase on scaled frequency grid (2/m) * f
                    f_scaled_grid = (2.0 / m_flt) * f_eval
                    _, _, phi_22_scaled = self.base_22(
                        f=f_scaled_grid,
                        mass1=m1,
                        mass2=m2,
                        chi1=s1,
                        chi2=s2,
                        distance=dist,
                        inclination=0.0,
                        phic=phic,
                        tc=tc,
                        f_ref=f_ref,
                        phasing_convention=phasing_convention,
                        reference_frame=reference_frame,
                        return_phase=True,
                    )
                    delta_target_u = _unsqueeze_to(delta_target, mf_grid)
                    phi_ins_grid = (m_flt / 2.0) * phi_22_scaled + delta_target_u

                    # C0 continuity connection at f_ins_p
                    carrier_p_at_ins, _ = eval_carrier_phi_and_deriv((2.0 / m_flt) * f_ins_p)
                    phi_ins_at_ins = (m_flt / 2.0) * carrier_p_at_ins + delta_target
                    dphi_int_shift = phi_ins_at_ins - eval_phi_int(f_ins_p)
                    dphi_int_shift_u = _unsqueeze_to(dphi_int_shift, mf_grid)

                    phi_int_grid = eval_phi_int(mf_grid) + dphi_int_shift_u
                    phi_rd_grid = eval_phi_rd(mf_grid) + dphi_int_shift_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    psi_lm = torch.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_grid,
                        torch.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_grid,
                            phi_rd_grid,
                        ),
                    )
                    exp_neg_i_psi = torch.complex(torch.cos(psi_lm), -torch.sin(psi_lm))
                    h_mode = amp_lm_phys.to(exp_neg_i_psi.dtype) * exp_neg_i_psi

                else:
                    # ---------------------------------------------------------
                    # Mode (3, 2): Spherical-Spheroidal Mode Mixing
                    # ---------------------------------------------------------
                    alpha2_32 = alpha2_22
                    g1 = XHMParameterFits.phase_32_s_dphi_p1(eta, s1, s2, S)
                    g2 = XHMParameterFits.phase_32_s_dphi_fring32(eta, s1, s2, S)
                    g3 = XHMParameterFits.phase_32_s_dphi_p4(eta, s1, s2, S)

                    f_nodes_s = torch.stack([
                        f_ring["22"],
                        f_ring["32"],
                        f_ring["32"] + 0.5 * f_damp["32"],
                    ], dim=-1)

                    f_damp_32_u = f_damp["32"].unsqueeze(-1) if f_damp["32"].dim() > 0 else f_damp["32"]
                    f_ring_32_u = f_ring["32"].unsqueeze(-1) if f_ring["32"].dim() > 0 else f_ring["32"]
                    lor_nodes_s = f_damp_32_u / ((f_damp_32_u**2) + (f_nodes_s - f_ring_32_u)**2)
                    m_s = torch.stack([
                        1.0 / (f_nodes_s**4),
                        lor_nodes_s,
                        torch.ones_like(f_nodes_s),
                    ], dim=-1)
                    rhs_s = torch.stack([
                        g1 - alpha2_32 / (f_nodes_s[..., 0]**2),
                        g2 - alpha2_32 / (f_nodes_s[..., 1]**2),
                        g3 - alpha2_32 / (f_nodes_s[..., 2]**2),
                    ], dim=-1)
                    sol_s = torch.linalg.solve(m_s, rhs_s)

                    alpha4_32 = sol_s[..., 0]
                    alpha_lam_32 = sol_s[..., 1]
                    dt_rd_32 = sol_s[..., 2]

                    # Alignment to (2,2) carrier in spheroidal basis
                    f0_align = f_ring["22"] + f_damp["22"]
                    f1_align = f_ring["22"]

                    delta_t_32_s = XHMParameterFits.phase_32_s_deltadphi(eta, s1, s2, S)
                    delta_phi_32_s = XHMParameterFits.phase_32_s_deltaphi(eta, s1, s2, S)

                    _, carrier_dp_at_f0 = eval_carrier_phi_and_deriv(f0_align)
                    carrier_p_at_f1, _ = eval_carrier_phi_and_deriv(f1_align)

                    def eval_dphi_32_s(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a2_u = _unsqueeze_to(alpha2_32, f_in)
                        a4_u = _unsqueeze_to(alpha4_32, f_in)
                        alam_u = _unsqueeze_to(alpha_lam_32, f_in)
                        dt_u = _unsqueeze_to(dt_rd_32, f_in)
                        lor = damp_u / ((damp_u**2) + (f_in - ring_u)**2)
                        return (
                            a2_u / (f_in**2)
                            + a4_u / (f_in**4)
                            + alam_u * lor
                            + dt_u
                        )

                    time_shift_32 = (carrier_dp_at_f0 + delta_t_32_s) - eval_dphi_32_s(f0_align)
                    dt_rd_32 = dt_rd_32 + time_shift_32

                    phi_32_s_base_at_f1 = (
                        -alpha2_32 / f1_align
                        - (1.0 / 3.0) * alpha4_32 / (f1_align**3)
                        + alpha_lam_32 * torch.atan((f1_align - f_ring["32"]) / f_damp["32"])
                        + dt_rd_32 * f1_align
                    )
                    phi_rd_32_s_shift = (carrier_p_at_f1 + delta_phi_32_s) - phi_32_s_base_at_f1

                    def eval_phi_32_s(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a2_u = _unsqueeze_to(alpha2_32, f_in)
                        a4_u = _unsqueeze_to(alpha4_32, f_in)
                        alam_u = _unsqueeze_to(alpha_lam_32, f_in)
                        dt_u = _unsqueeze_to(dt_rd_32, f_in)
                        shift_u = _unsqueeze_to(phi_rd_32_s_shift, f_in)
                        return (
                            -a2_u / f_in
                            - (1.0 / 3.0) * a4_u / (f_in**3)
                            + alam_u * torch.atan((f_in - ring_u) / damp_u)
                            + dt_u * f_in
                            + shift_u
                        )

                    # Intermediate phase for (3,2)
                    f_p1_32 = XHMParameterFits.phase_p1_32(eta, s1, s2, S)
                    f_p2_32 = XHMParameterFits.phase_p2_32(eta, s1, s2, S)
                    f_p3_32 = XHMParameterFits.phase_p3_32(eta, s1, s2, S)
                    f_p4_32 = XHMParameterFits.phase_p4_32(eta, s1, s2, S)

                    f_nodes_int_32 = torch.stack(c_pts[:4], dim=-1)

                    def lor_32(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        return damp_u / ((damp_u**2) + (f_in - ring_u)**2)

                    def dlor_32(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        return -2.0 * damp_u * (f_in - ring_u) / (((damp_u**2) + (f_in - ring_u)**2)**2)

                    eps_diff = 1e-6
                    dphi_rd_at_join = eval_dphi_32_s(f_rd_p)
                    d2phi_rd_at_join = (eval_dphi_32_s(f_rd_p + eps_diff) - eval_dphi_32_s(f_rd_p - eps_diff)) / (2.0 * eps_diff)

                    rows_32 = []
                    for i_pt in range(4):
                        fn_i = f_nodes_int_32[..., i_pt]
                        rows_32.append(torch.stack([
                            lor_32(fn_i), torch.ones_like(fn_i), 1.0 / fn_i, 1.0 / (fn_i**2), 1.0 / (fn_i**3), 1.0 / (fn_i**4)
                        ], dim=-1))
                    rows_32.append(torch.stack([
                        lor_32(f_rd_p), torch.ones_like(f_rd_p), 1.0 / f_rd_p, 1.0 / (f_rd_p**2), 1.0 / (f_rd_p**3), 1.0 / (f_rd_p**4)
                    ], dim=-1))
                    rows_32.append(torch.stack([
                        dlor_32(f_rd_p), torch.zeros_like(f_rd_p), -1.0 / (f_rd_p**2), -2.0 / (f_rd_p**3), -3.0 / (f_rd_p**4), -4.0 / (f_rd_p**5)
                    ], dim=-1))

                    m_int_32 = torch.stack(rows_32, dim=-2)
                    rhs_int_32 = torch.stack([f_p1_32, f_p2_32, f_p3_32, f_p4_32, dphi_rd_at_join, d2phi_rd_at_join], dim=-1)
                    sol_int_32 = torch.linalg.solve(m_int_32, rhs_int_32)

                    a_lam_32 = sol_int_32[..., 0]
                    a_0_32 = sol_int_32[..., 1]
                    a_1_32 = sol_int_32[..., 2]
                    a_2_32 = sol_int_32[..., 3]
                    a_3_32 = sol_int_32[..., 4]
                    a_4_32 = sol_int_32[..., 5]

                    def eval_dphi_int_32(f_in: torch.Tensor) -> torch.Tensor:
                        a_lam_u = _unsqueeze_to(a_lam_32, f_in)
                        a_0_u = _unsqueeze_to(a_0_32, f_in)
                        a_1_u = _unsqueeze_to(a_1_32, f_in)
                        a_2_u = _unsqueeze_to(a_2_32, f_in)
                        a_3_u = _unsqueeze_to(a_3_32, f_in)
                        a_4_u = _unsqueeze_to(a_4_32, f_in)
                        return (
                            a_lam_u * lor_32(f_in)
                            + a_0_u
                            + a_1_u / f_in
                            + a_2_u / (f_in**2)
                            + a_3_u / (f_in**3)
                            + a_4_u / (f_in**4)
                        )

                    def eval_phi_int_32(f_in: torch.Tensor) -> torch.Tensor:
                        damp_u = _unsqueeze_to(f_damp["32"], f_in)
                        ring_u = _unsqueeze_to(f_ring["32"], f_in)
                        a_lam_u = _unsqueeze_to(a_lam_32, f_in)
                        a_0_u = _unsqueeze_to(a_0_32, f_in)
                        a_1_u = _unsqueeze_to(a_1_32, f_in)
                        a_2_u = _unsqueeze_to(a_2_32, f_in)
                        a_3_u = _unsqueeze_to(a_3_32, f_in)
                        a_4_u = _unsqueeze_to(a_4_32, f_in)
                        return (
                            a_lam_u * torch.atan((f_in - ring_u) / damp_u)
                            + a_0_u * f_in
                            + a_1_u * torch.log(torch.clamp(f_in, min=1e-12))
                            - a_2_u / f_in
                            - 0.5 * a_3_u / (f_in**2)
                            - (1.0 / 3.0) * a_4_u / (f_in**3)
                        )

                    c_int_32 = eval_phi_32_s(f_rd_p) - eval_phi_int_32(f_rd_p)

                    # Alignment frequency f_align and target phase per Section 3 of SPEC-TW-2026-09-11-PARITY
                    f_align_geom_32 = torch.where(eta > 0.05, 0.6 * f_meco_22, f_meco_22)
                    f_align_hz_32 = f_align_geom_32 / m_sec
                    delta_target_32 = PI
                    phi_ins_32_grid = phi_22_carrier + delta_target_32

                    carrier_p_at_ins, _ = eval_carrier_phi_and_deriv(f_ins_p)
                    phi_ins_at_ins_32 = carrier_p_at_ins + delta_target_32
                    dphi_int_shift_32 = phi_ins_at_ins_32 - (eval_phi_int_32(f_ins_p) + c_int_32)
                    dphi_int_shift_32_u = _unsqueeze_to(dphi_int_shift_32, mf_grid)
                    c_int_32_u = _unsqueeze_to(c_int_32, mf_grid)

                    phi_int_32_grid = eval_phi_int_32(mf_grid) + c_int_32_u + dphi_int_shift_32_u
                    phi_rd_32_s_grid = eval_phi_32_s(mf_grid) + dphi_int_shift_32_u

                    f_ins_p_u = _unsqueeze_to(f_ins_p, mf_grid)
                    f_rd_p_u = _unsqueeze_to(f_rd_p, mf_grid)
                    phi_32_imr = torch.where(
                        mf_grid <= f_ins_p_u,
                        phi_ins_32_grid,
                        torch.where(
                            mf_grid <= f_rd_p_u,
                            phi_int_32_grid,
                            phi_rd_32_s_grid,
                        ),
                    )

                    # Complex mixing coefficients from XHMParameterFits
                    alpha322_re, alpha322_im = XHMParameterFits.mixing_alpha322(chi_f)
                    alpha333_re, alpha333_im = XHMParameterFits.mixing_alpha333(chi_f)
                    alpha322 = torch.complex(alpha322_re, alpha322_im)
                    alpha332 = torch.complex(alpha333_re, alpha333_im)

                    alpha322_u = _unsqueeze_to(alpha322, mf_grid)
                    alpha332_u = _unsqueeze_to(alpha332, mf_grid)

                    # Mode strain in spheroidal basis for ringdown:
                    exp_neg_i_psi_32_s = torch.complex(torch.cos(phi_rd_32_s_grid), -torch.sin(phi_rd_32_s_grid))
                    h32_s = amp_lm_phys.to(exp_neg_i_psi_32_s.dtype) * exp_neg_i_psi_32_s

                    # Transformed strain for ringdown (f > f_rd_a = f_ring_22)
                    h32_mixed = alpha322_u * h22_mode_strain + alpha332_u * h32_s

                    # Unmixed strain for f <= f_rd_a
                    exp_neg_i_psi_32 = torch.complex(torch.cos(phi_32_imr), -torch.sin(phi_32_imr))
                    h32_unmixed = amp_lm_phys.to(exp_neg_i_psi_32.dtype) * exp_neg_i_psi_32

                    f_rd_a_u = _unsqueeze_to(f_rd_a, mf_grid)
                    h_mode = torch.where(
                        mf_grid <= f_rd_a_u,
                        h32_unmixed,
                        h32_mixed,
                    )

            # Apply body exchange swap phase (-1)^m for odd-m modes when swapped
            if m % 2 != 0:
                h_mode = torch.where(_unsqueeze_to(swap_mask, h_mode), -h_mode, h_mode)

            # Spin-weighted spherical harmonics projection
            d_plus, d_cross = _swsh_projection_coefficients(l, m, inc_eval, phi0_eval)
            hp_lm = d_plus * h_mode
            hc_lm = d_cross * h_mode

            h_plus_total = h_plus_total + hp_lm
            h_cross_total = h_cross_total + hc_lm

            if return_modes:
                mode_dict[(l, m)] = (hp_lm, hc_lm)
            elif return_unprojected_modes:
                mode_dict[(l, m)] = h_mode
                # Negative-m mode satisfies equatorial reflection symmetry: h_{l, -m} = (-1)^l * conj(h_{l, m})
                sign_l = 1.0 if l % 2 == 0 else -1.0
                mode_dict[(l, -m)] = sign_l * torch.conj(h_mode)

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus_total = torch.where(f_eval > 0.0, h_plus_total, torch.zeros_like(h_plus_total))
        h_cross_total = torch.where(f_eval > 0.0, h_cross_total, torch.zeros_like(h_cross_total))

        if return_modes or return_unprojected_modes:
            return h_plus_total, h_cross_total, mode_dict
        return h_plus_total, h_cross_total

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
