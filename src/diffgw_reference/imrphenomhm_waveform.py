"""Independent, scalar HM composition for the named published calibration.

The default PhenomD coefficients and remnant fits are the printed Khan/Husa
values in the existing symbolic references. Only independently generated QNM
*data* are shared with production; no frontend evaluator or fit helper is used.
London's maps and Mishra's source PN variants are composed with the independent
Wigner projection. This is a bounded implementation reference, not a claim of
historical LAL calibration equivalence or NR accuracy across parameter space.
"""

from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import sympy as sp

from diffgw_core.hm_qnm_n61_data import N61_COEFFICIENTS
from diffgw_core.kerr_qnm_220_data import (
    DEGREE, KNOTS_220, COEFFS_220, KNOTS_2M20, COEFFS_2M20,
)
from diffgw_core.qnm_data import KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES
from . import imrphenomd_amplitude as damp
from . import imrphenomd_intrinsic as intrinsic
from . import imrphenomd_phase as dphase
from . import imrphenomd_waveform as carrier
from . import imrphenomhm_projection as projection
from .imrphenomhm_intrinsic_modes import SUPPORTED_MODES, public_intrinsic_mode
from .imrphenomhm_signed_projection import symbolic_phase_factor

SPEC_VERSION = "imrphenomhm-independent-waveform/v3"
SPA_CONVENTION_VERSION = "imrphenomhm-spa-sxs-carrier/v1"
REFERENCE_PHASE_VERSION = "imrphenomhm-orbital-reference-phase/v1"
COMPONENT_IDS = {
    "polarizations": "hm.waveform.independent_published.v1",
    "derivatives": "hm.waveform.independent_published_derivatives.v1",
    "orbital_polarizations": "hm.waveform.orbital_reference.polarizations.v1",
    "orbital_derivatives": "hm.waveform.orbital_reference.derivatives.v1",
    "orbital_covariance": "hm.waveform.orbital_reference.covariance.v1",
}
COMPONENT_SPEC_VERSIONS = {
    key: (REFERENCE_PHASE_VERSION if key.startswith("orbital_")
          else "imrphenomhm-independent-waveform/v1") for key in COMPONENT_IDS
}
COMPONENT_SOURCES = {
    component: (
        ("https://arxiv.org/pdf/1708.00404v4", "Eqs. (1)--(9)"),
        ("https://arxiv.org/html/1601.05588v1", "Eqs. (11)--(12)"),
        ("https://arxiv.org/html/1508.07253v2", "Table V; amplitude and phase"),
        ("https://arxiv.org/html/1508.07250v2", "Eqs. (17), (22)--(24)"),
        ("https://arxiv.org/pdf/1810.03550v2", "Eqs. (19)--(21): QNM polynomial"),
        ("generated-data", "independent Kerr roots and N=61 polynomial fit"),
    ) for component in COMPONENT_IDS.values()
}
COMPONENT_BINDINGS = {
    component: {
        "reference": "compose",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": (("test_published_smooth_derivatives",) if key == "derivatives"
                  else ("test_published_complete_modes_and_projection",
                        "test_published_symmetry_zeros")),
    } for key, component in COMPONENT_IDS.items()
}
for _key in (key for key in COMPONENT_IDS if key.startswith("orbital_")):
    _component = COMPONENT_IDS[_key]
    COMPONENT_SOURCES[_component] += (
        ("https://arxiv.org/html/2001.10914v2", "Eq. (14): orbital rotation law"),
        ("project-adapter", "HM orbital reference phase: C=pi/4-phi(r)+s*r; no extra mode gauge"),
    )
    COMPONENT_BINDINGS[_component]["tests"] = {
        "orbital_polarizations": ("test_orbital_complete_modes_and_projection",),
        "orbital_derivatives": ("test_orbital_reference_derivatives",),
        "orbital_covariance": ("test_orbital_reference_covariance", "test_reference_phase_reductions"),
    }[_key]
# Compose the previously derived signed-pair/SPA convention without changing
# either its equations or the existing published/orbital component versions.
for _key, _suffix, _tests in (
    ("spa_polarizations", "polarizations", ("test_spa_complete_modes_and_projection",)),
    ("spa_derivatives", "derivatives", ("test_spa_smooth_derivatives",
                                       "test_spa_reference_derivatives")),
):
    _component = f"hm.waveform.spa_sxs.{_suffix}.v1"
    COMPONENT_IDS[_key] = _component
    COMPONENT_SPEC_VERSIONS[_key] = SPA_CONVENTION_VERSION
    COMPONENT_SOURCES[_component] = COMPONENT_SOURCES[COMPONENT_IDS["polarizations"]] + (
        ("https://arxiv.org/pdf/0802.1249v3", "Eqs. (8.2)--(8.3), (9.3)--(9.4): leading mode phases"),
        ("https://arxiv.org/pdf/1904.04831v2", "Appendix C, Eqs. (C21)--(C24): SXS basis"),
        ("https://arxiv.org/html/2001.10914v2", "Eqs. (2), (4)--(9): signed-mode projection"),
        ("project-adapter", "22-preserving source-derived SPA/SXS phase, K=3*pi/4"),
    )
    COMPONENT_BINDINGS[_component] = {
        "reference": "compose",
        "torch": ("torchwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "jax": ("jaxwave.approximants.imrphenom_hm", "IMRPhenomHM"),
        "tests": _tests,
    }


@lru_cache(maxsize=None)
def _spa_phase_function(m):
    return sp.lambdify((), symbolic_phase_factor(m, 3 * sp.pi / 4),
                       modules="mpmath", docstring_limit=0)


SOURCE_PN_VARIANTS = {
    "hm_21_pn_variant": "mishra_eq12b",
    "hm_32_pn_variant": "mishra_eq12d",
    "hm_43_pn_variant": "mishra_eq12g",
}
BRANCHES = ("inspiral", "intermediate", "merger_ringdown")


def _number(value):
    return mp.mpf(str(sp.N(value, 60))) if isinstance(value, sp.Basic) else mp.mpf(str(value))


def qnm_frequencies(ell, m, final_spin, total_mass, final_mass):
    """Independent London polynomial sum, returning total-mass cyclic units."""
    spin = _number(final_spin)
    kappa = (mp.log(2 - spin) / mp.log(3)) ** (mp.mpf(1) / (2 + ell - abs(m)))
    coefficients = N61_COEFFICIENTS[(ell, m)]
    polynomial = mp.fsum(mp.mpc(_number(re), _number(im)) * kappa**j
                         for j, (re, im) in enumerate(coefficients))
    angular = (mp.mpf(m) / 2 + kappa * polynomial
               if (ell, m) in ((2, 2), (3, 3), (4, 4), (4, 3)) else polynomial)
    scale = _number(total_mass) / _number(final_mass) / (2 * mp.pi)
    return angular.real * scale, abs(angular.imag) * scale


@lru_cache(maxsize=None)
def _projection_functions(ell, m):
    return tuple(sp.lambdify((projection.IOTA, projection.AZIMUTH), expression,
                            modules="mpmath", docstring_limit=0)
                 for expression in projection.native_carrier_coefficients(ell, m))


@dataclass(frozen=True)
class WaveformSpec:
    """Cached independent carrier expressions and their scalar evaluators."""

    parameters: tuple
    carrier_spec: carrier.WaveformSpec
    final_mass: object
    final_spin: object
    base_qnm: tuple
    hm_qnm: dict
    alpha4: object
    alpha5: object
    m_sec: object
    slope: object
    amplitude_joins: tuple
    phase_joins: tuple
    amplitude_functions: tuple
    phase_functions: tuple

    def amplitude22(self, u):
        """Intrinsic 22 magnitude, including independently derived SI scale."""
        u = _number(u)
        if u > mp.mpf(".2"):
            return mp.mpf(0)
        branch = 0 if u <= self.amplitude_joins[0] else 1 if u <= self.amplitude_joins[1] else 2
        return abs(self.amplitude_functions[branch](u))

    def raw_phase(self, u):
        u = _number(u)
        branch = 0 if u <= self.phase_joins[0] else 1 if u <= self.phase_joins[1] else 2
        return self.phase_functions[branch](u)

    def positive_d_phase(self, u, f_ref_hz=None):
        u = _number(u)
        if u > mp.mpf(".2"):
            return mp.mpf(0)
        if f_ref_hz is None:
            return -mp.pi / 4 + self.raw_phase(u) - self.slope * u
        reference = self.m_sec * _number(f_ref_hz)
        if reference <= 0:
            raise ValueError("f_ref_hz must be positive")
        return self.raw_phase(u) - self.raw_phase(reference) - self.slope * (u - reference)

    def reference_phase_offset(self, f_ref_hz):
        """Independent carrier constant C(r); orbital modes transform by exp(-imC/2).

        This identity is restricted to mapped carrier phase evaluations below
        the D hard cutoff. It does not assert NR accuracy of the HM model.
        """
        if f_ref_hz is None:
            return mp.mpf(0)
        with mp.workdps(max(50, mp.mp.dps)):
            r = self.m_sec * _number(f_ref_hz)
            if r <= 0:
                raise ValueError("f_ref_hz must be positive")
            return mp.pi / 4 - self.raw_phase(r) + self.slope * r

    def evaluate(self, frequency_hz, *, transition="phenomd_runtime", f_ref_hz=None,
                 hm_reference_phase="legacy", hm_phase_convention="legacy",
                 inclination=0, phi0=0, phic=0, tc=0, modes=SUPPORTED_MODES):
        """Return (h_plus, h_cross, intrinsic modes) without frontend inputs.

        Frequencies are positive Hz; masses solar masses and distance Mpc.
        ``legacy`` preserves the v1 adapter. ``orbital`` changes the phase
        origin by m*C(r)/2 only, preserving the m=2 carrier. Neither choice
        changes the independent printed fits or mode maps.
        ``spa_sxs_v1`` applies the separately versioned leading-SPA factor
        before projection, with either reference-phase origin. It preserves
        m=2 and changes no fit, magnitude or frequency slope. The default
        ``hm_phase_convention="legacy"`` preserves the v1/v2 composition.
        Precision follows an enclosing mpmath context, with a 50-digit floor.
        """
        if hm_phase_convention not in ("legacy", "spa_sxs_v1"):
            raise ValueError("hm_phase_convention must be 'legacy' or 'spa_sxs_v1'")
        if hm_reference_phase not in ("legacy", "orbital"):
            raise ValueError("hm_reference_phase must be 'legacy' or 'orbital'")
        with mp.workdps(max(50, mp.mp.dps)):
            f = _number(frequency_hz)
            if f <= 0:
                raise ValueError("frequency_hz must be positive")
            x = self.m_sec * f
            m1, m2, s1, s2, _ = map(_number, self.parameters)
            gauge = (self.slope * self.m_sec * _number(f_ref_hz)
                     if f_ref_hz is not None and hm_reference_phase == "legacy" else 0)
            hp = hc = mp.mpc(0)
            result = {}
            for ell, m in modes:
                if (ell, m) not in SUPPORTED_MODES:
                    raise ValueError("unsupported HM mode")
                # The retained leading 33 term is proportional to mass
                # difference, including unequal-spin equal-mass binaries.
                if (ell, m) == (3, 3) and m1 == m2:
                    value = mp.mpc(0)
                else:
                    value = public_intrinsic_mode(
                        x, ell, m, m1, m2, s1, s2, self.amplitude22,
                        lambda u: self.positive_d_phase(u, f_ref_hz),
                        fr_lm=self.hm_qnm[(ell, m)][0], fr22=self.hm_qnm[(2, 2)][0],
                        fd_lm=self.hm_qnm[(ell, m)][1], fd22=self.hm_qnm[(2, 2)][1],
                        base_fr22=self.base_qnm[0], base_fd22=self.base_qnm[1],
                        alpha4=self.alpha4, alpha5=self.alpha5, transition=transition,
                        phic=phic, tc=tc, frequency_hz=f,
                        reference_gauge=(mp.mpf(m) / 2 - 1) * gauge,
                    )
                if hm_phase_convention == "spa_sxs_v1" and m != 2:
                    value *= _spa_phase_function(m)()
                result[(ell, m)] = value
                cp, cx = (fn(_number(inclination), _number(phi0))
                          for fn in _projection_functions(ell, m))
                hp += cp * value
                hc += cx * value
            return hp, hc, result


@lru_cache(maxsize=16)
def compose(mass1, mass2, chi1, chi2, distance_mpc=100):
    """Compose printed fits and clean-generated QNM data once per binary."""
    parameters = tuple(damp.exact(value) for value in
                       (mass1, mass2, chi1, chi2, distance_mpc))
    m1, m2, s1, s2, distance = parameters
    if not (m1 >= m2 > 0 and abs(s1) <= 1 and abs(s2) <= 1 and distance > 0):
        raise ValueError("require mass1>=mass2>0, |chi_i|<=1 and distance>0")
    with mp.workdps(60):
        final_mass, final_spin, fr, fd = intrinsic.remnant_qnm(
            m1, m2, s1, s2, knot_spins=KNOT_SPINS,
            cyclic_ordinates=ROUNDED_CYCLIC_ORDINATES,
            generated_splines={"degree": DEGREE,
                               "positive": (KNOTS_220, COEFFS_220),
                               "negative": (KNOTS_2M20, COEFFS_2M20)})
        spec = carrier.compose(m1, m2, s1, s2, distance, fr, fd)
        eta, _, _, _, chi_pn = damp.spin_coordinates(m1, m2, s1, s2)
        scale = carrier.physical_scale(damp.U / spec.m_sec, spec.m_sec,
                                       spec.d_sec, spec.eta) / sp.sqrt(5 / (16 * sp.pi))
        amplitude_functions = tuple(sp.lambdify(
            damp.U, sp.N(scale * spec.amplitude[branch], 60), modules="mpmath",
            docstring_limit=0, cse=True) for branch in BRANCHES)
        phase_functions = tuple(sp.lambdify(
            dphase.U, sp.N(spec.phase[branch], 60), modules="mpmath",
            docstring_limit=0, cse=True) for branch in BRANCHES)
        return WaveformSpec(
            parameters, spec, _number(final_mass), _number(final_spin),
            (_number(fr), _number(fd)),
            {lm: qnm_frequencies(*lm, final_spin, m1 + m2, final_mass)
             for lm in SUPPORTED_MODES},
            _number(dphase.fit_expr("alpha4", eta, chi_pn)),
            _number(dphase.fit_expr("alpha5", eta, chi_pn)),
            _number(spec.m_sec), _number(spec.slope),
            tuple(_number(spec.amplitude[key]) for key in ("f1", "f3")),
            tuple(_number(spec.phase[key]) for key in ("f1", "f2")),
            amplitude_functions, phase_functions,
        )
