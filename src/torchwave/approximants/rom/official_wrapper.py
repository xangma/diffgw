"""Public-facing clean-room wrapper for the externally supplied SEOBNRv4 ROM.

The HDF5 release is resolved from an explicit path, then
``TORCHWAVE_DATA_PATH``, ``LAL_DATA_PATH``, or TorchWave's external cache. It
is never downloaded during model construction or imported through LAL.
Remnant/QNM inputs use TorchWave's independently generated rounded-knot table
convention by default.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import torch

from torchwave.constants import MSUN_SEC
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from .official import OFFICIAL_ROM_SHA256, resolve_official_rom_path, sha256_file
from .bspline import load_official_rom
from .glue import OfficialROMGlue

_UNSET = object()


def independent_ringdown_mf(mass1, mass2, chi1, chi2):
    """Return total-mass QNM frequency in cycles, from independent fits."""
    mf = eob_final_mass_fraction(mass1, mass2, chi1, chi2)
    af = hofmann_final_spin(mass1, mass2, chi1, chi2, fit_order=(3, 4))
    return signed_qnm_220(af)[0] / (2 * torch.pi * mf)


def table_convention_ringdown_mf(mass1, mass2, chi1, chi2):
    """Return ROM ringdown frequency using the validated rounded-knot convention."""
    mf = eob_final_mass_fraction(mass1, mass2, chi1, chi2)
    af = hofmann_final_spin(mass1, mass2, chi1, chi2, fit_order=(3, 4))
    return signed_qnm_220_table_convention(af)[0] / (2 * torch.pi * mf)


class OfficialSEOBNRv4ROM:
    """Evaluate the public external ROM without a LAL runtime dependency.

    ``model_path`` may name the HDF5 file or its containing directory.  If it
    is omitted, ``TORCHWAVE_DATA_PATH`` and then ``LAL_DATA_PATH`` are
    searched, followed by TorchWave's external cache. ``flow``/``fhigh`` are
    aliases for the effective lower and half-open upper frequency support in
    Hz. Pass ``ringdown_mf=independent_ringdown_mf`` to select the
    full-precision evaluator explicitly. Tables are evaluated on CPU float64;
    outputs are transferred back to the input frequency device and returned as
    complex64 for float32 input or complex128 otherwise.
    """
    def __init__(self, model_path=None, *, node_policy="31_first_above",
                 ringdown_mf=None, flow=None, fhigh=None):
        # Resolve without asking the data policy layer to hash.  Hash exactly
        # once here, before any HDF5 table is opened, so corrupted canonical
        # assets cannot reach the numerical loader.
        path = resolve_official_rom_path(model_path, verify=False)
        digest = sha256_file(path)
        if digest != OFFICIAL_ROM_SHA256:
            raise ValueError(
                f"Unexpected SHA-256 for {Path(path).name}: {digest}; "
                f"expected {OFFICIAL_ROM_SHA256}"
            )
        self.model_path = Path(path)
        self._flow = flow
        self._fhigh = fhigh
        low = load_official_rom(path, "sub1")
        high = load_official_rom(path, "sub2")
        high_spin = load_official_rom(path, "sub3")
        provider = table_convention_ringdown_mf if ringdown_mf is None else ringdown_mf
        self._normal = OfficialROMGlue(low, high, ringdown_mf=provider,
                                       node_policy=node_policy, f_lower_hz=flow, f_final_hz=fhigh)
        self._high_spin = OfficialROMGlue(low, high_spin, ringdown_mf=provider,
                                          node_policy=node_policy, f_lower_hz=flow, f_final_hz=fhigh)
        self._ringdown = provider

    def __call__(self, frequency_hz=None, mass1=None, mass2=None, chi1=0., chi2=0., *, f=None,
                 distance=_UNSET, distance_mpc=None, inclination=0., phic=0., tc=0., f_ref=None,
                 flow=None, fhigh=None):
        if frequency_hz is None:
            frequency_hz = f
        elif f is not None:
            raise TypeError("provide either frequency_hz or f, not both")
        if frequency_hz is None:
            raise TypeError("frequency_hz (or f) is required")
        if mass1 is None or mass2 is None:
            raise TypeError("mass1 and mass2 are required")
        distance_given = distance is not _UNSET
        if not distance_given:
            distance = 100.
        if distance_mpc is not None:
            if distance_given:
                raise TypeError("provide either distance or distance_mpc, not both")
            distance = distance_mpc
        # Select sub3 in the high-spin/low-eta box inferred from the public
        # release's parameter grids; this dispatch is an interface diagnostic.
        input_frequency = torch.as_tensor(frequency_hz)
        out_device = input_frequency.device
        out_dtype = torch.complex64 if input_frequency.dtype == torch.float32 else torch.complex128
        # ROM tables are CPU float64 data; these transfers remain differentiable.
        f = input_frequency.to(device="cpu", dtype=torch.float64)
        dtype = self._normal.low.amp_coeff.dtype
        device = torch.device("cpu")
        m1, m2, s1 = (torch.as_tensor(x, dtype=dtype, device=device) for x in (mass1, mass2, chi1))
        vals = [torch.as_tensor(x, dtype=dtype, device=device) for x in
                (m1, m2, s1, chi2, distance, inclination, phic, tc,
                 0.0 if f_ref is None else f_ref)]
        for name, value in (("distance", distance), ("inclination", inclination),
                            ("phic", phic), ("tc", tc), ("f_ref", f_ref)):
            if value is not None:
                check = torch.as_tensor(value, dtype=dtype, device=device)
                if not torch.isfinite(check).all():
                    raise ValueError(f"{name} must be finite")
        if f_ref is not None and torch.any(torch.as_tensor(f_ref, dtype=dtype) < 0):
            raise ValueError("f_ref must be non-negative or None")
        effective_flow = self._flow if flow is None else flow
        effective_fhigh = self._fhigh if fhigh is None else fhigh
        bound_values = [torch.as_tensor(x, dtype=dtype, device=device) for x in
                        (effective_flow, effective_fhigh) if x is not None]
        params = torch.broadcast_shapes(*(x.shape for x in vals),
                                        *(x.shape for x in bound_values))
        m1, m2, s1, s2, dist, inc, phi, tcv, fref = (x.expand(params) for x in vals)
        swap = m2 > m1
        old_m1, old_s1 = m1, s1
        m1 = torch.where(swap, m2, m1)
        m2 = torch.where(swap, old_m1, m2)
        s1 = torch.where(swap, s2, s1)
        s2 = torch.where(swap, old_s1, s2)
        eta = (m1*m2)/(m1+m2).square()
        highspin = (s1 >= .995) & (eta <= .025)
        # Scalar calls preserve gradients and avoid mixing incompatible table domains.
        vals_p, vals_c = [], []
        for idx in (product(*(range(n) for n in params)) if params else [()]):
            # Bounds may be broadcast per intrinsic point just like the
            # physical parameters; select the scalar bound for this engine
            # call rather than passing a batch tensor to the scalar spline.
            def _bound(value):
                if value is None:
                    return None
                bound = torch.as_tensor(value, dtype=dtype, device=device)
                return bound.expand(params)[idx] if params else bound
            kw = dict(distance_mpc=dist[idx], inclination=inc[idx], phic=phi[idx], tc=tcv[idx],
                      fref_hz=fref[idx] if f_ref is not None else torch.zeros_like(fref[idx]),
                      f_lower_hz=_bound(effective_flow), f_final_hz=_bound(effective_fhigh))
            args = (m1[idx], m2[idx], s1[idx], s2[idx])
            engine = self._high_spin if bool(highspin.expand(params)[idx]) else self._normal
            p, c = engine(f, *args, **kw)
            vals_p.append(p.to(device=out_device, dtype=out_dtype))
            vals_c.append(c.to(device=out_device, dtype=out_dtype))
        if not params:
            return vals_p[0], vals_c[0]
        return torch.stack(vals_p).reshape(params + p.shape), torch.stack(vals_c).reshape(params + c.shape)
