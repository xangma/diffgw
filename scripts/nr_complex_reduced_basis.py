"""Small reduced-basis regression for independently supplied NR spectra.

The input ``values`` are assumed to be canonically normalized by the caller.
This module deliberately contains no waveform, gauge, or physics conventions:
it performs an un-centred complex SVD and polynomial regression only.

The mass-ratio convention is ``q >= 1``.  Regression uses the intrinsic
coordinate ``x = 4*q/(1+q)**2 - 1``, affinely scaled over the training range
to improve conditioning.  Predictions outside that range are rejected unless
``allow_extrapolation=True`` is passed to :meth:`ComplexReducedBasisModel.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


Tensor = torch.Tensor


def _complex_parts(t: Tensor) -> Dict[str, Any]:
    return {"real": t.real.tolist(), "imag": t.imag.tolist()}


def _from_complex_parts(d: Dict[str, Any], *, dtype: torch.dtype = torch.complex128) -> Tensor:
    real = torch.tensor(d["real"], dtype=torch.float64)
    imag = torch.tensor(d["imag"], dtype=torch.float64)
    return torch.complex(real, imag).to(dtype=dtype)


@dataclass(frozen=True)
class ComplexReducedBasisModel:
    """Fitted complex reduced basis and polynomial coordinate model."""

    basis: Tensor  # [rank, modes * bins], orthonormal rows in flattened space
    polynomial: Tensor  # [degree + 1, rank], coefficients in scaled x powers
    singular_values: Tensor
    q_min: float
    q_max: float
    x_center: float
    x_halfspan: float
    rank: int
    degree: int
    modes: int
    bins: int

    def predict(self, q: Tensor, *, allow_extrapolation: bool = False) -> Tensor:
        """Evaluate the fitted spectra at scalar or one-dimensional ``q``.

        The query computation is torch-only and keeps autograd through ``q``.
        The returned shape is ``[modes, bins]`` for a scalar and
        ``[batch, modes, bins]`` for a one-dimensional input.
        """
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q)
        if q.ndim not in (0, 1):
            raise ValueError(f"q must be scalar or 1D, got shape {tuple(q.shape)}")
        if not (q.is_floating_point() or q.is_complex()):
            raise TypeError("q must be a real floating-point tensor")
        if q.is_complex():
            raise TypeError("q must be real, not complex")
        q = q.to(dtype=torch.float64, device=self.basis.device)
        if not torch.isfinite(q).all().item():
            raise ValueError("q must contain only finite values")
        if (q < 1).any().item():
            raise ValueError("q must satisfy q >= 1")
        if not allow_extrapolation and ((q < self.q_min).any().item() or (q > self.q_max).any().item()):
            raise ValueError(
                f"q is outside the training range [{self.q_min:g}, {self.q_max:g}]; "
                "pass allow_extrapolation=True to override"
            )

        x = 4.0 * q / (1.0 + q).square() - 1.0
        z = (x - self.x_center) / self.x_halfspan
        powers = torch.stack([z.pow(k) for k in range(self.degree + 1)], dim=-1)
        coordinates = powers.to(torch.complex128) @ self.polynomial
        flat = coordinates @ self.basis
        if q.ndim == 0:
            return flat.reshape(self.modes, self.bins)
        return flat.reshape(q.shape[0], self.modes, self.bins)

    __call__ = predict

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible state made of real/imaginary lists."""
        return {
            "basis": _complex_parts(self.basis),
            "polynomial": _complex_parts(self.polynomial),
            "singular_values": self.singular_values.tolist(),
            "q_min": self.q_min,
            "q_max": self.q_max,
            "x_center": self.x_center,
            "x_halfspan": self.x_halfspan,
            "rank": self.rank,
            "degree": self.degree,
            "modes": self.modes,
            "bins": self.bins,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "ComplexReducedBasisModel":
        """Reconstruct a model returned by :meth:`to_dict`."""
        basis = _from_complex_parts(state["basis"])
        polynomial = _from_complex_parts(state["polynomial"])
        singular_values = torch.tensor(state["singular_values"], dtype=torch.float64)
        return cls(
            basis=basis,
            polynomial=polynomial,
            singular_values=singular_values,
            q_min=float(state["q_min"]),
            q_max=float(state["q_max"]),
            x_center=float(state["x_center"]),
            x_halfspan=float(state["x_halfspan"]),
            rank=int(state["rank"]),
            degree=int(state["degree"]),
            modes=int(state["modes"]),
            bins=int(state["bins"]),
        )


def _validate_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def fit_complex_basis(
    q: Tensor,
    values: Tensor,
    *,
    rank: int,
    degree: int,
    weights: Optional[Tensor] = None,
) -> ComplexReducedBasisModel:
    """Fit a truncated complex SVD basis and polynomial coordinates.

    Parameters are deliberately small and explicit.  ``q`` has shape ``[n]``;
    ``values`` has shape ``[n, modes, bins]`` and must be complex.  The SVD is
    un-centred, and the retained basis coordinates are fitted by weighted
    least squares in powers of a scaled intrinsic mass-ratio coordinate.
    """
    _validate_integer("rank", rank)
    _validate_integer("degree", degree)
    if rank < 1:
        raise ValueError("rank must be positive")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if not isinstance(q, torch.Tensor) or q.ndim != 1:
        raise ValueError("q must be a one-dimensional tensor")
    if not (q.is_floating_point() and not q.is_complex()):
        raise TypeError("q must be a real floating-point tensor")
    if q.numel() < 3:
        raise ValueError("at least three training samples are required")
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("values must have shape [n, modes, bins]")
    n, modes, bins = values.shape
    if n != q.numel() or modes < 1 or bins < 1:
        raise ValueError("values shape must be [len(q), modes >= 1, bins >= 1]")
    if not values.is_complex():
        raise TypeError("values must be a complex tensor")

    q = q.to(dtype=torch.float64)
    values = values.to(dtype=torch.complex128, device=q.device)
    if not torch.isfinite(q).all().item() or not torch.isfinite(values).all().item():
        raise ValueError("q and values must contain only finite values")
    if (q < 1).any().item():
        raise ValueError("q must satisfy q >= 1")

    if weights is None:
        weights = torch.ones(n, dtype=torch.float64, device=q.device)
    elif not isinstance(weights, torch.Tensor) or weights.ndim != 1 or weights.numel() != n:
        raise ValueError("weights must have shape [len(q)]")
    else:
        if weights.is_complex() or not weights.is_floating_point():
            raise TypeError("weights must be a real floating-point tensor")
        weights = weights.to(dtype=torch.float64, device=q.device)
    if not torch.isfinite(weights).all().item() or (weights < 0).any().item() or weights.sum().item() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive sum")

    intrinsic = 4.0 * q / (1.0 + q).square() - 1.0
    q_min = float(q.min().item())
    q_max = float(q.max().item())
    x_min = float(intrinsic.min().item())
    x_max = float(intrinsic.max().item())
    if degree > 0 and x_max <= x_min:
        raise ValueError("training q values must contain enough independent samples")
    x_center = 0.5 * (x_min + x_max)
    x_halfspan = 0.5 * (x_max - x_min) if x_max > x_min else 1.0
    z = (intrinsic - x_center) / x_halfspan
    design = torch.stack([z.pow(k) for k in range(degree + 1)], dim=-1)
    weighted_design = (design * weights.sqrt()[:, None]).to(torch.complex128)
    if torch.linalg.matrix_rank(weighted_design) < degree + 1:
        raise ValueError("training q values and positive weights are insufficiently independent for this degree")

    flat = values.reshape(n, modes * bins)
    u, singular_values, vh = torch.linalg.svd(flat, full_matrices=False)
    del u
    available = int(torch.linalg.matrix_rank(flat).item())
    if rank > min(n, modes * bins):
        raise ValueError(f"rank must be <= min(n, modes*bins)={min(n, modes * bins)}")
    if rank > available:
        raise ValueError(f"rank={rank} exceeds numerical data rank={available}")
    basis = vh[:rank]
    coordinates = flat @ basis.conj().transpose(-2, -1)
    weighted_coordinates = coordinates * weights.sqrt()[:, None]
    polynomial = torch.linalg.lstsq(weighted_design, weighted_coordinates).solution

    return ComplexReducedBasisModel(
        basis=basis,
        polynomial=polynomial,
        singular_values=singular_values,
        q_min=q_min,
        q_max=q_max,
        x_center=x_center,
        x_halfspan=x_halfspan,
        rank=rank,
        degree=degree,
        modes=modes,
        bins=bins,
    )


__all__ = ["ComplexReducedBasisModel", "fit_complex_basis"]
