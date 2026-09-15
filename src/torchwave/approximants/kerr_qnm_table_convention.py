"""Opt-in six-significant-digit natural-spline QNM compatibility convention.

The knot locations below were *inferred from black-box interpolation metadata*;
they are not published calibration data.  Values are regenerated from
TorchWave's independent ``signed_qnm_220`` solver and rounded componentwise to
six significant digits.  This module is diagnostic/compatibility-only and is
not imported by the default QNM path.
"""
import math
from functools import lru_cache
import torch

from .kerr_qnm_generated import signed_qnm_220

# Recovered interpolation-knot metadata (symmetric signed-spin grid).
KNOTS = tuple((
-0.9996, -0.9995, -0.9994, -0.9992, -0.999, -0.9989, -0.9988, -0.9987, -0.9986, -0.9985, -0.998, -0.9975, -0.997, -0.996, -0.995, -0.994, -0.992, -0.99, -0.988, -0.986, -0.984, -0.982, -0.98, -0.975, -0.97, -0.96, -0.95, -0.94, -0.92, -0.9, -0.88, -0.86, -0.84, -0.82, -0.8, -0.78, -0.76, -0.74, -0.72, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.9975, 0.998, 0.9985, 0.9986, 0.9987, 0.9988, 0.9989, 0.999, 0.9992, 0.9994, 0.9995, 0.9996
))


def _six(x):
    # Python's decimal formatting is componentwise and deterministic.
    return float(format(float(x), ".6g"))


@lru_cache(maxsize=1)
def _coefficients():
    # A first compatibility lookup is commonly made in inference mode.  Do
    # not let that mode mark the cached constants as inference tensors: such
    # tensors cannot safely participate in a later spin autograd graph.
    with torch.inference_mode(False), torch.no_grad():
        s = torch.tensor(KNOTS, dtype=torch.float64)
        raw = signed_qnm_220(s)
        y = torch.tensor([[_six(v) for v in row.tolist()] for row in raw], dtype=torch.float64)
    # y is [2,n]; natural cubic second derivatives, solved independently.
    h = s[1:] - s[:-1]
    n = s.numel()
    mat = torch.zeros((n, n), dtype=torch.float64)
    rhs = torch.zeros((n, 2), dtype=torch.float64)
    mat[0, 0] = mat[-1, -1] = 1.0
    for i in range(1, n - 1):
        mat[i, i - 1] = h[i - 1]
        mat[i, i] = 2.0 * (h[i - 1] + h[i])
        mat[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[:, i + 1] - y[:, i]) / h[i] - (y[:, i] - y[:, i - 1]) / h[i - 1])
    m = torch.linalg.solve(mat, rhs)  # [n,2]
    a = y[:, :-1].T
    b = (y[:, 1:] - y[:, :-1]).T / h[:, None] - h[:, None] * (2*m[:-1] + m[1:]) / 6.0
    c = m[:-1] / 2.0
    d = (m[1:] - m[:-1]) / (6.0 * h[:, None])
    return s, a, b, c, d, y


def signed_qnm_220_table_convention(spin):
    """Evaluate the opt-in rounded-knot natural cubic QNM convention.

    Returns ``(M_final*omega_real, M_final*|omega_imag|)`` with the input shape.  The
    input may be scalar or batched Torch floating point data and remains
    differentiable with respect to spin inside each spline interval.
    """
    if isinstance(spin, (float, int)):
        x = torch.tensor(spin, dtype=torch.float64)
    else:
        x = torch.as_tensor(spin)
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    if not torch.isfinite(x).all() or (x < -0.9996).any() or (x > 0.9996).any():
        raise ValueError("spin must be finite and in [-0.9996, 0.9996]")
    # MPS currently does not implement float64 indexing/arithmetic; use its
    # native precision there while retaining the caller's output dtype.  The
    # explicit mode guard also prevents a surrounding inference context from
    # turning the per-call copies into inference tensors before a later call
    # requests gradients.
    with torch.inference_mode(False):
        s, a, b, c, d, _ = _coefficients()
        calc_dtype = torch.float32 if x.device.type == "mps" else torch.float64
        s, a, b, c, d = (v.to(device=x.device, dtype=calc_dtype) for v in (s, a, b, c, d))
        work = x.to(device=x.device, dtype=calc_dtype)
        # ``right=True`` keeps an exact knot on the interval beginning there.
        idx = (torch.searchsorted(s, work.reshape(-1), right=True) - 1).clamp(min=0, max=s.numel() - 2)
        dx = work.reshape(-1) - s[idx]
        out = a[idx] + b[idx] * dx[:, None] + c[idx] * dx[:, None]**2 + d[idx] * dx[:, None]**3
        out = out.reshape(x.shape + (2,)).to(dtype=x.dtype)
    return out[..., 0], out[..., 1]


def rounded_knot_values():
    """Return regenerated rounded values, for provenance and validation."""
    return _coefficients()[-1].clone()
