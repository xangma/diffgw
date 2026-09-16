"""Clean-room Kerr quasi-normal mode (QNM) eigenproblem solver and table generation."""

from torchwave.qnm.solver import (
    KerrRoot,
    solve,
    angular_eigenvalue,
    radial_residual,
    angular_verification_report,
)
from torchwave.qnm.generate import (
    MODES,
    continuation,
    interpolator,
    render_splines,
    main as generate_main,
)
from torchwave.qnm import london

__all__ = [
    "KerrRoot",
    "solve",
    "angular_eigenvalue",
    "radial_residual",
    "angular_verification_report",
    "MODES",
    "continuation",
    "interpolator",
    "render_splines",
    "generate_main",
    "london",
]

