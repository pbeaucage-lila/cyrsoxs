"""CyRSoXS-JAX: JAX port of CyRSoXS for resonant soft X-ray scattering simulation."""

from cyrsoxs_jax.scatter import (
    scatter_3d,
    scatter_3d_batched,
    compute_q_grid_3d,
    project_polarization_magnitude_squared,
)
from cyrsoxs_jax.ewald import (
    ewald_projection,
    ewald_projection_from_polarization,
    compute_detector_q_grid,
    EwaldInterpolation,
)

__version__ = "0.1.0"

__all__ = [
    # Scatter3D
    "scatter_3d",
    "scatter_3d_batched",
    "compute_q_grid_3d",
    "project_polarization_magnitude_squared",
    # Ewald projection
    "ewald_projection",
    "ewald_projection_from_polarization",
    "compute_detector_q_grid",
    "EwaldInterpolation",
]
