"""CyRSoXS-JAX: JAX port of CyRSoXS for resonant soft X-ray scattering simulation."""

from cyrsoxs_jax.scatter import (
    scatter_3d,
    scatter_3d_batched,
    compute_q_grid_3d,
    project_polarization_magnitude_squared,
)

__version__ = "0.1.0"

__all__ = [
    "scatter_3d",
    "scatter_3d_batched",
    "compute_q_grid_3d",
    "project_polarization_magnitude_squared",
]
