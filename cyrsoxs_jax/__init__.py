"""CyRSoXS-JAX: JAX port of CyRSoXS for resonant soft X-ray scattering simulation."""

from cyrsoxs_jax.detector import (
    rotate_image,
    rotate_image_with_mask,
    average_rotations,
    rotate_and_average,
    apply_detector_rotation,
)

__version__ = "0.1.0"

__all__ = [
    "rotate_image",
    "rotate_image_with_mask",
    "average_rotations",
    "rotate_and_average",
    "apply_detector_rotation",
]
