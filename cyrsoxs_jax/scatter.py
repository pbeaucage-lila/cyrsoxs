"""Scatter3D and Ewald sphere projection for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array


def scatter_3d(polarization_fft: Array, q_grid: tuple[Array, Array, Array]) -> Array:
    """Compute 3D scattering amplitude from polarization FFT.

    This computes the scattering amplitude F(q) from the Fourier transform
    of the polarization field.

    Args:
        polarization_fft: FFT of polarization field (N, N, N, 3).
        q_grid: Tuple of (qx, qy, qz) arrays.

    Returns:
        3D scattering amplitude (N, N, N).
    """
    raise NotImplementedError("Scatter3D not yet implemented")


def ewald_sphere_projection(
    scatter_3d: Array,
    q_grid: tuple[Array, Array, Array],
    wavelength: float,
    detector_distance: float,
    detector_size: tuple[int, int],
) -> Array:
    """Project 3D scattering onto Ewald sphere for 2D detector.

    Args:
        scatter_3d: 3D scattering amplitude (N, N, N).
        q_grid: Tuple of (qx, qy, qz) arrays.
        wavelength: X-ray wavelength in nm.
        detector_distance: Sample-to-detector distance in mm.
        detector_size: (height, width) of detector in pixels.

    Returns:
        2D detector image.
    """
    raise NotImplementedError("Ewald sphere projection not yet implemented")


def apply_grazing_incidence(
    scatter_3d: Array,
    incident_angle: float,
    q_grid: tuple[Array, Array, Array],
) -> Array:
    """Apply grazing incidence geometry correction.

    Args:
        scatter_3d: 3D scattering amplitude.
        incident_angle: Grazing incidence angle in degrees.
        q_grid: Tuple of (qx, qy, qz) arrays.

    Returns:
        Corrected 3D scattering amplitude.
    """
    raise NotImplementedError("Grazing incidence correction not yet implemented")
