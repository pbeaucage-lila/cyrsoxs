"""Detector image handling and rotation averaging for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array


def rotate_image(image: Array, angle: float) -> Array:
    """Rotate a 2D image by a given angle.

    Args:
        image: 2D detector image.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    raise NotImplementedError("Image rotation not yet implemented")


def average_rotations(
    images: Array,
    angles: Array,
) -> Array:
    """Average multiple rotated detector images.

    Args:
        images: Stack of 2D images (N, H, W).
        angles: Rotation angles for each image in degrees.

    Returns:
        Averaged image (H, W).
    """
    return jnp.mean(images, axis=0)


def compute_detector_coordinates(
    detector_size: tuple[int, int],
    pixel_size: float,
    detector_distance: float,
    wavelength: float,
) -> tuple[Array, Array]:
    """Compute q-coordinates for detector pixels.

    Args:
        detector_size: (height, width) in pixels.
        pixel_size: Pixel size in mm.
        detector_distance: Sample-to-detector distance in mm.
        wavelength: X-ray wavelength in nm.

    Returns:
        Tuple of (qx, qy) 2D arrays.
    """
    ny, nx = detector_size
    k = 2 * jnp.pi / wavelength

    # Pixel positions relative to beam center
    x = (jnp.arange(nx) - nx // 2) * pixel_size
    y = (jnp.arange(ny) - ny // 2) * pixel_size

    # Convert to angles and then to q
    theta_x = jnp.arctan2(x, detector_distance)
    theta_y = jnp.arctan2(y, detector_distance)

    qx = k * jnp.sin(theta_x)
    qy = k * jnp.sin(theta_y)

    return jnp.meshgrid(qx, qy)


def apply_mask(image: Array, mask: Array) -> Array:
    """Apply a mask to a detector image.

    Args:
        image: 2D detector image.
        mask: Boolean mask (True = valid pixel).

    Returns:
        Masked image with invalid pixels set to NaN.
    """
    return jnp.where(mask, image, jnp.nan)
