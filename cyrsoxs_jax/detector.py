"""Detector image rotation and angle averaging for CyRSoXS-JAX.

Provides functions for:
- Rotating 2D detector images using affine transforms
- Averaging over multiple rotation angles (E-field polarization averaging)
- Applying detector geometry rotations for k-vector orientation
- Optional rotation masking for handling boundary pixels
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates
from typing import Tuple, Optional


def _compute_rotation_coords(
    shape: Tuple[int, int],
    angle: Array,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[Array, Array]:
    """Compute coordinate arrays for rotation about center.

    Args:
        shape: (height, width) of image.
        angle: Rotation angle in radians.
        center: Center of rotation (cy, cx). Defaults to image center.

    Returns:
        Tuple of (y_coords, x_coords) for map_coordinates.
    """
    h, w = shape
    if center is None:
        cy, cx = h / 2.0, w / 2.0
    else:
        cy, cx = center

    # Create output coordinate grid
    y_out = jnp.arange(h)
    x_out = jnp.arange(w)
    yy, xx = jnp.meshgrid(y_out, x_out, indexing='ij')

    # Shift to center
    yy_c = yy - cy
    xx_c = xx - cx

    # Inverse rotation (to find source coordinates)
    cos_a = jnp.cos(-angle)
    sin_a = jnp.sin(-angle)

    # Apply inverse rotation
    y_src = cos_a * yy_c - sin_a * xx_c + cy
    x_src = sin_a * yy_c + cos_a * xx_c + cx

    return y_src, x_src


def rotate_image(
    image: Array,
    angle: float,
    fill_value: float = jnp.nan,
) -> Array:
    """Rotate a 2D image by a given angle about its center.

    Uses bilinear interpolation matching CUDA NPP warpAffine behavior.

    Args:
        image: 2D detector image of shape (H, W).
        angle: Rotation angle in radians.
        fill_value: Value for pixels outside the image boundary.

    Returns:
        Rotated image of same shape.
    """
    h, w = image.shape

    # Compute source coordinates for each output pixel
    y_coords, x_coords = _compute_rotation_coords((h, w), angle)

    # Stack coordinates for map_coordinates
    coords = jnp.stack([y_coords, x_coords], axis=0)

    # Apply bilinear interpolation
    # mode='constant' with cval for out-of-bounds handling
    rotated = map_coordinates(
        image,
        coords,
        order=1,  # bilinear interpolation
        mode='constant',
        cval=fill_value,
    )

    return rotated


def rotate_image_with_mask(
    image: Array,
    angle: float,
) -> Tuple[Array, Array]:
    """Rotate image and return mask of valid (non-boundary) pixels.

    Args:
        image: 2D detector image of shape (H, W).
        angle: Rotation angle in radians.

    Returns:
        Tuple of (rotated_image, valid_mask).
        rotated_image has 0 for invalid pixels.
        valid_mask is True where pixels are valid.
    """
    # Rotate with NaN fill to identify boundary pixels
    rotated = rotate_image(image, angle, fill_value=jnp.nan)

    # Create mask: valid where not NaN
    valid_mask = ~jnp.isnan(rotated)

    # Replace NaN with 0 for accumulation
    rotated_clean = jnp.where(valid_mask, rotated, 0.0)

    return rotated_clean, valid_mask


def average_rotations(
    images: Array,
    use_mask: bool = False,
) -> Array:
    """Average multiple rotated detector images.

    Args:
        images: Stack of 2D images (N, H, W).
        use_mask: If True, track valid pixels per-location for proper averaging.

    Returns:
        Averaged image (H, W).
    """
    if use_mask:
        # Use NaN-aware mean - pixels with all NaN become 0
        valid = ~jnp.isnan(images)
        images_clean = jnp.where(valid, images, 0.0)
        count = jnp.sum(valid.astype(jnp.float32), axis=0)
        total = jnp.sum(images_clean, axis=0)
        # Avoid division by zero
        result = jnp.where(count > 0, total / count, 0.0)
        return result
    else:
        return jnp.mean(images, axis=0)


def rotate_and_average(
    projection: Array,
    angles: Array,
    use_mask: bool = False,
) -> Array:
    """Rotate a projection by multiple angles and average.

    This is the main function for E-field polarization averaging.
    For each angle, the projection is rotated and results are averaged.

    Args:
        projection: 2D Ewald sphere projection (H, W).
        angles: Array of rotation angles in radians (N,).
        use_mask: If True, track valid pixels for proper boundary handling.

    Returns:
        Averaged projection (H, W).
    """
    if use_mask:
        # Rotate each and track validity
        def rotate_single(angle):
            return rotate_image(projection, angle, fill_value=jnp.nan)

        rotated_stack = jax.vmap(rotate_single)(angles)
        return average_rotations(rotated_stack, use_mask=True)
    else:
        # Simple case: no masking
        def rotate_single(angle):
            return rotate_image(projection, angle, fill_value=0.0)

        rotated_stack = jax.vmap(rotate_single)(angles)
        return jnp.mean(rotated_stack, axis=0)


def apply_detector_rotation(
    image: Array,
    rotation_matrix: Array,
    k_rotation_matrix: Array,
) -> Array:
    """Apply final detector rotation for k-vector geometry.

    Transforms the averaged projection to detector coordinates
    based on the detector and k-vector rotation matrices.

    Args:
        image: 2D projection image (H, W).
        rotation_matrix: 3x3 detector rotation matrix.
        k_rotation_matrix: 3x3 k-vector rotation matrix.

    Returns:
        Rotated image in detector coordinates.
    """
    h, w = image.shape
    center = (h / 2.0, w / 2.0)

    # Combined rotation matrix
    combined = rotation_matrix @ k_rotation_matrix

    # Define 3 source points (center, bottom-center, right-center)
    src_pts = jnp.array([
        [w / 2.0, h / 2.0, 0.0],
        [w / 2.0, h * 1.0, 0.0],
        [w * 1.0, h / 2.0, 0.0],
    ])

    # Transform points relative to center
    src_centered = src_pts - jnp.array([center[1], center[0], 0.0])
    dst_centered = src_centered @ combined.T
    dst_pts = dst_centered + jnp.array([center[1], center[0], 0.0])

    # Compute affine coefficients from 3-point correspondence
    affine = _compute_affine_from_points(
        src_pts[:, :2],
        dst_pts[:, :2],
    )

    # Apply affine transform
    return _apply_affine_transform(image, affine)


def _compute_affine_from_points(
    src: Array,
    dst: Array,
) -> Array:
    """Compute 2x3 affine matrix from 3-point correspondence.

    Solves: dst = src @ A.T + b, returning [A | b].

    Args:
        src: Source points (3, 2).
        dst: Destination points (3, 2).

    Returns:
        Affine matrix (2, 3).
    """
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]

    u0, v0 = dst[0]
    u1, v1 = dst[1]
    u2, v2 = dst[2]

    denom = x0 * y1 - x1 * y0 - x0 * y2 + x2 * y0 + x1 * y2 - x2 * y1

    a00 = (u0 * y1 - u1 * y0 - u0 * y2 + u2 * y0 + u1 * y2 - u2 * y1) / denom
    a01 = -(u0 * x1 - u1 * x0 - u0 * x2 + u2 * x0 + u1 * x2 - u2 * x1) / denom
    a02 = (u0 * x1 * y2 - u0 * x2 * y1 - u1 * x0 * y2 + u1 * x2 * y0 + u2 * x0 * y1 - u2 * x1 * y0) / denom

    a10 = (v0 * y1 - v1 * y0 - v0 * y2 + v2 * y0 + v1 * y2 - v2 * y1) / denom
    a11 = -(v0 * x1 - v1 * x0 - v0 * x2 + v2 * x0 + v1 * x2 - v2 * x1) / denom
    a12 = (v0 * x1 * y2 - v0 * x2 * y1 - v1 * x0 * y2 + v1 * x2 * y0 + v2 * x0 * y1 - v2 * x1 * y0) / denom

    return jnp.array([[a00, a01, a02], [a10, a11, a12]])


def _apply_affine_transform(
    image: Array,
    affine: Array,
    fill_value: float = jnp.nan,
) -> Array:
    """Apply 2x3 affine transform to image.

    Args:
        image: 2D image (H, W).
        affine: Affine matrix (2, 3).
        fill_value: Value for out-of-bounds pixels.

    Returns:
        Transformed image.
    """
    h, w = image.shape

    # Create output grid
    y_out = jnp.arange(h)
    x_out = jnp.arange(w)
    yy, xx = jnp.meshgrid(y_out, x_out, indexing='ij')

    # Inverse affine to find source coordinates
    # For forward: dst = A @ src + b
    # Inverse: src = A^-1 @ (dst - b)
    A = affine[:, :2]
    b = affine[:, 2]

    A_inv = jnp.linalg.inv(A)

    # Apply inverse transform
    dst_coords = jnp.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)
    src_coords = (dst_coords - b) @ A_inv.T  # (N, 2)

    x_src = src_coords[:, 0].reshape(h, w)
    y_src = src_coords[:, 1].reshape(h, w)

    # Note: map_coordinates expects (row, col) = (y, x)
    coords = jnp.stack([y_src, x_src], axis=0)

    return map_coordinates(
        image,
        coords,
        order=1,
        mode='constant',
        cval=fill_value,
    )
