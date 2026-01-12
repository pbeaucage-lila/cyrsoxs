"""Rotation matrices and Euler angle handling for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array


def rotation_matrix_z(angle: float) -> Array:
    """Rotation matrix about Z axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def rotation_matrix_y(angle: float) -> Array:
    """Rotation matrix about Y axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rotation_matrix_x(angle: float) -> Array:
    """Rotation matrix about X axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def euler_to_rotation_matrix(
    alpha: float,
    beta: float,
    gamma: float,
    convention: str = "ZYZ",
) -> Array:
    """Convert Euler angles to rotation matrix.

    Args:
        alpha: First Euler angle in radians.
        beta: Second Euler angle in radians.
        gamma: Third Euler angle in radians.
        convention: Euler angle convention (default "ZYZ").

    Returns:
        3x3 rotation matrix.
    """
    if convention == "ZYZ":
        return rotation_matrix_z(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_z(gamma)
    elif convention == "ZXZ":
        return rotation_matrix_z(alpha) @ rotation_matrix_x(beta) @ rotation_matrix_z(gamma)
    else:
        raise ValueError(f"Unsupported Euler convention: {convention}")


def rotate_tensor(tensor: Array, rotation: Array) -> Array:
    """Rotate a 3x3 tensor by a rotation matrix.

    For a tensor T, returns R @ T @ R.T

    Args:
        tensor: 3x3 tensor to rotate.
        rotation: 3x3 rotation matrix.

    Returns:
        Rotated 3x3 tensor.
    """
    return rotation @ tensor @ rotation.T


def batch_euler_to_rotation(
    euler_angles: Array,
    convention: str = "ZYZ",
) -> Array:
    """Convert batch of Euler angles to rotation matrices.

    Args:
        euler_angles: Array of shape (..., 3) containing Euler angles.
        convention: Euler angle convention.

    Returns:
        Array of shape (..., 3, 3) containing rotation matrices.
    """
    from jax import vmap

    def single_euler_to_rotation(angles):
        return euler_to_rotation_matrix(angles[0], angles[1], angles[2], convention)

    # Flatten, apply, reshape
    original_shape = euler_angles.shape[:-1]
    flat_angles = euler_angles.reshape(-1, 3)
    flat_rotations = vmap(single_euler_to_rotation)(flat_angles)
    return flat_rotations.reshape(*original_shape, 3, 3)
