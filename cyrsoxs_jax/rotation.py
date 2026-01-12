"""Rotation matrix utilities for CyRSoXS JAX.

Provides rotation matrices for:
- Rotation from Euler angles
- Rotation aligning one vector to another
- Rodrigues rotation about arbitrary axis
- K-vector orientation rotations
- Detector rotation
- Matrix composition for combined transformations
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple, NamedTuple


class BaseConfiguration(NamedTuple):
    """Configuration for a single k-vector orientation.

    Matches CUDA BaseConfiguration struct.

    Attributes:
        matrix: 3x3 rotation matrix for this k-vector
        base_rot_angle: Rotation angle so that E.x ~ 1
        base_x: Transformed X axis
        base_y: Transformed Y axis
        base_z: Transformed Z axis (equals k)
    """
    matrix: Array
    base_rot_angle: Array
    base_x: Array
    base_y: Array
    base_z: Array


# Vector operations

def dot(a: Array, b: Array) -> Array:
    """Compute dot product of two 3D vectors.

    Args:
        a: Vector of shape (3,) or batch (N, 3)
        b: Vector of shape (3,) or batch (N, 3)

    Returns:
        Scalar or array of shape (N,)
    """
    return jnp.sum(a * b, axis=-1)


def cross(a: Array, b: Array) -> Array:
    """Compute cross product of two 3D vectors.

    Args:
        a: Vector of shape (3,) or batch (N, 3)
        b: Vector of shape (3,) or batch (N, 3)

    Returns:
        Vector of shape (3,) or batch (N, 3)
    """
    return jnp.cross(a, b)


def norm(v: Array) -> Array:
    """Compute L2 norm of a 3D vector.

    Args:
        v: Vector of shape (3,) or batch (N, 3)

    Returns:
        Scalar or array of shape (N,)
    """
    return jnp.sqrt(dot(v, v))


def normalize(v: Array) -> Array:
    """Normalize a 3D vector to unit length.

    Args:
        v: Vector of shape (3,) or batch (N, 3)

    Returns:
        Unit vector of same shape
    """
    n = norm(v)
    # Handle batched case
    if v.ndim > 1:
        return v / n[..., None]
    return v / n


# Matrix operations

def matmul(A: Array, B: Array, transpose_a: bool = False, transpose_b: bool = False) -> Array:
    """Multiply two 3x3 matrices with optional transposes.

    Args:
        A: Matrix of shape (3, 3)
        B: Matrix of shape (3, 3)
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B

    Returns:
        Matrix product of shape (3, 3)
    """
    if transpose_a:
        A = A.T
    if transpose_b:
        B = B.T
    return A @ B


def matvec(matrix: Array, vec: Array, transpose: bool = False) -> Array:
    """Multiply a 3x3 matrix by a 3D vector.

    Args:
        matrix: Matrix of shape (3, 3)
        vec: Vector of shape (3,)
        transpose: Whether to use transpose of matrix

    Returns:
        Vector of shape (3,)
    """
    if transpose:
        return matrix.T @ vec
    return matrix @ vec


def inverse_3x3(A: Array) -> Array:
    """Compute inverse of a 3x3 matrix.

    Args:
        A: Matrix of shape (3, 3)

    Returns:
        Inverse matrix of shape (3, 3)
    """
    return jnp.linalg.inv(A)


# Rotation matrices

def rotation_matrix_from_vectors(original: Array, target: Array) -> Array:
    """Compute rotation matrix that transforms original vector into target vector.

    Uses the method from:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Args:
        original: Source unit vector of shape (3,)
        target: Target unit vector of shape (3,)

    Returns:
        3x3 rotation matrix R such that R @ original = target
    """
    # Check if vectors are already aligned
    diff = jnp.abs(original - target)
    is_same = jnp.all(diff < 1e-10)

    def identity_case(_):
        return jnp.eye(3)

    def compute_rotation(_):
        dot_prod = dot(original, target)
        cross_vec = cross(original, target)
        norm_cross = norm(cross_vec)

        # Build G matrix
        G = jnp.array([
            [dot_prod, -norm_cross, 0.0],
            [norm_cross, dot_prod, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Compute (target - dot*original) / ||target - dot*original||
        scaled = dot_prod * original
        sub_vec = target - scaled
        norm_sub = norm(sub_vec)
        sub_normalized = sub_vec / norm_sub

        # Build F matrix: columns are original, sub_normalized, -cross_vec
        F = jnp.column_stack([original, sub_normalized, -cross_vec])

        # R = F @ G @ inv(F)
        inv_F = inverse_3x3(F)
        return F @ G @ inv_F

    return jax.lax.cond(is_same, identity_case, compute_rotation, None)


def rodrigues_rotation(v: Array, axis: Array, angle: Array) -> Array:
    """Rotate vector v about axis by angle using Rodrigues' formula.

    v_rot = v*cos(angle) + (axis x v)*sin(angle) + axis*(axis . v)*(1 - cos(angle))

    Args:
        v: Vector to rotate, shape (3,)
        axis: Unit rotation axis, shape (3,)
        angle: Rotation angle in radians

    Returns:
        Rotated vector of shape (3,)
    """
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    cross_prod = cross(axis, v)
    dot_prod = dot(axis, v)

    term1 = cos_a * v
    term2 = sin_a * cross_prod
    term3 = (1 - cos_a) * dot_prod * axis

    return term1 + term2 + term3


def rodrigues_rotation_matrix(axis: Array, angle: Array) -> Array:
    """Build rotation matrix from axis-angle representation using Rodrigues' formula.

    Args:
        axis: Unit rotation axis, shape (3,)
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    # Skew-symmetric cross-product matrix K
    K = jnp.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # R = I + sin(a)*K + (1-cos(a))*K^2
    I = jnp.eye(3)
    return I + sin_a * K + (1 - cos_a) * (K @ K)


def rotation_matrix_k(k: Array) -> Array:
    """Compute rotation matrix that maps (0,0,1) to given k-vector.

    Args:
        k: Target k-vector direction, shape (3,)

    Returns:
        3x3 rotation matrix
    """
    orig_k = jnp.array([0.0, 0.0, 1.0])
    return rotation_matrix_from_vectors(orig_k, k)


def rotation_matrix_euler(phi: Array, theta: Array, psi: Array) -> Array:
    """Compute rotation matrix from Euler angles (ZYZ convention).

    Args:
        phi: First rotation about Z axis
        theta: Second rotation about Y axis
        psi: Third rotation about Z axis

    Returns:
        3x3 rotation matrix
    """
    c1, s1 = jnp.cos(phi), jnp.sin(phi)
    c2, s2 = jnp.cos(theta), jnp.sin(theta)
    c3, s3 = jnp.cos(psi), jnp.sin(psi)

    return jnp.array([
        [c1*c2*c3 - s1*s3, -c1*c2*s3 - s1*c3, c1*s2],
        [s1*c2*c3 + c1*s3, -s1*c2*s3 + c1*c3, s1*s2],
        [-s2*c3, s2*s3, c2]
    ])


def rotation_matrix_euler_xyz(alpha: Array, beta: Array, gamma: Array) -> Array:
    """Compute rotation matrix from Euler angles (XYZ convention).

    Args:
        alpha: Rotation about X axis
        beta: Rotation about Y axis
        gamma: Rotation about Z axis

    Returns:
        3x3 rotation matrix
    """
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    cg, sg = jnp.cos(gamma), jnp.sin(gamma)

    # Rx @ Ry @ Rz
    return jnp.array([
        [cb*cg, -cb*sg, sb],
        [sa*sb*cg + ca*sg, -sa*sb*sg + ca*cg, -sa*cb],
        [-ca*sb*cg + sa*sg, ca*sb*sg + sa*cg, ca*cb]
    ])


def compute_base_rotation_angle(k: Array, rotation_matrix_k: Array) -> Tuple[Array, Array]:
    """Find base rotation angle such that E.x is approximately 1.

    This finds the rotation about k that aligns the transformed X axis
    to have minimal y-component (E field aligned with lab X).

    Args:
        k: K-vector direction, shape (3,)
        rotation_matrix_k: Rotation matrix mapping (0,0,1) to k

    Returns:
        Tuple of (rotation_matrix, base_angle)
    """
    X = jnp.array([1.0, 0.0, 0.0])

    # Transform X by the k-rotation
    shifted_x = matvec(rotation_matrix_k, X)

    # Search for angle that minimizes |rotated_x.y|
    # Using a fine grid search similar to CUDA implementation
    num_intervals = 1000
    angles = jnp.linspace(0, jnp.pi, num_intervals)

    def compute_y_component(angle):
        rotated = rodrigues_rotation(shifted_x, k, angle)
        return jnp.abs(rotated[1])

    y_components = jax.vmap(compute_y_component)(angles)
    min_idx = jnp.argmin(y_components)
    base_angle = angles[min_idx]

    # Check sign of x component and add pi if needed
    rotated_x = rodrigues_rotation(shifted_x, k, base_angle)
    base_angle = jnp.where(rotated_x[0] > 0, base_angle, base_angle + jnp.pi)

    # Compute the combined rotation matrix
    rotated_x_final = rodrigues_rotation(shifted_x, k, base_angle)
    rotation_matrix_x = rotation_matrix_from_vectors(shifted_x, rotated_x_final)
    combined = rotation_matrix_x @ rotation_matrix_k

    return combined, base_angle


# Complex vector rotation (for polarization)

def rotate_complex_vector(rotation_matrix: Array,
                          vec_x: Array, vec_y: Array, vec_z: Array,
                          transpose: bool = False) -> Tuple[Array, Array, Array]:
    """Rotate a complex 3D vector (polarization).

    Args:
        rotation_matrix: 3x3 rotation matrix
        vec_x, vec_y, vec_z: Complex components of the vector
        transpose: Whether to use transpose of rotation matrix

    Returns:
        Tuple of rotated (vec_x, vec_y, vec_z)
    """
    R = rotation_matrix.T if transpose else rotation_matrix

    new_x = R[0, 0] * vec_x + R[0, 1] * vec_y + R[0, 2] * vec_z
    new_y = R[1, 0] * vec_x + R[1, 1] * vec_y + R[1, 2] * vec_z
    new_z = R[2, 0] * vec_x + R[2, 1] * vec_y + R[2, 2] * vec_z

    return new_x, new_y, new_z


# Batched operations for voxel arrays

def rotate_vectors_batch(rotation_matrix: Array, vectors: Array) -> Array:
    """Apply rotation matrix to batch of vectors.

    Args:
        rotation_matrix: 3x3 rotation matrix
        vectors: Array of shape (..., 3)

    Returns:
        Rotated vectors of same shape
    """
    return vectors @ rotation_matrix.T


def compute_rotation_matrices_for_k_vectors(k_vectors: Array) -> Array:
    """Compute rotation matrices for a batch of k-vectors.

    Args:
        k_vectors: Array of shape (N, 3) k-vector directions

    Returns:
        Array of shape (N, 3, 3) rotation matrices
    """
    return jax.vmap(rotation_matrix_k)(k_vectors)


# Rotation composition

def compose_rotations(*matrices: Array) -> Array:
    """Compose multiple rotation matrices.

    Applies rotations right-to-left: compose(A, B, C) = A @ B @ C
    So vector v transformed is: A @ B @ C @ v

    Args:
        *matrices: Variable number of 3x3 rotation matrices

    Returns:
        Combined 3x3 rotation matrix
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = result @ m
    return result


def compute_detector_rotation_matrix(detector_coords: Array) -> Array:
    """Compute rotation matrix for detector coordinate system.

    Transforms from lab frame to detector frame based on detector normal.

    Args:
        detector_coords: Detector normal direction, shape (3,)

    Returns:
        3x3 rotation matrix
    """
    return rotation_matrix_k(detector_coords)


def compute_e_rotation_matrix(k: Array, e_rot_angle: Array) -> Array:
    """Compute rotation matrix for E-field polarization rotation.

    Rotates the E-field about the k-vector by specified angle.

    Args:
        k: K-vector direction, shape (3,)
        e_rot_angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix for E-field rotation
    """
    return rodrigues_rotation_matrix(k, e_rot_angle)


def compute_full_rotation(k: Array, base_config: BaseConfiguration,
                          e_rot_angle: Array) -> Array:
    """Compute full rotation matrix for a given E-field rotation.

    Combines base k-vector rotation with additional E-field rotation.

    Args:
        k: K-vector direction, shape (3,)
        base_config: Base configuration for this k-vector
        e_rot_angle: Additional E-field rotation angle

    Returns:
        Combined 3x3 rotation matrix
    """
    # Total angle is base angle plus additional rotation
    total_angle = base_config.base_rot_angle + e_rot_angle

    # Get the k-rotation matrix
    R_k = rotation_matrix_k(k)

    # Apply the E-rotation about k
    X = jnp.array([1.0, 0.0, 0.0])
    shifted_x = matvec(R_k, X)
    rotated_x = rodrigues_rotation(shifted_x, k, total_angle)

    # Build combined rotation
    R_x = rotation_matrix_from_vectors(shifted_x, rotated_x)
    return R_x @ R_k


# High-level configuration builders

def compute_base_configuration(k: Array) -> BaseConfiguration:
    """Compute base configuration for a k-vector.

    This determines the rotation matrix and base angle that aligns
    the transformed X axis with the lab X direction (E.x ~ 1).

    Args:
        k: K-vector direction (unit vector), shape (3,)

    Returns:
        BaseConfiguration with matrix, angle, and transformed axes
    """
    R_k = rotation_matrix_k(k)
    R_combined, base_angle = compute_base_rotation_angle(k, R_k)

    # Compute transformed axes
    X = jnp.array([1.0, 0.0, 0.0])
    Y = jnp.array([0.0, 1.0, 0.0])
    Z = jnp.array([0.0, 0.0, 1.0])

    base_x = matvec(R_combined, X)
    base_y = matvec(R_combined, Y)
    base_z = matvec(R_combined, Z)

    return BaseConfiguration(
        matrix=R_k,
        base_rot_angle=base_angle,
        base_x=base_x,
        base_y=base_y,
        base_z=base_z,
    )


def compute_all_base_configurations(k_vectors: Array) -> Tuple[Array, Array, Array, Array, Array]:
    """Compute base configurations for all k-vectors.

    Args:
        k_vectors: Array of shape (N, 3) k-vector directions

    Returns:
        Tuple of (matrices, base_angles, base_x, base_y, base_z)
        Each with leading dimension N
    """
    configs = jax.vmap(compute_base_configuration)(k_vectors)
    return (
        configs.matrix,
        configs.base_rot_angle,
        configs.base_x,
        configs.base_y,
        configs.base_z,
    )
