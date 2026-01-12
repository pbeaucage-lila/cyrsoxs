"""Tests for rotation matrix utilities."""

import jax
import jax.numpy as jnp
import pytest
from cyrsoxs_jax import rotation


class TestVectorOperations:
    """Tests for basic vector operations."""

    def test_dot_product(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = rotation.dot(a, b)
        expected = 1*4 + 2*5 + 3*6  # 32
        assert jnp.allclose(result, expected)

    def test_cross_product(self):
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])
        result = rotation.cross(a, b)
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_norm(self):
        v = jnp.array([3.0, 4.0, 0.0])
        result = rotation.norm(v)
        assert jnp.allclose(result, 5.0)

    def test_normalize(self):
        v = jnp.array([3.0, 4.0, 0.0])
        result = rotation.normalize(v)
        expected = jnp.array([0.6, 0.8, 0.0])
        assert jnp.allclose(result, expected)


class TestMatrixOperations:
    """Tests for matrix operations."""

    def test_matmul(self):
        A = jnp.eye(3)
        B = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
        result = rotation.matmul(A, B)
        assert jnp.allclose(result, B)

    def test_matvec(self):
        A = jnp.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])  # 90 deg rotation about X
        v = jnp.array([0.0, 1.0, 0.0])
        result = rotation.matvec(A, v)
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_inverse_3x3(self):
        A = jnp.array([[1.0, 2.0, 3.0],
                       [0.0, 1.0, 4.0],
                       [5.0, 6.0, 0.0]])
        A_inv = rotation.inverse_3x3(A)
        result = A @ A_inv
        assert jnp.allclose(result, jnp.eye(3), atol=1e-5)


class TestRotationMatrices:
    """Tests for rotation matrix computation."""

    def test_rotation_matrix_identity_case(self):
        v = jnp.array([0.0, 0.0, 1.0])
        R = rotation.rotation_matrix_from_vectors(v, v)
        assert jnp.allclose(R, jnp.eye(3))

    def test_rotation_matrix_transforms_vector(self):
        original = jnp.array([0.0, 0.0, 1.0])
        target = jnp.array([1.0, 0.0, 0.0])
        R = rotation.rotation_matrix_from_vectors(original, target)
        result = R @ original
        assert jnp.allclose(result, target, atol=1e-6)

    def test_rotation_matrix_orthogonal(self):
        original = jnp.array([0.0, 0.0, 1.0])
        target = rotation.normalize(jnp.array([1.0, 1.0, 1.0]))
        R = rotation.rotation_matrix_from_vectors(original, target)
        # Check R @ R.T = I
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
        # Check det(R) = 1
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


class TestRodriguesRotation:
    """Tests for Rodrigues rotation formula."""

    def test_rodrigues_zero_angle(self):
        v = jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = 0.0
        result = rotation.rodrigues_rotation(v, axis, angle)
        assert jnp.allclose(result, v)

    def test_rodrigues_90_degrees(self):
        v = jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi / 2
        result = rotation.rodrigues_rotation(v, axis, angle)
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_rodrigues_180_degrees(self):
        v = jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi
        result = rotation.rodrigues_rotation(v, axis, angle)
        expected = jnp.array([-1.0, 0.0, 0.0])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_rodrigues_rotation_matrix(self):
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi / 2
        R = rotation.rodrigues_rotation_matrix(axis, angle)
        v = jnp.array([1.0, 0.0, 0.0])
        result = R @ v
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(result, expected, atol=1e-6)


class TestKVectorRotation:
    """Tests for k-vector rotation matrices."""

    def test_rotation_matrix_k_z_axis(self):
        k = jnp.array([0.0, 0.0, 1.0])
        R = rotation.rotation_matrix_k(k)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_rotation_matrix_k_x_axis(self):
        k = jnp.array([1.0, 0.0, 0.0])
        R = rotation.rotation_matrix_k(k)
        orig = jnp.array([0.0, 0.0, 1.0])
        result = R @ orig
        assert jnp.allclose(result, k, atol=1e-6)

    def test_rotation_matrix_k_arbitrary(self):
        k = rotation.normalize(jnp.array([1.0, 1.0, 1.0]))
        R = rotation.rotation_matrix_k(k)
        orig = jnp.array([0.0, 0.0, 1.0])
        result = R @ orig
        assert jnp.allclose(result, k, atol=1e-6)


class TestEulerAngles:
    """Tests for Euler angle rotation matrices."""

    def test_euler_identity(self):
        R = rotation.rotation_matrix_euler(0.0, 0.0, 0.0)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_euler_xyz_identity(self):
        R = rotation.rotation_matrix_euler_xyz(0.0, 0.0, 0.0)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_euler_orthogonal(self):
        R = rotation.rotation_matrix_euler(0.5, 1.0, 0.3)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


class TestComplexVectorRotation:
    """Tests for complex polarization vector rotation."""

    def test_rotate_complex_vector(self):
        R = jnp.eye(3)
        vx = jnp.array(1.0 + 0.5j)
        vy = jnp.array(0.0 + 0.0j)
        vz = jnp.array(0.0 + 0.0j)
        rx, ry, rz = rotation.rotate_complex_vector(R, vx, vy, vz)
        assert jnp.allclose(rx, vx)
        assert jnp.allclose(ry, vy)
        assert jnp.allclose(rz, vz)


class TestBatchedOperations:
    """Tests for batched rotation operations."""

    def test_rotate_vectors_batch(self):
        R = rotation.rodrigues_rotation_matrix(
            jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2
        )
        vectors = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        result = rotation.rotate_vectors_batch(R, vectors)
        expected = jnp.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_compute_rotation_matrices_for_k_vectors(self):
        k_vectors = jnp.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        R_batch = rotation.compute_rotation_matrices_for_k_vectors(k_vectors)
        assert R_batch.shape == (2, 3, 3)
        # First should be identity (k = z)
        assert jnp.allclose(R_batch[0], jnp.eye(3), atol=1e-6)


class TestDifferentiability:
    """Tests for JAX differentiability."""

    def test_rodrigues_gradient(self):
        def rotate_and_sum(angle):
            v = jnp.array([1.0, 0.0, 0.0])
            axis = jnp.array([0.0, 0.0, 1.0])
            rotated = rotation.rodrigues_rotation(v, axis, angle)
            return jnp.sum(rotated)

        grad_fn = jax.grad(rotate_and_sum)
        grad = grad_fn(jnp.pi / 4)
        assert jnp.isfinite(grad)

    def test_euler_gradient(self):
        def euler_sum(angles):
            R = rotation.rotation_matrix_euler(angles[0], angles[1], angles[2])
            return jnp.sum(R)

        grad_fn = jax.grad(euler_sum)
        grad = grad_fn(jnp.array([0.1, 0.2, 0.3]))
        assert jnp.all(jnp.isfinite(grad))


class TestRotationComposition:
    """Tests for rotation composition."""

    def test_compose_identity(self):
        I = jnp.eye(3)
        result = rotation.compose_rotations(I, I, I)
        assert jnp.allclose(result, I)

    def test_compose_two_rotations(self):
        R1 = rotation.rodrigues_rotation_matrix(
            jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2
        )
        R2 = rotation.rodrigues_rotation_matrix(
            jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2
        )
        composed = rotation.compose_rotations(R1, R2)
        # Two 90 degree rotations = 180 degrees
        R_180 = rotation.rodrigues_rotation_matrix(
            jnp.array([0.0, 0.0, 1.0]), jnp.pi
        )
        assert jnp.allclose(composed, R_180, atol=1e-6)

    def test_detector_rotation_matrix(self):
        detector_coords = rotation.normalize(jnp.array([1.0, 0.0, 1.0]))
        R = rotation.compute_detector_rotation_matrix(detector_coords)
        # Check it maps z to detector normal
        z = jnp.array([0.0, 0.0, 1.0])
        result = R @ z
        assert jnp.allclose(result, detector_coords, atol=1e-6)


class TestBaseConfiguration:
    """Tests for base configuration computation."""

    def test_base_configuration_z_axis(self):
        k = jnp.array([0.0, 0.0, 1.0])
        config = rotation.compute_base_configuration(k)
        # For k = z, the base configuration should be near identity
        assert jnp.allclose(config.base_z, k, atol=1e-6)

    def test_base_configuration_arbitrary(self):
        k = rotation.normalize(jnp.array([1.0, 1.0, 1.0]))
        config = rotation.compute_base_configuration(k)
        # base_z should equal k
        assert jnp.allclose(config.base_z, k, atol=1e-5)
        # Axes should be orthogonal
        assert jnp.allclose(rotation.dot(config.base_x, config.base_y), 0, atol=1e-5)
        assert jnp.allclose(rotation.dot(config.base_x, config.base_z), 0, atol=1e-5)
        assert jnp.allclose(rotation.dot(config.base_y, config.base_z), 0, atol=1e-5)

    def test_compute_all_base_configurations(self):
        k_vectors = jnp.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            rotation.normalize(jnp.array([1.0, 1.0, 1.0])),
        ])
        matrices, angles, base_x, base_y, base_z = \
            rotation.compute_all_base_configurations(k_vectors)
        assert matrices.shape == (3, 3, 3)
        assert angles.shape == (3,)
        assert base_x.shape == (3, 3)


class TestERotation:
    """Tests for E-field rotation."""

    def test_e_rotation_matrix(self):
        k = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi / 4
        R = rotation.compute_e_rotation_matrix(k, angle)
        # Should be rotation about z
        expected = rotation.rodrigues_rotation_matrix(k, angle)
        assert jnp.allclose(R, expected, atol=1e-6)
