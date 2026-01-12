"""Tests for scatter3D computation."""

import jax.numpy as jnp
import pytest
from cyrsoxs_jax.scatter import (
    scatter_3d,
    compute_q_grid_3d,
    project_polarization_magnitude_squared,
)


class TestComputeQGrid3D:
    """Tests for compute_q_grid_3d function."""

    def test_grid_shape(self):
        """Test that q-grid has correct shape."""
        voxel = (16, 16, 16)
        qx, qy, qz = compute_q_grid_3d(voxel, phys_size=100.0)

        assert qx.shape == (16, 16, 16)
        assert qy.shape == (16, 16, 16)
        assert qz.shape == (16, 16, 16)

    def test_grid_range(self):
        """Test that q-grid starts at -pi/phys_size."""
        voxel = (8, 8, 8)
        phys_size = 100.0
        qx, qy, qz = compute_q_grid_3d(voxel, phys_size)

        # First element should be -pi/phys_size
        expected_start = -jnp.pi / phys_size
        assert jnp.isclose(qx[0, 0, 0], expected_start)
        assert jnp.isclose(qy[0, 0, 0], expected_start)
        assert jnp.isclose(qz[0, 0, 0], expected_start)

    def test_2d_mode_zero_qz(self):
        """Test that 2D mode sets qz to zero."""
        voxel = (8, 8, 8)
        qx, qy, qz = compute_q_grid_3d(voxel, phys_size=100.0, enable_2d=True)

        assert jnp.allclose(qz, 0.0)
        # qx and qy should still vary
        assert not jnp.allclose(qx, qx[0, 0, 0])

    def test_non_cubic_grid(self):
        """Test non-cubic grid dimensions."""
        voxel = (8, 16, 32)
        qx, qy, qz = compute_q_grid_3d(voxel, phys_size=100.0)

        assert qx.shape == (8, 16, 32)


class TestProjectPolarizationMagnitudeSquared:
    """Tests for project_polarization_magnitude_squared function."""

    def test_zero_polarization(self):
        """Test that zero polarization gives zero scatter."""
        polarization = jnp.zeros((4, 4, 4, 3), dtype=jnp.complex64)
        qx = jnp.ones((4, 4, 4))
        qy = jnp.zeros((4, 4, 4))
        qz = jnp.zeros((4, 4, 4))

        result = project_polarization_magnitude_squared(
            polarization, (qx, qy, qz), k_magnitude=1.0
        )

        assert jnp.allclose(result, 0.0)

    def test_polarization_parallel_to_q(self):
        """Test that polarization parallel to q gives zero scatter."""
        # When P is parallel to q, the projection onto perp plane is zero
        # P = (1, 0, 0), q = (1, 0, 0) should give zero
        polarization = jnp.zeros((4, 4, 4, 3), dtype=jnp.complex64)
        polarization = polarization.at[..., 0].set(1.0 + 0j)  # P = (1, 0, 0)

        k_mag = 1.0
        # q_total = k*k_vec + q_local. For q parallel to P:
        qx = jnp.ones((4, 4, 4)) * k_mag  # q points in x direction
        qy = jnp.zeros((4, 4, 4))
        qz = jnp.zeros((4, 4, 4))

        result = project_polarization_magnitude_squared(
            polarization, (qx, qy, qz), k_magnitude=k_mag
        )

        # Result should be zero (P perpendicular projection is zero)
        assert jnp.allclose(result, 0.0)

    def test_polarization_perpendicular_to_q(self):
        """Test that polarization perpendicular to q preserves magnitude."""
        # When P is perpendicular to q, projection preserves P
        # P = (0, 1, 0), q = (k, 0, 0)
        polarization = jnp.zeros((4, 4, 4, 3), dtype=jnp.complex64)
        polarization = polarization.at[..., 1].set(1.0 + 0j)  # P = (0, 1, 0)

        k_mag = 1.0
        qx = jnp.ones((4, 4, 4)) * k_mag
        qy = jnp.zeros((4, 4, 4))
        qz = jnp.zeros((4, 4, 4))

        result = project_polarization_magnitude_squared(
            polarization, (qx, qy, qz), k_magnitude=k_mag
        )

        # Result should be k^4 * |P|^2 = 1.0
        # The transformation is (k^2*I - q*q^T)*P with |q|=k
        # For P perp to q: result is k^2 * P, so |result|^2 = k^4 * |P|^2
        expected = k_mag ** 4 * 1.0
        assert jnp.allclose(result, expected)

    def test_complex_polarization(self):
        """Test with complex polarization values."""
        polarization = jnp.zeros((4, 4, 4, 3), dtype=jnp.complex64)
        # P = (0, 1+1j, 0), perpendicular to q = (k, 0, 0)
        polarization = polarization.at[..., 1].set(1.0 + 1.0j)

        k_mag = 1.0
        qx = jnp.ones((4, 4, 4)) * k_mag
        qy = jnp.zeros((4, 4, 4))
        qz = jnp.zeros((4, 4, 4))

        result = project_polarization_magnitude_squared(
            polarization, (qx, qy, qz), k_magnitude=k_mag
        )

        # |1+1j|^2 = 2, result should be k^4 * 2 = 2
        expected = k_mag ** 4 * 2.0
        assert jnp.allclose(result, expected)


class TestScatter3D:
    """Tests for scatter_3d function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        voxel = (8, 8, 8)
        polarization_fft = jnp.zeros((*voxel, 3), dtype=jnp.complex64)

        result = scatter_3d(
            polarization_fft,
            k_magnitude=1.0,
            voxel=voxel,
            phys_size=100.0,
            k_vector=(0.0, 0.0, 1.0),
        )

        assert result.shape == (8, 8, 8)

    def test_output_non_negative(self):
        """Test that output is non-negative (squared magnitude)."""
        voxel = (8, 8, 8)
        # Random complex polarization
        key = jnp.array([0, 1], dtype=jnp.uint32)
        polarization_fft = jnp.ones((*voxel, 3), dtype=jnp.complex64) * (0.5 + 0.5j)

        result = scatter_3d(
            polarization_fft,
            k_magnitude=1.0,
            voxel=voxel,
            phys_size=100.0,
            k_vector=(0.0, 0.0, 1.0),
        )

        assert jnp.all(result >= 0)

    def test_2d_mode(self):
        """Test 2D mode works correctly."""
        voxel = (8, 8, 4)
        polarization_fft = jnp.ones((*voxel, 3), dtype=jnp.complex64) * (0.1 + 0j)

        result = scatter_3d(
            polarization_fft,
            k_magnitude=1.0,
            voxel=voxel,
            phys_size=100.0,
            k_vector=(0.0, 0.0, 1.0),
            enable_2d=True,
        )

        assert result.shape == (8, 8, 4)
        # In 2D mode, result should be constant along z
        # (since qz=0 and polarization is uniform)
        assert jnp.allclose(result[:, :, 0], result[:, :, 1])

    def test_k_vector_direction_matters(self):
        """Test that different k-vector directions give different results."""
        voxel = (8, 8, 8)
        polarization_fft = jnp.ones((*voxel, 3), dtype=jnp.complex64) * (0.5 + 0.3j)

        result_z = scatter_3d(
            polarization_fft,
            k_magnitude=1.0,
            voxel=voxel,
            phys_size=100.0,
            k_vector=(0.0, 0.0, 1.0),
        )

        result_x = scatter_3d(
            polarization_fft,
            k_magnitude=1.0,
            voxel=voxel,
            phys_size=100.0,
            k_vector=(1.0, 0.0, 0.0),
        )

        # Results should be different (not identical)
        assert not jnp.allclose(result_z, result_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
