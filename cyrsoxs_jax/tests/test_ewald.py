"""Tests for Ewald sphere projection."""

import jax.numpy as jnp
import pytest
from cyrsoxs_jax.ewald import (
    ewald_projection,
    compute_detector_q_grid,
    EwaldInterpolation,
)


class TestDetectorQGrid:
    """Tests for compute_detector_q_grid."""

    def test_grid_shape(self):
        """Grid should have correct 2D shape."""
        voxel = (32, 32, 32)
        phys_size = 100.0

        qx, qy, (dx, dy, dz), start = compute_detector_q_grid(voxel, phys_size)

        assert qx.shape == (32, 32)
        assert qy.shape == (32, 32)

    def test_grid_range(self):
        """Grid should span from -pi/L to +pi/L."""
        voxel = (64, 64, 64)
        phys_size = 200.0

        qx, qy, (dx, dy, dz), start = compute_detector_q_grid(voxel, phys_size)

        expected_start = -jnp.pi / phys_size
        assert jnp.isclose(start, expected_start)
        assert jnp.isclose(qx[0, 0], expected_start)
        assert jnp.isclose(qy[0, 0], expected_start)

    def test_grid_spacing(self):
        """Grid spacing should be 2*pi/L/(N-1)."""
        voxel = (32, 32, 32)
        phys_size = 100.0

        qx, qy, (dx, dy, dz), start = compute_detector_q_grid(voxel, phys_size)

        expected_dx = (2 * jnp.pi / phys_size) / 31
        assert jnp.isclose(dx, expected_dx)
        assert jnp.isclose(dy, expected_dx)


class TestEwaldProjection:
    """Tests for ewald_projection."""

    def test_output_shape(self):
        """Projection should be 2D with shape (Nx, Ny)."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d, k_magnitude, voxel, phys_size, k_vector
        )

        assert projection.shape == (16, 16)

    def test_invalid_outside_ewald_sphere(self):
        """Points outside Ewald sphere should be NaN."""
        voxel = (32, 32, 32)
        phys_size = 100.0
        k_magnitude = 0.01  # Small k means small Ewald sphere
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d, k_magnitude, voxel, phys_size, k_vector
        )

        # With small k, most of the detector should be outside Ewald sphere
        nan_count = jnp.sum(jnp.isnan(projection))
        assert nan_count > 0

    def test_nearest_neighbor_interpolation(self):
        """Test nearest neighbor interpolation mode."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d,
            k_magnitude,
            voxel,
            phys_size,
            k_vector,
            interpolation=EwaldInterpolation.NEAREST_NEIGHBOR,
        )

        # Valid regions should have value 1.0
        valid_mask = ~jnp.isnan(projection)
        assert jnp.allclose(projection[valid_mask], 1.0)

    def test_trilinear_interpolation(self):
        """Test trilinear interpolation mode."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d,
            k_magnitude,
            voxel,
            phys_size,
            k_vector,
            interpolation=EwaldInterpolation.TRILINEAR,
        )

        # Valid regions should have value 1.0
        valid_mask = ~jnp.isnan(projection)
        assert jnp.allclose(projection[valid_mask], 1.0)

    def test_boundary_pixels_nan(self):
        """Boundary pixels (last row/column) should be NaN."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d, k_magnitude, voxel, phys_size, k_vector
        )

        # Last row and column should be NaN
        assert jnp.all(jnp.isnan(projection[-1, :]))
        assert jnp.all(jnp.isnan(projection[:, -1]))

    def test_2d_mode(self):
        """Test 2D morphology mode."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5
        k_vector = (0.0, 0.0, 1.0)

        scatter_3d = jnp.ones(voxel)
        projection = ewald_projection(
            scatter_3d,
            k_magnitude,
            voxel,
            phys_size,
            k_vector,
            enable_2d=True,
        )

        assert projection.shape == (16, 16)

    def test_k_vector_direction(self):
        """Test with different k-vector directions."""
        voxel = (16, 16, 16)
        phys_size = 100.0
        k_magnitude = 0.5

        scatter_3d = jnp.ones(voxel)

        # Test with k along z
        proj_z = ewald_projection(
            scatter_3d, k_magnitude, voxel, phys_size, (0.0, 0.0, 1.0)
        )

        # Test with tilted k
        proj_tilted = ewald_projection(
            scatter_3d, k_magnitude, voxel, phys_size, (0.1, 0.0, 0.99)
        )

        # Both should produce valid projections
        assert proj_z.shape == proj_tilted.shape
        assert jnp.any(~jnp.isnan(proj_z))
        assert jnp.any(~jnp.isnan(proj_tilted))
