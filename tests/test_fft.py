"""Tests for FFT module."""

import jax.numpy as jnp
import pytest

from cyrsoxs_jax.fft import forward_fft_3d, inverse_fft_3d, fftshift_3d, compute_q_grid


class TestFFT:
    """Tests for FFT functions."""

    def test_forward_inverse_roundtrip_3d(self):
        """Test that forward and inverse FFT are inverses."""
        field = jnp.ones((8, 8, 8))
        result = inverse_fft_3d(forward_fft_3d(field))
        assert jnp.allclose(result.real, field, atol=1e-10)

    def test_forward_inverse_roundtrip_4d(self):
        """Test roundtrip for 4D array (vector field)."""
        field = jnp.ones((8, 8, 8, 3))
        result = inverse_fft_3d(forward_fft_3d(field))
        assert jnp.allclose(result.real, field, atol=1e-10)

    def test_fftshift_3d(self):
        """Test FFT shift."""
        field = jnp.zeros((4, 4, 4))
        field = field.at[0, 0, 0].set(1.0)
        shifted = fftshift_3d(field)
        assert shifted[2, 2, 2] == 1.0

    def test_compute_q_grid(self):
        """Test q-grid computation."""
        n = 16
        physical_size = 1.0  # nm
        wavelength = 1.0  # nm
        qx, qy, qz = compute_q_grid(n, physical_size, wavelength)
        assert qx.shape == (n,)
        assert qy.shape == (n,)
        assert qz.shape == (n,)


class TestRotation:
    """Tests for rotation matrices."""

    def test_rotation_matrix_z_identity(self):
        """Test Z rotation by 0 gives identity."""
        from cyrsoxs_jax.rotation import rotation_matrix_z
        R = rotation_matrix_z(0.0)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        """Test rotation matrices are orthogonal."""
        from cyrsoxs_jax.rotation import rotation_matrix_z, rotation_matrix_y, rotation_matrix_x
        for angle in [0.1, 0.5, 1.0, jnp.pi / 4]:
            for rot_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = rot_fn(angle)
                assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-5)
                assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-5)

    def test_euler_to_rotation_zyz(self):
        """Test Euler to rotation matrix conversion."""
        from cyrsoxs_jax.rotation import euler_to_rotation_matrix
        # Identity rotation
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0, "ZYZ")
        assert jnp.allclose(R, jnp.eye(3), atol=1e-10)


class TestTypes:
    """Tests for data types."""

    def test_simulation_config_defaults(self):
        """Test SimulationConfig default values."""
        from cyrsoxs_jax.types import SimulationConfig
        config = SimulationConfig(energies=jnp.array([280.0, 285.0]))
        assert config.num_threads == 1
        assert config.physical_size == 1.0
        assert config.euler_convention == "ZYZ"
