"""Tests for FFT module."""

import jax.numpy as jnp
import pytest

from cyrsoxs_jax.fft import (
    forward_fft_3d,
    inverse_fft_3d,
    fftshift_3d,
    replace_dc_component,
    fft_pipeline,
    fft_pipeline_vectorized,
    compute_q_grid,
)


class TestFFT:
    """Tests for FFT functions."""

    def test_forward_inverse_roundtrip_3d(self):
        """Test that forward and inverse FFT are inverses."""
        field = jnp.ones((8, 8, 8), dtype=jnp.complex64)
        result = inverse_fft_3d(forward_fft_3d(field))
        assert jnp.allclose(result.real, jnp.ones((8, 8, 8)), atol=1e-5)

    def test_forward_inverse_roundtrip_4d(self):
        """Test roundtrip for 4D array (vector field)."""
        field = jnp.ones((8, 8, 8, 3), dtype=jnp.complex64)
        result = inverse_fft_3d(forward_fft_3d(field))
        assert jnp.allclose(result.real, jnp.ones((8, 8, 8, 3)), atol=1e-5)

    def test_forward_fft_random(self):
        """Test forward FFT on random field."""
        key = jnp.array([0, 1], dtype=jnp.uint32)
        field = jnp.linspace(0, 1, 64).reshape((4, 4, 4))
        fft_result = forward_fft_3d(field)
        assert fft_result.shape == (4, 4, 4)
        assert jnp.iscomplexobj(fft_result)

    def test_fftshift_3d(self):
        """Test FFT shift moves origin to center."""
        field = jnp.zeros((4, 4, 4))
        field = field.at[0, 0, 0].set(1.0)
        shifted = fftshift_3d(field)
        # For even-sized array, origin moves to (N/2, N/2, N/2)
        assert shifted[2, 2, 2] == 1.0
        assert shifted[0, 0, 0] == 0.0

    def test_fftshift_4d(self):
        """Test FFT shift for 4D vector field."""
        field = jnp.zeros((4, 4, 4, 3))
        field = field.at[0, 0, 0, 0].set(1.0)
        field = field.at[0, 0, 0, 1].set(2.0)
        field = field.at[0, 0, 0, 2].set(3.0)
        shifted = fftshift_3d(field)
        assert shifted[2, 2, 2, 0] == 1.0
        assert shifted[2, 2, 2, 1] == 2.0
        assert shifted[2, 2, 2, 2] == 3.0

    def test_invalid_ndim(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError):
            forward_fft_3d(jnp.ones((4, 4)))
        with pytest.raises(ValueError):
            inverse_fft_3d(jnp.ones((4, 4)))
        with pytest.raises(ValueError):
            fftshift_3d(jnp.ones((4, 4)))


class TestDCReplacement:
    """Tests for DC component replacement."""

    def test_replace_dc_component_3d(self):
        """Test DC replacement for 3D field."""
        # Create a field with known values
        field = jnp.zeros((4, 4, 4), dtype=jnp.complex64)
        # Set DC component
        field = field.at[0, 0, 0].set(100.0 + 0j)
        # Set the 6 face-adjacent neighbors
        field = field.at[0, 0, 1].set(1.0 + 1j)  # +x
        field = field.at[0, 0, 3].set(2.0 + 2j)  # -x (wrap)
        field = field.at[0, 1, 0].set(3.0 + 3j)  # +y
        field = field.at[0, 3, 0].set(4.0 + 4j)  # -y (wrap)
        field = field.at[1, 0, 0].set(5.0 + 5j)  # +z
        field = field.at[3, 0, 0].set(6.0 + 6j)  # -z (wrap)

        result = replace_dc_component(field)

        # Check DC is replaced with average of neighbors
        expected_avg = (1 + 2 + 3 + 4 + 5 + 6) / 6.0 + (1 + 2 + 3 + 4 + 5 + 6) / 6.0 * 1j
        assert jnp.allclose(result[0, 0, 0], expected_avg)

        # Check neighbors are unchanged
        assert result[0, 0, 1] == 1.0 + 1j
        assert result[0, 1, 0] == 3.0 + 3j
        assert result[1, 0, 0] == 5.0 + 5j

    def test_replace_dc_component_4d(self):
        """Test DC replacement for 4D vector field."""
        field = jnp.zeros((4, 4, 4, 3), dtype=jnp.complex64)
        # Set DC component for each polarization
        field = field.at[0, 0, 0, 0].set(100.0)
        field = field.at[0, 0, 0, 1].set(200.0)
        field = field.at[0, 0, 0, 2].set(300.0)
        # Set one neighbor
        field = field.at[0, 0, 1, 0].set(6.0)
        field = field.at[0, 0, 1, 1].set(12.0)
        field = field.at[0, 0, 1, 2].set(18.0)

        result = replace_dc_component(field)

        # DC should be replaced (original 100, 200, 300 should be gone)
        assert result[0, 0, 0, 0] != 100.0
        assert result[0, 0, 0, 1] != 200.0
        assert result[0, 0, 0, 2] != 300.0

    def test_replace_dc_preserves_other_values(self):
        """Test that DC replacement only modifies index [0,0,0]."""
        field = jnp.ones((4, 4, 4), dtype=jnp.complex64) * 5.0
        field = field.at[0, 0, 0].set(100.0)

        result = replace_dc_component(field)

        # All non-DC values should remain unchanged
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if not (i == 0 and j == 0 and k == 0):
                        assert result[i, j, k] == 5.0

    def test_invalid_dc_ndim(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError):
            replace_dc_component(jnp.ones((4, 4)))


class TestFFTPipeline:
    """Tests for the complete FFT pipeline."""

    def test_fft_pipeline_shapes(self):
        """Test that pipeline preserves shapes."""
        px = jnp.ones((8, 8, 8), dtype=jnp.complex64)
        py = jnp.ones((8, 8, 8), dtype=jnp.complex64) * 2
        pz = jnp.ones((8, 8, 8), dtype=jnp.complex64) * 3

        fx, fy, fz = fft_pipeline(px, py, pz)

        assert fx.shape == (8, 8, 8)
        assert fy.shape == (8, 8, 8)
        assert fz.shape == (8, 8, 8)

    def test_fft_pipeline_vectorized(self):
        """Test vectorized pipeline interface."""
        polarization = jnp.ones((8, 8, 8, 3), dtype=jnp.complex64)
        result = fft_pipeline_vectorized(polarization)
        assert result.shape == (8, 8, 8, 3)

    def test_pipeline_consistent_with_individual_calls(self):
        """Test that pipeline gives same result as individual calls."""
        px = jnp.linspace(0, 1, 64, dtype=jnp.complex64).reshape((4, 4, 4))
        py = jnp.linspace(1, 2, 64, dtype=jnp.complex64).reshape((4, 4, 4))
        pz = jnp.linspace(2, 3, 64, dtype=jnp.complex64).reshape((4, 4, 4))

        # Using pipeline
        fx_pipe, fy_pipe, fz_pipe = fft_pipeline(px, py, pz)

        # Manual steps
        fx_manual = fftshift_3d(replace_dc_component(forward_fft_3d(px)))
        fy_manual = fftshift_3d(replace_dc_component(forward_fft_3d(py)))
        fz_manual = fftshift_3d(replace_dc_component(forward_fft_3d(pz)))

        assert jnp.allclose(fx_pipe, fx_manual)
        assert jnp.allclose(fy_pipe, fy_manual)
        assert jnp.allclose(fz_pipe, fz_manual)


class TestQGrid:
    """Tests for q-space grid computation."""

    def test_compute_q_grid_cubic(self):
        """Test q-grid computation for cubic grid."""
        n = 16
        physical_size = 1.0
        wavelength = 1.0
        qz, qy, qx = compute_q_grid(n, physical_size, wavelength)
        assert qx.shape == (n,)
        assert qy.shape == (n,)
        assert qz.shape == (n,)

    def test_compute_q_grid_non_cubic(self):
        """Test q-grid computation for non-cubic grid."""
        nz, ny, nx = 8, 16, 32
        physical_size = 1.0
        wavelength = 1.0
        qz, qy, qx = compute_q_grid((nz, ny, nx), physical_size, wavelength)
        assert qx.shape == (nx,)
        assert qy.shape == (ny,)
        assert qz.shape == (nz,)

    def test_q_grid_zero_at_center(self):
        """Test that q=0 is at the center after fftshift."""
        n = 8
        physical_size = 1.0
        wavelength = 1.0
        qz, qy, qx = compute_q_grid(n, physical_size, wavelength)
        # For even n, zero frequency should be at index n/2
        center_idx = n // 2
        assert jnp.abs(qx[center_idx]) < 1e-10
        assert jnp.abs(qy[center_idx]) < 1e-10
        assert jnp.abs(qz[center_idx]) < 1e-10
