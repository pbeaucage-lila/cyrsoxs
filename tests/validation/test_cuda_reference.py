"""Validation tests comparing JAX port against CUDA CyRSoXS reference.

These tests load pre-computed reference data from CUDA CyRSoXS and compare
against JAX implementation at each pipeline stage.

Acceptance Criteria (from bead cy-pwd):
- Polarization: max relative error < 1e-5
- FFT: max relative error < 1e-5
- Final projection: max relative error < 1e-4
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from cyrsoxs_jax import polarization, fft

REFERENCE_DATA_DIR = Path(__file__).parent.parent / "reference_data"

# Tolerances from acceptance criteria
POLARIZATION_RTOL = 1e-5
FFT_RTOL = 1e-5
PROJECTION_RTOL = 1e-4


def load_reference(name: str) -> dict:
    """Load reference data from npz file."""
    path = REFERENCE_DATA_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"Reference data not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def max_relative_error(actual, expected):
    """Compute max relative error, handling zeros."""
    mask = jnp.abs(expected) > 1e-12
    if not jnp.any(mask):
        return jnp.max(jnp.abs(actual - expected))
    rel_err = jnp.abs(actual - expected) / jnp.abs(expected)
    return jnp.max(jnp.where(mask, rel_err, 0.0))


class TestPolarizationValidation:
    """Validate polarization computation against CUDA reference."""

    @pytest.fixture
    def sphere_data(self):
        """Load sphere morphology reference data."""
        return load_reference("sphere_polarization")

    def test_sphere_polarization(self, sphere_data):
        """Test polarization on sphere morphology."""
        morphology = jnp.array(sphere_data["morphology"])
        n_para = jnp.array(sphere_data["n_para"])
        n_perp = jnp.array(sphere_data["n_perp"])
        euler_angles = jnp.array(sphere_data["euler_angles"])
        incident_field = jnp.array(sphere_data["incident_field"])
        expected = jnp.array(sphere_data["polarization_cuda"])

        result = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )

        error = max_relative_error(result, expected)
        assert error < POLARIZATION_RTOL, f"Polarization error {error:.2e} > {POLARIZATION_RTOL}"

    @pytest.fixture
    def lamellar_data(self):
        """Load lamellar morphology reference data."""
        return load_reference("lamellar_polarization")

    def test_lamellar_polarization(self, lamellar_data):
        """Test polarization on lamellar morphology."""
        morphology = jnp.array(lamellar_data["morphology"])
        n_para = jnp.array(lamellar_data["n_para"])
        n_perp = jnp.array(lamellar_data["n_perp"])
        euler_angles = jnp.array(lamellar_data["euler_angles"])
        incident_field = jnp.array(lamellar_data["incident_field"])
        expected = jnp.array(lamellar_data["polarization_cuda"])

        result = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )

        error = max_relative_error(result, expected)
        assert error < POLARIZATION_RTOL, f"Polarization error {error:.2e} > {POLARIZATION_RTOL}"


class TestFFTValidation:
    """Validate FFT computation against CUDA reference."""

    @pytest.fixture
    def fft_data(self):
        """Load FFT reference data."""
        return load_reference("fft_reference")

    def test_forward_fft_3d(self, fft_data):
        """Test 3D forward FFT."""
        input_field = jnp.array(fft_data["input_field"])
        expected = jnp.array(fft_data["fft_cuda"])

        result = fft.forward_fft_3d(input_field)

        error = max_relative_error(result, expected)
        assert error < FFT_RTOL, f"FFT error {error:.2e} > {FFT_RTOL}"

    def test_fft_polarization_field(self, fft_data):
        """Test FFT of polarization field (4D)."""
        if "polarization_field" not in fft_data:
            pytest.skip("4D polarization FFT reference not available")

        input_field = jnp.array(fft_data["polarization_field"])
        expected = jnp.array(fft_data["polarization_fft_cuda"])

        result = fft.forward_fft_3d(input_field)

        error = max_relative_error(result, expected)
        assert error < FFT_RTOL, f"Polarization FFT error {error:.2e} > {FFT_RTOL}"


class TestProjectionValidation:
    """Validate final projection against CUDA reference."""

    @pytest.fixture
    def projection_data(self):
        """Load projection reference data."""
        return load_reference("projection_reference")

    def test_2d_projection(self, projection_data):
        """Test 2D detector projection."""
        pytest.skip("Projection not yet implemented in JAX port")


class TestEndToEndValidation:
    """End-to-end validation of full simulation pipeline."""

    @pytest.fixture
    def e2e_data(self):
        """Load end-to-end reference data."""
        return load_reference("end_to_end")

    def test_full_simulation(self, e2e_data):
        """Test complete simulation pipeline."""
        pytest.skip("Full simulation not yet implemented in JAX port")


class TestMultiMaterialValidation:
    """Validate multi-material handling."""

    @pytest.fixture
    def multi_mat_data(self):
        """Load multi-material reference data."""
        return load_reference("multi_material")

    def test_two_material_polarization(self, multi_mat_data):
        """Test polarization with two materials."""
        morphology = jnp.array(multi_mat_data["morphology"])
        n_para = jnp.array(multi_mat_data["n_para"])
        n_perp = jnp.array(multi_mat_data["n_perp"])
        euler_angles = jnp.array(multi_mat_data["euler_angles"])
        incident_field = jnp.array(multi_mat_data["incident_field"])
        expected = jnp.array(multi_mat_data["polarization_cuda"])

        result = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )

        error = max_relative_error(result, expected)
        assert error < POLARIZATION_RTOL, f"Multi-material error {error:.2e} > {POLARIZATION_RTOL}"


class TestRotationValidation:
    """Validate rotation matrix application."""

    @pytest.fixture
    def rotation_data(self):
        """Load rotation reference data."""
        return load_reference("rotation_reference")

    def test_polarization_with_rotation(self, rotation_data):
        """Test polarization with non-identity rotation matrix."""
        morphology = jnp.array(rotation_data["morphology"])
        n_para = jnp.array(rotation_data["n_para"])
        n_perp = jnp.array(rotation_data["n_perp"])
        euler_angles = jnp.array(rotation_data["euler_angles"])
        incident_field = jnp.array(rotation_data["incident_field"])
        rotation_matrix = jnp.array(rotation_data["rotation_matrix"])
        expected = jnp.array(rotation_data["polarization_cuda"])

        result = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field,
            rotation_matrix=rotation_matrix
        )

        error = max_relative_error(result, expected)
        assert error < POLARIZATION_RTOL, f"Rotation error {error:.2e} > {POLARIZATION_RTOL}"
