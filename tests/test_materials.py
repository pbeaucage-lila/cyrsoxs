"""Tests for optical constants and material handling."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyrsoxs_jax.materials import (
    OpticalConstantsRaw,
    combine_materials,
    compute_dielectric_tensor,
    compute_dielectric_tensor_from_delta_beta,
    create_vacuum_material,
    get_refractive_index_at_energy,
    interpolate_optical_constants,
    load_processed_optical_constants,
    load_raw_optical_constants,
)
from cyrsoxs_jax.types import MaterialData


class TestOpticalConstantsRaw:
    """Tests for OpticalConstantsRaw dataclass."""

    def test_to_material_data(self):
        """Test conversion to MaterialData."""
        constants = OpticalConstantsRaw(
            energies=jnp.array([280.0, 285.0, 290.0]),
            delta_para=jnp.array([0.001, 0.002, 0.001]),
            beta_para=jnp.array([0.0001, 0.0002, 0.0001]),
            delta_perp=jnp.array([0.0008, 0.0015, 0.0008]),
            beta_perp=jnp.array([0.00008, 0.00015, 0.00008]),
        )

        material_data = constants.to_material_data()

        assert material_data.num_materials == 1
        assert material_data.npara.shape == (3, 1)
        assert material_data.nperp.shape == (3, 1)
        assert jnp.allclose(material_data.energies, constants.energies)


class TestLoadProcessedOpticalConstants:
    """Tests for loading processed format files."""

    def test_load_processed_format(self):
        """Test loading EnergyDataN block format."""
        content = """EnergyData0:
{
Energy = 280.0;
BetaPara = 0.00042089;
BetaPerp = 0.00042103;
DeltaPara = 0.0008513;
DeltaPerp = 0.0011596;
}
EnergyData1:
{
Energy = 285.0;
BetaPara = 0.00015347;
BetaPerp = 0.00011305;
DeltaPara = -0.00045534;
DeltaPerp = -1.2197e-05;
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()

            constants = load_processed_optical_constants(f.name)

        assert len(constants.energies) == 2
        assert jnp.allclose(constants.energies, jnp.array([280.0, 285.0]))
        assert jnp.isclose(constants.beta_para[0], 0.00042089)
        assert jnp.isclose(constants.delta_para[1], -0.00045534)


class TestLoadRawOpticalConstants:
    """Tests for loading raw format files."""

    def test_load_raw_format(self):
        """Test loading tab-separated raw format."""
        content = """beta_para\tbeta_perp\tdelta_para\tdelta_perp\tdelta\tbeta\tenergy
0.0001\t0.00008\t0.001\t0.0008\t0.0009\t0.00009\t280.0
0.0002\t0.00015\t0.002\t0.0015\t0.00175\t0.000175\t285.0
0.0001\t0.00008\t0.001\t0.0008\t0.0009\t0.00009\t290.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()

            constants = load_raw_optical_constants(f.name)

        assert len(constants.energies) == 3
        assert jnp.allclose(constants.energies, jnp.array([280.0, 285.0, 290.0]))


class TestInterpolation:
    """Tests for optical constants interpolation."""

    def test_interpolate_to_new_energies(self):
        """Test interpolating to different energy values."""
        constants = OpticalConstantsRaw(
            energies=jnp.array([280.0, 290.0]),
            delta_para=jnp.array([0.001, 0.002]),
            beta_para=jnp.array([0.0001, 0.0002]),
            delta_perp=jnp.array([0.0008, 0.0016]),
            beta_perp=jnp.array([0.00008, 0.00016]),
        )

        target_energies = jnp.array([280.0, 285.0, 290.0])
        interpolated = interpolate_optical_constants(constants, target_energies)

        assert len(interpolated.energies) == 3
        # Check midpoint is linearly interpolated
        assert jnp.isclose(interpolated.delta_para[1], 0.0015)
        assert jnp.isclose(interpolated.beta_para[1], 0.00015)

    def test_interpolation_is_differentiable(self):
        """Test that interpolation can be differentiated."""
        constants = OpticalConstantsRaw(
            energies=jnp.array([280.0, 290.0]),
            delta_para=jnp.array([0.001, 0.002]),
            beta_para=jnp.array([0.0001, 0.0002]),
            delta_perp=jnp.array([0.0008, 0.0016]),
            beta_perp=jnp.array([0.00008, 0.00016]),
        )

        def loss_fn(target_energy):
            target_energies = jnp.array([target_energy])
            interp = interpolate_optical_constants(constants, target_energies)
            return interp.delta_para[0]

        grad = jax.grad(loss_fn)(285.0)
        # Gradient should be (0.002 - 0.001) / (290 - 280) = 0.0001
        assert jnp.isclose(grad, 0.0001, atol=1e-6)


class TestCombineMaterials:
    """Tests for combining multiple materials."""

    def test_combine_two_materials(self):
        """Test combining two materials into MaterialData."""
        mat1 = OpticalConstantsRaw(
            energies=jnp.array([280.0, 285.0]),
            delta_para=jnp.array([0.001, 0.002]),
            beta_para=jnp.array([0.0001, 0.0002]),
            delta_perp=jnp.array([0.0008, 0.0015]),
            beta_perp=jnp.array([0.00008, 0.00015]),
        )
        mat2 = OpticalConstantsRaw(
            energies=jnp.array([280.0, 285.0]),
            delta_para=jnp.array([0.0005, 0.001]),
            beta_para=jnp.array([0.00005, 0.0001]),
            delta_perp=jnp.array([0.0004, 0.0008]),
            beta_perp=jnp.array([0.00004, 0.00008]),
        )

        combined = combine_materials([mat1, mat2])

        assert combined.num_materials == 2
        assert combined.npara.shape == (2, 2)
        assert combined.nperp.shape == (2, 2)

    def test_combine_with_interpolation(self):
        """Test combining materials with different energy grids."""
        mat1 = OpticalConstantsRaw(
            energies=jnp.array([280.0, 290.0]),
            delta_para=jnp.array([0.001, 0.002]),
            beta_para=jnp.array([0.0001, 0.0002]),
            delta_perp=jnp.array([0.0008, 0.0016]),
            beta_perp=jnp.array([0.00008, 0.00016]),
        )
        mat2 = OpticalConstantsRaw(
            energies=jnp.array([275.0, 295.0]),
            delta_para=jnp.array([0.0005, 0.001]),
            beta_para=jnp.array([0.00005, 0.0001]),
            delta_perp=jnp.array([0.0004, 0.0008]),
            beta_perp=jnp.array([0.00004, 0.00008]),
        )

        target_energies = jnp.array([280.0, 285.0, 290.0])
        combined = combine_materials([mat1, mat2], target_energies)

        assert combined.num_materials == 2
        assert jnp.allclose(combined.energies, target_energies)


class TestVacuumMaterial:
    """Tests for vacuum material creation."""

    def test_create_vacuum(self):
        """Test creating vacuum optical constants."""
        energies = jnp.array([280.0, 285.0, 290.0])
        vacuum = create_vacuum_material(energies)

        assert jnp.allclose(vacuum.delta_para, 0.0)
        assert jnp.allclose(vacuum.beta_para, 0.0)
        assert jnp.allclose(vacuum.delta_perp, 0.0)
        assert jnp.allclose(vacuum.beta_perp, 0.0)


class TestDielectricTensor:
    """Tests for dielectric tensor computation."""

    def test_compute_from_refractive_index(self):
        """Test dielectric tensor from complex refractive index."""
        npara = 1 - 0.001 + 1j * 0.0001
        nperp = 1 - 0.0008 + 1j * 0.00008

        tensor = compute_dielectric_tensor(npara, nperp)

        assert tensor.shape == (3, 3)
        # Should be diagonal
        assert jnp.allclose(tensor - jnp.diag(jnp.diag(tensor)), 0.0)
        # Check values
        assert jnp.isclose(tensor[2, 2], npara ** 2)
        assert jnp.isclose(tensor[0, 0], nperp ** 2)

    def test_compute_from_delta_beta(self):
        """Test dielectric tensor from optical constants."""
        tensor = compute_dielectric_tensor_from_delta_beta(
            delta_para=0.001,
            beta_para=0.0001,
            delta_perp=0.0008,
            beta_perp=0.00008,
        )

        assert tensor.shape == (3, 3)
        # Check it matches expected values
        n_para = (1 - 0.001) + 1j * 0.0001
        n_perp = (1 - 0.0008) + 1j * 0.00008
        assert jnp.isclose(tensor[2, 2], n_para ** 2)
        assert jnp.isclose(tensor[0, 0], n_perp ** 2)


class TestGetRefractiveIndexAtEnergy:
    """Tests for energy-resolved refractive index lookup."""

    def test_get_at_exact_energy(self):
        """Test getting refractive index at an exact energy point."""
        material_data = MaterialData.from_optical_constants(
            delta_para=jnp.array([[0.001], [0.002]]),
            beta_para=jnp.array([[0.0001], [0.0002]]),
            delta_perp=jnp.array([[0.0008], [0.0015]]),
            beta_perp=jnp.array([[0.00008], [0.00015]]),
            energies=jnp.array([280.0, 285.0]),
        )

        npara, nperp = get_refractive_index_at_energy(material_data, 280.0)

        expected_npara = (1 - 0.001) + 1j * 0.0001
        assert jnp.isclose(npara[0], expected_npara)

    def test_get_at_interpolated_energy(self):
        """Test getting refractive index at interpolated energy."""
        material_data = MaterialData.from_optical_constants(
            delta_para=jnp.array([[0.001], [0.002]]),
            beta_para=jnp.array([[0.0001], [0.0002]]),
            delta_perp=jnp.array([[0.0008], [0.0016]]),
            beta_perp=jnp.array([[0.00008], [0.00016]]),
            energies=jnp.array([280.0, 290.0]),
        )

        npara, nperp = get_refractive_index_at_energy(material_data, 285.0)

        # At midpoint, should be average
        expected_npara = (1 - 0.0015) + 1j * 0.00015
        assert jnp.isclose(npara[0], expected_npara, atol=1e-6)
