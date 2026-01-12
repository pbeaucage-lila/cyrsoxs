"""Tests for polarization computation."""

import jax.numpy as jnp
import pytest
from cyrsoxs_jax import polarization


class TestPolarizationFieldEuler:
    """Tests for Euler angle polarization computation."""

    def test_output_shape(self):
        N, num_materials = 4, 2
        morphology = jnp.zeros((N, N, N, num_materials, 4))
        n_para = jnp.array([1.0 + 0.01j, 1.1 + 0.02j])
        n_perp = jnp.array([1.0 + 0.005j, 1.05 + 0.01j])
        euler_angles = jnp.zeros((N, N, N, 3))
        incident_field = jnp.array([1.0, 0.0, 0.0])

        P = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )
        assert P.shape == (N, N, N, 3)
        assert P.dtype == jnp.complex64

    def test_zero_volume_fraction_gives_zero(self):
        N, num_materials = 4, 1
        morphology = jnp.zeros((N, N, N, num_materials, 4))
        n_para = jnp.array([1.5 + 0.1j])
        n_perp = jnp.array([1.2 + 0.05j])
        euler_angles = jnp.zeros((N, N, N, 3))
        incident_field = jnp.array([1.0, 0.0, 0.0])

        P = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )
        assert jnp.allclose(P, 0.0)

    def test_isotropic_material(self):
        """Isotropic material (n_para = n_perp) should give scalar response."""
        N, num_materials = 4, 1
        morphology = jnp.zeros((N, N, N, num_materials, 4))
        morphology = morphology.at[:, :, :, 0, 2].set(1.0)  # vfrac = 1
        morphology = morphology.at[:, :, :, 0, 3].set(1.0)  # S = 1

        n = 1.5 + 0.1j
        n_para = jnp.array([n])
        n_perp = jnp.array([n])
        euler_angles = jnp.zeros((N, N, N, 3))
        incident_field = jnp.array([1.0, 0.0, 0.0])

        P = polarization.compute_polarization_field(
            morphology, n_para, n_perp, euler_angles, incident_field
        )
        # For isotropic, Py and Pz should be zero when E is along x
        assert jnp.allclose(P[:, :, :, 1], 0.0, atol=1e-6)
        assert jnp.allclose(P[:, :, :, 2], 0.0, atol=1e-6)


class TestPolarizationVectorMorphology:
    """Tests for vector morphology polarization computation."""

    def test_output_shape(self):
        N, num_materials = 4, 2
        morphology = jnp.zeros((N, N, N, num_materials, 4))
        n_para = jnp.array([1.0 + 0.01j, 1.1 + 0.02j])
        n_perp = jnp.array([1.0 + 0.005j, 1.05 + 0.01j])
        incident_field = jnp.array([1.0, 0.0, 0.0])

        P = polarization.compute_polarization_vector_morphology(
            morphology, n_para, n_perp, incident_field
        )
        assert P.shape == (N, N, N, 3)

    def test_zero_orientation_gives_zero(self):
        N, num_materials = 4, 1
        morphology = jnp.zeros((N, N, N, num_materials, 4))
        n_para = jnp.array([1.5 + 0.1j])
        n_perp = jnp.array([1.2 + 0.05j])
        incident_field = jnp.array([1.0, 0.0, 0.0])

        P = polarization.compute_polarization_vector_morphology(
            morphology, n_para, n_perp, incident_field
        )
        assert jnp.allclose(P, 0.0)


class TestBuildDielectricTensor:
    """Tests for dielectric tensor construction."""

    def test_output_shape(self):
        N = 4
        sx = jnp.ones((N, N, N))
        sy = jnp.zeros((N, N, N))
        sz = jnp.zeros((N, N, N))
        phi_a = jnp.ones((N, N, N))
        phi_ui = jnp.zeros((N, N, N))
        phi = jnp.ones((N, N, N))

        tensor = polarization.build_dielectric_tensor(
            sx, sy, sz, phi_a, phi_ui, phi,
            npar=2.0+0.1j, nper=1.5+0.05j, nsum=5.0+0.2j
        )
        assert tensor.shape == (N, N, N, 6)


class TestEffectiveDielectric:
    """Tests for effective dielectric computation."""

    def test_single_material(self):
        vfrac = jnp.array([1.0])
        tensors = jnp.array([[[1.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 3.0]]])
        result = polarization.compute_effective_dielectric(vfrac, tensors)
        assert jnp.allclose(result, tensors[0])

    def test_equal_mixing(self):
        vfrac = jnp.array([0.5, 0.5])
        t1 = jnp.eye(3) * 2.0
        t2 = jnp.eye(3) * 4.0
        tensors = jnp.stack([t1, t2])
        result = polarization.compute_effective_dielectric(vfrac, tensors)
        expected = jnp.eye(3) * 3.0
        assert jnp.allclose(result, expected)
