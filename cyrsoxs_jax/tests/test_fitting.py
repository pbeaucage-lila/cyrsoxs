"""Tests for the autodiff-ready fitting API."""

import jax
import jax.numpy as jnp
import pytest

from cyrsoxs_jax.fitting import (
    OpticalParams,
    MorphologyParams,
    FitParams,
    simulate_differentiable,
    make_forward_model,
    make_checkpointed_forward_model,
    mse_loss,
    chi_squared_loss,
    log_intensity_loss,
    create_fit_state,
    fit_step,
)
from cyrsoxs_jax.types import (
    Morphology,
    VoxelData,
    MaterialData,
    SimulationConfig,
    MorphologyType,
)


@pytest.fixture
def simple_morphology():
    """Create a simple test morphology."""
    nz, ny, nx = 8, 8, 8
    num_materials = 2

    # Create simple vector morphology data
    alignment = jnp.zeros((num_materials, nz, ny, nx, 3))
    alignment = alignment.at[0, :, :, :, 2].set(1.0)  # Material 0: z-aligned
    alignment = alignment.at[1, :, :, :, 0].set(1.0)  # Material 1: x-aligned

    unaligned = jnp.zeros((num_materials, nz, ny, nx))
    unaligned = unaligned.at[0].set(0.1)
    unaligned = unaligned.at[1].set(0.2)

    voxel_data = VoxelData.from_vector_morphology(
        alignment=alignment,
        unaligned_fraction=unaligned,
        num_materials=num_materials,
    )

    return Morphology(voxel_data=voxel_data, phys_size=1.0)


@pytest.fixture
def simple_config():
    """Create a simple simulation config."""
    return SimulationConfig(
        energies=jnp.array([285.0]),
        k_vectors=jnp.array([[0.0, 0.0, 1.0]]),
        start_angle=0.0,
        end_angle=0.0,
        increment_angle=1.0,
    )


@pytest.fixture
def simple_optical_params():
    """Create simple optical parameters."""
    return OpticalParams(
        delta_para=jnp.array([[1e-4, 2e-4]]),
        beta_para=jnp.array([[1e-5, 2e-5]]),
        delta_perp=jnp.array([[0.5e-4, 1e-4]]),
        beta_perp=jnp.array([[0.5e-5, 1e-5]]),
    )


class TestOpticalParams:
    """Tests for OpticalParams."""

    def test_to_complex(self, simple_optical_params):
        """Test conversion to complex refractive indices."""
        npara, nperp = simple_optical_params.to_complex()

        assert npara.shape == (1, 2)
        assert nperp.shape == (1, 2)
        assert jnp.iscomplexobj(npara)
        assert jnp.iscomplexobj(nperp)

    def test_from_material_data(self):
        """Test creation from MaterialData."""
        material_data = MaterialData.from_optical_constants(
            delta_para=jnp.array([[1e-4, 2e-4]]),
            beta_para=jnp.array([[1e-5, 2e-5]]),
            delta_perp=jnp.array([[0.5e-4, 1e-4]]),
            beta_perp=jnp.array([[0.5e-5, 1e-5]]),
            energies=jnp.array([285.0]),
        )

        optical_params = OpticalParams.from_material_data(material_data)

        assert optical_params.delta_para.shape == (1, 2)
        # Use relative tolerance for floating point comparison
        assert jnp.allclose(optical_params.delta_para, jnp.array([[1e-4, 2e-4]]), rtol=1e-3)

    def test_pytree_registration(self, simple_optical_params):
        """Test that OpticalParams is a valid pytree."""
        leaves = jax.tree_util.tree_leaves(simple_optical_params)
        assert len(leaves) == 4

        # Test roundtrip
        flat, treedef = jax.tree_util.tree_flatten(simple_optical_params)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
        assert jnp.allclose(reconstructed.delta_para, simple_optical_params.delta_para)


class TestMorphologyParams:
    """Tests for MorphologyParams."""

    def test_from_morphology(self, simple_morphology):
        """Test creation from Morphology."""
        morph_params = MorphologyParams.from_morphology(simple_morphology)

        assert morph_params.data.shape == simple_morphology.voxel_data.data.shape
        assert morph_params.morphology_type == MorphologyType.VECTOR_MORPHOLOGY

    def test_pytree_registration(self, simple_morphology):
        """Test that MorphologyParams is a valid pytree."""
        morph_params = MorphologyParams.from_morphology(simple_morphology)
        leaves = jax.tree_util.tree_leaves(morph_params)
        assert len(leaves) == 1


class TestDifferentiableSimulation:
    """Tests for differentiable simulation functions."""

    def test_simulate_differentiable_runs(
        self, simple_morphology, simple_config, simple_optical_params
    ):
        """Test that simulate_differentiable produces output."""
        result = simulate_differentiable(
            simple_optical_params,
            simple_morphology.voxel_data.data,
            float(simple_config.energies[0]),
            simple_config.k_vectors[0],
            simple_morphology.phys_size,
            simple_morphology.shape,
            int(simple_morphology.morphology_type),
            simple_config.angles,
        )

        nz, ny, nx = simple_morphology.shape
        assert result.shape == (ny, nx)

    def test_gradient_computes(
        self, simple_morphology, simple_config, simple_optical_params
    ):
        """Test that gradients can be computed."""
        # Create morphology with spatial variation for nonzero gradients
        nz, ny, nx = 8, 8, 8
        num_materials = 2

        # Create varying volume fractions across the volume
        x = jnp.linspace(0, 1, nx)
        y = jnp.linspace(0, 1, ny)
        z = jnp.linspace(0, 1, nz)
        zz, yy, xx = jnp.meshgrid(z, y, x, indexing='ij')

        alignment = jnp.zeros((num_materials, nz, ny, nx, 3))
        alignment = alignment.at[0, :, :, :, 2].set(xx)  # Varying alignment
        alignment = alignment.at[1, :, :, :, 0].set(1 - xx)

        unaligned = jnp.zeros((num_materials, nz, ny, nx))
        unaligned = unaligned.at[0].set(0.1 * yy)
        unaligned = unaligned.at[1].set(0.2 * (1 - yy))

        voxel_data = VoxelData.from_vector_morphology(
            alignment=alignment,
            unaligned_fraction=unaligned,
            num_materials=num_materials,
        )
        varying_morphology = Morphology(voxel_data=voxel_data, phys_size=1.0)

        def loss_fn(optical_params):
            result = simulate_differentiable(
                optical_params,
                varying_morphology.voxel_data.data,
                float(simple_config.energies[0]),
                simple_config.k_vectors[0],
                varying_morphology.phys_size,
                varying_morphology.shape,
                int(varying_morphology.morphology_type),
                simple_config.angles,
            )
            return jnp.mean(result**2)

        grads = jax.grad(loss_fn)(simple_optical_params)

        # Check that gradients exist and have correct shapes
        assert grads.delta_para.shape == simple_optical_params.delta_para.shape
        assert grads.beta_para.shape == simple_optical_params.beta_para.shape
        # Gradients should be finite (may be very small but not NaN/Inf)
        assert jnp.all(jnp.isfinite(grads.delta_para))
        assert jnp.all(jnp.isfinite(grads.beta_para))

    def test_make_forward_model(
        self, simple_morphology, simple_config, simple_optical_params
    ):
        """Test forward model factory."""
        forward = make_forward_model(simple_morphology, simple_config)

        result = forward(
            simple_optical_params, simple_morphology.voxel_data.data
        )

        nz, ny, nx = simple_morphology.shape
        assert result.shape == (ny, nx)

    def test_checkpointed_forward_model(
        self, simple_morphology, simple_config, simple_optical_params
    ):
        """Test checkpointed forward model."""
        forward = make_checkpointed_forward_model(simple_morphology, simple_config)

        result = forward(
            simple_optical_params, simple_morphology.voxel_data.data
        )

        nz, ny, nx = simple_morphology.shape
        assert result.shape == (ny, nx)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 2.1], [3.1, 4.1]])

        loss = mse_loss(pred, target)

        expected = jnp.mean((pred - target) ** 2)
        assert jnp.allclose(loss, expected)

    def test_mse_loss_with_mask(self):
        """Test MSE loss with mask."""
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 2.1], [3.1, 4.1]])
        mask = jnp.array([[1.0, 1.0], [0.0, 0.0]])

        loss = mse_loss(pred, target, mask)

        # Only top row should contribute
        expected = jnp.mean((pred[:1] - target[:1]) ** 2)
        assert jnp.allclose(loss, expected)

    def test_chi_squared_loss(self):
        """Test chi-squared loss computation."""
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 2.1], [3.1, 4.1]])
        uncertainty = jnp.array([[0.1, 0.1], [0.1, 0.1]])

        loss = chi_squared_loss(pred, target, uncertainty)

        expected = jnp.mean(((pred - target) / uncertainty) ** 2)
        assert jnp.allclose(loss, expected)

    def test_log_intensity_loss(self):
        """Test log intensity loss."""
        pred = jnp.array([[1.0, 10.0], [100.0, 1000.0]])
        target = jnp.array([[1.1, 11.0], [110.0, 1100.0]])

        loss = log_intensity_loss(pred, target)

        # Should be finite
        assert jnp.isfinite(loss)

    def test_loss_gradients(self):
        """Test that loss functions are differentiable."""
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 2.1], [3.1, 4.1]])

        grad_mse = jax.grad(lambda p: mse_loss(p, target))(pred)
        assert grad_mse.shape == pred.shape

        uncertainty = jnp.ones_like(pred) * 0.1
        grad_chi = jax.grad(lambda p: chi_squared_loss(p, target, uncertainty))(pred)
        assert grad_chi.shape == pred.shape

        grad_log = jax.grad(lambda p: log_intensity_loss(p, target))(pred)
        assert grad_log.shape == pred.shape


class TestFitState:
    """Tests for fitting state management."""

    def test_create_fit_state(self, simple_optical_params):
        """Test FitState creation."""
        pytest.importorskip("optax")
        import optax

        params = FitParams(optical=simple_optical_params, morphology_data=None)
        optimizer = optax.adam(1e-3)

        state = create_fit_state(params, optimizer)

        assert state.step == 0
        assert len(state.loss_history) == 0
        assert state.params.optical is not None

    def test_fit_step(self, simple_optical_params):
        """Test single optimization step."""
        pytest.importorskip("optax")
        import optax

        params = FitParams(optical=simple_optical_params, morphology_data=None)
        optimizer = optax.adam(1e-3)

        # Simple quadratic loss
        target = OpticalParams(
            delta_para=jnp.array([[2e-4, 3e-4]]),
            beta_para=jnp.array([[2e-5, 3e-5]]),
            delta_perp=jnp.array([[1e-4, 2e-4]]),
            beta_perp=jnp.array([[1e-5, 2e-5]]),
        )

        def loss_fn(params):
            diff_delta = params.optical.delta_para - target.delta_para
            diff_beta = params.optical.beta_para - target.beta_para
            return jnp.mean(diff_delta**2) + jnp.mean(diff_beta**2)

        state = create_fit_state(params, optimizer)
        new_state, loss_val = fit_step(state, loss_fn, optimizer)

        assert new_state.step == 1
        assert len(new_state.loss_history) == 1
        assert jnp.isfinite(loss_val)


class TestEndToEndFitting:
    """End-to-end fitting tests."""

    def test_fit_optical_constants(
        self, simple_morphology, simple_config, simple_optical_params
    ):
        """Test fitting optical constants to target pattern."""
        pytest.importorskip("optax")
        import optax

        # Generate "experimental" data with slightly different params
        target_optical = OpticalParams(
            delta_para=simple_optical_params.delta_para * 1.1,
            beta_para=simple_optical_params.beta_para * 1.1,
            delta_perp=simple_optical_params.delta_perp * 1.1,
            beta_perp=simple_optical_params.beta_perp * 1.1,
        )

        forward = make_forward_model(simple_morphology, simple_config)
        target_pattern = forward(target_optical, simple_morphology.voxel_data.data)

        # Set up fitting
        def loss_fn(params):
            pred = forward(params.optical, simple_morphology.voxel_data.data)
            return mse_loss(pred, target_pattern)

        params = FitParams(optical=simple_optical_params, morphology_data=None)
        optimizer = optax.adam(1e-4)

        state = create_fit_state(params, optimizer)
        initial_loss = loss_fn(state.params)

        # Run a few steps
        for _ in range(5):
            state, loss_val = fit_step(state, loss_fn, optimizer)

        # Loss should decrease
        assert loss_val < initial_loss
