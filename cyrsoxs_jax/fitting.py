"""Autodiff-ready fitting API for CyRSoXS-JAX.

This module provides differentiable forward models and utilities for fitting
morphology and material parameters to experimental scattering data using
gradient-based optimization.

Example:
    >>> # Basic fitting workflow
    >>> def loss(params):
    ...     pred = simulate_differentiable(params, morphology, config)
    ...     return jnp.mean((pred - experimental)**2)
    >>>
    >>> grads = jax.grad(loss)(params)
    >>> params = optax_update(params, grads)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from cyrsoxs_jax.types import (
    Morphology, MaterialData, SimulationConfig, VoxelData,
    MorphologyType,
)
from cyrsoxs_jax.simulate import (
    simulate_single_angle, energy_to_wavelength,
    _rotate_image_bilinear,
)
from cyrsoxs_jax.rotation import compute_base_configuration, compute_full_rotation


@dataclass
class OpticalParams:
    """Fittable optical constant parameters.

    Stores delta and beta components separately for fitting.
    All arrays have shape [num_energies, num_materials].

    Attributes:
        delta_para: Parallel delta component.
        beta_para: Parallel beta component.
        delta_perp: Perpendicular delta component.
        beta_perp: Perpendicular beta component.
    """
    delta_para: Array
    beta_para: Array
    delta_perp: Array
    beta_perp: Array

    def to_complex(self) -> tuple[Array, Array]:
        """Convert to complex refractive indices.

        Returns:
            Tuple of (npara, nperp) complex arrays.
        """
        npara = (1 - self.delta_para) + 1j * self.beta_para
        nperp = (1 - self.delta_perp) + 1j * self.beta_perp
        return npara, nperp

    @classmethod
    def from_material_data(cls, material_data: MaterialData) -> OpticalParams:
        """Extract fittable params from MaterialData."""
        return cls(
            delta_para=1 - material_data.npara.real,
            beta_para=material_data.npara.imag,
            delta_perp=1 - material_data.nperp.real,
            beta_perp=material_data.nperp.imag,
        )


@dataclass
class MorphologyParams:
    """Fittable morphology parameters.

    For vector morphology:
        alignment: Director vectors [num_materials, Nz, Ny, Nx, 3]
        unaligned: Unaligned fraction [num_materials, Nz, Ny, Nx]

    For Euler angles:
        s_param: Alignment parameter S
        theta: Rotation angle about X-axis
        psi: Second rotation angle about Z-axis
        vfrac: Volume fraction
    """
    data: Array  # [num_materials, Nz, Ny, Nx, 4]
    morphology_type: MorphologyType

    @classmethod
    def from_morphology(cls, morphology: Morphology) -> MorphologyParams:
        """Extract fittable params from Morphology."""
        return cls(
            data=morphology.voxel_data.data,
            morphology_type=morphology.voxel_data.morphology_type,
        )


class FitParams(NamedTuple):
    """Combined parameters for fitting.

    Use this to bundle parameters that should be optimized together.
    Only include arrays you want to fit - leave others as None.
    """
    optical: OpticalParams | None = None
    morphology_data: Array | None = None  # [num_materials, Nz, Ny, Nx, 4]


def _optical_params_flatten(op: OpticalParams):
    children = (op.delta_para, op.beta_para, op.delta_perp, op.beta_perp)
    return children, None


def _optical_params_unflatten(aux_data, children):
    delta_para, beta_para, delta_perp, beta_perp = children
    return OpticalParams(delta_para, beta_para, delta_perp, beta_perp)


def _morphology_params_flatten(mp: MorphologyParams):
    children = (mp.data,)
    aux_data = (mp.morphology_type,)
    return children, aux_data


def _morphology_params_unflatten(aux_data, children):
    (morphology_type,) = aux_data
    (data,) = children
    return MorphologyParams(data, morphology_type)


jax.tree_util.register_pytree_node(
    OpticalParams,
    _optical_params_flatten,
    _optical_params_unflatten,
)

jax.tree_util.register_pytree_node(
    MorphologyParams,
    _morphology_params_flatten,
    _morphology_params_unflatten,
)


def simulate_single_energy_diff(
    n_para: Array,
    n_perp: Array,
    morphology_data: Array,
    energy: float,
    k_vec: Array,
    phys_size: float,
    voxel_dims: tuple[int, int, int],
    morphology_type: MorphologyType,
    angles: Array,
) -> Array:
    """Differentiable simulation at a single energy and k-vector.

    This is the core differentiable forward model. All array inputs
    participate in gradient computation.

    Args:
        n_para: Parallel refractive indices [num_materials], complex.
        n_perp: Perpendicular refractive indices [num_materials], complex.
        morphology_data: Voxel data [num_materials, Nz, Ny, Nx, 4].
        energy: Photon energy in eV.
        k_vec: K-vector direction [3].
        phys_size: Physical voxel size in nm.
        voxel_dims: Tuple of (Nz, Ny, Nx).
        morphology_type: Type of morphology representation.
        angles: Array of rotation angles in degrees.

    Returns:
        2D scattering pattern [Ny, Nx].
    """
    wavelength = energy_to_wavelength(energy)
    k_mag = 2 * jnp.pi / wavelength
    incident_field = jnp.array([1.0, 0.0, 0.0])

    base_config = compute_base_configuration(k_vec)

    # Transpose for polarization functions
    morph_transposed = jnp.transpose(morphology_data, (1, 2, 3, 0, 4))

    nz, ny, nx = voxel_dims
    num_angles = len(angles)

    def process_angle(carry, angle_deg):
        projection_sum = carry
        angle_rad = (base_config.base_rot_angle + angle_deg) * jnp.pi / 180.0

        rotation_matrix = compute_full_rotation(
            k_vec, base_config, angle_deg * jnp.pi / 180.0
        )

        projection = simulate_single_angle(
            morph_transposed, n_para, n_perp, rotation_matrix,
            incident_field, k_vec, k_mag, phys_size, morphology_type,
        )

        projection_rotated = _rotate_image_bilinear(projection, angle_rad)
        return projection_sum + projection_rotated, None

    init_sum = jnp.zeros((ny, nx))
    projection_sum, _ = lax.scan(process_angle, init_sum, angles)

    return projection_sum / num_angles


@partial(jax.jit, static_argnames=['voxel_dims', 'morphology_type'])
def simulate_differentiable(
    optical_params: OpticalParams,
    morphology_data: Array,
    energy: float,
    k_vec: Array,
    phys_size: float,
    voxel_dims: tuple[int, int, int],
    morphology_type: int,
    angles: Array,
) -> Array:
    """JIT-compiled differentiable forward model.

    This function is designed for use with jax.grad(). All array
    parameters participate in autodiff.

    Args:
        optical_params: Fittable optical constants.
        morphology_data: Voxel data [num_materials, Nz, Ny, Nx, 4].
        energy: Photon energy in eV.
        k_vec: K-vector direction [3].
        phys_size: Physical voxel size in nm.
        voxel_dims: Static tuple of (Nz, Ny, Nx).
        morphology_type: Static int for morphology type.
        angles: Rotation angles in degrees.

    Returns:
        2D scattering pattern [Ny, Nx].
    """
    n_para, n_perp = optical_params.to_complex()
    # Get optical constants for single energy (index 0)
    n_para_e = n_para[0]
    n_perp_e = n_perp[0]

    mt = MorphologyType(morphology_type)

    return simulate_single_energy_diff(
        n_para_e, n_perp_e, morphology_data, energy,
        k_vec, phys_size, voxel_dims, mt, angles,
    )


def make_forward_model(
    morphology: Morphology,
    config: SimulationConfig,
    energy_idx: int = 0,
    k_idx: int = 0,
) -> Callable[[OpticalParams, Array], Array]:
    """Create a forward model function for fitting.

    Returns a function that takes (optical_params, morphology_data)
    and returns a 2D scattering pattern. This function is
    compatible with jax.grad() and jax.jit().

    Args:
        morphology: Base morphology (for metadata).
        config: Simulation configuration.
        energy_idx: Index of energy to simulate.
        k_idx: Index of k-vector to use.

    Returns:
        Forward model function.

    Example:
        >>> forward = make_forward_model(morphology, config)
        >>> def loss(optical_params, morph_data):
        ...     pred = forward(optical_params, morph_data)
        ...     return jnp.mean((pred - experimental)**2)
        >>> grads = jax.grad(loss, argnums=(0, 1))(opt_params, morph_data)
    """
    energy = float(config.energies[energy_idx])
    k_vec = config.k_vectors[k_idx]
    phys_size = morphology.phys_size
    voxel_dims = morphology.shape
    morphology_type = int(morphology.morphology_type)
    angles = config.angles

    def forward(optical_params: OpticalParams, morphology_data: Array) -> Array:
        return simulate_differentiable(
            optical_params, morphology_data, energy, k_vec,
            phys_size, voxel_dims, morphology_type, angles,
        )

    return forward


def mse_loss(prediction: Array, target: Array, mask: Array | None = None) -> Array:
    """Mean squared error loss.

    Args:
        prediction: Predicted scattering pattern.
        target: Experimental data.
        mask: Optional mask (1 = include, 0 = exclude).

    Returns:
        Scalar loss value.
    """
    diff_sq = (prediction - target) ** 2
    if mask is not None:
        diff_sq = diff_sq * mask
        return jnp.sum(diff_sq) / jnp.sum(mask)
    return jnp.mean(diff_sq)


def chi_squared_loss(
    prediction: Array,
    target: Array,
    uncertainty: Array,
    mask: Array | None = None,
) -> Array:
    """Chi-squared loss with experimental uncertainties.

    Args:
        prediction: Predicted scattering pattern.
        target: Experimental data.
        uncertainty: Standard deviation of each measurement.
        mask: Optional mask (1 = include, 0 = exclude).

    Returns:
        Reduced chi-squared value.
    """
    chi_sq = ((prediction - target) / uncertainty) ** 2
    if mask is not None:
        chi_sq = chi_sq * mask
        return jnp.sum(chi_sq) / jnp.sum(mask)
    return jnp.mean(chi_sq)


def log_intensity_loss(
    prediction: Array,
    target: Array,
    epsilon: float = 1e-10,
    mask: Array | None = None,
) -> Array:
    """Loss on log-transformed intensities.

    Useful when fitting over many orders of magnitude.

    Args:
        prediction: Predicted scattering pattern.
        target: Experimental data.
        epsilon: Small value to avoid log(0).
        mask: Optional mask.

    Returns:
        Scalar loss value.
    """
    log_pred = jnp.log(prediction + epsilon)
    log_target = jnp.log(target + epsilon)
    diff_sq = (log_pred - log_target) ** 2
    if mask is not None:
        diff_sq = diff_sq * mask
        return jnp.sum(diff_sq) / jnp.sum(mask)
    return jnp.mean(diff_sq)


# =============================================================================
# Gradient Checkpointing
# =============================================================================

def simulate_checkpointed(
    optical_params: OpticalParams,
    morphology_data: Array,
    energy: float,
    k_vec: Array,
    phys_size: float,
    voxel_dims: tuple[int, int, int],
    morphology_type: int,
    angles: Array,
) -> Array:
    """Memory-efficient simulation using gradient checkpointing.

    Uses jax.checkpoint to trade compute for memory during backprop.
    This is useful for large morphologies where storing all intermediate
    activations would exceed GPU memory.

    Args:
        Same as simulate_differentiable.

    Returns:
        2D scattering pattern [Ny, Nx].
    """
    @jax.checkpoint
    def _forward(optical_params, morphology_data):
        return simulate_differentiable(
            optical_params, morphology_data, energy, k_vec,
            phys_size, voxel_dims, morphology_type, angles,
        )

    return _forward(optical_params, morphology_data)


def make_checkpointed_forward_model(
    morphology: Morphology,
    config: SimulationConfig,
    energy_idx: int = 0,
    k_idx: int = 0,
) -> Callable[[OpticalParams, Array], Array]:
    """Create a memory-efficient forward model with gradient checkpointing.

    Same as make_forward_model but uses checkpointing to reduce memory
    usage during gradient computation.
    """
    energy = float(config.energies[energy_idx])
    k_vec = config.k_vectors[k_idx]
    phys_size = morphology.phys_size
    voxel_dims = morphology.shape
    morphology_type = int(morphology.morphology_type)
    angles = config.angles

    def forward(optical_params: OpticalParams, morphology_data: Array) -> Array:
        return simulate_checkpointed(
            optical_params, morphology_data, energy, k_vec,
            phys_size, voxel_dims, morphology_type, angles,
        )

    return forward


# =============================================================================
# Optax Integration
# =============================================================================

class FitState(NamedTuple):
    """State for iterative fitting.

    Attributes:
        params: Current parameter values.
        opt_state: Optax optimizer state.
        step: Current iteration number.
        loss_history: Array of loss values.
    """
    params: FitParams
    opt_state: any
    step: int
    loss_history: Array


def create_fit_state(
    params: FitParams,
    optimizer,
) -> FitState:
    """Initialize fitting state.

    Args:
        params: Initial parameter values.
        optimizer: Optax optimizer (e.g., optax.adam(1e-3)).

    Returns:
        Initialized FitState.
    """
    opt_state = optimizer.init(params)
    return FitState(
        params=params,
        opt_state=opt_state,
        step=0,
        loss_history=jnp.array([]),
    )


def fit_step(
    state: FitState,
    loss_fn: Callable[[FitParams], Array],
    optimizer,
) -> tuple[FitState, Array]:
    """Perform one optimization step.

    Args:
        state: Current fit state.
        loss_fn: Loss function taking FitParams and returning scalar.
        optimizer: Optax optimizer.

    Returns:
        Tuple of (new_state, loss_value).
    """
    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)

    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = _apply_updates(state.params, updates)

    new_history = jnp.append(state.loss_history, loss_val)

    new_state = FitState(
        params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1,
        loss_history=new_history,
    )

    return new_state, loss_val


def _apply_updates(params: FitParams, updates: FitParams) -> FitParams:
    """Apply optax updates to parameters."""
    new_optical = None
    new_morph = None

    if params.optical is not None and updates.optical is not None:
        new_optical = OpticalParams(
            delta_para=params.optical.delta_para + updates.optical.delta_para,
            beta_para=params.optical.beta_para + updates.optical.beta_para,
            delta_perp=params.optical.delta_perp + updates.optical.delta_perp,
            beta_perp=params.optical.beta_perp + updates.optical.beta_perp,
        )

    if params.morphology_data is not None and updates.morphology_data is not None:
        new_morph = params.morphology_data + updates.morphology_data

    return FitParams(optical=new_optical, morphology_data=new_morph)


def fit_loop(
    initial_params: FitParams,
    loss_fn: Callable[[FitParams], Array],
    optimizer,
    num_steps: int,
    callback: Callable[[int, FitParams, Array], None] | None = None,
) -> FitState:
    """Run fitting loop.

    Args:
        initial_params: Starting parameter values.
        loss_fn: Loss function.
        optimizer: Optax optimizer.
        num_steps: Number of optimization steps.
        callback: Optional callback(step, params, loss) called each step.

    Returns:
        Final FitState with optimized parameters.

    Example:
        >>> import optax
        >>> optimizer = optax.adam(1e-3)
        >>> def loss(params):
        ...     pred = forward(params.optical, params.morphology_data)
        ...     return mse_loss(pred, experimental)
        >>> final_state = fit_loop(params, loss, optimizer, 1000)
    """
    state = create_fit_state(initial_params, optimizer)

    for i in range(num_steps):
        state, loss_val = fit_step(state, loss_fn, optimizer)
        if callback is not None:
            callback(i, state.params, loss_val)

    return state


@partial(jax.jit, static_argnames=['num_steps'])
def fit_loop_jit(
    initial_params: FitParams,
    loss_and_grad_fn: Callable[[FitParams], tuple[Array, FitParams]],
    opt_state,
    optimizer_update,
    num_steps: int,
) -> tuple[FitParams, Array]:
    """JIT-compiled fitting loop using lax.fori_loop.

    For maximum performance when running many iterations.

    Args:
        initial_params: Starting parameters.
        loss_and_grad_fn: Function returning (loss, grads).
        opt_state: Initial optimizer state.
        optimizer_update: Optimizer update function.
        num_steps: Number of steps (static).

    Returns:
        Tuple of (final_params, final_loss).
    """
    def body_fn(i, carry):
        params, opt_state, _ = carry
        loss_val, grads = loss_and_grad_fn(params)
        updates, new_opt_state = optimizer_update(grads, opt_state, params)
        new_params = _apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    init_carry = (initial_params, opt_state, jnp.array(0.0))
    final_params, _, final_loss = lax.fori_loop(0, num_steps, body_fn, init_carry)

    return final_params, final_loss
