"""Main simulation orchestration for CyRSoXS-JAX."""

from typing import List
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from cyrsoxs_jax.types import (
    Morphology, OpticalConstants, ScatteringResult, SimulationConfig,
    MorphologyType, ReferenceFrame,
)
from cyrsoxs_jax.polarization import (
    compute_polarization_field,
    compute_polarization_vector_morphology,
)
from cyrsoxs_jax.fft import forward_fft_3d, fftshift_3d, compute_q_grid
from cyrsoxs_jax.rotation import (
    compute_base_configuration,
    compute_full_rotation,
    rodrigues_rotation_matrix,
    rotation_matrix_k,
)

# Physical constant: hc in eV*nm
HC_EV_NM = 1239.84197


def energy_to_wavelength(energy_ev: float) -> float:
    """Convert photon energy to wavelength in nm."""
    return HC_EV_NM / energy_ev


def _replace_dc_component(field: Array) -> Array:
    """Replace DC component [0,0,0] with average of 6 face-adjacent neighbors."""
    # Get neighbor values (with periodic boundaries)
    nz, ny, nx = field.shape[:3]
    neighbors = jnp.stack([
        field[1, 0, 0], field[nz-1, 0, 0],
        field[0, 1, 0], field[0, ny-1, 0],
        field[0, 0, 1], field[0, 0, nx-1],
    ])
    avg = jnp.mean(neighbors, axis=0)
    return field.at[0, 0, 0].set(avg)


def _compute_scatter_intensity(p_fft: Array, k_vec: Array) -> Array:
    """Compute scattered intensity from FFT of polarization.

    Args:
        p_fft: FFT of polarization field, shape (Nz, Ny, Nx, 3).
        k_vec: Incident k-vector direction, shape (3,).

    Returns:
        Scattered intensity |P - (P·k)k|², shape (Nz, Ny, Nx).
    """
    # Project out component along k: P_perp = P - (P·k)k
    p_dot_k = jnp.sum(p_fft * k_vec, axis=-1, keepdims=True)
    p_perp = p_fft - p_dot_k * k_vec
    return jnp.sum(jnp.abs(p_perp) ** 2, axis=-1)


def _ewald_projection_nearest(scatter_3d: Array, k_mag: float,
                               phys_size: float, k_vec: Array) -> Array:
    """Project 3D scattering onto Ewald sphere using nearest neighbor.

    For transmission geometry with k along z, the Ewald sphere intersection
    maps qx, qy detector coordinates to qz = k - sqrt(k² - qx² - qy²).
    """
    nz, ny, nx = scatter_3d.shape

    # Q-space grid
    dq = 2 * jnp.pi / (nx * phys_size)
    qx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=1.0) * nx * dq)
    qy = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=1.0) * ny * dq)
    qx_2d, qy_2d = jnp.meshgrid(qx, qy)

    # Ewald sphere: qz = k - sqrt(k² - qx² - qy²)
    q_perp_sq = qx_2d**2 + qy_2d**2
    valid = q_perp_sq < k_mag**2
    qz_ewald = jnp.where(valid, k_mag - jnp.sqrt(k_mag**2 - q_perp_sq), 0.0)

    # Convert to indices
    iz = jnp.round(qz_ewald / dq + nz // 2).astype(jnp.int32)
    iz = jnp.clip(iz, 0, nz - 1)

    # Extract values along Ewald sphere
    iy = jnp.arange(ny)[:, None] * jnp.ones(nx, dtype=jnp.int32)
    ix = jnp.ones(ny, dtype=jnp.int32)[:, None] * jnp.arange(nx)

    projection = scatter_3d[iz.astype(jnp.int32),
                           iy.astype(jnp.int32),
                           ix.astype(jnp.int32)]
    return jnp.where(valid, projection, 0.0)


def _rotate_image_bilinear(image: Array, angle: float) -> Array:
    """Rotate 2D image by angle (radians) using bilinear interpolation."""
    ny, nx = image.shape
    cy, cx = ny / 2.0, nx / 2.0

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)

    # Output coordinates
    y_out, x_out = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')

    # Inverse transform to find source coordinates
    y_centered = y_out - cy
    x_centered = x_out - cx
    y_src = cos_a * y_centered + sin_a * x_centered + cy
    x_src = -sin_a * y_centered + cos_a * x_centered + cx

    # Bilinear interpolation
    y0 = jnp.floor(y_src).astype(jnp.int32)
    x0 = jnp.floor(x_src).astype(jnp.int32)
    y1, x1 = y0 + 1, x0 + 1

    # Clamp to valid range
    y0c = jnp.clip(y0, 0, ny - 1)
    y1c = jnp.clip(y1, 0, ny - 1)
    x0c = jnp.clip(x0, 0, nx - 1)
    x1c = jnp.clip(x1, 0, nx - 1)

    # Weights
    wy = y_src - y0
    wx = x_src - x0

    # Gather and interpolate
    v00 = image[y0c, x0c]
    v01 = image[y0c, x1c]
    v10 = image[y1c, x0c]
    v11 = image[y1c, x1c]

    result = (v00 * (1 - wy) * (1 - wx) +
              v01 * (1 - wy) * wx +
              v10 * wy * (1 - wx) +
              v11 * wy * wx)

    # Mask out-of-bounds
    valid = (y_src >= 0) & (y_src < ny) & (x_src >= 0) & (x_src < nx)
    return jnp.where(valid, result, 0.0)


def simulate_single_angle(
    morphology_data: Array,
    n_para: Array,
    n_perp: Array,
    rotation_matrix: Array,
    incident_field: Array,
    k_vec: Array,
    k_mag: float,
    phys_size: float,
    morphology_type: MorphologyType,
) -> Array:
    """Simulate scattering at a single rotation angle.

    Returns:
        2D projection image.
    """
    # Compute polarization
    if morphology_type == MorphologyType.VECTOR_MORPHOLOGY:
        polarization = compute_polarization_vector_morphology(
            morphology_data, n_para, n_perp, incident_field, rotation_matrix
        )
    else:
        polarization = compute_polarization_field(
            morphology_data, n_para, n_perp, None, incident_field, rotation_matrix
        )

    # FFT each component
    p_fft = forward_fft_3d(polarization)

    # Replace DC component
    p_fft = _replace_dc_component(p_fft)

    # FFT shift
    p_fft = fftshift_3d(p_fft)

    # Compute scattered intensity
    scatter_3d = _compute_scatter_intensity(p_fft, k_vec)

    # Ewald sphere projection
    projection = _ewald_projection_nearest(scatter_3d, k_mag, phys_size, k_vec)

    return projection


def simulate_single_k_vector(
    morphology: Morphology,
    n_para: Array,
    n_perp: Array,
    k_vec: Array,
    wavelength: float,
    config: SimulationConfig,
) -> Array:
    """Simulate scattering for a single k-vector, averaging over rotation angles.

    Returns:
        Averaged 2D projection.
    """
    k_mag = 2 * jnp.pi / wavelength
    incident_field = jnp.array([1.0, 0.0, 0.0])  # Horizontal polarization

    # Get base configuration for this k-vector
    base_config = compute_base_configuration(k_vec)

    # Transpose morphology data for polarization functions
    # From [num_materials, Nz, Ny, Nx, 4] to [Nz, Ny, Nx, num_materials, 4]
    morphology_data = jnp.transpose(morphology.voxel_data.data, (1, 2, 3, 0, 4))

    angles_deg = config.angles
    num_angles = len(angles_deg)

    def process_angle(carry, angle_deg):
        projection_sum = carry
        angle_rad = (base_config.base_rot_angle + angle_deg) * jnp.pi / 180.0

        # Compute rotation matrix for this angle
        rotation_matrix = compute_full_rotation(k_vec, base_config,
                                                 angle_deg * jnp.pi / 180.0)

        # Simulate at this angle
        projection = simulate_single_angle(
            morphology_data, n_para, n_perp, rotation_matrix,
            incident_field, k_vec, k_mag, morphology.phys_size,
            morphology.morphology_type,
        )

        # Rotate projection by E-field angle
        projection_rotated = _rotate_image_bilinear(projection, angle_rad)

        return projection_sum + projection_rotated, None

    # Initialize accumulator
    nz, ny, nx = morphology.shape
    init_sum = jnp.zeros((ny, nx))

    # Scan over angles
    projection_sum, _ = lax.scan(process_angle, init_sum, angles_deg)

    # Average
    return projection_sum / num_angles


def simulate_single_energy(
    morphology: Morphology,
    n_para: Array,
    n_perp: Array,
    energy: float,
    config: SimulationConfig,
) -> List[ScatteringResult]:
    """Run simulation at a single energy for all k-vectors.

    Args:
        morphology: Morphology data structure.
        n_para: Parallel refractive indices, shape [num_materials].
        n_perp: Perpendicular refractive indices, shape [num_materials].
        energy: Photon energy in eV.
        config: Simulation configuration.

    Returns:
        List of ScatteringResult, one per k-vector.
    """
    wavelength = energy_to_wavelength(energy)
    results = []

    for i in range(len(config.k_vectors)):
        k_vec = config.k_vectors[i]
        projection = simulate_single_k_vector(
            morphology, n_para, n_perp, k_vec, wavelength, config
        )
        results.append(ScatteringResult(
            projection=projection,
            energy=energy,
            k_vector=k_vec,
            wavelength=wavelength,
        ))

    return results


def simulate(
    morphology: Morphology,
    material_data,  # MaterialData
    config: SimulationConfig,
) -> List[List[ScatteringResult]]:
    """Run a full CyRSoXS simulation.

    This is the main entry point for running scattering simulations.

    Args:
        morphology: Morphology data structure with voxel data and orientations.
        material_data: MaterialData with optical constants for all materials.
        config: Simulation configuration.

    Returns:
        Nested list of ScatteringResult: [energy][k_vector].
    """
    all_results = []

    for energy_idx, energy in enumerate(config.energies):
        # Get optical constants at this energy
        n_para, n_perp = material_data.get_at_energy(energy_idx)

        # Simulate at this energy
        results = simulate_single_energy(
            morphology, n_para, n_perp, float(energy), config
        )
        all_results.append(results)

    return all_results


# JIT-compiled version for performance
@partial(jax.jit, static_argnames=['morphology_type'])
def _simulate_single_angle_jit(
    morphology_data: Array,
    n_para: Array,
    n_perp: Array,
    rotation_matrix: Array,
    incident_field: Array,
    k_vec: Array,
    k_mag: float,
    phys_size: float,
    morphology_type: int,
) -> Array:
    """JIT-compiled single angle simulation."""
    mt = MorphologyType(morphology_type)
    return simulate_single_angle(
        morphology_data, n_para, n_perp, rotation_matrix,
        incident_field, k_vec, k_mag, phys_size, mt
    )
