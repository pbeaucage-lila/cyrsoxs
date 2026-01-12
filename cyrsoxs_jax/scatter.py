"""Scatter3D computation for CyRSoXS-JAX.

This module implements the 3D scattering intensity computation from
the FFT of the polarization field. The physics involves projecting
the polarization onto the plane perpendicular to the scattering vector
and computing the squared magnitude.
"""

import jax.numpy as jnp
from jax import Array


def compute_q_grid_3d(
    voxel: tuple[int, int, int],
    phys_size: float,
    enable_2d: bool = False,
) -> tuple[Array, Array, Array]:
    """Compute the q-space grid for scatter3D.

    The q-grid is defined as q = -pi/phys_size + idx * dq where
    dq = (2*pi/phys_size) / (N-1) for each dimension.

    Args:
        voxel: Tuple of (Nx, Ny, Nz) grid dimensions.
        phys_size: Physical size of the simulation box.
        enable_2d: If True, set qz = 0 for 2D morphology.

    Returns:
        Tuple of (qx, qy, qz) 3D meshgrid arrays.
    """
    nx, ny, nz = voxel

    # Compute grid spacing for each dimension
    dx = (2 * jnp.pi / phys_size) / (nx - 1)
    dy = (2 * jnp.pi / phys_size) / (ny - 1)
    dz = 0.0 if enable_2d else (2 * jnp.pi / phys_size) / (nz - 1)

    # Compute q coordinates: q = -pi/phys_size + idx * dq
    start = -jnp.pi / phys_size
    qx_1d = start + jnp.arange(nx) * dx
    qy_1d = start + jnp.arange(ny) * dy
    if enable_2d:
        qz_1d = jnp.zeros(nz)
    else:
        qz_1d = start + jnp.arange(nz) * dz

    # Create 3D meshgrid with 'ij' indexing to match CUDA's X,Y,Z ordering
    qx, qy, qz = jnp.meshgrid(qx_1d, qy_1d, qz_1d, indexing='ij')

    return qx, qy, qz


def project_polarization_magnitude_squared(
    polarization: Array,
    q_vec: tuple[Array, Array, Array],
    k_magnitude: float,
) -> Array:
    """Compute |(k^2*I - q*q^T) * P|^2 for all voxels.

    This projects the polarization onto the plane perpendicular to
    the scattering vector q, scaled by k^2, and returns the squared
    magnitude.

    The computation follows the CUDA implementation:
    For each component i:
        temp_i = (k^2 - q_i^2) * P_i - q_i * sum_{j!=i}(q_j * P_j)
    Result = sum_i |temp_i|^2

    Args:
        polarization: Complex polarization FFT (Nx, Ny, Nz, 3).
        q_vec: Tuple of (qx, qy, qz) 3D arrays for scattering vectors.
        k_magnitude: Magnitude of the incident k-vector.

    Returns:
        Scattering intensity (Nx, Ny, Nz).
    """
    qx, qy, qz = q_vec
    k_sq = k_magnitude * k_magnitude

    # Extract polarization components
    px = polarization[..., 0]  # Complex (Nx, Ny, Nz)
    py = polarization[..., 1]
    pz = polarization[..., 2]

    # Compute (k^2*I - q*q^T) * P for each row
    # Row 0: (k^2 - qx^2)*px - qx*qy*py - qx*qz*pz
    temp_x = (k_sq - qx * qx) * px - qx * (qy * py + qz * pz)

    # Row 1: -qx*qy*px + (k^2 - qy^2)*py - qy*qz*pz
    temp_y = -qx * qy * px + (k_sq - qy * qy) * py - qy * qz * pz

    # Row 2: -qx*qz*px - qy*qz*py + (k^2 - qz^2)*pz
    temp_z = -qx * qz * px - qy * qz * py + (k_sq - qz * qz) * pz

    # Sum of squared magnitudes: |temp_x|^2 + |temp_y|^2 + |temp_z|^2
    result = (
        temp_x.real ** 2 + temp_x.imag ** 2 +
        temp_y.real ** 2 + temp_y.imag ** 2 +
        temp_z.real ** 2 + temp_z.imag ** 2
    )

    return result


def scatter_3d(
    polarization_fft: Array,
    k_magnitude: float,
    voxel: tuple[int, int, int],
    phys_size: float,
    k_vector: tuple[float, float, float],
    enable_2d: bool = False,
) -> Array:
    """Compute 3D scattering intensity from polarization FFT.

    Implements the CyRSoXS scatter3D computation:
    1. Compute q-grid from voxel dimensions and physical size
    2. Add k*k_vector to get full scattering vector at each point
    3. Project polarization perpendicular to q and compute |P_perp|^2

    Args:
        polarization_fft: FFT of polarization field (Nx, Ny, Nz, 3), complex.
        k_magnitude: Magnitude of incident k-vector (2*pi/wavelength).
        voxel: Grid dimensions (Nx, Ny, Nz).
        phys_size: Physical size of simulation box in nm.
        k_vector: Normalized k-vector direction (kx, ky, kz), |k_vector|=1.
        enable_2d: If True, use 2D morphology mode (qz=0).

    Returns:
        3D scattering intensity array (Nx, Ny, Nz).
    """
    # Compute base q-grid
    qx, qy, qz = compute_q_grid_3d(voxel, phys_size, enable_2d)

    # Add k*k_vector contribution to q
    kx, ky, kz = k_vector
    qx_total = k_magnitude * kx + qx
    qy_total = k_magnitude * ky + qy
    qz_total = k_magnitude * kz + qz

    # Compute projected polarization magnitude squared
    scatter = project_polarization_magnitude_squared(
        polarization_fft,
        (qx_total, qy_total, qz_total),
        k_magnitude,
    )

    return scatter


def scatter_3d_batched(
    polarization_fft: Array,
    k_magnitude: float,
    voxel: tuple[int, int, int],
    phys_size: float,
    k_vectors: Array,
    enable_2d: bool = False,
) -> Array:
    """Compute 3D scattering for multiple k-vector orientations.

    This is useful for computing scattering at multiple energies or
    rotation angles efficiently.

    Args:
        polarization_fft: FFT of polarization field (Nx, Ny, Nz, 3), complex.
        k_magnitude: Magnitude of incident k-vector.
        voxel: Grid dimensions (Nx, Ny, Nz).
        phys_size: Physical size of simulation box in nm.
        k_vectors: Array of normalized k-vectors (N_angles, 3).
        enable_2d: If True, use 2D morphology mode.

    Returns:
        3D scattering intensity for each angle (N_angles, Nx, Ny, Nz).
    """
    from jax import vmap

    def compute_single(k_vec):
        return scatter_3d(
            polarization_fft,
            k_magnitude,
            voxel,
            phys_size,
            (k_vec[0], k_vec[1], k_vec[2]),
            enable_2d,
        )

    return vmap(compute_single)(k_vectors)
