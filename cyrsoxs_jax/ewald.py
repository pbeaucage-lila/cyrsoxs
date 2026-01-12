"""Ewald sphere projection for CyRSoXS-JAX.

This module implements the projection of 3D reciprocal space scattering
onto a 2D detector via the Ewald sphere construction. The Ewald sphere
represents the locus of points in reciprocal space that satisfy the
elastic scattering condition |k_f| = |k_i|.
"""

import jax.numpy as jnp
from jax import Array
from enum import IntEnum


class EwaldInterpolation(IntEnum):
    """Interpolation modes for Ewald sphere projection."""
    NEAREST_NEIGHBOR = 0
    TRILINEAR = 1


def compute_detector_q_grid(
    voxel: tuple[int, int, int],
    phys_size: float,
) -> tuple[Array, Array, Array, Array]:
    """Compute the 2D detector q-grid and grid spacing.

    Args:
        voxel: Tuple of (Nx, Ny, Nz) grid dimensions.
        phys_size: Physical size of the simulation box.

    Returns:
        Tuple of (qx_2d, qy_2d, dx, start) where qx_2d and qy_2d are
        2D meshgrid arrays for the detector, dx is (dx, dy, dz) spacing,
        and start is the q-space origin.
    """
    nx, ny, nz = voxel

    # Grid spacing
    dx = (2 * jnp.pi / phys_size) / (nx - 1)
    dy = (2 * jnp.pi / phys_size) / (ny - 1)
    dz = (2 * jnp.pi / phys_size) / (nz - 1)

    # Start position in q-space
    start = -jnp.pi / phys_size

    # 1D coordinates for detector
    qx_1d = start + jnp.arange(nx) * dx
    qy_1d = start + jnp.arange(ny) * dy

    # Create 2D meshgrid for detector plane
    qx_2d, qy_2d = jnp.meshgrid(qx_1d, qy_1d, indexing='ij')

    return qx_2d, qy_2d, (dx, dy, dz), start


def ewald_projection(
    scatter_3d: Array,
    k_magnitude: float,
    voxel: tuple[int, int, int],
    phys_size: float,
    k_vector: tuple[float, float, float],
    interpolation: EwaldInterpolation = EwaldInterpolation.TRILINEAR,
    enable_2d: bool = False,
) -> Array:
    """Project 3D scattering onto 2D detector via Ewald sphere.

    Maps each detector pixel to a point on the Ewald sphere in reciprocal
    space and interpolates the scatter3D value at that point.

    The Ewald sphere constraint is: |k_f|^2 = |k_i|^2 = k^2
    For a detector pixel at (qx, qy), we solve for qz:
        qz = -kz + sqrt(k^2 - (kx + qx)^2 - (ky + qy)^2)

    Args:
        scatter_3d: 3D scattering intensity array (Nx, Ny, Nz).
        k_magnitude: Magnitude of incident k-vector (2*pi/wavelength).
        voxel: Grid dimensions (Nx, Ny, Nz).
        phys_size: Physical size of simulation box in nm.
        k_vector: Normalized k-vector direction (kx, ky, kz).
        interpolation: Interpolation mode (NEAREST_NEIGHBOR or TRILINEAR).
        enable_2d: If True, use 2D morphology mode.

    Returns:
        2D projection array (Nx, Ny).
    """
    nx, ny, nz = voxel
    kx, ky, kz = k_vector

    # Get detector grid
    qx_2d, qy_2d, (dx, dy, dz), start = compute_detector_q_grid(voxel, phys_size)

    # Compute total q including k-vector contribution
    qx_total = k_magnitude * kx + qx_2d
    qy_total = k_magnitude * ky + qy_2d

    # Ewald sphere: k^2 = qx_total^2 + qy_total^2 + qz_total^2
    # Solve for qz: val = k^2 - qx_total^2 - qy_total^2
    k_sq = k_magnitude * k_magnitude
    val = k_sq - qx_total * qx_total - qy_total * qy_total

    # Points outside Ewald sphere (val < 0) are invalid
    valid = val >= 0

    # Compute qz on Ewald sphere (where valid)
    qz_ewald = -k_magnitude * kz + jnp.sqrt(jnp.maximum(val, 0))

    # Handle 2D mode
    if enable_2d:
        # In 2D mode, just use the XY slice at z=0
        projection = scatter_3d[:, :, 0]
        # Mask boundary pixels
        boundary_mask = (jnp.arange(nx)[:, None] == nx - 1) | (jnp.arange(ny)[None, :] == ny - 1)
        projection = jnp.where(boundary_mask, jnp.nan, projection)
        projection = jnp.where(valid, projection, jnp.nan)
        return projection

    # Convert qz to array index (floating point)
    z_idx_float = (qz_ewald - start) / dz

    # Boundary check: exclude last row/column and invalid z indices
    x_idx = jnp.arange(nx)[:, None]
    y_idx = jnp.arange(ny)[None, :]
    boundary_mask = (x_idx == nx - 1) | (y_idx == ny - 1)
    z_valid = (z_idx_float >= 0) & (z_idx_float < nz - 1)

    if interpolation == EwaldInterpolation.NEAREST_NEIGHBOR:
        # Round to nearest integer index
        z_idx = jnp.clip(jnp.round(z_idx_float).astype(jnp.int32), 0, nz - 1)
        projection = scatter_3d[x_idx, y_idx, z_idx]
    else:
        # Trilinear interpolation along z
        z_idx_low = jnp.clip(jnp.floor(z_idx_float).astype(jnp.int32), 0, nz - 2)
        z_idx_high = z_idx_low + 1

        # Interpolation weight
        z_frac = z_idx_float - z_idx_low

        # Get values at adjacent z planes
        val_low = scatter_3d[x_idx, y_idx, z_idx_low]
        val_high = scatter_3d[x_idx, y_idx, z_idx_high]

        # Linear interpolation
        projection = (1 - z_frac) * val_low + z_frac * val_high

    # Apply masks for invalid regions
    projection = jnp.where(valid & z_valid & ~boundary_mask, projection, jnp.nan)

    return projection


def ewald_projection_from_polarization(
    polarization_fft: Array,
    k_magnitude: float,
    voxel: tuple[int, int, int],
    phys_size: float,
    k_vector: tuple[float, float, float],
    interpolation: EwaldInterpolation = EwaldInterpolation.TRILINEAR,
    enable_2d: bool = False,
) -> Array:
    """Project polarization FFT onto 2D detector via Ewald sphere.

    This combines scatter3D computation with Ewald projection, computing
    the scattering intensity on-the-fly at the interpolated points.

    For trilinear interpolation, scatter3D is computed at both z planes
    and then interpolated. For nearest neighbor, scatter3D is computed
    only at the nearest voxel.

    Args:
        polarization_fft: FFT of polarization field (Nx, Ny, Nz, 3), complex.
        k_magnitude: Magnitude of incident k-vector (2*pi/wavelength).
        voxel: Grid dimensions (Nx, Ny, Nz).
        phys_size: Physical size of simulation box in nm.
        k_vector: Normalized k-vector direction (kx, ky, kz).
        interpolation: Interpolation mode (NEAREST_NEIGHBOR or TRILINEAR).
        enable_2d: If True, use 2D morphology mode.

    Returns:
        2D projection array (Nx, Ny).
    """
    from cyrsoxs_jax.scatter import scatter_3d

    # Compute full scatter3D and then project
    scatter = scatter_3d(
        polarization_fft,
        k_magnitude,
        voxel,
        phys_size,
        k_vector,
        enable_2d,
    )

    return ewald_projection(
        scatter,
        k_magnitude,
        voxel,
        phys_size,
        k_vector,
        interpolation,
        enable_2d,
    )
