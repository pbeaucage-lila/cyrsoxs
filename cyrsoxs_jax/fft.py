"""FFT pipeline for CyRSoXS-JAX.

This module implements the 3D FFT pipeline matching the CUDA implementation:
1. 3D FFT of polarization (X, Y, Z components)
2. Replace DC component with neighbor average
3. FFT shift to center zero frequency

The DC component replacement is critical for preventing artifacts in the
scattered intensity near q=0 that arise from the interpolation scheme.
"""

import jax.numpy as jnp
from jax import Array


def forward_fft_3d(field: Array) -> Array:
    """Compute 3D FFT of a field.

    Args:
        field: Real-space field of shape (Nz, Ny, Nx) or (Nz, Ny, Nx, 3)
               for vector fields with 3 components.

    Returns:
        Fourier-space field with same shape, complex dtype.
    """
    if field.ndim == 3:
        return jnp.fft.fftn(field)
    elif field.ndim == 4:
        return jnp.fft.fftn(field, axes=(0, 1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {field.ndim}D")


def inverse_fft_3d(field: Array) -> Array:
    """Compute inverse 3D FFT of a field.

    Args:
        field: Fourier-space field of shape (Nz, Ny, Nx) or (Nz, Ny, Nx, 3).

    Returns:
        Real-space field with same shape.
    """
    if field.ndim == 3:
        return jnp.fft.ifftn(field)
    elif field.ndim == 4:
        return jnp.fft.ifftn(field, axes=(0, 1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {field.ndim}D")


def fftshift_3d(field: Array) -> Array:
    """Shift zero-frequency component to center.

    Args:
        field: Fourier-space field of shape (Nz, Ny, Nx) or (Nz, Ny, Nx, 3).

    Returns:
        Shifted field with same shape.
    """
    if field.ndim == 3:
        return jnp.fft.fftshift(field)
    elif field.ndim == 4:
        return jnp.fft.fftshift(field, axes=(0, 1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {field.ndim}D")


def replace_dc_component(field: Array) -> Array:
    """Replace DC component with average of its 6 face-adjacent neighbors.

    The DC component at index [0, 0, 0] is replaced with the average of its
    6 face-adjacent neighbors using periodic boundary conditions. This prevents
    artifacts in the scattered intensity near q=0 that arise from interpolation.

    The 6 neighbors are:
        - (1, 0, 0)         -> index 1 in x
        - (Nx-1, 0, 0)      -> wrap-around in x (periodic)
        - (0, 1, 0)         -> index 1 in y
        - (0, Ny-1, 0)      -> wrap-around in y (periodic)
        - (0, 0, 1)         -> index 1 in z
        - (0, 0, Nz-1)      -> wrap-around in z (periodic)

    Args:
        field: Complex Fourier-space field of shape (Nz, Ny, Nx) or
               (Nz, Ny, Nx, 3) for vector fields.

    Returns:
        Field with DC component replaced by neighbor average.
    """
    if field.ndim == 3:
        return _replace_dc_scalar(field)
    elif field.ndim == 4:
        # Process each component separately
        px = _replace_dc_scalar(field[..., 0])
        py = _replace_dc_scalar(field[..., 1])
        pz = _replace_dc_scalar(field[..., 2])
        return jnp.stack([px, py, pz], axis=-1)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {field.ndim}D")


def _replace_dc_scalar(field: Array) -> Array:
    """Replace DC component for a scalar 3D field.

    Args:
        field: Complex field of shape (Nz, Ny, Nx).

    Returns:
        Field with DC component [0,0,0] replaced by neighbor average.
    """
    nz, ny, nx = field.shape

    # Get the 6 face-adjacent neighbors (with periodic boundary)
    # Neighbor at (0, 0, 1) - positive x
    n1 = field[0, 0, 1]
    # Neighbor at (0, 0, nx-1) - negative x (wrap-around)
    n2 = field[0, 0, nx - 1]
    # Neighbor at (0, 1, 0) - positive y
    n3 = field[0, 1, 0]
    # Neighbor at (0, ny-1, 0) - negative y (wrap-around)
    n4 = field[0, ny - 1, 0]
    # Neighbor at (1, 0, 0) - positive z
    n5 = field[1, 0, 0]
    # Neighbor at (nz-1, 0, 0) - negative z (wrap-around)
    n6 = field[nz - 1, 0, 0]

    # Compute average of 6 neighbors
    avg = (n1 + n2 + n3 + n4 + n5 + n6) / 6.0

    # Replace DC component at [0, 0, 0]
    return field.at[0, 0, 0].set(avg)


def fft_pipeline(
    polarization_x: Array,
    polarization_y: Array,
    polarization_z: Array,
) -> tuple[Array, Array, Array]:
    """Execute the complete FFT pipeline for polarization fields.

    This implements the CyRSoXS FFT pipeline:
    1. Compute 3D FFT of each polarization component
    2. Replace DC component with neighbor average (prevents q=0 artifacts)
    3. Apply FFT shift to center zero frequency

    Args:
        polarization_x: X component of polarization field (Nz, Ny, Nx), complex.
        polarization_y: Y component of polarization field (Nz, Ny, Nx), complex.
        polarization_z: Z component of polarization field (Nz, Ny, Nx), complex.

    Returns:
        Tuple of (fft_x, fft_y, fft_z) - Fourier-transformed, DC-replaced,
        and shifted polarization components.
    """
    # Step 1: Forward FFT
    fft_x = forward_fft_3d(polarization_x)
    fft_y = forward_fft_3d(polarization_y)
    fft_z = forward_fft_3d(polarization_z)

    # Step 2: Replace DC component with neighbor average
    fft_x = replace_dc_component(fft_x)
    fft_y = replace_dc_component(fft_y)
    fft_z = replace_dc_component(fft_z)

    # Step 3: FFT shift to center zero frequency
    fft_x = fftshift_3d(fft_x)
    fft_y = fftshift_3d(fft_y)
    fft_z = fftshift_3d(fft_z)

    return fft_x, fft_y, fft_z


def fft_pipeline_vectorized(polarization: Array) -> Array:
    """Execute the FFT pipeline on a vectorized polarization field.

    Alternative interface that processes all 3 components together.

    Args:
        polarization: Polarization field of shape (Nz, Ny, Nx, 3), complex,
                      where the last axis contains (pX, pY, pZ).

    Returns:
        Fourier-transformed, DC-replaced, and shifted polarization field
        of shape (Nz, Ny, Nx, 3).
    """
    # Step 1: Forward FFT on all components
    fft_field = forward_fft_3d(polarization)

    # Step 2: Replace DC component in each polarization component
    fft_field = replace_dc_component(fft_field)

    # Step 3: FFT shift to center zero frequency
    fft_field = fftshift_3d(fft_field)

    return fft_field


def compute_q_grid(
    n: int | tuple[int, int, int],
    physical_size: float,
    wavelength: float,
) -> tuple[Array, Array, Array]:
    """Compute q-space grid coordinates.

    Args:
        n: Grid size (single int for cubic, or (Nz, Ny, Nx) tuple).
        physical_size: Physical size per voxel in nm.
        wavelength: X-ray wavelength in nm.

    Returns:
        Tuple of (qz, qy, qx) 3D meshgrid arrays, with zero frequency centered.
    """
    if isinstance(n, int):
        nz, ny, nx = n, n, n
    else:
        nz, ny, nx = n

    # Compute q-space sampling
    dq_x = 2 * jnp.pi / (nx * physical_size)
    dq_y = 2 * jnp.pi / (ny * physical_size)
    dq_z = 2 * jnp.pi / (nz * physical_size)

    # Generate frequency arrays and apply fftshift to center zero
    qx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=1.0)) * nx * dq_x
    qy = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=1.0)) * ny * dq_y
    qz = jnp.fft.fftshift(jnp.fft.fftfreq(nz, d=1.0)) * nz * dq_z

    return qz, qy, qx
