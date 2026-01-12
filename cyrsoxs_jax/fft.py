"""FFT pipeline for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array


def forward_fft_3d(field: Array) -> Array:
    """Compute 3D FFT of a field.

    Args:
        field: Real-space field (N, N, N) or (N, N, N, 3).

    Returns:
        Fourier-space field with same shape.
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
        field: Fourier-space field (N, N, N) or (N, N, N, 3).

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
        field: Fourier-space field.

    Returns:
        Shifted field.
    """
    if field.ndim == 3:
        return jnp.fft.fftshift(field)
    elif field.ndim == 4:
        return jnp.fft.fftshift(field, axes=(0, 1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {field.ndim}D")


def compute_q_grid(n: int, physical_size: float, wavelength: float) -> tuple[Array, Array, Array]:
    """Compute q-space grid.

    Args:
        n: Grid size.
        physical_size: Physical size per voxel in nm.
        wavelength: X-ray wavelength in nm.

    Returns:
        Tuple of (qx, qy, qz) 1D arrays.
    """
    k = 2 * jnp.pi / wavelength
    dq = 2 * jnp.pi / (n * physical_size)
    q = jnp.fft.fftfreq(n, d=1.0) * n * dq
    return q, q, q
