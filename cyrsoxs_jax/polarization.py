"""Polarization computation for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array


def compute_polarization_field(
    morphology: Array,
    dielectric_tensors: Array,
    euler_angles: Array,
    incident_field: Array,
) -> Array:
    """Compute the polarization field P = (eps - 1) * E.

    Args:
        morphology: Volume fractions (N, N, N, num_materials).
        dielectric_tensors: Per-material dielectric tensors (num_materials, 3, 3).
        euler_angles: Euler angles (N, N, N, 3).
        incident_field: Incident electric field vector (3,).

    Returns:
        Polarization field (N, N, N, 3).
    """
    raise NotImplementedError("Polarization field computation not yet implemented")


def compute_effective_dielectric(
    volume_fractions: Array,
    dielectric_tensors: Array,
) -> Array:
    """Compute effective dielectric tensor via volume averaging.

    Args:
        volume_fractions: Material volume fractions (num_materials,).
        dielectric_tensors: Per-material dielectric tensors (num_materials, 3, 3).

    Returns:
        Effective dielectric tensor (3, 3).
    """
    return jnp.einsum("m,mij->ij", volume_fractions, dielectric_tensors)
