"""Optical constants and material handling for CyRSoXS-JAX."""

from typing import Dict

import jax.numpy as jnp
from jax import Array

from cyrsoxs_jax.types import OpticalConstants


def load_optical_constants(filepath: str) -> OpticalConstants:
    """Load optical constants from a file.

    Args:
        filepath: Path to optical constants file.

    Returns:
        OpticalConstants with delta and beta values.
    """
    raise NotImplementedError("Optical constants loading not yet implemented")


def interpolate_optical_constants(
    constants: OpticalConstants,
    energy: float,
) -> tuple[complex, complex]:
    """Interpolate optical constants to a specific energy.

    Args:
        constants: OpticalConstants to interpolate.
        energy: Target energy in eV.

    Returns:
        Tuple of (n_para, n_perp) complex refractive indices.
    """
    raise NotImplementedError("Optical constants interpolation not yet implemented")


def compute_dielectric_tensor(
    delta_para: float,
    delta_perp: float,
    beta_para: float,
    beta_perp: float,
) -> Array:
    """Compute the dielectric tensor from optical constants.

    Args:
        delta_para: Parallel delta component.
        delta_perp: Perpendicular delta component.
        beta_para: Parallel beta component.
        beta_perp: Perpendicular beta component.

    Returns:
        3x3 diagonal dielectric tensor.
    """
    n_para = 1 - delta_para + 1j * beta_para
    n_perp = 1 - delta_perp + 1j * beta_perp

    eps_para = n_para**2
    eps_perp = n_perp**2

    return jnp.diag(jnp.array([eps_perp, eps_perp, eps_para]))
