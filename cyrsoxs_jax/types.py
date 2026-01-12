"""Core data types for CyRSoXS-JAX."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SimulationConfig(NamedTuple):
    """Configuration for a CyRSoXS simulation."""

    energies: Array  # eV
    num_threads: int = 1
    physical_size: float = 1.0  # nm per voxel
    euler_convention: str = "ZYZ"


class Morphology(NamedTuple):
    """Morphology data structure."""

    voxel_data: Array  # (N, N, N, num_materials) volume fractions
    euler_angles: Array  # (N, N, N, 3) Euler angles
    physical_size: float  # nm per voxel


class OpticalConstants(NamedTuple):
    """Optical constants for a material."""

    delta_para: Array  # parallel component
    delta_perp: Array  # perpendicular component
    beta_para: Array  # parallel absorption
    beta_perp: Array  # perpendicular absorption
    energies: Array  # corresponding energies in eV


class ScatteringResult(NamedTuple):
    """Result of a scattering simulation."""

    intensity: Array  # 2D detector image
    q_x: Array  # q values in x direction
    q_y: Array  # q values in y direction
    energy: float  # photon energy in eV
