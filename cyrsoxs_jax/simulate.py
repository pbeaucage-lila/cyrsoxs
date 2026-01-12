"""Main simulation orchestration for CyRSoXS-JAX."""

from typing import List, Optional

import jax.numpy as jnp
from jax import Array

from cyrsoxs_jax.types import Morphology, OpticalConstants, ScatteringResult, SimulationConfig


def simulate(
    morphology: Morphology,
    optical_constants: List[OpticalConstants],
    config: SimulationConfig,
    polarization: str = "horizontal",
) -> List[ScatteringResult]:
    """Run a CyRSoXS simulation.

    This is the main entry point for running scattering simulations.

    Args:
        morphology: Morphology data structure with voxel data and orientations.
        optical_constants: List of optical constants for each material.
        config: Simulation configuration.
        polarization: Incident beam polarization ("horizontal" or "vertical").

    Returns:
        List of ScatteringResult, one per energy.
    """
    raise NotImplementedError("Full simulation pipeline not yet implemented")


def simulate_single_energy(
    morphology: Morphology,
    optical_constants: List[OpticalConstants],
    energy: float,
    polarization: str = "horizontal",
) -> ScatteringResult:
    """Run simulation at a single energy.

    Args:
        morphology: Morphology data structure.
        optical_constants: List of optical constants for each material.
        energy: Photon energy in eV.
        polarization: Incident beam polarization.

    Returns:
        ScatteringResult for this energy.
    """
    raise NotImplementedError("Single energy simulation not yet implemented")


def compute_form_factor(
    morphology: Morphology,
    q_grid: tuple[Array, Array, Array],
) -> Array:
    """Compute the form factor from morphology.

    Args:
        morphology: Morphology data structure.
        q_grid: Q-space grid coordinates.

    Returns:
        3D form factor array.
    """
    raise NotImplementedError("Form factor computation not yet implemented")


def energy_to_wavelength(energy_ev: float) -> float:
    """Convert photon energy to wavelength.

    Args:
        energy_ev: Photon energy in eV.

    Returns:
        Wavelength in nm.
    """
    # E = hc/lambda, hc = 1239.84 eV*nm
    return 1239.84 / energy_ev
