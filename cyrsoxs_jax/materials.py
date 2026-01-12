"""Optical constants and material handling for CyRSoXS-JAX.

This module provides functions for loading, interpolating, and manipulating
optical constants for X-ray scattering simulations. All operations are
JAX-compatible and differentiable for use in fitting workflows.

Supported file formats:
    - Raw format: Tab-separated columns (e.g., PEOlig2018.txt)
    - Processed format: Block-structured EnergyDataN format (e.g., Material0.txt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cyrsoxs_jax.types import Material, MaterialData


@dataclass
class OpticalConstantsRaw:
    """Raw optical constants data for a single material.

    Stores the energy-dependent optical constants before conversion
    to complex refractive index.

    Attributes:
        energies: Array of energy values in eV.
        delta_para: Parallel delta component at each energy.
        beta_para: Parallel beta component at each energy.
        delta_perp: Perpendicular delta component at each energy.
        beta_perp: Perpendicular beta component at each energy.
    """
    energies: Array
    delta_para: Array
    beta_para: Array
    delta_perp: Array
    beta_perp: Array

    def to_material_data(self) -> MaterialData:
        """Convert to MaterialData with complex refractive indices."""
        return MaterialData.from_optical_constants(
            delta_para=self.delta_para[:, jnp.newaxis],
            beta_para=self.beta_para[:, jnp.newaxis],
            delta_perp=self.delta_perp[:, jnp.newaxis],
            beta_perp=self.beta_perp[:, jnp.newaxis],
            energies=self.energies,
        )


def load_raw_optical_constants(
    filepath: str | Path,
    energy_col: int = 6,
    beta_para_col: int = 0,
    beta_perp_col: int = 1,
    delta_para_col: int = 2,
    delta_perp_col: int = 3,
    skip_header: int = 1,
) -> OpticalConstantsRaw:
    """Load optical constants from raw tab-separated format.

    This format is used for source optical constants files like PEOlig2018.txt.
    The file contains columns for beta and delta values at various energies.

    Args:
        filepath: Path to the optical constants file.
        energy_col: Column index for energy values (0-based).
        beta_para_col: Column index for parallel beta.
        beta_perp_col: Column index for perpendicular beta.
        delta_para_col: Column index for parallel delta.
        delta_perp_col: Column index for perpendicular delta.
        skip_header: Number of header lines to skip.

    Returns:
        OpticalConstantsRaw with loaded data.
    """
    data = np.loadtxt(filepath, skiprows=skip_header)

    # Sort by energy
    sort_idx = np.argsort(data[:, energy_col])
    data = data[sort_idx]

    # Remove duplicate energies (keep first occurrence)
    _, unique_idx = np.unique(data[:, energy_col], return_index=True)
    data = data[unique_idx]

    return OpticalConstantsRaw(
        energies=jnp.array(data[:, energy_col]),
        delta_para=jnp.array(data[:, delta_para_col]),
        beta_para=jnp.array(data[:, beta_para_col]),
        delta_perp=jnp.array(data[:, delta_perp_col]),
        beta_perp=jnp.array(data[:, beta_perp_col]),
    )


def load_processed_optical_constants(filepath: str | Path) -> OpticalConstantsRaw:
    """Load optical constants from processed block format.

    This format is used for pre-processed material files like Material0.txt,
    which contain EnergyDataN blocks with Energy, BetaPara, BetaPerp,
    DeltaPara, and DeltaPerp values.

    Args:
        filepath: Path to the processed material file.

    Returns:
        OpticalConstantsRaw with loaded data.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse blocks
    pattern = r'EnergyData\d+:\s*\{([^}]+)\}'
    blocks = re.findall(pattern, content)

    energies = []
    delta_para = []
    beta_para = []
    delta_perp = []
    beta_perp = []

    for block in blocks:
        values = {}
        for line in block.strip().split('\n'):
            line = line.strip()
            if '=' in line:
                key, val = line.split('=')
                key = key.strip()
                val = val.strip().rstrip(';')
                values[key] = float(val)

        energies.append(values.get('Energy', 0.0))
        beta_para.append(values.get('BetaPara', 0.0))
        beta_perp.append(values.get('BetaPerp', 0.0))
        delta_para.append(values.get('DeltaPara', 0.0))
        delta_perp.append(values.get('DeltaPerp', 0.0))

    # Sort by energy
    indices = np.argsort(energies)

    return OpticalConstantsRaw(
        energies=jnp.array([energies[i] for i in indices]),
        delta_para=jnp.array([delta_para[i] for i in indices]),
        beta_para=jnp.array([beta_para[i] for i in indices]),
        delta_perp=jnp.array([delta_perp[i] for i in indices]),
        beta_perp=jnp.array([beta_perp[i] for i in indices]),
    )


def interpolate_optical_constants(
    constants: OpticalConstantsRaw,
    target_energies: Array,
) -> OpticalConstantsRaw:
    """Interpolate optical constants to target energies.

    Uses linear interpolation, which is differentiable with JAX.

    Args:
        constants: Source optical constants data.
        target_energies: Array of target energy values in eV.

    Returns:
        OpticalConstantsRaw interpolated to target energies.
    """
    delta_para = jnp.interp(target_energies, constants.energies, constants.delta_para)
    beta_para = jnp.interp(target_energies, constants.energies, constants.beta_para)
    delta_perp = jnp.interp(target_energies, constants.energies, constants.delta_perp)
    beta_perp = jnp.interp(target_energies, constants.energies, constants.beta_perp)

    return OpticalConstantsRaw(
        energies=target_energies,
        delta_para=delta_para,
        beta_para=beta_para,
        delta_perp=delta_perp,
        beta_perp=beta_perp,
    )


def combine_materials(
    materials: Sequence[OpticalConstantsRaw],
    target_energies: Array | None = None,
) -> MaterialData:
    """Combine multiple materials into a single MaterialData.

    If target_energies is provided, all materials are interpolated to those
    energies. Otherwise, the energies from the first material are used.

    Args:
        materials: Sequence of OpticalConstantsRaw for each material.
        target_energies: Optional array of target energies.

    Returns:
        MaterialData with all materials combined.
    """
    if len(materials) == 0:
        raise ValueError("At least one material is required")

    if target_energies is None:
        target_energies = materials[0].energies

    # Interpolate all materials to target energies
    interpolated = [interpolate_optical_constants(m, target_energies) for m in materials]

    # Stack into arrays [num_energies, num_materials]
    delta_para = jnp.stack([m.delta_para for m in interpolated], axis=1)
    beta_para = jnp.stack([m.beta_para for m in interpolated], axis=1)
    delta_perp = jnp.stack([m.delta_perp for m in interpolated], axis=1)
    beta_perp = jnp.stack([m.beta_perp for m in interpolated], axis=1)

    return MaterialData.from_optical_constants(
        delta_para=delta_para,
        beta_para=beta_para,
        delta_perp=delta_perp,
        beta_perp=beta_perp,
        energies=target_energies,
    )


def create_vacuum_material(energies: Array) -> OpticalConstantsRaw:
    """Create optical constants for vacuum (all zeros).

    Args:
        energies: Array of energy values.

    Returns:
        OpticalConstantsRaw with zero optical constants.
    """
    zeros = jnp.zeros_like(energies)
    return OpticalConstantsRaw(
        energies=energies,
        delta_para=zeros,
        beta_para=zeros,
        delta_perp=zeros,
        beta_perp=zeros,
    )


def compute_dielectric_tensor(
    npara: complex | Array,
    nperp: complex | Array,
) -> Array:
    """Compute the dielectric tensor from complex refractive indices.

    The dielectric tensor for a uniaxial material is diagonal with
    eps_perp along x and y, and eps_para along z (the optic axis).

    Args:
        npara: Parallel (extraordinary) refractive index.
        nperp: Perpendicular (ordinary) refractive index.

    Returns:
        3x3 diagonal dielectric tensor.
    """
    eps_para = npara ** 2
    eps_perp = nperp ** 2

    return jnp.diag(jnp.array([eps_perp, eps_perp, eps_para]))


def compute_dielectric_tensor_from_delta_beta(
    delta_para: float | Array,
    beta_para: float | Array,
    delta_perp: float | Array,
    beta_perp: float | Array,
) -> Array:
    """Compute dielectric tensor from optical constants.

    Args:
        delta_para: Parallel delta component.
        beta_para: Parallel beta component.
        delta_perp: Perpendicular delta component.
        beta_perp: Perpendicular beta component.

    Returns:
        3x3 diagonal dielectric tensor.
    """
    n_para = (1 - delta_para) + 1j * beta_para
    n_perp = (1 - delta_perp) + 1j * beta_perp
    return compute_dielectric_tensor(n_para, n_perp)


def get_refractive_index_at_energy(
    material_data: MaterialData,
    energy: float,
) -> tuple[Array, Array]:
    """Get interpolated refractive index at a specific energy.

    This function is differentiable with respect to energy for fitting.

    Args:
        material_data: MaterialData containing optical constants.
        energy: Target energy in eV.

    Returns:
        Tuple of (npara, nperp) arrays of shape [num_materials].
    """
    # Find interpolation position
    energies = material_data.energies

    # Linear interpolation for complex values
    npara = _interp_complex(energy, energies, material_data.npara)
    nperp = _interp_complex(energy, energies, material_data.nperp)

    return npara, nperp


def _interp_complex(x: float, xp: Array, fp: Array) -> Array:
    """Interpolate complex array along first axis.

    Args:
        x: Target x value.
        xp: Array of x coordinates (sorted), shape [num_energies].
        fp: Array of shape [num_energies, num_materials] of complex values.

    Returns:
        Interpolated values of shape [num_materials].
    """
    # Interpolate each material separately
    def interp_single(fp_col):
        real = jnp.interp(x, xp, fp_col.real)
        imag = jnp.interp(x, xp, fp_col.imag)
        return real + 1j * imag

    # Apply along material axis (axis 1)
    return jax.vmap(interp_single, in_axes=1, out_axes=0)(fp)
