"""Morphology I/O for CyRSoXS HDF5 files.

This module provides functions to read morphology data from HDF5 files
in the CyRSoXS format, supporting both vector morphology and Euler angle
representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from cyrsoxs_jax.types import MorphologyOrder, MorphologyType, VoxelData


@dataclass
class MorphologyMetadata:
    """Metadata from a CyRSoXS morphology HDF5 file.

    Attributes:
        num_materials: Number of materials in the morphology.
        phys_size: Physical size of a voxel in nanometers.
        voxel_dims: Tuple of (Nx, Ny, Nz) voxel dimensions.
        morphology_type: Type of morphology (vector or Euler angles).
        morphology_order: Axis ordering in the file (XYZ or ZYX).
    """

    num_materials: int
    phys_size: float
    voxel_dims: tuple[int, int, int]
    morphology_type: MorphologyType
    morphology_order: MorphologyOrder


def _get_axis_order(dataset: h5py.Dataset) -> MorphologyOrder:
    """Determine axis ordering from HDF5 dimension labels."""
    try:
        label_0 = dataset.dims[0].label
        label_2 = dataset.dims[2].label
    except (AttributeError, IndexError):
        return MorphologyOrder.ZYX

    if label_0 == "Z" and label_2 == "X":
        return MorphologyOrder.ZYX
    elif label_0 == "X" and label_2 == "Z":
        return MorphologyOrder.XYZ
    else:
        return MorphologyOrder.ZYX


def _xyz_to_zyx(data: np.ndarray, num_components: int = 1) -> np.ndarray:
    """Convert XYZ ordered data to ZYX ordering."""
    if num_components == 1:
        return np.swapaxes(data, 0, 2)
    else:
        return np.swapaxes(data, 0, 2)


def _detect_morphology_type(f: h5py.File) -> MorphologyType:
    """Detect morphology type from HDF5 file structure."""
    if "Vector_Morphology" in f:
        return MorphologyType.VECTOR_MORPHOLOGY
    elif "Euler_Angles" in f:
        return MorphologyType.EULER_ANGLES
    elif "vector_morphology" in f:
        return MorphologyType.VECTOR_MORPHOLOGY
    elif "euler_angles" in f:
        return MorphologyType.EULER_ANGLES
    else:
        raise ValueError("Cannot detect morphology type from HDF5 file")


def _get_group_name(f: h5py.File, morphology_type: MorphologyType) -> str:
    """Get the actual group name handling case variations."""
    if morphology_type == MorphologyType.VECTOR_MORPHOLOGY:
        if "Vector_Morphology" in f:
            return "Vector_Morphology"
        elif "vector_morphology" in f:
            return "vector_morphology"
    else:
        if "Euler_Angles" in f:
            return "Euler_Angles"
        elif "euler_angles" in f:
            return "euler_angles"
    raise ValueError(f"Group not found for morphology type {morphology_type}")


def _get_params_group_name(f: h5py.File) -> str:
    """Get the parameters group name handling case variations."""
    if "Morphology_Parameters" in f:
        return "Morphology_Parameters"
    elif "morphology_parameter" in f:
        return "morphology_parameter"
    elif "morphology_parameters" in f:
        return "morphology_parameters"
    raise ValueError("Morphology parameters group not found")


def read_morphology_metadata(filepath: str | Path) -> MorphologyMetadata:
    """Read metadata from a CyRSoXS morphology HDF5 file.

    Args:
        filepath: Path to the HDF5 morphology file.

    Returns:
        MorphologyMetadata with file information.
    """
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        params_group = _get_params_group_name(f)
        morphology_type = _detect_morphology_type(f)
        group_name = _get_group_name(f, morphology_type)

        num_materials = int(f[params_group]["NumMaterial"][()])
        phys_size = float(f[params_group]["PhysSize"][()])

        if morphology_type == MorphologyType.VECTOR_MORPHOLOGY:
            sample_dataset = f[group_name]["Mat_1_unaligned"]
        else:
            sample_dataset = f[group_name]["Mat_1_Vfrac"]

        morphology_order = _get_axis_order(sample_dataset)
        dims = sample_dataset.shape

        if morphology_order == MorphologyOrder.ZYX:
            voxel_dims = (int(dims[2]), int(dims[1]), int(dims[0]))
        else:
            voxel_dims = (int(dims[0]), int(dims[1]), int(dims[2]))

    return MorphologyMetadata(
        num_materials=num_materials,
        phys_size=phys_size,
        voxel_dims=voxel_dims,
        morphology_type=morphology_type,
        morphology_order=morphology_order,
    )


def read_vector_morphology(filepath: str | Path) -> tuple[VoxelData, float]:
    """Read vector morphology data from HDF5 file.

    Args:
        filepath: Path to the HDF5 morphology file.

    Returns:
        Tuple of (VoxelData, phys_size).
    """
    filepath = Path(filepath)
    metadata = read_morphology_metadata(filepath)

    if metadata.morphology_type != MorphologyType.VECTOR_MORPHOLOGY:
        raise ValueError(
            f"File contains {metadata.morphology_type.name}, not vector morphology"
        )

    num_materials = metadata.num_materials
    nx, ny, nz = metadata.voxel_dims

    alignment_list = []
    unaligned_list = []

    with h5py.File(filepath, "r") as f:
        group_name = _get_group_name(f, MorphologyType.VECTOR_MORPHOLOGY)
        group = f[group_name]

        for mat_id in range(1, num_materials + 1):
            unaligned_name = f"Mat_{mat_id}_unaligned"
            alignment_name = f"Mat_{mat_id}_alignment"

            unaligned_data = np.array(group[unaligned_name])
            alignment_data = np.array(group[alignment_name])

            if metadata.morphology_order == MorphologyOrder.XYZ:
                unaligned_data = _xyz_to_zyx(unaligned_data)
                alignment_data = np.swapaxes(alignment_data, 0, 2)

            unaligned_list.append(unaligned_data)
            alignment_list.append(alignment_data)

    alignment = jnp.array(np.stack(alignment_list, axis=0))
    unaligned_fraction = jnp.array(np.stack(unaligned_list, axis=0))

    voxel_data = VoxelData.from_vector_morphology(
        alignment=alignment,
        unaligned_fraction=unaligned_fraction,
        num_materials=num_materials,
    )

    return voxel_data, metadata.phys_size


def read_euler_morphology(filepath: str | Path) -> tuple[VoxelData, float]:
    """Read Euler angle morphology data from HDF5 file.

    Args:
        filepath: Path to the HDF5 morphology file.

    Returns:
        Tuple of (VoxelData, phys_size).
    """
    filepath = Path(filepath)
    metadata = read_morphology_metadata(filepath)

    if metadata.morphology_type != MorphologyType.EULER_ANGLES:
        raise ValueError(
            f"File contains {metadata.morphology_type.name}, not Euler angles"
        )

    num_materials = metadata.num_materials
    nx, ny, nz = metadata.voxel_dims

    s_list = []
    theta_list = []
    psi_list = []
    vfrac_list = []

    with h5py.File(filepath, "r") as f:
        group_name = _get_group_name(f, MorphologyType.EULER_ANGLES)
        group = f[group_name]

        for mat_id in range(1, num_materials + 1):
            vfrac_name = f"Mat_{mat_id}_Vfrac"
            s_name = f"Mat_{mat_id}_S"
            theta_name = f"Mat_{mat_id}_Theta"
            psi_name = f"Mat_{mat_id}_Psi"

            vfrac_data = np.array(group[vfrac_name])

            if s_name in group:
                s_data = np.array(group[s_name])
                theta_data = np.array(group[theta_name])
                psi_data = np.array(group[psi_name])
                theta_data = np.where(s_data == 0, 0.0, theta_data)
                psi_data = np.where(s_data == 0, 0.0, psi_data)
            else:
                s_data = np.zeros_like(vfrac_data)
                theta_data = np.zeros_like(vfrac_data)
                psi_data = np.zeros_like(vfrac_data)

            if metadata.morphology_order == MorphologyOrder.XYZ:
                vfrac_data = _xyz_to_zyx(vfrac_data)
                s_data = _xyz_to_zyx(s_data)
                theta_data = _xyz_to_zyx(theta_data)
                psi_data = _xyz_to_zyx(psi_data)

            vfrac_list.append(vfrac_data)
            s_list.append(s_data)
            theta_list.append(theta_data)
            psi_list.append(psi_data)

    s_param = jnp.array(np.stack(s_list, axis=0))
    theta = jnp.array(np.stack(theta_list, axis=0))
    psi = jnp.array(np.stack(psi_list, axis=0))
    vfrac = jnp.array(np.stack(vfrac_list, axis=0))

    voxel_data = VoxelData.from_euler_angles(
        s_param=s_param,
        theta=theta,
        psi=psi,
        vfrac=vfrac,
        num_materials=num_materials,
    )

    return voxel_data, metadata.phys_size


def read_morphology(
    filepath: str | Path,
    morphology_type: MorphologyType | None = None,
) -> tuple[VoxelData, float]:
    """Read morphology data from a CyRSoXS HDF5 file.

    This is the main entry point for loading morphology data. It auto-detects
    the morphology type if not specified.

    Args:
        filepath: Path to the HDF5 morphology file.
        morphology_type: Optional explicit morphology type. If None, auto-detected.

    Returns:
        Tuple of (VoxelData, phys_size) where phys_size is in nanometers.

    Example:
        >>> voxel_data, phys_size = read_morphology("morphology.h5")
        >>> print(f"Loaded {voxel_data.num_materials} materials")
        >>> print(f"Voxel dimensions: {voxel_data.voxel_dims}")
    """
    filepath = Path(filepath)

    if morphology_type is None:
        metadata = read_morphology_metadata(filepath)
        morphology_type = metadata.morphology_type

    if morphology_type == MorphologyType.VECTOR_MORPHOLOGY:
        return read_vector_morphology(filepath)
    else:
        return read_euler_morphology(filepath)
