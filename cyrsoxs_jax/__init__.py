"""CyRSoXS-JAX: JAX port of CyRSoXS for resonant soft X-ray scattering simulation."""

from cyrsoxs_jax.types import (
    Voxel,
    VoxelData,
    Material,
    MaterialData,
    RotationMatrix,
    MorphologyType,
    MorphologyOrder,
    ReferenceFrame,
    EwaldsInterpolation,
    FFTWindowing,
)

from cyrsoxs_jax.materials import (
    OpticalConstantsRaw,
    load_raw_optical_constants,
    load_processed_optical_constants,
    interpolate_optical_constants,
    combine_materials,
    create_vacuum_material,
    compute_dielectric_tensor,
    compute_dielectric_tensor_from_delta_beta,
    get_refractive_index_at_energy,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Voxel",
    "VoxelData",
    "Material",
    "MaterialData",
    "RotationMatrix",
    "MorphologyType",
    "MorphologyOrder",
    "ReferenceFrame",
    "EwaldsInterpolation",
    "FFTWindowing",
    # Materials
    "OpticalConstantsRaw",
    "load_raw_optical_constants",
    "load_processed_optical_constants",
    "interpolate_optical_constants",
    "combine_materials",
    "create_vacuum_material",
    "compute_dielectric_tensor",
    "compute_dielectric_tensor_from_delta_beta",
    "get_refractive_index_at_energy",
]
