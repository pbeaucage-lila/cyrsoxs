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
from cyrsoxs_jax.morphology import (
    MorphologyMetadata,
    read_morphology,
    read_morphology_metadata,
    read_vector_morphology,
    read_euler_morphology,
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
from cyrsoxs_jax import rotation

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
    # Morphology
    "MorphologyMetadata",
    "read_morphology",
    "read_morphology_metadata",
    "read_vector_morphology",
    "read_euler_morphology",
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
    # Rotation
    "rotation",
]
