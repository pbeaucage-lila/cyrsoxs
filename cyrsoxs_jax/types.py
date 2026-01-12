"""Core data types and array structures for CyRSoXS-JAX.

This module defines JAX-compatible data structures matching the CUDA types
used in the original CyRSoXS implementation.

Array Shape Conventions:
    - Voxel data: [num_materials, Nz, Ny, Nx, 4] for s1 components
    - Polarization: [3, Nz, Ny, Nx] complex - for (pX, pY, pZ) components
    - Rotation matrices: [3, 3] or batched [N, 3, 3]
    - Material optical constants: [num_energies, num_materials] for each of npara/nperp
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class MorphologyType(IntEnum):
    """Type of morphology representation."""
    EULER_ANGLES = 0
    VECTOR_MORPHOLOGY = 1


class MorphologyOrder(IntEnum):
    """Axis ordering for morphology data."""
    ZYX = 0  # First axis is Z, second Y, third X
    XYZ = 1  # First axis is X, second Y, third Z
    INVALID = -1


class ReferenceFrame(IntEnum):
    """Reference frame for computing p vector."""
    MATERIAL = 0  # Rotate p along with E
    LAB = 1       # Compute in Lab axis frame


class EwaldsInterpolation(IntEnum):
    """Interpolation type for I(q)."""
    NEAREST_NEIGHBOUR = 0
    LINEAR = 1


class FFTWindowing(IntEnum):
    """FFT windowing type."""
    NONE = 0
    HANNING = 1


class Voxel(NamedTuple):
    """Single voxel data for one material.

    For Vector Morphology:
        sx: x component of director vector
        sy: y component of director vector
        sz: z component of director vector
        phi_ua: fraction of unaligned component

    For Euler Angles:
        s: fraction of aligned component (S parameter)
        theta: rotation angle about X-axis
        psi: second rotation angle about Z-axis
        vfrac: volume fraction of the material
    """
    s1_x: float  # sx or S
    s1_y: float  # sy or theta
    s1_z: float  # sz or psi
    s1_w: float  # phi_ua or vfrac


@dataclass
class VoxelData:
    """Voxel data for all materials in the simulation volume.

    Attributes:
        data: Array of shape [num_materials, Nz, Ny, Nx, 4] containing voxel data.
              The last dimension stores (s1_x, s1_y, s1_z, s1_w) for each voxel.
        morphology_type: Whether data represents Euler angles or vector morphology.
        voxel_dims: Tuple of (Nz, Ny, Nx) dimensions.
        num_materials: Number of materials in the morphology.
    """
    data: Array
    morphology_type: MorphologyType
    voxel_dims: tuple[int, int, int]
    num_materials: int

    @classmethod
    def from_vector_morphology(
        cls,
        alignment: Array,
        unaligned_fraction: Array,
        num_materials: int,
    ) -> VoxelData:
        """Create VoxelData from vector morphology arrays.

        Args:
            alignment: Array of shape [num_materials, Nz, Ny, Nx, 3] with (sx, sy, sz).
            unaligned_fraction: Array of shape [num_materials, Nz, Ny, Nx] with phi_ua.

        Returns:
            VoxelData instance.
        """
        voxel_dims = alignment.shape[1:4]
        data = jnp.concatenate([
            alignment,
            unaligned_fraction[..., jnp.newaxis]
        ], axis=-1)
        return cls(
            data=data,
            morphology_type=MorphologyType.VECTOR_MORPHOLOGY,
            voxel_dims=voxel_dims,
            num_materials=num_materials,
        )

    @classmethod
    def from_euler_angles(
        cls,
        s_param: Array,
        theta: Array,
        psi: Array,
        vfrac: Array,
        num_materials: int,
    ) -> VoxelData:
        """Create VoxelData from Euler angle arrays.

        Args:
            s_param: Alignment parameter S, shape [num_materials, Nz, Ny, Nx].
            theta: Rotation angle about X-axis, shape [num_materials, Nz, Ny, Nx].
            psi: Second rotation angle about Z-axis, shape [num_materials, Nz, Ny, Nx].
            vfrac: Volume fraction, shape [num_materials, Nz, Ny, Nx].

        Returns:
            VoxelData instance.
        """
        voxel_dims = s_param.shape[1:4]
        data = jnp.stack([s_param, theta, psi, vfrac], axis=-1)
        return cls(
            data=data,
            morphology_type=MorphologyType.EULER_ANGLES,
            voxel_dims=voxel_dims,
            num_materials=num_materials,
        )

    def get_material(self, mat_id: int) -> Array:
        """Get voxel data for a specific material.

        Args:
            mat_id: Material index (0-based).

        Returns:
            Array of shape [Nz, Ny, Nx, 4].
        """
        return self.data[mat_id]


@dataclass
class Material:
    """Optical constants for a material at a single energy.

    The refractive index is expressed as n = 1 - delta + i*beta,
    stored as complex: n = (1 - delta) + i*beta.

    Attributes:
        npara: Parallel component of refractive index (complex).
        nperp: Perpendicular component of refractive index (complex).
    """
    npara: complex
    nperp: complex

    @classmethod
    def from_optical_constants(
        cls,
        delta_para: float,
        beta_para: float,
        delta_perp: float,
        beta_perp: float,
    ) -> Material:
        """Create Material from optical constants (delta, beta).

        Args:
            delta_para: Parallel delta component.
            beta_para: Parallel beta component.
            delta_perp: Perpendicular delta component.
            beta_perp: Perpendicular beta component.

        Returns:
            Material instance.
        """
        return cls(
            npara=complex(1 - delta_para, beta_para),
            nperp=complex(1 - delta_perp, beta_perp),
        )


@dataclass
class MaterialData:
    """Optical constants for all materials across all energies.

    Attributes:
        npara: Array of shape [num_energies, num_materials] complex.
        nperp: Array of shape [num_energies, num_materials] complex.
        energies: Array of energy values in eV.
        num_materials: Number of materials.
    """
    npara: Array
    nperp: Array
    energies: Array
    num_materials: int

    @classmethod
    def from_optical_constants(
        cls,
        delta_para: Array,
        beta_para: Array,
        delta_perp: Array,
        beta_perp: Array,
        energies: Array,
    ) -> MaterialData:
        """Create MaterialData from optical constant arrays.

        Args:
            delta_para: Shape [num_energies, num_materials].
            beta_para: Shape [num_energies, num_materials].
            delta_perp: Shape [num_energies, num_materials].
            beta_perp: Shape [num_energies, num_materials].
            energies: Shape [num_energies].

        Returns:
            MaterialData instance.
        """
        npara = (1 - delta_para) + 1j * beta_para
        nperp = (1 - delta_perp) + 1j * beta_perp
        return cls(
            npara=npara,
            nperp=nperp,
            energies=energies,
            num_materials=delta_para.shape[1],
        )

    def get_at_energy(self, energy_idx: int) -> tuple[Array, Array]:
        """Get optical constants at a specific energy index.

        Returns:
            Tuple of (npara, nperp) arrays of shape [num_materials].
        """
        return self.npara[energy_idx], self.nperp[energy_idx]


@dataclass
class RotationMatrix:
    """3x3 rotation matrix.

    Stored in row-major order as a [3, 3] array.
    Compatible with JAX operations and JIT compilation.

    Attributes:
        matrix: Array of shape [3, 3].
    """
    matrix: Array

    @classmethod
    def identity(cls) -> RotationMatrix:
        """Create an identity rotation matrix."""
        return cls(matrix=jnp.eye(3))

    @classmethod
    def zeros(cls) -> RotationMatrix:
        """Create a zero matrix."""
        return cls(matrix=jnp.zeros((3, 3)))

    @classmethod
    def from_flat(cls, flat: Array) -> RotationMatrix:
        """Create from flat array of 9 elements (row-major)."""
        return cls(matrix=flat.reshape(3, 3))

    def __matmul__(self, other: RotationMatrix) -> RotationMatrix:
        """Matrix multiplication."""
        return RotationMatrix(matrix=self.matrix @ other.matrix)

    def transpose(self) -> RotationMatrix:
        """Return transposed matrix."""
        return RotationMatrix(matrix=self.matrix.T)

    def apply(self, vec: Array) -> Array:
        """Apply rotation to a 3D vector.

        Args:
            vec: Array of shape [3] or [..., 3].

        Returns:
            Rotated vector of same shape.
        """
        return jnp.einsum('ij,...j->...i', self.matrix, vec)

    def apply_transpose(self, vec: Array) -> Array:
        """Apply transposed rotation to a 3D vector.

        Args:
            vec: Array of shape [3] or [..., 3].

        Returns:
            Rotated vector of same shape.
        """
        return jnp.einsum('ji,...j->...i', self.matrix, vec)


# Pytree registration for JAX compatibility
def _voxeldata_flatten(vd: VoxelData):
    children = (vd.data,)
    aux_data = (vd.morphology_type, vd.voxel_dims, vd.num_materials)
    return children, aux_data


def _voxeldata_unflatten(aux_data, children):
    morphology_type, voxel_dims, num_materials = aux_data
    (data,) = children
    return VoxelData(data, morphology_type, voxel_dims, num_materials)


def _materialdata_flatten(md: MaterialData):
    children = (md.npara, md.nperp, md.energies)
    aux_data = (md.num_materials,)
    return children, aux_data


def _materialdata_unflatten(aux_data, children):
    (num_materials,) = aux_data
    npara, nperp, energies = children
    return MaterialData(npara, nperp, energies, num_materials)


def _rotationmatrix_flatten(rm: RotationMatrix):
    return (rm.matrix,), None


def _rotationmatrix_unflatten(aux_data, children):
    (matrix,) = children
    return RotationMatrix(matrix)


# Register pytrees
jax.tree_util.register_pytree_node(
    VoxelData,
    _voxeldata_flatten,
    _voxeldata_unflatten,
)

jax.tree_util.register_pytree_node(
    MaterialData,
    _materialdata_flatten,
    _materialdata_unflatten,
)

jax.tree_util.register_pytree_node(
    RotationMatrix,
    _rotationmatrix_flatten,
    _rotationmatrix_unflatten,
)
