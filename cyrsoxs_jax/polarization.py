"""Polarization computation for CyRSoXS-JAX."""

import jax.numpy as jnp
from jax import Array

ONE_BY_4PI = 1.0 / (4.0 * jnp.pi)


def compute_polarization_field(
    morphology: Array,
    n_para: Array,
    n_perp: Array,
    euler_angles: Array,
    incident_field: Array,
    rotation_matrix: Array | None = None,
) -> Array:
    """Compute the polarization field for uniaxial materials.

    Implements the Born approximation polarization: P = (1/4π) * (n²-I) · E
    where n² is the rotated dielectric tensor for each voxel.

    Args:
        morphology: Per-material data (N, N, N, num_materials, 4) where the last
            dimension contains [psi, theta, vfrac, S] (Euler angles, volume fraction,
            and alignment parameter).
        n_para: Complex parallel refractive index per material (num_materials,).
        n_perp: Complex perpendicular refractive index per material (num_materials,).
        euler_angles: Euler angles (N, N, N, 3) - [psi, theta, phi] (unused if
            morphology contains per-material angles).
        incident_field: Incident electric field vector (3,). Typically [1, 0, 0].
        rotation_matrix: Optional rotation matrix (3, 3) for k/E frame rotation.
            If None, identity is used.

    Returns:
        Complex polarization field (N, N, N, 3).
    """
    if rotation_matrix is None:
        rotation_matrix = jnp.eye(3)

    # Rotate incident field by rotation matrix: matVec = R @ E
    mat_vec = rotation_matrix @ incident_field

    # Square the refractive indices to get dielectric constants
    # n_para, n_perp are complex: (num_materials,)
    n_para_sq = n_para * n_para
    n_perp_sq = n_perp * n_perp
    n_sum_sq = (n_para + 2 * n_perp) ** 2

    # Initialize polarization accumulators
    shape = morphology.shape[:3]
    num_materials = morphology.shape[3]
    p_x = jnp.zeros(shape, dtype=jnp.complex64)
    p_y = jnp.zeros(shape, dtype=jnp.complex64)
    p_z = jnp.zeros(shape, dtype=jnp.complex64)

    # Loop over materials
    for m in range(num_materials):
        # Extract per-voxel material properties
        mat_prop = morphology[:, :, :, m, :]  # (N, N, N, 4)
        psi = mat_prop[:, :, :, 0]
        theta = mat_prop[:, :, :, 1]
        vfrac = mat_prop[:, :, :, 2]
        s_param = mat_prop[:, :, :, 3]

        # Compute orientation vector from Euler angles
        sx = jnp.cos(psi) * jnp.sin(theta)
        sy = jnp.sin(psi) * jnp.sin(theta)
        sz = jnp.cos(theta)

        # phi_a = aligned fraction, phi_ui = unaligned (isotropic) fraction
        phi_a = vfrac * s_param
        phi_ui = vfrac - phi_a
        phi = vfrac

        # Get material optical constants
        npar = n_para_sq[m]
        nper = n_perp_sq[m]
        nsum = n_sum_sq[m]

        # Build symmetric dielectric tensor elements and apply to E field
        # Tensor structure: [[0,1,2], [1,3,4], [2,4,5]]

        # Element (0,0): npar*sx² + nper*(sy² + sz²) + isotropic - phi
        nr_00 = phi_a * (npar * sx * sx + nper * (sy * sy + sz * sz)) + (phi_ui * nsum / 9.0) - phi
        p_x = p_x + nr_00 * mat_vec[0]

        # Element (0,1) = (1,0): (npar - nper)*sx*sy
        nr_01 = phi_a * (npar - nper) * sx * sy
        p_x = p_x + nr_01 * mat_vec[1]
        p_y = p_y + nr_01 * mat_vec[0]

        # Element (0,2) = (2,0): (npar - nper)*sx*sz
        nr_02 = phi_a * (npar - nper) * sx * sz
        p_x = p_x + nr_02 * mat_vec[2]
        p_z = p_z + nr_02 * mat_vec[0]

        # Element (1,1): npar*sy² + nper*(sx² + sz²) + isotropic - phi
        nr_11 = phi_a * (npar * sy * sy + nper * (sx * sx + sz * sz)) + (phi_ui * nsum / 9.0) - phi
        p_y = p_y + nr_11 * mat_vec[1]

        # Element (1,2) = (2,1): (npar - nper)*sy*sz
        nr_12 = phi_a * (npar - nper) * sy * sz
        p_y = p_y + nr_12 * mat_vec[2]
        p_z = p_z + nr_12 * mat_vec[1]

        # Element (2,2): npar*sz² + nper*(sx² + sy²) + isotropic - phi
        nr_22 = phi_a * (npar * sz * sz + nper * (sx * sx + sy * sy)) + (phi_ui * nsum / 9.0) - phi
        p_z = p_z + nr_22 * mat_vec[2]

    # Scale by 1/(4π)
    p_x = p_x * ONE_BY_4PI
    p_y = p_y * ONE_BY_4PI
    p_z = p_z * ONE_BY_4PI

    # Stack into (N, N, N, 3) array
    return jnp.stack([p_x, p_y, p_z], axis=-1)


def compute_polarization_vector_morphology(
    morphology: Array,
    n_para: Array,
    n_perp: Array,
    incident_field: Array,
    rotation_matrix: Array | None = None,
) -> Array:
    """Compute polarization field using vector morphology format.

    This variant uses the vector morphology format where orientation is stored
    as (sx, sy, sz, phi_ui) rather than Euler angles.

    Args:
        morphology: Per-material data (N, N, N, num_materials, 4) where the last
            dimension contains [sx, sy, sz, phi_ui] (orientation vector components
            and unaligned volume fraction).
        n_para: Complex parallel refractive index per material (num_materials,).
        n_perp: Complex perpendicular refractive index per material (num_materials,).
        incident_field: Incident electric field vector (3,). Typically [1, 0, 0].
        rotation_matrix: Optional rotation matrix (3, 3) for k/E frame rotation.

    Returns:
        Complex polarization field (N, N, N, 3).
    """
    if rotation_matrix is None:
        rotation_matrix = jnp.eye(3)

    mat_vec = rotation_matrix @ incident_field

    n_para_sq = n_para * n_para
    n_perp_sq = n_perp * n_perp
    n_sum_sq = (n_para + 2 * n_perp) ** 2

    shape = morphology.shape[:3]
    num_materials = morphology.shape[3]
    p_x = jnp.zeros(shape, dtype=jnp.complex64)
    p_y = jnp.zeros(shape, dtype=jnp.complex64)
    p_z = jnp.zeros(shape, dtype=jnp.complex64)

    for m in range(num_materials):
        mat_prop = morphology[:, :, :, m, :]
        sx = mat_prop[:, :, :, 0]
        sy = mat_prop[:, :, :, 1]
        sz = mat_prop[:, :, :, 2]
        phi_ui = mat_prop[:, :, :, 3]

        # In vector morphology: phi = phi_ui + |s|²
        phi = phi_ui + sx * sx + sy * sy + sz * sz

        npar = n_para_sq[m]
        nper = n_perp_sq[m]
        nsum = n_sum_sq[m]

        # Tensor element (0,0)
        nr_00 = npar * sx * sx + nper * (sy * sy + sz * sz) + (phi_ui * nsum / 9.0) - phi
        p_x = p_x + nr_00 * mat_vec[0]

        # Tensor element (0,1)
        nr_01 = (npar - nper) * sx * sy
        p_x = p_x + nr_01 * mat_vec[1]
        p_y = p_y + nr_01 * mat_vec[0]

        # Tensor element (0,2)
        nr_02 = (npar - nper) * sx * sz
        p_x = p_x + nr_02 * mat_vec[2]
        p_z = p_z + nr_02 * mat_vec[0]

        # Tensor element (1,1)
        nr_11 = npar * sy * sy + nper * (sx * sx + sz * sz) + (phi_ui * nsum / 9.0) - phi
        p_y = p_y + nr_11 * mat_vec[1]

        # Tensor element (1,2)
        nr_12 = (npar - nper) * sy * sz
        p_y = p_y + nr_12 * mat_vec[2]
        p_z = p_z + nr_12 * mat_vec[1]

        # Tensor element (2,2)
        nr_22 = npar * sz * sz + nper * (sx * sx + sy * sy) + (phi_ui * nsum / 9.0) - phi
        p_z = p_z + nr_22 * mat_vec[2]

    p_x = p_x * ONE_BY_4PI
    p_y = p_y * ONE_BY_4PI
    p_z = p_z * ONE_BY_4PI

    return jnp.stack([p_x, p_y, p_z], axis=-1)


def build_dielectric_tensor(
    sx: Array,
    sy: Array,
    sz: Array,
    phi_a: Array,
    phi_ui: Array,
    phi: Array,
    npar: complex,
    nper: complex,
    nsum: complex,
) -> Array:
    """Build the 6 unique elements of the symmetric dielectric tensor.

    Args:
        sx, sy, sz: Orientation vector components (N, N, N).
        phi_a: Aligned volume fraction (N, N, N).
        phi_ui: Unaligned volume fraction (N, N, N).
        phi: Total volume fraction (N, N, N).
        npar: Squared parallel refractive index (complex scalar).
        nper: Squared perpendicular refractive index (complex scalar).
        nsum: Squared sum term (n_para + 2*n_perp)² (complex scalar).

    Returns:
        Dielectric tensor elements (N, N, N, 6) in order [00, 01, 02, 11, 12, 22].
    """
    iso_term = phi_ui * nsum / 9.0

    nr_00 = phi_a * (npar * sx * sx + nper * (sy * sy + sz * sz)) + iso_term - phi
    nr_01 = phi_a * (npar - nper) * sx * sy
    nr_02 = phi_a * (npar - nper) * sx * sz
    nr_11 = phi_a * (npar * sy * sy + nper * (sx * sx + sz * sz)) + iso_term - phi
    nr_12 = phi_a * (npar - nper) * sy * sz
    nr_22 = phi_a * (npar * sz * sz + nper * (sx * sx + sy * sy)) + iso_term - phi

    return jnp.stack([nr_00, nr_01, nr_02, nr_11, nr_12, nr_22], axis=-1)


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
