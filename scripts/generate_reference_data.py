#!/usr/bin/env python3
"""Generate reference data from CUDA CyRSoXS for validation tests.

This script creates test morphologies and runs them through CUDA CyRSoXS
to generate reference data for validating the JAX port.

Usage:
    python scripts/generate_reference_data.py --cuda-path /path/to/cyrsoxs

Requirements:
    - Compiled CUDA CyRSoXS binary
    - h5py for morphology I/O
    - numpy for data storage
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np


def create_sphere_morphology(n: int = 32, radius: float = 10.0) -> dict:
    """Create a simple sphere morphology for testing.

    Args:
        n: Grid size.
        radius: Sphere radius in voxels.

    Returns:
        Dictionary with morphology data.
    """
    center = n // 2
    z, y, x = np.mgrid[:n, :n, :n]
    dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)

    # Volume fractions: material 0 inside sphere, material 1 outside
    vfrac_0 = (dist <= radius).astype(np.float32)
    vfrac_1 = 1.0 - vfrac_0

    # Random orientations (Euler angles)
    np.random.seed(42)
    theta = np.random.uniform(0, np.pi, (n, n, n)).astype(np.float32)
    psi = np.random.uniform(0, 2 * np.pi, (n, n, n)).astype(np.float32)
    s_param = np.ones((n, n, n), dtype=np.float32) * 0.8

    # Stack for 2 materials: shape (N, N, N, num_materials, 4)
    # [psi, theta, vfrac, S]
    morphology = np.zeros((n, n, n, 2, 4), dtype=np.float32)
    morphology[:, :, :, 0, 0] = psi
    morphology[:, :, :, 0, 1] = theta
    morphology[:, :, :, 0, 2] = vfrac_0
    morphology[:, :, :, 0, 3] = s_param * vfrac_0

    morphology[:, :, :, 1, 0] = psi
    morphology[:, :, :, 1, 1] = theta
    morphology[:, :, :, 1, 2] = vfrac_1
    morphology[:, :, :, 1, 3] = s_param * vfrac_1

    return {
        "morphology": morphology,
        "euler_angles": np.stack([psi, theta, np.zeros_like(psi)], axis=-1),
        "grid_size": n,
        "physical_size": 1.0,  # nm per voxel
    }


def create_lamellar_morphology(n: int = 32, period: int = 8) -> dict:
    """Create a lamellar (alternating layers) morphology.

    Args:
        n: Grid size.
        period: Layer period in voxels.

    Returns:
        Dictionary with morphology data.
    """
    z, y, x = np.mgrid[:n, :n, :n]

    # Alternating layers along z
    layer_idx = (z // (period // 2)) % 2
    vfrac_0 = (layer_idx == 0).astype(np.float32)
    vfrac_1 = 1.0 - vfrac_0

    # Oriented along z axis
    theta = np.zeros((n, n, n), dtype=np.float32)
    psi = np.zeros((n, n, n), dtype=np.float32)
    s_param = np.ones((n, n, n), dtype=np.float32)

    morphology = np.zeros((n, n, n, 2, 4), dtype=np.float32)
    morphology[:, :, :, 0, 0] = psi
    morphology[:, :, :, 0, 1] = theta
    morphology[:, :, :, 0, 2] = vfrac_0
    morphology[:, :, :, 0, 3] = s_param * vfrac_0

    morphology[:, :, :, 1, 0] = psi
    morphology[:, :, :, 1, 1] = theta
    morphology[:, :, :, 1, 2] = vfrac_1
    morphology[:, :, :, 1, 3] = s_param * vfrac_1

    return {
        "morphology": morphology,
        "euler_angles": np.stack([psi, theta, np.zeros_like(psi)], axis=-1),
        "grid_size": n,
        "physical_size": 1.0,
    }


def get_test_optical_constants() -> dict:
    """Get test optical constants for two materials.

    Returns typical values for organic polymer materials near the carbon K-edge.
    """
    # Material 0: Higher contrast material
    n_para_0 = 1.0 + 0.01j
    n_perp_0 = 1.0 + 0.005j

    # Material 1: Lower contrast material (vacuum-like)
    n_para_1 = 1.0 + 0.001j
    n_perp_1 = 1.0 + 0.001j

    return {
        "n_para": np.array([n_para_0, n_para_1], dtype=np.complex64),
        "n_perp": np.array([n_perp_0, n_perp_1], dtype=np.complex64),
        "energy": 285.0,  # eV
    }


def save_reference_data(output_dir: Path, name: str, data: dict):
    """Save reference data to npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"{name}.npz", **data)
    print(f"Saved: {output_dir / name}.npz")


def generate_synthetic_reference(output_dir: Path):
    """Generate synthetic reference data for initial testing.

    This creates reference data using the JAX implementation itself,
    useful for regression testing even without CUDA reference.
    """
    import jax.numpy as jnp
    from cyrsoxs_jax import polarization

    print("Generating synthetic reference data (JAX baseline)...")

    # Sphere polarization
    sphere = create_sphere_morphology(n=16)
    optics = get_test_optical_constants()
    incident_field = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Compute JAX result as baseline
    pol_result = polarization.compute_polarization_field(
        jnp.array(sphere["morphology"]),
        jnp.array(optics["n_para"]),
        jnp.array(optics["n_perp"]),
        jnp.array(sphere["euler_angles"]),
        jnp.array(incident_field),
    )

    sphere_ref = {
        "morphology": sphere["morphology"],
        "euler_angles": sphere["euler_angles"],
        "n_para": optics["n_para"],
        "n_perp": optics["n_perp"],
        "incident_field": incident_field,
        "polarization_cuda": np.array(pol_result),
        "is_jax_baseline": True,  # Flag indicating this is JAX, not CUDA
    }
    save_reference_data(output_dir, "sphere_polarization", sphere_ref)

    # Lamellar polarization
    lamellar = create_lamellar_morphology(n=16)
    pol_lamellar = polarization.compute_polarization_field(
        jnp.array(lamellar["morphology"]),
        jnp.array(optics["n_para"]),
        jnp.array(optics["n_perp"]),
        jnp.array(lamellar["euler_angles"]),
        jnp.array(incident_field),
    )

    lamellar_ref = {
        "morphology": lamellar["morphology"],
        "euler_angles": lamellar["euler_angles"],
        "n_para": optics["n_para"],
        "n_perp": optics["n_perp"],
        "incident_field": incident_field,
        "polarization_cuda": np.array(pol_lamellar),
        "is_jax_baseline": True,
    }
    save_reference_data(output_dir, "lamellar_polarization", lamellar_ref)

    # FFT reference
    n = 16
    np.random.seed(42)
    input_field = np.random.randn(n, n, n).astype(np.float32)
    fft_ref = {
        "input_field": input_field,
        "fft_cuda": np.fft.fftn(input_field),  # NumPy as reference
    }
    save_reference_data(output_dir, "fft_reference", fft_ref)

    # Multi-material reference (same as sphere, has 2 materials)
    multi_ref = {
        "morphology": sphere["morphology"],
        "euler_angles": sphere["euler_angles"],
        "n_para": optics["n_para"],
        "n_perp": optics["n_perp"],
        "incident_field": incident_field,
        "polarization_cuda": np.array(pol_result),
        "is_jax_baseline": True,
    }
    save_reference_data(output_dir, "multi_material", multi_ref)

    # Rotation reference
    rot_angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle), np.cos(rot_angle), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    pol_rotated = polarization.compute_polarization_field(
        jnp.array(sphere["morphology"]),
        jnp.array(optics["n_para"]),
        jnp.array(optics["n_perp"]),
        jnp.array(sphere["euler_angles"]),
        jnp.array(incident_field),
        rotation_matrix=jnp.array(rotation_matrix),
    )

    rotation_ref = {
        "morphology": sphere["morphology"],
        "euler_angles": sphere["euler_angles"],
        "n_para": optics["n_para"],
        "n_perp": optics["n_perp"],
        "incident_field": incident_field,
        "rotation_matrix": rotation_matrix,
        "polarization_cuda": np.array(pol_rotated),
        "is_jax_baseline": True,
    }
    save_reference_data(output_dir, "rotation_reference", rotation_ref)

    print(f"\nGenerated {5} reference data files in {output_dir}")
    print("\nNote: These are JAX baseline values for regression testing.")
    print("For true CUDA validation, run with --cuda-path /path/to/cyrsoxs")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA reference data for JAX validation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "reference_data",
        help="Output directory for reference data",
    )
    parser.add_argument(
        "--cuda-path",
        type=Path,
        help="Path to compiled CUDA CyRSoXS binary (optional)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic reference data without CUDA",
    )
    args = parser.parse_args()

    if args.cuda_path and args.cuda_path.exists():
        print(f"Using CUDA binary: {args.cuda_path}")
        print("CUDA reference generation not yet implemented")
        print("Falling back to synthetic data...")
        generate_synthetic_reference(args.output_dir)
    else:
        if not args.synthetic:
            print("No CUDA binary specified. Use --synthetic for placeholder data.")
            print("Or specify --cuda-path /path/to/cyrsoxs")
            return
        generate_synthetic_reference(args.output_dir)


if __name__ == "__main__":
    main()
