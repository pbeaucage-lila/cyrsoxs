"""CyRSoXS-JAX: JAX port of CyRSoXS for resonant soft X-ray scattering simulation."""

from cyrsoxs_jax.detector import (
    rotate_image,
    rotate_image_with_mask,
    average_rotations,
    rotate_and_average,
    apply_detector_rotation,
)
from cyrsoxs_jax.simulate import simulate, simulate_single_energy
from cyrsoxs_jax.types import (
    Morphology,
    VoxelData,
    MaterialData,
    SimulationConfig,
    ScatteringResult,
    MorphologyType,
)
from cyrsoxs_jax.fitting import (
    OpticalParams,
    MorphologyParams,
    FitParams,
    FitState,
    simulate_differentiable,
    make_forward_model,
    make_checkpointed_forward_model,
    mse_loss,
    chi_squared_loss,
    log_intensity_loss,
    create_fit_state,
    fit_step,
    fit_loop,
)

__version__ = "0.1.0"

__all__ = [
    # Detector
    "rotate_image",
    "rotate_image_with_mask",
    "average_rotations",
    "rotate_and_average",
    "apply_detector_rotation",
    # Simulation
    "simulate",
    "simulate_single_energy",
    # Types
    "Morphology",
    "VoxelData",
    "MaterialData",
    "SimulationConfig",
    "ScatteringResult",
    "MorphologyType",
    # Fitting API
    "OpticalParams",
    "MorphologyParams",
    "FitParams",
    "FitState",
    "simulate_differentiable",
    "make_forward_model",
    "make_checkpointed_forward_model",
    "mse_loss",
    "chi_squared_loss",
    "log_intensity_loss",
    "create_fit_state",
    "fit_step",
    "fit_loop",
]
