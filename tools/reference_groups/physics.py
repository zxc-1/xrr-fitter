"""Replay frozen R22 physics cases through the single R23 Parratt engine."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import warnings

import numpy as np

from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.physics.derivatives import parratt_reflectivity_jacobian
from xrr_fitter.physics.footprint import footprint_factor
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.reflectivity import instrument_reflectivity, qz_from_theta_deg
from xrr_fitter.physics.resolution import (
    GaussHermiteConvergenceWarning,
    gaussian_smear,
    theta_domain_smear,
)
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure


ARTIFACTS = ("golden/physics.json", "golden/physics.npz")
INPUTS = {
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
        29298,
        "613e86c22605b111ceb57fd6b3a63f93e3a330cfac65cc18d37b2f1a5c2407ee",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
        20247,
        "c2aae5beca68b95d5dd0f06659fdf73c7ddc8921aa46e76bda7e7d2cae35fa65",
    ),
}
INPUT_ORDER = tuple(INPUTS)
CONFIGURATION = {
    "q_grid": {"start": 0.005, "stop": 0.35, "count": 128},
    "theta_grid": {"start": 0.05, "stop": 2.5, "count": 128},
    "profile_step_a": 2.0,
    "relative_sigma": 0.002,
}


def _validate_input(value: object) -> bytes:
    try:
        input_class, path, size, digest = INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("physics input identity drift") from error
    if value.input_class != input_class or value.path != path:
        raise ValueError("physics input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("physics input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("physics input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("physics input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "physics":
        raise ValueError("physics group drift")
    if tuple(context.artifacts) != ARTIFACTS:
        raise ValueError("physics artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("physics configuration drift")
    if tuple(context.seeds):
        raise ValueError("physics seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("physics input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _complex_values(values: np.ndarray) -> list[dict[str, float]]:
    return [{"real": float(value.real), "imag": float(value.imag)} for value in values]


def _span_values(spans: tuple[object, ...]) -> list[dict[str, int]]:
    return [asdict(span) for span in spans]


def _case_arrays(stem: str, prefix: str, content: bytes, q_grid: np.ndarray, theta_grid: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray], object]:
    project = project_from_bytes(content)
    dataset = project.datasets[0]
    assert dataset.structure is not None
    stack = expand_structure(dataset.structure, dataset.beam.wavelength_a)
    depth, profile = sld_depth_profile(stack, step_a=CONFIGURATION["profile_step_a"])
    parratt = parratt_reflectivity(q_grid, stack)
    arrays = {
        f"{prefix}_abeles": parratt,
        f"{prefix}_footprint": footprint_factor(theta_grid, 0.2),
        f"{prefix}_instrument": instrument_reflectivity(theta_grid, stack, dataset.beam, background=1e-8, relative_sigma=CONFIGURATION["relative_sigma"]),
        f"{prefix}_parratt": parratt,
        f"{prefix}_q_resolution": gaussian_smear(q_grid, lambda values: parratt_reflectivity(values, stack), relative_sigma=CONFIGURATION["relative_sigma"], absolute_sigma_a_inv=1e-4),
        f"{prefix}_theta_resolution": theta_domain_smear(theta_grid, lambda values: parratt_reflectivity(qz_from_theta_deg(values, dataset.beam.wavelength_a), stack), sigma_theta_deg=0.01),
        f"{prefix}_profile_depth": depth,
        f"{prefix}_profile_sld_imag": profile.imag,
        f"{prefix}_profile_sld_real": profile.real,
        f"{prefix}_stack_roughness": stack.roughness_a,
        f"{prefix}_stack_sld_imag": stack.sld_a2.imag,
        f"{prefix}_stack_sld_real": stack.sld_a2.real,
        f"{prefix}_stack_thickness": stack.thickness_a,
    }
    summary = {
        "thickness_a": stack.thickness_a.tolist(),
        "roughness_a": stack.roughness_a.tolist(),
        "sld_a2": _complex_values(stack.sld_a2),
        "periodic_spans": _span_values(stack.periodic_spans),
    }
    return summary, arrays, stack


def _derivative_arrays(q_grid: np.ndarray, stack: object) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    parameter_count = 3
    q_tangent = np.empty((q_grid.size, parameter_count), dtype=np.float64)
    q_tangent[:, 0] = np.linspace(2e-4, 8e-4, q_grid.size)
    q_tangent[:, 1] = np.linspace(-3e-4, 4e-4, q_grid.size)
    q_tangent[:, 2] = 1e-4
    thickness_tangent = np.zeros((stack.thickness_a.size, parameter_count), dtype=np.float64)
    sld_tangent = np.zeros((stack.sld_a2.size, parameter_count), dtype=np.complex128)
    roughness_tangent = np.zeros((stack.roughness_a.size, parameter_count), dtype=np.float64)
    thickness_tangent[1, 0] = 0.7
    sld_tangent[1, 1] = 1.1e-7 + 0.4e-8j
    roughness_tangent[:, 2] = np.linspace(0.02, 0.08, stack.roughness_a.size)
    primal, jacobian = parratt_reflectivity_jacobian(q_grid, stack, q_tangent, thickness_tangent, sld_tangent, roughness_tangent)
    arrays = {
        "single_derivative_primal": primal,
        "single_parratt_jacobian": jacobian,
        "single_tangent_qz": q_tangent,
        "single_tangent_roughness": roughness_tangent,
        "single_tangent_sld_imag": sld_tangent.imag,
        "single_tangent_sld_real": sld_tangent.real,
        "single_tangent_thickness": thickness_tangent,
    }
    summary = {"case": "single-layer", "parameter_count": parameter_count, "primal_shape": list(primal.shape), "jacobian_shape": list(jacobian.shape)}
    return summary, arrays


def replay(context: object) -> dict[str, object]:
    """Build exact normalized physics artifacts from hash-bound projects."""
    contents = _validate_context(context)
    q_config = CONFIGURATION["q_grid"]
    theta_config = CONFIGURATION["theta_grid"]
    q_grid = np.linspace(q_config["start"], q_config["stop"], q_config["count"], dtype=np.float64)
    theta_grid = np.linspace(theta_config["start"], theta_config["stop"], theta_config["count"], dtype=np.float64)
    arrays: dict[str, np.ndarray] = {"q_grid": q_grid, "theta_grid": theta_grid}
    cases: dict[str, object] = {}
    stacks: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GaussHermiteConvergenceWarning)
        for stem, prefix in (("single-layer", "single"), ("mo-si-periodic", "mo_si")):
            summary, values, stack = _case_arrays(stem, prefix, contents[f"{stem}-project"], q_grid, theta_grid)
            cases[stem] = summary
            arrays.update(values)
            stacks[stem] = stack
    derivative, derivative_arrays = _derivative_arrays(q_grid, stacks["single-layer"])
    arrays.update(derivative_arrays)
    summary = {"schema": "xrr-r22-physics-reference-v1", "cases": cases, "derivative_case": derivative}
    return {ARTIFACTS[0]: summary, ARTIFACTS[1]: arrays}
