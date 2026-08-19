"""Internal instrument jacobian implementation."""

from __future__ import annotations

from functools import partial

import numpy as np

from xrr_fitter.evaluation_geometry import (
    _active_upper_tangent as _active_upper_tangent,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_cap_jacobians as _add_latent_roughness_cap_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_caps as _add_latent_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _allowed_missing_roughness_names as _allowed_missing_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _definition_name as _definition_name,
)
from xrr_fitter.evaluation_geometry import (
    _expand_structure_with_jacobian as _expand_structure_with_jacobian,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_caps as _fill_missing_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_jacobians as _fill_missing_roughness_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _gradient_slab_counts as _gradient_slab_counts,
)
from xrr_fitter.evaluation_geometry import (
    _interface_neighbor_indices as _interface_neighbor_indices,
)
from xrr_fitter.evaluation_geometry import (
    _interface_upper as _interface_upper,
)
from xrr_fitter.evaluation_geometry import (
    _is_public_interface as _is_public_interface,
)
from xrr_fitter.evaluation_geometry import (
    _is_roughness_definition as _is_roughness_definition,
)
from xrr_fitter.evaluation_geometry import (
    _latent_periodic_roughness_names as _latent_periodic_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _record_active_upper as _record_active_upper,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definition_map as _roughness_definition_map,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definitions as _roughness_definitions,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_upper_jacobians as _roughness_dynamic_upper_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_uppers as _roughness_dynamic_uppers,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_geometry_context as _roughness_geometry_context,
)
from xrr_fitter.evaluation_geometry import (
    _zero_roughness_values as _zero_roughness_values,
)
from xrr_fitter.evaluation_model import _masked_optional, _point_resolution_for_wavelength, _primary_wavelength
from xrr_fitter.evaluation_objective import log_residuals
from xrr_fitter.evaluation_parameters import EvaluationConstraintError, values_and_jacobians
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.parameters import (
    PhysicalValueError,
)
from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.model.structure import (
    ExpandedSlabLimitError,
)
from xrr_fitter.physics.derivatives import (
    parratt_reflectivity_jacobian,
    smear_with_widths_jacobian,
)
from xrr_fitter.physics.geometry import DifferentiableStack
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.reflectivity import (
    _background as _sampled_background,
)
from xrr_fitter.physics.reflectivity import (
    _mixed_kalpha_average,
    _powerlaw_term,
    qz_from_theta_deg,
)


def _qz_and_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    wavelength_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert incident angle and its forward tangents to wavelength-specific q.

    Degree-to-radian scaling participates in the derivative explicitly. The
    trailing parameter axis is broadcast against the leading sample shape and
    remains aligned with the flattened physics query.
    """
    try:
        qz = qz_from_theta_deg(theta_deg, wavelength_a)
    except ValueError as error:
        raise FloatingPointError("nonfinite qz conversion") from error
    theta_rad = np.deg2rad(theta_deg)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scale = 4.0 * np.pi / wavelength_a
            # The derivative of sine is evaluated in radians while the source
            # tangent remains per degree, hence the explicit pi/180 factor.
            qz_jacobian = scale * np.cos(theta_rad)[:, None] * (np.pi / 180.0) * theta_jacobian
    except FloatingPointError:
        qz_jacobian = np.full_like(theta_jacobian, np.nan)
    unstable = ~np.isfinite(qz_jacobian)
    if np.any(unstable):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            stable = 4.0 * np.pi * (np.pi / 180.0) * np.cos(theta_rad)[:, None] * theta_jacobian / wavelength_a
        qz_jacobian[unstable] = stable[unstable]
    if np.any(~np.isfinite(qz_jacobian)):
        raise FloatingPointError("nonfinite qz Jacobian")
    return qz, qz_jacobian


def _point_resolution_with_jacobian(
    problem: object,
    angle_offset_deg: float,
    angle_jacobian: np.ndarray,
    wavelength_a: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-row q widths and their angle-offset forward tangents.

    Imported q widths have zero candidate tangent. Angular widths are first
    normalized from FWHM when needed, then differentiated through the absolute
    cosine conversion used by the primal resolution preparation.
    """
    point_resolution = _point_resolution_for_wavelength(
        problem,
        angle_offset_deg,
        wavelength_a,
    )
    if point_resolution is None:
        return None, None
    point_jacobian = np.zeros(
        point_resolution.shape + (len(problem.variables),),
        dtype=float,
    )
    kind = problem.data.column_mapping.resolution_kind
    if kind in {"sigma_q_a_inv", "fwhm_q_a_inv"}:
        return point_resolution, point_jacobian
    raw = np.asarray(problem.data.resolution_raw, dtype=float)
    if kind.startswith("fwhm"):
        raw = raw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_theta_rad = np.deg2rad(raw / 2.0)
    theta_rad = np.deg2rad(problem.data.two_theta_deg / 2.0 + angle_offset_deg)
    cosine = np.cos(theta_rad)
    # The primal converter uses ``abs(cos(theta))``; its local derivative is
    # represented by sign(cosine) away from the exact cusp.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        derivative = (
            -(4.0 * np.pi / wavelength_a) * np.sign(cosine) * np.sin(theta_rad) * (np.pi / 180.0) * sigma_theta_rad
        )
        unstable = ~np.isfinite(derivative)
        if np.any(unstable):
            derivative[unstable] = (
                -4.0
                * np.pi
                * np.sign(cosine[unstable])
                * np.sin(theta_rad[unstable])
                * (np.pi / 180.0)
                * sigma_theta_rad[unstable]
                / wavelength_a
            )
        jacobian = derivative[:, None] * angle_jacobian[None, :]
    if np.any(~np.isfinite(derivative)) or np.any(~np.isfinite(jacobian)):
        raise FloatingPointError("nonfinite point-resolution Jacobian")
    return point_resolution, jacobian


def _value_or_zeros(value: np.ndarray | None, template: np.ndarray) -> np.ndarray:
    """Materialize an absent optional tangent with a shape-matched zero array.

    Callers may then use one algebraic expression for imported and absent point
    resolution without introducing a second derivative path.
    """
    return np.zeros_like(template) if value is None else value


def _resolution_width_jacobian(
    qz: np.ndarray,
    qz_jacobian: np.ndarray,
    relative_sigma: float,
    relative_sigma_jacobian: np.ndarray,
    absolute_sigma: float,
    absolute_sigma_jacobian: np.ndarray,
    point_sigma: np.ndarray | None,
    point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine relative, absolute, and imported q-width terms analytically.

    The primal width is the square root of summed variances. Its tangent uses
    the same terms before dividing only positive widths; an exact zero width
    retains an exact zero derivative and bypasses division.
    """
    point = _value_or_zeros(point_sigma, qz)
    point_jacobian = _value_or_zeros(point_sigma_jacobian, qz_jacobian)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            relative_term = relative_sigma * qz
            relative_jacobian = qz[:, None] * relative_sigma_jacobian[None, :] + relative_sigma * qz_jacobian
    except (FloatingPointError, OverflowError) as error:
        raise FloatingPointError("nonfinite resolution width Jacobian") from error
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            variance = relative_term**2 + absolute_sigma**2 + point**2
            variance_jacobian = (
                2.0 * relative_term[:, None] * relative_jacobian
                + 2.0 * absolute_sigma * absolute_sigma_jacobian[None, :]
                + 2.0 * point[:, None] * point_jacobian
            )
            # Zero total variance is a legitimate disabled-resolution state.
            # Its tangent is defined as zero instead of evaluating a 0/0 quotient.
            widths = np.sqrt(variance)
            width_jacobian = np.zeros_like(variance_jacobian)
            positive = widths > 0.0
            width_jacobian[positive] = variance_jacobian[positive] / (2.0 * widths[positive, None])
    except (FloatingPointError, OverflowError):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                widths = np.hypot(np.hypot(relative_term, absolute_sigma), point)
                width_jacobian = np.zeros_like(relative_jacobian)
                positive = widths > 0.0
                relative_weight = relative_term[positive] / widths[positive]
                absolute_weight = absolute_sigma / widths[positive]
                point_weight = point[positive] / widths[positive]
                width_jacobian[positive] = (
                    relative_weight[:, None] * relative_jacobian[positive]
                    + absolute_weight[:, None] * absolute_sigma_jacobian[None, :]
                    + point_weight[:, None] * point_jacobian[positive]
                )
        except (FloatingPointError, OverflowError) as error:
            raise FloatingPointError("nonfinite resolution width Jacobian") from error
    if np.any(~np.isfinite(widths)) or np.any(~np.isfinite(width_jacobian)):
        raise FloatingPointError("nonfinite resolution width Jacobian")
    return widths, width_jacobian


def _reflectivity_with_jacobian(
    differentiable: DifferentiableStack,
    query_qz: np.ndarray,
    query_qz_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bind a differentiable stack to the low-level Parratt tangent kernel.

    Values and every physical tangent array are forwarded together so periodic
    acceleration and branch conventions remain owned by physics.
    """
    return parratt_reflectivity_jacobian(
        query_qz,
        differentiable.stack,
        query_qz_jacobian,
        differentiable.thickness_jacobian,
        differentiable.sld_jacobian,
        differentiable.roughness_jacobian,
    )


def _theta_reflectivity_with_jacobian(
    differentiable: DifferentiableStack,
    wavelength_a: float,
    query_theta: np.ndarray,
    query_theta_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate arbitrary-shaped theta quadrature nodes and restore their shape.

    Smearing supplies a sample-by-node query grid. Flattening keeps Parratt's
    vector interface simple, while reshape restores both node axes and the
    trailing parameter dimension without reordering observations.
    """
    flat_theta = query_theta.ravel()
    flat_jacobian = query_theta_jacobian.reshape(
        -1,
        query_theta_jacobian.shape[-1],
    )
    qz, qz_jacobian = _qz_and_jacobian(
        flat_theta,
        flat_jacobian,
        wavelength_a,
    )
    result, jacobian = _reflectivity_with_jacobian(
        differentiable,
        qz,
        qz_jacobian,
    )
    return (
        result.reshape(query_theta.shape),
        jacobian.reshape(query_theta.shape + (flat_jacobian.shape[1],)),
    )


def _primal_theta_reflectivity(
    stack: SlabStack,
    wavelength_a: float,
    query_theta: np.ndarray,
) -> np.ndarray:
    """Provide the exact primal callback paired with theta-domain tangents.

    Adaptive smearing compares quadrature orders using this value-only function;
    it must therefore share q conversion and Parratt ordering with the tangent
    callback rather than deriving values from Jacobian output.
    """
    try:
        query_qz = qz_from_theta_deg(query_theta, wavelength_a)
    except ValueError as error:
        raise FloatingPointError("nonfinite theta-domain qz conversion") from error
    return parratt_reflectivity(query_qz, stack)


def _single_wavelength_smeared_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    differentiable: DifferentiableStack,
    wavelength_a: float,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    point_sigma: np.ndarray | None,
    point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate adaptive resolution smearing for one physical wavelength.

    Theta-domain resolution smears incident angles with the scalar instrument
    width. Q-domain resolution first converts candidate angles, combines all
    width sources, and smears q. Both paths pass paired primal callbacks so
    convergence decisions match ordinary model evaluation.
    """
    if problem.instrument.resolution_domain == "theta":
        sigma = values["instrument.sigma_theta_deg"]
        widths = np.full(theta_deg.shape, sigma, dtype=float)
        width_jacobian = np.broadcast_to(
            value_jacobians["instrument.sigma_theta_deg"],
            theta_jacobian.shape,
        ).copy()
        return smear_with_widths_jacobian(
            theta_deg,
            theta_jacobian,
            widths,
            width_jacobian,
            partial(
                _theta_reflectivity_with_jacobian,
                differentiable,
                wavelength_a,
            ),
            primal_function=partial(
                _primal_theta_reflectivity,
                differentiable.stack,
                wavelength_a,
            ),
        )
    if problem.instrument.resolution_domain != "q":
        raise ValueError("resolution_domain must be q or theta")
    qz, qz_jacobian = _qz_and_jacobian(
        theta_deg,
        theta_jacobian,
        wavelength_a,
    )
    widths, width_jacobian = _resolution_width_jacobian(
        qz,
        qz_jacobian,
        values["instrument.relative_sigma"],
        value_jacobians["instrument.relative_sigma"],
        values["instrument.absolute_sigma_a_inv"],
        value_jacobians["instrument.absolute_sigma_a_inv"],
        point_sigma,
        point_sigma_jacobian,
    )
    return smear_with_widths_jacobian(
        qz,
        qz_jacobian,
        widths,
        width_jacobian,
        partial(_reflectivity_with_jacobian, differentiable),
        primal_function=partial(parratt_reflectivity, stack=differentiable.stack),
    )


def _smeared_beam_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    primary_stack: DifferentiableStack,
    secondary_stack: DifferentiableStack | None,
    primary_point_sigma: np.ndarray | None,
    primary_point_sigma_jacobian: np.ndarray | None,
    secondary_point_sigma: np.ndarray | None,
    secondary_point_sigma_jacobian: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine primary and optional K-alpha-2 smeared values and tangents.

    Each wavelength owns its expanded SLD stack and point-width conversion.
    Mixed-beam fractions are fixed beam metadata, so the weighted derivative is
    the same linear combination as the primal reflectivity.
    """
    beam = problem.data.beam
    smeared, smeared_jacobian = _single_wavelength_smeared_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        primary_stack,
        _primary_wavelength(problem),
        values,
        value_jacobians,
        primary_point_sigma,
        primary_point_sigma_jacobian,
    )
    if beam.kind != "mixed_kalpha":
        return smeared, smeared_jacobian
    if secondary_stack is None:
        raise ValueError("secondary stack is required for mixed_kalpha")
    # Beam fractions are fixed acquisition metadata rather than fit coordinates.
    # Apply the identical linear mixture to primal values and every tangent row.
    secondary, secondary_jacobian = _single_wavelength_smeared_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        secondary_stack,
        beam.wavelength_2_a,
        values,
        value_jacobians,
        secondary_point_sigma,
        secondary_point_sigma_jacobian,
    )
    ratio = beam.intensity_ratio_21
    return (
        _mixed_kalpha_average(smeared, secondary, ratio),
        _mixed_kalpha_average(smeared_jacobian, secondary_jacobian, ratio),
    )


def _footprint_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    spill_angle_deg: float,
    spill_angle_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the capped sine-ratio footprint transmission.

    Zero spill angle disables footprint loss exactly. Below saturation the
    quotient rule includes candidate angle and spill-angle tangents; at or above
    one, the capped transmission has zero local derivative.
    """
    if spill_angle_deg == 0.0:
        return (
            np.ones(theta_deg.size, dtype=float),
            np.zeros_like(theta_jacobian),
        )
    theta_rad = np.deg2rad(theta_deg)
    spill_rad = np.deg2rad(spill_angle_deg)
    numerator = np.sin(theta_rad)
    denominator = np.sin(spill_rad)
    # ``minimum`` creates a saturated branch whose derivative is exactly zero.
    # Strictly sub-unity points retain the full sine-ratio quotient tangent.
    footprint = np.ones(theta_deg.size, dtype=float)
    jacobian = np.zeros_like(theta_jacobian)
    active = numerator < denominator
    if np.any(active):
        numerator_jacobian = np.cos(theta_rad[active])[:, None] * (np.pi / 180.0) * theta_jacobian[active]
        denominator_jacobian = np.cos(spill_rad) * (np.pi / 180.0) * spill_angle_jacobian
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                footprint[active] = numerator[active] / denominator
                jacobian[active] = (
                    numerator_jacobian * denominator - numerator[active, None] * denominator_jacobian[None, :]
                ) / denominator**2
        except FloatingPointError:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    footprint[active] = numerator[active] / denominator
                    jacobian[active] = (
                        numerator_jacobian / denominator
                        - footprint[active, None] * (denominator_jacobian / denominator)[None, :]
                    )
            except FloatingPointError as error:
                raise FloatingPointError("nonfinite footprint Jacobian") from error
    if np.any(~np.isfinite(footprint)) or np.any(~np.isfinite(jacobian)):
        raise FloatingPointError("nonfinite footprint Jacobian")
    return footprint, jacobian


def _scaled_signal_jacobian(
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    smeared: np.ndarray,
    smeared_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scale and footprint to smeared reflectivity with the product rule.

    Scale multiplies both the optical signal and footprint. Background remains
    outside this helper to preserve the frozen instrument order.
    """
    footprint, footprint_jacobian = _footprint_jacobian(
        theta_deg,
        theta_jacobian,
        values["instrument.footprint_spill_angle_deg"],
        value_jacobians["instrument.footprint_spill_angle_deg"],
    )
    scale = values["instrument.scale"]
    scale_jacobian = value_jacobians["instrument.scale"]
    # Keep multiplication grouping aligned with the primal instrument model.
    # This order also fixes the frozen floating-point derivative artifact.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            signal = scale * smeared * footprint
            signal_jacobian = (smeared * footprint)[:, None] * scale_jacobian[None, :] + scale * (
                footprint[:, None] * smeared_jacobian + smeared[:, None] * footprint_jacobian
            )
    except FloatingPointError as error:
        raise FloatingPointError("nonfinite scaled signal Jacobian") from error
    if np.any(~np.isfinite(signal)) or np.any(~np.isfinite(signal_jacobian)):
        raise FloatingPointError("nonfinite scaled signal Jacobian")
    return signal, signal_jacobian


def _scaled_powerlaw_jacobian(
    qz: np.ndarray,
    exponent: float,
    coefficient_jacobian: np.ndarray,
) -> np.ndarray:
    """Multiply power-law values by finite coefficients without overflow first."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        result = qz[:, None] ** (-exponent) * coefficient_jacobian[None, :]
    zero_columns = coefficient_jacobian == 0.0
    result[:, zero_columns] = 0.0
    unstable = ~np.isfinite(result)
    if np.any(unstable):
        q_indices, parameter_indices = np.nonzero(unstable)
        coefficients = coefficient_jacobian[parameter_indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            result[unstable] = np.sign(coefficients) * np.exp(
                np.log(np.abs(coefficients)) - exponent * np.log(qz[q_indices])
            )
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("nonfinite power-law amplitude Jacobian")
    return result


def _background_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate constant, linear-q, and optional power-law background.

    Q is recomputed at the beam's effective wavelength and carries angle-offset
    tangents. A disabled zero-amplitude power law returns before evaluating
    logarithms or negative powers, matching the primal inactive mode.
    """
    qz, qz_jacobian = _qz_and_jacobian(
        theta_deg,
        theta_jacobian,
        problem.data.beam.effective_wavelength_a,
    )
    background = values["instrument.background"]
    linear = values["instrument.linear_background_per_a_inv"]
    amplitude = values["instrument.powerlaw_background_amplitude"]
    exponent = values["instrument.powerlaw_background_exponent"]
    try:
        sampled = _sampled_background(qz, background, linear, amplitude, exponent)
    except ValueError as error:
        raise EvaluationConstraintError("constraint_violation:ValueError") from error
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            sampled_jacobian = (
                value_jacobians["instrument.background"][None, :]
                + qz[:, None] * value_jacobians["instrument.linear_background_per_a_inv"][None, :]
                + linear * qz_jacobian
            )
    except FloatingPointError as error:
        raise EvaluationConstraintError("constraint_violation:FloatingPointError") from error
    if np.any(~np.isfinite(sampled_jacobian)):
        raise EvaluationConstraintError("constraint_violation:FloatingPointError")
    amplitude_jacobian = value_jacobians["instrument.powerlaw_background_amplitude"]
    inactive = (amplitude <= 0.0, np.all(amplitude_jacobian == 0.0)) == (True, True)
    if inactive:
        return sampled, sampled_jacobian
    # Power-law differentiation includes both exponent and q-angle dependence.
    # This branch is skipped entirely when amplitude and its tangent are zero.
    exponent_jacobian = value_jacobians["instrument.powerlaw_background_exponent"]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if np.any(qz <= 0.0):
                raise FloatingPointError("power-law background requires positive qz")
            amplitude_term = _scaled_powerlaw_jacobian(
                qz,
                exponent,
                amplitude_jacobian,
            )
            if amplitude == 0.0:
                # Avoid manufacturing 0*inf in the inactive physical branch;
                # only the amplitude axis has a nonzero local tangent there.
                result_jacobian = sampled_jacobian + amplitude_term
            else:
                power_term = _powerlaw_term(qz, amplitude, exponent)
                power_jacobian = power_term[:, None] * (
                    -np.log(qz)[:, None] * exponent_jacobian[None, :] - exponent * qz_jacobian / qz[:, None]
                )
                result_jacobian = sampled_jacobian + amplitude_term + power_jacobian
    except FloatingPointError as error:
        raise EvaluationConstraintError("constraint_violation:FloatingPointError") from error
    if np.any(~np.isfinite(result_jacobian)):
        raise EvaluationConstraintError("constraint_violation:FloatingPointError")
    return sampled, result_jacobian


def _instrument_model_jacobian(
    problem: object,
    theta_deg: np.ndarray,
    theta_jacobian: np.ndarray,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
    primary_stack: DifferentiableStack,
    secondary_stack: DifferentiableStack | None,
    point_inputs: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose smearing, scale-footprint signal, and background derivatives.

    This order mirrors ``instrument_reflectivity`` exactly: wavelength-specific
    optics and resolution first, then beam mixing, scale times footprint, and
    finally additive background on the effective q grid.
    """
    smeared, smeared_jacobian = _smeared_beam_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        values,
        value_jacobians,
        primary_stack,
        secondary_stack,
        *point_inputs,
    )
    signal, signal_jacobian = _scaled_signal_jacobian(
        theta_deg,
        theta_jacobian,
        smeared,
        smeared_jacobian,
        values,
        value_jacobians,
    )
    background, background_jacobian = _background_jacobian(
        problem,
        theta_deg,
        theta_jacobian,
        values,
        value_jacobians,
    )
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            model = signal + background
            model_jacobian = signal_jacobian + background_jacobian
    except FloatingPointError as error:
        raise FloatingPointError("nonfinite instrument model Jacobian") from error
    if np.any(~np.isfinite(model)) or np.any(~np.isfinite(model_jacobian)):
        raise FloatingPointError("nonfinite instrument model Jacobian")
    return model, model_jacobian


def _differentiable_stacks(
    problem: object,
    values: dict[str, float],
    value_jacobians: dict[str, np.ndarray],
) -> tuple[float, DifferentiableStack, DifferentiableStack | None]:
    """Build primary and optional secondary tangent stacks from one value map.

    Separate expansion is required because material SLD depends on wavelength.
    Geometry and its unit tangents remain shared between the two expansions.
    """
    primary_wavelength = _primary_wavelength(problem)
    primary = _expand_structure_with_jacobian(
        problem,
        values,
        value_jacobians,
        primary_wavelength,
    )
    secondary = (
        _expand_structure_with_jacobian(
            problem,
            values,
            value_jacobians,
            problem.data.beam.wavelength_2_a,
        )
        if problem.data.beam.kind == "mixed_kalpha"
        else None
    )
    return primary_wavelength, primary, secondary


def _model_residual_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate fitted residuals and their analytic Jacobian in one traversal.

    Values and coordinate tangents are decoded once; this function does not call
    the primal evaluator. Positive-angle rows are differentiated through stack,
    resolution, beam, footprint, scale, and background operations, then inserted
    into full source order. The returned model values supply the residual from the
    same physical traversal that supplied the Jacobian.
    """
    try:
        values, value_jacobians = values_and_jacobians(problem, unit_vector)
        primary_wavelength, primary_stack, secondary_stack = _differentiable_stacks(
            problem,
            values,
            value_jacobians,
        )
    except (ExpandedSlabLimitError, PhysicalValueError) as error:
        raise EvaluationConstraintError(f"constraint_violation:{type(error).__name__}") from error
    theta = problem.data.two_theta_deg / 2.0 + values["instrument.angle_offset_deg"]
    model_mask = np.isfinite(theta) & (theta > 0.0) & (theta <= 90.0)
    if np.any(problem.data.fit_mask & ~model_mask):
        raise ValueError("cannot differentiate nonpositive fitted angle")
    theta_model = theta[model_mask]
    angle_jacobian = value_jacobians["instrument.angle_offset_deg"]
    theta_jacobian = np.broadcast_to(
        angle_jacobian,
        (theta_model.size, angle_jacobian.size),
    ).copy()
    primary_sigma, primary_sigma_jacobian = _point_resolution_with_jacobian(
        problem,
        values["instrument.angle_offset_deg"],
        angle_jacobian,
        primary_wavelength,
    )
    if problem.data.beam.kind == "mixed_kalpha":
        secondary_sigma, secondary_sigma_jacobian = _point_resolution_with_jacobian(
            problem,
            values["instrument.angle_offset_deg"],
            angle_jacobian,
            problem.data.beam.wavelength_2_a,
        )
    else:
        secondary_sigma, secondary_sigma_jacobian = None, None
    point_inputs = (
        _masked_optional(primary_sigma, model_mask),
        _masked_optional(primary_sigma_jacobian, model_mask),
        _masked_optional(secondary_sigma, model_mask),
        _masked_optional(secondary_sigma_jacobian, model_mask),
    )
    model, model_jacobian = _instrument_model_jacobian(
        problem,
        theta_model,
        theta_jacobian,
        values,
        value_jacobians,
        primary_stack,
        secondary_stack,
        point_inputs,
    )
    full_model = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    full_model[model_mask] = model
    full_jacobian = np.zeros(
        (problem.data.qz_a_inv.size, len(problem.variables)),
        dtype=float,
    )
    full_jacobian[model_mask] = model_jacobian
    fit_model = full_model[problem.data.fit_mask]
    residual = log_residuals(
        fit_model,
        problem.data.intensity_normalized[problem.data.fit_mask],
        problem.data.r_floor,
    )
    # d(log10(model + floor)) = d(model) / ((model + floor) * ln(10)).
    # Observations are constant, so they contribute no residual tangent.
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_jacobian = full_jacobian[problem.data.fit_mask] / (
            (fit_model + problem.data.r_floor)[:, None] * np.log(10.0)
        )
    finite_values = np.concatenate(
        (
            model.ravel(),
            model_jacobian.ravel(),
            fit_model,
            residual,
            residual_jacobian.ravel(),
        )
    )
    if np.any(~np.isfinite(finite_values)):
        raise FloatingPointError("nonfinite analytic model or residual Jacobian")
    return (
        np.array(residual, dtype=float, copy=True),
        np.array(residual_jacobian, dtype=float, copy=True),
        float(values["instrument.scale"]),
    )


def evaluate_model_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    """Return the analytic Jacobian of unweighted fitted log residuals."""
    _residual, jacobian, _scale = _model_residual_jacobian(problem, unit_vector)
    # Defensive copying prevents solver or analysis code from mutating a
    # derivative snapshot that may be shared with candidate publication.
    result = np.array(jacobian, dtype=float, copy=True)
    result.setflags(write=False)
    return result
