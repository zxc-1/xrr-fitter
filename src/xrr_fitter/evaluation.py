"""Stable facade for all shared numerical evaluation primitives."""

from __future__ import annotations

from functools import wraps

import xrr_fitter.evaluation_instrument_jacobian as _jacobian_impl
import xrr_fitter.evaluation_model as _model_impl
import xrr_fitter.evaluation_parameters as _parameters_impl
import xrr_fitter.evaluation_priors as _priors_impl
import xrr_fitter.evaluation_solver as _solver_impl
from xrr_fitter.evaluation_geometry import _fill_missing_roughness_caps as _fill_missing_roughness_caps
from xrr_fitter.evaluation_geometry import _gradient_slab_counts as _gradient_slab_counts
from xrr_fitter.evaluation_instrument_jacobian import (
    _background_jacobian as _background_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _differentiable_stacks as _differentiable_stacks,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _footprint_jacobian as _footprint_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _instrument_model_jacobian as _instrument_model_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _model_residual_jacobian as _model_residual_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _point_resolution_with_jacobian as _point_resolution_with_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _primal_theta_reflectivity as _primal_theta_reflectivity,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _qz_and_jacobian as _qz_and_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _reflectivity_with_jacobian as _reflectivity_with_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _resolution_width_jacobian as _resolution_width_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _scaled_powerlaw_jacobian as _scaled_powerlaw_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _scaled_signal_jacobian as _scaled_signal_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _single_wavelength_smeared_jacobian as _single_wavelength_smeared_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _smeared_beam_jacobian as _smeared_beam_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _theta_reflectivity_with_jacobian as _theta_reflectivity_with_jacobian,
)
from xrr_fitter.evaluation_instrument_jacobian import (
    _value_or_zeros as _value_or_zeros,
)
from xrr_fitter.evaluation_model import (
    _angle_layout as _angle_layout,
)
from xrr_fitter.evaluation_model import (
    _append_ideal_reflectivity_diagnostic as _append_ideal_reflectivity_diagnostic,
)
from xrr_fitter.evaluation_model import (
    _append_nevot_croce_diagnostic as _append_nevot_croce_diagnostic,
)
from xrr_fitter.evaluation_model import (
    _append_stack_diagnostics as _append_stack_diagnostics,
)
from xrr_fitter.evaluation_model import (
    _expanded_stacks as _expanded_stacks,
)
from xrr_fitter.evaluation_model import (
    _index_in_range as _index_in_range,
)
from xrr_fitter.evaluation_model import (
    _instrument_values as _instrument_values,
)
from xrr_fitter.evaluation_model import (
    _is_integer_index as _is_integer_index,
)
from xrr_fitter.evaluation_model import (
    _masked_optional as _masked_optional,
)
from xrr_fitter.evaluation_model import (
    _model_evaluation as _model_evaluation,
)
from xrr_fitter.evaluation_model import (
    _modeled_reflectivity as _modeled_reflectivity,
)
from xrr_fitter.evaluation_model import (
    _nevot_croce_affected as _nevot_croce_affected,
)
from xrr_fitter.evaluation_model import (
    _parameter_pair as _parameter_pair,
)
from xrr_fitter.evaluation_model import (
    _parameter_values as _parameter_values,
)
from xrr_fitter.evaluation_model import (
    _point_resolution_for_wavelength as _point_resolution_for_wavelength,
)
from xrr_fitter.evaluation_model import (
    _primary_wavelength as _primary_wavelength,
)
from xrr_fitter.evaluation_model import (
    _record_physics_diagnostic as _record_physics_diagnostic,
)
from xrr_fitter.evaluation_model import (
    evaluate_model as evaluate_model,
)
from xrr_fitter.evaluation_objective import (
    _float_vectors as _float_vectors,
)
from xrr_fitter.evaluation_objective import (
    _validated_qz as _validated_qz,
)
from xrr_fitter.evaluation_objective import (
    assign_fit_regions as assign_fit_regions,
)
from xrr_fitter.evaluation_objective import (
    log_residuals as log_residuals,
)
from xrr_fitter.evaluation_objective import (
    region_weights as region_weights,
)
from xrr_fitter.evaluation_objective import (
    robust_log_cost as robust_log_cost,
)
from xrr_fitter.evaluation_objective import (
    scale_prior_penalty as scale_prior_penalty,
)
from xrr_fitter.evaluation_parameters import (
    EvaluationConstraintError as EvaluationConstraintError,
)
from xrr_fitter.evaluation_parameters import (
    _apply_constraint_jacobians as _apply_constraint_jacobians,
)
from xrr_fitter.evaluation_parameters import (
    _apply_constraint_values as _apply_constraint_values,
)
from xrr_fitter.evaluation_parameters import (
    _declared_values as _declared_values,
)
from xrr_fitter.evaluation_parameters import (
    _decode_nonrough_values as _decode_nonrough_values,
)
from xrr_fitter.evaluation_parameters import (
    _initial_parameter_pair as _initial_parameter_pair,
)
from xrr_fitter.evaluation_parameters import (
    _physical_parameter_pair as _physical_parameter_pair,
)
from xrr_fitter.evaluation_parameters import (
    _readonly_vector as _readonly_vector,
)
from xrr_fitter.evaluation_parameters import (
    _roughness_value_jacobian as _roughness_value_jacobian,
)
from xrr_fitter.evaluation_parameters import (
    _unit_derivative as _unit_derivative,
)
from xrr_fitter.evaluation_parameters import (
    _validated_unit as _validated_unit,
)
from xrr_fitter.evaluation_parameters import (
    _zero_jacobian_pair as _zero_jacobian_pair,
)
from xrr_fitter.evaluation_parameters import (
    encode_physical_vector as encode_physical_vector,
)
from xrr_fitter.evaluation_parameters import (
    roughness_dynamic_uppers as roughness_dynamic_uppers,
)
from xrr_fitter.evaluation_priors import (
    _lognormal_grid_nodes as _lognormal_grid_nodes,
)
from xrr_fitter.evaluation_priors import (
    _lognormal_log_density as _lognormal_log_density,
)
from xrr_fitter.evaluation_priors import (
    _lognormal_support_grid as _lognormal_support_grid,
)
from xrr_fitter.evaluation_priors import (
    _normal_grid_nodes as _normal_grid_nodes,
)
from xrr_fitter.evaluation_priors import (
    _normal_log_density as _normal_log_density,
)
from xrr_fitter.evaluation_priors import (
    _prior_base_grid as _prior_base_grid,
)
from xrr_fitter.evaluation_priors import (
    _prior_coordinate as _prior_coordinate,
)
from xrr_fitter.evaluation_priors import (
    _prior_grid as _prior_grid,
)
from xrr_fitter.evaluation_priors import (
    _prior_norm as _prior_norm,
)
from xrr_fitter.evaluation_priors import (
    _soft_range_grid_nodes as _soft_range_grid_nodes,
)
from xrr_fitter.evaluation_priors import (
    _soft_range_log_density as _soft_range_log_density,
)
from xrr_fitter.evaluation_priors import (
    _uniform_cdf as _uniform_cdf,
)
from xrr_fitter.evaluation_priors import (
    _uniform_log_density as _uniform_log_density,
)
from xrr_fitter.evaluation_priors import (
    _uniform_log_width as _uniform_log_width,
)
from xrr_fitter.evaluation_priors import (
    _uniform_quantile as _uniform_quantile,
)
from xrr_fitter.evaluation_priors import (
    _unnormalized_density as _unnormalized_density,
)
from xrr_fitter.evaluation_priors import (
    prior_bounds as prior_bounds,
)
from xrr_fitter.evaluation_priors import (
    prior_cdf as prior_cdf,
)
from xrr_fitter.evaluation_priors import (
    prior_center_and_spread as prior_center_and_spread,
)
from xrr_fitter.evaluation_priors import (
    prior_inverse_cdf as prior_inverse_cdf,
)
from xrr_fitter.evaluation_priors import (
    prior_log_density as prior_log_density,
)
from xrr_fitter.evaluation_solver import (
    _empty_residual_jacobian as _empty_residual_jacobian,
)
from xrr_fitter.evaluation_solver import (
    _expected_derivative_value_error as _expected_derivative_value_error,
)
from xrr_fitter.evaluation_solver import (
    _least_squares_residual_parts as _least_squares_residual_parts,
)
from xrr_fitter.evaluation_solver import (
    _least_squares_row_count as _least_squares_row_count,
)
from xrr_fitter.evaluation_solver import (
    _scale_prior_jacobian as _scale_prior_jacobian,
)
from xrr_fitter.evaluation_solver import (
    _scale_prior_residual as _scale_prior_residual,
)
from xrr_fitter.evaluation_solver import (
    _scale_prior_residual_from_scale as _scale_prior_residual_from_scale,
)
from xrr_fitter.evaluation_solver import (
    cached_least_squares_callbacks as cached_least_squares_callbacks,
)
from xrr_fitter.model.parameters import PhysicalValueError as PhysicalValueError
from xrr_fitter.model.parameters import physical_to_unit as physical_to_unit
from xrr_fitter.model.parameters import unit_to_physical as unit_to_physical
from xrr_fitter.physics.parratt import parratt_reflectivity as parratt_reflectivity
from xrr_fitter.physics.stack import expand_structure as expand_structure

ORIGINAL_QZ_AND_JACOBIAN = _qz_and_jacobian
ORIGINAL_SMEARED_BEAM_JACOBIAN = _smeared_beam_jacobian
ORIGINAL_SCALED_SIGNAL_JACOBIAN = _scaled_signal_jacobian
ORIGINAL_BACKGROUND_JACOBIAN = _background_jacobian
ORIGINAL_INSTRUMENT_MODEL_JACOBIAN = _instrument_model_jacobian
ORIGINAL_MODEL_RESIDUAL_JACOBIAN = _model_residual_jacobian
ORIGINAL_EVALUATE_MODEL = evaluate_model


def _facade_hook(name: str, original):
    current = globals()[name]
    return original if getattr(current, "__wrapped__", None) is original else current


def _sync_compatibility_hooks() -> None:
    """Propagate facade-level test seams into split implementation modules."""
    for module in (_parameters_impl, _model_impl, _jacobian_impl):
        module._fill_missing_roughness_caps = _fill_missing_roughness_caps
    _model_impl.parratt_reflectivity = parratt_reflectivity
    _jacobian_impl._single_wavelength_smeared_jacobian = _single_wavelength_smeared_jacobian
    _jacobian_impl._smeared_beam_jacobian = _facade_hook(
        "_smeared_beam_jacobian",
        ORIGINAL_SMEARED_BEAM_JACOBIAN,
    )
    _jacobian_impl._scaled_signal_jacobian = _facade_hook(
        "_scaled_signal_jacobian",
        ORIGINAL_SCALED_SIGNAL_JACOBIAN,
    )
    _jacobian_impl._background_jacobian = _facade_hook(
        "_background_jacobian",
        ORIGINAL_BACKGROUND_JACOBIAN,
    )
    _jacobian_impl._qz_and_jacobian = _facade_hook(
        "_qz_and_jacobian",
        ORIGINAL_QZ_AND_JACOBIAN,
    )
    _jacobian_impl._instrument_model_jacobian = _facade_hook(
        "_instrument_model_jacobian",
        ORIGINAL_INSTRUMENT_MODEL_JACOBIAN,
    )
    model_system = _facade_hook(
        "_model_residual_jacobian",
        ORIGINAL_MODEL_RESIDUAL_JACOBIAN,
    )
    _jacobian_impl._model_residual_jacobian = model_system
    _solver_impl._model_residual_jacobian = model_system
    _solver_impl._scale_prior_jacobian = _scale_prior_jacobian
    primal = _facade_hook("evaluate_model", ORIGINAL_EVALUATE_MODEL)
    _solver_impl.evaluate_model = primal
    _priors_impl.evaluate_model = primal


@wraps(_parameters_impl.values_and_jacobians)
def values_and_jacobians(*args, **kwargs):
    _sync_compatibility_hooks()
    return _parameters_impl.values_and_jacobians(*args, **kwargs)


@wraps(_parameters_impl.values_by_name)
def values_by_name(*args, **kwargs):
    _sync_compatibility_hooks()
    return _parameters_impl.values_by_name(*args, **kwargs)


@wraps(_model_impl.evaluate_model)
def evaluate_model(*args, **kwargs):
    _sync_compatibility_hooks()
    return _model_impl.evaluate_model(*args, **kwargs)


@wraps(_model_impl.expanded_structure_jacobian)
def expanded_structure_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _model_impl.expanded_structure_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._qz_and_jacobian)
def _qz_and_jacobian(*args, **kwargs):
    return _jacobian_impl._qz_and_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._smeared_beam_jacobian)
def _smeared_beam_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl._smeared_beam_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._scaled_signal_jacobian)
def _scaled_signal_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl._scaled_signal_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._background_jacobian)
def _background_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl._background_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._instrument_model_jacobian)
def _instrument_model_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl._instrument_model_jacobian(*args, **kwargs)


@wraps(_jacobian_impl._model_residual_jacobian)
def _model_residual_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl._model_residual_jacobian(*args, **kwargs)


@wraps(_jacobian_impl.evaluate_model_jacobian)
def evaluate_model_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _jacobian_impl.evaluate_model_jacobian(*args, **kwargs)


@wraps(_solver_impl.least_squares_system)
def least_squares_system(*args, **kwargs):
    _sync_compatibility_hooks()
    return _solver_impl.least_squares_system(*args, **kwargs)


@wraps(_solver_impl.least_squares_residual)
def least_squares_residual(*args, **kwargs):
    _sync_compatibility_hooks()
    return _solver_impl.least_squares_residual(*args, **kwargs)


@wraps(_solver_impl.least_squares_residual_jacobian)
def least_squares_residual_jacobian(*args, **kwargs):
    _sync_compatibility_hooks()
    return _solver_impl.least_squares_residual_jacobian(*args, **kwargs)


@wraps(_solver_impl.least_squares_loss)
def least_squares_loss(*args, **kwargs):
    _sync_compatibility_hooks()
    return _solver_impl.least_squares_loss(*args, **kwargs)


@wraps(_priors_impl.problem_log_probability)
def problem_log_probability(*args, **kwargs):
    _sync_compatibility_hooks()
    return _priors_impl.problem_log_probability(*args, **kwargs)


@wraps(_priors_impl._parameter_prior_log_density)
def _parameter_prior_log_density(*args, **kwargs):
    _sync_compatibility_hooks()
    return _priors_impl._parameter_prior_log_density(*args, **kwargs)
