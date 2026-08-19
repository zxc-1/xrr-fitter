"""Shared architecture policy for the evaluation facade and implementations."""

EVALUATION_FACADE_MODULE = "evaluation"
EVALUATION_IMPLEMENTATION_MODULES = {
    "evaluation_geometry",
    "evaluation_instrument_jacobian",
    "evaluation_model",
    "evaluation_objective",
    "evaluation_parameters",
    "evaluation_priors",
    "evaluation_solver",
}
EVALUATION_BOUNDARY_MODULES = EVALUATION_IMPLEMENTATION_MODULES | {EVALUATION_FACADE_MODULE}

__all__ = [
    "EVALUATION_BOUNDARY_MODULES",
    "EVALUATION_FACADE_MODULE",
    "EVALUATION_IMPLEMENTATION_MODULES",
]
