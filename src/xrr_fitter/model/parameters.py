from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p

import numpy as np

TRANSFORMS = frozenset({"linear", "log", "roughness_fraction"})
# Reserved namespace used by compile-time generated parameter rules.  User
# dataset and constraint declarations must never claim this identifier.
RESERVED_DATASET_ID = "__drift__"

# kind -> number of scalar parameters; PRIOR_KINDS is derived so the two never
# drift. soft_range carries (low, high, std); normal/lognormal carry (loc, scale).
PRIOR_ARITY: dict[str, int] = {"uniform": 0, "normal": 2, "lognormal": 2, "soft_range": 3}
PRIOR_KINDS = frozenset(PRIOR_ARITY)
# Per kind, the parameter indices that must be strictly positive scales.
PRIOR_SCALE_INDICES: dict[str, tuple[int, ...]] = {
    "uniform": (),
    "normal": (1,),
    "lognormal": (1,),
    "soft_range": (2,),
}


def _prior_scales(kind: str, parameters: tuple[float, ...]) -> None:
    for index in PRIOR_SCALE_INDICES[kind]:
        if parameters[index] <= 0.0:
            raise ValueError(f"{kind} prior scale must be positive")


def _prior_ordering(kind: str, parameters: tuple[float, ...]) -> None:
    if kind == "soft_range" and not parameters[0] < parameters[1]:
        raise ValueError("soft_range prior requires low < high")


@dataclass(frozen=True, slots=True)
class PriorSpec:
    kind: str
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.kind not in PRIOR_KINDS:
            raise ValueError(f"unsupported prior kind: {self.kind}")
        arity = PRIOR_ARITY[self.kind]
        if len(self.parameters) != arity:
            raise ValueError(f"{self.kind} prior requires {arity} parameters")
        if not all(isfinite(value) for value in self.parameters):
            raise ValueError("prior parameters must be finite")
        _prior_scales(self.kind, self.parameters)
        _prior_ordering(self.kind, self.parameters)


def _prior_center(prior: PriorSpec) -> float | None:
    # Center in physical space, comparable to a definition's [lower, upper].
    # lognormal parameters live in log space, so its physical center is exp(loc).
    if prior.kind == "normal":
        return prior.parameters[0]
    if prior.kind == "lognormal":
        try:
            return exp(prior.parameters[0])
        except OverflowError:
            return float("inf")
    if prior.kind == "soft_range":
        return _interval_midpoint(prior.parameters[0], prior.parameters[1])
    return None


def _interval_midpoint(lower: float, upper: float) -> float:
    """Return the midpoint of two finite bounds without overflowing their sum."""
    lower, upper = float(lower), float(upper)
    span = upper - lower
    if isfinite(span):
        return lower + span / 2.0
    scale = max(abs(lower), abs(upper))
    return scale * ((lower / scale) * 0.5 + (upper / scale) * 0.5)


def _interval_half_width(lower: float, upper: float) -> float:
    """Return half of a finite interval even when its full span overflows."""
    lower, upper = float(lower), float(upper)
    span = upper - lower
    if isfinite(span):
        return span / 2.0
    scale = max(abs(lower), abs(upper))
    return scale * ((upper / scale - lower / scale) / 2.0)


def _definition_prior(
    name: str,
    transform: str,
    lower: float,
    upper: float,
    prior: PriorSpec | None,
) -> None:
    if prior is None:
        return
    if not isinstance(prior, PriorSpec):
        raise TypeError("prior must be a PriorSpec")
    center = _prior_center(prior)
    if center is None:
        return
    # roughness_fraction 先验在消费侧(evaluation._prior_coordinate)恒落在无量纲单位
    # 分数 [0,1] 上,因为物理粗糙度上界依赖此标量路径并不持有的几何快照。构造期校验
    # 必须匹配同一坐标系,否则合法分数中心会被 Å 下界误拒、Å 量级的伪中心会被误纳。
    low, high = (0.0, 1.0) if transform == "roughness_fraction" else (lower, upper)
    if not low <= center <= high:
        raise ValueError(f"{name} prior center must be within bounds")
    # A lognormal prior is valid only on a declaration whose entire support is
    # strictly positive.  Reject a zero endpoint as well as a negative one so
    # the persisted declaration cannot claim density where log(x) is undefined.
    if prior.kind == "lognormal" and low <= 0.0:
        raise ValueError(f"{name} lognormal prior requires a positive lower bound")


def _name(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _bounds(name: str, initial: float, lower: float, upper: float) -> None:
    if not all(isfinite(value) for value in (initial, lower, upper)):
        raise ValueError(f"{name} values must be finite")
    if lower > upper or not lower <= initial <= upper:
        raise ValueError(f"{name} initial value must be within bounds")


@dataclass(frozen=True, slots=True)
class ParameterSetting:
    name: str
    initial: float
    lower: float
    upper: float
    locked: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _bounds(self.name, self.initial, self.lower, self.upper)
        if not isinstance(self.locked, bool):
            raise TypeError("locked must be bool")


@dataclass(frozen=True, slots=True)
class ParameterPrior:
    """A per-parameter prior override stored separately from its declaration.

    A prior binds only a parameter ``name`` to a ``PriorSpec``; it never carries
    bounds or lock state. Keeping it apart from ``ParameterSetting`` lets a
    "restore defaults" action clear settings while leaving priors intact.
    """

    name: str
    prior: PriorSpec

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        if not isinstance(self.prior, PriorSpec):
            raise TypeError("prior must be a PriorSpec")


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    display_name: str
    unit: str
    category: str
    initial: float
    lower: float
    upper: float
    transform: str
    locked: bool
    integer: bool = False
    expert_only: bool = False
    sharing_key: str | None = None
    prior: PriorSpec | None = None
    constrained: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _name(self.display_name, "display_name")
        _name(self.category, "category")
        _bounds(self.name, self.initial, self.lower, self.upper)
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unsupported parameter transform: {self.transform}")
        if self.transform == "log" and self.lower <= 0.0:
            raise ValueError(f"log parameter bounds must be positive: {self.name}")
        if not all(
            isinstance(value, bool) for value in (self.locked, self.integer, self.expert_only, self.constrained)
        ):
            raise TypeError("parameter flags must be bool")
        if self.sharing_key is not None:
            _name(self.sharing_key, "sharing_key")
        _definition_prior(self.name, self.transform, self.lower, self.upper, self.prior)


def _effective_upper(
    definition: ParameterDefinition,
    dynamic_upper: float | None,
) -> float:
    """Allow candidate geometry to tighten, but never widen, declared bounds.

    Dynamic roughness bounds may carry a ``nextafter`` value immediately below
    a physical limit. Taking the exact minimum preserves that strict endpoint.
    """
    # Static metadata remains authoritative whenever geometry imposes no cap.
    # A supplied cap may tighten only; it cannot silently broaden the project.
    if dynamic_upper is not None:
        try:
            finite_dynamic = isfinite(dynamic_upper)
        except (TypeError, ValueError):
            finite_dynamic = False
        if isinstance(dynamic_upper, bool) or not finite_dynamic:
            raise PhysicalValueError(f"dynamic upper must be finite: {definition.name}")
        upper = min(definition.upper, float(dynamic_upper))
    else:
        upper = definition.upper
    if upper < definition.lower:
        raise PhysicalValueError(f"empty physical bounds: {definition.name}")
    return upper


class PhysicalValueError(ValueError):
    """An expected candidate value outside its compiled physical domain."""


def _log_interval_width(lower: float, upper: float) -> float:
    """Return a positive log-space interval without losing adjacent bounds.

    ``log(upper) - log(lower)`` can round to zero when two distinct floating
    point values are adjacent at very large or very small magnitudes.  The
    ``log1p`` form retains that local ratio while preserving the original
    subtraction order for ordinary intervals.
    """
    direct = log(upper) - log(lower)
    if isfinite(direct) and direct > 0.0:
        return direct
    relative = (upper - lower) / lower
    width = log1p(relative)
    if not isfinite(width) or width <= 0.0:
        raise ValueError("log bounds have no representable positive interval")
    return width


def _log10_ratio(value: float, reference: float) -> float:
    """Return ``log10(value/reference)`` while retaining adjacent positives."""
    if value == reference:
        return 0.0
    if value > reference:
        return _log_interval_width(reference, value) / log(10.0)
    return -_log_interval_width(value, reference) / log(10.0)


def _validate_physical_value(
    definition: ParameterDefinition,
    value: float,
    upper: float,
) -> None:
    """Reject invalid physical coordinates instead of clipping them.

    Clipping would make a malformed project or candidate appear to have fitted
    successfully at a boundary and would break encode/decode invertibility.
    """
    if not all((isfinite(value), value >= definition.lower, value <= upper)):
        raise PhysicalValueError(f"value outside bounds: {definition.name}")


def _log_physical_to_unit(
    definition: ParameterDefinition,
    value: float,
    upper: float,
) -> float:
    """Normalize one positive value in logarithmic physical space.

    Natural logarithms are used in both directions; their base cancels in the
    unit ratio. A dynamic upper therefore participates in the inverse exactly.
    """
    if min(definition.lower, value) <= 0.0:
        raise PhysicalValueError(f"log parameter must be positive: {definition.name}")
    if value == definition.lower:
        return 0.0
    if value == upper:
        return 1.0
    numerator = _log_interval_width(definition.lower, value)
    denominator = _log_interval_width(definition.lower, upper)
    return numerator / denominator


def _validate_unit_value(definition: ParameterDefinition, unit: float) -> None:
    """Reject nonfinite or out-of-domain solver coordinates before dispatch.

    Solvers normally honor box bounds, but imported checkpoints and direct API
    callers enter here too. No clipping or tolerance is applied at this layer.
    """
    if not all((isfinite(unit), unit >= 0.0, unit <= 1.0)):
        raise ValueError(f"unit value outside [0,1]: {definition.name}")


def _log_unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    upper: float,
) -> float:
    """Decode a logarithmic coordinate while preserving exact endpoints.

    Explicit branches at zero and one avoid exponential roundoff that could
    place a decoded value just outside its persisted physical interval.
    """
    if min(definition.lower, upper - definition.lower) <= 0.0:
        raise ValueError(f"invalid log bounds: {definition.name}")
    if unit == 0.0:
        return float(definition.lower)
    if unit == 1.0:
        return float(upper)
    span = _log_interval_width(definition.lower, upper)
    decoded = float(np.exp(log(definition.lower) + unit * span))
    return min(max(decoded, float(definition.lower)), float(upper))


def _linear_unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    upper: float,
) -> float:
    """Decode affine and geometry-tightened roughness coordinates exactly.

    Roughness shares this interpolation after its candidate-specific upper has
    been computed. Endpoint branches preserve project and checkpoint equality.
    """
    if unit == 0.0:
        return float(definition.lower)
    if unit == 1.0:
        return float(upper)
    lower = float(definition.lower)
    upper = float(upper)
    span = upper - lower
    if isfinite(span):
        return float(lower + unit * span)
    # A cross-zero interval can have finite endpoints but an unrepresentable
    # span. Normalize the endpoints first so the convex combination never
    # constructs that span as an intermediate value.
    scale = max(abs(lower), abs(upper))
    lower_scaled = lower / scale
    upper_scaled = upper / scale
    return float(scale * (lower_scaled * (1.0 - unit) + upper_scaled * unit))


def _linear_physical_to_unit(
    definition: ParameterDefinition,
    value: float,
    upper: float,
) -> float:
    """Normalize an affine value without overflowing a wide cross-zero span."""
    lower = float(definition.lower)
    upper = float(upper)
    value = float(value)
    if value == lower:
        return 0.0
    if value == upper:
        return 1.0
    span = upper - lower
    if isfinite(span):
        return (value - lower) / span
    scale = max(abs(lower), abs(upper))
    lower_scaled = lower / scale
    upper_scaled = upper / scale
    value_scaled = value / scale
    return (value_scaled - lower_scaled) / (upper_scaled - lower_scaled)


def physical_to_unit(
    definition: ParameterDefinition,
    value: float,
    dynamic_upper: float | None = None,
) -> float:
    """Encode one legal physical value in its declared unit interval.

    A zero-width effective domain has no meaningful inverse coordinate and maps
    to the canonical unit value zero. All declarations reject out-of-range
    values before dispatching to their persisted transform identifier.
    """
    upper = _effective_upper(definition, dynamic_upper)
    _validate_physical_value(definition, value, upper)
    if upper == definition.lower:
        return 0.0
    if definition.transform == "log":
        return _log_physical_to_unit(definition, value, upper)
    if definition.transform in {"linear", "roughness_fraction"}:
        return _linear_physical_to_unit(definition, value, upper)
    raise ValueError(f"unknown transform: {definition.transform}")


def unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    dynamic_upper: float | None = None,
) -> float:
    """Decode one finite unit coordinate using exact endpoint semantics.

    Dispatch is restricted to the three persisted transform identifiers.
    Unknown metadata remains an error instead of silently becoming affine.
    """
    _validate_unit_value(definition, unit)
    upper = _effective_upper(definition, dynamic_upper)
    if upper == definition.lower:
        return float(definition.lower)
    if definition.transform == "log":
        return _log_unit_to_physical(definition, unit, upper)
    if definition.transform in {"linear", "roughness_fraction"}:
        return _linear_unit_to_physical(definition, unit, upper)
    raise ValueError(f"unknown transform: {definition.transform}")


@dataclass(frozen=True, slots=True)
class ParameterCoordinate:
    parameter_index: int
    name: str
    transform: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter_index, int)
            or isinstance(self.parameter_index, bool)
            or self.parameter_index < 0
        ):
            raise ValueError("parameter_index must be a nonnegative integer")
        _name(self.name, "parameter name")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unsupported parameter transform: {self.transform}")


@dataclass(frozen=True, slots=True)
class ParameterValue:
    name: str
    value: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _bounds(self.name, self.value, self.lower, self.upper)


@dataclass(frozen=True, slots=True)
class ParameterReference:
    dataset_id: str
    parameter_name: str

    def __post_init__(self) -> None:
        _name(self.dataset_id, "dataset_id")
        _name(self.parameter_name, "parameter_name")


# Expression parameter constraints ------------------------------------------
#
# A ``ConstraintRule`` drives one target parameter from an expression over
# other parameters.  Expressions are small immutable trees of ``ConstraintNode``
# leaves (``ref`` / ``const``) joined by binary operators.  The three pure
# checks below are the *binding-time* validators; they live here (not in
# ``evaluation.py``) because ``ALLOWED["services"]`` excludes ``evaluation``
# while both services and evaluation may import ``model`` (plan 修正 4).  Every
# rejection is ``ValueError`` / ``TypeError`` — never ``EvaluationConstraintError``,
# which is silently swallowed as an invalid candidate (修正 10).

CONSTRAINT_BINARY_OPS = frozenset({"add", "sub", "mul", "div", "pow"})
CONSTRAINT_LEAF_OPS = frozenset({"ref", "const"})
CONSTRAINT_UNARY_OPS = frozenset({"sin", "cos"})
CONSTRAINT_OPS = CONSTRAINT_LEAF_OPS | CONSTRAINT_BINARY_OPS | CONSTRAINT_UNARY_OPS
MAX_CONSTRAINT_DEPTH = 64


def _reference_key(reference: ParameterReference) -> str:
    return f"{reference.dataset_id}::{reference.parameter_name}"


def _validate_ref_node(node: ConstraintNode) -> None:
    if not isinstance(node.reference, ParameterReference):
        raise TypeError("ref node requires a ParameterReference")
    if node.value is not None or node.operands:
        raise ValueError("ref node must not carry a value or operands")


def _validate_const_node(node: ConstraintNode) -> None:
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise TypeError("const node requires a numeric value")
    if not isfinite(node.value):
        raise ValueError("const value must be finite")
    object.__setattr__(node, "value", float(node.value))
    if node.reference is not None or node.operands:
        raise ValueError("const node must not carry a reference or operands")


def _validate_binary_node(node: ConstraintNode) -> None:
    if node.reference is not None or node.value is not None:
        raise ValueError(f"{node.op} node must not carry a reference or value")
    if len(node.operands) != 2:
        raise ValueError(f"{node.op} node requires exactly two operands")
    if any(not isinstance(operand, ConstraintNode) for operand in node.operands):
        raise TypeError("binary operands must be ConstraintNode values")
    if node.op == "pow" and node.operands[1].op != "const":
        raise ValueError("pow exponent must be a constant")


def _validate_unary_node(node: ConstraintNode) -> None:
    if node.reference is not None or node.value is not None:
        raise ValueError(f"unary op {node.op!r} takes no reference/value")
    if len(node.operands) != 1:
        raise ValueError(f"unary op {node.op!r} needs exactly one operand")
    if not isinstance(node.operands[0], ConstraintNode):
        raise TypeError("unary operand must be a ConstraintNode value")


@dataclass(frozen=True, slots=True)
class ConstraintNode:
    op: str
    reference: ParameterReference | None = None
    value: float | None = None
    operands: tuple[ConstraintNode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))
        if self.op not in CONSTRAINT_OPS:
            raise ValueError(f"unsupported constraint op: {self.op}")
        if self.op == "ref":
            _validate_ref_node(self)
        elif self.op == "const":
            _validate_const_node(self)
        elif self.op in CONSTRAINT_UNARY_OPS:
            _validate_unary_node(self)
        else:
            _validate_binary_node(self)


def _iter_references(node: ConstraintNode) -> Iterator[ParameterReference]:
    pending = [(node, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_CONSTRAINT_DEPTH:
            raise ValueError("constraint expression nesting exceeds the maximum depth")
        if current.op == "ref":
            yield current.reference
        pending.extend((operand, depth + 1) for operand in reversed(current.operands))


@dataclass(frozen=True, slots=True)
class ConstraintRule:
    target: ParameterReference
    expression: ConstraintNode

    def __post_init__(self) -> None:
        if not isinstance(self.target, ParameterReference):
            raise TypeError("constraint target must be a ParameterReference")
        if not isinstance(self.expression, ConstraintNode):
            raise TypeError("constraint expression must be a ConstraintNode")
        # A target appearing in its own expression is a one-hop cycle; reject it
        # at construction so a self-referential rule can never reach the forward
        # model.  Cycles that span several rules need the whole set and are
        # found by ``constraint_cycle_path``.
        if any(reference == self.target for reference in _iter_references(self.expression)):
            raise ValueError(f"constraint target {_reference_key(self.target)} must not appear in its own expression")


def constraint_cycle_path(rules: Sequence[ConstraintRule]) -> tuple[str, ...]:
    """Return one dependency cycle among ``rules`` as reference keys, else ().

    The graph has one node per constraint *target*; an edge ``t -> r`` exists
    when target ``t``'s expression references another target ``r``.  Independent
    variables that are not themselves targets cannot close a cycle and are
    ignored.  The returned path starts and ends on the same key so the caller
    can quote the whole loop in an error message (环路径写进异常文本).
    """
    target_keys = {_reference_key(rule.target) for rule in rules}
    adjacency: dict[str, list[str]] = {}
    for rule in rules:
        key = _reference_key(rule.target)
        bucket = adjacency.setdefault(key, [])
        for reference in _iter_references(rule.expression):
            dependency = _reference_key(reference)
            if dependency in target_keys and dependency not in bucket:
                bucket.append(dependency)

    visiting: dict[str, bool] = {}
    stack: list[str] = []

    def _walk(node: str) -> tuple[str, ...]:
        visiting[node] = True
        stack.append(node)
        for dependency in adjacency[node]:
            if visiting.get(dependency) is True:
                start = stack.index(dependency)
                return tuple(stack[start:]) + (dependency,)
            if dependency not in visiting:
                found = _walk(dependency)
                if found:
                    return found
        stack.pop()
        visiting[node] = False
        return ()

    for node in adjacency:
        if node not in visiting:
            found = _walk(node)
            if found:
                return found
    return ()


def validate_constraint_stage_split(
    rules: Sequence[ConstraintRule],
    definitions_by_reference: Mapping[ParameterReference, ParameterDefinition],
) -> None:
    """Reject constraints that break the two-phase evaluation ordering.

    Roughness values are decoded in a second phase whose dynamic upper bounds
    depend on already-decoded non-roughness values.  A non-roughness target may
    therefore not read a roughness independent variable — its value is not known
    yet when the target is resolved.  Integer parameters describe discrete
    topology and may be neither derived targets nor independent variables in an
    analytic constraint graph.  Unknown references (target or operand) are
    rejected rather than skipped.
    """
    for rule in rules:
        target_definition = definitions_by_reference.get(rule.target)
        if target_definition is None:
            raise ValueError(f"constraint references unknown parameter: {_reference_key(rule.target)}")
        if target_definition.integer:
            raise ValueError(f"constraint integer target is unsupported: {_reference_key(rule.target)}")
        target_is_roughness = target_definition.transform == "roughness_fraction"
        for reference in _iter_references(rule.expression):
            definition = definitions_by_reference.get(reference)
            if definition is None:
                raise ValueError(f"constraint references unknown parameter: {_reference_key(reference)}")
            if definition.integer:
                raise ValueError(f"constraint independent variable must not be integer: {_reference_key(reference)}")
            if not target_is_roughness and definition.transform == "roughness_fraction":
                raise ValueError(
                    f"non-roughness constraint target {_reference_key(rule.target)} "
                    f"cannot depend on roughness parameter {_reference_key(reference)}"
                )


def constraint_sharing_conflicts(
    rules: Sequence[ConstraintRule],
    sharing_rules: Sequence[SharingRule],
) -> tuple[ParameterReference, ...]:
    """Return targets driven by both a constraint and a sharing rule.

    A parameter cannot be both equality-shared and expression-driven; the two
    would fight over the same coordinate.  The result preserves rule order and
    de-duplicates so a caller can name each conflicting reference once.
    """
    shared_members = {member for rule in sharing_rules for member in rule.members}
    conflicts: list[ParameterReference] = []
    for rule in rules:
        if rule.target in shared_members and rule.target not in conflicts:
            conflicts.append(rule.target)
    return tuple(conflicts)


@dataclass(frozen=True, slots=True)
class SharingRule:
    sharing_key: str
    members: tuple[ParameterReference, ...]

    def __post_init__(self) -> None:
        _name(self.sharing_key, "sharing_key")
        members = tuple(self.members)
        object.__setattr__(self, "members", members)
        if len(members) < 2:
            raise ValueError("sharing rule requires at least two members")
        if any(not isinstance(member, ParameterReference) for member in members):
            raise TypeError("sharing rule members must be ParameterReference values")
        if len(members) != len(set(members)):
            raise ValueError("sharing rule members must be unique")


@dataclass(frozen=True, slots=True)
class JointFitLayout:
    """Read-only description of a joint fit's datasets and shared parameters.

    The layout is derived purely from persisted project fields, so it reports
    the *declared* structure without compiling problems or reading source
    data.  It therefore cannot assert that a shared parameter still exists or
    is free; it answers only "which datasets participate and which parameters
    are declared shared", which is exactly what a progress view needs to
    attribute a joint run to its members.
    """

    dataset_ids: tuple[str, ...]
    shared_parameters: tuple[SharingRule, ...]
