"""Immutable project persistence and workspace-state values.

This module owns the model-level shape of an XRR project. Its persisted
contract covers:

* exact source identity and import settings for every dataset;
* structure, instrument, parameter, and scale-prior choices;
* fitting configuration, checkpoints, and the last publishable analysis;
* cross-dataset parameter-sharing references; and
* the limited UI selection and splitter state already stored by R22 projects.

It does not read source files, resolve relative paths, or encode project JSON.
Those responsibilities belong to the I/O layer. Source-validation and update
records are data returned by that layer, not filesystem operations hidden in
the model.

Construction validates both local values and project-wide references. Dataset
records first validate their own source and attachments. The project root then
checks version identity, unique dataset IDs, sharing members, active selection,
and selected candidates. This ordering keeps malformed local values separate
from dangling cross-record references and preserves the errors used by callers.

State transition helpers always use ``dataclasses.replace`` and therefore
return a newly validated project. They cannot mutate a previous snapshot or
bypass the invariants applied while loading a persisted project.

Dataset identity and presentation are separate persisted concepts. ``dataset_id``
is the stable namespace key for sharing, selection, seeds, and export paths.
``display_name`` is user-facing metadata and may change without renumbering.
Loading fills a missing display name from ``dataset_id`` for older documents.
Validation checks its type and content locally but never feeds it into
cross-record identity. Source-stem allocation remains a service operation and
does not live in this model.
Display names never participate in persisted candidate ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import isclose, isfinite

from xrr_fitter.model.analysis import FitResult, StructureEvidence
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import FitCheckpoint, FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting, SharingRule
from xrr_fitter.model.structure import StructureSpec


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "xrr-fit-v1"


def _sha256(value: str, field_name: str) -> None:
    valid = len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    if not valid:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _dataset_id(value: str) -> None:
    if not value.strip():
        raise ValueError("dataset_id must not be empty")


def _selected_candidates(values: object) -> tuple[tuple[str, str], ...]:
    selected = tuple(tuple(value) for value in values)
    if any(len(value) != 2 or not value[0] or not value[1] for value in selected):
        raise ValueError("selected_candidate_ids must contain nonempty pairs")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_candidate_ids must be unique")
    return selected


def _splitter_sizes(
    workspace_values: object,
    left_values: object,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    workspace = tuple(workspace_values)
    left = tuple(left_values)
    if len(workspace) != 3 or len(left) != 2:
        raise ValueError("workspace splitter sizes have invalid dimensions")
    values = (*workspace, *left)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("workspace splitter sizes must be nonnegative integers")
    return workspace, left


def _nonnegative_index(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _bool_value(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True, slots=True)
class ScalePriorState:
    """Persist the exact three-state scale-prior decision.

    An unresolved value has no estimate, width, or reason. An inactive value
    records a positive configured width plus the reason compilation disabled
    it. An active value records a positive estimate and width without a reason.
    """

    enabled: bool
    s_hat: float | None = None
    tau_s_decades: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OxideDecision:
    """A versioned user decision for one suggested surface or backing oxide."""

    base_material: str
    oxide_material: str
    location: str
    accepted: bool
    oxide_table_version: str

    def __post_init__(self) -> None:
        for field_name in ("base_material", "oxide_material", "location", "oxide_table_version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.location not in {"surface", "backing"}:
            raise ValueError("location must be surface or backing")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")


@dataclass(frozen=True, slots=True)
class ProjectUiState:
    """The limited workspace layout and selection state persisted by projects."""

    active_dataset_id: str | None = None
    selected_candidate_ids: tuple[tuple[str, str], ...] = ()
    expert_mode: bool = False
    workspace_splitter_sizes: tuple[int, int, int] = (320, 680, 380)
    left_splitter_sizes: tuple[int, int] = (280, 480)
    plot_tab_index: int = 0

    def __post_init__(self) -> None:
        if self.active_dataset_id is not None:
            _dataset_id(self.active_dataset_id)
        selected = _selected_candidates(self.selected_candidate_ids)
        workspace, left = _splitter_sizes(
            self.workspace_splitter_sizes,
            self.left_splitter_sizes,
        )
        _nonnegative_index(self.plot_tab_index, "plot_tab_index")
        _bool_value(self.expert_mode, "expert_mode")
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "workspace_splitter_sizes", workspace)
        object.__setattr__(self, "left_splitter_sizes", left)


@dataclass(frozen=True, slots=True)
class DatasetProject:
    """One imported dataset and its immutable fit workspace attachments.

    ``source_path`` remains the persisted project path while ``source_sha256``
    binds the bytes originally imported. Fit masks retain source-row
    cardinality. Analysis results and checkpoints are optional snapshots, not
    implicit permission for service code to reuse stale values.
    """

    dataset_id: str
    source_path: str
    source_sha256: str
    beam: BeamSpec
    import_angle_offset_deg: float
    column_mapping: DataColumnMapping
    fit_mask: tuple[bool, ...]
    fit_range_two_theta_deg: tuple[float, float]
    structure: StructureSpec | None
    instrument: InstrumentSpec
    structure_evidence: StructureEvidence | None = None
    scale_prior: ScalePriorState = ScalePriorState(enabled=False)
    oxide_decisions: tuple[OxideDecision, ...] = ()
    parameter_settings: tuple[ParameterSetting, ...] = ()
    last_valid_result: FitResult | None = None
    checkpoint: FitCheckpoint | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        _validate_dataset_header(self)
        mask = _fit_mask(self.fit_mask)
        fit_range = _fit_range(self.fit_range_two_theta_deg)
        _validate_dataset_attachments(self)
        object.__setattr__(self, "fit_mask", mask)
        object.__setattr__(self, "fit_range_two_theta_deg", fit_range)
        object.__setattr__(self, "oxide_decisions", tuple(self.oxide_decisions))
        object.__setattr__(self, "parameter_settings", tuple(self.parameter_settings))
        object.__setattr__(self, "display_name", self.display_name or self.dataset_id)


@dataclass(frozen=True, slots=True)
class XrrProject:
    """The complete versioned project root consumed by service and I/O layers.

    ``base_directory`` is runtime resolution context and is deliberately
    excluded from equality. All other fields belong to the deterministic
    persisted value and participate in validation and comparison.
    """

    schema_version: int
    algorithm_version: str
    fit_config: FitConfig
    input_angle_kind: str
    batch_mode: str
    datasets: tuple[DatasetProject, ...]
    sharing_rules: tuple[SharingRule, ...] = ()
    ui_state: ProjectUiState = ProjectUiState()
    base_directory: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "sharing_rules", tuple(self.sharing_rules))
        validate_project(self)

    @property
    def master_seed(self) -> int:
        return self.fit_config.master_seed

    @classmethod
    def new(cls, datasets: tuple[DatasetProject, ...], master_seed: int) -> XrrProject:
        return cls(
            schema_version=SCHEMA_VERSION,
            algorithm_version=ALGORITHM_VERSION,
            fit_config=FitConfig.standard(master_seed),
            input_angle_kind="two_theta_deg",
            batch_mode="independent",
            datasets=tuple(datasets),
        )


class SourceStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    HASH_MISMATCH = "hash_mismatch"


@dataclass(frozen=True, slots=True)
class DatasetSourceValidation:
    """Observed source identity compared with one persisted dataset reference."""

    dataset_id: str
    status: SourceStatus
    expected_sha256: str
    actual_sha256: str | None
    message: str
    source_identity: str | None = None

    def __post_init__(self) -> None:
        _dataset_id(self.dataset_id)
        _sha256(self.expected_sha256, "expected_sha256")
        if self.actual_sha256 is not None:
            _sha256(self.actual_sha256, "actual_sha256")
        if not isinstance(self.status, SourceStatus):
            raise TypeError("status must be SourceStatus")

    @property
    def expected_hash(self) -> str:
        return self.expected_sha256

    @property
    def actual_hash(self) -> str | None:
        return self.actual_sha256

    @property
    def user_message(self) -> str:
        return self.message

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A stable project-validation problem suitable for user-facing reports."""

    code: str
    message: str
    dataset_id: str | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("validation issue code and message must not be empty")
        if self.dataset_id is not None:
            _dataset_id(self.dataset_id)


@dataclass(frozen=True, slots=True)
class ProjectValidation:
    """Source records and semantic issues collected for an entire project.

    A validation is successful only when no semantic issue exists and every
    dataset source has status ``OK``. The sequence is copied so a caller cannot
    change a completed validation report through its input list.
    """

    datasets: tuple[DatasetSourceValidation, ...]
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def valid(self) -> bool:
        return not self.issues and all(item.status is SourceStatus.OK for item in self.datasets)

    @property
    def is_valid(self) -> bool:
        return self.valid


@dataclass(frozen=True, slots=True)
class SourceUpdatePreview:
    """A non-mutating preview of a proposed dataset source replacement."""

    dataset_id: str
    current_source_path: str
    proposed_source_path: str
    expected_sha256: str
    observed_sha256: str
    changed: bool

    def __post_init__(self) -> None:
        _dataset_id(self.dataset_id)
        _sha256(self.expected_sha256, "expected_sha256")
        _sha256(self.observed_sha256, "observed_sha256")
        if not self.current_source_path or not self.proposed_source_path:
            raise ValueError("source update paths must not be empty")


def _positive_finite(value: float | None) -> bool:
    return value is not None and isfinite(value) and value > 0.0


def _unresolved_scale_prior(state: ScalePriorState) -> bool:
    # Unresolved means compilation has not yet made either a positive estimate
    # or an explicit disabled decision.
    return (
        not state.enabled
        and state.s_hat is None
        and state.tau_s_decades is None
        and state.reason is None
    )


def _inactive_scale_prior(state: ScalePriorState) -> bool:
    # Disabled compiled state retains the configured width and a user reason,
    # but must not retain a stale scale estimate.
    return (
        not state.enabled
        and state.s_hat is None
        and _positive_finite(state.tau_s_decades)
        and isinstance(state.reason, str)
        and bool(state.reason.strip())
    )


def _active_scale_prior(state: ScalePriorState) -> bool:
    # Active state owns both positive values and no contradictory reason.
    return (
        state.enabled
        and _positive_finite(state.s_hat)
        and _positive_finite(state.tau_s_decades)
        and state.reason is None
    )


def _validate_scale_prior(state: ScalePriorState) -> None:
    if not isinstance(state, ScalePriorState):
        raise TypeError("scale_prior must be ScalePriorState")
    valid = type(state.enabled) is bool and any(
        (
            _unresolved_scale_prior(state),
            _inactive_scale_prior(state),
            _active_scale_prior(state),
        )
    )
    if not valid:
        raise ValueError("invalid scale prior state")


def _validate_dataset_header(dataset: DatasetProject) -> None:
    _dataset_id(dataset.dataset_id)
    if dataset.display_name is not None:
        if not isinstance(dataset.display_name, str):
            raise TypeError("display_name must be str or None")
        if not dataset.display_name.strip():
            raise ValueError("display_name must not be empty")
    if not dataset.source_path:
        raise ValueError("source_path must not be empty")
    _sha256(dataset.source_sha256, "source_sha256")
    if not isinstance(dataset.beam, BeamSpec):
        raise TypeError("beam must be BeamSpec")
    if not isinstance(dataset.column_mapping, DataColumnMapping):
        raise TypeError("column_mapping must be DataColumnMapping")
    if not isinstance(dataset.instrument, InstrumentSpec):
        raise TypeError("instrument must be InstrumentSpec")
    if not isfinite(dataset.import_angle_offset_deg):
        raise ValueError("import_angle_offset_deg must be finite")


def _fit_mask(values: object) -> tuple[bool, ...]:
    mask = tuple(values)
    if not mask or any(not isinstance(value, bool) for value in mask):
        raise ValueError("fit_mask must be a nonempty bool sequence")
    return mask


def _fit_range(values: object) -> tuple[float, float]:
    fit_range = tuple(values)
    valid = (
        len(fit_range) == 2
        and all(isfinite(value) for value in fit_range)
        and fit_range[0] <= fit_range[1]
    )
    if not valid:
        raise ValueError("fit_range_two_theta_deg must be a finite ordered pair")
    return fit_range


def _optional_attachment(value: object, expected: type, field_name: str) -> None:
    if value is not None and not isinstance(value, expected):
        raise TypeError(f"{field_name} must be {expected.__name__} or None")


def _validate_oxide_decisions(values: object) -> None:
    if any(not isinstance(value, OxideDecision) for value in values):
        raise TypeError("oxide_decisions must contain OxideDecision values")


def _validate_parameter_settings(values: object) -> None:
    if any(not isinstance(value, ParameterSetting) for value in values):
        raise TypeError("parameter_settings must contain ParameterSetting values")


def _validate_dataset_attachments(dataset: DatasetProject) -> None:
    # Attachments are independently immutable; this boundary also resolves
    # uncertainty ownership against the candidate graph being attached.
    _optional_attachment(dataset.structure, StructureSpec, "structure")
    _optional_attachment(dataset.structure_evidence, StructureEvidence, "structure_evidence")
    _validate_scale_prior(dataset.scale_prior)
    _validate_oxide_decisions(dataset.oxide_decisions)
    _validate_parameter_settings(dataset.parameter_settings)
    _optional_attachment(dataset.last_valid_result, FitResult, "last_valid_result")
    _optional_attachment(dataset.checkpoint, FitCheckpoint, "checkpoint")
    if dataset.last_valid_result is not None:
        _validate_result_evidence(dataset.last_valid_result)


def _validate_project_header(project: XrrProject) -> None:
    # Version and mode checks precede graph traversal so unsupported documents
    # fail before their nested references are interpreted.
    schema_valid = (
        isinstance(project.schema_version, int)
        and not isinstance(project.schema_version, bool)
        and project.schema_version == SCHEMA_VERSION
    )
    if not schema_valid:
        raise ValueError("unsupported schema_version")
    expected_values = (
        (project.algorithm_version, ALGORITHM_VERSION, "algorithm_version"),
        (project.input_angle_kind, "two_theta_deg", "input_angle_kind"),
    )
    for actual, expected, field_name in expected_values:
        if actual != expected:
            raise ValueError(f"unsupported {field_name}")
    if project.batch_mode not in {"independent", "joint"}:
        raise ValueError("unsupported batch_mode")
    if not isinstance(project.fit_config, FitConfig):
        raise TypeError("fit_config must be FitConfig")
    if not isinstance(project.ui_state, ProjectUiState):
        raise TypeError("ui_state must be ProjectUiState")


def _project_dataset_ids(project: XrrProject) -> tuple[str, ...]:
    identifiers = tuple(dataset.dataset_id for dataset in project.datasets)
    invalid = any(not value.strip() for value in identifiers)
    if invalid or len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset_id values must be nonempty and unique")
    return identifiers


def validate_project(project: XrrProject) -> None:
    """Validate project identity plus cross-dataset sharing and UI references."""
    # Validation proceeds from namespace identity to cross-dataset fit graphs,
    # then to UI selections that point into those already validated graphs.
    _validate_project_header(project)
    dataset_ids = set(_project_dataset_ids(project))
    _validate_sharing(project.sharing_rules, dataset_ids)
    if project.batch_mode == "joint":
        _validate_joint_results(project.datasets)
    _validate_ui(project, dataset_ids)


def _validate_result_evidence(result: FitResult) -> None:
    # Legacy evidence may omit an owner. Once present, both uncertainty and its
    # nested MCMC report must resolve to the same result's stable candidate IDs.
    report = result.uncertainty
    if report is None:
        return
    candidate_ids = {candidate.candidate_id for candidate in result.candidates}
    owners = ((report.candidate_id, "uncertainty"),)
    if report.mcmc is not None:
        owners = (*owners, (report.mcmc.candidate_id, "mcmc"))
    for owner, label in owners:
        if owner is not None and owner not in candidate_ids:
            raise ValueError(f"{label} candidate_id references a missing candidate")


def _candidate_identity(candidate: object) -> tuple[object, ...]:
    # Joint projections legitimately have different local objectives and arrays.
    # Only global lineage and rank claims must remain identical across datasets.
    return (
        candidate.candidate_id,
        candidate.seed_index,
        candidate.valid,
        candidate.stop_reason,
        candidate.nfev,
        candidate.ranking_objective,
    )


def _result_identity(result: FitResult) -> tuple[object, ...]:
    # Joint search owns aligned lineage, selection, seeds, and stage history.
    # Confidence and analysis-enriched warnings remain dataset-local evidence.
    return (
        tuple(_candidate_identity(candidate) for candidate in result.candidates),
        result.best_index,
        result.child_seeds,
        result.stage_summaries,
    )


def _validate_joint_rank(candidates: tuple[object, ...]) -> None:
    # Invalid global candidates cannot claim a finite rank in any projection.
    if not candidates[0].valid:
        if any(candidate.ranking_objective is not None for candidate in candidates):
            raise ValueError("joint candidates have inconsistent invalid ranking")
        return
    objectives = tuple(candidate.objective for candidate in candidates)
    # A valid joint rank is the mean of aligned finite local objectives, never
    # a copied objective from whichever dataset happened to finish first.
    if any(not isfinite(value) for value in objectives):
        raise ValueError("joint candidates have nonfinite objective")
    ranking = candidates[0].ranking_objective
    expected = sum(objectives) / len(objectives)
    if ranking is None or not isclose(ranking, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("joint candidate global ranking does not match local mean")


def _validate_joint_results(datasets: tuple[DatasetProject, ...]) -> None:
    # Joint publication is atomic: either every dataset has a projection or none
    # does. Partial results would make best_index and UI selection ambiguous.
    results = tuple(
        dataset.last_valid_result
        for dataset in datasets
        if dataset.last_valid_result is not None
    )
    if results and len(results) != len(datasets):
        raise ValueError("joint results must exist for every dataset")
    if not results:
        return
    # Establish alignment before indexing candidates for mean-rank validation.
    identity = _result_identity(results[0])
    if any(_result_identity(result) != identity for result in results[1:]):
        raise ValueError("joint result candidates are not coherent across datasets")
    for index in range(len(results[0].candidates)):
        _validate_joint_rank(tuple(result.candidates[index] for result in results))


def _validate_sharing(rules: tuple[SharingRule, ...], dataset_ids: set[str]) -> None:
    if any(not isinstance(rule, SharingRule) for rule in rules):
        raise TypeError("sharing_rules must contain SharingRule values")
    keys = tuple(rule.sharing_key for rule in rules)
    if len(keys) != len(set(keys)):
        raise ValueError("sharing_key values must be unique")
    for rule in rules:
        if any(member.dataset_id not in dataset_ids for member in rule.members):
            raise ValueError("sharing rule references a missing dataset")


def _validate_ui(project: XrrProject, dataset_ids: set[str]) -> None:
    # Workspace selections persist stable IDs, never positional indices, and
    # therefore must resolve after dataset and candidate validation succeeds.
    state = project.ui_state
    if state.active_dataset_id is not None and state.active_dataset_id not in dataset_ids:
        raise ValueError("active_dataset_id references a missing dataset")
    by_id = {dataset.dataset_id: dataset for dataset in project.datasets}
    for dataset_id, candidate_id in state.selected_candidate_ids:
        if dataset_id not in by_id:
            raise ValueError("selected candidate references a missing dataset")
        result = by_id[dataset_id].last_valid_result
        if result is None:
            raise ValueError("selected candidate has no persisted result")
        candidate_ids = {candidate.candidate_id for candidate in result.candidates}
        if candidate_id not in candidate_ids:
            raise ValueError("selected candidate is absent from persisted result")


def _replace_dataset(project: XrrProject, dataset_id: str, value: DatasetProject) -> XrrProject:
    # Exact-one replacement prevents helpers from masking a corrupt namespace.
    indices = [index for index, dataset in enumerate(project.datasets) if dataset.dataset_id == dataset_id]
    if len(indices) != 1:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    datasets = list(project.datasets)
    datasets[indices[0]] = value
    return replace(project, datasets=tuple(datasets))


def with_active_dataset(project: XrrProject, dataset_id: str | None) -> XrrProject:
    """Return a project whose persisted workspace selects ``dataset_id``."""
    if dataset_id is not None and dataset_id not in {value.dataset_id for value in project.datasets}:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    return replace(project, ui_state=replace(project.ui_state, active_dataset_id=dataset_id))


def with_batch_mode(project: XrrProject, mode: str) -> XrrProject:
    """Return a project using an explicitly supported batch-fit mode."""
    if mode not in {"independent", "joint"}:
        raise ValueError(f"unsupported batch_mode: {mode}")
    if mode == "joint" and len(project.datasets) < 2:
        raise ValueError("joint batch mode requires at least two datasets")
    return replace(project, batch_mode=mode)


def with_dataset_fit_mask(
    project: XrrProject,
    dataset_id: str,
    fit_mask: tuple[bool, ...],
) -> XrrProject:
    """Replace one dataset mask without changing its persisted row cardinality."""
    # Mask transitions preserve source-row alignment; invalidation is performed
    # later by the service layer rather than hidden in this value operation.
    dataset = next((value for value in project.datasets if value.dataset_id == dataset_id), None)
    if dataset is None:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    mask = tuple(fit_mask)
    if len(mask) != len(dataset.fit_mask) or any(not isinstance(value, bool) for value in mask):
        raise ValueError("fit_mask must match the persisted dataset mask")
    return _replace_dataset(project, dataset_id, replace(dataset, fit_mask=mask))


def with_workspace_state(project: XrrProject, state: ProjectUiState) -> XrrProject:
    """Replace the persisted workspace state after validating its references."""
    if not isinstance(state, ProjectUiState):
        raise TypeError("state must be ProjectUiState")
    return replace(project, ui_state=state)
