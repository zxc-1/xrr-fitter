from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec


class AutomaticRole(StrEnum):
    MANUAL = "manual"
    UNROUTED = "unrouted"
    SINGLE = "single"
    JOINT = "joint"
    ISOLATED_RETRY = "isolated_retry"


class AutomaticStatus(StrEnum):
    NOT_RUN = "not_run"
    PENDING = "pending"
    REFINING = "refining"
    PASSED = "passed"
    REVIEW = "review"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MeasurementPreset:
    preset_id: str
    beam: BeamSpec
    instrument: InstrumentSpec
    import_angle_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.preset_id.strip():
            raise ValueError("preset_id must not be empty")
        if not isinstance(self.beam, BeamSpec):
            raise TypeError("beam must be BeamSpec")
        if not isinstance(self.instrument, InstrumentSpec):
            raise TypeError("instrument must be InstrumentSpec")
        if not isfinite(self.import_angle_offset_deg):
            raise ValueError("import_angle_offset_deg must be finite")


@dataclass(frozen=True, slots=True)
class DatasetAutomation:
    import_batch_id: str | None = None
    fit_group_id: str | None = None
    role: AutomaticRole = AutomaticRole.MANUAL
    status: AutomaticStatus = AutomaticStatus.NOT_RUN
    statistics_member: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, AutomaticRole):
            raise TypeError("role must be AutomaticRole")
        if not isinstance(self.status, AutomaticStatus):
            raise TypeError("status must be AutomaticStatus")
        if not isinstance(self.statistics_member, bool):
            raise TypeError("statistics_member must be bool")
        routed = self.role not in {AutomaticRole.MANUAL, AutomaticRole.UNROUTED}
        if routed and not self.fit_group_id:
            raise ValueError("routed automatic role requires fit_group_id")
        automatic = self.role is not AutomaticRole.MANUAL
        if automatic and not self.import_batch_id:
            raise ValueError("automatic role requires import_batch_id")
        requires_reason = self.status in {
            AutomaticStatus.REVIEW,
            AutomaticStatus.FAILED,
        }
        if requires_reason and not self.reason:
            raise ValueError("review or failed status requires reason")
        if self.statistics_member and self.status is not AutomaticStatus.PASSED:
            raise ValueError("statistics_member requires passed status")


@dataclass(frozen=True, slots=True)
class ImportFilePreview:
    source_path: str
    display_name: str
    dataset_id_stem: str | None
    layers_backing_to_surface: tuple[str, ...]
    substrate_group_id: str | None
    requires_substrate_choice: bool
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class ImportBatchPreview:
    import_batch_id: str
    preset: MeasurementPreset
    files: tuple[ImportFilePreview, ...]

    def __post_init__(self) -> None:
        if not self.import_batch_id.strip():
            raise ValueError("import_batch_id must not be empty")
        if not isinstance(self.preset, MeasurementPreset):
            raise TypeError("preset must be MeasurementPreset")
        files = tuple(self.files)
        if any(not isinstance(value, ImportFilePreview) for value in files):
            raise TypeError("files must contain ImportFilePreview values")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ImportFailure:
    source_path: str
    message: str
    recovery_action: str


@dataclass(frozen=True, slots=True)
class AutomaticLayerResult:
    dataset_id: str
    layer_index: int
    material_name: str
    thickness_a: float
    roughness_a: float
    sld_real_a2: float
    sld_imag_a2: float
    electron_density_a3: float
    nominal_density_g_cm3: float | None
    density_scale: float
    fitted_density_g_cm3: float | None
    density_note: str | None


@dataclass(frozen=True, slots=True)
class AutomaticDatasetSummary:
    dataset_id: str
    status: AutomaticStatus
    statistics_member: bool
    reason: str | None
    layers: tuple[AutomaticLayerResult, ...]


@dataclass(frozen=True, slots=True)
class LayerUniformitySummary:
    fit_group_id: str
    layer_index: int
    material_name: str
    count: int
    mean_thickness_a: float
    minimum_thickness_a: float
    maximum_thickness_a: float
    population_std_a: float
    cv_percent: float
    relative_range_percent: float


@dataclass(frozen=True, slots=True)
class AutomaticResultSummary:
    import_batch_id: str | None
    datasets: tuple[AutomaticDatasetSummary, ...]
    uniformity: tuple[LayerUniformitySummary, ...]
