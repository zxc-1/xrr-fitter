"""Qt-owned pointer to one immutable public project snapshot."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

import xrr_fitter.api as api


def _with_active_dataset(project: api.XrrProject) -> api.XrrProject:
    """Give an opened project a usable selection without marking it dirty."""
    if project.ui_state.active_dataset_id is None and project.datasets:
        return api.select_active_dataset(project, project.datasets[0].dataset_id)
    return project


class ProjectDocument(QObject):
    """Own project identity, persistence path, and unsaved-change state."""

    project_changed = Signal(object)
    path_changed = Signal(object)
    dirty_changed = Signal(bool)
    source_validation_changed = Signal(object)

    def __init__(self, project: api.XrrProject | None = None) -> None:
        super().__init__()
        self._project = api.new_project() if project is None else project
        self._path: Path | None = None
        self._dirty = False
        self._source_validation = api.inspect_sources(self._project)
        self._project_projections: list[Callable[[api.XrrProject], None]] = []

    @property
    def project(self) -> api.XrrProject:
        return self._project

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def source_validation(self) -> api.ProjectValidation:
        return self._source_validation

    @property
    def active_dataset_id(self) -> str | None:
        return self._project.ui_state.active_dataset_id

    def _set_dirty(self, value: bool) -> None:
        if self._dirty == value:
            return
        self._dirty = value
        self.dirty_changed.emit(value)

    def mark_dirty(self) -> None:
        self._set_dirty(True)

    def register_project_projection(
        self,
        projection: Callable[[api.XrrProject], None],
    ) -> None:
        if not callable(projection):
            raise TypeError("project projection must be callable")
        if projection in self._project_projections:
            raise ValueError("project projection is already registered")
        self._project_projections.append(projection)

    def unregister_project_projection(
        self,
        projection: Callable[[api.XrrProject], None],
    ) -> None:
        try:
            self._project_projections.remove(projection)
        except ValueError:
            return

    def replace_project(
        self,
        project: api.XrrProject,
        *,
        dirty: bool = True,
        source_validation: api.ProjectValidation | None = None,
    ) -> None:
        if not isinstance(project, api.XrrProject):
            raise TypeError("project must be an XrrProject")
        if source_validation is not None and not isinstance(
            source_validation,
            api.ProjectValidation,
        ):
            raise TypeError("source_validation must be a ProjectValidation")
        self._precommit_project(project)
        self._project = project
        if source_validation is not None:
            self._source_validation = source_validation
        self.project_changed.emit(project)
        if source_validation is not None:
            self.source_validation_changed.emit(source_validation)
        self._set_dirty(dirty)

    def _precommit_project(self, project: api.XrrProject) -> None:
        previous = self._project
        attempted: list[Callable[[api.XrrProject], None]] = []
        try:
            for projection in tuple(self._project_projections):
                attempted.append(projection)
                projection(project)
        except Exception as error:
            rollback_errors: list[str] = []
            for projection in reversed(attempted):
                try:
                    projection(previous)
                except Exception as rollback_error:
                    rollback_errors.append(f"{type(rollback_error).__name__}: {rollback_error}")
            if rollback_errors:
                error.add_note("project projection rollback failures: " + "; ".join(rollback_errors))
            raise

    def new(self) -> None:
        project = api.new_project()
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._source_validation = validation
        self._path = None
        self.project_changed.emit(self._project)
        self.source_validation_changed.emit(validation)
        self.path_changed.emit(None)
        self._set_dirty(False)

    def open(self, path: str | Path) -> None:
        target = Path(path)
        project = _with_active_dataset(api.load_project(target))
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._source_validation = validation
        self._path = target
        self.project_changed.emit(project)
        self.source_validation_changed.emit(validation)
        self.path_changed.emit(target)
        self._set_dirty(False)

    def save(self, path: str | Path | None = None) -> None:
        target = self._path if path is None else Path(path)
        if target is None:
            raise ValueError("project path is required for the first save")
        api.save_project(self._project, target)
        project = api.load_project(target)
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._source_validation = validation
        self.project_changed.emit(project)
        self.source_validation_changed.emit(validation)
        if target != self._path:
            self._path = target
            self.path_changed.emit(target)
        self._set_dirty(False)

    def select_active_dataset(self, dataset_id: str | None) -> None:
        self.replace_project(
            api.select_active_dataset(self._project, dataset_id),
            dirty=True,
        )

    def refresh_sources(self) -> api.ProjectValidation:
        validation = api.inspect_sources(self._project)
        self._source_validation = validation
        self.source_validation_changed.emit(validation)
        return validation

    def source_record(self, dataset_id: str) -> object:
        matches = tuple(record for record in self._source_validation.datasets if record.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown source validation dataset: {dataset_id}")
        return matches[0]

    def source_status(self, dataset_id: str) -> str:
        return str(self.source_record(dataset_id).status.value)

    def source_warning(self, dataset_id: str) -> str:
        record = self.source_record(dataset_id)
        if record.status.value == "ok":
            return ""
        actual = record.actual_sha256 or "不可用"
        return f"{record.message}\nexpected SHA-256: {record.expected_sha256}\nactual SHA-256: {actual}"

    def preview_source_update(
        self,
        dataset_id: str,
        replacement: str | Path | None = None,
    ) -> api.SourceUpdatePreview:
        return api.preview_source_update(self._project, dataset_id, replacement)

    def accept_source_update(
        self,
        preview: api.SourceUpdatePreview,
    ) -> tuple[str, ...]:
        before = self._parameter_setting_names(preview.dataset_id)
        project = api.accept_source_update(self._project, preview)
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._source_validation = validation
        self.project_changed.emit(project)
        self.source_validation_changed.emit(validation)
        self._set_dirty(True)
        retained = set(self._parameter_setting_names(preview.dataset_id))
        return tuple(name for name in before if name not in retained)

    def _parameter_setting_names(self, dataset_id: str) -> tuple[str, ...]:
        matches = tuple(dataset for dataset in self._project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return tuple(setting.name for setting in matches[0].parameter_settings)
