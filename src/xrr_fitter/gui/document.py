"""Qt-owned pointer to one immutable public project snapshot."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal

import xrr_fitter.api as api

MAX_UNDO_DEPTH = 50

# 「读者在看哪里」：翻页、拖分隔条、换当前数据集。它们随项目存盘，却不是编辑——不置脏、
# 不进撤销栈、不清 redo。候选选择与专家模式不在此列：前者决定导出哪一个解，后者决定拟合
# 走哪条路线，两者都改变项目的意思。
LAYOUT_FIELDS = (
    "workspace_splitter_sizes",
    "left_splitter_sizes",
    "plot_tab_index",
    "analysis_tab_index",
    "dock_state",
)
VIEW_FIELDS = ("active_dataset_id", *LAYOUT_FIELDS)


def _with_active_dataset(project: api.XrrProject) -> api.XrrProject:
    """Give an opened project a usable selection without marking it dirty."""
    if project.ui_state.active_dataset_id is None and project.datasets:
        return api.select_active_dataset(project, project.datasets[0].dataset_id)
    return project


def same_content(first: api.XrrProject, second: api.XrrProject) -> bool:
    """Report whether two snapshots differ at most in where the reader is looking.

    Fields are compared by identity: every view transition is a ``replace`` of
    ``ui_state`` alone, so an unedited field is the very same object, and
    identity never has to decide what equality means for the arrays a result
    carries.
    """
    if any(
        getattr(first, item.name) is not getattr(second, item.name)
        for item in fields(api.XrrProject)
        if item.name != "ui_state"
    ):
        return False
    before, after = first.ui_state, second.ui_state
    return all(
        getattr(before, item.name) == getattr(after, item.name)
        for item in fields(api.ProjectUiState)
        if item.name not in VIEW_FIELDS
    )


def _lost(before: api.XrrProject, after: api.XrrProject, attributes: tuple[str, ...]) -> tuple[str, ...]:
    """Datasets in both snapshots that held one of ``attributes`` and no longer do.

    A removed dataset is not counted: it went away, nothing was cleared from it.
    """
    remaining = {dataset.dataset_id: dataset for dataset in after.datasets}
    return tuple(
        dataset.dataset_id
        for dataset in before.datasets
        if dataset.dataset_id in remaining
        and any(
            getattr(dataset, name) is not None and getattr(remaining[dataset.dataset_id], name) is None
            for name in attributes
        )
    )


def dropped_results(before: api.XrrProject, after: api.XrrProject) -> tuple[str, ...]:
    return _lost(before, after, ("last_valid_result",))


def dropped_fit_work(before: api.XrrProject, after: api.XrrProject) -> tuple[str, ...]:
    """Results and checkpoints both: a checkpoint can hold hours of search."""
    return _lost(before, after, ("last_valid_result", "checkpoint"))


def with_view(project: api.XrrProject, view: api.XrrProject, names: tuple[str, ...]) -> api.XrrProject:
    """Return ``project`` showing ``view``'s values for the named view fields."""
    state = project.ui_state
    if all(getattr(state, name) == getattr(view.ui_state, name) for name in names):
        return project
    adopted = replace(state, **{name: getattr(view.ui_state, name) for name in names})
    return api.set_workspace_state(project, adopted)


class ProjectDocument(QObject):
    """Own project identity, persistence path, and unsaved-change state."""

    project_changed = Signal(object)
    # A successful new/open, rather than an edit or save of this document.
    document_replaced = Signal()
    path_changed = Signal(object)
    dirty_changed = Signal(bool)
    source_validation_changed = Signal(object)
    undo_state_changed = Signal(bool, bool)
    # An edit cleared these datasets' fit results. A run replacing its own result does not count.
    results_dropped = Signal(tuple)

    def __init__(self, project: api.XrrProject | None = None) -> None:
        super().__init__()
        self._project = api.new_project() if project is None else project
        # What is on disk (or what new/open started from): undo back to it reads clean again.
        self._saved_project = self._project
        self._path: Path | None = None
        self._dirty = False
        self._source_validation = api.inspect_sources(self._project)
        self._project_projections: list[Callable[[api.XrrProject], None]] = []
        self._undo_stack: deque[api.XrrProject] = deque(maxlen=MAX_UNDO_DEPTH)
        self._redo_stack: list[api.XrrProject] = []

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
        amend: bool = False,
    ) -> None:
        """Commit one edit.

        ``amend`` folds the commit into the undo step already on top of the
        stack instead of opening a new one: a fit writes several checkpoints and
        then its result, and the reader undoes that run as one action.
        """
        dropped = dropped_results(self._project, project)
        self._commit(project, dirty=dirty, source_validation=source_validation, amend=amend)
        if dropped:
            self.results_dropped.emit(dropped)

    def _commit(
        self,
        project: api.XrrProject,
        *,
        dirty: bool,
        source_validation: api.ProjectValidation | None,
        amend: bool,
    ) -> None:
        if not isinstance(project, api.XrrProject):
            raise TypeError("project must be an XrrProject")
        if source_validation is not None and not isinstance(
            source_validation,
            api.ProjectValidation,
        ):
            raise TypeError("source_validation must be a ProjectValidation")
        self._precommit_project(project)
        if not amend:
            self._undo_stack.append(self._project)
        self._redo_stack.clear()
        self._project = project
        if source_validation is not None:
            self._source_validation = source_validation
        self.project_changed.emit(project)
        if source_validation is not None:
            self.source_validation_changed.emit(source_validation)
        self._set_dirty(dirty)
        self._emit_undo_state()

    def apply_view(self, project: api.XrrProject) -> None:
        """Adopt a snapshot that changes only where the reader is looking.

        No dirty mark, no undo step, no lost redo.  A page or splitter change is
        stored without re-projecting: the widgets that reported it already show
        it, and redrawing every panel per click was what made paging slow.  A new
        active dataset changes what every panel shows, so it is projected.
        """
        if not same_content(self._project, project):
            raise ValueError("a view update must not change project content")
        previous = self._project
        if project.ui_state == previous.ui_state:
            return
        if project.ui_state.active_dataset_id == previous.ui_state.active_dataset_id:
            self._project = project
            return
        self._precommit_project(project)
        self._project = project
        self.project_changed.emit(project)

    def adopt_operation_result(self, project: api.XrrProject, *, amend: bool = False) -> None:
        """Commit what a background operation produced without moving the reader.

        The worker computed from the snapshot it started with, so its view is the
        one the run began on.  Pages and the active dataset moved on since then
        are carried over; mutating routes are locked while it runs, so the view is
        the only thing that can have changed underneath it.

        Not announced as dropped results: a run that writes a checkpoint over the
        result it is replacing is doing its job, not losing the reader's work.
        """
        self._commit(with_view(project, self._project, VIEW_FIELDS), dirty=True, source_validation=None, amend=amend)

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

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        return self._step(self._undo_stack, self._redo_stack)

    def redo(self) -> bool:
        return self._step(self._redo_stack, self._undo_stack)

    def _step(self, source, target) -> bool:
        """Restore the top of ``source`` while the reader stays on their page.

        A snapshot remembers the page its edit was made from; restoring that page
        made every undo look like a page change.  The active dataset does come
        back with the snapshot, so the reverted edit is the one on screen.
        """
        if not source:
            return False
        restored = with_view(source[-1], self._project, LAYOUT_FIELDS)
        self._precommit_project(restored)
        source.pop()
        target.append(self._project)
        self._project = restored
        self.project_changed.emit(restored)
        # 撤销回存盘那一刻就不再算未保存：否则改一下再撤销，关窗时还要问一次存不存。
        self._set_dirty(not same_content(restored, self._saved_project))
        self._emit_undo_state()
        return True

    def _emit_undo_state(self) -> None:
        self.undo_state_changed.emit(self.can_undo, self.can_redo)

    def new(self) -> None:
        project = api.new_project()
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._saved_project = project
        self._source_validation = validation
        self._path = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.project_changed.emit(self._project)
        self.source_validation_changed.emit(validation)
        self.path_changed.emit(None)
        self._set_dirty(False)
        self._emit_undo_state()
        self.document_replaced.emit()

    def open(self, path: str | Path) -> None:
        target = Path(path)
        project = _with_active_dataset(api.load_project(target))
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._saved_project = project
        self._source_validation = validation
        self._path = target
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.project_changed.emit(project)
        self.source_validation_changed.emit(validation)
        self.path_changed.emit(target)
        self._set_dirty(False)
        self._emit_undo_state()
        self.document_replaced.emit()

    def save(self, path: str | Path | None = None) -> None:
        target = self._path if path is None else Path(path)
        if target is None:
            raise ValueError("project path is required for the first save")
        api.save_project(self._project, target)
        project = api.load_project(target)
        validation = api.inspect_sources(project)
        self._precommit_project(project)
        self._project = project
        self._saved_project = project
        self._source_validation = validation
        self.project_changed.emit(project)
        self.source_validation_changed.emit(validation)
        if target != self._path:
            self._path = target
            self.path_changed.emit(target)
        self._set_dirty(False)

    def select_active_dataset(self, dataset_id: str | None) -> None:
        self.apply_view(api.select_active_dataset(self._project, dataset_id))

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
