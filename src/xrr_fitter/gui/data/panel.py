"""Dataset import panel backed exclusively by the supported application API.

The panel owns file selection, measurement presets, filename material previews,
substrate confirmation, and fit-mask editing. Every mutation adopts the immutable
project returned by ``xrr_fitter.api``; widgets never patch domain objects in
place. Batch failures remain visible per source while successful imports update
the active dataset and the compact source/instrument summary atomically.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.data.import_dialog import ImportDialog
from xrr_fitter.gui.data.mask_editor import MaskEditor
from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog
from xrr_fitter.gui.document import ProjectDocument

DATA_FILTER = "XRR 数据 (*.xy *.dat *.txt);;所有文件 (*)"
SUPPORTED_SUFFIXES = {".xy", ".dat", ".txt"}
TREE_HEADERS = ("数据集", "源文件", "光路", "仪器", "状态", "SHA-256", "拟合")
COMPACT_COLUMNS = (0, 4, 6)
DETAIL_COLUMNS = (1, 2, 3, 5)

# Precise, glanceable rendering of each source-validation status. The marker
# turns a buried "源文件异常" into an at-a-glance signal, and the label names
# the exact failure so the fix is obvious without opening the detail pane.
SOURCE_STATUS_VISUALS = {
    "ok": ("", "源文件正常"),
    "missing": ("⛔", "源文件缺失"),
    "unreadable": ("⚠", "源文件无法读取"),
    "hash_mismatch": ("⚠", "源文件内容已改变"),
}

# Glyph per persisted confidence class, so a dataset row states not just that a
# fit exists but how far to trust it. The glyphs mirror the results panel badge
# (defined locally to keep the data panel from importing the results module).
FIT_STATUS_VISUALS = {
    "可信": "●",
    "可用但相关": "◆",
    "多解": "▲",
    "不可信": "■",
}


class DataPanel(QWidget):
    """Render project datasets and commit data mutations transactionally."""

    datasets_imported = Signal(tuple)
    active_dataset_changed = Signal(object)
    mask_changed = Signal(str, tuple)
    instrument_changed = Signal(str, object)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("dataPanel")
        self.setAccessibleName("数据与掩膜")
        self._force_preset_dialog = False
        self._build_controls()
        self._mask_editor = MaskEditor(document, self)
        self._mask_editor.mask_changed.connect(self.mask_changed.emit)
        document.project_changed.connect(self._render_project)
        document.source_validation_changed.connect(self._render_project)
        self._render_project()

    def _build_controls(self) -> None:
        heading = QLabel("数据集")
        heading.setObjectName("dataPanelHeader")
        heading.setProperty("sectionHeader", True)
        self.import_files_button = QPushButton("导入文件")
        self.import_files_button.setObjectName("importFilesButton")
        self.import_files_button.setProperty("commandBar", True)
        self.import_folder_button = QPushButton("导入文件夹")
        self.import_folder_button.setObjectName("importFolderButton")
        self.import_folder_button.setProperty("commandBar", True)
        self.change_preset_button = QPushButton("更换测量预设")
        self.change_preset_button.setObjectName("changeMeasurementPresetButton")
        self.change_preset_button.setProperty("commandBar", True)
        self.change_preset_button.setAccessibleName("更换测量预设")
        self.change_preset_button.setToolTip("为后续导入选择新的光路和仪器预设")
        self.import_files_button.clicked.connect(self._import_files)
        self.import_folder_button.clicked.connect(self._import_folder)
        self.change_preset_button.clicked.connect(self._change_measurement_preset)
        self.import_files_shortcut = self._shortcut("importFilesShortcut", "Ctrl+I")
        self.import_folder_shortcut = self._shortcut(
            "importFolderShortcut",
            "Ctrl+Shift+I",
        )
        self.import_files_shortcut.activated.connect(self._import_files)
        self.import_folder_shortcut.activated.connect(self._import_folder)
        self.tree = QTreeWidget()
        self.tree.setObjectName("datasetTree")
        self.tree.setAccessibleName("数据集列表")
        self.tree.setColumnCount(len(TREE_HEADERS))
        self.tree.setHeaderLabels(TREE_HEADERS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        for column in DETAIL_COLUMNS:
            self.tree.setColumnHidden(column, True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.currentItemChanged.connect(self._select_tree_item)
        self.details_label = QLabel()
        self.details_label.setObjectName("datasetDetails")
        self.details_label.setProperty("mutedText", True)
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.hide()
        self.failure_table = QTableWidget(0, 3)
        self.failure_table.setObjectName("importFailureTable")
        self.failure_table.setHorizontalHeaderLabels(("文件", "问题", "恢复操作"))
        self.failure_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.failure_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.failure_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.failure_table.horizontalHeader().setStretchLastSection(True)
        self.failure_table.hide()
        top = QHBoxLayout()
        top.addWidget(heading)
        top.addStretch(1)
        top.addWidget(self.import_files_button)
        top.addWidget(self.import_folder_button)
        top.addWidget(self.change_preset_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.tree)
        layout.addWidget(self.details_label)
        layout.addWidget(self.failure_table)

    def _shortcut(self, name: str, keys: str) -> QShortcut:
        shortcut = QShortcut(QKeySequence(keys), self)
        shortcut.setObjectName(name)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        return shortcut

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(dataset.dataset_id for dataset in self.document.project.datasets)

    @property
    def active_dataset_id(self) -> str | None:
        return self.document.active_dataset_id

    def add_paths(
        self,
        paths,
        *,
        beam: api.BeamSpec,
        instrument: api.InstrumentSpec,
        column_mapping: api.DataColumnMapping | None = None,
        import_angle_offset_deg: float = 0.0,
    ) -> None:
        sources = tuple(Path(path) for path in paths)
        if not sources:
            raise ValueError("data import requires at least one path")
        candidate = self.document.project
        imported: list[str] = []
        for source in sources:
            candidate = api.add_dataset(
                candidate,
                source,
                instrument,
                column_mapping=column_mapping,
                import_angle_offset_deg=import_angle_offset_deg,
                beam=beam,
            )
            imported.append(candidate.datasets[-1].dataset_id)
        previous_active = self.active_dataset_id
        validation = api.inspect_sources(candidate)
        self.document.replace_project(candidate, source_validation=validation)
        self.datasets_imported.emit(tuple(imported))
        if self.active_dataset_id != previous_active:
            self.active_dataset_changed.emit(self.active_dataset_id)

    def add_folder(
        self,
        folder: str | Path,
        *,
        beam: api.BeamSpec,
        instrument: api.InstrumentSpec,
        recursive: bool,
        column_mapping: api.DataColumnMapping | None = None,
        import_angle_offset_deg: float = 0.0,
    ) -> None:
        root = Path(folder)
        entries = root.rglob("*") if recursive else root.glob("*")
        paths = tuple(
            sorted(
                (
                    path
                    for path in entries
                    if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
                ),
                key=lambda path: path.relative_to(root).as_posix().casefold(),
            )
        )
        self.add_paths(
            paths,
            beam=beam,
            instrument=instrument,
            column_mapping=column_mapping,
            import_angle_offset_deg=import_angle_offset_deg,
        )

    def import_paths(
        self,
        paths,
        *,
        preset: api.MeasurementPreset | None = None,
        column_mapping: api.DataColumnMapping | None = None,
    ) -> api.ProjectImportResult:
        sources = tuple(Path(path) for path in paths)
        if not sources:
            raise ValueError("data import requires at least one path")
        selected_preset = self.document.project.measurement_preset if preset is None else preset
        if selected_preset is None:
            raise ValueError("automatic import requires a measurement preset")
        preview = api.preview_import_batch(sources, selected_preset)
        substrate_choices = self._substrate_choices(preview)
        mappings = (
            None
            if column_mapping is None
            else {row.source_path: column_mapping for row in preview.files}
        )
        before_active = self.active_dataset_id
        result = api.import_dataset_batch(
            self.document.project,
            preview,
            substrate_choices,
            mappings,
        )
        validation = api.inspect_sources(result.updated_project)
        self.document.replace_project(
            result.updated_project,
            source_validation=validation,
        )
        self._render_failures(result.failures)
        if result.imported_dataset_ids:
            self.datasets_imported.emit(result.imported_dataset_ids)
            if self.active_dataset_id != before_active:
                self.active_dataset_changed.emit(self.active_dataset_id)
        return result

    def _substrate_choices(
        self,
        preview: api.ImportBatchPreview,
    ) -> dict[str, str]:
        choices: dict[str, str] = {}
        for row in preview.files:
            group_id = row.substrate_group_id
            if not row.requires_substrate_choice or group_id in choices:
                continue
            if group_id is None:
                raise ValueError("substrate choice requires a structure group")
            dialog = SubstrateDialog(row.layers_backing_to_surface, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                raise InterruptedError("substrate selection cancelled")
            choices[group_id] = dialog.substrate_token()
        return choices

    def _render_failures(self, failures: tuple[api.ImportFailure, ...]) -> None:
        self.failure_table.setRowCount(len(failures))
        for row, failure in enumerate(failures):
            values = (
                Path(failure.source_path).name,
                failure.message,
                failure.recovery_action,
            )
            for column, value in enumerate(values):
                self.failure_table.setItem(row, column, QTableWidgetItem(value))
        self.failure_table.setVisible(bool(failures))

    def set_fit_range(self, dataset_id: str, lower: float, upper: float) -> None:
        self._mask_editor.set_fit_range(dataset_id, lower, upper)

    def set_point_enabled(self, dataset_id: str, index: int, enabled: bool) -> None:
        self._mask_editor.set_point_enabled(dataset_id, index, enabled)

    def set_instrument(
        self,
        dataset_id: str,
        instrument: api.InstrumentSpec,
    ) -> None:
        updated = api.set_instrument(self.document.project, dataset_id, instrument)
        self.document.replace_project(updated)
        self.instrument_changed.emit(dataset_id, instrument)

    def _source_status_code(self, dataset_id: str) -> str:
        try:
            return self.document.source_status(dataset_id)
        except KeyError:
            return "ok"

    def status_text(self, dataset_id: str) -> str:
        code = self._source_status_code(dataset_id)
        if code != "ok":
            return SOURCE_STATUS_VISUALS.get(code, ("", "源文件异常"))[1]
        dataset = self._dataset(dataset_id)
        return "可拟合" if sum(dataset.fit_mask) >= 30 else "数据点不足"

    def status_marker(self, dataset_id: str) -> str:
        """Return the glanceable marker glyph for the dataset's source status."""
        code = self._source_status_code(dataset_id)
        return SOURCE_STATUS_VISUALS.get(code, ("", ""))[0]

    def fit_status_text(self, dataset_id: str) -> str:
        """Name the dataset's fit outcome: unfitted, or its confidence label."""
        result = self._dataset(dataset_id).last_valid_result
        return "未拟合" if result is None else str(result.confidence.value)

    def fit_status_marker(self, dataset_id: str) -> str:
        """Return the confidence glyph, or empty when the dataset is unfitted."""
        result = self._dataset(dataset_id).last_valid_result
        if result is None:
            return ""
        return FIT_STATUS_VISUALS.get(str(result.confidence.value), "")

    def _fit_status_tooltip(self, dataset_id: str) -> str:
        """Summarise the fit outcome so the compact cell can stay short."""
        result = self._dataset(dataset_id).last_valid_result
        if result is None:
            return "尚未拟合该数据集"
        best = result.best_candidate
        parts = [f"可信度：{result.confidence.value}", f"候选解 {len(result.candidates)} 个"]
        if best is not None:
            parts.append(f"最优目标值 J={best.objective:g}")
        if result.warnings:
            parts.append(f"告警 {len(result.warnings)} 条")
        return " · ".join(parts)

    def sha256_text(self, dataset_id: str) -> str:
        return self._dataset(dataset_id).source_sha256

    def beam_text(self, dataset_id: str) -> str:
        beam = self._dataset(dataset_id).beam
        if beam.kind == "monochromatic":
            return f"单色 λ={beam.wavelength_a:g} Å"
        return (
            f"混合 Kα λ₁={beam.wavelength_1_a:g} Å / "
            f"λ₂={beam.wavelength_2_a:g} Å · I₂/I₁={beam.intensity_ratio_21:g}"
        )

    def instrument_text(self, dataset_id: str) -> str:
        instrument = self._dataset(dataset_id).instrument
        identity = instrument.instrument_id or "默认仪器"
        footprint = self._footprint_text(instrument)
        domain = "θ" if instrument.resolution_domain == "theta" else "q"
        return (
            f"{identity} · {footprint} · 背景 {instrument.background_kind} · "
            f"分辨率 {domain}"
        )

    def _footprint_text(self, instrument: api.InstrumentSpec) -> str:
        if instrument.footprint_mode == "geometry":
            return (
                f"几何换算 {instrument.sample_length_mm:g}×"
                f"{instrument.beam_width_mm:g} mm"
            )
        if instrument.footprint_mode == "none":
            return "无足迹修正"
        return "拟合足迹"

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(
            dataset
            for dataset in self.document.project.datasets
            if dataset.dataset_id == dataset_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def _render_project(self, *_args) -> None:
        blocker = QSignalBlocker(self.tree)
        self.tree.clear()
        active_item = None
        for dataset in self.document.project.datasets:
            item = self._tree_item(dataset)
            self.tree.addTopLevelItem(item)
            if dataset.dataset_id == self.active_dataset_id:
                active_item = item
        if active_item is not None:
            self.tree.setCurrentItem(active_item)
        del blocker
        self.change_preset_button.setVisible(
            self.document.project.ui_state.expert_mode
        )
        self._render_details()

    def _render_details(self) -> None:
        dataset_id = self.active_dataset_id
        if dataset_id is None or not any(
            dataset.dataset_id == dataset_id
            for dataset in self.document.project.datasets
        ):
            self.details_label.clear()
            self.details_label.hide()
            return
        dataset = self._dataset(dataset_id)
        self.details_label.setText(
            "\n".join(
                (
                    f"源文件：{Path(dataset.source_path).name}",
                    f"光路：{self.beam_text(dataset_id)}",
                    f"仪器：{self.instrument_text(dataset_id)}",
                    f"状态：{self.status_text(dataset_id)} · "
                    f"SHA-256 {dataset.source_sha256[:12]}…",
                )
            )
        )
        self.details_label.setToolTip(
            f"{dataset.source_path}\nSHA-256：{dataset.source_sha256}"
        )
        self.details_label.show()

    def _tree_item(self, dataset: api.DatasetProject) -> QTreeWidgetItem:
        marker = self.status_marker(dataset.dataset_id)
        status = self.status_text(dataset.dataset_id)
        status_cell = f"{marker} {status}" if marker else status
        fit_marker = self.fit_status_marker(dataset.dataset_id)
        fit_status = self.fit_status_text(dataset.dataset_id)
        fit_cell = f"{fit_marker} {fit_status}" if fit_marker else fit_status
        values = (
            dataset.display_name or dataset.dataset_id,
            Path(dataset.source_path).name,
            self.beam_text(dataset.dataset_id),
            self.instrument_text(dataset.dataset_id),
            status_cell,
            dataset.source_sha256[:12],
            fit_cell,
        )
        item = QTreeWidgetItem(values)
        item.setData(0, Qt.ItemDataRole.UserRole, dataset.dataset_id)
        item.setToolTip(1, dataset.source_path)
        item.setToolTip(2, values[2])
        item.setToolTip(3, values[3])
        item.setToolTip(4, self._status_tooltip(dataset.dataset_id, status))
        item.setToolTip(5, dataset.source_sha256)
        item.setToolTip(6, self._fit_status_tooltip(dataset.dataset_id))
        return item

    def _status_tooltip(self, dataset_id: str, status: str) -> str:
        """Explain an abnormal source status inline; SHA detail lives here too."""
        if self._source_status_code(dataset_id) == "ok":
            return status
        warning = self.document.source_warning(dataset_id)
        return warning or status

    def _select_tree_item(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        dataset_id = str(current.data(0, Qt.ItemDataRole.UserRole))
        if dataset_id == self.active_dataset_id:
            return
        try:
            self.document.select_active_dataset(dataset_id)
        except Exception:
            blocker = QSignalBlocker(self.tree)
            self.tree.setCurrentItem(previous)
            del blocker
            return
        self.active_dataset_changed.emit(dataset_id)

    def _import_files(self) -> None:
        names, _selected = QFileDialog.getOpenFileNames(
            self,
            "导入 XRR 数据",
            "",
            DATA_FILTER,
        )
        if names:
            self._confirm_import(tuple(Path(name) for name in names), folder=False)

    def _import_folder(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "导入 XRR 数据文件夹")
        if name:
            self._confirm_import((Path(name),), folder=True)

    def _change_measurement_preset(self) -> None:
        """Scope a replacement preset to one file-selection attempt.

        Cancelling either chooser leaves the saved preset untouched. The transient
        flag is cleared even when import validation reports an error, so a later
        ordinary import can continue using the persisted measurement declaration.
        """
        self._force_preset_dialog = True
        try:
            self._import_files()
        finally:
            self._force_preset_dialog = False

    def _confirm_import(self, paths: tuple[Path, ...], *, folder: bool) -> None:
        try:
            preset = self.document.project.measurement_preset
            mapping = None
            recursive = False
            if preset is None or self._force_preset_dialog:
                dialog = ImportDialog(paths, folder_mode=folder, parent=self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                instrument = dialog.instrument_spec()
                preset = api.MeasurementPreset(
                    instrument.instrument_id or "default-measurement",
                    dialog.beam_spec(),
                    instrument,
                )
                mapping = dialog.column_mapping()
                recursive = dialog.recursive_folder_import()
            sources = self._folder_paths(paths[0], recursive) if folder else paths
            self.import_paths(sources, preset=preset, column_mapping=mapping)
        except (InterruptedError, OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "导入数据失败", f"{type(error).__name__}: {error}")

    def _folder_paths(self, folder: Path, recursive: bool) -> tuple[Path, ...]:
        entries = folder.rglob("*") if recursive else folder.glob("*")
        return tuple(
            sorted(
                (
                    path
                    for path in entries
                    if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
                ),
                key=lambda path: path.relative_to(folder).as_posix().casefold(),
            )
        )
