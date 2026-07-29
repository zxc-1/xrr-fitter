"""Dataset import panel backed exclusively by the supported application API."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.data.import_dialog import ImportDialog
from xrr_fitter.gui.data.mask_editor import MaskEditor
from xrr_fitter.gui.document import ProjectDocument


DATA_FILTER = "XRR 数据 (*.xy *.dat *.txt);;所有文件 (*)"
SUPPORTED_SUFFIXES = {".xy", ".dat", ".txt"}
TREE_HEADERS = ("数据集", "源文件", "光路", "仪器", "状态", "SHA-256")


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
        self._build_controls()
        self._mask_editor = MaskEditor(document, self)
        self._mask_editor.mask_changed.connect(self.mask_changed.emit)
        document.project_changed.connect(self._render_project)
        document.source_validation_changed.connect(self._render_project)
        self._render_project()

    def _build_controls(self) -> None:
        self.import_files_button = QPushButton("导入文件")
        self.import_files_button.setObjectName("importFilesButton")
        self.import_folder_button = QPushButton("导入文件夹")
        self.import_folder_button.setObjectName("importFolderButton")
        self.import_files_button.clicked.connect(self._import_files)
        self.import_folder_button.clicked.connect(self._import_folder)
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
        self.tree.currentItemChanged.connect(self._select_tree_item)
        buttons = QHBoxLayout()
        buttons.addWidget(self.import_files_button)
        buttons.addWidget(self.import_folder_button)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.tree)

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

    def status_text(self, dataset_id: str) -> str:
        dataset = self._dataset(dataset_id)
        try:
            if self.document.source_status(dataset_id) != "ok":
                return "源文件异常"
        except KeyError:
            pass
        return "可拟合" if sum(dataset.fit_mask) >= 30 else "数据点不足"

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

    def _tree_item(self, dataset: api.DatasetProject) -> QTreeWidgetItem:
        values = (
            dataset.display_name or dataset.dataset_id,
            Path(dataset.source_path).name,
            self.beam_text(dataset.dataset_id),
            self.instrument_text(dataset.dataset_id),
            self.status_text(dataset.dataset_id),
            dataset.source_sha256[:12],
        )
        item = QTreeWidgetItem(values)
        item.setData(0, Qt.ItemDataRole.UserRole, dataset.dataset_id)
        item.setToolTip(1, dataset.source_path)
        item.setToolTip(2, values[2])
        item.setToolTip(3, values[3])
        item.setToolTip(4, values[4])
        item.setToolTip(5, dataset.source_sha256)
        return item

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

    def _confirm_import(self, paths: tuple[Path, ...], *, folder: bool) -> None:
        dialog = ImportDialog(paths, folder_mode=folder, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            if folder:
                self.add_folder(
                    paths[0],
                    beam=dialog.beam_spec(),
                    instrument=dialog.instrument_spec(),
                    recursive=dialog.recursive_folder_import(),
                    column_mapping=dialog.column_mapping(),
                )
            else:
                self.add_paths(
                    paths,
                    beam=dialog.beam_spec(),
                    instrument=dialog.instrument_spec(),
                    column_mapping=dialog.column_mapping(),
                )
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "导入数据失败", f"{type(error).__name__}: {error}")
