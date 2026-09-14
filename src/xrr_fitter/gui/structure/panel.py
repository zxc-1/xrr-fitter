"""Project-aware structure and native-oxide workflow coordinator."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.structure.editor import StructureEditor


class StructurePanel(QWidget):
    """Commit structure workflows through API services after complete validation."""

    structure_changed = Signal(str, object)
    oxide_decision_changed = Signal(str, object)
    structure_analyzed = Signal(str, object)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("structurePanel")
        # The dock title bar already names this panel; see DataPanel for why the
        # heading is retained but hidden.
        heading = QLabel("样品结构")
        heading.setObjectName("structurePanelHeader")
        heading.setProperty("sectionHeader", True)
        heading.hide()
        self.initialize_button = QPushButton("初始化结构")
        self.initialize_button.setObjectName("initializeStructureButton")
        self.initialize_button.setAccessibleName("初始化结构")
        self.initialize_button.setProperty("primary", True)
        self.initialize_button.clicked.connect(self._initialize_default_structure)
        self.editor = StructureEditor(
            self.set_structure,
            self.accept_current_oxide,
            self.refuse_current_oxide,
            self,
            master_seed_source=lambda: self.document.project.master_seed,
        )
        # 编辑器上一次装的是哪一份数据集的结构。重载时靠它区分「同一份被自己改了」和
        # 「换了一份」——只有前者该落回原来选中的那一层。
        self._loaded_dataset_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(heading)
        layout.addWidget(self.initialize_button)
        layout.addWidget(self.editor)
        document.project_changed.connect(self._refresh)
        self._refresh()

    @property
    def active_dataset_id(self) -> str | None:
        return self.document.active_dataset_id

    @property
    def structure(self) -> api.StructureSpec | None:
        return self.editor.structure

    def set_structure(self, structure: api.StructureSpec) -> bool:
        dataset_id = self._require_active_dataset_id()
        current = self.document.project
        updated = api.set_structure(current, dataset_id, structure)
        if updated is current:
            return False
        self.document.replace_project(updated)
        persisted = self._dataset(updated, dataset_id).structure
        self.structure_changed.emit(dataset_id, persisted)
        return True

    def initialize_structure(
        self,
        backing_formula: str,
        backing_density_g_cm3: float,
    ) -> bool:
        formula = backing_formula.strip()
        structure = api.StructureSpec(
            api.MaterialSpec("Air", None, None, 0.0j),
            (),
            api.MaterialSpec(formula, formula, backing_density_g_cm3),
        )
        return self.set_structure(structure)

    def _initialize_default_structure(self) -> None:
        self.initialize_structure("Si", 2.329)

    def add_layer(self, layer: api.LayerSpec) -> None:
        self.editor.add_layer(layer)

    def add_periodic_block(self, block: api.PeriodicBlock) -> None:
        self.editor.add_periodic_block(block)

    def remove_component(self, index: int) -> None:
        self.editor.remove_component(index)

    def move_component(self, source: int, destination: int) -> bool:
        return self.editor.move_component(source, destination)

    def current_oxide_suggestion(self) -> api.OxideSuggestion | None:
        return self.editor.current_oxide_suggestion()

    def accept_current_oxide(self) -> None:
        """接受建议就是往层堆叠里插了一层——落盘之后要和手改一样宣布出去。

        这条路不经过 ``set_structure``，用的是 ``api.accept_oxide_suggestion`` 自己那条提交；
        但读者眼里两条路结果相同：层堆叠多了一层。底栏那句「结构已修改（未保存）」（设计稿
        帧③）听的是 ``structure_changed``，只发 ``oxide_decision_changed`` 会让这一次编辑
        无声无息。拒绝那一路不发——它只记一个决定，结构一个字没动。
        """
        suggestion = self._require_oxide_suggestion()
        dataset_id = self._require_active_dataset_id()
        updated = api.accept_oxide_suggestion(
            self.document.project,
            dataset_id,
            suggestion,
        )
        self.document.replace_project(updated)
        dataset = self._dataset(updated, dataset_id)
        decision = dataset.oxide_decisions[-1]
        self.structure_changed.emit(dataset_id, dataset.structure)
        self.oxide_decision_changed.emit(dataset_id, decision)

    def refuse_current_oxide(self) -> None:
        suggestion = self._require_oxide_suggestion()
        formula = suggestion.oxide_material.formula
        if formula is None:
            raise ValueError("oxide suggestion requires a formula-backed material")
        decision = api.OxideDecision(
            suggestion.base_material,
            formula,
            suggestion.location,
            False,
            suggestion.oxide_table_version,
        )
        dataset_id = self._require_active_dataset_id()
        updated = api.record_oxide_decision(
            self.document.project,
            dataset_id,
            decision,
        )
        self.document.replace_project(updated)
        self.oxide_decision_changed.emit(dataset_id, decision)

    def analyze_structure(self) -> api.StructureEvidence:
        dataset_id = self._require_active_dataset_id()
        evidence = api.analyze_structure(self.document.project, dataset_id)
        self.structure_analyzed.emit(dataset_id, evidence)
        return evidence

    def _require_oxide_suggestion(self) -> api.OxideSuggestion:
        suggestion = self.current_oxide_suggestion()
        if suggestion is None:
            raise ValueError("no current oxide suggestion")
        return suggestion

    def _require_active_dataset_id(self) -> str:
        dataset_id = self.active_dataset_id
        if dataset_id is None:
            raise ValueError("an active dataset is required")
        return dataset_id

    def _refresh(self, *_args) -> None:
        dataset_id = self.active_dataset_id
        if dataset_id is None:
            self.initialize_button.hide()
            self._loaded_dataset_id = None
            self.editor.clear()
            return
        dataset = self._dataset(self.document.project, dataset_id)
        if dataset.structure is None:
            self.initialize_button.show()
            self.initialize_button.setEnabled(True)
            self._loaded_dataset_id = None
            self.editor.clear()
            return
        self.initialize_button.hide()
        # 还是刚才那份数据集的结构，就落回刚才选中的那一层：改一层的数字走的正是这条重
        # 载路径，丢了选中等于让使用者为自己的每一次编辑重新点一次同一层。换数据集时不
        # 落回——那边的第 1 行是另一层。
        same_dataset = dataset_id == self._loaded_dataset_id
        self._loaded_dataset_id = dataset_id
        self.editor.load(dataset.structure, dataset.oxide_decisions, keep_selection=same_dataset)

    def _dataset(
        self,
        project: api.XrrProject,
        dataset_id: str,
    ) -> api.DatasetProject:
        matches = tuple(dataset for dataset in project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]
