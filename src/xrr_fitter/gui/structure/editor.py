"""Concrete structure value editor and deterministic tree renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.structure.dialogs import BackingDialog, LayerDialog, PeriodicDialog

CommitStructure = Callable[[api.StructureSpec], object]
OxideAction = Callable[[], object]
TREE_HEADERS = ("名称", "材料", "密度", "厚度 (nm)", "粗糙度 (nm)", "重复")

# The tree annotates a drifting block in its tooltip rather than adding a
# column, so these mirror the dialog's Chinese labels for kind and target.
DRIFT_KIND_LABELS = {"linear": "线性", "sine": "正弦", "random": "随机"}
DRIFT_TARGET_LABELS = {"thickness": "厚度", "roughness": "粗糙度"}


def _material_text(material: api.MaterialSpec) -> str:
    return material.formula.strip() if material.formula is not None else "显式 SLD"


def _density_text(material: api.MaterialSpec) -> str:
    density = material.bulk_density_g_cm3
    return "" if density is None else f"{density:g}"


def _nm(value_a: float) -> str:
    return f"{value_a / 10.0:g}"


def _transition_text(transition: api.InterfaceTransition) -> str:
    """Spell out the transition in the roughness column without adding one."""
    branches = "、".join(
        f"{branch.kind} 权重 {branch.weight:g} 宽度 {_nm(branch.thickness_a)} nm" for branch in transition.branches
    )
    return f"界面过渡取代粗糙度：{branches}；切片上限 {_nm(transition.microslab_max_a)} nm"


def _drift_tooltip(drift: api.DriftSpec) -> str:
    """Name the drift law and target in the block row without a new column."""
    kind = DRIFT_KIND_LABELS.get(drift.kind, drift.kind)
    target = DRIFT_TARGET_LABELS.get(drift.target, drift.target)
    return f"漂移：{kind} · 作用于{target}"


def _oxide_identity(value: object) -> tuple[object, ...]:
    material = value.oxide_material
    formula = material.formula if isinstance(material, api.MaterialSpec) else material
    return (
        value.base_material,
        formula,
        value.location,
        value.oxide_table_version,
    )


class StructureEditor(QWidget):
    """Edit a detached structure while a panel owns API publication."""

    def __init__(
        self,
        commit_structure: CommitStructure,
        accept_oxide: OxideAction,
        refuse_oxide: OxideAction,
        parent: QWidget | None = None,
        *,
        master_seed_source: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("structureEditor")
        self.setAccessibleName("结构编辑")
        self._commit_structure = commit_structure
        self._accept_oxide = accept_oxide
        self._refuse_oxide = refuse_oxide
        self._master_seed_source = master_seed_source
        self._structure: api.StructureSpec | None = None
        self._decisions: tuple[api.OxideDecision, ...] = ()
        self._suggestion: api.OxideSuggestion | None = None
        self._build_widgets()
        self.clear()

    def _build_widgets(self) -> None:
        self.tree = QTreeWidget()
        self.tree.setObjectName("structureTree")
        self.tree.setHeaderLabels(TREE_HEADERS)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._refresh_action_state)
        self.add_layer_button = self._button("添加普通层", "addLayerButton")
        self.add_periodic_button = self._button("添加周期块", "addPeriodicBlockButton")
        self.edit_backing_button = self._button("编辑基底", "editBackingButton")
        self.remove_button = self._button("删除", "removeComponentButton")
        self.up_button = self._button("上移", "moveComponentUpButton")
        self.down_button = self._button("下移", "moveComponentDownButton")
        self.oxide_button = self._button("添加自然氧化层", "oxideSuggestionButton")
        self.refuse_button = self._button("忽略建议", "oxideSuggestionRefuseButton")
        self.error_label = QLabel()
        self.error_label.setObjectName("structureValidationError")
        self.error_label.setProperty("statusKind", "error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._connect_actions()
        add_row = QHBoxLayout()
        add_row.setSpacing(theme.SPACE_XS)
        add_row.addWidget(self.add_layer_button)
        add_row.addWidget(self.add_periodic_button)
        add_row.addWidget(self.edit_backing_button)
        add_row.addStretch(1)
        edit_row = QHBoxLayout()
        edit_row.setSpacing(theme.SPACE_XS)
        edit_row.addWidget(self.remove_button)
        edit_row.addWidget(self.up_button)
        edit_row.addWidget(self.down_button)
        edit_row.addStretch(1)
        oxide_row = QHBoxLayout()
        oxide_row.setSpacing(theme.SPACE_XS)
        oxide_row.addWidget(self.oxide_button)
        oxide_row.addWidget(self.refuse_button)
        oxide_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addLayout(add_row)
        layout.addLayout(edit_row)
        layout.addLayout(oxide_row)
        layout.addWidget(self.tree)
        layout.addWidget(self.error_label)

    def _button(self, text: str, name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(name)
        button.setAccessibleName(text)
        button.setProperty("commandBar", True)
        return button

    def _connect_actions(self) -> None:
        self.add_layer_button.clicked.connect(self._choose_layer)
        self.add_periodic_button.clicked.connect(self._choose_periodic)
        self.edit_backing_button.clicked.connect(self._choose_backing)
        self.remove_button.clicked.connect(self._remove_selected)
        self.up_button.clicked.connect(lambda: self._move_selected(-1))
        self.down_button.clicked.connect(lambda: self._move_selected(1))
        self.oxide_button.clicked.connect(self._accept_oxide)
        self.refuse_button.clicked.connect(self._refuse_oxide)

    @property
    def structure(self) -> api.StructureSpec | None:
        return self._structure

    def load(
        self,
        structure: api.StructureSpec,
        decisions: tuple[api.OxideDecision, ...] = (),
    ) -> None:
        self._structure = structure
        self._decisions = tuple(decisions)
        self.error_label.hide()
        self._render()
        self._refresh_oxide()

    def clear(self) -> None:
        self._structure = None
        self._decisions = ()
        self._suggestion = None
        self.tree.clear()
        for button in (
            self.add_layer_button,
            self.add_periodic_button,
            self.edit_backing_button,
            self.remove_button,
            self.up_button,
            self.down_button,
        ):
            button.setEnabled(False)
        self.oxide_button.hide()
        self.refuse_button.hide()
        self.error_label.hide()

    def current_oxide_suggestion(self) -> api.OxideSuggestion | None:
        return self._suggestion

    def add_layer(self, layer: api.LayerSpec) -> None:
        structure = self._require_structure()
        self._commit_structure(replace(structure, components=(*structure.components, layer)))

    def add_periodic_block(self, block: api.PeriodicBlock) -> None:
        structure = self._require_structure()
        self._commit_structure(replace(structure, components=(*structure.components, block)))

    def set_backing(self, backing: api.MaterialSpec, roughness_a: float) -> None:
        structure = self._require_structure()
        self._commit_structure(
            replace(
                structure,
                backing=backing,
                backing_roughness_a=roughness_a,
            )
        )

    def remove_component(self, index: int) -> None:
        structure = self._require_structure()
        self._require_index(index, len(structure.components))
        components = (*structure.components[:index], *structure.components[index + 1 :])
        self._commit_structure(replace(structure, components=components))

    def move_component(self, source: int, destination: int) -> bool:
        structure = self._require_structure()
        count = len(structure.components)
        self._require_index(source, count)
        self._require_index(destination, count)
        if source == destination:
            return False
        components = list(structure.components)
        component = components.pop(source)
        components.insert(destination, component)
        self._commit_structure(replace(structure, components=tuple(components)))
        return True

    def _require_structure(self) -> api.StructureSpec:
        if self._structure is None:
            raise ValueError("active dataset has no structure")
        return self._structure

    def _require_index(self, index: int, count: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
            raise IndexError("component index out of range")

    def _render(self) -> None:
        structure = self._require_structure()
        self.tree.clear()
        self.tree.addTopLevelItem(self._medium_item("Air", structure.fronting))
        for index, component in enumerate(structure.components):
            self.tree.addTopLevelItem(self._component_item(component, index))
        self.tree.addTopLevelItem(self._medium_item("基底", structure.backing))
        self.add_layer_button.setEnabled(True)
        self.add_periodic_button.setEnabled(True)
        self.edit_backing_button.setEnabled(True)
        self._refresh_action_state()

    def _medium_item(self, name: str, material: api.MaterialSpec) -> QTreeWidgetItem:
        item = QTreeWidgetItem((name, _material_text(material), _density_text(material)))
        item.setData(0, Qt.ItemDataRole.UserRole, None)
        return item

    def _component_item(self, component: object, index: int) -> QTreeWidgetItem:
        if isinstance(component, api.PeriodicBlock):
            item = QTreeWidgetItem((component.name, "周期", "", "", "", str(component.repeats)))
            if component.drift is not None:
                item.setToolTip(0, _drift_tooltip(component.drift))
            for layer in component.layers:
                item.addChild(self._layer_item(layer))
        elif isinstance(component, api.GradientLayerSpec):
            item = QTreeWidgetItem(
                (component.name, "SLD gradient", "", _nm(component.thickness_a), _nm(component.roughness_a))
            )
        else:
            item = self._layer_item(component)
        item.setData(0, Qt.ItemDataRole.UserRole, index)
        return item

    def _layer_item(self, layer: api.LayerSpec) -> QTreeWidgetItem:
        item = QTreeWidgetItem(
            (
                layer.name,
                _material_text(layer.material),
                _density_text(layer.material),
                _nm(layer.thickness_a),
                _nm(layer.roughness_a),
            )
        )
        if layer.transition is not None:
            item.setToolTip(4, _transition_text(layer.transition))
        return item

    def _refresh_oxide(self) -> None:
        structure = self._require_structure()
        decided = {_oxide_identity(value) for value in self._decisions}
        self._suggestion = next(
            (value for value in api.suggest_oxide_layers(structure) if _oxide_identity(value) not in decided),
            None,
        )
        visible = self._suggestion is not None
        self.oxide_button.setVisible(visible)
        self.refuse_button.setVisible(visible)

    def _selected_index(self) -> int | None:
        item = self.tree.currentItem()
        if item is None or item.parent() is not None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def _refresh_action_state(self, *_args) -> None:
        index = self._selected_index()
        count = 0 if self._structure is None else len(self._structure.components)
        self.remove_button.setEnabled(index is not None)
        self.up_button.setEnabled(index is not None and index > 0)
        self.down_button.setEnabled(index is not None and index + 1 < count)

    def _remove_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            self.remove_component(index)

    def _move_selected(self, offset: int) -> None:
        index = self._selected_index()
        if index is not None:
            self.move_component(index, index + offset)

    def _choose_layer(self) -> None:
        LayerDialog(self, commit_layer=self.add_layer).exec()

    def _choose_periodic(self) -> None:
        structure = self._require_structure()
        PeriodicDialog(
            self,
            commit_block=self.add_periodic_block,
            master_seed=self._master_seed(),
            block_offset=len(structure.components),
        ).exec()

    def _master_seed(self) -> int:
        if self._master_seed_source is None:
            return 0
        return self._master_seed_source()

    def _choose_backing(self) -> None:
        structure = self._require_structure()
        BackingDialog(
            structure.backing,
            structure.backing_roughness_a,
            self,
            commit_backing=self.set_backing,
        ).exec()
