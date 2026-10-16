"""The modal form that builds one periodic block: its layer table and its optional drift law."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.structure.fields import dialog_buttons, number_editor, show_rejection, table_cell

# The amount field means a different physical quantity per drift law, so its
# label follows the selected kind rather than staying a single fixed string.
DRIFT_AMOUNT_LABELS = {
    "linear": "每周期增量",
    "sine": "正弦幅度",
    "random": "随机幅度",
}

# QSpinBox stores a signed 32-bit int, so a 64-bit project master seed must be
# folded into range before it can seed the derived random-drift field.
DRIFT_SEED_MODULUS = 2_147_483_648

PERIODIC_COLUMNS = ("名称", "化学式", "密度 (g/cm³)", "厚度 (nm)", "粗糙度 (nm)")

# 「增量」读起来像一个长度，其实是乘在基值上的比例（``fit.drift``：第 k 个周期取
# base·(1 + s·c_k)）。不说清楚，读者会把 0.5 nm 的漂移填成 0.5——那是每周期涨一半。
DRIFT_AMOUNT_TOOLTIP = (
    "无量纲比例 s：第 k 个周期的厚度（或粗糙度）= 基值 × (1 + s·c_k)。"
    "线性 c_k = k；正弦 c_k = sin(2πk/周期 + 相位)；随机 c_k 取自 [-1, 1] 的均匀分布。"
    "例如 0.01 表示每个周期比上一周期厚 1%（线性）。"
)


def _cell_text(table: QTableWidget, row: int, column: int) -> str:
    item = table.item(row, column)
    return "" if item is None else item.text().strip()


def _prefill(layer: api.LayerSpec) -> tuple[str, str, str, str, str]:
    """编辑已有周期块时一行里各格写什么；``repr`` 保证没改过的格子能逐位比回原层。"""
    material = layer.material
    density = "" if material.bulk_density_g_cm3 is None else repr(material.bulk_density_g_cm3)
    return (
        layer.name,
        material.formula or "",
        density,
        repr(layer.thickness_a / 10.0),
        repr(layer.roughness_a / 10.0),
    )


class PeriodicDialog(QDialog):
    """Collect one ordered periodic cell using an explicit layer table."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        commit_block: Callable[[api.PeriodicBlock], object] | None = None,
        master_seed: int | None = None,
        block_offset: int | None = None,
        block: api.PeriodicBlock | None = None,
    ) -> None:
        super().__init__(parent)
        title = "添加周期块" if block is None else "编辑周期块"
        self.setObjectName("periodicBlockDialog")
        self.setWindowTitle(title)
        self.setAccessibleName(title)
        self.setModal(True)
        self._commit_block = commit_block
        self._block: api.PeriodicBlock | None = None
        # 编辑已有块时，块顶粗糙度这张表里没有格子；不带上它，确定一次就把它悄悄抹掉。
        self._top_roughness_a = None if block is None else block.top_roughness_a
        self._master_seed = master_seed or 0
        self._block_offset = block_offset or 0
        self.name_editor = QLineEdit()
        self.name_editor.setObjectName("periodicNameInput")
        self.repeats_editor = QSpinBox()
        self.repeats_editor.setObjectName("periodicRepeatsInput")
        self.repeats_editor.setRange(1, 1_000_000)
        self.repeats_editor.setValue(2)
        self._build_table()
        self.error_label = QLabel()
        self.error_label.setObjectName("periodicDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._build_drift_inputs()
        self.buttons = dialog_buttons("periodicDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self.drift_kind.currentIndexChanged.connect(self._sync_drift_inputs)
        self.repeats_editor.valueChanged.connect(self._sync_drift_warning)
        self.drift_seed.valueChanged.connect(self._sync_drift_seed_source)
        self._arrange()
        self._sync_drift_inputs()
        if block is not None:
            self._load_block(block)
        # 预填完了才接：预填时先写化学式、后写密度，接早了会先塞一个名义密度进去。
        self.table.itemChanged.connect(self._fill_nominal_density)

    def _build_table(self) -> None:
        self.table = QTableWidget(0, len(PERIODIC_COLUMNS))
        self.table.setObjectName("periodicLayerTable")
        self.table.setHorizontalHeaderLabels(PERIODIC_COLUMNS)
        # 每一行编辑已有块时对应的原层：没改过的行原样写回，密度倍率、SLD 覆盖这些表里没有
        # 的字段不会因为确定了一次就丢。新加的行没有原层。
        self._templates: list[api.LayerSpec | None] = []
        self.add_row_button = QPushButton("＋ 添加一行")
        self.add_row_button.setObjectName("periodicAddRowButton")
        self.add_row_button.clicked.connect(self._append_row)
        self.remove_row_button = QPushButton("删除选中行")
        self.remove_row_button.setObjectName("periodicRemoveRowButton")
        self.remove_row_button.setToolTip("一个周期至少要有一层")
        self.remove_row_button.clicked.connect(self._remove_row)
        for _ in range(2):
            self._append_row()

    def _append_row(self, template: api.LayerSpec | None = None) -> None:
        self.table.insertRow(self.table.rowCount())
        self._templates.append(template if isinstance(template, api.LayerSpec) else None)
        self.remove_row_button.setEnabled(self.table.rowCount() > 1)

    def _remove_row(self) -> None:
        if self.table.rowCount() <= 1:
            return
        row = self.table.currentRow()
        row = self.table.rowCount() - 1 if row < 0 else row
        self.table.removeRow(row)
        del self._templates[row]
        self.remove_row_button.setEnabled(self.table.rowCount() > 1)

    def _load_block(self, block: api.PeriodicBlock) -> None:
        self.name_editor.setText(block.name)
        self.repeats_editor.setValue(block.repeats)
        self.table.setRowCount(0)
        self._templates.clear()
        for row, layer in enumerate(block.layers):
            self._append_row(layer)
            for column, text in enumerate(_prefill(layer)):
                self.table.setItem(row, column, QTableWidgetItem(text))
        if block.drift is not None:
            self._load_drift(block.drift)

    def _load_drift(self, drift: api.DriftSpec) -> None:
        self.drift_kind.setCurrentIndex(self.drift_kind.findData(drift.kind))
        self.drift_target.setCurrentIndex(self.drift_target.findData(drift.target))
        self.drift_amount.setValue(drift.amount)
        if drift.kind == "sine":
            self.drift_period.setValue(drift.period)
            self.drift_phase.setValue(drift.phase)
        if drift.kind == "random":
            self.drift_seed.setValue(drift.seed)

    def _fill_nominal_density(self, item: QTableWidgetItem) -> None:
        """化学式那一格填的是表里有的材料、密度格还空着，就带入名义密度。"""
        if item.column() != 1:
            return
        nominal = api.INITIAL_DENSITIES_G_CM3.get(item.text().strip())
        density = self.table.item(item.row(), 2)
        if nominal is not None and (density is None or not density.text().strip()):
            self.table.setItem(item.row(), 2, QTableWidgetItem(f"{nominal:g}"))

    def _build_drift_inputs(self) -> None:
        self.drift_kind = QComboBox()
        self.drift_kind.setObjectName("periodicDriftKind")
        self.drift_kind.addItem("无", None)
        self.drift_kind.addItem("线性", "linear")
        self.drift_kind.addItem("正弦", "sine")
        self.drift_kind.addItem("随机", "random")
        self.drift_target = QComboBox()
        self.drift_target.setObjectName("periodicDriftTarget")
        self.drift_target.addItem("厚度", "thickness")
        self.drift_target.addItem("粗糙度", "roughness")
        self.drift_amount = number_editor("periodicDriftAmount", -1_000_000.0, 0.0)
        self.drift_amount.setToolTip(DRIFT_AMOUNT_TOOLTIP)
        self.drift_amount_label = QLabel("每周期增量")
        self.drift_amount_label.setToolTip(DRIFT_AMOUNT_TOOLTIP)
        self.drift_period = number_editor("periodicDriftPeriod", 0.000001, 4.0)
        self.drift_phase = number_editor("periodicDriftPhase", -1_000_000.0, 0.0)
        self.drift_seed = QSpinBox()
        self.drift_seed.setObjectName("periodicDriftSeed")
        self.drift_seed.setRange(0, DRIFT_SEED_MODULUS - 1)
        seed = (self._master_seed + self._block_offset) % DRIFT_SEED_MODULUS
        self._default_drift_seed = seed
        self.drift_seed.setValue(seed)
        self.drift_seed_source = QLabel(
            f"种子来源：工程种子 {self._master_seed} + 块偏移 {self._block_offset}，折叠至 32 位 = {seed}"
        )
        self.drift_seed_source.setObjectName("periodicDriftSeedSource")
        self.drift_seed_source.setWordWrap(True)
        self.drift_warning = QLabel()
        self.drift_warning.setObjectName("periodicDriftWarning")
        self.drift_warning.setWordWrap(True)

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("名称", self.name_editor)
        form.addRow("重复次数", self.repeats_editor)
        self._drift_form = QFormLayout()
        self._drift_form.addRow("漂移类型", self.drift_kind)
        self._drift_form.addRow("作用目标", self.drift_target)
        self._drift_form.addRow(self.drift_amount_label, self.drift_amount)
        self._drift_form.addRow("正弦周期", self.drift_period)
        self._drift_form.addRow("正弦相位", self.drift_phase)
        self._drift_form.addRow("随机种子", self.drift_seed)
        rows = QHBoxLayout()
        rows.addWidget(self.add_row_button)
        rows.addWidget(self.remove_row_button)
        rows.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table)
        layout.addLayout(rows)
        layout.addLayout(self._drift_form)
        layout.addWidget(self.drift_seed_source)
        layout.addWidget(self.drift_warning)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _accept_fields(self) -> None:
        try:
            layers = tuple(self._layer_at(row) for row in range(self.table.rowCount()))
            candidate = api.PeriodicBlock(
                self.name_editor.text().strip(),
                layers,
                self.repeats_editor.value(),
                top_roughness_a=self._top_roughness_a,
                drift=self._drift_spec(),
            )
            if self._commit_block is not None:
                self._commit_block(candidate)
        except (TypeError, ValueError) as error:
            show_rejection(self, self.error_label, error)
            return
        self._block = candidate
        self.error_label.hide()
        self.accept()

    def _sync_drift_inputs(self, *_args: object) -> None:
        """Show only the fields the selected drift law actually consumes.

        A hidden field would otherwise feed a value the model rejects, so the
        row visibility mirrors ``DriftSpec``'s per-kind requirements exactly.
        """
        kind = self.drift_kind.currentData()
        active = kind is not None
        self._drift_form.setRowVisible(self.drift_target, active)
        self._drift_form.setRowVisible(self.drift_amount, active)
        self._drift_form.setRowVisible(self.drift_period, kind == "sine")
        self._drift_form.setRowVisible(self.drift_phase, kind == "sine")
        self._drift_form.setRowVisible(self.drift_seed, kind == "random")
        self.drift_seed_source.setVisible(kind == "random")
        self.drift_warning.setVisible(active)
        if active:
            self.drift_amount_label.setText(DRIFT_AMOUNT_LABELS[kind])
        self._sync_drift_warning()

    def _sync_drift_warning(self, *_args: object) -> None:
        repeats = self.repeats_editor.value()
        self.drift_warning.setText(
            f"漂移块退出矩阵幂快路径：{repeats} 个周期的雅可比复杂度为 O({repeats})，而非无漂移时的 O(log {repeats})。"
        )

    def _sync_drift_seed_source(self, *_args: object) -> None:
        seed = self.drift_seed.value()
        if seed == self._default_drift_seed:
            text = f"种子来源：工程种子 {self._master_seed} + 块偏移 {self._block_offset}，折叠至 32 位 = {seed}"
        else:
            text = f"种子来源：用户编辑 = {seed}（默认折叠值 {self._default_drift_seed}）"
        self.drift_seed_source.setText(text)

    def _drift_spec(self) -> api.DriftSpec | None:
        kind = self.drift_kind.currentData()
        if kind is None:
            return None
        target = self.drift_target.currentData()
        amount = self.drift_amount.value()
        if kind == "sine":
            return api.DriftSpec(
                kind,
                target,
                amount,
                period=self.drift_period.value(),
                phase=self.drift_phase.value(),
            )
        if kind == "random":
            return api.DriftSpec(kind, target, amount, seed=self.drift_seed.value())
        return api.DriftSpec(kind, target, amount)

    def _layer_at(self, row: int) -> api.LayerSpec:
        template = self._templates[row]
        texts = tuple(_cell_text(self.table, row, column) for column in range(len(PERIODIC_COLUMNS)))
        if template is not None and texts == _prefill(template):
            return template
        fields = tuple(self._cell(row, column) for column in range(len(PERIODIC_COLUMNS)))
        name, formula, density, thickness, roughness = fields
        material = api.MaterialSpec(formula, formula, float(density))
        return api.LayerSpec(
            name,
            material,
            float(thickness) * 10.0,
            roughness_a=float(roughness) * 10.0,
        )

    def _cell(self, row: int, column: int) -> str:
        return table_cell(self.table, row, column, "periodic layer")

    def block(self) -> api.PeriodicBlock:
        if self._block is None:
            raise LookupError("periodic block has not been confirmed")
        return self._block
