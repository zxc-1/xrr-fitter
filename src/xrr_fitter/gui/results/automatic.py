"""Automatic point/layer and group-uniformity result tables."""

from __future__ import annotations

from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

STATUS_TEXT = {
    "passed": "通过",
    "refining": "精修中",
    "review": "需复核",
    "failed": "失败",
    "pending": "待处理",
    "not_run": "未运行",
}


def automatic_status_text(status: object) -> str:
    value = getattr(status, "value", status)
    return STATUS_TEXT.get(str(value), str(value))


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"


def _set_row(table: QTableWidget, row: int, values: tuple[str, ...]) -> None:
    for column, value in enumerate(values):
        table.setItem(row, column, QTableWidgetItem(value))


class AutomaticPointLayerTable(QTableWidget):
    HEADERS = (
        "点位",
        "状态",
        "统计",
        "材料",
        "厚度 (Å)",
        "粗糙度 (Å)",
        "SLD 实部 (Å⁻²)",
        "SLD 吸收部 (Å⁻²)",
        "有效电子密度 (Å⁻³)",
        "密度结果 (g/cm³)",
        "密度说明",
    )

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setObjectName("automaticPointLayerTable")
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def project_summary(self, summary: object) -> None:
        rows = tuple((dataset, layer) for dataset in summary.datasets for layer in dataset.layers)
        self.setRowCount(len(rows))
        self.setVisible(bool(rows))
        for row, (dataset, layer) in enumerate(rows):
            _set_row(
                self,
                row,
                (
                    dataset.dataset_id,
                    automatic_status_text(dataset.status),
                    "是" if dataset.statistics_member else "否",
                    layer.material_name,
                    _number(layer.thickness_a),
                    _number(layer.roughness_a),
                    _number(layer.sld_real_a2),
                    _number(layer.sld_imag_a2),
                    _number(layer.electron_density_a3),
                    _number(layer.fitted_density_g_cm3),
                    layer.density_note or "",
                ),
            )


class AutomaticUniformityTable(QTableWidget):
    HEADERS = (
        "组",
        "层",
        "数量",
        "均值 (Å)",
        "最小值 (Å)",
        "最大值 (Å)",
        "总体标准差 (Å)",
        "CV (%)",
        "相对极差 (%)",
    )

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setObjectName("automaticUniformityTable")
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def project_summary(self, summary: object) -> None:
        self.setRowCount(len(summary.uniformity))
        self.setVisible(bool(summary.uniformity))
        for row, value in enumerate(summary.uniformity):
            _set_row(
                self,
                row,
                (
                    value.fit_group_id,
                    f"{value.layer_index}: {value.material_name}",
                    str(value.count),
                    _number(value.mean_thickness_a),
                    _number(value.minimum_thickness_a),
                    _number(value.maximum_thickness_a),
                    _number(value.population_std_a),
                    _number(value.cv_percent),
                    _number(value.relative_range_percent),
                ),
            )
