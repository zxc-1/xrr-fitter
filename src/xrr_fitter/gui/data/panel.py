"""Dataset import panel backed exclusively by the supported application API.

The panel owns file selection, measurement presets, filename material previews,
substrate confirmation, and fit-mask editing. Every mutation adopts the immutable
project returned by ``xrr_fitter.api``; widgets never patch domain objects in
place. Batch failures remain visible per source while successful imports update
the active dataset and the compact source/instrument summary atomically.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.data.dataset_card import DATASET_CARD_ROLE, DatasetCardDelegate, dataset_card
from xrr_fitter.gui.data.import_dialog import ImportDialog
from xrr_fitter.gui.data.mask_editor import MaskEditor
from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.metrics import UNKNOWN
from xrr_fitter.gui.fitting.metrics import _trend as metrics_trend
from xrr_fitter.gui.navigation.panel import STRUCTURE_STEP_INDEX
from xrr_fitter.gui.sizing import ContentSizedTree

DATA_FILTER = "XRR 数据 (*.xy *.dat *.txt);;所有文件 (*)"
SUPPORTED_SUFFIXES = {".xy", ".dat", ".txt"}
# 七列仍是模型的形状——详情标签和各列 tooltip 都从这里取值，所以列没有删。显示则
# 收成一列自绘卡片（见 dataset_card）：设计稿的 ``.ds`` 没有表头也没有第二列。
# 这里此前还有 ``COMPACT_COLUMNS = (0, 4, 6)`` 和 ``DETAIL_COLUMNS = (1, 2, 3, 5)``
# 划分「显示哪几列、收起哪几列」；一列卡片之后这条界线只剩「第 0 列」一句，两个常量
# 都没有读者了。
TREE_HEADERS = ("数据集", "源文件", "光路", "仪器", "状态", "SHA-256", "拟合")

# Precise, glanceable rendering of each source-validation status. The marker
# turns a buried "源文件异常" into an at-a-glance signal, and the label names
# the exact failure so the fix is obvious without opening the detail pane.
SOURCE_STATUS_VISUALS = {
    "ok": ("", "源文件正常"),
    "missing": ("⛔", "源文件缺失"),
    "unreadable": ("⚠", "源文件无法读取"),
    "hash_mismatch": ("⚠", "源文件内容已改变"),
}

# 设计稿帧③ 左栏底下那句话（HTML 640）。它不清点数据集，回答的是改结构这一步真正会问的
# 那个问题：我编的这叠层落在哪几条曲线上、哪些东西仍旧各集一份。清点在这一步是废话——
# 数据集树就在这句话上面，几行数得出来。
SHARED_STRUCTURE_SUMMARY = "结构对全部可拟合数据集共享 · 每集独立仪器/标度"

# 批量导入把「文件 → 数据集 + 层堆叠」这两步推断摆出来。两者都只从文件名（或同名父文件夹）
# 推得，源文件一个字节都没读，所以推错了只看数据集树是看不出来的：树里显示的是推断结果本身。
BATCH_PREVIEW_HEADERS = ("文件", "数据集", "层堆叠")
# 空格与「没认出材料」在表里长得一样，而后者的含义是「这一条得手工建层」。
BATCH_PREVIEW_NO_STACK = "未识别，需手工建层"
BATCH_PREVIEW_STACK_SEPARATOR = " / "


def _readonly_table(object_name: str, headers: tuple[str, ...]) -> QTableWidget:
    """左栏那两张只读小表：默认不占位，按内容显隐。

    两张表回答的是同一批导入的两半（哪些进来了、成了什么；哪些没进来、怎么补），所以
    形状也保持一致——最后一列拉伸，其余按内容收，栏只有 264px 宽（见 ``_build_add_button``）。
    """
    table = QTableWidget(0, len(headers))
    table.setObjectName(object_name)
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.hide()
    return table


class _GlyphToolButton(QToolButton):
    """一枚只有字形那么大的工具按钮。

    ``QToolButton::sizeHint`` 在样式表被问到之前就替文字左右各垫一个空格的宽度
    （``textSize.width() + fm.horizontalAdvance(u' ') * 2``），所以样式表里的
    ``padding: 0px`` 收不掉它：一枚 16px 的 ``＋`` 量出来是 28px 宽的按钮。设计稿
    ``.nav-sec`` 是 ``justify-content:space-between`` 加 12px 内缩，字形的右边缘就该离
    栏边 12px；28px 的盒子把 16px 的字形居中，右边缘落到 12+(28-16)/2=18px，偏出 6px。
    这里直接按当前字体的字形盒报尺寸——字号是样式表给的，所以读 ``fontMetrics()``
    而不是写死 16，样式表换字号时尺寸跟着走。
    """

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        text = self.text()
        if not text:
            # 没有文字就没有字形盒可量（图标按钮走这条），把判断交回 Qt。
            return super().sizeHint()
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(text), metrics.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        # 两个钩子必须一起收：可及性契约量的是 ``width() >= minimumSizeHint().width()``，
        # 只改 sizeHint 会让按钮比自己声明的最小宽度还窄，读成「被裁窄了」。
        return self.sizeHint()


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
        # Untruncated detail fields, kept so a resize can re-elide them to the new
        # width without re-deriving them from the project.
        # 运行中卡片小字改报这一次运行，所以这三样跟着进度事件走。
        self._running = False
        self._run_objective: float | None = None
        self._run_trend = ""
        # 项目走到哪一步。卡片小字与页脚在结构那一步换说法，而步骤不是项目字段——它由
        # ``window_layout.apply_step_scope`` 从外面递进来，所以在窗口接线之前它是 ``None``。
        self._step: int | None = None
        # 帧⑤ 那句 walkers 下界。它不是这一栏自己算得出的话（两个数一个在候选解里、一个在右栏
        # 的 spin box 里），所以由 ``window_layout.refresh_sampling_footer`` 递进来；不在那一屏
        # 时是 ``None``，页脚照旧读自己的清点。
        self._sampling_summary: str | None = None
        self._build_controls()
        self._mask_editor = MaskEditor(document, self)
        self._mask_editor.mask_changed.connect(self.mask_changed.emit)
        document.project_changed.connect(self._render_project)
        document.source_validation_changed.connect(self._render_project)
        self._render_project()

    def _build_controls(self) -> None:
        # ``.nav-sec{...}数据集 ＋``：分区抬头替下了原先那句「面板标题会重名所以藏起来」
        # 的注释——停靠标题栏已经换成普通栏，栏里没有别处写着这两个字了。
        heading = QLabel("数据集")
        heading.setObjectName("dataPanelHeader")
        theme.apply_section_heading(heading, tracking_px=theme.RAIL_SECTION_TRACKING_PX)
        self.section_heading = heading
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
        self.dataset_add_button = self._build_add_button()
        self._build_tree()
        self.failure_table = _readonly_table("importFailureTable", ("文件", "问题", "恢复操作"))
        self.batch_preview_table = _readonly_table("importBatchPreviewTable", BATCH_PREVIEW_HEADERS)
        top = QHBoxLayout()
        # ``.nav-sec{padding:var(--md) var(--md) var(--sm)}``：内缩挂在抬头这一行上，不挂
        # 到栏的外边距——数据集行的 12px 由委托自己画（``CARD_PAD_H_PX``），栏一旦有外边距
        # 就会连列表一起再缩一次。
        top.setContentsMargins(theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_SM)
        top.addWidget(heading)
        top.addStretch(1)
        top.addWidget(self.dataset_add_button)
        self.summary_label = QLabel()
        self.summary_label.setObjectName("datasetSummary")
        self.summary_label.setProperty("mutedText", True)
        self.summary_label.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addLayout(top)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.tree)
        layout.addWidget(self.batch_preview_table)
        layout.addWidget(self.failure_table)

    def _build_add_button(self) -> QToolButton:
        """一枚 ``＋``，三个导入命令挂在它的菜单里。

        这三颗按钮此前并排摊在抬头行里。左栏实测 264px，光「导入文件」「导入文件夹」
        两颗就吃掉 144px，专家模式再加「更换测量预设」共 242px——第三颗量出来是 0 宽，
        也就是专家模式下它根本画不出来。设计稿那行只有 ``数据集 ＋``。

        托的是同一批 ``QPushButton``，不是菜单项的副本：可及名、tooltip 和点击处理都在
        它们身上，既有测试里的 ``findChild(QPushButton, "importFilesButton").click()``
        因此照旧成立。按钮本身仍是 ``QPushButton``，所以「更换测量预设」的专家模式可见性
        还是那一句 ``setVisible``。
        """
        button = _GlyphToolButton()
        button.setObjectName("datasetAddButton")
        button.setAccessibleName("导入数据集")
        button.setToolTip("导入数据集（点击选择文件，展开可选文件夹或更换测量预设）")
        # 默认动作走最常用那条路：最常见的导入是选文件，不该先让人展开一层菜单。
        default = QAction("＋", self)
        default.setObjectName("datasetAddAction")
        default.triggered.connect(self._import_files)
        button.setDefaultAction(default)
        menu = QMenu(button)
        menu.setObjectName("datasetAddMenu")
        for hosted in (self.import_files_button, self.import_folder_button, self.change_preset_button):
            action = QWidgetAction(menu)
            action.setDefaultWidget(hosted)
            menu.addAction(action)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
        return button

    def _build_tree(self) -> None:
        """一列自绘的卡片列表：表头和其余六列都收起来。

        七列仍在模型里——详情标签和各列 tooltip 都从它们读，删列会连带删掉那些出处。
        收起的是显示：设计稿的 ``.ds`` 列表没有表头，也没有第二列。
        """
        # 一行起步：设计稿的 ``.ds`` 列表就是有几个数据集画几行，⑤⑥ 两帧两个数据集时列表
        # 紧挨着「分析管线」。默认的三行楼层会替只有一两集的项目预留空行，在按内容定高的机架
        # 里那正好成了列表和管线之间的一段空白。
        self.tree = ContentSizedTree(row_floor=1)
        self.tree.setObjectName("datasetTree")
        self.tree.setAccessibleName("数据集列表")
        self.tree.setColumnCount(len(TREE_HEADERS))
        self.tree.setHeaderLabels(TREE_HEADERS)
        self.tree.setRootIsDecorated(False)
        self.tree.setHeaderHidden(True)
        # 斑马纹是给多列表格分行用的。卡片自己有边界（两行一组、选中那张有 accent 左条），
        # 再加隔行底色会和 ``.ds.on`` 的高亮抢同一个信号。
        self.tree.setAlternatingRowColors(False)
        self.tree.setUniformRowHeights(True)
        for column in range(1, len(TREE_HEADERS)):
            self.tree.setColumnHidden(column, True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.setItemDelegateForColumn(0, DatasetCardDelegate(self.tree))
        self.tree.currentItemChanged.connect(self._select_tree_item)

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
                (path for path in entries if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES),
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
        self._render_batch_preview(preview)
        substrate_choices = self._substrate_choices(preview)
        mappings = None if column_mapping is None else {row.source_path: column_mapping for row in preview.files}
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

    def _render_batch_preview(self, preview: api.ImportBatchPreview) -> None:
        """把这一批的文件名推断摆出来，在真正导入之前。

        填表放在 ``import_dataset_batch`` 之前而不是之后：``_substrate_choices`` 会为
        ``Si+…`` 的一组弹模态问基底，那一刻这张表已经答得出「这批是哪些文件、各自认出了
        什么」，而对话框问的正是同一批里的一件事。
        """
        self.batch_preview_table.setRowCount(len(preview.files))
        for row, file in enumerate(preview.files):
            stack = BATCH_PREVIEW_STACK_SEPARATOR.join(file.layers_backing_to_surface)
            values = (
                Path(file.source_path).name,
                # 数据集编号推不出来时退回显示名（``ImportFilePreview`` 允许它为空，
                # 而「层堆叠」那句话在这一列是另一件事，不能借来当兜底）。
                file.dataset_id_stem or file.display_name,
                stack or BATCH_PREVIEW_NO_STACK,
            )
            for column, value in enumerate(values):
                self.batch_preview_table.setItem(row, column, QTableWidgetItem(value))
        self.batch_preview_table.setVisible(bool(preview.files))

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

    def point_count_text(self, dataset_id: str) -> str:
        """How many points this curve carries, as the design writes it: 「512 点」.

        设计稿每一帧的数据集行都有这个数，而它此前在界面上没有出处。它是「可拟合 / 数据点
        不足」那句判定的依据——只给判定不给数，读者看到「数据点不足」也不知道差多少。

        数的是掩码长度而不是 ``sum``：这一行报的是这个文件里有多少点，范围裁剪之后还剩多少
        是画布上那件事。源文件出问题也照旧报——这个数是导入当时数出来的，存在档里，不因为
        文件被移开而变成未知。
        """
        return f"{len(self._dataset(dataset_id).fit_mask)} 点"

    def import_summary_text(self) -> str:
        """Aggregate the datasets into a one-line readiness overview.

        A batch import can produce many rows; this collapses them into "how many
        are ready and how many need attention" so the user need not scan every
        row. Fittability reuses the per-row judgement so the two never disagree.
        """
        datasets = self.document.project.datasets
        total = len(datasets)
        if total == 0:
            return ""
        fittable = sum(1 for dataset in datasets if self.status_text(dataset.dataset_id) == "可拟合")
        attention = total - fittable
        if attention == 0:
            return f"共 {total} 个数据集 · 全部可拟合"
        return f"共 {total} 个数据集 · 可拟合 {fittable} · 需注意 {attention}"

    def summary_text(self) -> str:
        """数据集树底下那句话，按项目走到的那一步取。

        结构那一步不清点（设计稿帧③ 的页脚），其余各帧仍是清点（帧①）。改结构时读者问的
        不是「有几个数据集」——树就在这句话上面，几行数得出来——而是「我编的这叠层落在哪几条
        曲线上、哪些东西仍旧各集一份」。

        而「对全部可拟合数据集共享」是能核的事实，不是标语：``api.set_structure`` 只在联合
        批量下把结构传播给全部数据集，独立模式下只落在指名的那一集。所以这句话只在真的每条
        可拟合曲线都拿着同一叠层时才写，否则退回清点——写一句假的比不写更糟。

        不确定度那一屏（帧⑤）的页脚整句让给 walkers 下界，所以它排在最前：那一屏的读者盯的
        是「下一次采样会不会被拦」，而数据集清点在同一屏上的树里已经数得出来。页脚在设计稿里
        本来就是逐帧换人的（帧① 清点、帧③ 结构共享、帧⑤ walkers），不是一句常驻文案。
        """
        if self._sampling_summary is not None:
            return self._sampling_summary
        if self.structure_step_is_current() and self._structure_is_shared():
            return SHARED_STRUCTURE_SUMMARY
        return self.import_summary_text()

    def structure_step_is_current(self) -> bool:
        """项目此刻停在结构那一步吗。

        卡片小字与页脚都问这一句：设计稿帧③ 的小字停在点数、页脚写结构管着哪几条曲线，
        两处的依据是同一个下标。
        """
        return self._step == STRUCTURE_STEP_INDEX

    def _structure_is_shared(self) -> bool:
        """每条可拟合曲线都拿着同一叠层吗——而且至少得有一条。

        比的是 ``StructureSpec`` 本身。它是冻结数据类，相等即每一层的材料、厚度、粗糙度
        都相等，也就是「同一叠层」这句话的字面意思；用等值而不是集合，因为层里嵌着的
        规格不保证可哈希。
        """
        structures = [
            dataset.structure
            for dataset in self.document.project.datasets
            if self.status_text(dataset.dataset_id) == "可拟合"
        ]
        if not structures or any(structure is None for structure in structures):
            return False
        return all(structure == structures[0] for structure in structures[1:])

    def fit_status_text(self, dataset_id: str) -> str:
        """Name the dataset's fit outcome: unfitted, or its confidence label."""
        result = self._dataset(dataset_id).last_valid_result
        return "未拟合" if result is None else str(result.confidence.value)

    def fit_status_marker(self, dataset_id: str) -> str:
        """Return the confidence glyph, or empty when the dataset is unfitted.

        The shape comes from ``theme.CONFIDENCE_GLYPHS`` so this row and the
        results badge cannot disagree about what a verdict looks like. An
        unfitted dataset stays blank rather than borrowing the badge's hollow
        ring: the adjacent cell already reads "未拟合", and a glyph in a dense
        list would imply a result exists.
        """
        result = self._dataset(dataset_id).last_valid_result
        if result is None:
            return ""
        return theme.CONFIDENCE_GLYPHS.get(str(result.confidence.value), "")

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

    def display_name_text(self, dataset_id: str) -> str:
        """The dataset's name as the list writes it, falling back to its id.

        导入时给了显示名就用显示名，没给就用 ``dataset_id``——列表里每行总得有个
        可读的抬头，空着的一行没法点也没法说是哪条。
        """
        dataset = self._dataset(dataset_id)
        return dataset.display_name or dataset.dataset_id

    def sha256_text(self, dataset_id: str) -> str:
        return self._dataset(dataset_id).source_sha256

    def beam_text(self, dataset_id: str) -> str:
        beam = self._dataset(dataset_id).beam
        if beam.kind == "monochromatic":
            return f"单色 λ={beam.wavelength_a:g} Å"
        return (
            f"混合 Kα λ₁={beam.wavelength_1_a:g} Å / λ₂={beam.wavelength_2_a:g} Å · I₂/I₁={beam.intensity_ratio_21:g}"
        )

    def instrument_text(self, dataset_id: str) -> str:
        instrument = self._dataset(dataset_id).instrument
        identity = instrument.instrument_id or "默认仪器"
        footprint = self._footprint_text(instrument)
        domain = "θ" if instrument.resolution_domain == "theta" else "q"
        return f"{identity} · {footprint} · 背景 {instrument.background_kind} · 分辨率 {domain}"

    def _footprint_text(self, instrument: api.InstrumentSpec) -> str:
        if instrument.footprint_mode == "geometry":
            return f"几何换算 {instrument.sample_length_mm:g}×{instrument.beam_width_mm:g} mm"
        if instrument.footprint_mode == "none":
            return "无足迹修正"
        return "拟合足迹"

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(dataset for dataset in self.document.project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def set_running(self, running: bool) -> None:
        """一次拟合开始或结束了。

        运行中每张卡的小字改报这一次运行；跑完之后回到点数与判定，因为运行中的 J 从那
        一刻起就是上一轮的数了。
        """
        self._running = running
        if not running:
            self._run_objective = None
            self._run_trend = ""
        self._render_project()

    def set_run_progress(self, progress: api.FitProgress) -> None:
        """一次进度事件。联合批量下它不带数据集——J 是几条曲线共享的那一个。"""
        objective = float(progress.best_objective)
        self._run_trend = metrics_trend(self._run_objective, objective)
        self._run_objective = objective
        self._render_project()

    def run_subline(self, _dataset_id: str) -> str:
        """运行中那一行小字，不运行时是空串。

        趋势只留箭头：``≈ 平台``「下降」这些词在 264px 的卡片小字里挤掉文件名的宽度，
        而箭头本身已经说清是降是平。第一次读数没有可比对象，所以它没有箭头。
        """
        if not self._running or self._run_objective is None:
            return ""
        arrow = "" if self._run_trend in ("", UNKNOWN) else self._run_trend[0]
        return f"拟合中 · J={self._run_objective:.2f}{arrow}"

    def set_step(self, step: int | None) -> None:
        """项目走到了哪一步。

        与 ``set_running`` 同一种形状：存下来再重画，因为卡片小字和页脚都从这个值取措辞。
        步骤没变就不重画——``apply_step_scope`` 在每次项目变更后都会再喊一遍，而重画要清空
        并重建整棵树。
        """
        if step == self._step:
            return
        self._step = step
        self._render_project()

    def set_sampling_summary(self, summary: str | None) -> None:
        """页脚这一屏归 walkers 下界（``summary`` 不是 ``None``），或者交还给清点。

        与 ``set_step`` 不同，这一路只重刷页脚那一行而不重画整棵树：这句话的一半来自右栏那个
        walkers spin box，读者每按一下箭头它就得改口，而重画会清空并重建全部数据集行。
        """
        if summary == self._sampling_summary:
            return
        self._sampling_summary = summary
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        """页脚那一行：有话就写上并露面，没话就整行收起来。"""
        summary = self.summary_text()
        self.summary_label.setText(summary)
        self.summary_label.setVisible(bool(summary))

    def _render_project(self, *_args) -> None:
        # 联合批量下这一列是一次运行的成员表，不是几条各自拟合的曲线——设计稿帧④ 的抬头
        # 因此写「联合拟合 · 数据集」。
        joint = self.document.project.batch_mode == "joint"
        self.section_heading.setText("联合拟合 · 数据集" if joint else "数据集")
        blocker = QSignalBlocker(self.tree)
        self.tree.clear()
        active_item = None
        for dataset in self.document.project.datasets:
            item = self._tree_item(dataset)
            self.tree.addTopLevelItem(item)
            if dataset.dataset_id == self.active_dataset_id:
                active_item = item
        if active_item is not None:
            # The selection highlight dims when the tree loses focus, so bold the
            # active row's name to keep it identifiable no matter where focus went.
            font = active_item.font(0)
            font.setBold(True)
            active_item.setFont(0, font)
            self.tree.setCurrentItem(active_item)
        del blocker
        self.change_preset_button.setVisible(self.document.project.ui_state.expert_mode)
        self._refresh_summary()

    def _tree_item(self, dataset: api.DatasetProject) -> QTreeWidgetItem:
        marker = self.status_marker(dataset.dataset_id)
        # 点数与判定同格：设计稿把「512 点 · 已拟合」写在一行里，而分成两列会让 264px 的
        # 紧凑视图再挤掉一列名字的宽度。判定的措辞由 ``status_text`` 单独给出，因为
        # ``import_summary_text`` 拿它做等值比较。
        status = f"{self.point_count_text(dataset.dataset_id)} · {self.status_text(dataset.dataset_id)}"
        status_cell = f"{marker} {status}" if marker else status
        fit_marker = self.fit_status_marker(dataset.dataset_id)
        fit_status = self.fit_status_text(dataset.dataset_id)
        fit_cell = f"{fit_marker} {fit_status}" if fit_marker else fit_status
        values = (
            self.display_name_text(dataset.dataset_id),
            Path(dataset.source_path).name,
            self.beam_text(dataset.dataset_id),
            self.instrument_text(dataset.dataset_id),
            status_cell,
            dataset.source_sha256[:12],
            fit_cell,
        )
        item = QTreeWidgetItem(values)
        item.setData(0, Qt.ItemDataRole.UserRole, dataset.dataset_id)
        item.setData(0, DATASET_CARD_ROLE, dataset_card(self, dataset.dataset_id))
        item.setToolTip(1, dataset.source_path)
        item.setToolTip(2, values[2])
        item.setToolTip(3, values[3])
        item.setToolTip(4, self._status_tooltip(dataset.dataset_id, status))
        item.setToolTip(5, dataset.source_sha256)
        item.setToolTip(6, self._fit_status_tooltip(dataset.dataset_id))
        # 收起的那几列连同源文件/光路/仪器一起归到第 0 列：设计稿的左栏只有抬头、列表和
        # 底部合计，没有那块四行详情，而这些字段仍是判断「拟合的是哪一条曲线」的依据。
        # 悬停落在卡片上，所以它们跟着卡片走，而不是另立一个占掉列表高度的标签。
        item.setToolTip(
            0,
            "\n".join(
                (
                    f"源文件：{values[1]}",
                    f"光路：{values[2]}",
                    f"仪器：{values[3]}",
                    self._status_tooltip(dataset.dataset_id, status),
                    self._fit_status_tooltip(dataset.dataset_id),
                    f"SHA-256：{dataset.source_sha256}",
                )
            ),
        )
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
                    angle_convention=dialog.angle_convention(),
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
                (path for path in entries if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES),
                key=lambda path: path.relative_to(folder).as_posix().casefold(),
            )
        )
