"""Candidate audit rows and stable-ID selection projection."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QFrame, QListWidget, QListWidgetItem, QWidget

from xrr_fitter.gui.results.candidate_row import (
    CANDIDATE_ROW_ROLE,
    ROW_PAD_V_PX,
    CandidateRowDelegate,
    candidate_is_selectable,
    candidate_row,
)

# 空列表的高度：Qt 给一张滚动视图的默认高是 192px，与内容无关，三张这样的列表叠在
# 结果栏里就把它撑破了。有行的时候高度按行算（见 ``sizeHint``），一行都没有的时候留
# 三行的位置，让那句「运行拟合后……」有地方写——占位文字挤在一行高里会被裁掉。
#
# 上限取消了：设计稿这一节是检视器那一栏的末段，整段展开，列表内再套一层滚动条会把
# 末行藏在折叠线下，而这一栏本来就自己会滚。
VISIBLE_ROW_FLOOR = 3


def candidate_is_mcmc_ready(candidate: object | None, result: object | None) -> bool:
    if candidate is None or result is None or result.uncertainty is None:
        return False
    return bool(
        candidate_is_selectable(candidate)
        and len(candidate.unit_vector) > 0
        and result.uncertainty.candidate_id == candidate.candidate_id
    )


def _objective_text(candidate: object) -> str:
    local = f"局部目标值 J={candidate.objective:.12g}"
    ranking = getattr(candidate, "ranking_objective", None)
    if ranking is None:
        return local
    return f"{local} · 全局排序目标值 J={ranking:.12g}"


def _audit_state(candidate: object) -> str:
    if candidate_is_selectable(candidate):
        return "有效"
    if getattr(candidate, "stop_reason", "") == "early_eliminated":
        return "仅供检查 · 早期淘汰 · 不参与收敛统计"
    reason = getattr(candidate, "stop_reason", "unknown")
    return f"仅供检查 · 无效 · {reason}"


def candidate_line(candidate: object, *, selected: bool, recommended: bool) -> str:
    parts = [str(candidate.candidate_id), _objective_text(candidate), _audit_state(candidate)]
    if recommended and candidate_is_selectable(candidate):
        parts.append("推荐")
    if selected:
        parts.append("查看中")
    return " · ".join(parts)


def active_dataset(project: object) -> object | None:
    dataset_id = project.ui_state.active_dataset_id
    return next(
        (dataset for dataset in project.datasets if dataset.dataset_id == dataset_id),
        None,
    )


def persisted_candidate_id(project: object, dataset_id: str) -> str | None:
    return next(
        (candidate_id for owner, candidate_id in project.ui_state.selected_candidate_ids if owner == dataset_id),
        None,
    )


def result_verdict(result: object) -> str | None:
    """整份结果的判定，读成行与判定卡共用的那个字符串。

    ``FitResult.confidence`` 是 ``ConfidenceClass``（``StrEnum``），行只认字符串——
    ``theme.CONFIDENCE_GLYPHS`` 的键就是这几个字。旧存档里这个字段可能是空的，此时
    返回 ``None``，被采信的那一行画空心圈（「还没测」），不是第五档可信度。
    """
    confidence = getattr(result, "confidence", None)
    if confidence is None:
        return None
    return str(getattr(confidence, "value", confidence))


class CandidateList(QListWidget):
    """Render every retained candidate without hiding archived evidence."""

    candidate_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset_id: str | None = None
        self._result: object | None = None
        self._recommended_id: str | None = None
        self._confidence: str | None = None
        self._inspection_result: object | None = None
        self._inspection_candidate_id: str | None = None
        self.setObjectName("candidateList")
        self.setAccessibleName("拟合候选解")
        self.setAccessibleDescription("用方向键在候选解之间切换，证据面板会随当前行更新")
        self.setToolTip("用方向键或点击切换候选解；证据面板随当前行更新")
        # 换行关掉：DisplayRole 上挂的是那句散文，开着换行它会把一行撑成三四行高，而
        # 这一行根本不画那句话（见 ``CandidateRowDelegate.sizeHint``）。
        self.setWordWrap(False)
        # 行与行之间的缝由委托算进行高。``setSpacing`` 四边都插，首行上方和末行下方于是
        # 各多出一半，而设计稿这一段紧贴着上一节的横线开始。
        self.setSpacing(0)
        self.setViewportMargins(0, 0, 0, 0)
        # 外框交给外层那张卡：这张列表本来就装在检视器的「候选解」一段里，自己再画一道
        # 就是双线。样式表里那道通用 ``border`` 另由 ``#candidateList`` 抹掉。
        self.setFrameShape(QFrame.Shape.NoFrame)
        # 整段展开，不自己滚：检视器那一栏会滚，列表内再套一层就把末行藏了。
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 分段自绘。散文行留在 DisplayRole 上供读屏，画面按设计稿分段对齐。
        self.setItemDelegate(CandidateRowDelegate(self))
        self.currentItemChanged.connect(self._current_item_changed)

    def sizeHint(self) -> QSize:
        """列表高就是它装的那几行，一像素不多。

        不滚动就得把高度让给内容。留着 Qt 默认那份高度，检视器里这一段会先撑出一片空白；
        按行算而每行的高由委托给出（字高 + 上下内边距 + 行间那道缝），所以这里只是把
        ``sizeHintForRow`` 加起来——缝已经在里面了。
        """
        width = super().sizeHint().width()
        if self.count() == 0:
            unit = self.fontMetrics().height() + 2 * ROW_PAD_V_PX
            return QSize(width, VISIBLE_ROW_FLOOR * unit + 2 * self.frameWidth())
        rows = sum(self.sizeHintForRow(row) for row in range(self.count()))
        return QSize(width, rows + 2 * self.frameWidth())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() > 0:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QPen(self.palette().placeholderText().color()))
        painter.drawText(
            self.viewport().rect(),
            Qt.AlignmentFlag.AlignCenter,
            "运行拟合后，候选解将显示在此处",
        )
        painter.end()

    @property
    def result(self) -> object | None:
        return self._result

    @property
    def recommended_id(self) -> str | None:
        return self._recommended_id

    def project_result(
        self,
        dataset_id: str,
        result: object,
        persisted_id: str | None,
    ) -> str | None:
        best = result.best_candidate
        self._recommended_id = None if best is None else best.candidate_id
        # 判定属于整份结果，行上画的是「被采信的那一行的判定」——所以在这里读一次存起来，
        # 后面切候选（``inspect``）时那枚 ● 不会因为焦点移开就没了。
        self._confidence = result_verdict(result)
        inspection_is_current = (
            self._dataset_id == dataset_id
            and self._inspection_result is result
            and self._inspection_candidate_id is not None
        )
        if not inspection_is_current:
            self.clear_inspection()
        self._dataset_id = dataset_id
        self._result = result
        visible_id = self._inspection_candidate_id or persisted_id or self._recommended_id
        self.load(
            result.candidates,
            selected_id=visible_id,
            recommended_id=self._recommended_id,
            confidence=self._confidence,
        )
        return visible_id

    def inspect(self, candidate_id: str) -> object:
        candidate = self.candidate(candidate_id)
        if candidate is None or self._result is None:
            raise ValueError("fit result is no longer current")
        self._inspection_result = self._result
        self._inspection_candidate_id = candidate_id
        self.load(
            self._result.candidates,
            selected_id=candidate_id,
            recommended_id=self._recommended_id,
            confidence=self._confidence,
        )
        return candidate

    def candidate(self, candidate_id: str | None) -> object | None:
        if candidate_id is None or self._result is None:
            return None
        return next(
            (candidate for candidate in self._result.candidates if candidate.candidate_id == candidate_id),
            None,
        )

    def clear_inspection(self) -> None:
        self._inspection_result = None
        self._inspection_candidate_id = None

    def clear_projection(self) -> None:
        self._dataset_id = None
        self._result = None
        self._recommended_id = None
        self._confidence = None
        self.clear_inspection()
        self.clear_candidates()

    def load(
        self,
        candidates: tuple[object, ...],
        *,
        selected_id: str | None,
        recommended_id: str | None,
        confidence: str | None = None,
    ) -> None:
        blocker = QSignalBlocker(self)
        self.clear()
        selected_row = -1
        for row, candidate in enumerate(candidates):
            selected = candidate.candidate_id == selected_id
            recommended = candidate.candidate_id == recommended_id
            line = candidate_line(candidate, selected=selected, recommended=recommended)
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, candidate.candidate_id)
            # 名字换成序号之后，求解器 ID 只剩这里一处出口：跟日志或存档对照时要拿得到
            # ``E-13``，而序号是随这次结果的排序来的，换一次拟合就换一个含义。
            item.setToolTip(line)
            # 画面走分段自绘，可访问文本仍是上面那句散文：自绘接管的是排版，
            # 不是信息量。
            item.setData(
                CANDIDATE_ROW_ROLE,
                candidate_row(
                    candidate,
                    selected=selected,
                    # 判定画在被采信的那一行上：整份结果只有一个判定，而它判的正是这个解。
                    adopted=recommended,
                    confidence=confidence,
                    ordinal=row,
                ),
            )
            self.addItem(item)
            if selected:
                selected_row = row
        self.setCurrentRow(selected_row)
        del blocker
        # 行数变了高度就变了：这张列表按内容占高，不通知一次布局它还按上一份的行数占位。
        self.updateGeometry()

    def clear_candidates(self) -> None:
        blocker = QSignalBlocker(self)
        self.clear()
        del blocker
        self.updateGeometry()

    def candidate_count(self) -> int:
        return self.count()

    def candidate_text(self, row: int) -> str:
        item = self.item(row)
        if item is None:
            raise IndexError(f"candidate row out of range: {row}")
        return item.text()

    def selected_candidate_id(self) -> str | None:
        item = self.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is not None:
            self.candidate_requested.emit(str(current.data(Qt.ItemDataRole.UserRole)))
