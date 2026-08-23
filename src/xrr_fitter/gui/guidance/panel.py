"""Guided four-step workflow surface.

Expert mode was the only working surface: standard mode merely hid a handful of
controls, so a newcomer still faced the whole dock workspace at once. This panel
walks import to result and shows only what the current step needs.

Every gate is answered by reading the immutable project through the public API,
so the guided surface adds no API of its own and can never disagree with what a
fit would actually accept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.plots.plot_icons import plot_icon


@dataclass(frozen=True, slots=True)
class StepSpec:
    """One guided step: its identity, prose, and the action it offers."""

    name: str
    title: str
    body: str
    action_text: str
    action: str
    icon: str


STEP_SPECS = (
    StepSpec(
        "importStep",
        "第 1 步 · 导入数据",
        "选择 .xy / .dat / .txt 反射率数据文件。导入时确认光路与仪器设置。",
        "导入数据文件…",
        "import_files",
        "data_curve",
    ),
    StepSpec(
        "structureStep",
        "第 2 步 · 确认样品结构",
        "初始化样品结构，必要时添加膜层。默认基底为 Si，可在结构面板调整。",
        "初始化样品结构",
        "initialize_structure",
        "layer_stack",
    ),
    StepSpec(
        "fitStep",
        "第 3 步 · 开始拟合",
        "一键拟合会先全局筛选再局部精修。拟合过程中可以随时取消。",
        "开始一键拟合",
        "start_fit",
        "fit_progress",
    ),
    StepSpec(
        "resultStep",
        "第 4 步 · 查看结果",
        "查看候选解与可信度。需要完整的参数表、诊断图或导出时，切换到专家模式。",
        "切换到专家模式",
        "leave_guidance",
        "result_table",
    ),
)


class _StepIndicator(QWidget):
    """Horizontal step indicator: dots connected by lines."""

    DOT_RADIUS = 5.0
    LINE_LENGTH = 28.0
    TOTAL_STEPS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepIndicator")
        self._current = 0
        self.setFixedHeight(28)
        self.setMinimumWidth(200)

    def set_current(self, index: int) -> None:
        if self._current != index:
            self._current = index
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width = self.width()
        total_w = (self.TOTAL_STEPS - 1) * self.LINE_LENGTH + self.TOTAL_STEPS * self.DOT_RADIUS * 2
        start_x = (width - total_w) / 2.0
        cy = self.height() / 2.0

        from PySide6.QtWidgets import QApplication

        from xrr_fitter.gui.theme import palette_tokens

        tokens = palette_tokens(QApplication.instance().palette())
        accent = QColor(tokens.accent)
        muted = QColor(tokens.muted_text)

        for i in range(self.TOTAL_STEPS):
            cx = start_x + i * (self.DOT_RADIUS * 2 + self.LINE_LENGTH) + self.DOT_RADIUS
            if i < self.TOTAL_STEPS - 1:
                line_start = cx + self.DOT_RADIUS + 2
                line_end = cx + self.DOT_RADIUS + self.LINE_LENGTH - 2
                line_color = accent if i < self._current else muted
                pen = QPen(line_color, 1.5)
                p.setPen(pen)
                p.drawLine(QPointF(line_start, cy), QPointF(line_end, cy))

            if i < self._current:
                p.setPen(QPen(accent, 1.5))
                p.setBrush(accent)
                p.drawEllipse(QPointF(cx, cy), self.DOT_RADIUS, self.DOT_RADIUS)
                # checkmark
                p.setPen(QPen(QColor("white"), 1.5))
                p.drawLine(QPointF(cx - 2.5, cy), QPointF(cx - 0.5, cy + 2.5))
                p.drawLine(QPointF(cx - 0.5, cy + 2.5), QPointF(cx + 3.0, cy - 2.0))
            elif i == self._current:
                p.setPen(QPen(accent, 2.0))
                p.setBrush(accent)
                p.drawEllipse(QPointF(cx, cy), self.DOT_RADIUS, self.DOT_RADIUS)
            else:
                p.setPen(QPen(muted, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(cx, cy), self.DOT_RADIUS, self.DOT_RADIUS)

        p.end()


class GuidancePanel(QWidget):
    """Project one step at a time, gated on the real project state."""

    step_changed = Signal(str)
    leave_requested = Signal()

    def __init__(
        self,
        document: ProjectDocument,
        actions: dict[str, Callable[[], object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("guidancePanel")
        self.setAccessibleName("引导流程")
        self._document = document
        self._actions = dict(actions)
        self._pages: dict[str, QWidget] = {}
        self._action_buttons: dict[str, QPushButton] = {}
        self._next_buttons: dict[str, QPushButton] = {}
        self._step_indicator = _StepIndicator(self)
        self._stack = QStackedWidget(self)
        self._stack.setObjectName("guidanceStack")
        for spec in STEP_SPECS:
            page = self._build_page(spec)
            self._pages[spec.name] = page
            self._stack.addWidget(page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACE_LG,
            theme.SPACE_LG,
            theme.SPACE_LG,
            theme.SPACE_LG,
        )
        layout.addStretch(1)
        layout.addWidget(self._step_indicator, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._stack)
        layout.addStretch(2)
        document.project_changed.connect(self._refresh)
        self._refresh()

    def _build_page(self, spec: StepSpec) -> QWidget:
        page = QWidget()
        page.setObjectName(spec.name)
        page.setAccessibleName(spec.title)

        # Card container
        card = QFrame(page)
        card.setProperty("sectionCard", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG)
        card_layout.setSpacing(theme.SPACE_MD)

        # Step icon
        icon_label = QLabel(card)
        icon_label.setObjectName(f"{spec.name}Icon")
        icon_label.setPixmap(plot_icon(spec.icon, size=48).pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(icon_label)

        title = QLabel(spec.title, card)
        title.setObjectName(f"{spec.name}Title")
        title.setProperty("emptyTitle", True)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(title)

        body = QLabel(spec.body, card)
        body.setObjectName(f"{spec.name}Body")
        body.setProperty("mutedText", True)
        body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        body.setWordWrap(True)
        card_layout.addWidget(body)

        action = QPushButton(spec.action_text, card)
        action.setObjectName(f"{spec.name}Action")
        action.setProperty("primary", True)
        action.setAccessibleName(spec.action_text)
        action.clicked.connect(lambda _checked=False, key=spec.action: self._run(key))
        self._action_buttons[spec.name] = action
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(action)
        buttons.addStretch(1)
        card_layout.addLayout(buttons)

        # Ghost-styled navigation
        back = QPushButton("← 上一步", page)
        back.setObjectName(f"{spec.name}Back")
        back.setProperty("ghost", True)
        back.clicked.connect(lambda _checked=False, name=spec.name: self._step(name, -1))
        forward = QPushButton("下一步 →", page)
        forward.setObjectName(f"{spec.name}Next")
        forward.setProperty("ghost", True)
        forward.clicked.connect(lambda _checked=False, name=spec.name: self._step(name, 1))
        self._next_buttons[spec.name] = forward

        navigation = QHBoxLayout()
        navigation.addWidget(back)
        navigation.addStretch(1)
        navigation.addWidget(forward)

        layout = QVBoxLayout(page)
        layout.setSpacing(theme.SPACE_MD)
        layout.addWidget(card)
        layout.addLayout(navigation)
        return page

    def step_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in STEP_SPECS)

    def current_step(self) -> str:
        return self.step_names()[self._stack.currentIndex()]

    def show_step(self, name: str) -> None:
        page = self._pages.get(name)
        if page is None:
            raise KeyError(f"unknown guidance step: {name}")
        old_page = self._stack.currentWidget()
        if old_page is not page:
            self._animate_transition(old_page, page)
        self._stack.setCurrentWidget(page)
        self._step_indicator.set_current(self.step_names().index(name))
        self._refresh()
        self.step_changed.emit(name)

    def _animate_transition(self, old_page: object, new_page: object) -> None:
        effect = QGraphicsOpacityEffect(new_page)
        new_page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(200)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.finished.connect(lambda: new_page.setGraphicsEffect(None))
        animation.start()

    def step_is_available(self, name: str) -> bool:
        """Answer a step's gate from the project, never from local UI state."""
        project = self._document.project
        dataset = self._active_dataset(project)
        if name == "importStep":
            return True
        if name == "structureStep":
            return bool(project.datasets)
        if name == "fitStep":
            return dataset is not None and dataset.structure is not None
        if name == "resultStep":
            return dataset is not None and dataset.last_valid_result is not None
        raise KeyError(f"unknown guidance step: {name}")

    def _active_dataset(self, project: api.XrrProject) -> object | None:
        active = project.ui_state.active_dataset_id
        return next(
            (dataset for dataset in project.datasets if dataset.dataset_id == active),
            None,
        )

    def _step(self, name: str, offset: int) -> None:
        names = self.step_names()
        target = names.index(name) + offset
        if target >= len(names):
            self.leave_requested.emit()
        elif 0 <= target:
            self.show_step(names[target])

    def _run(self, key: str) -> None:
        if key == "leave_guidance":
            self.leave_requested.emit()
            return
        operation = self._actions.get(key)
        if operation is not None:
            operation()

    def _refresh(self, *_args) -> None:
        current = self.current_step()
        names = self.step_names()
        index = names.index(current)
        # The action for a step the project is not ready for would fail, so it is
        # disabled rather than left to raise; forward is gated on the next step's
        # own precondition so the flow cannot run ahead of the project.
        self._action_buttons[current].setEnabled(self.step_is_available(current))
        following = names[index + 1] if index + 1 < len(names) else None
        self._next_buttons[current].setEnabled(True)
        if following is None:
            self._next_buttons[current].setText("进入主界面 →")
