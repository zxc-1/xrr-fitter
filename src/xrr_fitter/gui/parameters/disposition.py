"""设计稿帧③「参数化 · 厚度 d」：当前这个量此刻怎么被对待。

三档说的是拟合器拿它怎么办，两枚徽章说的是另外两件同样只属于这一个量的事。这几样此
前散在三处——锁在参数表名字格的勾里，先验在表的最后一列，共享要切到另一个标签页——
读者没法一眼回答「我现在改的这个数，拟合时到底算什么」。
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.badge import Badge
from xrr_fitter.gui.parameters.table import FREEDOM_TOOLTIPS, prior_summary

FREE_LABEL = "自由"
FIXED_LABEL = "固定"
RANGE_ONLY_LABEL = "仅范围"

#: 没有先验时说的话。留空会被读成「先验没加载出来」，而「无」在这里有确切含义——上下限
#: 之内一律等权。
NO_PRIOR_TEXT = "先验：无（均匀）"

#: 三档各自说的是拟合器拿这个量怎么办。这几句从参数表借过来，不在这里另写一份：名字格的勾
#: 和这三个圆点表示的是同一个档位，两处措辞不同会被读成两件事。
#: 被约束表达式驱动时三档都点不动，这句话既说清为什么，也指出去哪里改。
CONSTRAINT_DRIVEN_TOOLTIP = (
    "由约束表达式驱动：值是算出来的，三档都不适用。要改回自由、仅范围或固定，请到「约束」页删掉那条规则。"
)


def sharing_summary(
    definition: api.ParameterDefinition,
    rules: Sequence[api.SharingRule],
) -> str:
    """这个量和几份数据集绑在一起，没绑时是空串。

    数的是数据集而不是成员条目：联合拟合里一个数据集可能贡献多个成员，报成员数会把
    「跨 3 集」说成「跨 5 项」。
    """
    key = definition.sharing_key
    if key is None:
        return ""
    members = tuple(member for rule in rules if rule.sharing_key == key for member in rule.members)
    return "" if not members else f"跨 {len({member.dataset_id for member in members})} 集共享"


class ParameterDisposition(QWidget):
    """当前参数的三档处置与两枚只读结论。"""

    # 第二个参数写 ``object`` 而不是 ``ParameterFreedom``/``str``：``Signal(str, str)`` 会把
    # StrEnum 过一遍 QString 再还回来，接收端拿到的是普通字符串，档位的类型信息在信号上就丢
    # 了。``object`` 原样传，接收端不必重新包一次。
    freedom_requested = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("parameterDisposition")
        self.free_button = QRadioButton(FREE_LABEL)
        self.free_button.setObjectName("parameterFreeChoice")
        self.fixed_button = QRadioButton(FIXED_LABEL)
        self.fixed_button.setObjectName("parameterFixedChoice")
        self.range_button = QRadioButton(RANGE_ONLY_LABEL)
        self.range_button.setObjectName("parameterRangeOnlyChoice")
        # 档位到按钮只建一张表，正反查都从它出发。两处各写一份对应关系的话，新增一档时改
        # 漏一处不会报错，只会静默点不亮。
        self._buttons: dict[api.ParameterFreedom, QRadioButton] = {
            api.ParameterFreedom.FREE: self.free_button,
            api.ParameterFreedom.FIXED: self.fixed_button,
            api.ParameterFreedom.RANGE_ONLY: self.range_button,
        }
        self._choices = QButtonGroup(self)
        # 三档互斥，但「没有当前行」时一档都不该亮着。独占组不允许取消选中，所以退出
        # 独占来清空，清完再恢复——留一档亮着会被读成「当前那个量是自由的」。
        for button in self._buttons.values():
            self._choices.addButton(button)
        self.prior_badge = Badge("parameterPriorBadge")
        self.sharing_badge = Badge("parameterSharingBadge", kind="info")

        choices = QHBoxLayout()
        choices.setContentsMargins(0, 0, 0, 0)
        choices.setSpacing(theme.SPACE_MD)
        choices.addWidget(self.free_button)
        choices.addWidget(self.fixed_button)
        choices.addWidget(self.range_button)
        choices.addStretch(1)
        badges = QHBoxLayout()
        badges.setContentsMargins(0, 0, 0, 0)
        badges.setSpacing(theme.SPACE_SM)
        badges.addWidget(self.prior_badge)
        badges.addWidget(self.sharing_badge)
        badges.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addLayout(choices)
        layout.addLayout(badges)

        self._name: str | None = None
        self.free_button.clicked.connect(lambda: self._request_freedom(api.ParameterFreedom.FREE))
        self.fixed_button.clicked.connect(lambda: self._request_freedom(api.ParameterFreedom.FIXED))
        self.range_button.clicked.connect(lambda: self._request_freedom(api.ParameterFreedom.RANGE_ONLY))
        self.show_definition(None, quantity=None)

    def show_definition(
        self,
        definition: api.ParameterDefinition | None,
        *,
        quantity: str | None,
        freedom: api.ParameterFreedom | None = None,
        sharing_rules: Sequence[api.SharingRule] = (),
    ) -> None:
        """把这一段切到 ``definition`` 上；``None`` 表示此刻没有当前行。

        ``freedom`` 是这个量在 setting 里的档位。声明里只有 ``locked``，两态装不下「仅范围」，
        所以不传时按声明回落成两态——手里没有 setting 可查的调用方读到的就是声明的意思。
        """
        self._name = None if definition is None else definition.name
        if definition is None:
            self._check(None)
            self.prior_badge.setText("")
            self.sharing_badge.setText("")
            return
        constrained = definition.constrained
        current = api.ParameterFreedom.from_locked(definition.locked) if freedom is None else freedom
        # 被约束驱动的量，值是算出来的，三档都不适用。亮在「固定」是因为拟合器确实不动它
        # （编译出来的声明 ``locked`` 为真），三档一起禁用加上那句 tooltip 才说清「不是你选了
        # 固定，是那条约束替你定了」；留一档可点意味着点一下就悄悄删掉一条规则。
        self._check(self.fixed_button if constrained else self._buttons[current])
        for gear, button in self._buttons.items():
            button.setEnabled(not constrained)
            button.setToolTip(CONSTRAINT_DRIVEN_TOOLTIP if constrained else FREEDOM_TOOLTIPS[gear])
        self._show_badges(definition, quantity=quantity, sharing_rules=sharing_rules)

    def _show_badges(
        self,
        definition: api.ParameterDefinition,
        *,
        quantity: str | None,
        sharing_rules: Sequence[api.SharingRule],
    ) -> None:
        """两枚结论徽标：先验说这个量还被什么约束着，共享说它跟谁绑在一起。"""
        summary = prior_summary(definition)
        self.prior_badge.set_kind("info" if summary else "mut")
        self.prior_badge.setText(f"先验：{summary}" if summary else NO_PRIOR_TEXT)
        shared = sharing_summary(definition, sharing_rules)
        # 没共享就整枚不占地方：一排结论里，「未共享」和「共享」一样重会喧宾夺主。
        self.sharing_badge.setText(f"共享：{shared}{quantity or ''}" if shared else "")

    def _check(self, chosen: QRadioButton | None) -> None:
        self._choices.setExclusive(False)
        for button in self._buttons.values():
            button.setChecked(button is chosen)
            if chosen is None:
                button.setEnabled(False)
        self._choices.setExclusive(True)

    def _request_freedom(self, freedom: api.ParameterFreedom) -> None:
        if self._name is not None:
            self.freedom_requested.emit(self._name, freedom)
