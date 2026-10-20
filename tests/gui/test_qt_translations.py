"""The strings Qt writes itself read in the application's language.

Every message box's standard buttons, the editing menu of every text field and the
macOS application menu come from Qt's own catalogue. Without a translator they read
``&Yes`` / ``OK`` / ``Copy`` in an otherwise Chinese window. The probe runs in a
fresh process because only a newly created application is given the translator; the
test session's shared application is left as the tests set it up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = r"""
import json
from PySide6.QtWidgets import QLineEdit, QMessageBox
from xrr_fitter.gui.application import create_application

app = create_application(["probe"])
box = QMessageBox()
box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Ok)
# 菜单是输入框的子对象；输入框得留着引用，否则临时对象一回收，菜单跟着被删。
edit = QLineEdit()
menu = edit.createStandardContextMenu()
print(json.dumps({
    "buttons": sorted(button.text() for button in box.buttons()),
    "menu": [action.text().split("\t")[0] for action in menu.actions() if action.text()],
}))
"""


def test_a_launched_application_speaks_chinese_in_qts_own_strings() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        timeout=120,
    )
    shown = json.loads(completed.stdout.strip().splitlines()[-1])

    assert shown["buttons"] == sorted(["是(&Y)", "否(&N)", "确定"])
    assert {"复制(&C)", "粘贴(&P)", "全选"} <= set(shown["menu"])
