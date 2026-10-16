from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _isolated_user_settings(tmp_path_factory):
    """``QSettings`` 写到本次会话的临时目录，而不是开发机的用户配置。

    明暗偏好（F52）存在 ``QSettings`` 里；不隔开的话，一条点了深色开关的用例会改掉开发者自己
    的外观，而开发机上存着的深色又会让后面的用例换一套外观去量。
    """
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("qsettings")))


def _running_application():
    """The session's ``QApplication``, without importing Qt widgets into a non-GUI run."""
    widgets = sys.modules.get("PySide6.QtWidgets")
    return None if widgets is None else widgets.QApplication.instance()


def _appearance_snapshot():
    application = _running_application()
    if application is None:
        return None
    from PySide6.QtGui import QPalette

    from xrr_fitter.gui import theme

    return application.styleSheet(), QPalette(application.palette()), theme.LIGHT_PALETTE


def _restore_appearance(snapshot) -> None:
    application = _running_application()
    if application is None:
        return
    from PySide6.QtCore import QSettings

    from xrr_fitter.gui import theme

    sheet, palette, light = ("", None, None) if snapshot is None else snapshot
    if application.styleSheet() != sheet:
        application.setStyleSheet(sheet)
    if palette is not None and application.palette() != palette:
        application.setPalette(palette)
    # 深色切换会记住「来时的亮色板」；留着它，下一条用例切回亮色时拿到的是别人的板。
    theme.LIGHT_PALETTE = light
    # 明暗偏好落在 QSettings（已由 ``_isolated_user_settings`` 重定向到临时目录）；点过开关的
    # 用例不清掉它，后面每一扇新窗口都会按深色开。
    QSettings().clear()


@pytest.fixture(autouse=True)
def _application_appearance_is_restored():
    """把用例留在会话唯一那个 ``QApplication`` 上的外观还原回去。

    ``create_application`` 会装上 ``theme.apply_theme`` 的样式表，多数 GUI 用例却不走它。命令栏
    明暗开关那条用例（``test_full_control_operations``）点两下之后样式表就留下了，排在后面的用例
    量到的是发货外观，单跑量到的是裸界面：眉标字体与导入对话框尺寸两条用例单跑能过、宽跑失败。
    要量发货外观的用例自己装、自己卸（见 ``test_shipped_theme_contracts``）。没有起过 Qt 的用例
    什么也不做。
    """
    snapshot = _appearance_snapshot()
    yield
    _restore_appearance(snapshot)


@pytest.fixture
def load_tool_module():
    def load(name: str):
        path = ROOT / "tools" / f"{name}.py"
        if not path.is_file():
            pytest.fail(f"missing implementation: tools/{name}.py", pytrace=False)
        spec = importlib.util.spec_from_file_location(f"r23_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    return load
