"""左栏是导航，不只是进度条：走过的那几步点得回去。

设计稿帧③ 把「结构」画成 current、「参数」画成待办，可那一屏里结构已经建好了——照
``_determine_step``（只看项目里存了什么）算出来的是「参数」。两句话并不矛盾：一个说的是
项目最远走到哪，一个说的是读者此刻在看哪一步。此前左栏只表达前者，于是设计稿这一屏在软件
里根本到不了；更要紧的是右栏与画布跟着这一步换内容（``STEP_INSPECTOR_SECTIONS``），拟合
出了结果之后选中层那张卡就再也够不着了——改一层得先把结果删掉。

规则两条：点走过的步就停在那一步；项目自己往前走一步（拟合出了结果、结构立起来了），选择
让位给新到达的那一步——否则拟合跑完，屏幕还停在读者半小时前点的「结构」上。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QWidget

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _structure(layers: int = 1) -> api.StructureSpec:
    return api.StructureSpec(
        AIR,
        tuple(api.LayerSpec(f"film{index}", SIO2, 40.0 + index, roughness_a=3.0) for index in range(layers)),
        SI,
    )


def _project(tmp_path, *, layers: int = 0):
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    if layers:
        project = api.set_structure(project, "curve", _structure(layers))
    return api.select_active_dataset(project, "curve")


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    return window, document


def _click(window, title: str) -> None:
    row = window.pipeline_nav.findChild(QWidget, f"pipelineStep_{title}")
    assert row is not None, title
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)


def _sub(window, title: str) -> str:
    label = window.pipeline_nav.findChild(QLabel, f"pipelineDescription_{title}")
    assert label is not None, title
    return label.text()


def test_clicking_a_step_the_project_already_passed_moves_the_rail_back_to_it(qtbot, tmp_path) -> None:
    """结构建好之后管线报「参数」，而设计稿帧③ 停在「结构」——那一屏是点回来的。

    这是唯一能回到结构编辑那几段的路：右栏在参数与结果两步换成别的卡，画布在结果态收起
    层堆叠。没有这条路，加完最后一层就等于把结构编辑器关掉了。
    """
    window, _document = _window(qtbot, _project(tmp_path, layers=2))
    assert window.pipeline_nav.current_step_index() == 2

    _click(window, "结构")

    assert window.pipeline_nav.current_step_index() == 1


def test_a_step_the_project_has_not_reached_stays_out_of_reach(qtbot, tmp_path) -> None:
    """点「拟合」不该把左栏挪过去：那一步还没到，挪过去右栏就换成一张空的运行卡。

    走过的步能回去，是因为回去看的东西项目里已经有了；往前点没有这个前提。
    """
    window, _document = _window(qtbot, _project(tmp_path, layers=2))

    _click(window, "拟合")

    assert window.pipeline_nav.current_step_index() == 2


def test_the_project_moving_forward_takes_the_rail_back_from_a_manual_pick(qtbot, tmp_path) -> None:
    """读者点了「数据」，随后结构立起来了——左栏该跟去新到达的那一步。

    手点的落点只在项目状态不变的这段时间里有效。否则拟合跑完，屏幕还停在半小时前点的
    那一步，而结果就在旁边没人看见。
    """
    window, document = _window(qtbot, _project(tmp_path))
    _click(window, "数据")
    assert window.pipeline_nav.current_step_index() == 0

    document.replace_project(api.set_structure(document.project, "curve", _structure(2)))
    qtbot.wait(1)

    assert window.pipeline_nav.current_step_index() == 2


def test_the_structure_step_says_it_is_being_edited_while_the_rail_is_on_it(qtbot, tmp_path) -> None:
    """设计稿帧③ 这一行是「4 层 · 编辑中」，不是「4 层」。

    「编辑中」不是另一个项目字段，它就是「你在这儿」——同一个结构，站在这一步上看是正在
    改的东西，站在别的步上看是已经定下来的层数。
    """
    window, _document = _window(qtbot, _project(tmp_path, layers=4))
    assert _sub(window, "结构") == "6 层"

    _click(window, "结构")

    assert _sub(window, "结构") == "6 层 · 编辑中"


def test_the_oxide_note_gives_way_to_the_you_are_here_suffix(qtbot, tmp_path) -> None:
    """站在结构步上时这一行只写「编辑中」——设计稿帧③ 那叠层里也有一层氧化。

    帧③ 的层列表第一行就是「SiO₂ · 表面氧化层」，而左栏写的是「4 层 · 编辑中」，不是
    「4 层 · 含表面氧化 · 编辑中」。层数后面只挂一截：264px 的栏宽装不下两截，而站在这一步上
    时「你在这儿」比这叠层长什么样更急着说——氧化层此刻就摆在画布的第一行，看得见。
    """
    oxide = api.LayerSpec("SiO2 native oxide", SIO2, 34.2, roughness_a=3.0)
    structure = api.StructureSpec(AIR, (oxide, *_structure(2).components), SI)
    window, _document = _window(qtbot, api.set_structure(_project(tmp_path), "curve", structure))
    assert _sub(window, "结构") == "5 层 · 含表面氧化"

    _click(window, "结构")

    assert _sub(window, "结构") == "5 层 · 编辑中"


def test_the_parameters_step_reads_pending_review_once_a_structure_exists(qtbot, tmp_path) -> None:
    """设计稿帧③ 这一行是「待复核」：结构有了、边界一个都没动过，就是等着复核。

    此前这一行退回「设定参数边界与先验」——那句话在空项目上也是同一句，于是读者分不清
    「还没轮到这一步」和「轮到了，等你看」。
    """
    window, _document = _window(qtbot, _project(tmp_path, layers=2))

    assert _sub(window, "参数") == "待复核"


def test_going_back_to_the_structure_step_brings_the_layer_editor_with_it(qtbot, tmp_path) -> None:
    """点回「结构」得真的把结构编辑器交回来，否则这条路只是换了个高亮的圆点。

    画布的层堆叠和右栏的「选中层」都跟着这一步走，所以这条断了，回到结构那一步就什么也
    改不了。
    """
    window, _document = _window(qtbot, _project(tmp_path, layers=2))

    _click(window, "结构")
    qtbot.wait(1)

    assert window.structure_panel.isVisibleTo(window)
    assert window.inspector_cards["inspectorSelectedLayer"].isVisibleTo(window)
