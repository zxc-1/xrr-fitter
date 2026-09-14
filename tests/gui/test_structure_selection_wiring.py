"""层堆叠里的选中，接到右栏的「选中层」卡上。

帧③ 的两半是一条路径的两端：在画布的层堆叠里点一层，右栏立刻变成那一层的字段；
改完提交，结构走的还是编辑器原本那条 commit。断在中间任何一处，卡片就成了摆设。
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QWidget

import xrr_fitter.api as api

pytest.importorskip("pytestqt")


def _structure() -> api.StructureSpec:
    layer = api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=30.0)
    oxide = api.LayerSpec("SiO2", api.MaterialSpec("SiO2", "SiO2", 2.2), 21.0, roughness_a=4.0)
    return api.StructureSpec(
        api.MaterialSpec("Air", "N", 0.0012),
        (oxide, layer),
        api.MaterialSpec("c-Si", "Si", 2.33),
        backing_roughness_a=3.0,
    )


def _editor(qtbot):
    from xrr_fitter.gui.structure.editor import StructureEditor

    commits: list[api.StructureSpec] = []
    editor = StructureEditor(commits.append, lambda: None, lambda: None)
    qtbot.addWidget(editor)
    editor.load(_structure())
    return editor, commits


def test_selecting_a_layer_announces_it_with_its_index(qtbot) -> None:
    """编辑器发出 (index, component)，右栏据此绑定；这是两半之间唯一的接口。"""
    editor, _commits = _editor(qtbot)
    seen: list[tuple[object, object]] = []
    editor.component_selected.connect(lambda index, component: seen.append((index, component)))

    editor._select_component(1)

    assert seen[-1][0] == 1
    assert seen[-1][1].name == "a-Si"


def test_replacing_a_component_keeps_its_place_in_the_stack(qtbot) -> None:
    """替换而不是删了重加：索引不变，前后的层原样留在原位。"""
    editor, commits = _editor(qtbot)
    thicker = api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 500.0, roughness_a=30.0)

    editor.replace_component(1, thicker)

    assert len(commits) == 1
    components = commits[0].components
    assert [component.name for component in components] == ["SiO2", "a-Si"]
    assert components[1].thickness_a == pytest.approx(500.0)


def test_the_window_wires_the_stack_selection_into_the_inspector(qtbot) -> None:
    """整机接线：点一层，右栏那张卡就写上它的名字。"""
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    editor = window.structure_panel.editor
    editor.load(_structure())

    editor._select_component(1)

    assert window.selected_layer_panel.formula_editor.text() == "a-Si（自定义密度）"
    assert window.findChild(QLabel, "inspectorSelectedLayerTitle").text() == "选中层 · a-Si"


def test_the_card_caption_carries_the_layer_name_like_the_design(qtbot) -> None:
    """设计稿的抬头是一整行「选中层 · a-Si 非晶硅」，不是抬头一行、层名再占一行。

    卡壳的标题同时是这张卡的无障碍名称，所以层名写进抬头既省一行，也让读屏进入这张
    卡时直接听见改的是哪一层。
    """
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    heading = window.findChild(QLabel, "inspectorSelectedLayerTitle")
    assert heading is not None
    assert heading.text() == "选中层"

    window.structure_panel.editor.load(_structure())
    window.structure_panel.editor._select_component(1)

    assert heading.text() == "选中层 · a-Si"
    card = window.findChild(QWidget, "inspectorSelectedLayer")
    assert card.accessibleName() == "选中层 · a-Si"


def _project(tmp_path) -> api.Project:
    """真走一遍导入与校验的项目：声明只有在结构提交进项目之后才存在。

    模块顶上的 ``_structure`` 是直接喂给编辑器的，不过 ``api.set_structure`` 那道物理
    校验——它的 30 Å 粗糙度配 21 Å 的氧化层是过不了的。这里用设计稿帧③ 那一组几何。
    """
    curve = tmp_path / "curve.xy"
    curve.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", "N", 0.0012),
        (
            api.LayerSpec("SiO2", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2, roughness_a=5.1),
            api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
        ),
        api.MaterialSpec("c-Si", "Si", 2.33),
        backing_roughness_a=3.0,
    )
    project = api.add_dataset(api.new_project(), curve, api.InstrumentSpec())
    return api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")


def test_editing_a_layer_leaves_that_layer_selected(qtbot, tmp_path) -> None:
    """改完一格之后，右栏还停在刚改的那一层上。

    ``replace_component`` 存在的理由就是「换内容不换位置」，而位置保住了、选中没保住等于
    没保住：改完厚度想接着改粗糙度，得先回堆叠里把同一层再点一次。中间还会闪过一次空
    卡——上一秒写着 a-Si 的那张卡，因为自己那次编辑而变成「未选择」。
    """
    from PySide6.QtCore import Qt

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.structure_panel.editor._select_component(1)

    window.selected_layer_panel.thickness_editor.setValue(12.0)
    qtbot.keyClick(window.selected_layer_panel.thickness_editor, Qt.Key.Key_Return)

    card = window.selected_layer_panel
    assert card.thickness_editor.isEnabled()
    assert window.findChild(QLabel, "inspectorSelectedLayerTitle").text() == "选中层 · a-Si"
    assert card.thickness_editor.value() == pytest.approx(12.0)


def test_switching_datasets_does_not_carry_the_selection_across(qtbot, tmp_path) -> None:
    """换数据集时选中不跟着走：另一份结构的第 1 层是另一层，不是刚才那一层。

    保留选中是为了「同一份结构被自己的编辑刷新了」，不是为了「第 1 行永远选中」。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = _project(tmp_path)
    second = tmp_path / "second.xy"
    second.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {900.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    project = api.add_dataset(project, second, api.InstrumentSpec())
    document = ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.structure_panel.editor._select_component(1)

    document.replace_project(api.select_active_dataset(document.project, "second"))

    assert window.findChild(QLabel, "inspectorSelectedLayerTitle").text() == "选中层"


def test_the_rails_read_the_same_declarations_the_parameter_table_reads(qtbot, tmp_path) -> None:
    """整机接线：选中一层，卡上三根条报的界限和参数表里那三行是同一份。

    这两处都在讲「拟合器能把这个数推到哪儿」。卡片要是自己编一套，同一层的厚度就会在
    右栏说一个范围、在参数表里说另一个，而两处都没有标注自己说的是哪一种范围。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.railbar import format_number

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)

    window.structure_panel.editor._select_component(1)

    card = window.selected_layer_panel
    declaration = next(item for item in window.parameters_panel.definitions if item.name == "component.1.thickness_a")
    low = declaration.lower / 10.0
    high = declaration.upper / 10.0
    assert card.thickness_rail.text() == f"{format_number(low)} ≤ 48.7 ≤ {format_number(high)}"
    assert card.thickness_rail.fraction() == pytest.approx((48.7 - low) / (high - low))


def test_a_reloaded_declaration_set_reaches_the_rails(qtbot, tmp_path) -> None:
    """结构一改，声明整批换掉，条上的界限必须跟着换。

    厚度的上下界是从曲线的 q 范围和这一层自己的初值推出来的，改一层的厚度就会推出另
    一段。条还挂着上一批界限，读者会照着一段已经不存在的范围去判断「还能推多远」。

    重发声明和「堆叠重新选中这一层」谁先谁后没有保证，所以这一条量的是两条路合起来的
    结果，而不是其中某一条。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument(_project(tmp_path))
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.structure_panel.editor._select_component(1)
    before = window.selected_layer_panel.thickness_rail.text()

    thinner = api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 120.0, roughness_a=4.4)
    window.structure_panel.editor.replace_component(1, thinner)
    qtbot.wait(1)

    after = window.selected_layer_panel.thickness_rail.text()
    assert after != ""
    assert after != before
