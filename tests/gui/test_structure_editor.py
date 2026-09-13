from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
)

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _document(tmp_path, *, beam=None):
    from xrr_fitter.gui.document import ProjectDocument

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="structure-gui"),
        beam=beam,
    )
    return ProjectDocument(project)


def _panel(qtbot, tmp_path, *, beam=None):
    from xrr_fitter.gui.structure.panel import StructurePanel

    panel = StructurePanel(_document(tmp_path, beam=beam))
    qtbot.addWidget(panel)
    return panel


def _bare(backing=SI) -> api.StructureSpec:
    return api.StructureSpec(AIR, (), backing)


def _layer(name="film", material=SIO2, thickness=40.0) -> api.LayerSpec:
    return api.LayerSpec(name, material, thickness, roughness_a=3.0)


def _periodic() -> api.PeriodicBlock:
    return api.PeriodicBlock(
        "Mo/Si",
        (
            api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0),
            api.LayerSpec("Si", SI, 40.0),
        ),
        repeats=8,
    )


def test_structure_edit_replaces_active_dataset_project_structure(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    structure = api.StructureSpec(AIR, (_layer(),), SI)
    updated = api.set_structure(original, "sample", structure)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_structure",
        lambda project, dataset_id, value: (
            calls.append((project, dataset_id, value)),
            updated,
        )[1],
    )
    events: list[tuple[str, object]] = []
    panel.structure_changed.connect(lambda key, value: events.append((key, value)))

    panel.set_structure(structure)

    assert calls == [(original, "sample", structure)]
    assert panel.document.project is updated
    assert panel.structure == structure
    assert events == [("sample", structure)]


def test_imported_dataset_can_initialize_default_structure_from_visible_button(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    button = panel.findChild(QPushButton, "initializeStructureButton")
    add_layer = panel.findChild(QPushButton, "addLayerButton")

    assert button is not None and button.isVisibleTo(panel)
    assert add_layer is not None and add_layer.isEnabled() is False

    qtbot.mouseClick(button, Qt.LeftButton)

    assert panel.structure == _bare()
    assert button.isHidden()
    assert add_layer.isEnabled() is True


def test_structure_edit_failure_rolls_back_project_editor_and_signal(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "set_structure",
        lambda *_args: (_ for _ in ()).throw(ValueError("structure rejected")),
    )

    with pytest.raises(ValueError, match="structure rejected"):
        panel.set_structure(_bare())

    assert panel.document.project is before
    assert panel.structure is None
    assert events == []


def test_structure_editor_validates_with_active_mixed_kalpha_beam(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    beam = api.BeamSpec("mixed_kalpha")
    panel = _panel(qtbot, tmp_path, beam=beam)
    original = panel.document.project
    structure = _bare()
    updated = api.set_structure(original, "sample", structure)
    observed: list[api.BeamSpec] = []

    def set_structure(project, dataset_id, candidate):
        observed.append(project.datasets[0].beam)
        return updated

    monkeypatch.setattr(api, "set_structure", set_structure)

    panel.set_structure(structure)

    assert observed == [beam]


def test_structure_editor_adds_and_removes_ordinary_layer(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_bare())
    layer = _layer()

    panel.add_layer(layer)
    assert panel.structure.components == (layer,)

    panel.remove_component(0)
    assert panel.structure.components == ()


def test_structure_editor_adds_and_moves_periodic_block(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    layer = _layer()
    block = _periodic()
    panel.set_structure(api.StructureSpec(AIR, (layer,), SI))

    panel.add_periodic_block(block)
    changed = panel.move_component(1, 0)

    assert changed is True
    assert panel.structure.components == (block, layer)


@pytest.mark.parametrize(
    ("method", "arguments"),
    (
        ("remove_component", (-1,)),
        ("remove_component", (1,)),
        ("move_component", (-1, 0)),
        ("move_component", (0, 1)),
        ("move_component", (1, 0)),
    ),
)
def test_structure_editor_rejects_invalid_component_indices_transactionally(
    qtbot,
    tmp_path,
    method,
    arguments,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(),), SI))
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)

    with pytest.raises(IndexError, match="component index out of range"):
        getattr(panel, method)(*arguments)

    assert panel.document.project is before
    assert events == []


def test_structure_editor_noop_move_emits_nothing_and_preserves_identity(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(),), SI))
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)

    changed = panel.move_component(0, 0)

    assert changed is False
    assert panel.document.project is before
    assert events == []


def test_structure_editor_tree_keeps_fixed_roots_and_periodic_children_contained(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    block = _periodic()
    panel.set_structure(api.StructureSpec(AIR, (_layer(), block), SI))
    tree = panel.findChild(QTreeWidget, "structureTree")

    assert tree.topLevelItemCount() == 4
    assert tree.topLevelItem(0).text(0) == "空气"
    # 基底那一行报的是它的材料名（设计稿写的是「c-Si · 晶体硅基底」），不是「基底」这个
    # 占位词——半无限介质也该说清自己是什么。
    assert tree.topLevelItem(3).text(0) == "Si"
    periodic = tree.topLevelItem(2)
    assert periodic.text(0) == "Mo/Si"
    assert periodic.childCount() == 2
    assert [periodic.child(index).text(0) for index in range(2)] == ["Mo", "Si"]


def test_structure_tree_fits_the_dock_width_without_horizontal_scrolling(
    qtbot,
    tmp_path,
) -> None:
    # 面板被拖窄到 320px 时，行里让步的必须是名字那一栏。``QLabel`` 的最小宽度就是整句
    # 话的宽度，压不下去——不放开它，放不下的那一截会把右端的读数（「3 参数」）顶出
    # 可视区，而读数是这一行唯一的定量信息，名字被截断还能从副行和色块认出来。
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(), _periodic()), SI))
    panel.resize(320, 400)
    panel.show()
    qtbot.waitExposed(panel)
    tree = panel.findChild(QTreeWidget, "structureTree")

    assert tree.columnWidth(0) <= tree.viewport().width()
    assert tree.horizontalScrollBar().maximum() == 0

    row = tree.itemWidget(tree.topLevelItem(1), 0)
    reading = row.findChild(QLabel, "structureLayerReading")
    assert reading.geometry().right() <= row.width()
    assert reading.width() >= reading.sizeHint().width()


def test_add_layer_dialog_commits_explicit_nm_fields_through_direct_method(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "layerNameInput").setText("cap")
    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(4.5)
    dialog.findChild(QDoubleSpinBox, "layerRoughnessInput").setValue(0.3)
    buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.layer() == api.LayerSpec("cap", SIO2, 45.0, roughness_a=3.0)


def test_add_layer_dialog_keeps_full_stack_error_open_then_commits_correction(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_bare())
    dialog = LayerDialog(panel, commit_layer=panel.add_layer)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit, "layerNameInput").setText("too-rough")
    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(1.0)
    roughness = dialog.findChild(QDoubleSpinBox, "layerRoughnessInput")
    roughness.setValue(1.0)
    buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.isVisible()
    assert error is not None and error.isVisible()
    assert "roughness_a must be below 4.9 A" in error.text()
    assert panel.structure.components == ()

    roughness.setValue(0.1)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert tuple(layer.name for layer in panel.structure.components) == ("too-rough",)


def _drop_row(tree, source_row: int, target_row: int, *, below: bool = False) -> None:
    """把 ``source_row`` 拖到 ``target_row`` 上，走真实的 ``dropEvent``。

    ``qtbot`` 没有拖放动作，原生 drag 会阻塞在系统的事件循环里；能测的这一层就是树
    收到 drop 之后怎么算目标行，所以直接把事件递给它。
    """
    from PySide6.QtCore import QMimeData, QPoint
    from PySide6.QtGui import QDropEvent
    from PySide6.QtWidgets import QAbstractItemView, QApplication

    # 先量这一条：树没宣告可拖放时，Qt 基类的 ``dropEvent`` 会去解引用从未建立的拖放
    # 状态而直接段错误，撞在这里才看得出是「没接拖放」而不是解释器崩了。
    assert tree.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    tree.setCurrentItem(tree.topLevelItem(source_row))
    rect = tree.visualItemRect(tree.topLevelItem(target_row))
    offset = rect.height() - 2 if below else 2
    position = QPoint(rect.center().x(), rect.top() + offset)
    event = QDropEvent(
        position,
        Qt.DropAction.MoveAction,
        QMimeData(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tree.dropEvent(event)
    QApplication.processEvents()


def test_dragging_a_stack_row_reorders_the_structure(qtbot, tmp_path) -> None:
    """卡片副标题承诺「拖动排序」，拖一行就要真的换层序。

    树的第一行和最后一行是空气与基底两个半无限介质，不是组件；行号要先映射回组件
    索引，否则拖到第 2 行会被当成拖到组件 1。
    """
    panel = _panel(qtbot, tmp_path)
    oxide = _layer("SiO2", SIO2, 21.0)
    film = _layer("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0)
    panel.set_structure(api.StructureSpec(AIR, (oxide, film), SI))
    tree = panel.editor.tree

    # 行序是 空气 / SiO2 / a-Si / 基底：把 a-Si 拖到 SiO2 上面。
    _drop_row(tree, 2, 1)

    assert [component.name for component in panel.structure.components] == ["a-Si", "SiO2"]


def test_dropping_a_row_onto_the_bounding_media_leaves_the_order_alone(qtbot, tmp_path) -> None:
    """空气和基底是半无限介质，层不能排到它们外面去，拖过去就该什么也不发生。"""
    panel = _panel(qtbot, tmp_path)
    oxide = _layer("SiO2", SIO2, 21.0)
    film = _layer("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0)
    panel.set_structure(api.StructureSpec(AIR, (oxide, film), SI))
    tree = panel.editor.tree

    _drop_row(tree, 2, 0)
    _drop_row(tree, 1, 3, below=True)

    assert [component.name for component in panel.structure.components] == ["SiO2", "a-Si"]


def _design_structure() -> api.StructureSpec:
    """设计稿帧③ 层堆叠里那四行：空气 / SiO₂ / a-Si / c-Si。"""
    return api.StructureSpec(
        AIR,
        (
            api.LayerSpec("SiO₂ · 表面氧化层", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2, roughness_a=5.1),
            api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
        ),
        api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
        backing_roughness_a=3.0,
    )


def _rows(tree) -> list:
    """每个顶层行的行控件（设计稿 ``.lyr``）。"""
    return [tree.itemWidget(tree.topLevelItem(index), 0) for index in range(tree.topLevelItemCount())]


def test_the_layer_stack_is_the_design_s_row_list_not_a_six_column_table(qtbot, tmp_path) -> None:
    """设计稿帧③ 的层堆叠是四行 ``.lyr``：色块 / 名字加副行 / 右端读数。

    这张卡此前是六列表格（名称 · 材料 · 密度 · 厚度 · 粗糙度 · 重复）。264px 的画布列里
    六列每列摊到四十来个像素，2.19 被截成 ``2.…``，而列头占掉的一行高度换来的是六个此刻
    没人在读的字。设计稿把同样的信息压成一行两级文字：名字一级，「厚度 · 粗糙 · 密度」
    退到副行——要判断的是这个搭法对不对，不是逐列对数字。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_design_structure())
    tree = panel.findChild(QTreeWidget, "structureTree")

    assert tree.columnCount() == 1
    assert tree.header().isHidden() is True

    rows = _rows(tree)
    assert [row.name_text() for row in rows] == [
        "空气",
        "SiO₂ · 表面氧化层",
        "a-Si · 非晶硅薄膜",
        "c-Si · 晶体硅基底",
    ]
    assert [row.detail_text() for row in rows] == [
        "入射介质 · 半无限",
        "厚度 3.42 · 粗糙 0.51 · 密度 2.19",
        "厚度 48.70 · 粗糙 0.44 · 密度 2.28",
        "粗糙 0.30 · 密度 2.33（锁定）",
    ]
    assert [row.reading_text() for row in rows] == ["SLD 0 · 半无限", "3 参数", "3 参数", "半无限"]


def test_the_substrate_says_its_density_can_never_be_fitted(qtbot, tmp_path) -> None:
    """设计稿帧③ 基底那行写的是「密度 2.33（锁定）」，别的行的密度后面什么也不跟。

    这不是一句修辞：``default_parameter_definitions`` 给基底只发了一个声明
    （``backing.roughness_a``）。层的密度有 ``component.N.density_scale`` 可以放开，基底
    的密度连声明都没有，任何拟合都动不了它。副行报的是「这一行有什么」，而基底行的两个数
    里一个能拟合一个不能——不点出来，读者只会去参数表里找那个永远找不到的密度。

    右端读数「半无限」说的是另一件事（它没有厚度），两个都留着才说得全。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_design_structure())
    tree = panel.findChild(QTreeWidget, "structureTree")
    rows = _rows(tree)

    assert rows[-1].detail_text().endswith("（锁定）")
    assert "（锁定）" not in rows[1].detail_text()


def test_the_bounding_media_rows_are_set_apart_from_the_layers_that_carry_parameters(
    qtbot,
    tmp_path,
) -> None:
    """设计稿 ``.lyr.semi``：空气和基底压一档底色，选中的层戴 ``.lyr.sel`` 的左缘强调条。

    半无限介质和可编辑的层在同一张清单里，读者要先分清哪两行是「边界条件」——它们没有
    厚度、点了也编辑不了。设计稿靠底色分档，而不是靠读者去读副行才发现。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_design_structure())
    tree = panel.findChild(QTreeWidget, "structureTree")
    rows = _rows(tree)

    assert [row.property("semiInfinite") for row in rows] == [True, False, False, True]

    tree.setCurrentItem(tree.topLevelItem(2))

    assert [row.property("rowSelected") for row in rows] == [False, False, True, False]


def test_a_periodic_block_reports_its_repeats_in_the_row_it_owns(qtbot, tmp_path) -> None:
    """重复那一列没了，周期数改由块自己的副行报——它本来就只对这一种行有意义。

    六列里 ``重复`` 有值的只有周期块，其余每一行都空着；一列常驻只为一种行服务，代价
    是另外三行各让出一截宽度。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_periodic(),), SI))
    tree = panel.findChild(QTreeWidget, "structureTree")
    block = _rows(tree)[1]

    assert block.detail_text() == "周期 ×8 · 2 层"
    assert block.reading_text() == "6 参数"


def test_an_auto_added_oxide_reads_as_the_design_writes_it_not_as_generated_text(qtbot, tmp_path) -> None:
    """设计稿帧③ 那一行写「SiO₂ · 表面氧化层」。

    服务层给自动加入的氧化层生成的名字是 ``"SiO2 native oxide"``——那个串是
    ``services.materials`` 认领这一层的钥匙，不是给人看的字。显示层把它翻过来，中文
    界面上就不会突然冒出一行英文，化学式也排成设计稿的下标。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (api.LayerSpec("SiO2 native oxide", SIO2, 34.2),), SI))
    tree = panel.findChild(QTreeWidget, "structureTree")

    assert tree.topLevelItem(1).text(0) == "SiO₂ · 表面氧化层"


def test_the_row_widget_is_the_only_thing_that_paints_a_layer_name(qtbot, tmp_path) -> None:
    """设计稿一行只有一个名字，实现里却有两份。

    行是 ``setItemWidget`` 放进树里的一个控件，而 ``QTreeWidgetItem`` 自己的显示文本
    仍由树的委托绘制在它底下。行控件的背景是透明的（底色只有半无限那两行有），所以
    委托画的那份名字从行控件底下透出来，和行自己的名字叠印成一团。

    显示文本不能删——读屏器与按文本定位的测试都靠它。让委托那一份不着墨即可：字还在，
    只是由行控件来画。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (api.LayerSpec("a-Si", SIO2, 34.2),), SI))
    tree = panel.findChild(QTreeWidget, "structureTree")

    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        assert item.text(0) != ""
        assert tree.itemWidget(item, 0).name_text() == item.text(0)
