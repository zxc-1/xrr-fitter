from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QWidget

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
MO = api.MaterialSpec("Mo", "Mo", 10.28)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="stack-gui"),
    )
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def _layer(name="film", material=SIO2, thickness=40.0) -> api.LayerSpec:
    return api.LayerSpec(name, material, thickness, roughness_a=3.0)


def _periodic() -> api.PeriodicBlock:
    """A Mo/Si cell of 65 A repeated 8 times, so 520 A of expanded thickness."""
    return api.PeriodicBlock(
        "Mo/Si",
        (api.LayerSpec("Mo", MO, 25.0), api.LayerSpec("Si", SI, 40.0)),
        repeats=8,
    )


def _stack(structure, height=236):
    from xrr_fitter.gui.structure.stack import stack_bands

    return stack_bands(structure, height)


def _view(qtbot, structure, *, height=236):
    from xrr_fitter.gui.structure.stack import StackView

    view = StackView()
    qtbot.addWidget(view)
    view.resize(120, height)
    view.load(structure)
    return view


def _click(qtbot, view, band) -> None:
    qtbot.mouseClick(view, Qt.LeftButton, pos=QPoint(60, band.top + band.height // 2))


def test_bands_size_each_component_in_proportion_to_its_thickness() -> None:
    """A table of numbers does not show that one layer dwarfs another.

    Reading 20 nm against 60 nm takes arithmetic; a band three times as tall
    states the same ratio without any.
    """
    structure = api.StructureSpec(AIR, (_layer("thin", thickness=20.0), _layer("thick", thickness=60.0)), SI)

    bands = _stack(structure)

    thin, thick = bands[1], bands[2]
    assert thick.height == 3 * thin.height


def test_bands_run_from_fronting_to_backing_and_fill_their_box() -> None:
    """The diagram is a section through the sample, so order and extent are the message."""
    structure = api.StructureSpec(AIR, (_layer("cap"), _layer("film")), SI)

    bands = _stack(structure, height=236)

    assert tuple(band.index for band in bands) == (None, 0, 1, None)
    assert bands[0].top == 0
    assert sum(band.height for band in bands) == 236
    for earlier, later in zip(bands, bands[1:], strict=False):
        assert later.top == earlier.top + earlier.height


def test_semi_infinite_media_get_a_fixed_cap_rather_than_a_thickness_share() -> None:
    """Fronting and backing have no thickness to be proportional to."""
    from xrr_fitter.gui.structure.stack import MEDIUM_BAND_H

    bands = _stack(api.StructureSpec(AIR, (_layer(),), SI))

    assert bands[0].height == MEDIUM_BAND_H
    assert bands[-1].height == MEDIUM_BAND_H
    assert "Si" in bands[-1].detail


def test_a_layer_thousands_of_times_thinner_still_gets_a_visible_band() -> None:
    """A 2 A native oxide under a 4000 A film is 0.1 px of an honest share.

    Rounding it away would hide the layer the diagram exists to show, so thin
    bands are lifted to a floor and the surplus is charged to the tall ones.
    """
    from xrr_fitter.gui.structure.stack import MIN_BAND_H

    structure = api.StructureSpec(AIR, (_layer("oxide", thickness=2.0), _layer("film", thickness=4000.0)), SI)

    bands = _stack(structure, height=236)

    assert bands[1].height >= MIN_BAND_H
    assert bands[2].height > bands[1].height
    assert sum(band.height for band in bands) == 236


def test_periodic_block_draws_as_one_band_weighted_by_its_expanded_thickness() -> None:
    """Expanding 8 repeats into 16 bands would spend the whole box on one component."""
    structure = api.StructureSpec(AIR, (_layer("cap", thickness=130.0), _periodic()), SI)

    bands = _stack(structure, height=236)

    cap, block = bands[1], bands[2]
    assert block.index == 1
    assert "8" in block.detail
    assert block.height == 4 * cap.height


def test_clicking_a_band_selects_the_component_it_stands_for(qtbot) -> None:
    """The diagram is the spatial view of the stack, so it is where a user points."""
    structure = api.StructureSpec(AIR, (_layer("cap"), _layer("film")), SI)
    view = _view(qtbot, structure)
    picked: list[int] = []
    view.component_selected.connect(picked.append)

    _click(qtbot, view, view.bands()[2])

    assert picked == [1]
    assert view.selected_index() == 1


def test_clicking_a_medium_band_selects_nothing(qtbot) -> None:
    """Air and the substrate are not editable components, so they are not targets."""
    view = _view(qtbot, api.StructureSpec(AIR, (_layer(),), SI))
    picked: list[int] = []
    view.component_selected.connect(picked.append)

    _click(qtbot, view, view.bands()[0])

    assert picked == []
    assert view.selected_index() is None


def test_clicking_the_diagram_moves_the_tree_selection(qtbot, tmp_path) -> None:
    """Two views of one stack must not disagree about which layer is current.

    行的右键菜单作用在树的当前行上，所以只点亮示意图的选中会让「删除」指到别处去。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer("cap"), _layer("film")), SI))
    view = panel.findChild(QWidget, "structureStack")
    assert view is not None

    _click(qtbot, view, view.bands()[2])

    assert panel.editor.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == 1
    assert panel.editor.remove_action.isEnabled()


def test_selecting_a_tree_row_highlights_the_matching_band(qtbot, tmp_path) -> None:
    """The sync runs both ways, so the diagram tracks selections made in the tree."""
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer("cap"), _layer("film")), SI))
    view = panel.findChild(QWidget, "structureStack")

    panel.editor.tree.setCurrentItem(panel.editor.tree.topLevelItem(2))

    assert view.selected_index() == 1


def test_clearing_the_editor_empties_the_diagram(qtbot, tmp_path) -> None:
    """A dataset with no structure has no section to draw."""
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(),), SI))
    view = panel.findChild(QWidget, "structureStack")

    panel.editor.clear()

    assert view.bands() == ()


def _swatch_colour(tree, item) -> str | None:
    """The colour actually drawn in a row's swatch, read off the pixmap it carries.

    Reading the pixmap rather than a data role kept beside it means the assertion
    is about what reaches the screen: a role recorded in parallel can agree with
    the diagram while the swatch painted from it does not.
    """
    from xrr_fitter.gui.structure.rows import SWATCH_OBJECT

    row = tree.itemWidget(item, 0)
    label = None if row is None else row.findChild(QLabel, SWATCH_OBJECT)
    pixmap = None if label is None else label.pixmap()
    if pixmap is None or pixmap.isNull():
        return None
    image = pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2).name()


def test_each_tree_row_carries_the_colour_of_its_own_band(qtbot, tmp_path) -> None:
    """设计稿层行的行首是色块（`.lyr > .sw`）：把这一行和图上那条带对上。

    图和树讲的是同一个结构的两种写法，中间只差一个视觉挂钩——没有它，读者得靠数
    行序自己配对，而周期块在图上是一条带、在树里是一行带子行，序号并不同步。
    """
    from xrr_fitter.gui import theme

    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer("cap"), _periodic()), SI))
    view = panel.findChild(QWidget, "structureStack")
    tree = panel.editor.tree

    bands = {band.index: band.fill for band in view.bands()}
    for row in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(row)
        index = item.data(0, Qt.ItemDataRole.UserRole)
        expected = theme.DATA_NEUTRAL if index is None else bands[index]
        assert _swatch_colour(tree, item) == QColor(expected).name(), f"row {row} swatch does not match its band"


def test_the_bounding_media_do_not_spend_a_data_hue(qtbot, tmp_path) -> None:
    """设计系统写明「空气与基底一律取中性灰不占序列色相」。

    半无限介质不是被拟合的层，占掉一个序列色相就让第一层看起来是第二层。
    """
    from xrr_fitter.gui import theme

    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer("cap"),), SI))
    tree = panel.editor.tree

    media = (tree.topLevelItem(0), tree.topLevelItem(tree.topLevelItemCount() - 1))

    assert [_swatch_colour(tree, item) for item in media] == [QColor(theme.DATA_NEUTRAL).name()] * 2
    assert _swatch_colour(tree, tree.topLevelItem(1)) == QColor(theme.DATA_SEQUENCE[0]).name()


def test_a_child_layer_of_a_periodic_block_is_not_given_its_own_colour(qtbot, tmp_path) -> None:
    """周期块在图上只有一条带，所以子行没有可对应的颜色。

    给子行也发一个色块会读成「这两层在图上各有一条带」，而它们合起来才是那条带。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_periodic(),), SI))

    tree = panel.editor.tree
    block = tree.topLevelItem(1)

    assert block.childCount() == 2
    assert [_swatch_colour(tree, block.child(row)) for row in range(2)] == [None, None]
