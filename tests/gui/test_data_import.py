"""Qt data-import contracts for public API routing and explicit instrument input.

The suite keeps dialog validation, filename material parsing, active selection,
and immutable project adoption observable at the panel boundary.
"""

from __future__ import annotations

from math import asin, degrees
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QTreeWidget,
)
from tests.gui.data_import_support import _instrument, _panel, _saved_preset, _write_curve

import xrr_fitter.api as api


def test_saved_measurement_preset_skips_the_full_import_dialog(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from dataclasses import replace

    from xrr_fitter.gui.data.import_dialog import ImportDialog
    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))
    source = _write_curve(tmp_path / "P1 Zr.xy")
    monkeypatch.setattr(
        ImportDialog,
        "exec",
        lambda _dialog: (_ for _ in ()).throw(AssertionError("dialog opened")),
    )

    panel.import_paths((source,))

    assert panel.document.project.measurement_preset is _saved_preset() or (
        panel.document.project.measurement_preset == _saved_preset()
    )


def test_first_automatic_import_persists_measurement_configuration(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    panel = _panel(qtbot)
    source = _write_curve(tmp_path / "P1 Zr.xy")

    def accept(dialog: ImportDialog):
        dialog.select_beam_kind("monochromatic")
        dialog.instrument_id.setText("first-use-lab")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ImportDialog, "exec", accept)

    panel._confirm_import((source,), folder=False)

    preset = panel.document.project.measurement_preset
    assert preset is not None
    assert preset.preset_id == "first-use-lab"
    assert preset.instrument.instrument_id == "first-use-lab"
    assert preset.beam == api.BeamSpec("monochromatic", wavelength_a=1.5406)


@pytest.mark.parametrize("select_source", (False, True))
def test_cancelled_measurement_preset_change_does_not_affect_next_import(
    qtbot,
    tmp_path,
    monkeypatch,
    select_source: bool,
) -> None:
    """Keep a cancelled replacement request local to one UI action.

    The empty selection covers cancelling the native file chooser. Selecting a
    source and rejecting ``ImportDialog`` covers cancellation after the source is
    known. Neither path may force the next ordinary import back through the full
    measurement dialog.
    """
    from dataclasses import replace

    from xrr_fitter.gui.data.import_dialog import ImportDialog
    from xrr_fitter.gui.document import ProjectDocument

    panel = _panel(
        qtbot,
        ProjectDocument(replace(api.new_project(), measurement_preset=_saved_preset())),
    )
    source = _write_curve(tmp_path / "P1 Zr.xy")
    selected = [str(source)] if select_source else []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: (selected, ""),
    )
    monkeypatch.setattr(
        ImportDialog,
        "exec",
        lambda _dialog: QDialog.DialogCode.Rejected,
    )

    panel._change_measurement_preset()

    assert panel._force_preset_dialog is False


def test_ambiguous_substrate_is_requested_once_per_structure_group(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from dataclasses import replace

    from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog
    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))
    sources = (
        _write_curve(tmp_path / "P1 Si+Zr.xy"),
        _write_curve(tmp_path / "P2 Si+Zr.xy"),
    )
    dialogs: list[SubstrateDialog] = []

    def accept(dialog: SubstrateDialog):
        dialogs.append(dialog)
        dialog.substrate_editor.setText("Al2O3")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SubstrateDialog, "exec", accept)

    panel.import_paths(sources)

    assert len(dialogs) == 1
    assert len(panel.document.project.datasets) == 2


def test_successful_automatic_import_keeps_dataset_pending(
    qtbot,
    tmp_path,
) -> None:
    from dataclasses import replace

    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))

    result = panel.import_paths((_write_curve(tmp_path / "P1 Zr.xy"),))

    dataset = panel.document.project.datasets[0]
    assert result.imported_dataset_ids == ("P1",)
    assert dataset.automation.import_batch_id == result.import_batch_id
    assert dataset.automation.status.value == "pending"
    assert dataset.last_valid_result is None
    assert panel.document.project.batch_mode == "independent"


def test_data_panel_imports_multiple_xy_files_and_selects_active_dataset(
    qtbot,
    tmp_path,
) -> None:
    paths = (
        _write_curve(tmp_path / "first.xy"),
        _write_curve(tmp_path / "second.xy", scale=800.0),
    )
    panel = _panel(qtbot)
    events: list[tuple[str, ...]] = []
    active: list[str | None] = []
    panel.datasets_imported.connect(events.append)
    panel.active_dataset_changed.connect(active.append)

    panel.add_paths(
        paths,
        beam=api.BeamSpec("monochromatic"),
        instrument=_instrument(),
    )

    assert events == [("first", "second")]
    assert active == ["first"]
    assert panel.dataset_ids == ("first", "second")
    assert panel.active_dataset_id == "first"
    assert panel.document.project.ui_state.active_dataset_id == "first"
    assert panel.status_text("first") == "可拟合"
    assert len(panel.sha256_text("first")) == 64


def test_import_allocates_duplicate_stem_ids_and_preserves_active_dataset(
    qtbot,
    tmp_path,
) -> None:
    first = _write_curve(tmp_path / "first" / "sample.xy")
    second = _write_curve(tmp_path / "second" / "sample.xy", scale=800.0)
    panel = _panel(qtbot)
    beam = api.BeamSpec("monochromatic")
    instrument = _instrument()

    panel.add_paths((first, second), beam=beam, instrument=instrument)
    panel.add_paths((first,), beam=beam, instrument=instrument)

    assert panel.dataset_ids == ("sample", "sample-2", "sample-3")
    assert panel.active_dataset_id == "sample"
    assert panel.document.project.ui_state.active_dataset_id == "sample"


def test_import_routes_mutation_only_through_add_dataset_and_renders_returned_ids(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    first_path = _write_curve(tmp_path / "a" / "sample.xy")
    second_path = _write_curve(tmp_path / "b" / "sample.xy", scale=900.0)
    initial = api.new_project()
    first = api.add_dataset(initial, first_path, _instrument())
    second = api.add_dataset(first, second_path, _instrument())
    from xrr_fitter.gui.document import ProjectDocument

    panel = _panel(qtbot, ProjectDocument(initial))
    calls: list[tuple[object, ...]] = []
    returned = iter((first, second))

    def add_dataset(project, path, instrument, **kwargs):
        calls.append((project, Path(path), instrument, kwargs))
        return next(returned)

    monkeypatch.setattr(api, "add_dataset", add_dataset)
    monkeypatch.setattr(
        api,
        "import_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GUI import bypassed add_dataset")),
    )
    beam = api.BeamSpec("mixed_kalpha")

    panel.add_paths(
        (first_path, second_path),
        beam=beam,
        instrument=_instrument(),
    )

    assert panel.dataset_ids == ("sample", "sample-2")
    assert [call[0] for call in calls] == [initial, first]
    assert [call[1] for call in calls] == [first_path, second_path]
    assert all(call[3]["beam"] is beam for call in calls)


def test_data_panel_import_failure_is_atomic_and_emits_no_success(
    qtbot,
    tmp_path,
) -> None:
    good = _write_curve(tmp_path / "good.xy")
    broken = tmp_path / "broken.xy"
    broken.write_text("not numeric XRR data\n", encoding="utf-8")
    panel = _panel(qtbot)
    before = panel.document.project
    events: list[tuple[str, ...]] = []
    panel.datasets_imported.connect(events.append)

    with pytest.raises(ValueError, match="broken[.]xy"):
        panel.add_paths(
            (good, broken),
            beam=api.BeamSpec("monochromatic"),
            instrument=_instrument(),
        )

    assert panel.document.project is before
    assert panel.dataset_ids == ()
    assert events == []


def test_data_panel_rejects_empty_import_without_success_signal(qtbot) -> None:
    panel = _panel(qtbot)
    events: list[tuple[str, ...]] = []
    panel.datasets_imported.connect(events.append)

    with pytest.raises(ValueError, match="at least one path"):
        panel.add_paths(
            (),
            beam=api.BeamSpec("monochromatic"),
            instrument=_instrument(),
        )

    assert panel.dataset_ids == ()
    assert events == []


def test_data_panel_folder_import_filters_and_sorts_deterministically(
    qtbot,
    tmp_path,
) -> None:
    folder = tmp_path / "folder"
    for relative in ("b.XY", "A.xy", "zeta.dat", "nested/C.XY", "nested/d.txt"):
        _write_curve(folder / relative)
    (folder / "ignored.csv").write_text("ignored\n", encoding="utf-8")
    panel = _panel(qtbot)

    panel.add_folder(
        folder,
        beam=api.BeamSpec("monochromatic"),
        instrument=_instrument(),
        recursive=True,
    )

    assert panel.dataset_ids == ("A", "b", "C", "d", "zeta")


def test_data_panel_shows_source_beam_and_instrument_summaries(qtbot, tmp_path) -> None:
    source = _write_curve(tmp_path / "sample.xy")
    spill_angle = degrees(asin(0.1 / 10.0))
    instrument = api.InstrumentSpec(
        instrument_id="lab-01",
        footprint_mode="geometry",
        footprint_spill_angle_deg=spill_angle,
        sample_length_mm=10.0,
        beam_width_mm=0.1,
        background_kind="linear",
        resolution_domain="theta",
    )
    panel = _panel(qtbot)

    panel.add_paths(
        (source,),
        beam=api.BeamSpec("mixed_kalpha"),
        instrument=instrument,
    )

    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    item = tree.topLevelItem(0)
    assert item.text(0) == "sample"
    assert item.toolTip(1) == str(source)
    assert panel.beam_text("sample").startswith("混合 Kα")
    assert panel.instrument_text("sample") == ("lab-01 · 几何换算 10×0.1 mm · 背景 linear · 分辨率 θ")
    assert item.toolTip(5) == panel.sha256_text("sample")


def test_the_dataset_row_says_how_many_points_it_carries(qtbot, tmp_path) -> None:
    """设计稿每一帧的数据集行都写着点数：「θ/2θ · 512 点 · 已拟合」。

    点数是读者判断这条曲线值不值得拟合的第一个数，也是「可拟合 / 数据点不足」这句判定
    的依据——只给判定不给数，读者看到「数据点不足」也不知道差多少。它和判定同格，因为
    分开两列会让 264px 的紧凑视图再挤掉一列名字的宽度。

    数的是掩码长度而不是参与拟合的点数：这一行报的是这个文件里有多少点，范围裁剪之后
    还剩多少是画布上那件事。
    """
    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)

    panel.add_paths((source,), beam=api.BeamSpec("monochromatic"), instrument=_instrument())

    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    assert panel.point_count_text("sample") == "32 点"
    assert tree.topLevelItem(0).text(4) == "32 点 · 可拟合"


def test_the_point_count_leads_a_failed_source_status_too(qtbot, tmp_path) -> None:
    """源文件出问题时点数还是它上一次记下的那个数，判定换成故障那句。

    这一格的两半答的是两件事：有多少点，以及这条曲线现在能不能用。源文件缺失只推翻
    后者——存档里那份点数是导入当时数出来的，不因为文件被移开而变成未知。
    """
    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)
    panel.add_paths((source,), beam=api.BeamSpec("monochromatic"), instrument=_instrument())
    source.unlink()
    panel.document.refresh_sources()

    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    assert tree.topLevelItem(0).text(4) == "⛔ 32 点 · 源文件缺失"


def test_import_shortcuts_do_not_conflict_with_project_open(qtbot) -> None:
    panel = _panel(qtbot)
    files = panel.findChild(QShortcut, "importFilesShortcut")
    folder = panel.findChild(QShortcut, "importFolderShortcut")

    assert files.key() == QKeySequence("Ctrl+I")
    assert folder.key() == QKeySequence("Ctrl+Shift+I")
    assert files.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert files.key() != QKeySequence(QKeySequence.StandardKey.Open)


def test_data_panel_renders_precise_source_status_with_marker(qtbot, tmp_path) -> None:
    panel = _panel(qtbot)
    source = _write_curve(tmp_path / "original.xy")
    panel.add_paths(
        (source,),
        beam=api.BeamSpec("monochromatic"),
        instrument=_instrument(),
    )
    dataset_id = panel.dataset_ids[0]
    tree = panel.tree

    # A healthy source renders with no marker and a positive status label.
    assert panel.status_text(dataset_id) == "可拟合"
    assert panel.status_marker(dataset_id) == ""

    # Removing the source file makes the status specific and glanceable: the
    # marker ("⛔") flags attention and the label names the exact failure.
    source.unlink()
    panel.document.refresh_sources()

    assert panel.status_text(dataset_id) == "源文件缺失"
    assert panel.status_marker(dataset_id) == "⛔"
    # The tree cell must combine marker + label so both appear in the list.
    item = tree.topLevelItem(0)
    status_column = 4
    assert "⛔" in item.text(status_column)
    assert "源文件缺失" in item.text(status_column)


def test_data_panel_shows_fit_status_per_dataset(qtbot) -> None:
    # Multi-dataset work needs an at-a-glance answer to "which curves are done,
    # and how trustworthy is each result?" without opening every dataset. The
    # tree therefore carries a fit column: unfitted datasets read "未拟合", and
    # fitted ones surface the persisted confidence label plus a glyph.
    from dataclasses import replace

    from tests.support.model_cases import (
        dataset_project,
        final_fit_result,
        fit_candidate,
        project,
    )

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.model.analysis import ConfidenceClass

    trusted = replace(
        final_fit_result(fit_candidate("candidate-a", 0.2)),
        confidence=ConfidenceClass.TRUSTED,
    )
    fitted = dataset_project("fitted", result=trusted)
    pending = dataset_project("pending", result=None)
    value = replace(project(fitted, pending), base_directory="/private/tmp")
    panel = _panel(qtbot, ProjectDocument(value))

    assert panel.fit_status_text("pending") == "未拟合"
    assert panel.fit_status_text("fitted") == "可信"
    assert panel.fit_status_marker("fitted") == "●"
    assert panel.fit_status_marker("pending") == ""

    tree = panel.tree
    fit_column = 6
    assert tree.headerItem().text(fit_column) == "拟合"
    rows = {
        tree.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole): tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
    }
    assert "可信" in rows["fitted"].text(fit_column)
    assert "●" in rows["fitted"].text(fit_column)
    assert "未拟合" in rows["pending"].text(fit_column)


def test_dataset_row_glyph_is_sourced_from_theme(qtbot, monkeypatch) -> None:
    # The dataset row and the results badge state the same verdict, so they must
    # state it with the same shape. Two literal glyph tables agree only until one
    # is edited, and theme is where the design's shapes are authored — repointing
    # a verdict there and watching the row follow is what proves the row reads
    # that source rather than keeping a copy of its own.
    from dataclasses import replace

    from tests.support.model_cases import (
        dataset_project,
        final_fit_result,
        fit_candidate,
        project,
    )

    import xrr_fitter.gui.theme as theme
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.model.analysis import ConfidenceClass

    monkeypatch.setitem(theme.CONFIDENCE_GLYPHS, "多解", "✦")
    multiple = replace(
        final_fit_result(fit_candidate("candidate-a", 0.2)),
        confidence=ConfidenceClass.MULTIPLE,
    )
    value = replace(project(dataset_project("fitted", result=multiple)), base_directory="/private/tmp")
    panel = _panel(qtbot, ProjectDocument(value))

    assert panel.fit_status_marker("fitted") == "✦"


def test_active_dataset_row_stays_emphasised_when_tree_loses_focus(qtbot) -> None:
    # Qt's selection highlight fades when the tree loses focus, so after clicking
    # into the plot or parameters the user can no longer tell which dataset is
    # active. A bold name persists regardless of focus, keeping the active row
    # identifiable at a glance across the whole workspace.
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, project

    from xrr_fitter.gui.document import ProjectDocument

    first = dataset_project("first")
    second = dataset_project("second")
    value = replace(project(first, second), base_directory="/private/tmp")
    document = ProjectDocument(value)
    document.select_active_dataset("second")
    panel = _panel(qtbot, document)

    tree = panel.tree
    rows = {
        tree.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole): tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
    }
    assert rows["second"].font(0).bold()
    assert not rows["first"].font(0).bold()


def test_data_panel_summarises_dataset_overview(qtbot, tmp_path) -> None:
    # After a batch import the tree can hold many rows; a one-line aggregate
    # answers "how many are ready and how many need a look" without scanning
    # every row. The count reuses the same per-row fittability judgement.
    panel = _panel(qtbot)
    assert panel.import_summary_text() == ""

    first = _write_curve(tmp_path / "first.xy")
    second = _write_curve(tmp_path / "second.xy")
    panel.add_paths(
        (first, second),
        beam=api.BeamSpec("monochromatic"),
        instrument=_instrument(),
    )
    assert panel.import_summary_text() == "共 2 个数据集 · 全部可拟合"
    assert not panel.summary_label.isHidden()

    # A source problem is surfaced in the aggregate, not only in one buried row.
    first.unlink()
    panel.document.refresh_sources()
    assert panel.import_summary_text() == "共 2 个数据集 · 可拟合 1 · 需注意 1"


def _stack() -> api.StructureSpec:
    """一叠最简单的层：一层 SiO₂ 压在硅基底上。"""
    return api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.2), 20.0, roughness_a=2.0),),
        api.MaterialSpec("Si", "Si", 2.329),
        backing_roughness_a=3.0,
    )


def test_the_structure_step_says_what_the_structure_covers_instead_of_the_census(qtbot, tmp_path) -> None:
    """设计稿帧③ 的页脚不清点数据集，写的是「结构对全部可拟合数据集共享 · 每集独立仪器/标度」。

    帧① 那句「共 4 个数据集 · 可拟合 3 · 需注意 1」（HTML 381）回答的是「导进来这批能用几
    条」——那是数据那一步的问题。到了结构这一步（HTML 640），读者要知道的是「我在编的这叠
    层管着哪几条曲线，哪些东西不跟着走」；没有这一句，改一处层厚会被读成只改了当前这一集，
    而改仪器又会被读成一起改了别的集。

    共享是能量的事实，不是标语：``services/structures.set_structure`` 只在联合批量下把结构
    传播给全部数据集，独立模式下只落在指名的那一集。所以这句话只在真的每条可拟合曲线都拿着
    同一叠层时才写，否则退回清点——写一句假的比不写更糟。
    """
    panel = _panel(qtbot)
    panel.add_paths(
        (_write_curve(tmp_path / "first.xy"), _write_curve(tmp_path / "second.xy")),
        beam=api.BeamSpec("monochromatic"),
        instrument=_instrument(),
    )
    first_id, second_id = (dataset.dataset_id for dataset in panel.document.project.datasets)

    panel.set_step(1)
    # 还没有结构：这一步无从说「共享」什么。
    assert panel.summary_text() == "共 2 个数据集 · 全部可拟合"

    stack = _stack()
    panel.document.replace_project(api.set_structure(panel.document.project, first_id, stack))
    # 只有一条拿着这叠层，「对全部可拟合数据集共享」此刻是假的。
    assert panel.summary_text() == "共 2 个数据集 · 全部可拟合"

    panel.document.replace_project(api.set_structure(panel.document.project, second_id, stack))
    assert panel.summary_text() == "结构对全部可拟合数据集共享 · 每集独立仪器/标度"
    assert panel.summary_label.text() == "结构对全部可拟合数据集共享 · 每集独立仪器/标度"

    # 别的步骤仍旧清点：这句话回答的是结构那一步的问题。
    panel.set_step(4)
    assert panel.summary_text() == "共 2 个数据集 · 全部可拟合"
    assert panel.summary_label.text() == "共 2 个数据集 · 全部可拟合"


def test_the_chosen_angle_convention_reaches_the_imported_datasets(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    """对话框里选的约定必须一路走到数据集，否则这个开关只是个装饰。

    通道是 ``MeasurementPreset``：``import_angle_offset_deg`` 已经在那里，两者都是「这台
    仪器输出的角度列怎么解释」。preset 跟项目一起存下来，所以下一批同仪器的数据默认沿用
    同一个约定，不必每次重选。
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    panel = _panel(qtbot)
    source = _write_curve(tmp_path / "P1 Zr.xy")

    def accept(dialog: ImportDialog):
        dialog.select_beam_kind("monochromatic")
        dialog.instrument_id.setText("grazing-lab")
        dialog.theta_convention.setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ImportDialog, "exec", accept)

    panel._confirm_import((source,), folder=False)

    project = panel.document.project
    assert project.measurement_preset.angle_convention == "theta"
    assert project.datasets[0].angle_convention == "theta"
    # 同一份文件按 2θ 读一遍来比：入射角归一到散射角，拟合区间正好翻倍。
    plain = api.add_dataset(api.new_project(), source, _instrument())
    assert project.datasets[0].fit_range_two_theta_deg == tuple(
        2.0 * value for value in plain.datasets[0].fit_range_two_theta_deg
    )


def test_the_batch_preview_lists_the_dataset_and_stack_each_filename_declares(qtbot, tmp_path) -> None:
    """批量导入把「这个文件成了哪个数据集、认出了哪叠层」逐行摆出来。

    已存预设的批量导入一个对话框都不弹（``_confirm_import`` 直接走 ``import_paths``），
    而数据集编号和整叠层都是从文件名推的。推成什么样此前只有一个个点开数据集反推得出来，
    二十个文件就是二十次。
    """
    from dataclasses import replace

    from xrr_fitter.gui.data.panel import BATCH_PREVIEW_HEADERS
    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))

    # 没导过东西就没有预览可看——这张表跟 ``failure_table`` 一样默认不占位。
    assert panel.batch_preview_table.isHidden()

    panel.import_paths(
        (
            _write_curve(tmp_path / "P1 Zr.xy"),
            _write_curve(tmp_path / "P2 Nb.xy"),
        )
    )

    table = panel.batch_preview_table
    assert not table.isHidden()
    assert [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())] == list(
        BATCH_PREVIEW_HEADERS
    )
    assert table.rowCount() == 2
    # 第一列必须是磁盘上的文件名，不是 ``display_name``——后者已经是解析结果，跟第二列
    # 一字不差，两列写同一个值就等于少了一列。
    assert [table.item(row, 0).text() for row in range(2)] == ["P1 Zr.xy", "P2 Nb.xy"]
    assert [table.item(row, 1).text() for row in range(2)] == ["P1", "P2"]
    assert [table.item(row, 2).text() for row in range(2)] == ["Zr", "Nb"]


def test_the_batch_preview_says_so_when_a_filename_declares_no_stack(qtbot, tmp_path) -> None:
    """文件名里没有材料就把这件事写出来，而不是留一格空白。

    这样的文件照样导入成功，只是没有自动结构。空白格看起来跟「还没算出来」一个样，而这一
    格的实际含义是「这一条得手工建层」——一批里混进几个，导入前看见与导完逐个点开发现，
    差的是整批的返工。
    """
    from dataclasses import replace

    from xrr_fitter.gui.data.panel import BATCH_PREVIEW_NO_STACK
    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))

    result = panel.import_paths((_write_curve(tmp_path / "unnamed.xy"),))

    assert result.imported_dataset_ids == ("unnamed",)
    assert panel.batch_preview_table.item(0, 2).text() == BATCH_PREVIEW_NO_STACK
