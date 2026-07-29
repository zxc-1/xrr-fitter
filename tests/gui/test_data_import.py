from __future__ import annotations

from math import asin, degrees
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QTreeWidget,
)

import xrr_fitter.api as api


def _write_curve(path: Path, *, scale: float = 1000.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, document=None):
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    panel = DataPanel(ProjectDocument() if document is None else document)
    qtbot.addWidget(panel)
    return panel


def _instrument() -> api.InstrumentSpec:
    return api.InstrumentSpec(instrument_id="gui-import")


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GUI import bypassed add_dataset")
        ),
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
    assert panel.instrument_text("sample") == (
        "lab-01 · 几何换算 10×0.1 mm · 背景 linear · 分辨率 θ"
    )
    assert item.toolTip(5) == panel.sha256_text("sample")


def test_import_dialog_requires_explicit_beam_choice(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    assert dialog.beam_kind() is None
    assert dialog.import_button().isEnabled() is False
    assert dialog.validation_text() == "请选择光路类型：单色 / 混合 Kα"

    dialog.select_beam_kind("mixed_kalpha")

    assert dialog.import_button().isEnabled() is True
    assert dialog.beam_spec() == api.BeamSpec("mixed_kalpha")


def test_import_dialog_displays_and_uses_monochromatic_wavelength(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    dialog.select_beam_kind("monochromatic")
    editor = dialog.findChild(QDoubleSpinBox, "monochromaticWavelengthEditor")
    assert editor is not None
    editor.setValue(1.2345)

    assert dialog.beam_spec().wavelength_a == pytest.approx(1.2345)


def test_import_dialog_column_mapping_cancel_and_validation(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    assert dialog.column_mapping() is None

    dialog.set_column_mapping(
        two_theta=2,
        intensity=3,
        intensity_sigma=4,
        resolution=5,
        resolution_kind="sigma_q_a_inv",
    )
    assert dialog.column_mapping() == api.DataColumnMapping(2, 3, 4, 5, "sigma_q_a_inv")

    dialog.cancel_column_mapping()
    assert dialog.column_mapping() is None
    with pytest.raises(ValueError, match="distinct nonnegative"):
        dialog.set_column_mapping(two_theta=0, intensity=0)


def test_column_mapping_dialog_blocks_invalid_mapping_with_inline_error(qtbot) -> None:
    from xrr_fitter.gui.data.import_dialog import ColumnMappingDialog

    dialog = ColumnMappingDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QSpinBox, "twoThetaColumnEditor").setValue(0)
    dialog.findChild(QSpinBox, "intensityColumnEditor").setValue(0)
    buttons = dialog.findChild(QDialogButtonBox)

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "columnMappingError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert error is not None and error.isVisible()
    assert "distinct nonnegative" in error.text()


def test_import_dialog_exposes_real_instrument_choices_and_geometry_fields(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    footprint = dialog.findChild(QComboBox, "footprintModeEditor")
    background = dialog.findChild(QComboBox, "backgroundModelEditor")
    resolution = dialog.findChild(QComboBox, "resolutionDomainEditor")
    assert [footprint.itemData(index) for index in range(footprint.count())] == [
        "geometry",
        "fit",
        "none",
    ]
    footprint.setCurrentIndex(footprint.findData("geometry"))
    dialog.findChild(QDoubleSpinBox, "sampleLengthEditor").setValue(10.0)
    dialog.findChild(QDoubleSpinBox, "beamWidthEditor").setValue(0.1)
    background.setCurrentIndex(background.findData("powerlaw"))
    resolution.setCurrentIndex(resolution.findData("theta"))

    instrument = dialog.instrument_spec()

    assert instrument.footprint_mode == "geometry"
    assert instrument.footprint_spill_angle_deg == pytest.approx(degrees(asin(0.01)))
    assert instrument.background_kind == "powerlaw"
    assert instrument.resolution_domain == "theta"


def test_import_action_keeps_parented_dialog_as_modal_window(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(source)], ""),
    )

    def reject(dialog):
        observed.update(
            parent=dialog.parent(),
            is_window=dialog.isWindow(),
            is_modal=dialog.isModal(),
        )
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ImportDialog, "exec", reject)

    panel.findChild(QPushButton, "importFilesButton").click()

    assert observed == {"parent": panel, "is_window": True, "is_modal": True}


def test_import_shortcuts_do_not_conflict_with_project_open(qtbot) -> None:
    panel = _panel(qtbot)
    files = panel.findChild(QShortcut, "importFilesShortcut")
    folder = panel.findChild(QShortcut, "importFolderShortcut")

    assert files.key() == QKeySequence("Ctrl+I")
    assert folder.key() == QKeySequence("Ctrl+Shift+I")
    assert files.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert files.key() != QKeySequence(QKeySequence.StandardKey.Open)
