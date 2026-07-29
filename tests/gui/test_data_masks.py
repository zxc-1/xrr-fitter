from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import xrr_fitter.api as api


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    source = _write_curve(tmp_path / "sample.xy")
    document = ProjectDocument()
    panel = DataPanel(document)
    qtbot.addWidget(panel)
    panel.add_paths(
        (source,),
        beam=api.BeamSpec("monochromatic"),
        instrument=api.InstrumentSpec(instrument_id="mask-test"),
    )
    return panel, source


def test_data_panel_composes_range_and_point_masks_without_changing_source(
    qtbot,
    tmp_path,
) -> None:
    panel, _source = _panel(qtbot, tmp_path)
    dataset_id = panel.active_dataset_id
    before = panel.document.project.datasets[0]
    events: list[tuple[str, tuple[bool, ...]]] = []
    panel.mask_changed.connect(lambda key, mask: events.append((key, tuple(mask))))

    panel.set_fit_range(dataset_id, 0.15, 0.45)
    panel.set_point_enabled(dataset_id, 8, False)

    dataset = panel.document.project.datasets[0]
    assert dataset.source_path == before.source_path
    assert dataset.source_sha256 == before.source_sha256
    assert dataset.fit_range_two_theta_deg == pytest.approx((0.15, 0.45))
    assert dataset.fit_mask[8] is False
    assert dataset.fit_mask[0] is False
    assert len(events) == 2
    assert events[-1] == (dataset_id, dataset.fit_mask)


def test_invalid_mask_inputs_do_not_mutate_state(qtbot, tmp_path) -> None:
    panel, _source = _panel(qtbot, tmp_path)
    dataset_id = panel.active_dataset_id
    before = panel.document.project
    events: list[object] = []
    panel.mask_changed.connect(events.append)

    with pytest.raises(ValueError, match="finite lower <= upper"):
        panel.set_fit_range(dataset_id, float("nan"), 1.0)
    with pytest.raises(ValueError, match="at least one valid point"):
        panel.set_fit_range(dataset_id, -10.0, -1.0)
    with pytest.raises(IndexError, match="point index out of range"):
        panel.set_point_enabled(dataset_id, 32, False)

    assert panel.document.project is before
    assert events == []


def test_mask_editor_commits_only_through_set_fit_mask(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _source = _panel(qtbot, tmp_path)
    dataset_id = panel.active_dataset_id
    original = panel.document.project
    expected_mask = np.asarray(original.datasets[0].fit_mask, dtype=bool)
    expected_mask[8] = False
    updated = api.set_fit_mask(original, dataset_id, expected_mask)
    calls: list[tuple[object, ...]] = []

    def set_fit_mask(project, key, mask):
        calls.append((project, key, mask.copy()))
        return updated

    monkeypatch.setattr(api, "set_fit_mask", set_fit_mask)

    panel.set_point_enabled(dataset_id, 8, False)

    assert len(calls) == 1
    assert calls[0][0] is original
    assert calls[0][1] == dataset_id
    assert calls[0][2].dtype == np.bool_
    assert np.array_equal(calls[0][2], expected_mask)
    assert panel.document.project is updated


def test_instrument_change_commits_only_through_set_instrument(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _source = _panel(qtbot, tmp_path)
    dataset_id = panel.active_dataset_id
    original = panel.document.project
    instrument = api.InstrumentSpec(
        instrument_id="updated",
        footprint_mode="none",
    )
    updated = api.set_instrument(original, dataset_id, instrument)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_instrument",
        lambda project, key, value: (
            calls.append((project, key, value)),
            updated,
        )[1],
    )

    panel.set_instrument(dataset_id, instrument)

    assert calls == [(original, dataset_id, instrument)]
    assert panel.document.project is updated
    assert panel.instrument_text(dataset_id).startswith("updated · 无足迹修正")


def test_mask_change_failure_is_atomic_and_emits_no_success(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _source = _panel(qtbot, tmp_path)
    dataset_id = panel.active_dataset_id
    before = panel.document.project
    events: list[object] = []
    panel.mask_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "set_fit_mask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("mask rejected")),
    )

    with pytest.raises(ValueError, match="mask rejected"):
        panel.set_point_enabled(dataset_id, 8, False)

    assert panel.document.project is before
    assert events == []


def test_initial_fit_range_uses_service_valid_points(qtbot, tmp_path) -> None:
    path = tmp_path / "invalid-first-point.xy"
    rows = ["0.05 -5.0"]
    rows.extend(
        f"{0.07 + index * 0.02:.6f} {1200.0 / (index + 1):.12g}"
        for index in range(31)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    panel = DataPanel(ProjectDocument())
    qtbot.addWidget(panel)
    panel.add_paths(
        (path,),
        beam=api.BeamSpec("monochromatic"),
        instrument=api.InstrumentSpec(),
    )

    assert panel.document.project.datasets[0].fit_range_two_theta_deg[0] == pytest.approx(
        0.07
    )
