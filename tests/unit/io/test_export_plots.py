from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib._pylab_helpers import Gcf
import pytest

from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    prepared_data,
    project,
)
from xrr_fitter.io.export_plots import (
    fit_overview_png,
    parameter_trends_png,
    residuals_png,
    sld_profile_png,
)
from xrr_fitter.io.export_tables import DatasetExportData, ExportReplayIdentity
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _context(
    dataset_id: str = "curve",
    *,
    excluded: tuple[tuple[int, int], ...] = (),
) -> DatasetExportData:
    data = prepared_data(size=40)
    mask = np.array(data.fit_mask, copy=True)
    for start, stop in excluded:
        mask[start:stop] = False
    data = replace(data, fit_mask=mask)
    model = data.intensity_normalized * 0.97
    residual = np.log10(model + data.r_floor) - np.log10(
        data.intensity_normalized + data.r_floor
    )
    diagnostics = (
        PhysicsDiagnostic("review-diagnostic", "diagnostic text", (0, 2, 4)),
    )
    candidate = replace(
        fit_candidate(),
        qz_a_inv=data.qz_a_inv,
        model_normalized=model,
        log_residuals_decades=residual,
        weighted_residuals=residual / 0.05,
        sld_depth_a=np.array([0.0, 20.0, 40.0]),
        sld_profile_a2=np.array([0.0, 2.0e-5 + 1.0e-7j, 1.0e-5], dtype=complex),
        diagnostics=diagnostics,
    )
    result = final_fit_result(candidate)
    dataset = dataset_project(dataset_id, result=result)
    dataset = replace(
        dataset,
        source_sha256=data.source_sha256,
        fit_mask=tuple(bool(value) for value in mask),
        fit_range_two_theta_deg=(
            float(data.two_theta_deg[0]),
            float(data.two_theta_deg[-1]),
        ),
    )
    value = project(dataset)
    return DatasetExportData(
        value,
        dataset,
        data,
        ((dataset.dataset_id, f"001-{dataset_id}-aaaaaaaa"),),
        candidate,
        ExportReplayIdentity(1, 10101, 20202),
        False,
    )


def _project_contexts(*dataset_ids: str) -> tuple[DatasetExportData, ...]:
    originals = tuple(_context(dataset_id) for dataset_id in dataset_ids)
    value = project(*(context.dataset for context in originals))
    mapping = tuple(
        (dataset_id, f"{index:03d}-dataset-aaaaaaaa")
        for index, dataset_id in enumerate(dataset_ids, start=1)
    )
    return tuple(
        DatasetExportData(
            project=value,
            dataset=context.dataset,
            data=context.data,
            directory_mapping=mapping,
            selected=context.selected,
            replay_identity=context.replay_identity,
            matching_surface_oxide_rejection=(
                context.matching_surface_oxide_rejection
            ),
        )
        for context in originals
    )


def test_export_plots_are_deterministic_pngs_and_close_every_figure() -> None:
    context = _context()
    before = tuple(Gcf.get_all_fig_managers())

    first = (
        fit_overview_png(context),
        sld_profile_png(context),
        residuals_png(context),
    )
    second = (
        fit_overview_png(context),
        sld_profile_png(context),
        residuals_png(context),
    )

    assert first == second
    assert all(value.startswith(b"\x89PNG\r\n\x1a\n") for value in first)
    assert tuple(Gcf.get_all_fig_managers()) == before
    assert context.dataset.structure is context.project.datasets[0].structure
    assert context.selected.diagnostics[0].code == "review-diagnostic"


def test_export_parameter_trends_are_deterministic_and_use_project_order() -> None:
    contexts = _project_contexts("first", "second")
    before = tuple(Gcf.get_all_fig_managers())

    first = parameter_trends_png(tuple(reversed(contexts)))
    second = parameter_trends_png(contexts)

    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert tuple(Gcf.get_all_fig_managers()) == before


def test_export_parameter_trends_render_unicode_dataset_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    contexts = _project_contexts("样品一", "样品二")
    tick_calls: list[tuple[tuple[str, ...], object]] = []
    original = Axes.set_xticks

    def capture(axis, ticks, labels=None, *args, **kwargs):
        if labels is not None:
            tick_calls.append((tuple(labels), kwargs.get("fontproperties")))
        return original(axis, ticks, labels, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xticks", capture)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"Glyph .* missing from font.*",
            category=UserWarning,
        )
        payload = parameter_trends_png(tuple(reversed(contexts)))

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert tick_calls[0][0] == ("样品一", "样品二")
    assert isinstance(tick_calls[0][1], FontProperties)


def test_export_pngs_ignore_process_global_matplotlib_style() -> None:
    first, second = _project_contexts("first", "second")
    expected = (
        fit_overview_png(first),
        sld_profile_png(first),
        residuals_png(first),
        parameter_trends_png((first, second)),
    )

    with matplotlib.rc_context(
        {
            "axes.facecolor": "#00ff00",
            "axes.labelsize": 19,
            "font.family": "serif",
            "font.size": 23,
            "savefig.transparent": True,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    ):
        observed = (
            fit_overview_png(first),
            sld_profile_png(first),
            residuals_png(first),
            parameter_trends_png((first, second)),
        )

    assert observed == expected


def test_export_residual_plot_shades_disjoint_exclusions_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    context = _context(excluded=((10, 13), (30, 34)))
    qz = context.selected.qz_a_inv
    expected = (
        (float(qz[10]), float(qz[12])),
        (float(qz[30]), float(qz[33])),
    )
    spans: list[tuple[float, float]] = []
    original = Axes.axvspan

    def capture(axis, lower, upper, *args, **kwargs):
        spans.append((float(lower), float(upper)))
        return original(axis, lower, upper, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvspan", capture)

    payload = residuals_png(context)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert spans == [*expected, *expected]
    included_between = float(qz[20])
    assert all(
        not lower <= included_between <= upper
        for lower, upper in expected
    )
