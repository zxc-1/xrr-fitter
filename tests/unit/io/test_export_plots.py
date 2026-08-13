from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from hashlib import sha256

import matplotlib
import numpy as np
import pytest
from matplotlib._pylab_helpers import Gcf
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    prepared_data,
    project,
)

from xrr_fitter.io.export_plots import (
    BAND_PAIRS,
    fit_overview_png,
    parameter_trends_png,
    residuals_png,
    sld_profile_png,
)
from xrr_fitter.io.export_tables import DatasetExportData, ExportReplayIdentity
from xrr_fitter.model.analysis import SldUncertaintyBands, UncertaintyReport
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
    residual = np.log10(model + data.r_floor) - np.log10(data.intensity_normalized + data.r_floor)
    diagnostics = (PhysicsDiagnostic("review-diagnostic", "diagnostic text", (0, 2, 4)),)
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
        (dataset_id, f"{index:03d}-dataset-aaaaaaaa") for index, dataset_id in enumerate(dataset_ids, start=1)
    )
    return tuple(
        DatasetExportData(
            project=value,
            dataset=context.dataset,
            data=context.data,
            directory_mapping=mapping,
            selected=context.selected,
            replay_identity=context.replay_identity,
            matching_surface_oxide_rejection=(context.matching_surface_oxide_rejection),
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


def test_export_parameter_trends_use_stable_order_labels_for_unicode_dataset_ids(
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
    assert tick_calls[0] == (("1", "2"), None)


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
    assert all(not lower <= included_between <= upper for lower, upper in expected)


def _zero_width_bands() -> SldUncertaintyBands:
    # Five quantile faces on one depth grid; the two published pairs are present
    # so both fill_between groups render. Distinct rows keep the banded PNG
    # byte-different from the bandless one without needing a real MCMC replay.
    depth = np.linspace(0.0, 40.0, 4)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    real = np.tile(np.arange(len(levels), dtype=float)[:, None], (1, depth.size))
    return SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label="基底界面",
        sample_count=500,
        total_samples=2000,
        failure_rate=0.0,
    )


def _context_with_bands() -> DatasetExportData:
    # Fold a band into the selected result while preserving object identity:
    # `replace` keeps the candidates tuple (so `selected` still belongs to the
    # result) and `project` rebuilds around the same dataset the export owns.
    base = _context()
    report = UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        sld_bands=_zero_width_bands(),
    )
    result = replace(base.result, uncertainty=report)
    dataset = replace(base.dataset, last_valid_result=result)
    return DatasetExportData(
        project(dataset),
        dataset,
        base.data,
        base.directory_mapping,
        base.selected,
        base.replay_identity,
        base.matching_surface_oxide_rejection,
    )


# Frozen from the three renderers at ``bb5f253^`` in the locked Matplotlib
# environment, before SLD-band drawing existed.  A fresh render is compared to
# both size and digest so this is not the former same-function self-comparison.
BANDLESS_PNG_BASELINES = {
    fit_overview_png: (24674, "6c4d698235b052044b8d8502439c526fc4a59411ba15519cc13c57c107ced107"),
    sld_profile_png: (28382, "2a344e8e2d86aa0a392d3a0111c543d6d916013e1b980101c8c86f485f223d0e"),
    residuals_png: (27318, "3962537e44db0923454d7d79db3fac0c31e25678c13309be5ddf3c325f3a17d1"),
}


@pytest.mark.parametrize(("render", "committed"), BANDLESS_PNG_BASELINES.items())
def test_export_png_without_bands_matches_the_committed_render(render, committed) -> None:
    context = _context()
    assert context.result.uncertainty is None or context.result.uncertainty.sld_bands is None

    payload = render(context)

    assert (len(payload), sha256(payload).hexdigest()) == committed


def test_export_band_legend_uses_the_published_quantile_labels() -> None:
    assert tuple(label for _pair, _alpha, label in BAND_PAIRS) == ("16–84%", "2.5–97.5%")


def test_sld_profile_png_with_bands_differs_and_stays_deterministic() -> None:
    banded = _context_with_bands()
    first = sld_profile_png(banded)
    assert first == sld_profile_png(banded)
    assert first != sld_profile_png(_context())


def test_sld_profile_png_takes_its_caption_from_the_band_object(monkeypatch) -> None:
    banded = _context_with_bands()
    calls = []
    original = SldUncertaintyBands.caption
    monkeypatch.setattr(
        SldUncertaintyBands,
        "caption",
        lambda self: calls.append(1) or original(self),
    )
    sld_profile_png(banded)
    assert len(calls) == 1


def test_sld_profile_png_renders_cjk_caption_without_missing_glyphs(caplog) -> None:
    # 导出的 SLD 剖面用 set_title 打上中文可信带说明；默认字体 DejaVu Sans 没有
    # CJK 字形，导出层又不能复用 GUI 的字体助手（架构禁止 io->gui），所以它必须自行
    # 解析 CJK 字族。镜像 GUI 的字形守卫：warnings 与 matplotlib 日志两条通道都不得
    # 出现 "Glyph ... missing"，否则中文说明在导出 PNG 里会渲染成豆腐块。
    banded = _context_with_bands()
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = sld_profile_png(banded)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert not [warning for warning in caught if "Glyph" in str(warning.message)]
    assert "glyph" not in caplog.text.lower()
