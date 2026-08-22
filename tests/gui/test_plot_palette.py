"""Matplotlib figures follow the resolved theme palette.

Qt stylesheets do not reach a Figure, so the diagnostic plots kept a hardcoded
white background while the surrounding controls already tracked the platform
appearance. On a dark desktop that left eight glaring white panels. Structural
colours (background, text, grid, spines) now derive from the same tokens the
stylesheet uses; the Okabe-Ito data series colours stay fixed because they are
legible against either background.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from xrr_fitter.gui import theme


def _relative_luminance(value: tuple[float, ...]) -> float:
    """WCAG relative luminance for an RGB(A) tuple of 0..1 channels."""
    channels = []
    for raw in value[:3]:
        channels.append(raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    values = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (values[1] + 0.05) / (values[0] + 0.05)


def test_plot_palette_differs_between_light_and_dark_tokens() -> None:
    light = theme.plot_palette(theme.LIGHT_TOKENS)
    dark = theme.plot_palette(theme.DARK_TOKENS)

    assert light.background != dark.background
    assert light.foreground != dark.foreground


def test_dark_plot_palette_is_actually_dark() -> None:
    dark = theme.plot_palette(theme.DARK_TOKENS)

    assert _relative_luminance(dark.background) < 0.2
    assert _relative_luminance(dark.foreground) > 0.5


def test_plot_text_meets_minimum_contrast_in_both_palettes() -> None:
    """Axis labels and the empty-state message must stay readable."""
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        palette = theme.plot_palette(tokens)
        assert _contrast(palette.foreground, palette.background) >= 4.5
        # The muted empty-state text is large and decorative, so it only needs
        # to clear the 3:1 threshold for non-body text.
        assert _contrast(palette.muted, palette.background) >= 3.0


def test_diagnostic_views_paint_the_resolved_background(qtbot) -> None:
    import numpy as np

    from xrr_fitter.gui.plots.diagnostics import build_tabs
    from xrr_fitter.gui.plots.live import LiveReflectivityPlot

    tabs, views = build_tabs()
    qtbot.addWidget(tabs)
    application = QApplication.instance()
    assert application is not None
    expected = theme.plot_palette(theme.palette_tokens(application.palette())).background

    for view in views.values():
        # The four reflectivity panes render through pyqtgraph, which the Qt
        # stylesheet and apply_figure_palette never reach, so they take the
        # resolved background explicitly; QColor stores channels at reduced
        # precision, so that hand-off round-trips to only ~1e-6 and is compared
        # with a tolerance. The matplotlib views paint the palette straight onto
        # the figure and axes at full float precision, so they still match ==.
        if isinstance(view, LiveReflectivityPlot):
            assert np.allclose(view.background_color(), expected, atol=1e-6)
        else:
            assert view.figure.get_facecolor() == expected
            assert view.axes.get_facecolor() == expected


def test_palette_survives_a_real_draw_that_clears_the_axes(qtbot, monkeypatch) -> None:
    """A real draw leaves the whole figure on the palette, not just at build.

    The tick and title colours are what ``Axes.clear()`` actually discards, so
    they are the load-bearing assertions here. The dark palette is used because
    the Matplotlib default white would be indistinguishable from the light one.
    """
    import numpy as np
    from tests.support.model_cases import prepared_data

    from xrr_fitter.gui.plots import diagnostics
    from xrr_fitter.gui.plots.reflectivity import draw_log

    monkeypatch.setattr(
        diagnostics,
        "current_plot_palette",
        lambda: theme.DARK_PLOT_PALETTE,
    )
    tabs, views = diagnostics.build_tabs()
    qtbot.addWidget(tabs)
    # The log pane now renders through pyqtgraph, so this exercises the
    # matplotlib clear-and-redraw path on the candidates view, a single-axes
    # figure that still owns the .axes/.figure draw_log clears and repaints.
    view = views["candidates"]
    expected = theme.DARK_PLOT_PALETTE

    draw_log(view, prepared_data(size=4), None)

    assert view.axes.get_facecolor() == expected.background
    assert view.figure.get_facecolor() == expected.background
    assert np.allclose(view.axes.title.get_color(), expected.foreground)
    assert np.allclose(view.axes.yaxis.label.get_color(), expected.foreground)


def test_empty_state_message_uses_the_palette_rather_than_a_fixed_grey(qtbot) -> None:
    from xrr_fitter.gui.plots.diagnostics import build_tabs, draw_empty

    tabs, views = build_tabs()
    qtbot.addWidget(tabs)
    # draw_empty writes its centred message onto a matplotlib axes; the log pane
    # is now pyqtgraph, so the candidates view is the single-axes mpl stand-in.
    view = views["candidates"]

    draw_empty(view, "标题")

    message = next(text for text in view.axes.texts if text.get_text() == "暂无可用数据")
    application = QApplication.instance()
    assert application is not None
    expected = theme.plot_palette(theme.palette_tokens(application.palette())).muted
    assert message.get_color() == expected
