"""Cross-cutting consistency of the confidence double-encoding system.

The system spans theme.py (the canonical glyph map, the one verdict->kind map, plus
the "no verdict yet" fallback pair), results/panel.py (CONFIDENCE_STATUS_KINDS, an
alias of theme's map), data/dataset_card.py (the rail card's glyph), data/panel.py
(the dataset row's marker) and chrome.py (the status bar's verdict span). Glyphs and
colours each have a single home in theme; no surface restates either. 设计稿把四个
判定画成四个颜色（``.cf-trusted/.cf-correlated/.cf-multiple/.cf-untrusted``），底栏
帧⑤ 的「判定：◆ 可用但相关」用的就是 ``--info``，所以任何一处把它折到 ``warn`` 上都是
与设计稿相左。这些测试守住「只有一张表」这条不变式。
"""

from __future__ import annotations


def _dict_constants(module: object) -> list[tuple[str, dict]]:
    """模块里的字典常量，按名字排序——下面两处扫描都从这里起步。"""
    return sorted(
        ((name, value) for name, value in vars(module).items() if isinstance(value, dict)),
        key=lambda item: item[0],
    )


def _glyph_tables(module: object, authored: set[str]) -> list[str]:
    """这个模块里有没有第二张「判定→字形」表：值全是字符串，且与 theme 那张有交集。"""
    return [
        name
        for name, value in _dict_constants(module)
        if all(isinstance(entry, str) for entry in value.values()) and set(value.values()) & authored
    ]


def _colour_tables(module: object, verdicts: set[str], kinds: set[str]) -> list[str]:
    """这个模块里有没有第二张「判定→颜色档」表。theme 那张本身按同一性排除，别名不算副本。"""
    from xrr_fitter.gui import theme

    return [
        name
        for name, value in _dict_constants(module)
        if value is not theme.CONFIDENCE_COMPARISON_KINDS and set(value) & verdicts and set(value.values()) <= kinds
    ]


def test_every_surface_reads_glyphs_from_theme() -> None:
    """No surface may keep a glyph table of its own beside theme's.

    Four surfaces render the same verdict — the results badge, the dataset card,
    the status-bar dot, and the dataset row's marker — and a local copy in any of
    them agrees with theme only until one side is edited. The per-surface tests
    prove each one follows a repointed theme glyph; this one states the invariant
    they share, so a fifth surface added later has an obvious rule to meet.

    ``data.dataset_card`` is the fourth, added with the ``.ds`` card list; it is
    listed here rather than left to the docstring's promise, which is what the
    "a fourth surface added later" clause was waiting for.
    """
    from xrr_fitter.gui import chrome, theme
    from xrr_fitter.gui.data import dataset_card
    from xrr_fitter.gui.data import panel as data_panel
    from xrr_fitter.gui.results import panel as results_panel

    authored = set(theme.CONFIDENCE_GLYPHS.values()) | {theme.CONFIDENCE_FALLBACK_GLYPH}
    for module in (chrome, data_panel, dataset_card, results_panel):
        tables = _glyph_tables(module, authored)
        assert tables == [], f"{module.__name__} keeps a second glyph table: {tables}"


def test_panel_status_kinds_cover_every_theme_verdict() -> None:
    """results/panel.py must assign a status kind to every verdict theme draws."""
    from xrr_fitter.gui.results.panel import CONFIDENCE_STATUS_KINDS
    from xrr_fitter.gui.theme import CONFIDENCE_GLYPHS

    for verdict in CONFIDENCE_GLYPHS:
        assert verdict in CONFIDENCE_STATUS_KINDS, f"panel missing verdict: {verdict}"


def test_no_surface_keeps_a_second_verdict_colour_table() -> None:
    """Colours, like glyphs, have exactly one home: theme's comparison map.

    chrome used to keep ``QUALITY_STATUS_KINDS``, which folded 可用但相关 onto
    ``warn`` on the argument that the bar shows one verdict at a time. 设计稿否了
    这个说法：帧⑤ 底栏那一段写的是 ``<b class="cf-correlated">◆ 可用但相关</b>``，
    与右栏候选行同一个 ``--info``。所以这里扫的是「有没有第二张判定→颜色表」，
    而不再是「那处分歧有没有变」。
    """
    from xrr_fitter.gui import chrome, theme
    from xrr_fitter.gui.data import panel as data_panel
    from xrr_fitter.gui.results import panel as results_panel

    verdicts = set(theme.CONFIDENCE_GLYPHS)
    kinds = set(theme.CONFIDENCE_COMPARISON_KINDS.values())
    for module in (chrome, data_panel, results_panel):
        tables = _colour_tables(module, verdicts, kinds)
        assert tables == [], f"{module.__name__} keeps a second verdict colour table: {tables}"


def test_panel_kinds_are_theme_kinds_themselves() -> None:
    """The panel's name for the map is an alias, not a copy that can drift."""
    from xrr_fitter.gui.results.panel import CONFIDENCE_STATUS_KINDS
    from xrr_fitter.gui.theme import CONFIDENCE_COMPARISON_KINDS

    assert CONFIDENCE_STATUS_KINDS is CONFIDENCE_COMPARISON_KINDS


def test_the_four_verdicts_keep_four_distinct_kinds() -> None:
    """可用但相关 must stay separable from 多解 everywhere, per the design."""
    from xrr_fitter.gui.theme import CONFIDENCE_COMPARISON_KINDS, CONFIDENCE_GLYPHS

    assert set(CONFIDENCE_COMPARISON_KINDS) == set(CONFIDENCE_GLYPHS)
    assert CONFIDENCE_COMPARISON_KINDS["可用但相关"] == "info"
    assert CONFIDENCE_COMPARISON_KINDS["多解"] == "warn"
    assert len(set(CONFIDENCE_COMPARISON_KINDS.values())) == len(CONFIDENCE_COMPARISON_KINDS)
