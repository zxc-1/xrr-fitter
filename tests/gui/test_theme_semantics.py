"""Semantic state colours resolve through theme tokens, not inline stylesheets.

The confidence badge used to carry four hardcoded hex values. They were the
light-theme colours, so on a dark desktop the badge kept emitting the light
green while every control around it had already switched. Worse, the colour
reached the screen through an inline ``setStyleSheet`` — the exact practice the
theme module exists to remove — while the ``semanticColor`` dynamic property it
also set was consumed by no stylesheet rule at all: computed on every refresh
and never painted.

The badge therefore joins the existing ``statusKind`` channel. Confidence has
four levels against three status colours, so an informational token covers the
"usable but correlated" case: it is a state, not a call to action, and must stay
distinguishable from the accent colour that marks primary buttons.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel

from xrr_fitter.gui import theme

STATUS_KINDS = ("ok", "info", "warn", "error")


def _palette(window: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    return palette


def test_every_status_kind_has_a_token_in_both_appearances() -> None:
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        colours = {getattr(tokens, kind) for kind in STATUS_KINDS}
        assert len(colours) == len(STATUS_KINDS)


def test_informational_token_tracks_the_appearance() -> None:
    """A state colour that ignores the palette is the bug being fixed."""
    assert theme.LIGHT_TOKENS.info != theme.DARK_TOKENS.info


def test_informational_token_stays_distinct_from_the_accent() -> None:
    """Reusing the accent would make a status badge read as a button."""
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        assert tokens.info != tokens.accent


@pytest.mark.parametrize("kind", STATUS_KINDS)
def test_stylesheet_paints_each_status_kind_from_the_resolved_token(kind: str) -> None:
    """A property no rule consumes is how the old badge lost its colour."""
    for window, tokens in (("#FFFFFF", theme.LIGHT_TOKENS), ("#1E1F22", theme.DARK_TOKENS)):
        sheet = theme.build_stylesheet(_palette(window))
        assert f'QLabel[statusKind="{kind}"]' in sheet
        assert f'QLabel[statusKind="{kind}"] {{ color: {getattr(tokens, kind)}; }}' in sheet


def test_confidence_badge_keeps_the_emphasis_the_inline_style_carried() -> None:
    """The inline rule set a weight as well as a colour; both had to survive.

    Moving the badge onto the token channel is only lossless if the stylesheet
    replaces every declaration the inline string carried, so the verdict still
    reads as the panel's headline rather than as one more label.
    """
    sheet = theme.build_stylesheet(_palette("#FFFFFF"))

    assert "QLabel#confidenceBadge { font-weight: 600; }" in sheet


@pytest.mark.parametrize("kind", STATUS_KINDS)
def test_set_status_kind_accepts_every_painted_kind(qtbot, kind: str) -> None:
    label = QLabel()
    qtbot.addWidget(label)

    theme.set_status_kind(label, kind)

    assert label.property("statusKind") == kind


def test_set_status_kind_still_rejects_an_unpainted_kind(qtbot) -> None:
    """The guard is what keeps a typo from silently losing the colour."""
    label = QLabel()
    qtbot.addWidget(label)

    with pytest.raises(ValueError):
        theme.set_status_kind(label, "informational")
