"""Data-series hues have one source, and labels drawn on them stay readable.

Four series colours lived as bare hex literals in the two plot modules, and the
structure diagram repeated the same Okabe-Ito set with a comment claiming it
matched "the plot palette's data colours" — a claim nothing enforced. A hue
could drift in one module and keep its meaning in the other, which is the one
failure a colour vocabulary exists to prevent.

Unlike the token colours these deliberately do *not* follow the appearance. A
series that changed hue with the desktop theme would invalidate every reference
a user has already formed ("the blue curve is the data"), so instead each series
hue is required to hold up against both canvas backgrounds.

The band labels are the opposite case: they are drawn *on* the fills, so a fixed
colour cannot work for six of them. The previous fixed white reached 2.25:1 on
the orange fill against the 4.5:1 body-text minimum, so the label colour has to
be chosen per fill.
"""

from __future__ import annotations

import pytest

from xrr_fitter.gui import theme

# The three series that are drawn as lines or markers, so they answer to the
# non-text contrast minimum. The fit-range band is excluded on purpose: it is a
# translucent span, not an outline.
SERIES_COLOURS = ("DATA_OBSERVED", "DATA_CANDIDATE", "DATA_PREVIEW")

WCAG_TEXT_MIN = 4.5
WCAG_NON_TEXT_MIN = 3.0


def _relative_luminance(value: str) -> float:
    channels = []
    for index in (1, 3, 5):
        raw = int(value[index : index + 2], 16) / 255.0
        channels.append(raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _canvas_backgrounds() -> tuple[str, str]:
    """The two figure backgrounds a series has to remain visible against."""
    return ("#FFFFFF", "#1E1F22")


def test_each_series_role_names_a_distinct_hue() -> None:
    """Roles that share a figure must not share a colour.

    Observed data, the committed candidate and the in-flight preview are drawn
    on the same axes at the same time, so a collision between any two of them
    makes the plot unreadable rather than merely inelegant.
    """
    hues = [getattr(theme, name) for name in SERIES_COLOURS]

    assert len(set(hues)) == len(hues)


@pytest.mark.parametrize("name", SERIES_COLOURS)
def test_series_hue_holds_up_against_both_canvas_backgrounds(name: str) -> None:
    """This is what lets the series colours stay fixed while tokens switch.

    A hue is only allowed to ignore the appearance if it is legible in both, so
    the fixed-hue decision is enforced here instead of asserted in a comment.
    """
    hue = getattr(theme, name)

    for background in _canvas_backgrounds():
        assert _contrast(hue, background) >= WCAG_NON_TEXT_MIN


def test_fit_range_hue_stays_clear_of_every_series() -> None:
    """The range band shares the axes with all three series."""
    hues = {getattr(theme, name) for name in SERIES_COLOURS}

    assert theme.DATA_RANGE not in hues


def test_layer_sequence_seats_a_realistic_stack_before_repeating() -> None:
    """Six fills covers the common multilayer without a neighbour collision."""
    assert len(theme.DATA_SEQUENCE) >= 6
    assert len(set(theme.DATA_SEQUENCE)) == len(theme.DATA_SEQUENCE)


def test_bounding_media_stay_outside_the_layer_sequence() -> None:
    """Air and the substrate are not layers, so they cannot look like one.

    They are semi-infinite and get a constant strip; borrowing a sequence hue
    would make the diagram read as though the stack had two more components.
    """
    assert theme.DATA_NEUTRAL not in theme.DATA_SEQUENCE


@pytest.mark.parametrize("fill", (*theme.DATA_SEQUENCE, theme.DATA_NEUTRAL))
def test_every_fill_carries_a_readable_label(fill: str) -> None:
    """The regression this file was written for.

    A fixed white label managed 2.25:1 on the orange fill and 2.31:1 on the sky
    blue. Both are captions a user reads, so the body-text minimum applies.
    """
    assert _contrast(theme.band_label_colour(fill), fill) >= WCAG_TEXT_MIN


def test_label_colour_actually_switches_rather_than_favouring_one_end() -> None:
    """Guards against a "fix" that just swaps white for black everywhere.

    Black clears the minimum on five fills but only reaches 4.05:1 on the deep
    blue, so a single replacement colour cannot pass the parametrised check
    above; the choice has to follow the fill.
    """
    chosen = {theme.band_label_colour(fill) for fill in theme.DATA_SEQUENCE}

    assert len(chosen) == 2
