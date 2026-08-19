"""Plot contract cases, partition 6: candidate heatmaps; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


def _heatmap(panel, key):
    """The single image drawn on a heatmap view, as a plain array."""
    images = panel.view(key).axes.images
    assert len(images) == 1
    return np.asarray(images[0].get_array(), dtype=float)


def _row_labels(panel, key):
    return tuple(text.get_text() for text in panel.view(key).axes.get_yticklabels())


def _placeholder_text(panel, key):
    return tuple(text.get_text() for text in panel.view(key).axes.texts)


def _ranked_result(data):
    """Three candidates whose objective order differs from their argument order.

    A result carries the minimum-objective candidate first, so that one cannot be
    moved; the other two arrive out of order, which is what the row ranking fixes.
    """
    return final_fit_result(
        _candidate(data, "candidate-a", objective=0.1),
        _candidate(data, "candidate-b", objective=0.3),
        _candidate(data, "candidate-c", objective=0.2),
    )


def test_residual_heatmap_gives_each_candidate_a_row_ranked_best_first(qtbot) -> None:
    """Ranking by objective puts the fit worth reading at the top row."""
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_ranked_result(data))

    matrix = _heatmap(panel, "residual_map")

    assert matrix.shape == (3, 4)
    # candidate-c holds the lower objective, so it precedes candidate-b despite
    # arriving after it. The selected candidate is marked so a row ties back to
    # the other panes.
    assert _row_labels(panel, "residual_map") == (
        "▶ candidate-a",
        "candidate-c",
        "candidate-b",
    )


def test_residual_heatmap_colour_scale_is_symmetric_about_zero(qtbot) -> None:
    """An over- and an under-shoot of one size must read as equally far out."""
    data = prepared_data(size=4)
    candidate = _candidate(data, weighted_residuals=np.array([-0.5, 0.0, 1.0, 2.0]))
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))

    image = panel.view("residual_map").axes.images[0]

    assert image.get_clim() == (-2.0, 2.0)


def test_residual_heatmap_labels_columns_with_the_qz_each_index_holds(qtbot) -> None:
    """qz is not uniform in the stored angle, so ticks name indices, not an extent."""
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_ranked_result(data))
    view = panel.view("residual_map")
    view.canvas.draw()

    positions = tuple(view.axes.get_xticks())
    labels = tuple(text.get_text() for text in view.axes.get_xticklabels())
    qz = np.asarray(_candidate(data, "candidate-b").qz_a_inv, dtype=float)

    assert view.axes.get_xlabel() == "qz (Å⁻¹)"
    assert labels == tuple(f"{qz[int(index)]:.3g}" for index in positions)


def test_heatmaps_leave_out_candidates_that_may_only_be_inspected(qtbot) -> None:
    """A row for a candidate whose objective is not finite would carry no reading."""
    data = prepared_data(size=4)
    result = final_fit_result(
        _candidate(data, "candidate-a", objective=0.1),
        _candidate(data, "candidate-b", objective=0.2, valid=False),
        _candidate(data, "candidate-c", objective=0.3, stop_reason="early_eliminated"),
    )
    panel = _panel(qtbot, data=data, result=result)

    assert _row_labels(panel, "residual_map") == ("▶ candidate-a",)
    assert _row_labels(panel, "parameter_map") == ("▶ candidate-a",)


def test_parameter_heatmap_places_each_parameter_within_its_own_bounds(qtbot) -> None:
    """A thickness and a unitless scale share no range, so each column normalizes alone."""
    data = prepared_data(size=4)
    scale, thickness = (
        replace(_candidate(data).parameters[0], name="instrument.scale", value=1.4),
        replace(
            _candidate(data).parameters[0],
            name="component.0.thickness_a",
            value=60.0,
            lower=20.0,
            upper=100.0,
        ),
    )
    candidate = _candidate(data, parameters=(scale, thickness))
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))
    view = panel.view("parameter_map")
    view.canvas.draw()

    # scale: (1.4-0.5)/(1.5-0.5); thickness: (60-20)/(100-20)
    assert _heatmap(panel, "parameter_map") == pytest.approx(np.array([[0.9, 0.5]]))
    assert view.axes.images[0].get_clim() == (0.0, 1.0)
    assert tuple(text.get_text() for text in view.axes.get_xticklabels()) == (
        "instrument.scale",
        "component.0.thickness_a",
    )


def test_parameter_heatmap_draws_a_pinned_parameter_mid_scale(qtbot) -> None:
    """A parameter with no room to move holds no position within its bounds."""
    data = prepared_data(size=4)
    pinned = replace(_candidate(data).parameters[0], value=2.0, lower=2.0, upper=2.0)
    panel = _panel(qtbot, data=data, result=final_fit_result(_candidate(data, parameters=(pinned,))))

    assert _heatmap(panel, "parameter_map") == pytest.approx(np.array([[0.5]]))


def test_heatmap_redraws_keep_exactly_one_colour_key(qtbot) -> None:
    """A colour key arrives as an extra axes, so a stale one would eat the width."""
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_ranked_result(data))

    assert len(_colour_keys(panel.view("residual_map"))) == 1
    assert len(_colour_keys(panel.view("parameter_map"))) == 1

    panel.set_result(_ranked_result(data), "candidate-b")

    assert len(_colour_keys(panel.view("residual_map"))) == 1
    assert len(_colour_keys(panel.view("parameter_map"))) == 1


def test_heatmaps_state_why_they_are_empty_before_a_fit_runs(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))

    for key in ("residual_map", "parameter_map"):
        assert _placeholder_text(panel, key) == ("暂无可比较候选",)
        assert len(panel.view(key).axes.images) == 0


def test_parameter_heatmap_refuses_candidates_with_different_parameter_sets(qtbot) -> None:
    data = prepared_data(size=4)
    renamed = replace(_candidate(data).parameters[0], name="component.0.thickness_a")
    result = final_fit_result(
        _candidate(data, "candidate-a", objective=0.1),
        _candidate(data, "candidate-b", objective=0.2, parameters=(renamed,)),
    )
    panel = _panel(qtbot, data=data, result=result)

    assert _placeholder_text(panel, "parameter_map") == ("候选参数集合不一致，无法逐列并排比较",)
    assert _heatmap(panel, "residual_map").shape == (2, 4)


def test_a_heatmap_colour_key_is_left_out_of_the_reference_grid(qtbot) -> None:
    """A key is a legend for another axes, so gridding it adds lines with no reading."""
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_ranked_result(data))
    view = panel.view("residual_map")
    view.canvas.draw()

    key = _colour_keys(view)[0]

    assert key.xaxis._major_tick_kw.get("gridOn", False) is False
    # The image itself is skipped too: a grid would overdraw the cells it reads.
    assert view.axes.xaxis._major_tick_kw.get("gridOn", False) is False
