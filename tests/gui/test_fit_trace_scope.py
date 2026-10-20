"""A per-stage objective axis cannot connect separate stages or datasets."""

import numpy as np
import pytest
from tests.gui.test_fit_batch_interaction import _frame

from xrr_fitter.gui.fitting.progress import ProgressView


@pytest.mark.parametrize("next_owner,next_stage", (("a", "B"), ("b", "A")))
def test_objective_trace_starts_again_at_stage_or_owner_boundary(qtbot, next_owner, next_stage):
    view = ProgressView()
    qtbot.addWidget(view)
    try:
        view.set_progress(_frame("a", "A", completed=10, objective=8.0))
        view.set_progress(_frame(next_owner, next_stage, completed=1, objective=4.0))
        view.set_progress(_frame(next_owner, next_stage, completed=2, objective=3.0))

        x, y = view.trace_plot.trace_item.getData()
        np.testing.assert_array_equal(x, [0.1, 0.2])
        np.testing.assert_array_equal(y, [4.0, 3.0])
    finally:
        view.freeze()
