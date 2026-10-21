"""An objective trace never stitches separate datasets together or rolls back at a stage change.

The axis is the member's own overall completion (``test_fit_progress_members``): a stage change keeps
the earlier points and moves on to the right, and another dataset's frames never land on this line.
"""

import numpy as np
from tests.gui.test_fit_batch_interaction import _frame

from xrr_fitter.gui.fitting.progress import ProgressView


def test_a_stage_change_keeps_the_trace_and_moves_right(qtbot):
    view = ProgressView()
    qtbot.addWidget(view)
    try:
        view.set_progress(_frame("a", "A", completed=10, objective=8.0))
        view.set_progress(_frame("a", "B", completed=1, objective=4.0))
        view.set_progress(_frame("a", "B", completed=2, objective=3.0))

        x, y = view.trace_plot.trace_item.getOriginalDataset()
        np.testing.assert_array_equal(y, [8.0, 4.0, 3.0])
        assert list(x) == sorted(x) and len(set(x)) == 3
    finally:
        view.freeze()


def test_another_datasets_frames_never_land_on_the_trace(qtbot):
    view = ProgressView()
    qtbot.addWidget(view)
    try:
        view.set_progress(_frame("a", "A", completed=10, objective=8.0))
        view.set_progress(_frame("b", "A", completed=1, objective=4.0))
        view.set_progress(_frame("b", "A", completed=2, objective=3.0))

        _x, y = view.trace_plot.trace_item.getOriginalDataset()
        np.testing.assert_array_equal(y, [8.0])
    finally:
        view.freeze()
