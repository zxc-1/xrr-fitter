"""Canonical Slice 8 plot tests re-exported from maintainable case partitions."""

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from tests.gui.plot_cases_1 import *  # noqa: F401,F403
from tests.gui.plot_cases_2 import *  # noqa: F401,F403
from tests.gui.plot_cases_3 import *  # noqa: F401,F403
from tests.gui.plot_cases_4 import *  # noqa: F401,F403
from tests.gui.plot_cases_5 import *  # noqa: F401,F403
from tests.gui.plot_cases_6 import *  # noqa: F401,F403


@pytest.fixture(autouse=True)
def _deliver_plot_deferred_deletes() -> None:
    yield
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
