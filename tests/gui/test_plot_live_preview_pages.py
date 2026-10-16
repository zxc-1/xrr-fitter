"""实时预览两页都画、色标里认得出、搜索一结束两页一起撤。

审计实测：搜索中的模型只画在对数页上，原始页看不到；色标里也没有这条绿线的名字，读者不知道它是什么。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QLabel, QWidget
from tests.gui.plot_support import _panel
from tests.support.model_cases import prepared_data


def _key_captions(panel, key: str) -> tuple[str, ...]:
    row = panel.findChild(QWidget, f"plotCard:{key}SeriesKey")
    assert row is not None, f"the {key} card has no series key row"
    if row.isHidden():
        return ()
    return tuple(text for label in row.findChildren(QLabel) if (text := label.text()))


def test_the_search_preview_draws_on_both_reflectivity_pages_and_is_keyed(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    model = np.full(data.qz_a_inv.size, 0.5)

    assert panel.set_preview_curve(data.qz_a_inv, model)

    raw = panel.view("raw").preview_item
    assert panel.view("log").preview_item is not None
    assert raw is not None
    np.testing.assert_allclose(raw.yData, model * data.normalization)
    for key in ("log", "raw"):
        assert _key_captions(panel, key) == ("搜索中模型",), key


def test_ending_the_search_drops_the_preview_from_both_pages_and_their_keys(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    panel.set_preview_curve(data.qz_a_inv, np.full(data.qz_a_inv.size, 0.5))

    panel.clear_preview_curve()

    for key in ("log", "raw"):
        assert panel.view(key).preview_item is None, key
        assert _key_captions(panel, key) == (), key
