"""候选比较图与热图按结果列表里的序号称呼候选（候选 A / B / C），与列表对得上。

审计实测：结果列表把候选叫「候选 A」「候选 B」，候选比较图的图例和热图的行名却写求解器 ID
（``E-0``、``automatic-refit-2``），读者在两处之间对不上号。

序号是列表那一行的位置，而列表不按求解器产出的先后排：推荐解排第一，其余按目标值。所以期望的名字
从列表本身读出来，不在这里另写一份——图跟着列表走，列表的排法改了，这里不必跟着改。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from tests.gui.plot_support import _candidate, _panel, final_fit_result
from tests.support.model_cases import prepared_data

from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
from xrr_fitter.gui.results.candidates import CandidateList


def _ranked(data):
    """结果里的先后是 a、b、c，目标值的先后是 a、c、b：列表的序号与结果里的先后因此不是一回事。"""
    return final_fit_result(
        _candidate(data, "candidate-a", objective=0.1),
        _candidate(data, "candidate-b", objective=0.3),
        _candidate(data, "candidate-c", objective=0.2),
    )


def _listed(qtbot, result) -> tuple[tuple[str, str], ...]:
    """结果列表自上而下的 ``(candidate_id, 名字)``。"""
    listing = CandidateList()
    qtbot.addWidget(listing)
    listing.project_result("curve", result, None)
    items = (listing.item(row) for row in range(listing.count()))
    return tuple((item.data(Qt.ItemDataRole.UserRole), item.data(CANDIDATE_ROW_ROLE).title) for item in items)


def test_the_candidate_comparison_keys_candidates_by_their_list_ordinal(qtbot) -> None:
    data = prepared_data(size=4)
    result = _ranked(data)
    listed = _listed(qtbot, result)
    objectives = {candidate.candidate_id: candidate.objective for candidate in result.candidates}
    assert dict(listed)["candidate-b"] != "候选 B", "前提：列表的序号不是结果里的先后"

    panel = _panel(qtbot, data=data, result=result)
    labels = [text.get_text() for text in panel.view("candidates").axes.get_legend().get_texts()]

    assert [" · ".join(label.split(" · ")[:2]) for label in labels] == [
        f"{title} · J={objectives[candidate_id]:g}" for candidate_id, title in listed
    ]


def test_the_heatmaps_name_rows_by_the_same_ordinals(qtbot) -> None:
    data = prepared_data(size=4)
    result = _ranked(data)
    titles = dict(_listed(qtbot, result))
    panel = _panel(qtbot, data=data, result=result)
    # 行序仍按目标值（a、c、b），名字跟列表；正在看的那一个带 ▶。
    expected = ("▶ " + titles["candidate-a"], titles["candidate-c"], titles["candidate-b"])

    for key in ("residual_map", "parameter_map"):
        rows = tuple(text.get_text() for text in panel.view(key).axes.get_yticklabels())
        assert rows == expected, key
