from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest
from tests.support.synthetic_recovery import build_corpus
from tests.support.synthetic_recovery_runs import _CaseOutcome


def _case(category="single_layer"):
    return next(case for case in build_corpus() if case.category == category)


def _record(case=None):
    case = case or _case()
    outcome = _CaseOutcome(
        case.case_id,
        errors_by_family=(("thickness_period", (0.01,)),),
        closed_included=1,
    )
    return json.loads(json.dumps({"outcome": asdict(outcome), "elapsed_seconds": 2.5}))


def test_record_roundtrip_preserves_every_outcome_field(load_tool_module) -> None:
    module = load_tool_module("statistical_records")
    case = _case()
    payload = _record(case)

    outcome, elapsed = module.decode_record(case, payload)
    encoded = module.encode_record(outcome, elapsed)

    assert json.loads(json.dumps(encoded)) == payload
    assert outcome == _CaseOutcome(case.case_id, (("thickness_period", (0.01,)),), closed_included=1)
    assert elapsed == 2.5


@pytest.mark.parametrize("elapsed", [True, -1, float("nan"), float("inf"), "2.5", None])
def test_record_rejects_invalid_case_duration(elapsed, load_tool_module) -> None:
    module = load_tool_module("statistical_records")
    payload = _record()
    payload["elapsed_seconds"] = elapsed
    with pytest.raises(ValueError, match="elapsed"):
        module.decode_record(_case(), payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "another-case"),
        ("closed_included", True),
        ("closed_included", -1),
        ("closed_included", 2),
        ("open_interval_total", 100),
        ("open_interval_total", 1),
        ("open_interval_covered", 1),
        ("downgraded", 0),
        ("downgraded", True),
        ("production_acf_downgraded", True),
        ("raw_acf_flag", True),
        ("errors_by_family", [["thickness_period", [float("nan")]]]),
        ("errors_by_family", [["thickness_period", [float("inf")]]]),
        ("errors_by_family", [["thickness_period", [-0.1]]]),
        ("errors_by_family", [["thickness_period", [True]]]),
        ("errors_by_family", [["not-a-metric", [0.01]]]),
        ("errors_by_family", [["thickness_period", [0.01]], ["thickness_period", [0.01]]]),
        ("errors_by_family", [["thickness_period", []]]),
    ],
)
def test_record_rejects_invalid_metric_or_category_evidence(field, value, load_tool_module) -> None:
    module = load_tool_module("statistical_records")
    payload = _record()
    payload["outcome"][field] = value
    with pytest.raises(ValueError):
        module.decode_record(_case(), payload)


@pytest.mark.parametrize("mutation", ["record-extra", "outcome-extra", "outcome-missing", "not-object"])
def test_record_requires_exact_schema(mutation, load_tool_module) -> None:
    module = load_tool_module("statistical_records")
    payload = _record()
    if mutation == "record-extra":
        payload["ignored"] = "forbidden"
    elif mutation == "outcome-extra":
        payload["outcome"]["ignored"] = "forbidden"
    elif mutation == "outcome-missing":
        del payload["outcome"]["closed_included"]
    else:
        payload = []
    with pytest.raises(ValueError, match="schema"):
        module.decode_record(_case(), payload)


@pytest.mark.parametrize(
    "category,flags",
    [
        ("ambiguous", {"downgraded": True}),
        ("model_error", {"production_acf_downgraded": True, "raw_acf_flag": True}),
        ("mixed_kalpha_mono", {}),
    ],
)
def test_nonmetric_cases_preserve_only_their_original_evidence(category, flags, load_tool_module) -> None:
    module = load_tool_module("statistical_records")
    case = _case(category)
    outcome = _CaseOutcome(case.case_id, **flags)
    payload = json.loads(json.dumps(module.encode_record(outcome, 0.5)))

    assert module.decode_record(case, payload) == (outcome, 0.5)
    invalid = copy.deepcopy(payload)
    invalid["outcome"]["closed_included"] = 1
    invalid["outcome"]["errors_by_family"] = [["thickness_period", [0.01]]]
    with pytest.raises(ValueError, match="category"):
        module.decode_record(case, invalid)
