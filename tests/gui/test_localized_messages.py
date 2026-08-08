"""Chinese localization of public API status and error messages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _messages():
    from xrr_fitter.gui import messages

    return messages


def test_ready_message_is_localized() -> None:
    messages = _messages()

    text = messages.readiness_text("ready")

    assert "就绪" in text
    assert "ready" not in text


def test_empty_project_message_guides_import() -> None:
    messages = _messages()

    text = messages.readiness_text("project has no datasets")

    assert "导入" in text
    assert "datasets" not in text


def test_missing_structure_message_keeps_dataset_identity() -> None:
    messages = _messages()

    text = messages.readiness_text("dataset curve-7 has no structure")

    assert "curve-7" in text
    assert "结构" in text
    assert "has no structure" not in text


def test_unknown_readiness_message_passes_through_verbatim() -> None:
    messages = _messages()

    assert messages.readiness_text("结构尚未准备") == "结构尚未准备"


def test_missing_measurement_preset_message_is_localized() -> None:
    """Standard mode surfaces this preflight verdict, so it must be translated.

    An untranslated string reaches the fit panel whenever a project has data but
    no measurement preset, which is the normal state right after a manual
    import.
    """
    messages = _messages()

    text = messages.readiness_text("automatic fit requires a measurement preset")

    assert "预设" in text
    assert "measurement preset" not in text


@pytest.mark.parametrize(
    ("exception_type", "fragment"),
    (
        ("RuntimeError", "运行"),
        ("ValueError", "无效"),
        ("OSError", "文件"),
        ("UnexpectedError", "失败"),
    ),
)
def test_operation_error_text_replaces_exception_type_with_chinese(
    exception_type: str,
    fragment: str,
) -> None:
    messages = _messages()
    error = SimpleNamespace(
        exception_type=exception_type,
        message="walkers must stay even",
        detail="",
    )

    text = messages.operation_error_text(error)

    assert "walkers must stay even" in text
    assert exception_type not in text
    assert fragment in text


def test_ready_state_text_matches_main_window_contract() -> None:
    messages = _messages()

    assert messages.readiness_text("ready") == messages.READY_TEXT


def test_operation_error_text_appends_recovery_advice_for_known_types() -> None:
    messages = _messages()
    error = SimpleNamespace(
        exception_type="OSError",
        message="source file vanished",
        detail="",
    )

    text = messages.operation_error_text(error)

    # A known failure type carries a concrete next step, not just a diagnosis.
    assert "source file vanished" in text
    assert "建议：" in text
    assert "重新链接数据源" in text


def test_operation_error_text_omits_advice_for_unknown_types() -> None:
    messages = _messages()
    error = SimpleNamespace(
        exception_type="RuntimeError",
        message="unexpected worker crash",
        detail="",
    )

    text = messages.operation_error_text(error)

    # No invented advice for types without a documented recovery move.
    assert "unexpected worker crash" in text
    assert "建议：" not in text
