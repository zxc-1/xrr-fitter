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
