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


def test_a_readiness_message_that_is_already_chinese_passes_through_verbatim() -> None:
    """已经是中文的消息照原样上屏——它不需要翻译，替它换一句分类结论只会丢掉信息。

    分英文与中文两档，靠的是「这句话里有没有汉字」：GUI 自己构造的 readiness（拟合面板的
    ``尚未检查拟合条件``）本来就是中文，而 API 报的诊断串一律是英文。
    """
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
    ("message", "fragment"),
    (
        ("current fit mask is not fit-ready", "范围"),
        ("no runnable automatic datasets", "自动"),
        ("automatic fit does not support cross-dataset constraints", "约束"),
    ),
)
def test_the_preflight_verdicts_the_status_bar_shows_are_all_localized(message: str, fragment: str) -> None:
    """这三句 preflight 都会走到底栏第一段，而那一段在设计稿五帧里一律是中文。

    第一句最常撞上：把拟合角度范围拖窄到掩码里剩不下 30 点，``fit/problem.py`` 就抛
    ``ValueError("current fit mask is not fit-ready")``，preflight 拿 ``str(error)`` 当消息，
    30px 底栏于是写出一句英文。译文比原文多说了一句「怎么办」——原文只说「没就绪」，而读者
    真正要知道的是把范围放回去。
    """
    messages = _messages()

    text = messages.readiness_text(message)

    assert fragment in text
    assert text != message
    # 译过之后屏上不该再留英文碎片：底栏那一段没有第二行可以放原文。
    assert not any("a" <= character.lower() <= "z" for character in text), text


def test_an_untranslated_english_verdict_does_not_reach_the_screen(caplog) -> None:
    """没登记过的诊断串照原样上屏就等于把开放集合的英文接到设计稿的位置上。

    ``preflight_fit`` 的兜底是 ``str(error)``——fit 编译栈里任意一个 ``ValueError`` 都能走到
    这里，逐条登记永远追不上。所以这一档报分类结论，原文交给日志：屏上守住那一段的语言与
    长度，专家要的原文仍取得回，而不是被静默丢掉。
    """
    import logging

    messages = _messages()

    with caplog.at_level(logging.DEBUG, logger="xrr_fitter.gui.messages"):
        text = messages.readiness_text("walkers must stay even")

    assert text == messages.UNKNOWN_READINESS_TEXT
    assert "walkers must stay even" in caplog.text


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
