"""Structure step status bar: stack scale, selected layer, unsaved edits."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _span_text(window, name: str) -> str:
    """一段状态栏读出来是什么样：说明文字与取值按排版顺序连起来。"""
    from xrr_fitter.gui.status_bar import StatusSpan

    span = window.findChild(StatusSpan, name)
    assert span is not None, name
    layout = span.layout()
    pieces = []
    for index in range(layout.count()):
        label = layout.itemAt(index).widget()
        if isinstance(label, QLabel):
            pieces.append(label.text())
    return "".join(pieces)


def _window_with_structure(qtbot, tmp_path):
    """A window whose active dataset already carries a structure."""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "r.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (
            api.LayerSpec("SiO₂", api.MaterialSpec("SiO2", "SiO2", 2.20), 3.1, roughness_a=1.5),
            api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=1.5),
            api.LayerSpec("Ni", api.MaterialSpec("Ni", "Ni", 8.908), 80.0, roughness_a=5.0),
        ),
        api.MaterialSpec("Si", "Si", 2.329),
        backing_roughness_a=3.0,
    )
    project = api.set_structure(project, "r", structure)
    project = api.set_expert_mode(project, True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    return window


def _structure_status_labels(window):
    """结构步骤状态栏的那几个标签，按规格统一拿出来。"""
    scale = window.findChild(QLabel, "structureScaleStatus")
    selected = window.findChild(QLabel, "selectedLayerStatus")
    return {"scale": scale, "selected": selected}


def test_the_structure_step_reports_the_stack_scale_and_the_selected_layer(qtbot, tmp_path) -> None:
    """设计稿帧③ 的状态栏读的是这一步的事：「4 层 · 9 个自由参数」与「选中：a-Si」。

    状态栏在设计稿里逐帧换内容——帧① 报判定与 J，帧③ 报这一叠层有多大、手上选的是哪一
    层。此前它是固定的那几段，于是站在结构这一步上，屏幕最下面那一行讲的是一件与手上动作
    无关的事（拟合判定），而「选了哪一层」除了右栏抬头之外无处可读。

    选中那一段只报名字的前半截：设计稿写「选中：a-Si」，不是把「· 非晶硅薄膜」也拖进来——
    状态栏一行里同时还站着层数与下一步，说明性的后半截在这里只会挤掉别的。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    labels = _structure_status_labels(window)
    free = sum(1 for item in window.parameters_panel.definitions if not item.locked)

    assert labels["scale"].isVisibleTo(window) is True
    assert labels["scale"].text() == f"5 层 · {free} 个自由参数"

    window.structure_panel.editor.stack.component_selected.emit(0)
    qtbot.wait(1)
    assert _span_text(window, "statusSelectedSpan") == "选中：SiO₂"

    apply_step_scope(window, 4)
    assert labels["scale"].isVisibleTo(window) is False
    assert labels["selected"].isVisibleTo(window) is False


def test_the_structure_step_puts_the_next_action_where_the_verdict_would_be(qtbot, tmp_path) -> None:
    """帧③ 的状态栏右半边是「选中：a-Si」与「下一步：开始拟合」，没有判定、批量与数据集。

    这一步还没有判定可报——那三段留在屏幕上，读者会拿上一次运行的结论去读手上这一叠还没
    拟过的层。腾出来的位置交给「下一步」：站在结构这一步，最该知道的是改完往哪走。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)

    following = window.findChild(QLabel, "nextActionStatus")
    assert following is not None
    assert following.isVisibleTo(window) is True
    assert _span_text(window, "statusNextSpan") == "下一步：开始拟合"
    for name in ("fitQualityStatus", "fitQualityDot", "batchModeStatus", "activeDatasetStatus"):
        label = window.findChild(QLabel, name)
        assert label is not None, name
        assert label.isVisibleTo(window) is False, name

    apply_step_scope(window, 4)
    assert following.isVisibleTo(window) is False
    assert window.findChild(QLabel, "activeDatasetStatus").isVisibleTo(window) is True


def test_the_next_action_is_the_one_coloured_word_in_the_bar(qtbot, tmp_path) -> None:
    """设计稿帧③ 把「开始拟合」写成 ``<b style="color:var(--accent)">``（HTML 733 行）。

    底栏这一行整条是次要色的白文，唯一带色的就是这个词——它是这一屏唯一的出口。字重已经在
    ``add_text(bold=True)`` 里，缺的是色：``theme`` 里那条 ``QLabel[statusKind="accent"]``
    早就为这一处写着（连注释都点着「下一步：开始拟合」），只是没人给标签挂上这个属性。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)

    following = window.findChild(QLabel, "nextActionStatus")
    assert following.property("statusKind") == "accent"
    caption = window.findChild(QLabel, "nextActionCaption")
    assert caption.property("statusKind") in (None, "")


def _edit_the_stack(window) -> None:
    """把手上这叠层改一下——走的是读者拖 SLD 或改数字之后同一条路。"""
    from dataclasses import replace

    structure = window.document.project.datasets[0].structure
    edited = replace(structure, backing_roughness_a=structure.backing_roughness_a + 0.7)
    assert window.structure_panel.set_structure(edited) is True


def test_an_unsaved_stack_edit_is_what_the_bar_reports_on_the_structure_step(qtbot, tmp_path) -> None:
    """设计稿帧③ 底栏第一段是 warn 圆点 +「结构已修改（未保存）」（HTML 729 行）。

    这一屏读者刚动过层——底栏第一段该说的就是这件事，而它此前照样念「就绪」。「就绪」在这
    里不算错（拟合前提确实齐了），但它答的是另一个问题，于是「改过、还没落盘」这件事在整
    个窗口里除了标题栏那个圆点之外无处可读，而标题栏的圆点对任何改动都亮，不单指结构。

    判据是两件事的与：结构确实被提交过一次编辑，且这次编辑还没存。只看 ``is_dirty`` 太松
    ——切换活动数据集也会置脏；只看「站在结构这一步」是撒谎，读者可以什么都没改就走过来。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    dot = window.findChild(QLabel, "fitReadinessDot")
    assert label.text() == messages.READY_TEXT

    _edit_the_stack(window)

    assert label.text() == "结构已修改（未保存）"
    assert dot.property("statusKind") == "warn"


def test_accepting_the_suggested_oxide_counts_as_an_unsaved_stack_edit(qtbot, tmp_path) -> None:
    """设计稿帧③ 那一屏正是「点了建议氧化层、接受了、还没存」——底栏得认这一次编辑。

    接受建议往层堆叠里插进一整层，四行里的 SiO₂ 就是它。它却走的是自己那条提交路（``api.
    accept_oxide_suggestion`` 直接落盘到文档），不经过 ``set_structure``；于是「改过结构」这
    件事在这条路上没人宣布，底栏照念「就绪」。用户视角里两条路没有分别：都是层堆叠多了/
    少了东西。所以宣布口必须一致——谁把新结构落了盘，谁就得说一声。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.window_layout import apply_step_scope

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    project = api.set_structure(
        project,
        "curve",
        api.StructureSpec(
            api.MaterialSpec("Air", None, None, 0.0j),
            (api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),),
            api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
        ),
    )
    window = MainWindow(ProjectDocument(api.select_active_dataset(project, "curve")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    dot = window.findChild(QLabel, "fitReadinessDot")
    assert label.text() == messages.READY_TEXT
    assert window.structure_panel.current_oxide_suggestion() is not None

    window.structure_panel.accept_current_oxide()

    assert len(window.document.project.datasets[0].structure.components) == 2
    assert label.text() == "结构已修改（未保存）"
    assert dot.property("statusKind") == "warn"


def test_saving_the_project_takes_the_unsaved_stack_note_back_down(qtbot, tmp_path) -> None:
    """存过之后这句话必须消失，否则它报的是「改过」而不是「没存」。

    锁存器不能只加不减：存盘把项目落到磁盘上，「未保存」当场不成立；此后再置脏（比如换活
    动数据集）也不该把这句话叫回来——它说的是结构，不是任何改动。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    _edit_the_stack(window)
    assert label.text() == "结构已修改（未保存）"

    window.document.save(tmp_path / "saved.xrrproj.json")

    assert label.text() == messages.READY_TEXT
    window.document.mark_dirty()
    assert label.text() == messages.READY_TEXT


def test_a_live_run_outranks_the_unsaved_stack_note(qtbot, tmp_path) -> None:
    """在跑的时候底栏第一段是「拟合进行中」——设计稿帧④ 如此，而且它更紧要。

    两句话都想占第一段：正在跑是此刻唯一会自己变化的事，改没改存没存等它跑完再说也不迟。
    跑完之后那句话得自己回来，锁存器不因为跑过一次就作废。
    """
    from xrr_fitter.gui.operation_state import refresh_operation_state
    from xrr_fitter.gui.window_layout import apply_step_scope

    class _LiveJob:
        def cancel(self) -> None:
            return None

        def force_stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    _edit_the_stack(window)

    window.fit_panel.controller._job = _LiveJob()
    refresh_operation_state(window)
    assert label.text() == "拟合进行中"

    window.fit_panel.controller._job = None
    refresh_operation_state(window)
    assert label.text() == "结构已修改（未保存）"
