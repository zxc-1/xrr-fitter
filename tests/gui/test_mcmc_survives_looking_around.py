"""一条 MCMC 链属于它开跑时的科学输入，不属于读者此刻翻到的那一页。

审计实测：链跑着的时候切一次 tab 或换一个数据集，完成时整条链被当作「项目已改变」丢弃。
过期判据此前比的是项目对象的身份，而翻页本身就会换掉那个对象。这里钉住：只看了别处时链照常
落到它自己的数据集，读者停着的页与数据集保持不动；真正改了内容（见 ``test_results`` 里换候选
那一例）仍然拒收。
"""

from __future__ import annotations

from dataclasses import replace

from tests.gui.test_results import _mcmc_report, _two_candidate_result
from tests.support.model_cases import dataset_project

import xrr_fitter.api as api


def _project(*dataset_ids: str) -> api.XrrProject:
    datasets = tuple(dataset_project(name, result=_two_candidate_result()) for name in dataset_ids)
    value = replace(api.XrrProject.new(datasets, master_seed=1201), base_directory="/private/tmp")
    value = api.select_active_dataset(value, dataset_ids[0])
    return api.set_expert_mode(value, True)


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _with_chain(project: api.XrrProject, dataset_id: str) -> api.XrrProject:
    """把一条链挂到 ``dataset_id`` 的结果上，相当于 worker 交回来的项目。"""

    def attach(dataset):
        if dataset.dataset_id != dataset_id:
            return dataset
        result = dataset.last_valid_result
        uncertainty = replace(result.uncertainty, mcmc=_mcmc_report())
        return replace(dataset, last_valid_result=replace(result, uncertainty=uncertainty))

    return replace(project, datasets=tuple(attach(dataset) for dataset in project.datasets))


def _started(window, monkeypatch):
    monkeypatch.setattr(window.result_panel.controller, "start_mcmc", lambda *_args: True)
    failures: list[api.OperationError] = []
    window.result_panel.operation_failed.connect(failures.append)
    source = window.document.project
    assert window.result_panel.start_mcmc() is True
    return source, failures


def _chain(window, dataset_id: str):
    dataset = next(item for item in window.document.project.datasets if item.dataset_id == dataset_id)
    return dataset.last_valid_result.uncertainty.mcmc


def test_a_chain_survives_a_page_change(qtbot, monkeypatch) -> None:
    window = _window(qtbot, _project("curve"))
    source, failures = _started(window, monkeypatch)

    window.plot_panel.select_view("raw")
    window.result_panel.controller.mcmc_finished.emit(_with_chain(source, "curve"))

    assert failures == []
    assert _chain(window, "curve") is not None
    assert window.document.project.ui_state.plot_tab_index == 1
    assert window.plot_panel.current_view_key() == "raw"


def test_a_chain_lands_on_its_own_dataset_after_the_reader_moved_on(qtbot, monkeypatch) -> None:
    window = _window(qtbot, _project("curve", "other"))
    source, failures = _started(window, monkeypatch)

    window.select_active_dataset("other")
    window.result_panel.controller.mcmc_finished.emit(_with_chain(source, "curve"))

    assert failures == []
    assert _chain(window, "curve") is not None
    assert _chain(window, "other") is None
    assert window.document.active_dataset_id == "other"
