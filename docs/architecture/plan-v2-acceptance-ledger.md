# 计划 v2 §11 七条完成判据 · 验收台账

## 0. 这份文档解决什么

`deliverables/gui-redesign-缺失修补方案-v2.md` §11 列了 7 条完成判据，此前散在各批的施工
记录里，没有一处把「现在是什么状态、用哪条命令复核」摆在一起。本文是那一处。

数字都是 2026-09-03 在工作树 `gui-reskin` 实测的。**七条判据现已全部结项**——最后一项（判据 7
的路线 2 确认）按 §11.7 与 AGENTS §3 归用户裁决，2026-09-03 的裁决是**保留路线 2**，见 §5。

判据本身没要求、但同日一并核完的一项在 §6：G1–G27 + B1 逐项点名「回归时哪条测试会红」。

## 1. 七条判据现状

| 判据 | 复核命令 | 结果 |
|---|---|---|
| 1 设计稿 6 帧可追溯 | `design-frames-traceability.md` §3 的自解析循环 | 33 条锚点 `OK`，`MISSING=0`；另补反向核查（设计稿→实现）见该文 §5：20 按钮 + 14 tab 全部落地，抓到并修掉 1 处文案偏离 |
| 2 `tests/gui` + `tests/architecture` 全量绿 | 见 §2 前两行 | 1311 + 194 passed |
| 3 ruff / hygiene / radon 绿 | §9 三条命令，见 §2 | 全绿（hygiene 只剩工作树固有项） |
| 4 dock_state 往返 / undo/redo / 引导-专家共享路径保持 | 见 §3 | 三项各有测试看着 |
| 5 每批 RED-GREEN 证据 | 各批会话记录；本次新增一条见 §3 | 有 |
| 6 Group 3 四项各出后端契约 + 指纹金标全程绿 | 见 §4 | 契约齐；金标 named run 绿 |
| 7 B1 三列不可拖拽 + 宽度固定；走路线 2 须单独获确认 | 见 §5 | 代码与测试就绪，**2026-09-03 获用户确认保留路线 2** |

## 2. 本次实测

```bash
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/architecture -q -p no:cacheprovider -p no:randomly   # 194 passed, 43s
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/gui -q -p no:cacheprovider -p no:randomly            # 1311 passed, 293s（G21 文案修补后重跑）
env -u PYTHONPATH QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit tests/integration -q -p no:cacheprovider -p no:randomly --import-mode=importlib   # 2260 passed, 160s
.venv/bin/ruff check --no-cache src/xrr_fitter/gui tools tests   # All checks passed!
.venv/bin/ruff format --check src/xrr_fitter/gui                 # 80 files already formatted
.venv/bin/python tools/check_radon.py                            # 退 0
.venv/bin/python tools/check_hygiene.py                          # 见下
```

`ruff` 记得带 `--no-cache`（或跑完 `rm -rf .ruff_cache`）：`.ruff_cache` 落在仓库里会让
`check_hygiene.py` 立刻非 0，和 `__pycache__` 一个道理。**`ruff format` 没有 `--no-cache`
这个开关**，所以只要跑过它就得手动 `rm -rf .ruff_cache`，否则紧接着的 hygiene 会多报 6 条
`generated: .ruff_cache/...` + 1 条 `ownership`。

两处与 §9 写法的差别，如实记录：

- **`tests/architecture` 不能按 §9 写的 `--import-mode=importlib` 跑。** `test_distribution.py`
  顶层 `from tools import …`（`tools/` 无 `__init__.py`），importlib 下 `ModuleNotFoundError`，
  少收 9 个用例还报 1 个 collection error。上面用默认 prepend，收得比 §9 那条更全。
- **`check_hygiene.py` 在工作树里不可能空手。** 它用 `rglob("*")` 扫磁盘，本树自带的 `.venv`
  （每树必须独立，否则 `.pth` 指回主树 src 假绿）整棵都会被报成 `generated`。滤掉 `.venv`
  之后只剩一条 `git-control: .git: .git must be a normal directory`——linked worktree 的 `.git`
  是文件而非目录，干净检出里两者都不存在。
- 另跑了两遍更宽的 ruff：本次改动过的 231 个 `.py`（含新增文件）`ruff check` 与
  `ruff format --check` 均干净；`ruff check .` 全仓亦退 0。也就是说 §9 那两条命令通过，
  不是靠把存量错误挡在范围外。

## 3. 判据 4 的三项分别由谁看着

- **dock_state 往返**：四条测试，列在 `workspace-columns-migration.md` §5。路线 2 下这句读作
  **数据往返**（存得住、读得回、原样写回），不是布局恢复。
- **undo/redo**：本次补的
  `tests/gui/test_shell_menus.py::test_the_edit_menu_still_undoes_and_redoes_a_project_change`。
  补它的理由是**此前全仓没有任何测试提到 `undoAction` / `redoAction` / `can_undo`**——
  `ProjectDocument` 的两个栈一直有覆盖，但连着它们的那根 `undo_state_changed` 连线没有。
  重设计整条重装了 `chrome.py` 的菜单栏，这根线掉了的表现是「撤销永远是灰的」，而那种坏法
  在别处一个断言都碰不到。RED 证据：把 `chrome.py:445` 的 `redo.setEnabled(can_redo)` 临时
  改成 `setEnabled(False)`，该用例即 `assert (False, False) == (False, True)` 失败；改回即绿。
- **引导-专家共享路径**：`test_expert_mode_toggle_updates_only_project_ui_state`、
  `test_expert_mode_flag_is_orthogonal_to_guidance_visibility`、
  `test_guidance_toggle_does_not_alter_pipeline_nav_step`、
  `test_command_bar_segment_leaves_the_persisted_expert_depth_alone`。

## 4. 判据 6 的契约都在计划正文里

| Group 3 项 | 契约落在哪 |
|---|---|
| G16 导出 `formats` | §12.1 |
| G9 参数三态 | §12.2（定案落 setting 侧） |
| G25 角度检测（连带 G18 切换） | §5.4 + §12.3（G18 需新字段，金标同步） |
| G10 暂停 + G11 跳过 | 计划 §5.1（**交付语义与它有偏离**，见 `design-frames-traceability.md` §2.2） |
| 降级进 Group 3 的 G12 / G27 | §5.6 / §5.7 + §12.6 |

指纹金标单独点名跑过：
`tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints`
连同两条 dock_state 往返用例共 9 passed。硬约束照守：**没有给任何 `fit/` 层 dataclass 加字段**。

## 5. 判据 7：路线确认已获（2026-09-03，保留路线 2）

B1 交付的是路线 2（改建 `QSplitter` + 钉死手柄），计划 §12.5 正文写的是「定案：走路线 1」。
§11.7 要求的两份产物已出：架构迁移设计 + 金标测试变更方案，都在
`workspace-columns-migration.md`（结论：`tests/architecture/test_public_api.py` 一行未改，
没有删除或放宽任何既有断言）。**2026-09-03 用户裁决：保留路线 2**——AGENTS §3 要的那次单独
确认即为此，本项据此结项。要签的扩门禁规模在 §5.1 量过。

这不是绕过 §12.5，走的是 §12.5 自带的重启条款：原文写「路线 2（换回 QSplitter）**仅在
「必须纯 splitter、彻底弃用 dock」成为硬需求时**才重启评估，届时另出架构迁移设计并按
AGENTS §3 单独确认」——两项前置（迁移设计、单独确认）现已齐备。硬需求的论证在迁移文档
§2/§3：路线 1 的低风险性建立在「`window_layout.py` 是一份 156 行的表现层模块」这个前提上，
而 R23 之后那份模块已不存在，照字面回退会把 §11.1 要求的两件事退回去。

计划正文仍与裁决不一致（还写着路线 1）。那份文件在主检出且未纳版本控制，本工作树改不到也
不代改，建议的收口写法见 §5.2。回退路线 1 的步骤仍留在迁移文档 §6，只作备料——裁决已定；
其第 5 步要删掉「不是停靠区」类断言，属缩小门禁，真要走还得按 AGENTS §3 再单独确认一次。

### 5.1 这次确认到底在签什么：锁定面量出来是 10 条 GUI 断言，架构门禁 0 条

「走路线 2 须单独获确认」的理由是扩门禁，那就得先说清扩了多大。把 B1 的 39 条用例按引用的
容器分类（`tests/gui/test_workspace_columns.py`）：

| 类别 | 条数 | 说明 |
|---|---|---|
| 引用顶层 `workspaceSplitter` | 10 | 路线 2 的真实锁定面；其中只有 `test_the_workspace_is_a_splitter_not_a_dock_area` 与 `test_the_three_columns_are_the_splitter_in_reading_order` 断言机制本身，其余 8 条量的是宽度预算 / 拖不动 / reset / 往返，换路线后意图仍成立、只需改取控件的方式 |
| 引用 `canvasSplitter` 或 `leftSplitter` | 3 | 画布内纵向分栏与导航栏内分栏，与 B1 走哪条路线无关 |
| 不涉及任何 splitter | 26 | 只量行为：栏序、可见性、宽度预算、持久化语义 |

架构层一条都没加：`tests/architecture/test_public_api.py` 里与此相关的只有既有的
`set_dock_state`（`:102/:149/:193`），迁移文档说的「一行未改」在这一侧复核成立。

两条路线对设计稿的还原是同一个结果（`grid-template-columns:264px 1fr 340px`，两侧定宽、
中间吃剩余、栏缝不可拖），所以「哪条更贴设计稿」不是区分点。机制差别：路线 1 不用 splitter，
宽度成为常量；路线 2 留着 splitter 但把手柄 `setEnabled(False)` 并把光标改回箭头
（`window_layout.py:638 _pin_columns`），`setSizes` 仍通，于是宽度还是可持久化的数据
（`ui_state.workspace_splitter_sizes`，由 `test_column_widths_round_trip_through_a_saved_project`
看着）。计划 §0.3 的「真正差异只剩『可拖拽 vs 固定不可拖拽』一点」在路线 2 下已经满足。

### 5.2 计划 §12.5 的收口写法（留给用户落笔）

改动量是标题一行 + 追加一段；§12.5 下面「施工基线」三条是路线 1 的改点，随之作废，不必删，
标一句即可。可直接粘的写法：

```markdown
### 12.5 B1 路线【定案：走路线 1 → 2026-09-03 改判路线 2】

**改判（2026-09-03）**：按本节自带的重启条款——「路线 2 仅在『必须纯 splitter、彻底弃用
dock』成为硬需求时才重启评估，届时另出架构迁移设计并按 AGENTS §3 单独确认」——两项前置均
已完成：迁移设计见 `docs/architecture/workspace-columns-migration.md`（含金标变更方案），
AGENTS §3 的单独确认由用户于 2026-09-03 给出，锁定面为 10 条 GUI 断言、架构门禁 0 条新增。
故 B1 交付路线 2：保留 `QSplitter` 并禁用全部手柄（`window_layout.py:_pin_columns`）。
下面「施工基线」三条属路线 1，随本次改判一并作废。`dock_state` 契约与 `api.set_dock_state`
金标未动，`dock_state` 对 GUI 成为死 API（迁移文档 §3）。
```


## 6. G1–G27 + B1 各由哪条测试看着

判据 1 证明每项在实现里有位置，判据 2 证明整套测试是绿的。这一节回答夹在两者中间的那个问题：
**某一项被改坏时，到底有没有测试会红。** §3 第二项那个 undo/redo 的洞就是从这个角度找出来的
——它当时在判据 1、2、3 下全绿，因为全仓没有任何断言碰到它。所以 28 项逐一点名，一次核完。

结论：**28 项各有至少一条宿主测试**，没有第二个 undo/redo 式的洞。这一节没有带出任何新的
施工项。下表未写目录的文件都在 `tests/gui/` 下。

| 项 | 宿主测试（该项回归时会红的那条） |
|---|---|
| G1 引导/专家 segment | `test_shell_menus.py::test_command_bar_offers_the_designs_projection_segment`（另有 `::test_command_bar_segment_switches_the_visible_projection`、`::test_command_bar_segment_and_the_view_menu_cannot_disagree`） |
| G2 独立/联合 segment | `test_batch_segment.py::test_the_command_bar_carries_the_designs_batch_segment`、`::test_guided_mode_puts_the_batch_segment_away` |
| G3 明暗切换 | `test_shell_chrome.py::test_the_appearance_toggle_sits_at_the_far_right_and_flips_the_palette` |
| G4 画布 modebar | `test_visual_contracts.py::test_plot_modebar_shares_one_row_with_the_view_tabs`、`::test_no_two_modebar_controls_wear_the_same_glyph` |
| G5 状态栏模式文字 | `test_design_fidelity.py::test_the_status_bar_states_which_guided_step_is_open`（:198 断言 `fitReadinessStatus` 文字 == 「引导模式」） |
| G6 层堆叠只读预览 | `test_design_fidelity.py::test_each_guided_row_carries_the_colour_of_its_layer`、`test_sld_regions.py::test_only_the_real_layers_take_a_colour` |
| G7 引导卡双 CTA | `test_guidance.py::test_the_guided_actions_sit_in_one_row_at_the_bodys_left_edge` |
| G8 层列表拖拽重排 | `test_structure_editor.py::test_dragging_a_stack_row_reorders_the_structure`、`::test_dropping_a_row_onto_the_bounding_media_leaves_the_order_alone` |
| G9 参数三态 | `tests/unit/services/test_parameters.py::test_a_range_only_setting_grows_a_soft_range_prior_clamped_to_the_declaration`、`test_parameter_disposition.py::test_the_setting_gear_wins_over_the_declaration` |
| G10 暂停 | `tests/unit/services/test_workers.py::test_the_pause_probe_holds_the_worker_until_it_is_resumed`、`test_fit_progress.py::test_the_pause_control_sits_before_stop_and_toggles_to_resume` |
| G11 跳过本阶段 | `tests/unit/fit/test_resume.py::test_a_skip_request_ends_the_current_stage_and_moves_to_the_next`、`test_fit_progress.py::test_the_skip_control_asks_the_worker_to_drop_the_current_stage` |
| G12 各数据集目标值表 | `tests/unit/fit/test_joint_progress.py::test_joint_progress_carries_the_per_dataset_objective_breakdown`、`test_live_metrics.py::test_a_joint_run_splits_the_objective_when_the_backend_reports_the_breakdown` |
| G13 联合模式横幅 | `test_fit_progress.py::test_progress_view_shows_joint_layout_banner`、`::test_the_joint_banner_says_which_side_of_the_split_a_number_is_on` |
| G14 MCMC 后验直方图 | `test_uncertainty_dialog.py::test_the_posterior_page_gives_every_parameter_its_own_panel`、`::test_the_posterior_page_says_so_when_no_sampling_was_run` |
| G15 Δχ²=1 参考线 | `test_uncertainty_dialog.py::test_the_profile_page_marks_where_the_interval_closed`、`::test_the_profile_page_omits_the_reference_line_without_a_threshold` |
| G16 导出多格式 | `tests/unit/services/test_exports.py::test_default_publication_is_byte_identical_when_optional_formats_are_added` |
| G17 manifest 预览 | `test_export_dialog.py::test_export_option_dialog_manifest_preview_describes_run` |
| G18 角度约定切换 | `test_import_dialog.py::test_choosing_the_incident_angle_convention_relabels_the_preview`、`test_data_import.py::test_the_chosen_angle_convention_reaches_the_imported_datasets` |
| G19 批量导入预览表 | `test_visual_m7_dialogs.py::test_import_dialog_previews_every_file_with_its_row_count`、`::test_import_dialog_keeps_a_failed_file_visible_and_marked` |
| G20 六段管线 stepper | `test_pipeline_nav.py::test_the_stepper_names_the_six_canonical_stages`（该文件另有 8 条） |
| G21 不确定度四子页 | `test_uncertainty_dialog.py::test_uncertainty_view_carries_the_four_evidence_pages`、`::test_each_page_draws_the_owned_evidence` |
| G22 不确定度子管线 | `test_uncertainty_methods.py::test_the_four_methods_keep_the_order_the_design_walks_them_in`（同文件其余 7 条共用同一助手） |
| G23 帧④ 三 tab | `test_canvas_top.py::test_the_running_canvas_opens_with_the_three_tabs_frame_four_names` |
| G24 导入高级选项 | `test_design_fidelity.py::test_the_advanced_box_states_the_header_skip_and_the_log_option_it_does_not_have` |
| G25 角度自动检测 API | `tests/unit/services/test_datasets.py::test_detecting_the_angle_convention_forwards_the_readers_verdict` |
| G26 结构相关性提示 | `test_design_fidelity.py::test_the_inspector_warns_that_thickness_and_density_can_trade_off` |
| G27 四项实时指标 | `test_live_metrics.py::test_the_solver_telemetry_rows_read_a_dash_until_the_solver_publishes_them` |
| B1 三栏定宽不可拖拽 | `test_workspace_columns.py::test_the_workspace_is_a_splitter_not_a_dock_area`、`::test_the_three_columns_are_the_splitter_in_reading_order`（该文件共 39 条） |

### 6.1 别拿判据 1 的锚点去 grep `tests/`

`design-frames-traceability.md` §1 那 33 条锚点是**实现侧**符号，多半是私有的
（`_install_mode_segment` / `_build_secondary` / `_pin_columns` / `class UncertaintyPages`）。
测试钉的是行为，走 `objectName` 与公开 API，所以拿锚点扫 `tests/` 会得到 14 项 0 命中
（G1、G2、G5、G7、G10、G11、G14、G17、G18、G20、G21、G23、G24、G27、B1）——**全部是私有符号
造成的假空洞，不是覆盖缺口**。上表是换成测试侧的名字重扫得到的。

三个具体的坑，记下来省得下次再踩：

- **G4 的锚点 `MODE_SPECS` 在 `tests/` 里撞名。** 它在测试里只命中 chrome 的
  `WORKSPACE_MODE_SPECS`（那是 G1 的 segment 规格表，`test_expert_views.py:99/110/122`），
  一次都没命中 `plots/interactions.py` 自己的 `MODE_SPECS`。G4 的测试侧身份是上表那两条
  modebar 用例。§3 那条自解析循环只扫 `src/`，不受影响，照样报 `OK`。
- **G18 的 `twoTheta` 同样撞名。** `test_import_dialog.py` 里的 `twoTheta` 只匹配既有的
  `twoThetaColumnEditor`（列序号编辑器），与角度约定那两枚单选无关。
- **G8 的关键断言不在用例体里。** `tree.dragDropMode() == InternalMove` 写在共享助手
  `test_structure_editor.py:331 _drop_row` 的开头（注释交代了理由：树没宣告可拖放时 Qt 基类的
  `dropEvent` 会直接段错误）。按「最近的上一个 `def test_`」回溯会把它错记到隔壁用例上，真正
  跑到它的是上表那两条。


## 7. 判据之外补做的一项：`bootstrap_sample_count`（2026-09-04）

不在 G1–G27 + B1 那 28 项里，也不是任何判据要求的。来源是判据 1 的反向核查在
`design-frames-traceability.md` §5 记下的那条已知数据限制——设计稿帧⑤ 的
`重采样次数 200`（HTML L967-968）与左栏 `200 次 · 失败 2%`（L902）读不出来，因为
`UncertaintyReport` 上没有这个字段。用户批准了补它（AGENTS §3 的那次确认针对的是**工程文件
持久化 schema 变更**：`uncertainty` 段多一个 optional 键）。

字段落点与设计取舍写在 `design-frames-traceability.md` §5.1，这里只记验收面：

| 面 | 结果 |
|---|---|
| RED | 5 个宿主 12 条断言红：unit 侧 8 条全部 `AttributeError: 'UncertaintyReport' object has no attribute 'bootstrap_sample_count'`，GUI 侧 4 条读数不符 |
| GREEN | 同两条命令 64 passed / 47 passed |
| 全量 | `tests/unit`+`tests/integration` 2268 passed（160s）、`tests/gui` 1315 passed（269s）、`tests/architecture` 194 passed（39s） |
| 门禁 | 触及的 10 个文件 `ruff check --no-cache` + `ruff format --check` 干净；`tools/check_radon.py` 退 0；hygiene 只剩工作树固有那一条 |
| 指纹 | `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 单点复跑 1 passed |

**为什么加字段没动指纹**：`bootstrap_sample_count` 加在 `model/analysis.py` 的
`UncertaintyReport` 上，那不是 `fit/` 层 dataclass，不进 `FitCheckpoint` 的指纹口径；也没有加
到 `BootstrapResult` 上——那会移动 `bootstrap_provenance_sha256`。工程文件侧靠「只在非 0 时写
键」保住旧文件重新编码逐位不变，`test_result_without_a_resample_count_omits_key` 与
`test_result_without_a_resample_count_key_still_decodes` 两条正是钉这一点的。

§1 那张表的判据 2 / 判据 3 数字随之上浮（1311→1315、2260→2268），复核命令未变。
