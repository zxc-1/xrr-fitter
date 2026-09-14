# 工作区三栏：从停靠面板迁到定宽分栏（B1 迁移设计与金标变更方案）

## 0. 这份文档解决什么

`deliverables/gui-redesign-缺失修补方案-v2.md` §12.5 对 B1 定案路线 1（锁死 dock 外观），
已交付的外壳走的是路线 2（改建为分栏）。§11.7 规定：改走路线 2 须先产出「架构迁移设计 +
金标测试变更方案」并单独获确认（依 AGENTS §3）。本文是该产物。**该确认已于 2026-09-03 由
用户给出：保留路线 2**（台账见 `plan-v2-acceptance-ledger.md` §5，扩门禁规模见其 §5.1）。
本文范围不因此扩大；§6 的回退步骤保留为备料。

结论先说：**门禁没有被放宽**。`tests/architecture/test_public_api.py` 的金标一行未改，
`api.set_dock_state` 仍导出且签名不变，`dock_state` 仍在 schema 里并且无损往返。路线 2 的
真实代价只有一条：GUI 不再消费 `dock_state`，它对 GUI 已是死 API。

## 1. 现状与设计稿的差（§5.5 的判定）

- 设计稿的壳是网格：`grid-template-columns:264px 1fr 340px`（CSS L104）。两侧是定值，只有
  画布是 `1fr`。
- 旧壳是 5 个 `QDockWidget`，`DOCK_FEATURES = Movable | Floatable | Closable`。
- §5.5 的判定原文：**不是缺三栏，是三栏可被用户拖散**。所以 B1 的可观测判据是
  「三栏不可拖拽 + 宽度固定」，不是「有三栏」。

## 2. 为什么落地成路线 2

路线 1 的成立前提是「这只是表现层改动」：`window_layout.py` 在 R23 基线上是 156 行，只做
停靠装配。重设计把栏内容搬进同一模块之后（导航栏 rail、画布列三段、检查器分段滚动），该
模块 791 行，其中绝大部分是设计稿要求的栏内结构。此时重新停靠不再是「去掉两个 feature」：

- `QDockWidget` 自带标题栏与关闭按钮，设计稿两者都不画（§11.1）。
- 右侧三段在停靠区里回到一次只看一段的 tab，设计稿要求「判定 / 参数 / 候选解 同屏可读」。

也就是说，路线 1 的低风险性来自它假设的那份 156 行模块，而那份模块已经不存在了；按 §12.5
的字面回退，会把 §11.1 要求的两件事重新退回去。这是超出 §12.5 定案的取舍，需要用户裁决；
若裁决为回退，工作量与步骤见 §6。

## 3. 迁移后的结构

| 元素 | 位置 | 说明 |
|---|---|---|
| `QSplitter`（水平） | `window_layout.py:771` `install_workspace` | `objectName="workspaceSplitter"`，`handleWidth(1)` 对齐设计稿 `1px solid var(--border)` |
| 三列 | 同上 | `navigationColumn` / `canvasColumn` / `inspectorColumn` |
| 定宽单一真源 | `window_layout.py:61-62` | `LEFT/RIGHT_COLUMN_WIDTH` 取自 `ProjectUiState().workspace_splitter_sizes`（`model/project.py:163` = `(264, 716, 340)`），不另写字面量 |
| 钉死用户那条路 | `window_layout.py:638` `_pin_columns` | 每道手柄 `setEnabled(False)` + 箭头指针；`setSizes` 仍有效 |
| 默认与复位 | `window_layout.py:658` `apply_default_columns` | 初装、首个 `showEvent`、`reset_layout` 三处共用同一套算术 |
| 只有画布吃余量 | `gui/workspace.py:157-174` `configure_splitters` | 中列 stretch=1，两侧=0，且三列均不可折叠 |

「不可拖拽」钉的是用户那条路，不是所有路：`setSizes` 照常工作，默认宽度、`reset_layout`、
从工程读回宽度都走它。Qt 不会把请求宽度压到控件自身最小宽度以下，所以栏内容超预算时会
悄悄多占——这正是各条 `fits_its_budget` 断言在量的东西，它们保持绿。

## 4. 金标测试变更方案：不需要变更

| 金标 | 是否需要改 | 依据 |
|---|---|---|
| `tests/architecture/test_public_api.py` 导出名单（`:102`） | 否 | `set_dock_state` 仍在 `api.__all__`（`api.py:262`） |
| 同上 · 签名金标（`:149`） | 否 | 仍是 `(project: 'XrrProject', state: 'str') -> 'XrrProject'`，`services/projects.py:209` 未改 |
| 同上 · 分组金标（`:193`） | 否 | `workspace` 组仍是 `("set_expert_mode", "set_workspace_state", "set_dock_state")` |
| `tests/gui/test_workspace.py`（R23 基线契约） | 否 | 它的往返断言跑在合成根 `_workspace_root()` 上，与真实窗口是否钉宽无关；`:120`/`:122` 保持绿 |

这一栏是 AGENTS §3「扩大门禁须先确认」真正针对的东西，本次没有发生：**没有删除或放宽任何
一条既有断言**。路线 2 新增的覆盖全部是加断言（`tests/gui/test_workspace_columns.py`，39 条）。

需要改的只有一条我自己写的用例：`tests/gui/plot_cases_3.py` 里
`test_persisted_column_and_plot_tab_changes_mark_project_dirty` 原先手工 `emit(splitterMoved)`
断言「拖出来的宽度会持久化」。钉死之后 Qt 不可能再发这个信号，那条断言只能靠手工 emit 成立，
等于自证。已改为断言捕获后工程里的列宽等于外壳当下的几何——真话，且仍覆盖捕获路径。

## 5. `dock_state`：不再消费，但无损保留

§11.4 把「dock_state 持久化往返」列为完成判据。路线 2 下这句要读作**数据往返**（存得住、
读得回、原样写回），不是布局恢复——没有停靠布局可恢复了。链路逐段有覆盖：

- schema 保留字段：`model/project.py:174` `dock_state: str = ""`，类型校验在 `:187-188`。
- codec 可选键：`project_codec.py:287` 写、`:306` 列入 `optional`、`:327` 读（缺键回默认空串），
  所以停靠版写的文件和本版写的文件互相都能开。
- 捕获不摧毁它：`gui/workspace.py:131-149` `capture_project` 用 `replace(current, …)` 逐字段
  替换，只动它拥有的五个字段，`dock_state` 原样带过。这是全链路唯一会重写 `ui_state` 的地方。

覆盖它的测试：`tests/unit/model/test_dock_state.py`（模型往返）、
`tests/unit/io/test_dock_state_codec.py`（编解码往返 + 缺键 + 参考工程）、
`tests/gui/test_workspace_columns.py` 两条（陈旧值不致打不开工程；捕获+落盘+读回后值不变）。

残留代价，如实记录：`api.set_dock_state` 对 GUI 已是死 API——只有测试和外部调用者会用它。
删它要动公共 API 金标，那才是扩门禁，因此不删。

## 6. 若裁决为退回路线 1（备料；2026-09-03 的裁决是不退）

不是「去掉两个 feature」那么小，需要按序做完：

1. 把栏内结构从 `window_layout.py` 抽成独立装配单元（导航栏、画布列三段、检查器分段滚动），
   否则重建停靠会把这些结构一起拆掉。
2. 重建 5 个 `QDockWidget`，`DOCK_FEATURES` 只留 `Closable`，用 `setFixedWidth` /
   `resizeDocks` 钉 264/340。
3. 恢复 `dock_state` 的存取（`saveState`/`restoreState` 两端），并接回捕获与恢复路径。
4. 与 §11.1 的冲突另行裁决：停靠区必然带回标题栏、关闭按钮，右侧三段必然回到 tab；设计稿
   这三样都不画。
5. 相应删除 `tests/gui/test_workspace_columns.py` 中「不是停靠区」类断言——**这一步是缩小
   门禁，须按 AGENTS §3 单独确认。**

## 7. B1 可观测判据（§11.7）现状

| 判据 | 覆盖 | 结果 |
|---|---|---|
| 三栏不可拖拽 | `test_the_side_columns_cannot_be_dragged_to_another_width` | 绿（先 RED：手柄可用 → 钉死后过） |
| 宽度固定 264 / 340 | `test_the_side_columns_open_at_their_documented_widths`、`test_only_the_canvas_absorbs_the_spare_width`、`test_reset_layout_returns_the_columns_to_their_budgets` | 绿 |
| 一像素缝仍是设计稿的边框线 | `test_the_seam_between_columns_is_one_pixel_of_border` | 绿（禁用手柄没有改变它的涂色） |
| `dock_state` 往返 | §5 四条 | 绿 |
| 公共 API 金标全程保持 | `tests/architecture/test_public_api.py` 未改动 | 绿 |

`tests/gui` 全量 1309 passed（钉死之后重跑）。
