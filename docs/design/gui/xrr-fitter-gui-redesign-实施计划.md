# XRR-Fitter 全新 GUI · 可落地实施计划

> 历史设计资料（2026-08 至 2026-09），于 2026-09-15 整理入库。正文中的计划、状态和命令保留原编制时点语境，不是当前待办。当前完成情况见[设计追溯](../../architecture/design-frames-traceability.md)和[验收台账](../../architecture/plan-v2-acceptance-ledger.md)。

> 配套设计稿：`docs/design/gui/xrr-fitter-gui-redesign.html`（9 段预览：主工作台 / 引导四步 / 结构编辑+SLD / 九阶段进度 / 不确定度 / 导入导出 / 设计系统 / 能力覆盖）
> 目标读者：直接照此施工的工程师。每个里程碑都给出**文件级改动 + RED→GREEN 测试 + 验证命令**。
> 编制日期：2026-08-23

---

## 0. 一句话结论

**这不是推倒重建，而是表现层重塑（presentation-layer reskin）。** 既有 GUI（`src/xrr_fitter/gui/` 下 58 个文件）已经是一套成熟、近乎完整的实现，覆盖设计稿几乎全部能力。落地动作是：**在保留 `ProjectDocument` + `api` 边界 + 后端 + QDockWidget 机制 + `dock_state` 持久化契约不动的前提下，把「5-dock 停靠」视觉重塑为设计稿的「三区管线导航」，并把主题令牌 / 置信度双编码 / 图表调色板统一到设计稿规范。**

核心工作量集中在 3 个文件与 1 个新模块：
- `gui/window_layout.py`（157 行，**唯一骨架装配点**）——重排布局
- `gui/theme.py`（398 行，已含明暗双令牌）——合入设计令牌
- `gui/plots/` 子树——图表调色板与 SLD 可信带样式
- **新增** `gui/navigation/`（管线导航区，仅依赖 `gui` + `api`）

---

## 1. 现状核实（落地依据，均已读码验证）

### 1.1 既有 GUI 已覆盖设计稿全部 16 项能力

| # | 设计稿能力 | 既有实现位置 | 状态 |
|---|---|---|---|
| 1 | 导入与源校验 | `gui/data/panel.py` + `data/import_dialog.py` | ✅ 已实现 |
| 2 | 样品结构编辑 | `gui/structure/panel.py`+`editor.py`+`stack.py`+`dialogs.py` | ✅ |
| 3 | 参数化（先验/约束/共享） | `gui/parameters/panel.py`+`constraints.py`+`sharing.py`+`dialogs.py` | ✅ |
| 4 | 一键自动拟合 | `gui/fitting/panel.py`+`controller.py` | ✅ |
| 5 | 批量拟合（独立/联合） | `api.set_batch_mode`/`import_dataset_batch`/`describe_joint_layout` | ✅ 后端就绪 |
| 6 | 九阶段进度 0–1000 | `gui/fitting/progress.py`（`STAGE_WEIGHTS`/`STAGE_LABELS`） | ✅ **完全吻合** |
| 7 | 候选解与切换 | `gui/results/candidates.py` + `api.select_candidate` | ✅ |
| 8 | 置信度判定 | `model/analysis.py:ConfidenceClass` + `gui/chrome.py` 映射 | ✅ |
| 9 | 反射率/残差/qz⁴ | `gui/plots/live.py`（`LiveReflectivityPlot`, LIVE_PANE_KEYS） | ✅ |
| 10 | SLD 剖面与可信带 | `gui/plots/sld.py`+`sld_state.py`+`sld_drag.py` | ✅ |
| 11 | 相关矩阵 | `gui/results/uncertainty.py` + `gui/plots/diagnostics.py` | ✅ |
| 12 | Profile 似然 | `api.UncertaintyReport`（analysis 内部计算） | ✅ 后端就绪 |
| 13 | Bootstrap | `progress.py` bootstrap 阶段 + `UncertaintyReport` | ✅ |
| 14 | MCMC 后验 | `api.run_mcmc`/`start_mcmc_job`/`McmcReport` + `uncertainty.py:McmcControls` | ✅ |
| 15 | 导出 ORSO/Excel/图/清单 | `gui/export/dialog.py` + `api.export_result`/`ExportManifest` | ✅ |
| 16 | 引导↔专家双模式 | `gui/guidance/panel.py` + `MainWindow.set_guidance_visible` + `api.set_expert_mode` | ✅ |

**推论：没有一项能力需要新写后端。** 所有改动都在表现层。

### 1.2 三条硬约束（越过即门禁红）

**约束 A — 依赖边界（`tests/architecture/test_dependency_rules.py`）**
```python
ALLOWED["gui"] = {"gui", "api"}          # GUI 只能 import gui.* 和 api.*
THIRD_PARTY_OWNER_ALLOWLIST["PySide6"] = {"gui"}
THIRD_PARTY_PREFIX_ALLOWLIST = {
    "numpy":      ("gui.plots.",),        # numpy/matplotlib 只能在 gui.plots.* 子树
    "matplotlib": ("gui.plots.",),
}
PACKAGE_EDGE_EXCEPTIONS = {"gui.data.import_dialog": {"io.xy"}}  # 已批准例外
```
- 新增的任何布局/导航代码**只能** `import xrr_fitter.api as api` 与同层 `gui.*`，**禁止**触碰 `model`/`services`/`io`/`fit`/`analysis`。
- 任何要用 `numpy`/`matplotlib` 画图的新代码**必须落在 `gui.plots.*` 子树**，否则 `test_dependency_rules` 立即红。
- **重要修正**：`import_dialog.py:32` 的 `from xrr_fitter.io.xy import read_xy` **不是越界 bug，而是架构测试里显式登记的批准例外**。改造时**保持不动**，不要"顺手修复"它。
- 禁止：动态导入（`__import__`/`importlib`/`exec`/`eval`/`getattr` 导入）、`import *`、多模块循环依赖（全量扫描 + 强连通分量检测，见 `test_all_production_modules_pass_exhaustive_rules_and_have_no_cycles`）。

**约束 B — dock 布局持久化契约（`tests/gui/test_dock_layout.py`）**
既有 8 个测试深度耦合 QDockWidget，且 `dock_state` 走 `ui_state.dock_state`（base64 `saveState()`）经 `api.set_dock_state` 往返存盘：
```
MainWindow.capture_dock_layout() → project.ui_state.dock_state → api.save_project
  → 重开 → restoreState → dock 可见性一致
MainWindow.reset_dock_layout()   → 恢复 _default_dock_state
坏 base64 → 回退默认布局（不得让项目打不开）
```
→ **保留 QDockWidget 机制**，这套契约与测试几乎零改动；若改 QSplitter 三栏则全部作废（详见 §2.2）。

**约束 C — checkpoint 指纹漂移（记忆 `xrr-checkpoint-fingerprint-drift`）**
给进入 checkpoint 的 dataclass 加字段（即便有默认值）会让指纹漂移、旧 checkpoint 无法 resume，只有 `test_frozen_stage_search` 抓得到。
→ 纯表现层改造**不碰 `fit/` 层任何 dataclass**，天然规避；执行时不得为 UI 需要往 fit 层塞字段。

### 1.3 设计稿 API 符号 → 真实 `api.py` 符号（修正表）

设计稿 `#cover` 引用了若干**不存在于 `api.py`（已核实 100 个导出符号）** 的臆造名。落地时**一律以真名为准**：

| 设计稿臆造名 | 真实 `api.py` 符号 | 功能位置 |
|---|---|---|
| `detect_angle_convention` | `inspect_sources` / `preview_source_update` / `accept_source_update` / `SourceUpdatePreview` | 源校验流程 |
| `set_prior` | `set_parameter_priors` / `validate_parameter_priors` / `ParameterPrior` / `PriorSpec` / `ScalePriorState` | 参数先验 |
| `set_constraint` | `set_constraint_rules` / `validate_constraint_rules` / `ConstraintNode` / `ConstraintRule` | 约束 |
| `set_sharing` | `set_sharing_rules` / `validate_sharing_rules` / `SharingRule` | 参数共享 |
| `fit_batch` | `set_batch_mode` / `import_dataset_batch` / `preview_import_batch` / `describe_joint_layout` / `JointFitLayout` / `DatasetProject` | 批量/联合 |
| `profile_likelihood` | `UncertaintyReport`（analysis 内部算 profile，经此暴露） | 置信区间 |
| `run_manifest` | `ExportManifest` / `export_result` | 导出清单 |
| `FitStage` | `FitProgress` + `gui/fitting/progress.py:STAGE_LABELS` | 阶段进度 |

设计稿中**真实存在**的符号（可直接照抄）：`import_data`/`inspect_sources`/`set_structure`/`suggest_oxide_layers`/`describe_parameters`/`set_expert_mode`/`fit_automatically`/`start_fit_job`/`start_automatic_fit_job`/`start_mcmc_job`/`summarize_automatic_results`/`select_candidate`/`ConfidenceClass`/`run_mcmc`/`sld_nominal_profile`/`sld_uncertainty_bands`/`select_active_dataset`/`FitResult`/`XrrProject`/`OperationEvent`/`set_dock_state`/`set_workspace_state`/`set_fit_mask`。

---

## 2. 落地策略

### 2.1 不动的部分（保留清单 — 施工红线）

- **状态核心**：`ProjectDocument(QObject)` + 单一不可变 `api.XrrProject` 快照 + `replace_project` 原子提交 + undo/redo。
- **API 边界**：`gui` 只经 `xrr_fitter.api` 触达后端，100 个导出符号即全部词汇表。
- **后端**：`model`/`services`/`fit`/`analysis`/`io`/`physics` 一行不改。
- **dock 机制与持久化**：QDockWidget + `dock_state` + `capture/restore/reset_dock_layout` + `api.set_dock_state`。
- **引导↔专家共享代码路径**：`_guidance_actions` 把引导步骤路由到专家面板的**同一批按钮**（`data_panel.import_files_button.click` 等），不得让引导另写一套逻辑。
- **9 阶段进度模型**：`progress.py` 的 `STAGE_WEIGHTS`/`stage_position`/`PROGRESS_RESOLUTION=1000`。
- **5 个面板 widget 本体**：DataPanel / StructurePanel / ParametersPanel / FitPanel / ResultsPanel 的内部实现全部复用，只**重新父化与重新编排容器**。

### 2.2 布局三方案取舍

设计稿三区 = 管线导航（左 264px） + 自适应画布（中 1fr） + 上下文检视器（右 340px）。现状 = 5-dock 停靠 + 中央 stack(guidance↔plot)。

| 方案 | 做法 | dock_state 契约 | test_dock_layout | 还原度 | 结论 |
|---|---|---|---|---|---|
| A 纯 dock 重排 | 只改 `DOCK_SPECS` 的 area/size/tabify 分组 | 不动 | 不动 | 中（仍是可拖拽 dock 观感） | 可选 |
| B QSplitter 三栏 | `setCentralWidget(QSplitter[nav,canvas,inspector])` | **作废** | **8 个全重写** | 高 | ❌ 破坏面过大，不推荐 |
| **C 保留 dock + 新增导航区（推荐）** | 保留 dock 机制；新增左侧「管线导航」专属区；数据/结构/参数/拟合/结果 dock 按三区重编排；导航区不可关闭 | **保留** | 增量微调 | 高 | ✅ **采用** |

**方案 C 细化**：
- **左区（管线导航）**：新增 `gui/navigation/panel.py:PipelineNav`（继承 `QWidget`，仅依赖 `gui`+`api`），承载设计稿的「导入→结构→参数→拟合→结果」步骤脉络 + 数据集列表 + 结构层列表。以一个 `DockWidgetMovable` 但**去掉 `Closable`** 的固定左 dock 承载（或 `QMainWindow` 持久左栏）。保留 `test_dataset_list_height_follows_the_datasets_it_holds` 的 sizeHint 自适应行为。
- **中区（自适应画布）**：`central_stack`（guidance↔plot）**保持不变**——`test_the_plot_is_the_central_widget_not_a_dock` 继续绿。
- **右区（上下文检视器）**：parameters/fit/results 三 dock 继续 tabify，靠右，宽度对齐设计稿 340px（`resizeDocks` 调值）。

> 若用户要求像素级还原设计稿的「固定不可拖拽三栏」，再评估方案 B 的成本；默认按 C 施工。

---

## 3. 设计令牌与视觉规范落地（里程碑 M1）

`gui/theme.py` **已有** `ThemeTokens` / `LIGHT_TOKENS` / `DARK_TOKENS` / `PlotPalette` / `LIGHT_PLOT_PALETTE` / `DARK_PLOT_PALETTE` / `palette_tokens()`（按窗口明度<128 自动切换），且注释明确「数据色相跨明暗固定」——**与设计稿理念一致**。

**改动**：
1. 把设计稿 `#f7 设计系统` 段的 token 值（表面/墨色/强调/描边/圆角/间距）核对并合入 `LIGHT_TOKENS`/`DARK_TOKENS`，保持字段名不变。
2. 数据调色板改为设计稿的 **Okabe-Ito 色序**，写入 `LIGHT_PLOT_PALETTE`/`DARK_PLOT_PALETTE`，**保持跨明暗色相固定**（遵循既有注释约束）。
3. 置信度**双编码**（颜色+形状+文字）：`ConfidenceClass` 五类（可信●/可用但相关◆/多解▲/不可信■/不可用○）→ 在 `chrome.py`（状态色）与 `results/`（图例）统一形状+标签，不得仅靠颜色区分。
4. 半无限介质（air/基底）取中性色；发散型（残差）用双色相+中性灰中点——落到 `PlotPalette` 的 diverging 槽位。

**验证**：`tests/gui/test_theme_semantics.py` + `test_plot_palette.py` 作为 RED→GREEN 锚点。先改这两个测试断言新 token/色序 → 看红 → 改 `theme.py` → 看绿。用 dataviz skill 的 `validate_palette.js` 跑 Okabe-Ito 色序做 CVD 校验（ΔE≥8）。

---

## 4. 三区骨架重塑（里程碑 M2 — 核心）

**唯一改动入口**：`gui/window_layout.py:install_workspace(window, document)`。

现状装配（157 行）逐段处置：
| 现状代码 | 处置 |
|---|---|
| 构造 5 个 panel + PlotPanel + export_button（L91–102） | **不动** |
| `DOCK_SPECS` 5 dock addDockWidget（L34–40, L111–115） | 重排：data/structure 归入左「管线导航」区容器；parameters/fit/results 归右「检视器」区 |
| `central_stack`（guidance↔plot, L118–123） | **不动** |
| tabify 右侧三 dock + raise parameters（L133–135） | 保留，作为右区「检视器」tab |
| `resizeDocks([data,parameters],[320,380])`（L136–140） | 调为设计稿 264/340 基准 |
| `_default_dock_state = saveState()`（L141） | **不动**（reset 依赖它） |
| `_guidance_actions` 路由（L75–85） | **不动** |

**新增**：`gui/navigation/panel.py`（新目录，含 `__init__.py`）：
- `PipelineNav(QWidget)`：垂直布局，顶部步骤脉络（导入→结构→参数→拟合→结果，高亮当前步），中部数据集列表（复用 `data_panel` 的 tree 或镜像其 model），底部结构层列表。
- 只 `import xrr_fitter.api as api` + `from xrr_fitter.gui...`；**不碰后端**。
- 通过 `document` 信号（`project_changed` 等）刷新当前步高亮，与 `set_guidance_visible`/`expert_mode` 联动。

**RED→GREEN**：
1. 新增 `tests/gui/test_pipeline_nav.py`：断言 `window.navigation` 存在、当前步随 `document` 状态高亮、导航区不可关闭。先写测试看红。
2. 调整 `test_dock_layout.py`：若 data/structure 改由导航区承载，更新 `DOCK_NAMES` 相关断言（保留 dock_state 往返测试原样绿）。
3. 实现 → 全 `tests/gui/` 绿。

---

## 5. 分区能力视觉对齐（里程碑 M3–M7）

| 里程碑 | 对应设计稿帧 | 主要文件 | 动作 | RED→GREEN 锚点 |
|---|---|---|---|---|
| **M3 引导四步 + 双模式** | ②① | `guidance/panel.py`（`STEP_SPECS`）, `main_window.py:set_guidance_visible`, `parameters/panel.py:expertModeToggle` | 引导四步版式对齐；确认 `expert_mode`(document 持久) 与 guidance(view 层) 两个正交模式的切换动线 | `test_guidance.py` / `test_expert_views.py` |
| **M4 结构编辑 + SLD** | ③ | `structure/panel.py`+`editor.py`+`stack.py`, `plots/sld.py`+`sld_drag.py` | 层堆叠卡片版式；SLD 可拖拽剖面 + 16–84% 可信带样式对齐（伴随窗口，非 tab） | `test_structure_editor.py` / SLD 相关 plot_cases |
| **M5 九阶段进度** | ④ | `fitting/progress.py`, `fitting/panel.py`, `plots/live.py` | **进度模型不改**（已吻合）；仅进度条 0–1000 + 9 阶段标签 + 实时 log/raw/qz⁴/residual 四面板样式对齐 | `test_fit_progress.py`（注意 pyqtgraph logmode getData 陷阱，记忆 `xrr-pyqtgraph-logmode-getdata`） |
| **M6 不确定度** | ⑤ | `results/uncertainty.py`（`UncertaintyView`/`McmcControls`）, `plots/diagnostics.py`, `plots/heatmaps.py` | 相关矩阵热图 + Profile 似然 + Bootstrap + MCMC 后验版式；置信度双编码贯穿 | `test_uncertainty_dialog.py` |
| **M7 导入/导出** | ⑥ | `data/import_dialog.py`, `export/dialog.py`（`ExportWorkflow`/`OrtOptionDialog`） | 导入源校验向导 + 导出 ORSO/Excel/图/清单版式；**`import_dialog→io.xy` 例外保持不动** | `test_export_dialog.py` / `test_data_masks.py` |

---

## 6. 里程碑总表与工序

| MS | 名称 | 依赖 | 交付 | 门禁 |
|---|---|---|---|---|
| M0 | 基线锁定 | — | 全测试绿基线快照 + 环境就绪 | 见 §7 全量 |
| M1 | 设计令牌统一 | M0 | `theme.py` 令牌+调色板 | theme_semantics/plot_palette + validate_palette.js |
| M2 | 三区骨架 | M1 | `window_layout.py` 重排 + `navigation/` | test_dock_layout + test_pipeline_nav + architecture |
| M3 | 引导+双模式 | M2 | guidance 版式 | test_guidance/test_expert_views |
| M4 | 结构+SLD | M2 | structure/sld 版式 | test_structure_* + plot_cases |
| M5 | 九阶段进度 | M2 | progress/live 版式 | test_fit_progress |
| M6 | 不确定度 | M2 | uncertainty/diagnostics 版式 | test_uncertainty_dialog |
| M7 | 导入导出 | M2 | import/export 版式 | test_export_dialog |
| M8 | 收口 | M1–M7 | 全门禁 + 打包冒烟 | §8 全量 |

**每个里程碑内部工序（AGENTS.md 要求，改布局属"运行时行为变更+重构+项目作用域" → 强制 TDD）**：
```
① 先改/新增该里程碑锚点测试，表达新契约   → 跑测试看 RED（留证据）
② 最小实现                                → 跑测试看 GREEN（留证据）
③ 跑该层全量（tests/gui + tests/architecture）→ 无回归
④ ruff + hygiene + radon                  → 门禁绿
⑤ 清理临时产物，汇报：改了什么/怎么验证/剩余风险
```

---

## 7. 验证与门禁命令集（可复制）

> 记忆 `xrr-test-suite-commands`：**必须加 `--import-mode=importlib`**。GUI 测试需 offscreen。
> 记忆 `xrr-worktree-env-setup`：worktree 里每树必须独立 `.venv`，装完删 egg-info，worktree 固有 1 个 `.git-is_dir` 失败可接受。

```bash
# 0. 单元 + 架构（每次里程碑收尾）
QT_QPA_PLATFORM=offscreen python -m pytest tests/architecture --import-mode=importlib -q

# 1. GUI 全量（pytest-qt，离屏）
QT_QPA_PLATFORM=offscreen python -m pytest tests/gui --import-mode=importlib -q

# 2. 单个锚点测试（举例 M2）
QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_dock_layout.py tests/gui/test_pipeline_nav.py --import-mode=importlib -q

# 3. 代码风格门禁（pre-commit：ruff --fix + ruff-format，按 staged 文件）
ruff check src/xrr_fitter/gui tools tests
ruff format --check src/xrr_fitter/gui

# 4. 卫生 + 圈复杂度门禁
python tools/check_hygiene.py
python tools/check_radon.py

# 5. 综合校验（对应 CI verify.yml）
python tools/verify.py

# 6. Windows 打包冒烟（对应 windows-executable.yml；pyinstaller==6.21.0 在 [tool.xrr.windows-packaging]，非 optional-deps）
pyinstaller packaging/windows/xrr-fitter.spec --noconfirm

# 7. 调色板 CVD 校验（dataviz skill）
node <skill>/scripts/validate_palette.js "<Okabe-Ito hex 序列>" --mode light
node <skill>/scripts/validate_palette.js "<Okabe-Ito hex 序列>" --mode dark
```

**CI 对齐**：`.github/workflows/` 三条流水 `verify.yml` / `windows-executable.yml` / `pr-verify.yml`。本地 §7 全绿后再推，避免 CI 反复。

---

## 8. 风险登记与规避

| # | 风险 | 触发条件 | 规避 |
|---|---|---|---|
| R1 | 依赖门禁红 | 新导航/布局代码 import `model`/`services`/`io`；或在 `gui.plots.*` 外用 numpy/matplotlib | 新代码只 `import api` + `gui.*`；画图代码一律落 `gui.plots.*` |
| R2 | dock 持久化契约破裂 | 误改 QSplitter 或删 `capture/restore/reset_dock_layout` | 采用方案 C，保留 dock 机制；`dock_state` 往返测试须保持绿 |
| R3 | checkpoint 指纹漂移 | 为 UI 往 `fit/` 层 dataclass 加字段 | 纯表现层，不碰 fit 层；`test_frozen_stage_search` 守门 |
| R4 | 误"修复"已批准例外 | 把 `import_dialog→io.xy` 当越界删掉 | 它是 `PACKAGE_EDGE_EXCEPTIONS` 登记例外，**保持不动** |
| R5 | pyqtgraph logmode 断言错 | `setLogMode(y=True)` 后 `getData()` 返回 log10 空间 | 断言比 `log10(max(v,floor))`，非线性值（记忆 `xrr-pyqtgraph-logmode-getdata`） |
| R6 | 明暗数据色相漂移 | 合入 Okabe-Ito 时让暗色模式换色相 | 遵循 theme.py 既有约束，跨明暗色相固定 |
| R7 | 9 阶段口径误改 | 以为要新增/删阶段 | `progress.py` 已定义 9 阶段（权重和=1.0、分辨率=1000），仅视觉对齐 |
| R8 | 并发写落错分支 | 多 claude 进程并发（记忆 `xrr-repo-layout-and-concurrency`） | 单进程施工或独立 worktree；提交前确认 HEAD |

---

## 9. 完成判据（Definition of Done）

1. 设计稿三区（管线导航/自适应画布/上下文检视器）在真实 GUI 中呈现，明暗双模式均达标。
2. 16 项能力全部可达，且经由**真实 `api.py` 符号**（§1.3 修正表）驱动。
3. `tests/gui` + `tests/architecture` 全量绿（离屏 + `--import-mode=importlib`），无回归。
4. `ruff` / `check_hygiene.py` / `check_radon.py` / `verify.py` 全绿。
5. Windows PyInstaller 打包冒烟通过。
6. `dock_state` 持久化往返、undo/redo、引导↔专家共享代码路径全部保持。
7. 每个里程碑留有 RED→GREEN 证据；汇报含：改了什么 / 怎么验证 / 清理了什么 / 剩余风险。

---

## 附：施工顺序建议

```
M0 基线  →  M1 令牌  →  M2 三区骨架 ──┬─→ M3 引导+双模式
                                      ├─→ M4 结构+SLD
                                      ├─→ M5 九阶段进度
                                      ├─→ M6 不确定度
                                      └─→ M7 导入导出
                                            ↓
                                      M8 全门禁 + 打包收口
```
M3–M7 在 M2 骨架就位后可**独立并行**（不同文件、无共享写集），符合 AGENTS.md 子代理边界；M1/M2/M8 为串行关键路径，由主线直接处理。

---

## 进度小节（收口交接 · 2026-08-24）

> 本节由施工主线在上下文将满时写入，记录 M1–M8 的落地状态与最后一轮新鲜门禁证据。改动全部留在 worktree `.claude/worktrees/gui-reskin`，**未提交、未合并**（遵循「提交/合并前先问用户」）。

### 里程碑状态

| 里程碑 | 状态 | 落地方式 | 锚点证据 |
|---|---|---|---|
| M1 设计令牌 | ✅ 已落地 | Okabe-Ito 数据色板 + 明暗双模式令牌入 `theme.py` | `test_theme_semantics.py` / `test_plot_palette.py` 绿 |
| M2 三区骨架 | ✅ 已落地 | 新增 `navigationDock`（管线导航），保留 6 个 QDockWidget，**未改 QSplitter** | `test_dock_layout.py` 绿；boot smoke `docks=6` |
| M3 引导+双模式 | ✅ 已落地 | 引导流程 + 导航协调 + 明暗切换 | `test_guidance.py` / `test_guidance_nav_coordination.py` / `test_pipeline_nav*.py` 绿 |
| M4 结构+SLD | ✅ 已落地 | SLD 可信带固定 `DATA_CANDIDATE` 橙，内外带仅以透明度区分（0.28/0.14） | `plot_cases_*.py` / `test_results.py` 绿 |
| M5 九阶段进度 | ✅ 已落地 | 进度视觉对齐既有 9 阶段（权重和=1.0），未增删阶段 | `plot_cases_5.py` 绿 |
| M6 不确定度 | ✅ 已落地 | 相关矩阵改用 `diverging_colormap()`（随明暗重解析），替换固定 `coolwarm` | `plot_cases_1.py` 新增色相锚点绿 |
| M7 导入/导出 | ✅ 核验-报告（无代码改动） | 对话框经 app 级 `setStyleSheet` 自动继承 reskin；见下「M7 裁定」 | `test_export_dialog.py` / `test_data_masks.py` / `test_data_import.py` = 41 passed |

### M7 裁定（为何无代码改动）
- 导入/导出对话框通过 `theme.apply_theme` 的 `application.setStyleSheet(...)` **自动继承** reskin 后的 QSS（`QPushButton`/`QGroupBox`/`sectionCard` 均已全局改版），无需逐对话框改样式。
- 导入对话框已用 `QGroupBox("光路")`/`QGroupBox("仪器")` 分组，且 `_refresh_preview` 通过 `_preview_error` 显式暴露解析失败（符合「坏文件不静默吞掉」）。
- 设计稿 frame ⑥ 的批量预览表 / 多格式勾选（CSV·SVG·分项 PNG）/ manifest 预览属**新增能力**，需扩 `api.py`（现仅 `export_result(..., include_ort=bool)`）——属「reskin 不是重建」的越界项，**故意延后**。
- 无任何对话框把 OK 按钮提升为 `primary`（既定 chrome 约定）；`StyledDialog` 仅服务「关于」对话框。强推二者会破坏约定或构成跨切面扩面，均**拒绝**。

### M8 门禁（最后一轮新鲜证据）
- 全量：`tests/gui tests/architecture` → **866 passed, 1 warning, 退出 0**（离屏 + `--import-mode=importlib`）。唯一告警为 `diagnostics.py:133` constrained_layout 既有告警，良性。
- `ruff check` → All checks passed!；`ruff format --check` → 466 files already formatted。
- `tools/check_radon.py` → 退出 0；`tools/check_hygiene.py` → 退出 0（发现项均为 worktree 固有：`.git` 文件、`.venv`/`.pytest_cache`/`.ruff_cache` 归属与符号链接、`__pycache__`）。
- Boot smoke（离屏）→ `BOOT_OK window=mainWindow docks=6`（dataDock/fitDock/navigationDock/parametersDock/resultsDock/structureDock），Scheme C dock 契约完整、无 QSplitter。
- Windows PyInstaller 打包冒烟（DoD §9.5）→ **本 darwin worktree 不可运行**（PyInstaller 按记忆 `xrr-windows-lock-gate` 故意不入依赖，打包门禁仅 Windows-CI）。建议 CI 侧补验。

### 剩余风险 / 欠账
- M6 相关矩阵仅统一为主题 diverging 色相（蓝↔灰↔红），与设计稿「蓝↔近白↔橙」的中点白仍有细微差；改中点需动色板令牌，属扩面，未做。
- M7 frame ⑥ 富功能（批量向导 / 多格式导出 / manifest 版式）需 API 扩展，超 reskin 边界，未做。
- 对话框 OK 按钮的 primary 化属跨切面样式扩面，为守既定 chrome 约定未做。
- 变更未提交/未合并，等用户确认后再进入合并流程。
