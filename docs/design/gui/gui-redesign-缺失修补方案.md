# GUI 重设计 · 缺失元素修补方案

> 历史设计资料（2026-08 至 2026-09），于 2026-09-15 整理入库。正文中的计划、状态和命令保留原编制时点语境，不是当前待办。当前完成情况见[设计追溯](../../architecture/design-frames-traceability.md)和[验收台账](../../architecture/plan-v2-acceptance-ledger.md)。

> 配套文档：`docs/design/gui/xrr-fitter-gui-redesign.html`（设计稿）、`docs/design/gui/xrr-fitter-gui-redesign-实施计划.md`（reskin 计划，M1-M8 已完成）
> 编制日期：2026-09-03
> 审计方法：逐帧对比设计稿 HTML 元素与 `src/xrr_fitter/gui/` 源码，rg 交叉验证

---

## 0. 背景

reskin 计划（M1-M8）已完成表现层重塑：三区骨架、设计令牌、调色板、置信度双编码、各面板版式对齐。

但逐元素审计发现设计稿中 **22 个交互元素/组件在当前代码中缺失或仅部分实现**。这些不是视觉微调，而是设计稿明确画出的 UI 控件、交互流或信息展示。

本方案按「影响 x 复杂度」分 4 批排优先级，给出文件级改动目标和实现路线。

---

## 1. 缺失清单（按设计稿帧编号）

### 帧 1 主工作台 · 结果态

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G1 | 命令栏 segment 控件：引导/专家 切换 | L354-356 `.seg` | `guidanceModeAction` 在菜单里，不在工具栏 | 中 |
| G2 | 命令栏 segment 控件：独立/联合 批量切换 | L357 `.seg` | `batch_selector` 是 QComboBox 在 fitting/panel.py:66，不在命令栏 | 中 |
| G3 | 明暗切换按钮 | L361 `.btn.icon` | chrome.py 无此控件 | 低 |
| G4 | modebar 工具条（平移/缩放/复位/选区） | L390 `.modebar` | plots/ 下未找到 modebar 实现 | 中 |
| G5 | 状态栏 置信度/模式 指示区 | L474-479 | 状态栏存在（chrome.py:336-361）但无置信度标记和模式文字 | 低 |

### 帧 2 引导模式 · 四步

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G6 | 引导卡片内嵌层预览（无拖拽的简化堆叠） | L572-581 `.stack` | guidance/panel.py 无层预览 widget | 低 |
| G7 | 双 CTA 按钮（主操作 + 次操作） | L582-584 `.ctarow` | guidance/panel.py 有 action buttons 但不是 primary+secondary 双按钮 | 低 |

### 帧 3 结构编辑 · SLD 剖面

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G8 | 拖拽手柄（层拖动排序） | L656-663 `.drag` | structure/editor.py 只有 up/down 按钮，无 DragDrop/InternalMove | 中 |
| G9 | 参数三态 radio（自由/固定/仅范围） | L713-714 `.radio` | parameters/ 下未找到三态 radio 控件 | 中 |

### 帧 4 拟合进行中

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G10 | 暂停按钮 | L761, L850 | 无 pause 功能，仅有 cancel/force_stop（chrome.py:279-296） | 高 |
| G11 | 跳过本阶段按钮 | L850 | 无 skip_stage 功能 | 中 |
| G12 | 各数据集目标值表（联合拟合时） | L838-845 `.grid` | fitting/panel.py 无 per-dataset objective table | 中 |
| G13 | 联合拟合横幅说明 | L793-794 `.jbanner` | fitting/ 下无 joint banner | 低 |

### 帧 5 不确定度诊断

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G14 | MCMC 后验直方图（含 P16/P50/P84 标线） | L919-934 `.hist` | results/uncertainty.py 有 McmcControls 但无直方图 plot | 中 |
| G15 | Profile 似然 delta-chi-sq=1 参考线 | L937-949 `.pl-chart` | plots/diagnostics.py 无 delta_chi 参考线 | 低 |
| G21 | 画布四子选项卡（相关矩阵/Profile似然/SLD可信带/MCMC后验） | L911 `.tabs` | UncertaintyView 是单一 QPlainTextEdit，无 QTabWidget 切换 | 高 |
| G22 | 左侧导航不确定度方法子管线（4步：相关矩阵/自助抽样/Profile似然/MCMC） | L899-904 `.pipe` | navigation/panel.py 无不确定度方法步骤 | 中 |

### 帧 6 导入/导出

| # | 设计元素 | 设计稿位置 | 现状 | 严重度 |
|---|---|---|---|---|
| G16 | 导出多格式勾选（数据:ORSO/Excel/CSV + 图件:PNG/SVG + 随附:参数表/相关矩阵/manifest） | L1031-1053 `.fmt-chk` | export/dialog.py 仅 `include_ort=bool`，无三组勾选 | 中 |
| G17 | 导出 manifest 预览 | L1062-1080 `.manifest` | api 有 ExportManifest 但 dialog 未展示预览 | 中 |
| G18 | 导入角度约定切换（2theta/theta） | L1010-1012 `.radio` | import_dialog.py 只有 mono/mixed K-alpha radio，无角度约定 | 中 |
| G19 | 导入批量预览表 | L1014-1026 `.batch-tbl` | import_dialog.py 无 batch preview table | 中 |
| G20 | 管线步骤导航（左侧纵向步骤脉络） | navigationDock 已有框架 | 导航区存在但无步骤脉络样式（高亮当前步/完成步/未达步） | 低 |

---

## 2. 优先级分批

### 分批原则

- **Batch A（P0 高优）**：拟合控制流 — 缺失会导致用户无法暂停/跳过长时间拟合，体验硬伤
- **Batch B（P1 功能补全）**：导入/导出增强 + 诊断图表 — 缺失意味着设计稿承诺的完整工作流断裂
- **Batch C（P2 交互打磨）**：命令栏/画布工具/结构拖拽/参数三态 — 提升操作效率
- **Batch D（P3 视觉收尾）**：引导预览/双CTA/状态栏指示/暗色切换/步骤样式 — 视觉完善

---

## 3. Batch A — 拟合控制流（P0）

### A1: 暂停按钮（G10）

**目标**：拟合进行中允许用户暂停（挂起当前阶段，保留状态），再次点击恢复。

**涉及文件**：
- `gui/chrome.py` — 在 cancel 按钮旁新增 pause toggle button
- `gui/fitting/controller.py` — 接入 pause/resume 信号
- `api.py` — 确认 `FitJob` 或 worker 是否支持 pause；若不支持，需在 `fit/` 层新增 pause 机制

**风险**：
- 若 `fit/` 层不支持 pause（当前 `start_fit_job` 返回的 job 仅有 cancel），则需要后端改动，**超出纯表现层边界**
- 需确认 pause 不会破坏 checkpoint 指纹（记忆 `xrr-checkpoint-fingerprint-drift`）

**前置调研**：
1. `rg 'class FitJob' src/` — 确认 job 接口是否有 pause 预留
2. `rg 'threading|Event|pause|suspend' src/xrr_fitter/fit/` — 确认线程控制机制
3. 若无 pause 支持 → 此项需 API+后端改动，标记为「需架构评审」

**验证**：`tests/gui/test_fit_progress.py` 新增 pause/resume 场景

### A2: 跳过本阶段按钮（G11）

**目标**：拟合进行中允许跳过当前阶段，直接进入下一阶段。

**涉及文件**：
- `gui/chrome.py` 或 `gui/fitting/panel.py` — skip 按钮
- `gui/fitting/controller.py` — 发出 skip 信号
- `api.py` / `fit/` — 确认是否支持阶段跳过

**风险**：与 A1 相同，若后端不支持则需架构评审。

**前置调研**：
1. `rg 'stage|Stage|next_stage|skip' src/xrr_fitter/fit/` — 确认阶段推进机制
2. 若 9 阶段是严格顺序依赖（每阶段输入=上阶段输出），跳过可能产出无意义结果 → 需确认语义

**验证**：`tests/gui/test_fit_progress.py` 新增 skip 场景

---

## 4. Batch B — 导入/导出增强 + 诊断图表（P1）

### B1: 导出多格式勾选 + manifest 预览（G16 + G17）

**目标**：导出对话框提供 ORSO(.ort) / Excel(.xlsx) / CSV 数据格式 + PNG/SVG 图片格式的独立勾选，底部展示即将导出的文件清单预览。

**涉及文件**：
- `gui/export/dialog.py` — 替换单一 `include_ort` checkbox 为多格式勾选组；新增 manifest 预览区
- `api.py` — 当前 `export_result(..., include_ort=bool)` 接口需扩展为接受格式列表
- `gui/export/` 下可能需要新增 manifest 预览 widget

**依赖**：`api.ExportManifest` 已存在，可用于驱动预览列表。

**风险**：API 签名变更需确保向后兼容（旧项目调用不崩）。

**验证**：`tests/gui/test_export_dialog.py` 扩展格式勾选和 manifest 断言

### B2: 导入角度约定切换 + 批量预览表（G18 + G19）

**目标**：导入对话框新增 2theta/theta 角度约定 radio；批量导入时展示文件-数据集预览表。

**涉及文件**：
- `gui/data/import_dialog.py` — 新增角度约定 radio group + batch preview QTableWidget + 高级选项组（列分隔符下拉、跳过表头行、强度列取对数 checkbox，对应设计稿 L1015-1022）
- `api.py` — 确认 `inspect_sources` / `preview_source_update` 是否已返回角度信息

**前置调研**：
1. `rg 'angle|theta|2theta|convention' src/xrr_fitter/api.py` — 确认 API 是否有角度参数
2. `rg 'preview_import_batch' src/xrr_fitter/api.py` — 确认批量预览 API

**验证**：`tests/gui/test_data_import.py` 新增角度约定和批量预览断言

### B3: 不确定度画布四子选项卡（G21）

**目标**：不确定度面板中央画布改为 QTabWidget，承载四个子页：相关矩阵 / Profile 似然 / SLD 可信带 / MCMC 后验。当前 `UncertaintyView` 是单一 `QPlainTextEdit` 显示纯文本证据，无法承载图表。

**涉及文件**：
- `gui/results/uncertainty.py` — `UncertaintyView` 从 QPlainTextEdit 改为 QTabWidget（或 QStackedWidget + 顶部 tab bar），4 页分别嵌入对应 plot widget
- `gui/plots/diagnostics.py` — 确认各诊断图可独立嵌入 tab page

**风险**：这是 G14（MCMC 直方图）和 G15（Profile 参考线）的**结构前置依赖**，必须先完成。

**验证**：`tests/gui/test_uncertainty_dialog.py` 新增 tab 切换断言

### B4: MCMC 后验直方图（G14）【依赖 B3】

**目标**：不确定度面板 MCMC 后验 tab 新增参数直方图，带 P16/P50/P84 竖线标注。

**涉及文件**：
- `gui/plots/diagnostics.py` — 新增 `McmcHistogramPlot` 类
- `gui/results/uncertainty.py` — 在 MCMC tab 接入直方图 widget

**约束**：直方图涉及 numpy/matplotlib，**必须放在 `gui/plots/`** 子树（依赖门禁 R1）。

**依赖**：`api.McmcReport` 已包含 chain samples，足够驱动直方图。

**验证**：`tests/gui/test_uncertainty_dialog.py` 或新增 `plot_cases` 断言

### B5: Profile 似然 delta-chi-sq=1 参考线（G15）【依赖 B3】

**目标**：Profile likelihood 图上叠加 delta-chi-squared = 1 水平参考线，标注置信区间边界。

**涉及文件**：
- `gui/plots/diagnostics.py` — 在 profile likelihood plot 中 addLine

**验证**：plot_cases 新增参考线断言

### B6: 各数据集目标值表（G12）

**目标**：联合拟合时，进度区展示每个数据集的独立目标函数值表格。

**涉及文件**：
- `gui/fitting/panel.py` — 新增 per-dataset objective QTableWidget
- `gui/fitting/controller.py` — 从 progress 回调中提取 per-dataset 信息

**前置调研**：`rg 'per_dataset|dataset_objective|individual' src/xrr_fitter/fit/` — 确认 fit 层是否上报分数据集目标

**验证**：`tests/gui/test_fit_progress.py` 新增联合拟合 objective 表断言

---

## 5. Batch C — 交互打磨（P2）

### C1: 命令栏 segment 控件（G1 + G2）

**目标**：工具栏新增两组 segmented control：引导/专家 模式切换、独立/联合 批量切换。替代当前菜单入口。

**涉及文件**：
- `gui/chrome.py` — 新增 `QToolButton` 组或自定义 `SegmentedControl` widget
- 保留菜单入口作为 keyboard-accessible 备选

**实现路线**：
1. 新建 `gui/widgets/segmented_control.py`（可复用 segment widget）
2. chrome.py `_install_toolbar` 中嵌入两组 segment
3. 信号连接到既有 `set_guidance_visible` / `api.set_batch_mode`

**验证**：`tests/gui/test_chrome.py` 或新增 test_command_bar 断言

### C2: 结构层拖拽手柄（G8）

**目标**：结构编辑器层列表支持拖拽排序（拖拽手柄图标），替代/补充 up/down 按钮。

**涉及文件**：
- `gui/structure/editor.py` — 启用 `QListWidget.setDragDropMode(InternalMove)` + 自定义 drag handle delegate
- `gui/structure/stack.py` — 确认 model 层支持 reorder

**风险**：拖拽排序需确保与 `api.set_structure` 的 layer 顺序语义一致。

**验证**：`tests/gui/test_structure_editor.py` 新增 drag reorder 断言

### C3: 参数三态 radio（G9）

**目标**：参数表每行新增三态选择：自由 / 固定 / 仅范围。替代当前的自由/固定 checkbox。

**涉及文件**：
- `gui/parameters/table.py` — delegate 改为三态 radio 或三态 combobox
- 确认 `api.describe_parameters` / `api.set_parameter_priors` 是否支持「仅范围」状态

**前置调研**：
1. `rg 'free|fixed|bounded|range_only' src/xrr_fitter/api.py` — 确认参数状态枚举

**验证**：`tests/gui/test_parameter_table.py` 新增三态断言

### C4: 画布 modebar（G4）

**目标**：图表区右上角叠加浮动工具条：平移 / 框选缩放 / 复位 / 框选遮罩。

**涉及文件**：
- `gui/plots/` 下新增 `modebar.py`（或集成到 `live.py`）
- 利用 pyqtgraph 内置 ViewBox 交互模式（pan/rect-zoom/autoRange）

**约束**：modebar 若需 icon 资源，放 `gui/resources/`。

**验证**：新增 modebar 功能测试

### C5: 不确定度方法子管线（G22）

**目标**：左侧导航区在「结果」步骤展开后，展示不确定度方法子管线：相关矩阵 → 自助抽样 → Profile 似然 → MCMC，带完成/进行中/未达状态。

**涉及文件**：
- `gui/navigation/panel.py` — 在结果步骤下新增 sub-pipeline widget，4 个方法步骤带状态图标
- 通过 `document` 信号刷新各方法步骤状态

**依赖**：G20（管线步骤样式）提供步骤渲染基础。

**验证**：`tests/gui/test_pipeline_nav.py` 新增不确定度子管线断言

---

### D1: 引导卡片层预览（G6）

**目标**：引导模式「结构」步骤卡片内嵌只读层堆叠预览。

**涉及文件**：`gui/guidance/panel.py` — 新增 mini layer stack widget（只读，无拖拽）

### D2: 双 CTA 按钮（G7）

**目标**：引导卡片底部改为 primary + secondary 双按钮布局。

**涉及文件**：`gui/guidance/panel.py` — 调整 action button 样式和布局

### D3: 状态栏置信度/模式指示（G5）

**目标**：状态栏新增置信度徽章（形状+颜色双编码）和当前模式文字。

**涉及文件**：`gui/chrome.py` `_install_status_bar` — 添加 confidence label + mode label

### D4: 明暗切换按钮（G3）

**目标**：工具栏新增明暗主题切换图标按钮。

**涉及文件**：`gui/chrome.py` — 新增 theme toggle button，连接 `theme.apply_theme`

### D5: 联合拟合横幅（G13）

**目标**：联合拟合模式下，拟合面板顶部展示说明横幅。

**涉及文件**：`gui/fitting/panel.py` — 新增 joint banner widget，按 batch_mode 切换可见性

### D6: 管线步骤样式完善（G20）

**目标**：导航区步骤脉络补全「当前步高亮/完成步打勾/未达步灰显」的视觉状态。

**涉及文件**：`gui/navigation/panel.py` — 完善 PipelineNav 步骤项样式

---

## 7. 工序总览

```
Batch A（P0 拟合控制）────────────────────────────────────────┐
  A1 暂停按钮 ─── 需前置调研 fit 层 pause 支持               │
  A2 跳过阶段 ─── 需前置调研阶段跳过语义                     │
                                                              ↓
Batch B（P1 功能补全）────────────────────────────────────────┐
  B1 导出多格式+manifest ─── 需 API 扩展                      │
  B2 导入角度约定+批量预览                                    │
  B3 不确定度画布四子选项卡 ─── G14/G15 的结构前置            │
  B4 MCMC 后验直方图 ─── 依赖 B3                              │
  B5 Profile 似然参考线 ─── 依赖 B3                           │
  B6 各数据集目标值表                                         │
                                                              ↓
Batch C（P2 交互打磨）────────────────────────────────────────┐
  C1 命令栏 segment ─── 可与 B 批并行                         │
  C2 结构层拖拽                                               │
  C3 参数三态 radio                                           │
  C4 画布 modebar                                             │
  C5 不确定度方法子管线                                       │
                                                              ↓
Batch D（P3 视觉收尾）────────────────────────────────────────┘
  D1-D6 独立小项，无阻塞依赖
```

**串行关键路径**：A1/A2 需先确认后端能力 → 决定是否需要 API/fit 层改动 → 若需要，则 A 批会拉长。B3（四子选项卡）是 B4/B5 的硬前置。

**可并行**：B1/B2（对话框）与 B3-B6（诊断图表）互不干扰；C1-C5 五项文件无交叉；D1-D6 各自独立。

---

## 8. 架构约束提醒

以下约束与 reskin 计划相同，此处重申以免施工越界：

1. **依赖边界**：gui 只能 import `api` + `gui.*`；numpy/matplotlib 只能在 `gui.plots.*`
2. **dock 持久化契约**：保留 QDockWidget + dock_state 往返；不得改 QSplitter
3. **checkpoint 指纹**：不碰 `fit/` 层 dataclass 字段（记忆 `xrr-checkpoint-fingerprint-drift`）
4. **import_dialog -> io.xy 例外**：已批准例外，保持不动
5. **pyqtgraph logmode**：getData() 返回 log10 空间（记忆 `xrr-pyqtgraph-logmode-getdata`）
6. **引导-专家共享代码路径**：引导步骤路由到专家面板同一批按钮，不得另写逻辑

---

## 9. 验证门禁（可复制命令）

```bash
# GUI 全量
QT_QPA_PLATFORM=offscreen python -m pytest tests/gui --import-mode=importlib -q

# 架构
QT_QPA_PLATFORM=offscreen python -m pytest tests/architecture --import-mode=importlib -q

# ruff
ruff check src/xrr_fitter/gui tools tests
ruff format --check src/xrr_fitter/gui

# 卫生 + 圈复杂度
python tools/check_hygiene.py
python tools/check_radon.py
```

---

## 10. 每项施工 TDD 工序

```
1. 先改/新增锚点测试，表达新契约       -> 跑测试看 RED（留证据）
2. 最小实现                            -> 跑测试看 GREEN（留证据）
3. 跑该层全量 tests/gui + architecture  -> 无回归
4. ruff + hygiene + radon              -> 门禁绿
5. 清理临时产物，汇报改了什么/验证/风险
```

---

## 11. 完成判据

1. 设计稿 6 帧全部 UI 元素在实现中可追溯（每项有 rg 可验证的代码路径）
2. `tests/gui` + `tests/architecture` 全量绿，无回归
3. ruff / hygiene / radon 门禁绿
4. dock_state 持久化往返、undo/redo、引导-专家共享路径全部保持
5. 每批有 RED-GREEN 证据
