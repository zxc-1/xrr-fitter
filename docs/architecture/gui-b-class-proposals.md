# GUI B 类改进设计提案

> **状态**：2026-08-08 已完成决策，见「分类总览」的状态列。本文档本身不修改任何
> `api` 边界或运行时代码，仅记录实施方案、取舍与验收标准。
>
> **背景**：GUI 优化的九大类改进中，纯 GUI（A 类）项已在
> `worktree-gui-progress-liveness` 分支逐项落地。本文档收录剩余需要触及
> `api` 边界或需产品决策的项目。每项给出：动机、现状证据、设计方案、边界影响、
> TDD 切入点、风险与验收标准。
>
> **实施顺序**：#12.2 绘图配色（纯 GUI，最小）→ B1 可停靠布局（已签核 api）→
> B2 双模式（最大）。

---

## 分类总览

| 编号 | 项目 | 类别 | 边界影响 | 状态（2026-08-08） |
|------|------|------|----------|----------------|
| #12.2 | 暗色主题 | A（改判） | 不触及 api | 已决策：只让绘图跟随系统调色板 |
| B1 | 可停靠面板布局 | B | `ProjectUiState.dock_state` + 编解码 + `api.set_dock_state` | **已签核**，待实施 |
| B2 | 向导 / 专家双模式 | A（纯呈现层） | 预计不触及 api | 已决策：要做分步引导流程 |
| #5.2 | 联合拟合进度归属 | B（已落地方案 B） | `api` 暴露 `describe_joint_layout` 只读查询 | 已落地 |
| #10.2 | 自动保存草稿 | — | 复用现有 `api.save_project` | **已实现**，见下 |
| #10.3 | 项目历史版本 | C | 需 API 支持版本快照 | 已决策：暂不立项 |

> A 类（纯 GUI）项不在本文档范围内，已直接实现并提交。

---

## #12.2 暗色主题（已改判为 A 类）

> **状态：2026-08-08 决策为「只修绘图，跟随系统」，改判 A 类，不触及 api。**
> 本节原先称"当前只有单一浅色主题"，该描述不准确，已按实测更正如下。

### 动机
系统切换到暗色外观后，控件已随之变暗，但 8 张 matplotlib 诊断图仍是纯白，在暗
环境下刺眼。这是本项要解决的真实缺口。

### 现状证据（实测更正）
- 主题在启动时全局应用一次：`src/xrr_fitter/gui/application.py:22` 调
  `apply_theme(application)`，作用于整个 `QApplication`。
- `theme.py` **已有 `LIGHT_TOKENS` 与 `DARK_TOKENS` 两套令牌**，且
  `palette_tokens(palette)` 按 `QPalette.Window` 的 lightness 是否小于 128 自动
  选择。实测强制暗色调色板确实产出不同的样式表，因此**控件跟随系统暗色已可工作**。
- 真正的缺口只在绘图：`Figure` 不吃 Qt 样式表。实测 figure 与 axes 的 facecolor
  均为硬编码纯白 `(1.0, 1.0, 1.0, 1.0)`。
- 因此本项**不需要** `theme_mode` 字段或 `set_theme_mode`：跟随系统即可，代价是
  放弃手动三选一与 high_contrast 模式。
- UI 偏好已有持久化先例：`ProjectUiState`（`src/xrr_fitter/model/project.py:135`）
  已持久化 `expert_mode`、`plot_tab_index`、`workspace_splitter_sizes` 等；
  编解码在 `io/project_codec.py:140`（`_ui_to_dict`）与 `:151`（`_ui_from_dict`）。
- 绘图诊断图用的是各自 `Figure`，不受 `QApplication` 样式表影响，需要独立的
  matplotlib 颜色处理（见下「绘图一致性」）。

### 为什么最终不是 B 类
原提案认为"记住用户选择"必须持久化 `theme_mode`，因而跨 api 边界。用户已决定
**跟随系统外观**而不提供手动选择，于是无需持久化任何主题偏好：`palette_tokens()`
每次从当前 `QPalette` 解析即可。api 边界不受影响。

仍需人工视觉验收（对比度、绘图配色），单测只能覆盖"取到了 palette"。

### 设计方案（仅绘图层，纯 GUI）
- `theme.py` 新增 `plot_palette(tokens) -> PlotPalette`，从已有的 `ThemeTokens`
  派生绘图用色（figure/axes 背景、前景文本、网格、脊线）。
- 绘图函数从 palette 取色而非硬编码。硬编码色共 9 处，集中在
  `plots/reflectivity.py`(6)、`plots/panel.py`(2)、`plots/diagnostics.py`(1)。
- 数据系列色（`#0072B2` 蓝、`#009E73` 绿、`#E69F00` 橙）取自 Okabe-Ito 色盲安全
  调色板，在两种背景下都可辨，**保持不变**；只改背景、文本、网格、脊线等结构色。

**绘图一致性（关键风险点）**
- matplotlib `Figure` 不吃 Qt 样式表。需要在 `diagnostics.py` 的绘图入口按当前
  mode 设置 `figure.set_facecolor` / `axes` 前景色 / 文本色 / 网格色。
- 建议：新增 `plot_palette(mode) -> PlotPalette`，绘图函数从 palette 取色而非硬编码。
  当前硬编码色（如 `#8A8A8E` 空状态灰、`#009E73` 预览绿、`#E69F00` 拟合范围橙）
  需要在暗色下替换为对比度达标的对应色。
- 这一步是本提案工作量与风险的主要来源，需要逐个诊断图人工核对。

**应用时机**
- 绘图函数在每次绘制时从当前调色板取 palette，因此系统外观切换后的重绘自然跟上，
  无需运行时切换逻辑，也无需菜单项。

### TDD 切入点
1. `plot_palette` 从暗色令牌派生的背景与从浅色令牌派生的不同（纯函数单测）。
2. 绘图后 figure/axes 的 facecolor 等于 palette 的背景色，而非硬编码纯白。
3. 暗色下前景文本与背景的相对亮度差满足最小对比度阈值。

### 风险
- 硬编码色需一次性收敛到 palette，否则暗色下会有刺眼的残留浅色元素。
- 必须人工视觉验收；单测覆盖不了"看起来对"。

### 验收标准
- 系统切暗色后，控件与全部诊断图一并变暗，无残留白底图。
- 数据系列色保持 Okabe-Ito 不变，在两种背景下均可辨。
- 绘图单测全绿；复杂度门禁不退化；api 边界零改动。

---

## B1 可停靠面板布局

> **状态：2026-08-08 用户已签核 api 边界，待实施。**

### 动机
当前三列固定，`setChildrenCollapsible(False)` 且三列均不可折叠，无法适应"这一步只
关心结构"或"只想看大图"的场景。这是调研中最强的横向共识：GenX 一开始就可停靠，
Refl1D 已从固定四象限迁移到 golden-layout 可拖拽布局。

### 设计方案

**模型层（已签核触及 api）**
- `ProjectUiState` 新增 `dock_state: str = ""`，存 `QMainWindow.saveState()` 的
  base64 文本（`QByteArray` 不能直接进 JSON）。空串表示"用默认布局"。
- `api.set_dock_state(project, state) -> XrrProject`，与 `set_expert_mode` 对称。
- 编解码：`_ui_to_dict` / `_ui_from_dict` 各加一键；缺失时回落 `""`，旧项目可读。

**GUI 层**
- 六个 `QDockWidget`：数据集、样品结构、绘图、参数、拟合、结果。绘图作为
  `setCentralWidget` 而非 dock（它是工作主体，始终存在）。
- 「视图 ▸ 面板」子菜单列出各 dock 的 `toggleViewAction()`，天然获得显示/隐藏。
- 新增「视图 ▸ 重置布局」，在用户把布局拖乱后可恢复默认。

**跨版本兼容（签核时已明确的已知风险）**
- `saveState()` 是 Qt 版本相关的不透明字节串。`restoreState()` 返回 `bool`，失败
  时**必须回落到默认布局而不是报错**——升级 Qt 后打开旧项目不能打不开。
- base64 解码失败、字节串损坏同样回落默认，不抛异常。

### TDD 切入点
1. 模型往返（RED）：`set_dock_state` → 编码 → 解码 → 状态保持；旧 payload 缺字段
   → 回落 `""`。
2. 回落（RED）：喂入损坏的 base64 / 非法字节串 → 窗口仍可构造，使用默认布局。
3. GUI（RED）：隐藏某 dock → `toggleViewAction` 反映；重置布局恢复全部可见。

### 风险
- 会使 `tests/gui/test_accessibility.py`、`test_project_document.py`、
  `test_workspace.py` 中对 `projectColumn`/`plotColumn`/`analysisColumn` 与三列顺序
  的断言失效，需同步重写。
- `workspace_splitter_sizes` / `left_splitter_sizes` 在 dock 布局下失去意义。保留
  字段以维持旧文件可读，但不再驱动布局；需在字段 docstring 注明其已弃用语义，
  避免后来者误以为它仍有效。

### 验收标准
- 六个面板可拖拽重排、浮动、堆叠；布局随项目保存/打开往返一致。
- 损坏或跨版本不兼容的 `dock_state` 回落默认布局且不报错。
- 旧项目文件（无 `dock_state`）打开正常。
- 复杂度门禁不退化；无障碍名称与键盘顺序在新布局下重新覆盖。

---

## B2 向导 / 专家双模式

> **状态：2026-08-08 用户已决策要做，预计纯呈现层，待实施。**

### 动机
`expert_mode` 目前只是"加显控件"（本轮优化后 33 vs 30），不是两套工作面。非专家
用户仍要面对完整三列布局。参考 Rigaku SmartLab Studio II 的 "Guidance" 模式与
LEPTOS wizard：同一软件服务两类用户，不为新手牺牲专家密度。

### 设计方案
- 新增 `gui/guidance/` 分步容器，四步：导入数据 → 确认结构 → 一键拟合 → 看结果。
- 每步只显示该步需要的控件，目标 10 个以内；下一步在前一步满足前置条件后才可用
  （复用现有 `api.preflight_*` 判断，不新增校验逻辑）。
- 与专家模式互为顶层视图切换，各自保持状态：从引导切到专家再切回，不丢进度。
- 预计不触及 api：步骤前置条件全部可由现有只读查询回答。**实施前需确认这一点**，
  若某步需要新的只读查询，则该查询需单独签核。

### TDD 切入点
1. 每步的可见控件数不超过约定上限（防回归）。
2. 前置条件未满足时下一步不可用；满足后可用。
3. 引导 ↔ 专家切换后，项目状态与已完成步骤不丢失。

### 风险
- 本清单中工作量最大的一项。
- 四步划分需与实际使用流程核对，避免引导反而挡路。
- 若引导流程需要新的 api 只读查询，须停下来先取得签核。

### 验收标准
- 引导流程可独立完成"导入到看结果"的闭环，无需切到专家模式。
- 每步控件数在约定上限内。
- 双向切换不丢状态；复杂度门禁不退化。

---

## #5.2 联合拟合进度归属

> **状态：已按方案 B 落地。** `api.describe_joint_layout(project) -> api.JointFitLayout`
> 已实现（`services/projects.py` + `model/parameters.py`，经 `api.py` 导出），联合拟合
> 开始时由 `FitPanel` 计算布局并在 `ProgressView` 顶部横幅展示参与数据集与共享参数组。
> 下文保留原设计记录；方案 A（逐步 scope 归属）仍未实施，如需要可另行立项。

### 动机
联合拟合同时优化多个数据集的共享参数。当前进度反馈无法告诉用户"此刻在优化哪个数据集、
哪些是共享参数"，多数据集联合运行时用户只能看到一个笼统的阶段进度条。

### 现状证据
- `FitProgress`（`src/xrr_fitter/model/fitting.py:424`）已有 `dataset_id` 字段，但
  **联合流水线固定发 `None`**：`fit/joint_pipeline.py:308`
  `callback(FitProgress(None, stage, completed, total, best, message))`。
- 这不是缺陷而是语义真实：联合拟合优化的是跨数据集的**共享参数**，某一时刻没有单一
  "当前数据集"。独立批量模式才有逐数据集归属（`services/fitting.py` 多处按 dataset 发
  `FitProgress`）。
- 联合问题结构（哪些数据集参与、哪些参数共享、哪些是各数据集局部参数）存在于
  `fit/joint_problem.py` / `joint_sharing.py`，但**未通过 `api` 暴露给 GUI**。

### 为什么是 B 类
要"明确当前优化的数据集和参数"，GUI 需要读到联合问题的结构与每步进度归属。这要求
`api` 暴露联合问题的只读视图（参与的数据集集合、共享参数名、每个阶段针对的子问题），
且 `FitProgress` 或其伴随消息要携带足够的归属信息。这跨越了 `api` 边界。

### 设计方案（两个候选，需产品选型）

**方案 A：进度携带结构化归属（改动较大）**
- `api` 暴露 `JointFitLayout`（只读）：`shared_parameters: tuple[str, ...]`、
  `datasets: tuple[str, ...]`、以及每阶段的 `active_datasets`。
- `FitProgress` 增加可选 `scope` 字段（如 `"joint:shared"` / `"joint:local:<dataset_id>"`），
  联合流水线 `_emit` 按当前阶段填充。
- GUI 进度面板据此显示"正在优化：共享参数（数据集 A、B、C）"或"正在细化：数据集 B 局部参数"。
- 优点：信息最完整。缺点：改动 `FitProgress` 结构 + 流水线多处发射点 + 编解码（若持久化）。

**方案 B：静态布局 + 阶段标签（改动较小，推荐先行）**
- `api` 仅暴露 `describe_joint_layout(project) -> JointFitLayout`：拟合开始前一次性给出
  参与数据集与共享参数清单。
- 进度面板在联合运行时固定展示这份布局（"本次联合拟合：数据集 A/B/C，共享参数 X/Y"），
  阶段进度仍用现有 `stage` 文本，不逐步归属到单个数据集。
- 优点：不动 `FitProgress`、不碰流水线发射逻辑，只加一个只读查询函数；对用户已解决
  "在优化什么"的核心困惑。缺点：不显示逐步的"当前子问题"。

### TDD 切入点
- 方案 B：`describe_joint_layout` 对一个联合项目返回正确的数据集集合与共享参数名（纯函数，
  单测直接断言）；GUI 联合运行时进度面板文本包含这些名字。
- 方案 A：额外覆盖 `_emit` 在各阶段填的 `scope` 正确；`FitProgress` 往返编解码。

### 风险
- 方案 A 触及热路径的进度发射与可能的序列化，需保证不破坏现有独立模式归属与
  `FitProgress` 单调性契约（`completed <= total`、`best_objective` 非 NaN）。
- 联合"共享参数"的准确表述需与算法语义核对（`joint_sharing.py`），避免 GUI 措辞误导。

### 验收标准
- 联合拟合开始后，进度区明确列出参与数据集与共享参数（至少方案 B 的静态布局）。
- 独立批量模式的逐数据集归属不受影响。
- 新增 `api` 查询/字段有单测；`FitProgress` 契约与复杂度门禁不退化。

---

## #10.2 / #10.3 自动保存与项目历史版本（C 类）

### 定位
这两项超出"纯 GUI"与"简单 B 类字段"，涉及产品策略与新的持久化能力，归为 C 类，
**需用户先做产品决策**再进入设计与实施。此处仅列出待决问题与约束，不预设方案。

### #10.2 自动保存草稿 / 定期提示

> **状态：已实现，无需再决策。** 实现位于 `src/xrr_fitter/gui/project/autosave.py`
> （`AutosaveController`）。本节原先记录的待决问题已由该实现回答，保留于此仅作
> 决策记录。

已落地的策略：
- 触发：定时，间隔 `AUTOSAVE_INTERVAL_MS = 120_000`（2 分钟），且仅在项目
  dirty 且已有保存位置时写。
- 存储：独立旁挂草稿 `<path>.autosave`，不写回原项目文件。
- 与 dirty 标记的关系：写草稿**不清** dirty；只有用户手动保存才清，并同时
  `discard_draft()` 删除草稿。
- 恢复入口：`MainWindow._offer_draft_recovery()` 在打开项目时检测残留草稿并询问；
  接受则以 dirty 状态载入，拒绝则删除草稿。磁盘上存在草稿即意味着上次会话非正常
  结束。
- `api` 影响：无新增边界，复用现有 `api.save_project` / `api.load_project`。

### #10.3 项目历史版本
待决问题：
- 版本模型：每次保存一个快照？里程碑手动打点？保留多少、如何裁剪（容量上限）？
- 存储形式：项目文件内嵌历史，还是旁挂版本目录？对现有项目编解码的兼容性？
- 恢复语义：回到历史版本是覆盖当前，还是派生新项目？

约束：
- 项目模型当前是单一不可变快照；引入历史需要新的 `api` 能力与编解码扩展，且必须
  保持旧项目文件可读（向后兼容）。
- 需明确历史与外部真实数据/签核证据的边界，避免把不该进版本库的数据纳入历史。

### 建议
先落地 #10.2 的最小安全形态（关闭前提示 + 可选定时草稿，原子写、独立草稿文件），
在验证用户实际需求后再评估 #10.3 的版本模型。两者都应在用户明确策略后单独立项。
