# GUI B 类改进设计提案

> **状态**：设计草案，等待用户决策后再实施。本文档不修改任何 `api` 边界或运行时代码，
> 仅记录跨越 `xrr_fitter.api` 边界的 GUI 改进的实施方案、取舍与验收标准。
>
> **背景**：GUI 优化的九大类改进中，纯 GUI（A 类）项已在
> `worktree-gui-progress-liveness` 分支逐项落地。本文档收录剩余需要触及
> `api` 边界或需产品决策的项目，供实施前评审。每项给出：动机、现状证据、
> 设计方案、边界影响、TDD 切入点、风险与验收标准。

---

## 分类总览

| 编号 | 项目 | 类别 | 边界影响 | 需要的前置决策 |
|------|------|------|----------|----------------|
| #12.2 | 暗色/高对比度主题切换 | B | `ProjectUiState` 加字段 + 编解码 | 主题是否随项目持久化 |
| #5.2 | 联合拟合进度归属 | B | 需 API 暴露联合问题结构 | 进度粒度与展示形式 |
| #10.2 | 自动保存草稿 / 定期提示 | C | 需 API 支持后台写盘 | 自动保存策略与存储位置 |
| #10.3 | 项目历史版本 | C | 需 API 支持版本快照 | 版本模型与容量上限 |

> A 类（纯 GUI）项不在本文档范围内，已直接实现并提交。

---

## #12.2 暗色 / 高对比度主题切换

### 动机
当前应用只有单一浅色主题（`src/xrr_fitter/gui/theme.py` 的 `apply_theme`）。长时间在暗
环境使用、或有视觉辅助需求（高对比度）的用户无法切换。这是无障碍与专业细节的常见诉求。

### 现状证据
- 主题在启动时全局应用一次：`src/xrr_fitter/gui/application.py:22` 调
  `apply_theme(application)`，作用于整个 `QApplication`。
- 主题令牌与样式表集中在 `src/xrr_fitter/gui/theme.py`，是单一浅色调色板。
- UI 偏好已有持久化先例：`ProjectUiState`（`src/xrr_fitter/model/project.py:135`）
  已持久化 `expert_mode`、`plot_tab_index`、`workspace_splitter_sizes` 等；
  编解码在 `io/project_codec.py:140`（`_ui_to_dict`）与 `:151`（`_ui_from_dict`）。
- 绘图诊断图用的是各自 `Figure`，不受 `QApplication` 样式表影响，需要独立的
  matplotlib 颜色处理（见下「绘图一致性」）。

### 为什么是 B 类
主题偏好若要"记住用户选择"，必须落在被持久化的模型里。这要求：
1. 在 `ProjectUiState` 增加 `theme_mode` 字段（`api` 边界内的模型变更）；
2. 在 `project_codec` 的 `_ui_to_dict` / `_ui_from_dict` 增加往返编解码；
3. `api` 暴露一个纯函数变体（如 `set_theme_mode(project, mode)`），与既有
   `set_expert_mode` 对称。
另外主题需要人工视觉验证（对比度、可读性、绘图配色），无法仅靠单测收口。

### 设计方案

**模型层（api 边界内）**
- `ProjectUiState` 新增 `theme_mode: str = "light"`，取值限定
  `{"light", "dark", "high_contrast"}`，在 `__post_init__` 校验（复用现有
  枚举/取值校验风格）。
- `api.set_theme_mode(project, mode) -> XrrProject`，返回替换了 `ui_state.theme_mode`
  的新不可变项目，与 `set_expert_mode`（`model/project.py`）完全对称。
- 编解码：`_ui_to_dict` 增加 `"theme_mode": value.theme_mode`；`_ui_from_dict`
  的允许键集合与构造参数同步增加。**向后兼容**：`_ui_from_dict` 对缺失 `theme_mode`
  的旧项目回落到 `"light"`（旧文件读取不报错）。

**主题层（纯 GUI）**
- `theme.py` 由「单一调色板」改为「按 mode 解析令牌」：
  `resolve_tokens(mode) -> ThemeTokens`，`apply_theme(app, mode)` 按令牌生成样式表。
- 三套令牌：light（现状）、dark、high_contrast。保持相同的 objectName 选择器，
  只改颜色令牌，避免布局回归。

**绘图一致性（关键风险点）**
- matplotlib `Figure` 不吃 Qt 样式表。需要在 `diagnostics.py` 的绘图入口按当前
  mode 设置 `figure.set_facecolor` / `axes` 前景色 / 文本色 / 网格色。
- 建议：新增 `plot_palette(mode) -> PlotPalette`，绘图函数从 palette 取色而非硬编码。
  当前硬编码色（如 `#8A8A8E` 空状态灰、`#009E73` 预览绿、`#E69F00` 拟合范围橙）
  需要在暗色下替换为对比度达标的对应色。
- 这一步是本提案工作量与风险的主要来源，需要逐个诊断图人工核对。

**运行时切换**
- 主窗口「视图」菜单加主题子菜单（三选一，`QActionGroup` 互斥）。
- 切换时：`document` 走 `api.set_theme_mode` 得到新项目 → 触发 dirty →
  `apply_theme(app, mode)` 重刷样式表 → 通知绘图面板按新 palette 重绘。
- 启动/打开项目时读取 `ui_state.theme_mode` 并应用。

### TDD 切入点
1. 模型往返（RED）：`set_theme_mode` 改 mode → 编码→解码→ mode 保持；旧 payload
   缺字段 → 解码回落 `"light"`。（`tests/unit/io/test_project_codec.py`、
   `tests/unit/model/test_project_state.py`）
2. 校验（RED）：非法 mode 抛 `ValueError`。
3. GUI（RED）：菜单切到 dark → `apply_theme` 收到 `"dark"`；`document.project.ui_state.theme_mode == "dark"`。
4. 绘图 palette（RED）：`plot_palette("dark").background` 与 light 不同；空状态文本色
   在 dark 下满足最小对比度阈值（可用相对亮度差断言）。

### 风险
- **绘图配色**是最大工作量，且必须人工视觉验收；单测只能覆盖"取到了 palette"，
  覆盖不了"看起来对"。
- 硬编码色散落在多个绘图函数（`reflectivity.py`、`sld.py`、`diagnostics.py`），
  需一次性收敛到 palette，否则暗色下会有刺眼的残留浅色元素。
- 高对比度模式的准确定义（对比度目标、是否加粗、是否改焦点环）需产品确认。

### 验收标准
- 三种主题均可从菜单切换并即时生效（含所有诊断图）。
- 主题随项目保存/打开往返一致；旧项目文件打开默认浅色且不报错。
- 暗色/高对比度下无残留浅色控件或低对比度文本（人工核对清单逐项签核）。
- 模型/编解码/校验单测全绿；复杂度门禁不退化。

---

## #5.2 联合拟合进度归属

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
待决问题：
- 触发策略：定时（每 N 分钟）？按操作数？仅"关闭前提示"？
- 存储位置：写回原项目文件，还是独立的 `.autosave` 草稿？后者需明确恢复入口与冲突处理。
- 与"未保存标记（`[*]`，已实现）"的关系：自动保存后是否清 dirty？
- `api` 影响：需要一个安全的后台写盘路径，且不得与用户手动保存竞争写同一文件。

约束：
- 不得静默覆盖用户文件；自动产物与正式保存必须可区分、可回滚。
- 后台写盘不得阻塞 UI，也不得在写一半时崩溃留下损坏文件（需原子写：临时文件 + rename）。

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
