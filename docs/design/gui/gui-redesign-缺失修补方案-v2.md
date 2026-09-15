# GUI 重设计 · 缺失元素修补方案 v2（审计修订版）

> 历史设计资料（2026-08 至 2026-09），于 2026-09-15 整理入库。正文中的计划、状态和命令保留原编制时点语境，不是当前待办。当前完成情况见[设计追溯](../../architecture/design-frames-traceability.md)和[验收台账](../../architecture/plan-v2-acceptance-ledger.md)。

> 配套文档：`docs/design/gui/xrr-fitter-gui-redesign.html`（设计稿）、`docs/design/gui/gui-redesign-缺失修补方案.md`（v1，本文件的修订对象）
> 修订日期：2026-09-03
> 修订依据：对 v1 逐条现状核实（4 组只读 agent + rg/源码交叉验证）后重写。所有结论均带 `文件:行号` 证据。

---

## 0. 本次修订相对 v1 的变更摘要

v1 方向正确、分批合理，但多轮逐元素核实发现 **5 处必须修正 + 5 处遗漏补录（G23–G25 + A2/A3 复核定案的 G26/G27）+ 1 处本文档自身的误判撤销 + 1 处扩边界项（用户已确认纳入，落 §5.5）**，否则施工会踩空或越界。本版据此：

1. **撤销「剔除 G20」（本文档自身首轮误判，详见 §0.3）** —— 首轮我以 `guidance/panel.py` 的 `_StepIndicator` 为由剔除 G20，**判断错误**：`_StepIndicator`（`panel.py:84`「Horizontal step indicator」）是**引导模式横向 4 点指示器**，与设计稿要求的**专家模式左栏纵向六段管线 stepper（导入→结构→参数→拟合→结果→导出）** 不是同一物；后者全仓库零实现。G20 恢复为真实缺失，且与 visual-diff 报的「A1 IA 主脊」是同一物（不双计）。
2. **修正 4 处致命路径错误** —— v1 的 G20/G22/C5/D6 把改动落到 `gui/navigation/panel.py`；该文件与整个 `navigation/` 子包**不存在**。真实路径是 `gui/guidance/panel.py`。
3. **缩小 G5 范围** —— v1 称「状态栏无置信度标记」，事实是 `fitQualityStatus`/`fitQualityDot` 已是置信度徽章（`chrome.py:390-412`）。缺的只是「模式文字（引导/专家）」。
4. **重定级 G16 导出**（v1 批次编号曾叫「B1」，**勿与 §5.5 的布局项 B1 混淆**，本版全文「B1」一律指固定三区布局）—— 产物集是固定的（`services/exports.py:155-160`，只有 `.ort` 可开关）；**SVG 从零新增**，CSV 序列化器已存在（`io/export_tables.py:667 parameters_csv_bytes`）但**未接入产物集**（详见 §0.2 复核）。这是后端 + 逐位复现门禁问题，不是「换勾选框」。
5. **重定级 C3（G9）** —— 模型层没有「仅范围」第三态（`model/parameters.py` 只有 `locked`/`constrained` 两布尔）。需改模型 + 服务层，非纯 delegate。
6. **修正 B3（G21）前提** —— 相关矩阵热图、SLD 可信带**已存在**（`plots/sld.py`），且 `plots/diagnostics.py` 已有双层 QTabWidget。B3 是「整合进既有 tab 体系」，不是「从零搭容器」。
7. **补 3 处遗漏条目** —— 新增 **G23 帧④三 tab（含全新「目标值轨迹」组件）**、**G24 导入高级选项三控件**、**G25 角度约定检测 API 缺口**。

因此本版清单从 v1 的「G1–G22」调整为 **22 项真实缺失（含 G20，见 §0.3 撤销剔除）+ 5 项新增（G23–G25 遗漏补录，G26–G27 由 A2/A3 复核定案）= 27 项**。其中 visual-diff 报的「A1 纵向管线 stepper」经去重即 **G20，已并入清单**（同一物不双计，见 §0.3 ①）；A3 的「各数据集目标值表」并入 **G12** 不双计；「B1 固定三区布局」不属表现层缺失而是**架构级扩边界项，用户已确认纳入**，单列 §5.5（独立高风险主项，见 §0.3 ②）。

---

## 0.1 评审决议（2026-09-03，用户确认）

针对 §5 Group 3「须开工前确认」的四项，已获用户逐项拍板：

| 决议项 | 结论 | 影响 |
|---|---|---|
| Group 3 实现范围 | **四项全部纳入**：G10+G11 暂停/跳过、G16 CSV+SVG、G9 参数三态、G25 角度自动检测。无一砍除或延后。 | Group 3 从「待评审」升级为「已确认待施工」；每项仍须先出后端契约设计再落代码。 |
| G16 导出接口形态 | **改为 `formats` 列表参数**（非 `include_ort` 式逐格式布尔叠加）。 | `export_result` 签名变更：不是加参数，而是重构入参为格式集合。**所有既有调用点 + 架构测试须同步迁移**，向后兼容需在迁移层显式处理，不能靠默认值糊过去。 |
| G10 暂停语义 | **真·挂起/恢复**（进程内 pause/resume，非取消后从 checkpoint 续跑的轻量替代）。 | 必须在 `fit/` + `services/workers.py` 新增可挂起点；**指纹漂移风险最高**，须保证不动 `fit/` 层 dataclass 字段且 `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 通过。 |

**当前阶段边界**：本轮仅更新方案文档，**不落地任何代码**。四项的后端契约设计（签名、事件模型、指纹影响验证）待单独排期。

---

## 0.2 独立验证结果（2026-09-03，命令级复核）

四项决议 + 遗漏主张经本人 `rg` 逐条复核（非子代理转述），钉死如下证据。**多处推翻了子代理与前一版转述**：

| 断言 | 复核结果 | 处置 |
|---|---|---|
| skip / pause / resume 在 src 是否零命中 | ✅ `rg 'skip_stage\|skip_phase\|def pause\|is_paused\|def resume\|suspend' src/xrr_fitter` **零命中** | 确认 G10/G11 需从零建后端 |
| 参数是否只有两态 + 是否有指纹风险 | ✅ `ParameterSetting.locked: bool`（`model/parameters.py:136`）为唯一态；`soft_range` 已是 prior（`:16`）；`ParameterSetting`/`ParameterDefinition` 大量流入 `fit/parameters.py` 阶段编译 → **指纹漂移风险真实** | G9 硬约束坐实 |
| G16 是否「已是 formats 列表」 | ❌ 推翻：签名是 `include_ort: bool = False`（`exports.py:198`）；**无** `formats` / `ReportFormat` / SVG 任何命中 | 「重构为 formats」决议仍成立 |
| G16 是否「CSV 完全没实现」 | ❌ 修正：CSV 序列化器**已存在**（`io/export_tables.py:667 parameters_csv_bytes`），仅未接入产物集（`exports.py:155-160`）；SVG 才是从零新增 | 见 §5.2 修正后措辞 |
| G23 第三 tab 是否「日志面板」 | ❌ 推翻：HTML L785 原文 `<span class="tab">目标值轨迹</span>`；frame④（L743-884）全区**零** `日志`/`log` 命中 | 保持 §6.1 原判（目标值轨迹），不改 |
| ⌘K 命令面板 / 全局搜索 / 通知中心 / 参数过滤 / 候选多选对比 / 导出 toast（**我前一轮误转述为 visual-diff 主张**） | ❌ 逐一核验 HTML **全不成立**：均**零命中**；「对比」仅出现在无障碍要求「对比度 4.5:1」；「锁定」是参数锁定徽章（`.lock` L446/L704，已由 G9 覆盖），非独立层可见性控件 | 这 6 项确不存在，**但它们并非 visual-diff 的真实主张**（我前一轮转述错误）。visual-diff 实际报的是 A1 纵向 stepper / A2 / A3 / B1–B6 —— 已在 **§0.3** 逐条复核并纠正 |

---

## 0.3 第二轮验证：三处反转 + IA 骨架遗漏（2026-09-03，均本人 grep 复核）

首轮 §0.2 之后，三个后端/视觉子代理回传，经本人独立 grep 复核，**推翻了包括本文档自身在内的三处结论**：

| 反转项 | 旧结论（含本文档） | grep 复核后的硬事实 | 处置 |
|---|---|---|---|
| **冻结指纹金标测试名** | 全文 5 处引用 `test_frozen_stage_search` | ❌ **该测试不存在**：`rg` 确认它只出现在**注释**里（`parameters.py:214`、`test_checkpoint.py:214`）。真实金标是 `tests/unit/fit/test_checkpoint.py:172 test_prior_defaults_do_not_perturb_frozen_fingerprints`（硬编码 fingerprint） | 全文 5 处已改为真实名；记忆 `xrr-checkpoint-fingerprint-drift` 经核已用真实名（`:19-20`）并已注明旧名弃用，无需再改 |
| **G11 跳过阶段语义** | 「阶段跳过入口」当作待加 UI + 待定义语义 | ❌ omit-middle **架构不可行**：`pipeline.py:118 _stage_candidates` 用 `next(... if v.stage==parent)`，父阶段缺失即 `StopIteration`；`:210` 显式 raise「stage B requires committed stage-A starts」。唯一安全语义 = **early-stop** | §5.1 已锁定为 early-stop |
| **G20 管线步骤样式（本文档自己剔除的）** | 首轮以 `_StepIndicator` 为由**剔除 G20** | ❌ 剔除错误：`_StepIndicator`（`panel.py:84`「Horizontal step indicator」）是引导模式横向 4 点指示器；**专家模式左栏纵向六段管线 stepper 零实现**（`rg "管线\|pipeline\|stepper" src/xrr_fitter/gui/` 仅命中 `main_window.py:207` 一句无关注释） | G20 恢复为真实缺失（见 §1 表） |

### 骨架层遗漏：分清「可并入清单」与「须单独拍板」两类

visual-diff 报了 A1（纵向 stepper）/ A2 / A3 / B1–B6。经本人核对，去重后归为两类，**关键是别把它们混为一个模糊的「IA 重构」**：

**① A1 = G20，同一物，已并入清单（不双重计数）**
- A1「纵向六段管线 stepper」与刚恢复的 **G20 是同一个缺失**（设计稿每帧左栏，HTML L372/L631/L772/L894，理据 L312-333）。visual-diff 的 A1 只是从「IA 主脊」角度重述 G20。
- **这个 stepper 组件本身可以作为一个 widget 装进现有导航 dock，不必拆 dock**——因此归 G20，留在清单内（Group 1/2 表现层），**不撞架构契约**。§4.7 的 G22 不确定度子管线正是挂在它下面。

**② B1 = 固定三区 vs dock 布局（用户已确认纳入，详见 §5.5）**
- 设计稿是固定三栏 `264px 管线导航 | 自适应画布 | 340px 检视器`（CSS L104），且明确声明「布局骨架…未沿用现有 dock 布局」（L333/L1156）。
- **grep 复核纠正了本项定性**：仓库**当前 dock 布局本就是三列**（`window_layout.py:1`「Stable three-column MainWindow」，左 data/structure｜中 guidance/plot stack｜右 parameters/fit/results tab），且**曾经就是 QSplitter 固定三区、后主动迁到 dock**（`workspace.py:33-38` docstring）。**真正差异只剩「可拖拽 vs 固定不可拖拽」一点**，不是「缺三列」。
- 用户已拍板**纳入为独立高风险主项**。§5.5 给出两条路线：路线 1（去掉 dock 的 Movable/Floatable + 钉死宽度，不动 dock_state 契约，风险等同 Group 1，**推荐先做**）；路线 2（真换回 QSplitter，撤销迁移、撞 `api.set_dock_state` 金标，blast radius 最大）。**本轮不落代码。**

**③ A2/A3 已逐 panel 复核，定案纳入清单（原「存疑」解除）**：

- **A2 结构诊断相关性提示（HTML L721-725，`insp-sec`/`h`「结构诊断」/`hint` 🔎「厚度 d 与密度 ρ 可能相关」）** → **真实缺失，纳入为 G26**。`structure/panel.py` 只有 init 按钮 + `StructureEditor`，`editor.py` 无 hint 渲染。但底层 `analyze_structure`/`StructureEvidence`（`services/` + `model/analysis.py`）**已能算出相关性证据**，只是未在结构面板渲染 → 归 **Group 2**（接既有后端，表现层补渲染）。
- **A3 拟合实时指标明细（HTML L826-846，`insp-sec`/`kv` 行：当前阶段/全局目标J/迭代次数/函数评估/接受率/步长 + 各数据集目标值 `table.grid`）** → **部分实现，缺失部分纳入为 G27**。`ProgressView`（`fitting/progress.py`）已有阶段名（`stage_label:229`）、全局 J（`detail_label:233-236`，best=best_objective）、耗时（`meta_label:263` 覆盖 stage/J/time）；**缺 迭代次数/函数评估(nfev)/接受率/步长 四项指标**——`FitProgress`（`model/progress.py:66-97`）字段无承载它们的项。各数据集目标值 `table.grid` **就是 G12**（已列，§5.6），不重复计。故 G27 = 那四项标量指标，需扩 `FitProgress` 字段 → 归 **Group 3**（与 G12 同为 progress 载体扩展，不触发指纹漂移，见 §5.6）。

**结论修订**：清单从「25 项」增补 A2/A3 为 **27 项（新增 G26 结构诊断相关性提示、G27 拟合实时指标四项）**；G20/A1 仍同一物不双计，A3 的 per-dataset 表并入 G12 不双计。**B1（固定三区）经用户确认为独立高风险主项，落 §5.5**。清单外无其他悬置项。

---

## 1. 现状核实结论表（v1 每条的准确性判定）

| # | v1 判定 | 核实结论 | 证据 |
|---|---|---|---|
| G1 引导/专家 segment | 缺失 | **准确** | `guidanceModeAction` 在「视图」菜单 `chrome.py:210-219`；工具栏 `_install_toolbar` `chrome.py:99-116` 不含它 |
| G2 独立/联合 segment | 缺失 | **准确** | `batch_selector` 是 `QComboBox`（`fitting/panel.py:66`），面板内，仅专家模式可见（:295） |
| G3 明暗切换 | 缺失 | **准确** | `chrome.py` 无 dark/light toggle；仅 `import theme` + `set_status_kind` 调用 |
| G4 画布 modebar | 缺失 | **准确（措辞收紧）** | 无自定义浮动工具条；但 pyqtgraph 面板自带右键缩放/复位。表述应为「无统一 平移/框选/复位/选区 工具条」 |
| G5 状态栏置信度/模式 | 「无置信度标记」 | **部分准确 → 缩小范围** | 置信度徽章已存在 `fitQualityStatus`/`fitQualityDot`（`chrome.py:390-412`）；**仅缺模式文字** |
| G6 引导层预览 | 缺失 | **准确** | `guidance/panel.py` 用静态 `layer_stack` 图标，非层堆叠预览 |
| G7 双 CTA | 缺失 | **准确** | 引导卡是单 primary + ghost 前进/后退，非 primary+secondary |
| G8 层拖拽 | 缺失 | **准确** | `structure/editor.py:137-138` 仅 up/down 按钮，无 DragDrop |
| G9 参数三态 | 缺失 | **准确（低估：需模型改动）** | `parameters/table.py` 两态 checkbox；`model/parameters.py` 无 range_only 态 |
| G10 暂停 | 缺失 | **准确（需后端）** | worker 仅 `cancel`/`force_stop`（`services/workers.py:299,303`）；全仓库 `pause`/`suspend` 零命中 |
| G11 跳过阶段 | 缺失 | **准确（需后端）** | `fit/stages.py` `STAGE_ORDER=("A".."E")` 固定；切片函数是 resume 用；无 `skip_stage` |
| G12 各数据集目标值表 | 缺失 | **准确** | `fitting/panel.py` 无 per-dataset objective table |
| G13 联合横幅 | 缺失 | **准确** | `fitting/` 无 joint banner |
| G14 MCMC 直方图 | 缺失 | **准确** | `UncertaintyView` 仅文本，无直方图 |
| G15 Profile Δχ²=1 线 | 缺失 | **准确** | `plots/sld.py:_draw_profiles` 无参考线 |
| G16 导出多格式 | 缺失 | **准确（低估：SVG 从零新增；CSV 序列化器已存在但未接入产物集）** | `services/exports.py:155-160` 固定产物，仅 `.ort` 可开关；`io/export_tables.py:667 parameters_csv_bytes` 存在但未接入 |
| G17 manifest 预览 | 缺失 | **准确** | `ExportManifest` 存在；dialog 仅导出后 summary，无导出前预览 |
| G18 角度约定切换 | 缺失 | **准确** | `import_dialog.py:186-189` 仅 mono/mixed Kα radio |
| G19 批量预览表 | 缺失 | **准确（后端已就绪）** | 无 QTableWidget；但 `preview_import_batch`/`inspect_sources` 已存在且 panel.py 已调用 |
| G20 管线步骤样式 | 缺失 | **准确（本轮撤销剔除，见 §0.3）** | `_StepIndicator`（`guidance/panel.py:84-146`）是**引导模式横向 4 点**指示器，非专家模式左栏**纵向六段管线** stepper；后者全仓库零实现（`rg` 仅命中 `main_window.py:207` 无关注释）。G20 = 该纵向 stepper，真实缺失 |
| G21 四子选项卡 | 缺失 | **准确（前提需修正）** | `UncertaintyView` 是单 `QPlainTextEdit`；但相关热图/SLD 带已存在于 `plots/sld.py`，`diagnostics.py` 已有双层 tab |
| G22 不确定度子管线 | 缺失 | **结论对，路径错** | 结论成立；文件应为 `guidance/panel.py`，非 `navigation/panel.py` |

**新增（v1 遗漏）：**

| # | 设计元素 | 现状 | 证据 |
|---|---|---|---|
| G23 | 帧④拟合面板三 tab（进度 / 实时反射率 / 目标值轨迹） | 缺失；面板是纵向 QVBoxLayout，无 tab | `fitting/panel.py:65-111`；`目标值轨迹`/收敛曲线组件全仓库零实现 |
| G24 | 导入高级选项（列分隔符下拉 / 跳过表头行 / 强度列取对数） | 缺失；`ColumnMappingDialog` 仅索引列映射 | `data/import_dialog.py:76-148` |
| G25 | 角度约定自动检测 API（`detect_angle_convention`） | 缺失；设计稿能力表提到但 `api` 无此符号 | `api.py` 无 `detect_angle`/`convention` 定义 |
| G26 | 结构诊断相关性提示（结构面板内「厚度 d 与密度 ρ 可能相关」等 hint） | 缺失渲染；后端 `analyze_structure`/`StructureEvidence` 已能算证据但结构面板不展示 | HTML L721-725；`structure/panel.py`/`editor.py` 无 hint 渲染；`services/` + `model/analysis.py` 有 `analyze_structure`/`StructureEvidence` |
| G27 | 拟合实时指标四项（迭代次数 / 函数评估 nfev / 接受率 / 步长） | 缺失；`ProgressView` 仅有阶段名/全局 J/耗时，无这四项 | HTML L826-846；`fitting/progress.py:229-263`；`FitProgress`（`model/progress.py:66-97`）无承载字段 |

---

## 2. 重新分批：按「改动面」而非优先级

审计的核心发现是——决定能否开工的轴是**改动触达哪一层**，不是优先级。据此重排为 4 组：

- **Group 1 · 纯表现层**：只碰 `gui/*`，不需后端/模型/API 改动。风险最低，可立即 TDD。
- **Group 2 · GUI 层接既有后端**：后端能力已就绪，只需在 GUI 接线（可能需破坏性确认或整合既有组件）。
- **Group 3 · 需跨层 / 后端 / 架构评审**：触达 `fit/`、`services/`、`model/`、`api`，可能影响 checkpoint 指纹或逐位复现门禁。**开工前须单独确认。**
- **Group 4 · 遗漏补录**：v1 未列，需先纳入清单再排期。

---

## 3. Group 1 — 纯表现层（可立即施工，风险低）

| 条目 | 目标 | 涉及文件 | 备注 |
|---|---|---|---|
| G3 明暗切换 | 工具栏加主题切换按钮 | `gui/chrome.py`（连 `theme.apply_theme`） | 确认 `theme` 模块暴露 toggle 入口 |
| G5 模式文字 | 状态栏补「引导/专家」模式文字（**置信度徽章已有，勿重复造**） | `gui/chrome.py:_install_status_bar` | 只加 mode label |
| G6 引导层预览 | 引导「结构」卡内嵌只读层堆叠 | `gui/guidance/panel.py` | 可复用 `structure/stack.py` 的 `StackView`（只读模式） |
| G7 双 CTA | 引导卡底改 primary + secondary 双按钮 | `gui/guidance/panel.py:190-254` | 保持引导-专家共享按钮路径（§8.6） |
| G13 联合横幅 | 联合模式下拟合面板顶部说明横幅 | `gui/fitting/panel.py` | 按 `batch_mode` 切可见性 |
| G4 modebar | 图表区右上浮动工具条（平移/框选缩放/复位/选区） | ~~`gui/plots/` 新增 `modebar.py`~~ → **交付落在既有 `gui/plots/interactions.py`**（`MODE_SPECS` + `_install_mode_buttons`，与既有 ViewBox 模式表同住，`plots/` 下无 `modebar.py`） | 包 pyqtgraph 原生 ViewBox 模式；**qz⁴R/残差面板只读**（`interactions.py:566-567`），modebar 需按面板启用 |

---

## 4. Group 2 — GUI 层接既有后端（后端已就绪）

### 4.1 G1 + G2 + C1：命令栏 segment 控件

- **目标**：工具栏新增两组 segmented control（引导/专家、独立/联合）。
- **文件**：新建 `gui/widgets/segmented_control.py`；`chrome.py:_install_toolbar` 嵌入；保留菜单入口作 keyboard 备选。
- **接线**：引导/专家 → `set_guidance_visible`（`main_window.py:255`）；独立/联合 → `api.set_batch_mode`。
- **⚠️ 破坏性确认（v1 遗漏）**：`api.set_batch_mode` 在 <2 数据集时抛 `ValueError`，且切换会**清空 results 和 selected_candidate_ids**（`services/projects.py:287-309`）。segment 直连会无预警丢结果——**必须先弹确认对话框**。
- **验证**：`tests/gui/test_chrome.py` 或新增 `test_command_bar`，含破坏性确认路径断言。

### 4.2 G17：导出 manifest 预览

- **目标**：导出对话框在导出**前**展示即将产出的文件清单预览。
- **文件**：`gui/export/dialog.py`（`ExportManifest` 已存在，`export_summary` `:64-71` 可复用渲染逻辑，但需从「导出后 summary」前移为「导出前预览」）。
- **验证**：`tests/gui/test_export_dialog.py` 加导出前预览断言。

### 4.3 G19 + G24：导入批量预览表 + 高级选项

- **目标**：批量导入展示文件-数据集预览表；新增列分隔符下拉 / 跳过表头行 / 强度列取对数三控件。
- **文件**：`gui/data/import_dialog.py`（`preview_import_batch`/`inspect_sources` 已就绪且 panel.py 已调用，仅缺 UI 呈现）；`ColumnMappingDialog:76-148` 扩展三个高级选项。
- **⚠️ 注意**：`import_dialog → io.xy` 是已批准例外（§8.4），扩展列解析时不得破坏该边界。
- **验证**：`tests/gui/test_data_import.py` 加批量预览 + 高级选项断言。

### 4.4 G21 + G14 + G15：不确定度四子选项卡（整合，非新建）

- **目标**：`UncertaintyView` 从单 `QPlainTextEdit` 改为承载四子页（相关矩阵 / Profile / SLD 可信带 / MCMC）。
- **⚠️ 前提修正（v1 错）**：相关热图（`plots/sld.py:draw_uncertainty`，imshow+colorbar）和 SLD 可信带（`_draw_bands`，16-84%/2.5-97.5% fill_between）**已实现**；`plots/diagnostics.py` **已有双层 QTabWidget**（`reflectivity_tabs` + `analysis_tabs`，后者含残差热图/候选比较/相关性/批量趋势）。**本项是把这些既有图整合进不确定度面板的 tab，不是从零搭容器；勿误伤 `analysis_tabs`。**
- **子项**：
  - G14 MCMC 直方图（P16/P50/P84 标线）——新增 `McmcHistogramPlot`，**必须放 `gui/plots/`**（numpy/matplotlib 依赖门禁 §8.1）。`api.McmcReport` 含 chain samples 足够驱动。
  - G15 Profile Δχ²=1 参考线——`plots/sld.py:_draw_profiles` 加 `axhline`。
- **验证**：`tests/gui/test_uncertainty_dialog.py` 加 tab 切换 + 直方图 + 参考线断言。

### 4.5 G8：结构层拖拽

- **目标**：层列表支持拖拽排序，补充现有 up/down。
- **文件**：`gui/structure/editor.py`（启用 InternalMove + drag handle delegate）；确认 reorder 与 `api` 层顺序语义一致。
- **验证**：`tests/gui/test_structure_editor.py` 加 drag reorder 断言。

### 4.6 G12：各数据集目标值表 → **已确认降级 Group 3（见 §5.6）**

- **前置调研已出结论（触发降级）**：`rg 'per_dataset|dataset_objective|individual' src/xrr_fitter/fit src/xrr_fitter/services` **零命中**；三处载体均只携标量——`FitProgress`（`model/progress.py:66-97`）单 `dataset_id` + 单 `best_objective`，`FitStageSummary`（`model/fitting.py:523-548`）单 `best_objective`，`FitCandidate`（`:412-432`）单 `objective`/`ranking_objective`。**fit 层不上报分数据集目标值**，无法纯前端从既有回调取得 → **本项按 §4.6 既定规则降级到 Group 3，详见 §5.6。**

### 4.7 G22 + C5：不确定度方法子管线

- **目标**：「结果」步骤下展开不确定度方法子管线（相关矩阵→自助抽样→Profile→MCMC），带完成/进行中/未达状态。
- **⚠️ 路径修正**：文件是 **`gui/guidance/panel.py`**（复用已实现的 `_StepIndicator` 三态渲染），**不是 v1 写的 `gui/navigation/panel.py`（该文件不存在）**。
- **验证**：`tests/gui/test_pipeline_nav.py` 或对应引导面板测试加子管线断言。

### 4.8 G26：结构诊断相关性提示

- **目标**：结构面板内渲染结构诊断提示（设计稿 HTML L721-725：`insp-sec`/`h`「结构诊断」/`hint` 🔎「厚度 d 与密度 ρ 可能相关」等）。
- **⚠️ 定性（A2 复核定案，§0.3 ③）**：**后端已就绪，仅缺渲染**——`analyze_structure`/`StructureEvidence`（`services/` + `model/analysis.py`）已能算出相关性证据；`structure/panel.py` 只有 init 按钮 + `StructureEditor`，`editor.py` 无 hint 渲染。故归 Group 2（接既有后端，表现层补渲染），不触后端/指纹。
- **文件**：`gui/structure/panel.py`（新增诊断提示区，调 `analyze_structure` 取 `StructureEvidence` 渲染）。
- **前置确认**：核实 `analyze_structure` 是否经 `api` 暴露（gui 只能 import `api`+`gui.*`，§8.1）；若仅在 `services/` 未导出，需先确认 API 暴露路径。
- **验证**：`tests/gui/test_structure_editor.py` 或新增结构面板测试加诊断提示断言。

---

## 5. Group 3 — 跨层 / 后端 / 架构（已确认，待施工）

> 以下每项都超出纯表现层边界，触达 `fit/`/`services/`/`model/`/`api`。**四项均已获用户确认纳入实现范围（见 §0.1）**。本轮仍不落地代码——每项须先产出「后端契约设计 + 指纹/复现门禁影响验证」，再进 TDD。

### 5.1 G10 暂停 + G11 跳过阶段【已确认 · 真·挂起/恢复｜G11 交付按设计稿字面「跳过本阶段」，见下第 3 条】

- **决议**：采用**真·挂起/恢复**语义（进程内 pause/resume），不退回「取消 + checkpoint 续跑」的轻量替代。
- **现状（签名级复核）**：取消基于 spawn-context 的 `multiprocessing.Event`——`services/workers.py:329 cancellation = context.Event()`，`:330` 作为 `context.Process(target=..., args=(request, queue, cancellation))` 的进程参数传入。**`cancelled` 不是 dataclass 字段、也不是 keyword 参数**，而是把 `cancellation.is_set`（bound method，`Callable[[], bool]`）作为**位置参数**沿纯 handler 传递（`workers.py:77-82 fit_worker_handler`、`:108-114 automatic_worker_handler`、`:125-134 mcmc_worker_handler`）。`OperationJob.cancel()`（`:299-301`）set Event，`force_stop()`（`:303-309`）set 后 terminate。`workers.py` 本身无探针，仅 handler 返回后查 `result.cancelled`（`:83`、`:115`）；真正的探针在 fit 层：`pipeline.py:406-411 _raise_if_cancelled`（`if cancelled is None or not cancelled(): return` 否则 `raise SearchCancelled`），调用于 `:430`/`:447`；阶段内联 `_poll(cancelled)` 在 `stages.py:235-237`、`local_search.py:77-79`、`joint_solvers.py:29-31`、`global_search.py:532-533`。
- **G11「跳过」语义已被架构锁死 = early-stop，非 omit-middle（硬结论）**：`_stage_candidates`（`pipeline.py:117-120`，`next(v for v in reversed(state.summaries) if v.stage==stage)`）读父阶段产物，父阶段缺失即 `StopIteration`；阶段循环 `for stage in remaining:` 起于 `pipeline.py:192`（`remaining` 在 `:180` 新跑 `("A","B","C","D","E")` 或 `:188` resume `plan.remaining_stages`），A 分支 `:193-207`、B `:208-219`、C/D `:220-242`、E `:243-253`，阶段收尾 `state=state.append(outcome)` `:254` → `_publish_checkpoint` `:256` → 循环后 `return _result(request, state, seeds)` `:257`。字面「省略中间阶段」需为每阶段定义「父缺失时回退输入源」= 重构阶段图，超出「修补」范畴。**故 G11 锁定为 early-stop（在某阶段后提前收尾、把当前 best 当最终结果发布），天然安全、零指纹风险**。注意 `remaining_stages`（`stages.py:220-229`）是 resume 的严格后缀机制，不是 skip，勿复用。
- **交付偏离本条锁定（2026-09-03，需知情）**：G11 最终**按设计稿字面语义交付「跳过本阶段」**（帧④ HTML L850 的按钮字面就是 `⏭ 跳过本阶段`，旁边才是 `⏸ 暂停`），不是上一条锁定的 early-stop：`services/workers.py:skip_stage` 作废当前这一个阶段，搜索接着往下跑，已跑完的阶段与候选全部保留；阶段边界出口是 `fit/pipeline.py` 的 `except StageSkipped`。让它可行的改动是**放宽上一条认定为「须重构阶段图」的那处查找**——`pipeline.py:118-127`：请求的那一阶段没有 summary 时，父代回退到**最近一个真正跑完的阶段**（`state.summaries[-1]`），不再 `StopIteration`。这是 `fit/` 层查找规则的改动，**超出下一条「签名级落点」所描述的范围**，故单独记此条。锁定的 early-stop 仍在，但降为退化分支：阶段 A 被跳过时一个候选都没有，`pipeline.py:312-313` 直接 `break` 落到既有 `return _result(request, state, seeds)`、绕开 `SearchCancelled`——与下一条 (b) 的设计一致；界面上没有单独的「提前收尾」控件，设计稿也没画。风险收敛面：该回退只在「某一阶段被跳过」之后才可能触发（正常跑每阶段先 `state.append(outcome)` 再被下一阶段读到，resume 的 `remaining_stages` 是严格后缀，父代都在）。**下一条的硬约束照守：没有给任何 `fit/` 层 dataclass 加字段**，`test_prior_defaults_do_not_perturb_frozen_fingerprints` 全程绿。覆盖：`tests/unit/fit/test_resume.py:367/381/408`、`tests/unit/services/test_workers.py:312`（开关一次性，下一阶段不会连着被跳）、`tests/gui/test_fit_progress.py:855`。完整记录见 `docs/architecture/design-frames-traceability.md` §2.2。
- **需改（签名级落点）**：`services/workers.py` + `fit/`——(a) pause：在 `workers.py:329` 旁与 `cancellation` 并列新建第二个 `context.Event()`（pause），加入 `:330` 的 `Process` args，把一个 `paused`/`wait` callable 沿现有 `cancelled` 的**位置参数链**传入 handler；阻塞式探针可与 `cancelled` 探针共址（`pipeline.py:430`/`:447` 及 `stages.py:235-237`/`local_search.py:77-79` 的 `_poll`/`_raise_if_cancelled` 一线），恢复时不重启进程；(b) early-stop：加 request-stop 探针，插在阶段边界 `pipeline.py:256`（`_publish_checkpoint` 后）与 `:192`（`for` 顶）之间，命中则 `break` 跳出循环、落到既有 `:257 return _result(request, state, seeds)` 用已累积 `state` 正常封装，**绕开 `SearchCancelled` 异常路径**。
- **硬约束（最高风险）**：pause/early-stop 全部照抄 `cancelled` 的传参方式——**进程间 `Event` + bound-method 位置参数**（`cancelled` 就是这样传的，既非 dataclass 字段也非 keyword 参数）；**绝不给任何 `fit/` 层 dataclass（`FitEvaluationContext`/`FitSearchRequest`/`FitCheckpoint`/`ParameterDefinition`/`FitStageSummary`/`FitCandidate`）加字段**——否则 checkpoint 指纹漂移（记忆 `xrr-checkpoint-fingerprint-drift`，`tests/unit/fit/test_checkpoint.py:172 test_prior_defaults_do_not_perturb_frozen_fingerprints` 硬编码 fingerprint 会抓）。
- **UI 区分**：Escape 已绑 cancel（`fitting/panel.py:89-92`），且 cancel 有可见延迟（worker 无法放弃正在评估的 seed）。pause 按钮必须与 cancel 在语义与视觉上明确区分。
- **验证**：`tests/gui/test_fit_progress.py` 加 pause/resume/early-stop 场景 + `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 必须保持绿（运行须 `PYTHONPATH=src`，xrr_fitter 未装成包）。

### 5.2 G16 导出 CSV / SVG【已确认 · formats 列表参数】

- **决议**：`export_result` 入参**重构为 `formats` 列表**（格式集合），不是在 `include_ort` 之外继续叠加逐格式布尔。
- **现状（签名级复核）**：现签名 `export_result(result: XrrProject | ProjectFitResult, output_dir: str | Path, *, include_ort: bool = False) -> ExportManifest`（`exports.py:194-199`，keyword-only）。每数据集固定 6 产物 `_dataset_artifacts`（`exports.py:153-161`：`fit_result.xlsx`/`fit_result.json`/`fit_overview.png`/`sld_profile.png`/`residuals.png`/`run_log.txt`），`include_ort` 时追加 `fit_result.ort`（`:162-171`）；根级 `_root_artifacts`（`:175-191`）产 `compatibility_summary.xlsx`，多数据集追加 `batch_summary.xlsx`/`parameter_trends.png`。**产物集从不产 CSV/SVG**；CSV 序列化器 `io/export_tables.py:667 parameters_csv_bytes(context: DatasetExportData) -> bytes` 已存在但**未被 `_dataset_artifacts` 引用**（当前固定集不含 `parameters.csv`），SVG 则需从零新增。
- **调用点全清单（签名变更须同步迁移）**：
  - 定义 `exports.py:194`
  - 生产调用：`cli/commands.py:114 include_ort=arguments.ort`、`gui/export/dialog.py:155 include_ort=include_ort`
  - 测试调用（**含硬编码 include_ort 断言**）：`tests/integration/test_export_workflow.py:90`、`tests/support/approved_workflow_capture.py:63`、`tests/unit/services/test_exports.py:92/126/167/189(include_ort=False)/205(include_ort=True)`
  - 架构金标：`tests/architecture/test_public_api.py:166` 硬编码 `"(result: 'XrrProject | ProjectFitResult', output_dir: 'str | Path', *, include_ort: 'bool' = False) -> 'ExportManifest'"`，`:189 "export": ("export_result",)` 分组断言——**签名一改，此两行金标必须同步改**。
- **需改**：
  1. `export_result` 签名从 `include_ort: bool` 改为 `formats: <格式集合>`；上列**全部调用点 + `test_public_api.py:166` 金标**同步迁移。
  2. 向后兼容在迁移层显式处理（旧 `include_ort=True/False` 映射到对应 formats），**不靠默认值糊过去**；`test_exports.py:189/205` 的 include_ort 断言须改写为 formats 语义。
  3. CSV 接入既有序列化器 `io/export_tables.py:667 parameters_csv_bytes`（无需从零写，仅需把它挂进 `_dataset_artifacts`）；SVG 需新增产物生成器。
- **硬约束**：新产物必须满足逐位复现（记忆 `xrr-export-byte-identity-check`：固定 `PYTHONHASHSEED=0` + master_seed + 共享源）。
- **验证**：`tests/unit/services/test_exports.py` + `tests/gui/test_export_dialog.py` 扩展格式断言；逐位复现核验命令按记忆固定环境跑。

### 5.3 G9 参数三态（仅范围）【已确认 · 模型 + 服务层】

- **现状（签名级复核）**：状态由两个独立布尔表达，无第三态。`ParameterSetting`（`model/parameters.py:130-142`）唯一态是 `locked: bool = False`；`ParameterDefinition`（`:163-195`）有 `locked: bool`（必填，`:173`）+ `constrained: bool = False`（`:178`），`__post_init__`（`:189-192`）校验二者均为 bool。**无 `range_only`/`bounded` 第三态**；`soft_range` 不是字段，而是 `PriorSpec.kind` 的取值（`:16 PRIOR_ARITY` 含 `"soft_range": 3`）。`set_parameter_settings(project, dataset_id, settings: Sequence[ParameterSetting])`（`services/parameters.py:271-275`）；`validate_parameter_settings`（`:126-129`）在 `:136-137` 对 `definition.constrained` 参数直接 `raise ValueError("cannot set constrained parameter")`——即受约束态由 definition 侧承载，settings 侧无法覆写。
- **需改**：`model/parameters.py` 状态枚举 + `services/parameters.py` 的 `set_parameter_settings`/`validate_parameter_settings` 语义 + `parameters/table.py` delegate 改三态。
- **硬约束（已坐实指纹风险 + 解法）**：`ParameterDefinition` **确被指纹化**——`checkpoint_identity`（`fit/checkpoint.py:91-107`）的 `parameter_settings_fingerprint = _fingerprint(problem.parameter_definitions)`，经 `_canonical_dataclass`（`:43-49`）**逐字段**序列化。给 `ParameterDefinition` 新增第三态字段会改变字段集合 → **必然指纹漂移**。**唯一安全解法**：按现有 `POST_FREEZE_OMITTED_DEFAULTS`（`checkpoint.py:24-40`，已含 `("ParameterDefinition","prior"): None`、`("ParameterDefinition","constrained"): False`）为新字段登记其默认态，`_is_unset_post_freeze_field`（`:33-40`）在字段等于默认时省略，才能让未配置运行复现历史指纹。注意 `ParameterSetting` 不直接进 `checkpoint_identity`（payload 用 `parameter_definitions` 而非 settings），故三态若只落在 setting 侧则无指纹风险——**契约设计须先定「第三态落 definition 还是 setting」**，这决定是否触发上述省略登记。
- **验证**：`tests/gui/test_parameter_table.py` + 模型层单测加三态断言 + `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 保持绿。

### 5.4 G18 + G25 角度约定切换 + 检测 API【G25 已确认纳入】

- **决议**：**G25 自动检测纳入实现范围**（不再仅 G18 手动）。
- **现状（签名级复核）**：`import_dialog.py:186-189` 只有 mono/mixed Kα radio；`rg 'detect_angle_convention|angle_convention' src` **全 src 零命中**——`api` 无任何 convention 探测/枚举符号。现有角度语义**仅由标量偏移 `import_angle_offset_deg` 表达**，分散在 `services/datasets.py:117/245/436/577/688` 的导入路径上，且 `import_data(path, beam, import_angle_offset_deg=0.0, column_mapping=None) -> PreparedData`（`services/datasets.py:114-119`）与 `add_dataset`（金标 `test_public_api.py:131` 硬编码带 `import_angle_offset_deg: 'float' = 0.0`）都带此参数。相关 preview 结构在 `model/automation.py`：`MeasurementPreset`（`:58`）/`ImportFilePreview`（`:91`）/`ImportBatchPreview`（`:106`）。
- **需改**：
  - G18（手动切换 2θ/θ）：纯 UI radio + 传参给既有 `import_angle_offset_deg` 标量路径——归 Group 2，可先做。**注意现有模型只有标量偏移、无「约定」枚举**，2θ/θ 若需语义化区分（非仅偏移量）可能要新增字段，契约设计阶段须定清。
  - G25（自动检测）：在 `services/datasets.py` + `api` **从零新增** `detect_angle_convention` 符号（当前完全不存在）；`api.py` 新增须同步 `test_public_api.py` 金标签名 + `__all__` 分组断言。UI 侧在导入对话框接入「自动检测」入口，结果回填 G18 的 radio。
- **工序**：G18 手动切换先落地（Group 2）；G25 自动检测后端契约设计后接入。

---

### 5.5 B1 固定三区布局【用户已确认纳入 · 独立高风险主项】

- **决议**：用户拍板**纳入方案，标为独立高风险主项**（§0.3 ②）。
- **关键事实（grep 复核，改变了本项定性）**：仓库**曾经就是 QSplitter 固定三区，后已主动迁移到 dock 布局**——`gui/workspace.py:33-38` docstring 原文「Panel geometry moved to the dock layout, which persists separately as an opaque `dock_state`. Splitters are therefore optional here」；`WorkspaceSnapshot` 仍保留 `workspace_splitter_sizes`/`left_splitter_sizes` 字段、`WorkspaceView.from_root`（`:39-46`）仍 `findChild(QSplitter, "workspaceSplitter")` 但**现恒为 None**（迁移遗留的兼容壳）。且 `gui/window_layout.py:1` docstring 是「Stable three-column MainWindow」——**当前 dock 布局本就是三列**：左（dataDock+structureDock，`DOCK_SPECS` L34-40）| 中（`central_stack`：guidance/plot，L119-123）| 右（parameters/fit/results **tab 化**，L133-135）。
- **因此差异只剩一点**：设计稿要「**固定宽度、不可拖拽**」（264/flex/340，CSS L104），现状是「**可拖拽/可浮动/可 tab**」（`DOCK_FEATURES = Movable|Floatable|Closable`，`window_layout.py:42-46`）。**不是缺三列，是三列可被用户拖散。**
- **撞的契约**：`api.set_dock_state` 签名被架构测试金标锁死（`tests/architecture/test_public_api.py:140` 硬编码 `"(project, state) -> XrrProject"`，`:95`/`:184` 列在导出集）；dock_state 往返在 `main_window.py:284-328`（saveState→比对 `_default_dock_state`→`api.set_dock_state`→`restoreState`）；`window_layout.py:141` 存 `_default_dock_state`。
- **两条路线（供你二次拍板取舍）**：
  1. **低风险 · 锁死 dock 外观（推荐先做）**：不拆 QDockWidget、不动 `dock_state` 契约，只从 `DOCK_FEATURES` 去掉 `DockWidgetMovable|DockWidgetFloatable`（保留/去掉 Closable 另议），并用 `setFixedWidth`/`resizeDocks` 钉死 264/340。**外观即达「固定三区、不可拖拽」，且 dock_state 往返、架构签名、`_default_dock_state` 全部不动**——本质降为表现层改动，风险等同 Group 1。代价：仍是 QMainWindow dock 内核，不是纯 QSplitter。
  2. **高风险 · 真·换回 QSplitter 三区**：重建 `workspaceSplitter`/`leftSplitter`（复用 `WorkspaceSnapshot` 里仍在的字段），废弃 dock 布局。**代价**：撤销一次已完成的 splitter→dock 迁移、孤立所有既存项目已持久化的 `dock_state`、`api.set_dock_state` 语义须重定义（撞金标签名测试）、`main_window.py:284-328` 往返逻辑重写、`test_public_api.py` 金标须改。blast radius 最大。
- **建议**：先走路线 1 验证外观达标；仅当「必须是纯 splitter、彻底弃用 dock」成为硬需求，再评估路线 2。**本轮不落代码**；路线选定后，路线 1 归 Group 1、路线 2 须单独出架构迁移设计 + 金标测试变更方案。
- **验证**：路线 1——`tests/gui` 断言 dock 不可移动 + 宽度固定 + dock_state 往返仍绿 + `tests/architecture` 全绿；路线 2——须新增架构迁移测试并重写 `test_public_api.py` 金标（属扩门禁，依 AGENTS §3 须再确认）。

---

### 5.6 G12 各数据集目标值表【从 Group 2 降级 · 需后端上报】

- **降级理由**：§4.6 前置调研坐实 fit 层无分数据集目标值上报（`FitProgress`/`FitStageSummary`/`FitCandidate` 全是聚合标量）。故 GUI 无法纯前端取数，须后端先新增上报。
- **需改**：在 fit 层产生分数据集目标值并沿回调上报。**关键约束**：`FitProgress`（`model/progress.py:66-97`）是流经进程队列的进度载体，但**它不进 `checkpoint_identity`**（`checkpoint.py:91-107` 只哈希 data/structure/instrument/config/parameter_definitions），所以给 `FitProgress` 加字段**不触发指纹漂移**——与 §5.1 的 `fit/` dataclass 禁令不同，progress 载体是安全的扩展点。仍须确认 `FitProgress` 未被别处指纹化/序列化进 checkpoint。
- **候选形态**：给 `FitProgress` 增一个 `dataset_objectives: tuple[tuple[str, float], ...] | None = None`（或等价映射）字段，联合拟合时填充、独立拟合时留 `None`；GUI `controller.py` 从回调提取，`fitting/panel.py` 渲染 QTableWidget。
- **验证**：`tests/gui/test_fit_progress.py` 加联合 objective 表断言 + 相关 fit 层单测；确认 `test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 不受影响（应天然绿，因 progress 不进指纹）。

---

### 5.7 G27 拟合实时指标四项（迭代次数 / nfev / 接受率 / 步长）【A3 复核定案 · 需后端上报】

- **定性（A3 复核定案，§0.3 ③）**：`ProgressView`（`fitting/progress.py`）已有阶段名（`stage_label:229`）、全局 J（`detail_label:233-236`，best=best_objective）、耗时（`meta_label:263`）；**缺 迭代次数 / 函数评估 nfev / 接受率 / 步长 四项标量指标**——设计稿 HTML L826-846 的 `insp-sec`/`kv` 行明确画出。各数据集目标值 `table.grid` 就是 G12（§5.6），不重复计。
- **需改**：这四项当前无载体——`FitProgress`（`model/progress.py:66-97`）字段无承载它们的项。须在 fit 层于阶段迭代时产生并沿回调上报。
- **关键约束（与 §5.6 同源 · 安全扩展点）**：`FitProgress` 是流经进程队列的进度载体，但**它不进 `checkpoint_identity`**（`checkpoint.py:91-107` 只哈希 data/structure/instrument/config/parameter_definitions），故给 `FitProgress` 加字段**不触发指纹漂移**——与 §5.1 的 `fit/` dataclass 禁令不同，progress 载体是安全扩展点。仍须确认 `FitProgress` 未被别处序列化进 checkpoint。
- **候选形态**：给 `FitProgress` 增标量字段承载四项（如 `iteration: int | None`、`nfev: int | None`、`acceptance_rate: float | None`、`step_size: float | None`），各阶段能算则填、否则留 `None`；GUI `controller.py` 从回调提取，`fitting/progress.py` 在 `meta_label` 区补四项显示。注意各阶段（全局筛选 / 局部精修 / MCMC）能给出的指标不同——步长/接受率对 MCMC 有意义，nfev/迭代次数对局部精修有意义，契约设计须逐阶段定清哪几项可填。
- **可与 G12 合并施工**：G12（§5.6）与 G27 同为「扩 `FitProgress` 字段 + GUI 渲染」，同属 progress 载体扩展、同不触指纹，可同一后端契约一并设计、一并迁移。
- **验证**：`tests/gui/test_fit_progress.py` 加四项指标断言 + 相关 fit 层单测；`test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 应天然绿（progress 不进指纹）。

---

## 6. Group 4 — 遗漏补录

### 6.1 G23 帧④三 tab（进度 / 实时反射率 / 目标值轨迹）

- **现状**：拟合面板是纵向 `QVBoxLayout`（`fitting/panel.py:65-111`），无 tab。
- **可复用**：进度视图（`ProgressView`，`:53`）、实时反射率（`LiveReflectivityPlot`，经 `preview_available` 信号 `:133`）。
- **全新组件**：**「目标值轨迹 / 收敛曲线」全仓库零实现**，需在 `gui/plots/` 新建（numpy/matplotlib 依赖门禁）。需确认 progress 回调是否上报可绘制的目标值序列——**若不上报，此组件的后端部分归 Group 3。**
- **参照**：帧①四 tab（`diagnostics.py` TAB_SPECS「对数反射率/原始数据与模型/qz⁴·R/加权残差」）已完全吻合设计稿，可作 tab 结构参照。

### 6.2 G24 导入高级选项

见 §4.3（已并入 Group 2 的导入项）。

### 6.3 G25 角度检测 API

见 §5.4（自动检测部分归 Group 3）。

---

## 7. 建议工序

```
Group 1（纯表现层）──────── 立即可做，无阻塞，先积累 TDD 节奏
  G3 明暗 · G5 模式文字 · G6 层预览 · G7 双CTA · G13 横幅 · G4 modebar

Group 2（接既有后端）────── 后端就绪，注意破坏性确认与既有组件整合
  G1+G2 segment（+破坏性确认） · G17 manifest预览 · G19+G24 导入 ·
  G21+G14+G15 不确定度tab（整合既有图） · G8 拖拽 · G12 目标值表* · G22 子管线（改 guidance/panel.py） ·
  G26 结构诊断提示（渲染既有 analyze_structure 证据）
  * G12 若 fit 层不上报 per-dataset 目标 → 降级 Group 3

Group 3（已确认，先出契约设计再 TDD）─ 触达 fit/services/model/api，涉指纹与复现门禁
  G10+G11 暂停/跳过（真·挂起/恢复） · G16 CSV/SVG（formats 列表） · G9 参数三态 · G25 角度检测API ·
  G27 实时指标四项（扩 FitProgress，与 G12 同源可合并施工）

Group 4（遗漏补录）───────
  G23 帧④三tab（含全新目标值轨迹组件，后端部分可能落 Group 3）

B1 固定三区布局（独立高风险主项，§5.5）─ 先走路线1（去 Movable/Floatable+钉宽，风险≈Group1）
  仅当「必须纯 QSplitter、弃用 dock」成硬需求，才评估路线2（撞 set_dock_state 金标，须再确认）
```

**关键路径**：Group 1 无依赖，最先做。Group 2 中 G21 的 tab 整合是 G14/G15 的结构落点（但相关图已存在，不阻塞）。Group 3 四项已确认纳入（§0.1），但每项须先产出后端契约设计 + 指纹/复现门禁影响验证，再动代码；G10（真·挂起/恢复）指纹漂移风险最高，G16（formats 列表重构）调用点迁移面最大。

---

## 8. 架构约束（与 v1 相同，重申）

1. **依赖边界**：gui 只 import `api` + `gui.*`；numpy/matplotlib 只在 `gui.plots.*`
2. **dock 持久化契约**：保留 QDockWidget + dock_state 往返；不动 QSplitter。**唯一例外是 §5.5 B1**——其路线 1 仍守此约束（只锁 dock 外观），路线 2 才主动挑战本条（撤销 dock 迁移、重定义 `api.set_dock_state`），且路线 2 依 AGENTS §3 属扩门禁须单独确认后方可动
3. **checkpoint 指纹**：不碰 `fit/` 层 dataclass 字段（记忆 `xrr-checkpoint-fingerprint-drift`）
4. **import_dialog → io.xy 例外**：已批准，保持不动
5. **pyqtgraph logmode**：`getData()` 返回 log10 空间（记忆 `xrr-pyqtgraph-logmode-getdata`）
6. **引导-专家共享代码路径**：引导步骤路由到专家面板同一批按钮，不另写逻辑
7. **导出逐位复现**（本版新增强调）：新增导出产物须满足逐位一致（记忆 `xrr-export-byte-identity-check`）

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
6. Group 3 四项（已确认纳入，§0.1）各自在落地前产出后端契约设计（签名/事件模型 + 指纹影响 + 逐位复现门禁验证），且 `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 全程保持绿
7. B1（§5.5）：路线 1 落地后三列不可拖拽 + 宽度固定，且 dock_state 往返与 `tests/architecture/test_public_api.py` 金标全程保持绿；若改走路线 2，须先产出架构迁移设计 + 金标测试变更方案并单独获确认（扩门禁，依 AGENTS §3）

---

## 12. 开工前决策清单（定案）+ 后端契约草案

> 本节把 §0.1 与 §5 中标注「须先出契约设计」的悬置项**逐条拍死到签名级**，接手方直接照此实现，无需再回头确认。仍不落代码——以下代码块是**契约草案**（proposed signatures），非既有源码。凡改动金标签名处均已点名对应 `test_public_api.py` 行。

### 12.1 G16 导出 `formats` 契约【定案】

**定型**：新增 `ExportFormat` 字符串枚举，`export_result` 入参从 `include_ort: bool` 改为 `formats: Sequence[ExportFormat]`，默认集**逐位复现现有默认产物**（不含 CSV/ORT/SVG）。

```python
# services/exports.py（契约草案）
from enum import StrEnum

class ExportFormat(StrEnum):
    XLSX = "xlsx"   # fit_result.xlsx（既有）
    JSON = "json"   # fit_result.json（既有）
    PNG  = "png"    # fit_overview/sld_profile/residuals.png（既有）
    ORT  = "ort"    # fit_result.ort（既有，原 include_ort=True）
    CSV  = "csv"    # parameters.csv —— 复用 io/export_tables.py:667 parameters_csv_bytes（新接入）
    SVG  = "svg"    # 矢量图件（从零新增生成器）

DEFAULT_FORMATS: tuple[ExportFormat, ...] = (ExportFormat.XLSX, ExportFormat.JSON, ExportFormat.PNG)

def export_result(
    result: XrrProject | ProjectFitResult,
    output_dir: str | Path,
    *,
    formats: Sequence[ExportFormat] = DEFAULT_FORMATS,
) -> ExportManifest: ...
```

**迁移映射（在调用点显式做，不靠默认值糊）**：旧 `include_ort=True` → `formats=(*DEFAULT_FORMATS, ExportFormat.ORT)`；旧 `include_ort=False` → `formats=DEFAULT_FORMATS`。
- `cli/commands.py:114`：`include_ort=arguments.ort` → 由 `arguments.ort` 拼 formats。
- `gui/export/dialog.py:155`：勾选框集合直接映射到 `formats`。
- 测试调用 `test_exports.py:189/205`：`include_ort=False/True` 断言改写为 `formats=` 语义。
- **金标同步**：`test_public_api.py:166` 硬编码签名串改为 `"(result: 'XrrProject | ProjectFitResult', output_dir: 'str | Path', *, formats: 'Sequence[ExportFormat]' = ...) -> 'ExportManifest'"`；`ExportFormat` 若经 `api` 暴露，`:189` 导出集需加符号。
- **逐位复现**：默认集必须与当前 6 产物逐位一致（记忆 `xrr-export-byte-identity-check`：固定 `PYTHONHASHSEED=0`+master_seed+共享源核验）。
- **⚠️ 架构边界（源码核实）**：`services/exports.py` 被门禁**禁止依赖 `analysis` 或 numpy**（`exports.py` 内 `render_orso` 处注释明写：协方差改由 model 层 `UncertaintyReport.covariance` 透传）。故 **SVG 生成器不能写在 `services.exports` 里**，须与既有 `fit_overview_png`/`sld_profile_png` 同落 io/绘图层，`exports.py` 仅以 `ArtifactProducer` 引用其 bytes 产出。CSV 同理复用 `io/export_tables.py:667`。

### 12.2 G9 参数三态落点契约【定案：落 setting 侧】

**定型**：第三态**落 `ParameterSetting`（setting 侧），不落 `ParameterDefinition`**。理由已坐实——`ParameterSetting` 不进 `checkpoint_identity`（§5.3），故 setting 侧扩展**零指纹漂移**，无需登记 `POST_FREEZE_OMITTED_DEFAULTS`。

```python
# model/parameters.py（契约草案）
from enum import StrEnum

class ParameterFreedom(StrEnum):
    FREE       = "free"        # 自由变动
    FIXED      = "fixed"       # 固定（旧 locked=True）
    RANGE_ONLY = "range_only"  # 仅范围：在 soft_range 内受限变动，不自由外推

@dataclass(frozen=True, slots=True)
class ParameterSetting:
    # 用 freedom 取代旧 locked: bool；旧 locked=True → FIXED，locked=False → FREE
    freedom: ParameterFreedom = ParameterFreedom.FREE
```

**服务层语义**（`services/parameters.py`）：`set_parameter_settings`/`validate_parameter_settings` 把 `RANGE_ONLY` 翻译为「保留/附加 soft_range prior 且标记非自由外推」；`FIXED` 等价旧 locked。`validate_parameter_settings:136-137` 对 definition 侧 `constrained` 的 raise 保持不动（约束态仍由 definition 承载，settings 不覆写）。
- **兼容**：读旧项目时 `locked` 字段映射到 `freedom`（迁移函数处理）。
- **金标**：`ParameterSetting` 不在 `test_public_api.py` 签名断言里（非顶层 API 函数），但若 `ParameterFreedom` 经 `api` 暴露需同步导出集。
- **验证**：`test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 天然绿（setting 不进指纹）。

### 12.3 G18 角度约定契约【定案：修正定性——需新字段，非纯 Group 2】

**定型 + 定性修正**：2θ↔θ 是**轴变换（×2 / ÷2）**，既有标量 `import_angle_offset_deg`（加性偏移）**无法表达**。故 G18 手动切换需**新增 `angle_convention` 字段**，触及 `import_data`/`add_dataset` 签名 → **金标 `test_public_api.py:131` 须同步**。本项因此**从 Group 2 上修为 Group 3-adjacent**（带金标变更），不再是纯 UI。

**⚠️ 默认方向（源码核实，纠正前一版草案）**：数据模型原生轴是 **`two_theta_deg`**（`datasets.py:222/248/605` 全走 `two_theta_deg`），且当前导入路径**无任何 ×2/÷2 转换**——即**现状默认「输入即 2θ」**。故向后兼容的默认必须是 `angle_convention="two_theta"`（不变换，逐位复现现有导入结果）；需做变换的是 `"theta"` 分支（输入是入射角 θ，导入时 ×2 归一到 `two_theta_deg`）。**默认绝不能设 `"theta"`，否则所有既有项目重导即角度翻倍。**

```python
# services/datasets.py（契约草案）
from typing import Literal
AngleConvention = Literal["two_theta", "theta"]

def import_data(
    path: str | Path,
    beam: ...,
    *,
    import_angle_offset_deg: float = 0.0,
    angle_convention: AngleConvention = "two_theta",   # 新增；默认=现状（输入即 2θ，不变换）
    column_mapping: ... | None = None,
) -> PreparedData: ...
# add_dataset 同步加 angle_convention（默认同样 "two_theta"；金标 test_public_api.py:131 签名串同步）
```

`"theta"` 时导入路径对角度列 ×2 归一到 `two_theta_deg`，再叠加 `import_angle_offset_deg`（偏移语义不变）。G25 自动检测（§5.4）回填此字段。

### 12.4 G23 目标值轨迹契约【定案：纯客户端累积，无后端】

**定型**：「目标值轨迹 / 收敛曲线」**不需后端新增序列上报**。每次 `FitProgress` 发布已带 `best_objective`（`model/progress.py`），GUI 侧在 `gui/plots/` 新建 `ObjectiveTrajectoryPlot`，**从连续 progress 事件累积 (publish-index 或 wall-clock, best_objective) 点列**自绘。x 轴用「发布序号/耗时」即可，**无需 G27 的 nfev**；仅当要求以 nfev 为 x 轴时才依赖 G27 的 `nfev` 字段。
- **归属修正**：G23 轨迹组件 = Group 2（客户端累积既有 progress 流），**后端零改动**（原 §6.1「后端部分归 Group 3」的担忧解除）。
- **约束**：组件在 `gui/plots/`（numpy/matplotlib 门禁 §8.1）；logmode getData 陷阱（记忆 `xrr-pyqtgraph-logmode-getdata`）适用。

### 12.5 B1 路线【定案：走路线 1 → 2026-09-03 改判路线 2】

**改判（2026-09-03）**：按本节自带的重启条款——「路线 2（换回 QSplitter）**仅在「必须纯 splitter、彻底弃用 dock」成为硬需求时**才重启评估，届时另出架构迁移设计并按 AGENTS §3 单独确认」——两项前置均已完成：架构迁移设计与金标测试变更方案见 `docs/architecture/workspace-columns-migration.md`，AGENTS §3 要求的单独确认由用户于 2026-09-03 给出。**故 B1 交付路线 2**：保留 `QSplitter` 三列并禁用全部手柄（`window_layout.py:_pin_columns`，`setSizes` 仍有效，默认宽度 / `reset_layout` / 从工程读回都走它）。硬需求的论证在迁移文档 §2：路线 1 的「只改表现层」前提建立在「`window_layout.py` 是一份 156 行的停靠装配模块」之上，重设计把栏内结构搬进同一模块后该文件已 791 行，照字面回退会把 §11.1 禁止的两件事（dock 标题栏 / 关闭按钮、右栏三段退回 tab）重新退回去。锁定面已量化：39 条 B1 用例中 10 条引用顶层 `workspaceSplitter`（仅 2 条断言机制本身），**架构门禁 0 条新增**——`tests/architecture/test_public_api.py` 一行未改，`api.set_dock_state` 仍导出且签名不变，`dock_state` 仍在 schema 内无损往返（对 GUI 成为死 API，迁移文档 §3/§5）。台账：`docs/architecture/plan-v2-acceptance-ledger.md` §5/§5.1。**下面「定型」与「施工基线」三条属路线 1，随本次改判一并作废，保留仅为存档。**

**定型**：**采用路线 1（锁死 dock 外观）**，不走路线 2。即：`window_layout.py:42-46` 的 `DOCK_FEATURES` 去掉 `DockWidgetMovable|DockWidgetFloatable`（`Closable` 保留），用 `setFixedWidth`/`resizeDocks` 钉死 264/340。**不拆 QDockWidget、不动 `dock_state` 契约、不动 `api.set_dock_state` 金标**——降为表现层改动，风险等同 Group 1。路线 2（换回 QSplitter）**仅在「必须纯 splitter、彻底弃用 dock」成为硬需求时**才重启评估，届时另出架构迁移设计并按 AGENTS §3 单独确认。

**施工基线（源码核实，落笔即改点）**：
- `window_layout.py:136-139` 现状 `resizeDocks([dataDock, parametersDock], [320, 380], Horizontal)`——**改的是这两个数（320→264、380→340），不是从零加**。宽度锁死配合 `setFixedWidth`（或 `setMinimumWidth==setMaximumWidth`）钉死左右两列，中列画布保持自适应。
- **右列是 tab 化的**：`window_layout.py:133-135` 把 `parametersDock`/`fitDock`/`resultsDock` 用 `tabifyDockWidget` 叠成一列并 `parametersDock.raise_()`。设计稿 340px 检视器对应的正是这一叠 tab 列，**去掉 Movable/Floatable 后 tab 切换仍保留**（Closable 保留即可关单页），符合设计稿「固定三区」但内部可切换的形态。
- `_default_dock_state = window.saveState()`（:140）在改宽后重新落定，`dock_state` 往返契约不变。

### 12.6 G27 分阶段可填指标契约【定案】

**定型**：`FitProgress` 新增四个可空标量（progress 载体不进指纹，§5.7），各阶段按下表填，不适用留 `None`：

| 字段 | 全局筛选(A/B) | 局部精修(C/D) | MCMC(E) |
|---|---|---|---|
| `iteration: int \| None` | 代数 | 迭代数 | 步数 |
| `nfev: int \| None` | ✓ | ✓ | — |
| `acceptance_rate: float \| None` | — | — | ✓ |
| `step_size: float \| None` | — | ✓(信赖域) | ✓(提议步长) |

与 G12 的 `dataset_objectives`（§5.6）**同属 `FitProgress` 扩展，同一后端契约一并设计、一并迁移**。
