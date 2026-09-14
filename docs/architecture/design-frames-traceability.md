# 设计稿 6 帧 → 实现追溯（计划 §11 判据 1）

## 0. 这份文档解决什么

`deliverables/gui-redesign-缺失修补方案-v2.md` §11 判据 1：**「设计稿 6 帧全部 UI 元素在实现
中可追溯（每项有 rg 可验证的代码路径）」**。此前每项都各自跑过 RED→GREEN，但从没有一处把
G1–G27 + B1 的锚点摆在一起、一次核完。本文就是那份清单。

用法：§3 是一条可复制的命令，逐项 `rg -F` 锚点，缺一项就打印 `MISSING`。最近一次执行
（2026-09-03，本工作树 `gui-reskin`）**33 条锚点全部命中，MISSING=0**。

§2 如实记录两处与计划 §3/§5.1「落点」栏不一致的交付，其中 G11 是语义层面的偏离，需要知情。

## 1. 追溯表

| 项 | 设计稿元素（计划原表措辞） | rg 锚点 | 文件 |
|---|---|---|---|
| G1 | 引导/专家 segment（工具栏） | `_install_mode_segment` | `gui/chrome.py` |
| G1 | 同上 · segment 控件本体 | `class SegmentedControl` | `gui/segmented.py` |
| G2 | 独立/联合 segment（工具栏，非面板内 combobox） | `batchModeSegment` | `gui/chrome.py` |
| G3 | 明暗切换 | `appearanceToggleButton` | `gui/chrome.py` |
| G4 | 画布 modebar（平移/框选/复位/选区） | `MODE_SPECS`、`_install_mode_buttons` | `gui/plots/interactions.py` |
| G5 | 状态栏模式文字（置信度徽章早已存在，只补模式） | `GUIDED_READINESS_TEXT` | `gui/chrome.py` |
| G6 | 引导「结构」卡内嵌只读层堆叠预览 | `component_fill`（复用 `structure/stack.py`） | `gui/guidance/panel.py` |
| G7 | 引导卡双 CTA（primary + secondary） | `_build_secondary` | `gui/guidance/panel.py` |
| G8 | 层列表拖拽重排 | `InternalMove` | `gui/structure/reorder.py` |
| G9 | 参数三态（含 `range_only`） | `RANGE_ONLY` | `model/parameters.py` |
| G10 | 暂停（真·挂起/恢复） | `def pause` / `is_paused` / `def resume` | `services/workers.py` |
| G10 | 同上 · 停车点探针 | `def pause_aware_probe` | `services/fitting.py` |
| G10 | 同上 · 帧④ 按钮 | `pauseFitButton`（`⏸ 暂停` / `▶ 继续`） | `gui/fitting/panel.py` |
| G11 | 跳过本阶段 | `def skip_stage` | `services/workers.py` |
| G11 | 同上 · 阶段边界出口 | `except StageSkipped` | `fit/pipeline.py` |
| G11 | 同上 · 帧④ 按钮 | `skipStageButton`（`⏭ 跳过本阶段`） | `gui/fitting/panel.py` |
| G12 | 各数据集目标值表 | `dataset_objectives`、`"各数据集目标值"` | `gui/fitting/metrics.py` |
| G13 | 联合模式横幅 | `mark_banner` | `gui/fitting/progress.py` |
| G14 | MCMC 后验直方图 | `class McmcHistogramPlot` | `gui/plots/posterior.py` |
| G15 | Profile 似然 Δχ²=1 参考线 | `axvline`（SLD 侧 `axhline` 在 `plots/sld.py`） | `gui/plots/posterior.py` |
| G16 | 导出多格式（`formats` 列表参数，非 `include_ort` 布尔） | `normalize_export_formats` | `services/exports.py` |
| G17 | 导出前 manifest 预览 | `_build_manifest_preview` | `gui/export/dialog.py` |
| G18 | 角度约定切换（2θ / θ 两档） | `twoThetaConventionOption` | `gui/data/import_dialog.py` |
| G19 | 批量导入预览表 | `_refresh_preview`（`QTableWidget`） | `gui/data/import_dialog.py` |
| G20 | 专家左栏纵向六段管线 stepper | `class StepRow`（步表在 `navigation/panel.py:PIPELINE_STEPS`） | `gui/navigation/steps.py` |
| G21 | 不确定度四子选项卡 | `class UncertaintyPages` | `gui/plots/posterior.py` |
| G22 | 不确定度子管线（引导侧） | `uncertainty_method_rows` | `gui/navigation/methods.py` |
| G23 | 帧④ 三 tab（进度 / 实时反射率 / 目标值轨迹） | `CANVAS_TAB_TITLES` | `gui/fitting/progress.py` |
| G24 | 导入高级选项（分隔符 / 跳过表头行 / 强度取对数） | `HEADER_SKIP_CAPTION` | `gui/data/import_dialog.py` |
| G25 | 角度约定自动检测 API | `detect_angle_convention`（在 `__all__` 内） | `api.py` |
| G26 | 结构诊断相关性提示 | `structureCorrelationHint`（原文见 `STRUCTURE_CORRELATION_HINT_TEXT`） | `gui/window_layout.py` |
| G27 | 拟合实时指标四项（迭代/nfev/接受率/步长） | `METRIC_ACCEPTANCE`（四项在 `METRIC_KEYS`） | `gui/fitting/metrics.py` |
| B1 | 三栏定宽不可拖拽 | `_pin_columns`（详见 `workspace-columns-migration.md`） | `gui/window_layout.py` |

## 2. 两处与计划落点表不一致

### 2.1 G4：modebar 落在 `plots/interactions.py`，没有新建 `plots/modebar.py`

计划 §3 的落点栏写「`gui/plots/` 新增 `modebar.py`」。实际把四枚模式按钮做进了既有的
`plots/interactions.py`（`MODE_SPECS` 与既有 ViewBox 模式表同住，`_install_mode_buttons` 装配），
`gui/plots/` 下没有 `modebar.py`。这只是文件位置的差别：设计稿 `.modebar` 的四枚字形
`✥ ⤢ ⌂ ▭` 与「qz⁴R/残差面板只读」的按面板启用都在这一处，判据 1 照样可追溯。

### 2.2 G11：交付的是设计稿字面语义，不是 §5.1 锁定的 early-stop

**这一条是语义偏离，不是位置差别。**

- §0.2 的硬结论：omit-middle「架构不可行」——`_stage_candidates` 用
  `next(v for v in reversed(state.summaries) if v.stage == parent)`，父阶段缺失即 `StopIteration`；
  故 §5.1 把 G11 锁定为 early-stop（某阶段后提前收尾、把当前 best 当最终结果发布）。
- 设计稿帧④（HTML L850）画的按钮是 **`跳过本阶段`**，旁边才是 `⏸ 暂停`。交付按设计稿的字面
  语义做了：`skip_stage` 作废当前这一个阶段，搜索接着往下跑，已跑完的阶段与候选都留着。
- 让它可行的改动在 `fit/pipeline.py:118-127`：请求的那一阶段没有 summary 时，父代**回退到最近
  一个真正跑完的阶段**（`state.summaries[-1]`），不再 `StopIteration`。这是 `fit/` 层查找规则的
  改动，超出 §5.1「签名级落点」所描述的范围。
- 计划锁定的 early-stop 仍在，但只作为退化分支：阶段 A 被跳过时一个候选都没有，
  `pipeline.py:312-313` 直接 `break` 落到既有 `return _result(request, state, seeds)`，
  绕开 `SearchCancelled`。界面上没有单独的「提前收尾」控件，设计稿也没画。

风险收敛面，如实说：那条回退只在「某一阶段被跳过」之后才可能触发——正常跑每个阶段都会先
`state.append(outcome)` 再被下一阶段读到，resume 的 `remaining_stages` 是严格后缀，父代都在。
硬约束照守：**没有给任何 `fit/` 层 dataclass 加字段**，
`tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints` 全程绿。
覆盖：`tests/unit/fit/test_resume.py:367/381/408`（探针抛 `StageSkipped` 的出口）、
`tests/unit/services/test_workers.py:312`（开关一次性，下一阶段不会连着被跳）、
`tests/gui/test_fit_progress.py:855`（按钮把请求送到 worker）。

## 3. 复核命令

表格自己就是核验清单：「rg 锚点」栏的第一个反引号词 + 「文件」栏拼成一次 `rg -F`。在工作树根跑：

```bash
while IFS='|' read -r _ item _ anchor file _; do
  a=${anchor#*\`}; a=${a%%\`*}
  f=${file#*\`}; f=${f%%\`*}
  rg -F -q "$a" "src/xrr_fitter/$f" && echo "OK      $item" || echo "MISSING $item  $f  $a"
done < <(rg '^\| (G[0-9]+|B1) \|' docs/architecture/design-frames-traceability.md)
```

2026-09-03 实测：33 行全部 `OK`，`MISSING` 0 条。改了文件名或改名了符号，这条命令会当场报出来。

同日一并复核了另外两条判据（供交叉参考，细节仍归各自的判据）：`tests/architecture` 194 passed、
`tests/gui` 1310 passed（判据 2）；`tools/check_radon.py` 退 0、改动过的 231 个 `.py`
`ruff check` 与 `ruff format --check` 均干净、`tools/check_hygiene.py` 除工作树固有的
`.git must be a normal directory` 与本树自带 `.venv` 外无条目（判据 3）。

## 4. 这份文档不覆盖什么

- **不替代各项自己的测试**。锚点只证明「元素在实现里有位置」，行为对不对由各项的 GUI/单元
  用例负责（判据 2、5）；本文只回答判据 1。哪一项由哪条测试看着，逐项列在
  `plan-v2-acceptance-ledger.md` §6。**§1 的锚点不能拿去 grep `tests/`**：它们是实现侧符号，
  多半私有，那样扫会得到 14 项假空洞；三处具体撞名（G4 `MODE_SPECS`、G18 `twoTheta`、
  G8 断言藏在共享助手里）记在该节 §6.1。
- **不重复 B1**。三栏的判据、钉死方式、`dock_state` 的去向与金标变更方案在
  `workspace-columns-migration.md`，本文只留一条锚点。
- **不裁决 §12.5**。B1 走的是路线 2，计划正文仍写「定案：走路线 1」；这项确认按 §11.7 与
  AGENTS §3 归用户，2026-09-03 的裁决是**保留路线 2**，记在
  `plan-v2-acceptance-ledger.md` §5（含 §12.5 的收口写法建议）。本文与那份迁移文档都只是备料。

## 5. 反向核查：设计稿 → 实现（2026-09-03 补做）

§1 是**正向**追溯：清单 28 项 → 代码。它有一个结构性盲点——**设计稿上有、但当初就没被写进
清单的元素，正向扫描永远发现不了**，因为它只走清单里的行。所以补做一次反向扫描：从设计稿
HTML 逐帧提取可见控件，再去 `src/` 找落点。

口径：帧①–⑥ 的正文范围（HTML L341–1077，`<h2>` 之后到「令牌」节之前），提取全部 `class="btn*"`
与 `class="tab*"` 字面，去重。非帧的三节（@312 理据、@1078 令牌、@1141 能力对照表）不算界面元素。

| 类别 | 条数 | 结果 |
|---|---|---|
| 按钮字面（新建 / 打开 / 保存 / ⚡ 一键拟合 / 导出… / ☾ / 切换 / 看起来没问题，开始拟合 → / ＋ 我要手动加\删层 / ＋ 添加层 / 建议氧化层 / 周期结构… / ⏸ 暂停 / ⏹ 停止 / ⏭ 跳过本阶段 / ⏹ 停止并保留最优 / ▶ 运行 MCMC / 取消 / 导入 4 个（跳过 1 个失败）/ 导出到文件夹…） | 20 | 20/20 有实现落点 |
| tab 字面（对数反射率 / 原始数据与模型 / qz⁴·R / 加权残差 / 样品结构 / SLD 深度剖面 / 参数总览 / 进度 / 实时反射率 / 目标值轨迹 / 相关矩阵 / Profile 似然 / SLD 可信带 / MCMC 后验） | 14 | 14/14 有实现落点 |
| 帧⑤ 右栏 `insp-sec` 标题与 kv 键 | 22 | 19 条字面命中，3 条另有交代，见下 |

三条不逐字命中的，逐条落实：

- **「MCMC 收敛诊断」**（段标题）：段本身在，读数换了措辞——
  `gui/results/uncertainty.py:235-237` 出 `最大 split-Rhat` / `最小 ESS` / `MCMC 边界命中（可疑）`，
  控件 `mcmcWalkers`、`mcmcBurnIn` 在 `:391-393`，walkers 下界校验在 `:475`。
- **「最强相关」/「次强」**（两行 kv）：**有意合并**成一行 `强相关：…` 聚合全部强相关对
  （`uncertainty.py:150`），判读话术进 `CORRELATION_CALLOUT_TEXT`（`:100`）。理由写在
  `:97-99` 的注释里：一次拟合可能有多对强相关，点名「最强/次强」会漏掉第三对。
- **「燃烧期 burn-in」**：交付为英文 `burn-in`（`mcmcBurnIn` 的标签），语义同一。

### 5.1 「重采样次数」由已知限制转为已补（新增字段）

这条原先记作「报不出来」的已知数据限制：`UncertaintyReport` 上没有重采样次数字段，检视区读
不出帧⑤ L967-968 的 `重采样次数 200`，帧⑤ 左栏那行也只能报参数个数。现已补齐字段，落点：

- `model/analysis.py:334` 新增 `bootstrap_sample_count: int = 0`，校验在 `:175`——非负整数，
  且 `bootstrap_performed=False` 时只能是 0（两个字段互相打脸时界面读哪个都会错）。
- `analysis/report.py:307` 在报告落地时填入，值由 `_bootstrap_request_count` 从证据**反解**
  （成功样本数 ÷（1 − 失败率）），不回头读预算 `problem.config.budget`：调用方可以递进一份
  用别的 `sample_count` 跑出来的 bootstrap，读数得跟着证据走。反解是精确的——
  `_collect_bootstrap_samples` 用两条 `RuntimeError` 钉住 `kept == count - failures`。
- `io/codec_results.py` 只在非 0 时写这个键，读侧进 optional 白名单：该字段之前存下的工程文件
  仍能加载，且重新编码逐位不变。
- `gui/results/uncertainty.py:150` 检视区出 `Bootstrap 重采样次数：…`，排在失败率**之前**
  （比例要先给基数）；`gui/navigation/methods.py:66` 左栏出设计稿字面 `200 次 · 失败 2%`。
- 两处 0 都读作「未记录」而不是「抽了 0 次」：检视区印 `未记录`，左栏退回 `N 参数 · 失败 x%`。

全军覆没（`failure_rate == 1.0`，成功 0 个）时分母为 0，次数无从得知，同样记 0，而
`bootstrap_performed` 保持 True。`analysis/joint.py` 不动——联合路径本来不跑自助抽样，默认 0
即正确。

### 5.2 反向核查抓到的一处真实偏离（已修）

G21 的第二个 tab：设计稿八处都写 **「Profile 似然」**（帧⑤ tab 在 HTML L911，另有 L873 帧
标题、L903 左栏子管线那一步、L930 画布内标题、L1147 能力对照表），实现原先写「参数剖面」
（`gui/plots/posterior.py:59`），且被 `tests/gui/test_uncertainty_dialog.py` 逐字钉住。

这不只是与设计稿不符，仓库内部也不自洽：G22 的方法表 `gui/navigation/methods.py:35` 用的是
`("Profile 似然", "逐参数扫描目标函数的谷宽")`，`tests/gui/test_step_scoped_views.py:275` 的注释
甚至照抄设计稿写「`.canvas-top` 只有四个标签（相关矩阵 / Profile 似然 …」。同一份证据在导航
里叫一个名、在画布标签上叫另一个名。

已改为设计稿原词。RED→GREEN 证据：先把断言改成设计稿字面，用例报
`At index 1 diff: '参数剖面' != 'Profile 似然'`；再改 `posterior.py:59`，
`test_uncertainty_dialog.py` + `test_step_scoped_views.py` 48 passed，`tests/gui` 全量
1311 passed。

**没有连带改画布内的子图标题**：`plots/sld.py:842/899` 与 `plots/diagnostics.py:317` 的
`set_title("参数剖面似然与区间")` 是分析页把相关矩阵与剖面两张子图并排画进同一 figure 时给
右半边的 axes 标题，设计稿没有这个并排形态；那句措辞受宽度约束（理由记在 `sld.py:928-932`：
标题 222px / 面板 835px，改短会与「相关矩阵」压字）。tab 是导航标签，axes 标题是排版内的说明，
两者不同层。
