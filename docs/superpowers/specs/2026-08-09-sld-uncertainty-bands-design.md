# SLD 剖面不确定度带设计

## 目标

把参数不确定度传播到 SLD 深度剖面，输出 68% 与 95% 的置信带。参照 Refl1D 的
`uncertainty.py` 做法。当前 `io/export_plots.py:74` 的 `sld_profile_png` 只画一条确定的
real/imag 曲线，读者无法判断剖面的哪一部分是数据支撑的、哪一部分是模型假定的。

## 原料

链路已完整，无需新的物理实现：

- `McmcReport.samples_physical` 已持久化（`io/codec_results.py:70`）。
- `physics/stack.py:151 rebuild_structure(structure, values)` 从物理值映射重建结构，
  文档明确"never mutates the compiled problem's declared structure"，可安全并行。
- `physics/stack.py:248 expand_structure(structure, wavelength_a)` 展开为 `SlabStack`。
- `physics/sld_profile.py:53 sld_depth_profile(stack, step_a=0.5)` 是纯函数，返回只读数组。

每条抽样的重放即三步串联。

## 对齐

难点只有一个：每条抽样的 depth 网格不同，因为厚度在变。`physics/sld_profile.py:11`
的 `_depth_grid` 起点为 `-max(10.0, 5.0 * roughness_a[0])`，终点为
`total + max(10.0, 5.0 * roughness_a[-1])`，两端都随抽样浮动。

采 Refl1D 的 `align_profiles` 解法：选一个对齐界面把深度平移到零点，再插值到公共网格。

- 默认对齐**基底界面**（`interfaces[-1]`，最深处）。理由：XRR 的基底通常是已知单晶，
  物理上最确定，把不确定度累积推向表面侧符合实际认知。
- 提供 `align="surface"` 备选，对齐 `interfaces[0]`。
- 公共网格取所有抽样对齐后 depth 范围的**交集**，步长沿用 `step_a`。取交集而非并集，
  避免边缘出现只有少数抽样支撑的伪带宽。
- 实部虚部**分别**取分位数。不对复数取模：那会丢掉吸收信息，而虚部恰是 XRR 密度对比的
  主要来源。

分位数取 16/50/84（68%）与 2.5/97.5（95%），与 Refl1D 一致。

## 代码边界

- 新增 `src/xrr_fitter/analysis/sld_bands.py`。放 `analysis` 而非 `io`：`ALLOWED["io"]`
  只有 `{io, model}`，无法 import `physics`，而重放必须调用 `expand_structure`。
- 新增 `model/analysis.py` 中一个只读结果类型（公共深度网格加六条分位曲线），作为
  `analysis` 与 `io` 的边界载体。
- 扩 `src/xrr_fitter/io/export_plots.py:74 sld_profile_png`：两条中位线加两组
  `fill_between`，透明度区分 68/95。
- 扩 `src/xrr_fitter/gui/plots/sld.py`。
- 不改 `physics/sld_profile.py`。
- 扩 `src/xrr_fitter/api.py`：暴露不确定度带的计算入口，让 GUI 无需经 `services` 取结果。

带的存在与否由 `McmcReport` 是否存在决定。无采样时渲染行为与当前**逐位相同**，这保证
增量兼容。

## 界面

`draw_sld`（`sld.py:32`）现在画选中候选的实/虚部，外加其他候选实部的淡色叠加。带加进来后
图上元素会变多，需要一条明确的优先级：**带只画选中候选的**，其他候选仍只有淡色中线。理由是
两个候选各带两组 `fill_between` 会互相遮挡到不可读。

带的显隐由一个复选框控制，默认开启（有 `McmcReport` 时）。无 `McmcReport` 时复选框禁用并
用 tooltip 说明"需要先运行 MCMC"——不是隐藏控件，因为隐藏会让用户以为软件没有这个能力。

68/95 两组带用透明度区分，图例文本写明分位数（`16–84%`、`2.5–97.5%`）而非笼统的"1σ/2σ"。
理由与 orso-validation 里 dQ 的 FWHM-vs-1σ 歧义同源：分位数是无歧义的，σ 需要额外假定
高斯性，而后验通常不是高斯的。

抽稀比例与对齐界面的选择要在图注里可见（设计正文已定"抽稀比例写进图注"），GUI 与
`export_plots.py:74` 共用同一段图注文本生成逻辑，避免屏幕上和导出图里说法不一致。

对齐界面默认基底，但界面提供切换（`QComboBox` 列出可选界面）。理由：默认基底对多层膜是对的，
但研究表面形貌的用户需要对齐最顶界面，而"带宽在哪里为零"完全取决于这个选择——这不是可以替
用户定死的参数。

## 失败与状态

- 抽样数上限默认 500，超出则均匀抽稀，抽稀比例写进图注。理由：`step_a=0.5` 加典型数千
  条抽样，每条都要 `rebuild_structure` + `expand_structure` + 剖面重建，不抽稀会让导出
  耗时不可接受；而 500 条对 84 分位的估计误差已远小于系统误差。
- 单条抽样重放失败（参数组合导致 `expand_structure` 报错）时记录并跳过，失败率超过 5%
  则整体失败而非静默出一条基于少数抽样的带。
- 交集为空（抽样厚度差异极大）时明确报错，不退化成单曲线。
- `samples_physical` 与当前结构的参数名不匹配时报错，不按位置猜测对应关系。

## 验证

- 退化测例：零抽样方差时带宽必须**恰好为零**，中位线与 `sld_depth_profile` 直算结果
  逐位相等。
- 对齐正确性测例：构造两条只差一个整体厚度平移的抽样，基底对齐后带宽应为零。
- 实虚分离测例：构造只有虚部变化的抽样，实部带宽应为零而虚部不为零。
- 抽稀确定性测例：同一 `McmcReport` 两次抽稀得到相同子集（抽稀必须走确定性种子而非
  `np.random` 默认状态）。
- 无 `McmcReport` 时 `sld_profile_png` 输出与改动前逐位相同。
- 图像回归沿用项目既有 png 测试姿势。
- GUI 测例：无 `McmcReport` 时带复选框禁用且 tooltip 非空；`draw_sld` 输出与改动前
  逐位相同。
- GUI 测例：切换对齐界面后重绘，带宽为零的位置随之移动。
- GUI 测例：屏幕图注与 `sld_profile_png` 图注文本相同。

## 非目标

- 不做基于协方差的线性传播带。MCMC 抽样重放是唯一实现路径；协方差传播在厚度参数上是
  已知不可靠的近似。
- 不做 bootstrap 抽样的带（bootstrap 结果不保留完整参数向量集合）。
- 不改变 `sld_depth_profile` 的 `step_a` 默认值或网格策略。
- 不在带上标注单个界面位置的不确定度（那是另一个可视化需求）。
