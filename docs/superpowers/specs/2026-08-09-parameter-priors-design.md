# 参数先验分布设计

## 目标

把参数的先验知识从"硬边界"升级为"分布"。当前 `ParameterDefinition` 只有 `lower`/`upper`
（`model/parameters.py:40`），全项目零 `invcdf`/`ppf`，`prior` 只指标度先验
（`model/fitting.py` 的 `scale_prior_enabled`/`scale_prior_tau_decades`/`scale_prior_center`/
`scale_prior_reason`）。参照 refnx 的 `Bounds`+`invcdf` 与 bumps 的 `Parameter.dev(std, mean)`。

真实场景是"我知道这层大概 100 Å 上下 5 Å"，硬边界只能表达"在 80 到 120 之间"，两者的
信息量差一个量级。

## 声明与数学分离

`MODEL_ALLOWED["parameters"] = set()`（`tests/architecture/test_dependency_rules.py:115`）
禁止 `model/parameters.py` import 任何其他 model 模块。因此 model 层只放自洽声明：

```python
@dataclass(frozen=True)
class PriorSpec:
    kind: str                    # "uniform" | "normal" | "lognormal" | "soft_range"
    parameters: tuple[float, ...]
```

`ParameterDefinition` 增加 `prior: PriorSpec | None = None`。默认 `None` 即当前硬边界
行为，**所有现存工程文件零迁移**。

数学放 `evaluation.py`，它已有 `scale_prior_penalty`（`:251`）与 `problem_log_probability`：

- `prior_log_density(definition, value) -> float` — 进目标函数
- `prior_inverse_cdf(definition, u) -> float` — 单位立方体映射
- `prior_bounds(definition) -> tuple[float, float]` — 硬截断，供差分进化用

`prior_inverse_cdf` 即使当前不做 nested sampling 也要现在实现。它是 refnx
`Objective.prior_transform` 的等价物：有了它，接任何单位立方体采样器都是零成本；事后
补则会牵动 `analysis/mcmc.py` 的整条初始化路径（`_initial_walker_matrix`、
`_validated_initial_state`、`_initial_log_probability`）。

## 与既有机制的边界

**先验与硬边界共存，不替代。** `normal` 先验在 `[lower, upper]` 外截断并重归一化。不重
归一化会让差分进化的边界投影与先验密度不一致，产生静默的采样偏差。

**`roughness_fraction` 的先验作用在分数空间。** 该 transform 的物理上界由几何动态决定
（`_roughness_dynamic_uppers`），`_effective_upper` 明确"may tighten, but never widen"。
先验若定义在埃空间，其归一化常数会随几何变化，导致目标函数不连续。

**`soft_range` 借 bumps 的形式**：区间内平坦，区间外高斯衰减。这比截断正态更贴合"我知道
大概范围但不想硬卡死"的真实用法。

## ConfidenceClass 判据的语义修订

这是本设计唯一的语义破坏点，必须单独审。

`analysis/report.py:169-173` 的 `boundary_hits` 判据是 unit 坐标贴边：

```python
if value <= fraction or value >= 1.0 - fraction
```

它直接驱动 `analysis/automatic.py:34` 的 `"parameter boundary hit"` 质量决策。引入先验后
"参数贴边"的含义分裂成两件事：贴的是硬边界（数据把参数推到了模型允许的极限，可疑），
还是落在先验的低密度区（数据与先验冲突，这是**信息**而非缺陷）。

修订方案：`boundary_hits` 保留原义，仅对**无先验**参数判定。有先验的参数改用新判据
`prior_conflicts`：后验中位数偏离先验中心超过 `k` 倍先验标准差。两者都进
`UncertaintyReport` 与 reason codes，语义不混淆。

**不能只加不改**：若沿用旧判据，有强先验的参数会因为先验把它拉到硬边界附近而被误判为
`UNTRUSTED`，分级会静默失真。

## 代码边界

- 改 `src/xrr_fitter/model/parameters.py`：新增 `PriorSpec`，`ParameterDefinition` 加字段。
- 改 `src/xrr_fitter/evaluation.py`：三个先验函数，接入 `problem_log_probability`。
- 改 `src/xrr_fitter/analysis/mcmc.py`：初始化按先验抽样而非按边界均匀。
- 改 `src/xrr_fitter/analysis/report.py` 与 `model/analysis.py`：新增 `prior_conflicts`。
- 改 `src/xrr_fitter/io/codec_declarations.py`：序列化 `PriorSpec`。
- 不改 `fit/` 的优化器路径；差分进化仍在 `prior_bounds` 给出的箱内搜索。
- 扩 `src/xrr_fitter/api.py`：新增 `validate_parameter_priors` / `set_parameter_priors`，
  照 `set_parameter_settings`（`ParametersPanel.set_parameter` 在 `panel.py:131` 的用法）
  按 `dataset_id` 分片，因为先验和参数设置一样是数据集级的。
- 改 `src/xrr_fitter/gui/parameters/{table,panel}.py`。

## 界面

先验挂在参数表上，不另开面板。理由：先验是某个参数的属性，和它的初值/边界并列；拆到别处
用户就得在两个视图间对照参数名。

`ParameterTable`（`table.py:11 HEADERS`）加第七列"先验"，渲染成摘要文本（如
`normal(2.5, 0.3)` 或空表示无先验），列本身**不可编辑**——先验有 kind 和可变元数，塞不进
单元格文本。编辑走右键菜单：`ParametersPanel._show_row_context_menu`（`panel.py:212`）
现在只有"恢复默认值"一项，加"编辑先验"与"清除先验"。

`PriorDialog(QDialog)` 照 `gui/structure/dialogs.py` 的模式：`QComboBox` 选 kind，参数字段
随 kind 切换显隐，`_accept_fields` 一次性构造 `PriorSpec` 并调回调，失败把消息写进
`priorDialogError` 标签且不关闭对话框。构造期错误（元数不匹配、中心越界、lognormal 用于
可能取负的参数）全部在这里可见。

**单位一致性。** `table.py:14 _uses_nm` 对 `.thickness_a` / `.roughness_a` 做 nm 显示换算，
先验的中心与标准差必须走同一换算，否则用户会给一个差 10 倍的先验。对话框用
`display_unit(name)` 标注单位，提交前经 `to_persisted_values` 的同一 scale 折回 Å。
`roughness_fraction` 参数的先验作用在 fraction 空间（无量纲），对话框显式标注"分数"，
不做 nm 换算。

**判据分裂要在结果视图里体现。** `prior_conflicts` 是新判据，`gui/results/uncertainty.py`
既有的 `boundary_hits` 呈现旁边要并列显示 `prior_conflicts`，且两者的文案必须能区分
"贴硬边界（可疑）"与"与先验冲突（信息）"。用同一个列表混排会把设计里刚拆开的语义在界面上
重新糊回去。

## 失败与状态

- `PriorSpec.parameters` 元数与 `kind` 不匹配时在构造期报错，不延迟到求值。
- 先验中心落在 `[lower, upper]` 外时报错。这几乎总是输入错误，静默截断会产生一个密度
  几乎为零的先验并让优化器看到平坦区域。
- 截断后归一化常数下溢（先验与边界几乎不重叠）时报错。
- `lognormal` 用于可能取零或负值的参数（如 `angle_offset_deg`）时报错。

## 验证

- 每种 `kind` 的 `prior_inverse_cdf(prior_cdf(x)) == x` 往返测。
- **`prior=None` 时全量测试逐位不变。** 这是本项的核心安全网，跑完整测试集比对，不抽样。
- 截断归一化测例：截断正态数值积分应得 1.0。
- MCMC 端到端：合成数据加一个偏离真值的强先验，后验应可预测地被拉向先验；先验放宽后
  应回到真值。
- 判据修订测例：有先验参数被先验拉到硬边界附近时，`boundary_hits` 不得报警而
  `prior_conflicts` 应按后验偏离量正确报警。
- `roughness_fraction` 先验在几何变化时归一化常数不变。
- GUI 测例：对 `.thickness_a` 参数在对话框里以 nm 输入的先验，持久化后的 `PriorSpec`
  数值为 Å。这条挡的是最容易出现的 10 倍错误。
- GUI 测例：构造期错误使 `priorDialogError` 可见且项目不变。
- GUI 测例：`prior_conflicts` 与 `boundary_hits` 在结果视图中分开呈现。

## 非目标

- 不实现 nested sampling 本体，只准备单位立方体映射。
- 不做多参数联合先验（相关先验）。
- 不改变标度先验（`scale_prior_*`）的既有实现或语义。
