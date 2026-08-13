# 参数先验分布实施计划

**Spec:** `docs/superpowers/specs/2026-08-09-parameter-priors-design.md`

**Goal:** 把参数的先验知识从硬边界升级为分布。`ParameterDefinition` 新增 `prior: PriorSpec | None = None`，支持四种 kind（`uniform` / `normal` / `lognormal` / `soft_range`），在 `[lower, upper]` 内截断并重归一化。数学放 `evaluation.py`，提供 `prior_log_density` / `prior_inverse_cdf` / `prior_bounds` 三个函数，接入 `problem_log_probability`。`prior=None` 时全链路与 HEAD **逐位相同**。

**Architecture:** 声明与数学分离是 `MODEL_ALLOWED["parameters"] = set()` 强制的——`model/parameters.py` 不能 import 任何东西，所以 `PriorSpec` 只能是自洽的 kind + 元数元组，所有分布计算落在 `evaluation.py`。先验与硬边界**共存不替代**：差分进化仍在 `[lower, upper]` 箱内搜索（`prior_bounds` 就是这个箱），先验只改变 `problem_log_probability` 的密度，因此只影响 MCMC 与不确定度分析，不影响任何优化器路径。

---

## 对 spec 的修正

核对 HEAD 代码后，spec 有 14 处与仓库现状不符或未覆盖的交互。逐条给出证据与处理。

### 修正 1（致命）：`problem_log_probability` 的逐位重放约束，先验项不能随意加

`evaluation.py:2185` 那段有一条显式注释锁死了求和顺序：

```python
# Preserve the frozen sampler grouping: sum point losses before dividing by
# the robust scale. Replacing this with mean * count changes retained log
# probabilities by a few ULPs even though the expressions are algebraically
# equivalent, which breaks deterministic checkpoint and reference replay.
```

`McmcReport.log_probability` 被持久化（`io/codec_results.py:75` 附近的编解码），检查点重放要求逐位一致。spec 说「接入 `problem_log_probability`」，但没说加在哪。**加法位置不同会改变浮点结果**：`(data + prior_a) + prior_b` 与 `data + (prior_a + prior_b)` 不等价。

**处理：先验项必须加在函数体**最末尾**，在既有 `scale_prior_center` 分支之后，且用一条独立的累加循环。** 精确形态：

```python
log_probability = -float(data_loss) / (2.0 * c_decades**2)
if problem.scale_prior_center is not None:      # 既有分支，位置不动
    prior = _scale_prior_residual(problem, observed)
    assert prior is not None
    log_probability -= 0.5 * prior**2
log_probability += _parameter_prior_log_density(problem, unit)   # 新增，只此一行
return float(log_probability)
```

`_parameter_prior_log_density` 内部对各参数先验求和，**无先验时必须返回精确 `0.0`**（不是 `-0.0`、不是 `sum([])` 的结果——`sum([])` 返回 int `0`，`float + 0` 是逐位无损的，但显式 `return 0.0` 更明确）。`x + 0.0 == x` 对所有有限 `x` 成立，包括 `-inf` 之外的一切，所以 `prior=None` 的逐位不变由 IEEE 754 保证而非靠测试运气。

**必须有一条测例直接钉这个：** 对同一个无先验问题，改动前后 `problem_log_probability` 的返回值用 `==` 比较（不是 `allclose`）。这是全计划最重要的一条断言。

### 修正 2：MCMC 初始化不是「按边界均匀」，spec 的第三条代码边界前提错了

spec 说「改 `analysis/mcmc.py`：初始化按先验抽样而非按边界均匀」。仓库的实现（`analysis/mcmc.py:286` `_problem_walkers`）是：

```python
proposal = rng.normal(center, 0.01, size=center.size)
```

`center` 来自 `_validated_candidate(problem, candidate)`（`:395`）——**收敛后的拟合最优解**，抽样是围绕它的半径 0.01 单位球，并用 `isfinite(problem_log_probability(...))` 拒绝无效点。这里从来没有「按边界均匀」这回事。

**处理：不修改 `analysis/mcmc.py` 的初始化算法。** 理由：走链者已从最优解出发，先验通过 `problem_log_probability` 自动生效在 `isfinite` 拒绝与后续的 Metropolis 接受率里；改初始化分布会破坏 `child_seed` 的可重复性契约（`_problem_seeds`（`:280`）用 `SeedSequence.spawn(2)` 分出初始化与采样两个种子，改抽样分布会让所有既有检查点的重放失效）。`analysis/mcmc.py` 仍会在报告阶段读取已映射样本以计算 `prior_conflicts`，但通用采样器与 walker 初始化保持不变。spec 提到的 `_initial_walker_matrix` / `_validated_initial_state` / `_initial_log_probability` 三个函数都在 `run_affine_invariant` 的通用采样器侧（`:119`/`:166`/`:153`），与先验初始化无关。

### 修正 3：`prior_inverse_cdf` 本轮无消费者，但仍要实现

spec 要求现在就实现 `prior_inverse_cdf`，理由是「接任何单位立方体采样器都是零成本」。核实：全项目零 `invcdf`/`ppf`（`rg` 无命中），而修正 2 已确认 MCMC 不用它。所以它本轮**没有生产消费者**。

**处理：照 spec 实现，但验收方式改为纯数学往返测，不假装有集成点。** `prior_inverse_cdf(definition, prior_cdf(definition, x)) == x`（`rtol=1e-10`）对四个 kind 各测一遍。这需要同时实现 `prior_cdf`——spec 的验证一节写了 `prior_cdf(x)` 却没在代码边界里列出这个函数，是遗漏。四个函数一起实现（`prior_log_density` / `prior_cdf` / `prior_inverse_cdf` / `prior_bounds`），往返测是 `prior_inverse_cdf` 唯一的正确性证据。

不实现它的代价确实如 spec 所说：事后补会牵动初始化路径。实现它的代价只是一个未被调用的纯函数加一组测例，`tools/check_radon.py` 不会因未使用而报警（它只看复杂度）。

### 修正 4：`boundary_hits` 有两套实现且引用面达 8 个模块，spec 严重低估影响面

spec 只提 `analysis/report.py:169-173` 一处，实际有**两套语义不同的实现**：

- `report.py:169`：unit 坐标贴边，`value <= fraction or value >= 1.0 - fraction`，**不区分 transform**。
- `mcmc.py:354` `_near_boundary`：**按 transform 分派**——`linear`/`roughness_fraction` 用 unit 坐标，`log` 用物理值相对 `upper - lower` 跨度，未知 transform 抛 `ValueError`。

下游引用共 30 处横跨 8 个模块：`analysis/automatic.py:34`（质量决策）、`analysis/classification.py:163`（reason code `"boundary_hit"`）、`analysis/joint.py:114,151,203`、`analysis/profiles.py:1096`（**决定给哪些参数跑剖面**）、`io/codec_results.py:75,94,117,137,161,187`（两处序列化 + 两处必需键集合）、`model/analysis.py:433,469`（两个报告类型的字段）、`gui/results/uncertainty.py:115,211`。

**处理：本轮不修改 `boundary_hits` 的判据，改为纯新增 `prior_conflicts` 并列。** spec 的「仅对无先验参数判定 `boundary_hits`」是对的判断但代价被低估：改它意味着 `profiles.py:1096` 的剖面选择集合变化（有先验参数不再被选中跑剖面，而剖面是 `profiles_closed` 的输入，进而改 `ConfidenceClass`）、`codec_results.py` 两处必需键的语义变化、以及 `classification.py` 的 reason code 集合变化。这是一条横跨 analysis 全域的语义改动，**放进本计划会让「`prior=None` 逐位不变」这条核心安全网无法验证**——因为改判据会改变有先验时的分级，而分级的正确性没有独立的数学判据可测（不像先验密度有解析式）。

替代方案（本轮采用）：
- `boundary_hits` 语义与实现**一字不动**，两套实现都不动。
- 新增 `prior_conflicts: tuple[str, ...]` 字段，与 `boundary_hits` **并列**进 `UncertaintyReport` 与 `McmcReport`，判据是「后验中位数偏离先验中心超过 `k` 倍先验标准差」。
- `prior_conflicts` **不进** `automatic.py` 的质量决策，也**不进** `classification.py` 的 reason codes。它是纯信息性输出，只在报告与界面上呈现。
- 结果视图并列显示两者且文案区分（spec 这一条保留，见修正 10）。

**spec 说的「不能只加不改」是一个真实风险，本轮明确接受它并记入剩余风险：** 有强先验的参数若被先验拉到硬边界附近，`boundary_hits` 仍会报警，`ConfidenceClass` 可能因此偏保守（报 `CORRELATED` 而非 `TRUSTED`）。这是**偏保守**方向的失真，不会让用户误信一个不该信的结果。反方向（漏报）才是危险的。判据修订应作为独立计划推进，且需要先冻结本轮的 `prior_conflicts` 语义。

### 修正 5：`k` 倍标准差的 `k` 必须可配置且有归属

spec 说「偏离先验中心超过 `k` 倍先验标准差」，没说 `k` 在哪。仓库既有的类似阈值都挂在 `problem.config.confidence` 下，其类型是 `model/fitting.py:166 ConfidenceThresholds`（**不是** `ConfidenceConfig`，全项目无此名），字段 `boundary_fraction: float = 0.005`（`:173`，被 `report.py:168` 与 `mcmc.py:360` 读）与 `strong_correlation: float = 0.95`（`:174`，被 `report.py:177` 读）。

**处理：`k` 放 `ConfidenceThresholds`，命名 `prior_conflict_sigmas`，默认 `3.0`。** 与 `boundary_fraction` / `strong_correlation` 同一个配置对象，读法一致（`problem.config.confidence.prior_conflict_sigmas`）。默认 3.0 的理由：正态先验下 3σ 外的后验中位数对应先验密度已降到峰值的 1.1%，是「数据与先验实质冲突」的常规判据；比这更松（2σ，5%）会在正常拟合噪声下频繁误报。

`ConfidenceThresholds` 是 `@dataclass(frozen=True, slots=True)`，其 `__post_init__`（`:177`）对**全部字段**统一要求有限且非负：

```python
values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
if any(not isfinite(value) or value < 0.0 for value in values):
    raise ValueError("confidence thresholds must be finite and nonnegative")
```

这条遍历自动覆盖新字段，所以 `prior_conflict_sigmas` **不需要自己写校验**就得到「有限且 ≥ 0」。但它需要的是「> 0」（0σ 会让每个有先验参数都报冲突），所以要在 `__post_init__` 末尾单加一条 `if self.prior_conflict_sigmas <= 0.0: raise ValueError(...)`，照 `strong_correlation` 那条区间检查的写法追加。

**这会改 `model/fitting.py` 与其编解码**，spec 的代码边界里没有这两处。编解码侧见修正 8 末段：`_fit_config_to_dict`/`_from_dict` 在 `codec_declarations.py:103`/`:128`，是显式列键式，要手工加并按 optional 读取。

### 修正 6：`soft_range` 的元数与数学形式 spec 未定义

spec 说 `soft_range`「区间内平坦，区间外高斯衰减」，借 bumps 的形式，但没给元数个数与衰减尺度。四个 kind 的 `parameters` 元组长度必须在构造期校验（spec 的失败一节要求「元数与 kind 不匹配时在构造期报错」），所以这个必须定死。

**处理：四个 kind 的元数与语义如下表，写进 `docs/algorithm.md`。**

| kind | `parameters` | 未归一化密度 `p(x)`，`x ∈ [lower, upper]` |
| --- | --- | --- |
| `uniform` | `()` | `1.0` |
| `normal` | `(mean, std)` | `exp(-0.5 * ((x - mean) / std)**2)` |
| `lognormal` | `(log_mean, log_std)` | `exp(-0.5 * ((log(x) - log_mean) / log_std)**2) / x` |
| `soft_range` | `(low, high, std)` | `1.0` 若 `low <= x <= high`；否则以 `std` 为尺度的高斯衰减 |

`lognormal` 的 `parameters` 是**对数空间**的均值与标准差（不是物理空间的），这必须写进文档与对话框标注，否则用户给 `(100.0, 5.0)` 意思是「100 Å 上下 5 Å」而实际得到 `exp(100)`。这是本设计里最容易造成十几个数量级错误的一处。

`normal` 的 `std > 0`、`lognormal` 的 `log_std > 0`、`soft_range` 的 `low < high` 且 `std > 0`，全部构造期校验。

### 修正 7：截断重归一化的归一化常数必须缓存，否则 MCMC 逐点重算

spec 要求「`normal` 先验在 `[lower, upper]` 外截断并重归一化」，理由正确（不重归一化会让差分进化的边界投影与先验密度不一致）。但 `problem_log_probability` 在 MCMC 里被调用 `walkers × steps` 次，每次重算归一化常数（`normal` 需要两次 `erf`，`soft_range` 需要数值积分或分段解析）是纯浪费。

**处理：归一化常数在 `_parameter_prior_log_density` 内部按 `(definition.name, definition.lower, definition.upper)` 缓存于模块级 `dict`，且必须是**纯函数缓存**（相同键必得相同值）。** 用 `functools.lru_cache` 包一个接受**不可变标量**的私有函数——不要缓存 `definition` 对象本身（`ParameterDefinition` 是 `frozen=True, slots=True`，可哈希，但把它当键会让缓存跨候选/跨数据集累积无界）。

**`roughness_fraction` 参数不缓存也不需要缓存**，见修正 9。

### 修正 8（致命）：序列化落点不是 `codec_declarations.py`，而且现有实现会自动破坏旧文件兼容

spec 说「改 `io/codec_declarations.py`：序列化 `PriorSpec`」。**落点错了。** `rg -n 'ParameterDefinition' src/xrr_fitter/io/codec_declarations.py` 零命中；真正的编解码在 `io/codec_candidates.py:41`/`:45`：

```python
def _parameter_definition_to_dict(value: ParameterDefinition) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _parameter_definition_from_dict(value: object) -> ParameterDefinition:
    fields = set(ParameterDefinition.__dataclass_fields__)
    return ParameterDefinition(**_mapping(value, fields, "parameter definition"))
```

两侧都由 `__dataclass_fields__` **自动派生**，被 `io/codec_results.py:214`（写）与 `:265`（读）消费。这个自动派生使 `ParameterDefinition` 加字段成为一次**双向破坏**，而且两个方向都不会有人主动去改代码：

1. **写路径立刻炸。** `_parameter_definition_to_dict` 会把 `PriorSpec` 对象原样放进 dict，`project_to_bytes`（`project_codec.py:516`）的 `json.dumps` 无 `default=` 参数，遇到非 JSON 原生类型抛 `TypeError`，被捕获后转成 `ProjectSchemaError`。**任何带先验的项目一保存就失败**，而且错误信息是泛化的 dumps 报错，排查代价很高。
2. **读路径让所有旧文件失效。** `fields = set(ParameterDefinition.__dataclass_fields__)` 是 `_mapping` 的 `required` 集合（`codec_common.py:80` 同时拒 `missing` 与 `extra`），加了 `prior` 字段后旧文件缺该键即 `missing` 报错。**这不是「忘了改」会发生的问题，是「什么都不改」就必然发生的问题。**

**处理：`_parameter_definition_to_dict` / `_from_dict` 必须改为显式字段处理，放弃 `__dataclass_fields__` 自动派生。** 精确形态：

```python
_DEFINITION_FIELDS = frozenset(ParameterDefinition.__dataclass_fields__) - {"prior"}


def _parameter_definition_to_dict(value: ParameterDefinition) -> dict[str, object]:
    payload = {field: getattr(value, field) for field in _DEFINITION_FIELDS}
    if value.prior is not None:
        payload["prior"] = _prior_to_dict(value.prior)
    return payload


def _parameter_definition_from_dict(value: object) -> ParameterDefinition:
    payload = dict(_mapping(value, set(_DEFINITION_FIELDS), "parameter definition", optional={"prior"}))
    prior = payload.pop("prior", None)
    return ParameterDefinition(**payload, prior=None if prior is None else _prior_from_dict(prior))
```

`_DEFINITION_FIELDS` 从 `__dataclass_fields__` 减去 `prior` 而非硬编码 12 个名字，这样以后再加字段仍会被既有的往返测抓到，不会静默漏发。

**为什么 `prior is None` 时必须完全不发键，而不是发 `"prior": None`：** `OPTIONAL_FIELDS`（`io/codec_common.py:19`）不是「字段可缺省」白名单，而是 `_allows_null`（`:117`）用的「该键的值允许为 JSON `null`」白名单，由 `_validate_nulls`（`:127`）消费，而 `_validate_nulls` 在读路径与**写路径**（`project_codec.py:517`，`project_to_bytes` 内第一件事）都跑。发 `None` 会让所有无先验工程（即绝大多数）在第一次保存时被拒绝写盘。采取条件发键后，`codec_common.py` **完全不需要改**，`OPTIONAL_FIELDS` 也不需要加 `prior`。

同一条推理适用于 `ParameterSetting`：`project_codec.py:114`/`:118` 是同样的 `__dataclass_fields__` 自动派生。**本计划不给 `ParameterSetting` 加任何字段**（先验挂在 `ParameterDefinition` 上，由 `services/parameters.py:37 _default_definitions` 派生），所以那两个函数不动。若实施中发现先验需要持久化在 `ParameterSetting` 侧，那是范围变更，先停下来说明。

**`ConfidenceThresholds`（修正 5 的新字段）的编解码是第三处同类风险。** 它在 `codec_declarations.py:103 _fit_config_to_dict` / `:128 _from_dict` 内，那两个函数是显式列键的（不是自动派生），所以要手工加 `prior_conflict_sigmas` 并按 `optional` 读取。实施时先 `sed -n '103,160p' src/xrr_fitter/io/codec_declarations.py` 看清 `confidence` 子对象的确切构造方式再改。

### 修正 9：`roughness_fraction` 先验的作用空间与归一化常数不变性

spec 说「`roughness_fraction` 的先验作用在分数空间，先验若定义在埃空间，其归一化常数会随几何变化，导致目标函数不连续」。判断正确，核实：`_effective_upper` 的动态上界由 `_roughness_dynamic_uppers`（`evaluation.py:352`）按几何逐次算出，确实会随厚度候选变化。

**处理：`prior_log_density` 对 `transform == "roughness_fraction"` 的参数，`x` 取**unit 坐标**（`[0, 1]` 的分数）而非物理 Å 值，截断区间恒为 `[0, 1]`，归一化常数因此是常数、与几何无关，可安全缓存（修正 7）。** 这也让「归一化常数在几何变化时不变」成为**结构性保证**而非需要测试保护的性质——但 spec 要求的那条测例仍要写，它验的是实现没有意外读物理值。

**由此产生一个必须写进文档与对话框的用户可见后果：** `roughness_fraction` 参数的先验中心与标准差是**无量纲分数**，不是 nm 也不是 Å。用户想表达「粗糙度大概是几何上限的一半」得填 `0.5`。这与其他参数的单位语义完全不同，对话框必须显式标注「分数」（spec 已要求，保留）。

### 修正 10：结果视图的两个判据并列，`gui/results/uncertainty.py` 有两处

spec 要求 `prior_conflicts` 与 `boundary_hits` 在结果视图分开呈现，文案区分「贴硬边界（可疑）」与「与先验冲突（信息）」。核实呈现点有**两处**：`gui/results/uncertainty.py:115`（`UncertaintyReport` 侧，`boundaries = _joined(report.boundary_hits) or "无"`）与 `:211`（`McmcReport` 侧，`f"MCMC 边界命中：..."`）。

**处理：两处都加并列行，文案分别为「先验冲突（信息）」与「MCMC 先验冲突（信息）」，照既有 `_joined(...) or "无"` 的形态。** spec 只说「旁边要并列显示」，漏了 MCMC 那一处；只改一处会让 MCMC 报告里的先验冲突不可见。

### 修正 11：`api.py` 的两个新函数需要按 `dataset_id` 分片，且 `validate_*` 的返回契约要定

spec 说「扩 `src/xrr_fitter/api.py`：新增 `validate_parameter_priors` / `set_parameter_priors`，照 `set_parameter_settings`」。**`api.py` 里不能写函数。** 核实：`api.py` 全文 210 行，`rg -n '^def '` 零命中，只有 `from ... import (...)` 与末尾 `__all__ = (...)`（`:117`）——它是纯 re-export 门面。真正的实现在 `services/parameters.py:172 set_parameter_settings`，由 `api.py:79` re-export。

`services/parameters.py:172` 的既有形态（这是要照抄的模板）：

```python
def set_parameter_settings(
    project: XrrProject,
    dataset_id: str,
    settings: Sequence[ParameterSetting],
) -> XrrProject:
    """Persist validated settings and invalidate their dependent fit state."""
    index = dataset_index(project, dataset_id)
    dataset = project.datasets[index]
    if dataset.structure is None:
        raise ValueError(f"dataset has no structure: {dataset_id}")
    data = _prepared_current(project, dataset)
    definitions = _default_definitions(project, dataset, data, dataset.structure)
    validated = validate_parameter_settings(definitions, settings)
    if validated == dataset.parameter_settings:
        return project
    updated = replace(dataset, parameter_settings=validated)
    return _replace_invalidated(project, index, updated, clear_evidence=False)
```

由此**契约已定，不需要实施时再猜**：`validate_*` 是**返回已校验值**式（`validate_parameter_settings(definitions, settings) -> tuple[ParameterSetting, ...]`），非法输入直接抛异常，不返回错误列表；`set_*` 返回新的 `XrrProject`，无变化时**返回原对象**（`if validated == ...: return project`，这是 GUI 侧判断「是否需要标脏」的依据），并走 `_replace_invalidated` 使依赖的拟合状态失效。

**处理：实现落 `services/parameters.py`，`api.py` 只在 `from xrr_fitter.services.parameters import (...)` 块与 `__all__` 里各加两行。** 三点必须照抄而非自创：

1. **`clear_evidence` 取值。** `set_parameter_settings` 用 `clear_evidence=False`。先验改变会改变 `problem_log_probability`，因此已有的不确定度报告与 MCMC 结果全部失效——但**结构证据（structure evidence）不受先验影响**，所以先验也用 `clear_evidence=False`。实施时读 `_replace_invalidated` 确认它清掉了 `uncertainty` 与 `mcmc`；若没清，需要额外处理，那是范围内的必要修改。
2. **无变化时返回原 `project` 对象**（不是等值的新对象）。这条是行为契约，要有测例钉 `set_parameter_priors(p, id, same) is p`。
3. **先验存哪。** `ParameterDefinition` 是 `_default_definitions` 每次**派生**出来的（`services/parameters.py:37`），不持久化在 dataset 上——所以「设置先验」不能像设置 setting 那样直接写进 `dataset.parameter_settings`。**这是一个未解决的落点问题，实施第一步必须先解决它**：要么给 `DatasetProject` 加一个 `parameter_priors: tuple[...]` 字段（则 `project_codec.py` 的 dataset 编解码要改，且 `_default_definitions` 要把它合进派生结果），要么给 `ParameterSetting` 加 `prior` 字段（则修正 8 末段说的「不给 `ParameterSetting` 加字段」失效，`project_codec.py:114`/`:118` 的自动派生问题原样重演）。**倾向前者**：先验与 `initial`/`lower`/`upper` 的生命周期不同（先验是用户长期知识，setting 是本次拟合的起点），混在一个类型里会让「恢复默认值」这个既有菜单项的语义变得含糊——它该不该清先验？分开存则答案明确：不该。

**这条是本计划唯一一处「计划阶段未定死」的设计决策**，Task 5 的第一步就是它，不要在写完 evaluation 数学后才去想。

### 修正 12：`UncertaintyReport` 有三处构造点，`prior_conflicts` 必须带默认值

spec 说「改 `analysis/report.py` 与 `model/analysis.py`：新增 `prior_conflicts`」，但没说构造点有几处。核实 `rg -n 'UncertaintyReport(' src/`：

- `analysis/report.py:269` — 主计算路径。
- `analysis/joint.py:145` — 联合拟合路径。
- `io/codec_results.py:174` — 反序列化路径。

`McmcReport` 同理有独立的构造点。报告值对象必须带默认值，服务层随后可用 `replace(...)` 给 joint 报告附加映射后的 union；这也让「读旧结果文件」能为缺失的 `prior_conflicts` 落到空值。

**处理：`prior_conflicts: tuple[str, ...] = ()` 带默认值追加在字段末尾。** 这样 codec 反序列化在旧文件缺该键时落到默认空元组，语义正确（旧结果没有先验，自然无冲突）。单候选与 MCMC 在 analysis 层填值；joint 的基础报告先保持默认值，再由 service composition 层复制并附加 local-to-global union。

带默认值的代价是漏传不会报错——所以要有一条测例断言 `analysis/report.py` 路径产出的报告在有先验且冲突时 `prior_conflicts` 非空，否则「忘了接线」会静默表现为「从不报冲突」。这条测例比字段本身重要。

`io/codec_results.py` 侧同样走修正 8 的条件发键：`prior_conflicts == ()` 时**不发键**（避免为空元组污染所有旧格式结果文件的往返比对），非空时发数组。读侧用 `_mapping(..., optional={"prior_conflicts"})`。

### 修正 13：`ConfidenceThresholds.__post_init__` 会自动校验新字段，但 `0.0` 能通过

修正 5 要给 `ConfidenceThresholds` 加 `prior_conflict_sigmas: float = 3.0`。核实它的 `__post_init__`（`model/fitting.py:176`）：

```python
def __post_init__(self) -> None:
    values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("confidence thresholds must be finite and nonnegative")
    if not 0.0 <= self.strong_correlation <= 1.0:
        raise ValueError("strong_correlation must be in [0, 1]")
```

第一条检查遍历 `__dataclass_fields__`，所以新字段**自动**获得「有限且非负」校验，不需要改这个循环。但 `prior_conflict_sigmas = 0.0` 会通过——而 0 倍标准差意味着后验中位数只要不精确等于先验中心就报冲突，即**每个有先验的参数永远报冲突**，判据完全失效且不报错。

**处理：追加一条显式检查 `if self.prior_conflict_sigmas <= 0.0: raise ValueError(...)`，写在 `strong_correlation` 那条之后。** 注意不要把它并进第一个 `any(...)` 循环——那个循环是全字段通用的，`boundary_fraction = 0.0` 是合法的（意为「不做贴边判定」），语义不同不能合并。

同时 `ConfidenceThresholds` 是 `frozen=True, slots=True` 且被 `FitConfig.confidence` 以**默认实例** `ConfidenceThresholds()` 持有（`model/fitting.py:196`）——加带默认值的字段不影响任何既有构造点，这是安全的。

### 修正 14（致命）：三处构造点手上的统计量不同，`prior_conflicts` 判据的输入必须分别定义

修正 12 定了「`prior_conflicts` 加到哪几处」，但没定「用什么算」。核实三个构造点手上的数据（`sed -n '250,295p' report.py`、`120,160p' joint.py`、`mcmc.py:410`）后发现判据文字「后验中位数偏离先验中心超过 `k` 倍先验标准差」对三处含义完全不同，**不定义清楚实施者到 `report.py` 会卡住**：

- **`analysis/report.py:269`** 手上只有 `best.unit_vector`——**单点估计，没有后验分布**，算不出「中位数」。它能算的是「点估计偏离先验中心超过 `k·σ`」。
- **joint 基础报告**手上有 global ensemble，但没有可直接用于 shared/local 坐标的 per-dataset prior；最终实现在 service composition 层对 winning candidate 做 local 判据，再映射成 global-variable union。
- **`McmcReport`（`mcmc.py:410`）** 手上有 `samples_physical`，能算真后验中位数，是三者中唯一名副其实的。

**处理，两条硬约束：**

1. **判据统计量按可得性分级，但函数签名统一。** 在 `evaluation.py` 加 `prior_conflicts(names, centers, spreads, estimates, k) -> tuple[str, ...]`，`estimates` 是「该参数的代表值」——有样本时传中位数，只有点估计时传点估计。**函数不关心 estimate 从哪来**，这样三处复用同一个纯函数，差异只在调用方喂什么。`report.py` 喂点估计并在文案上体现（见下），MCMC 侧喂中位数。
2. **`centers`/`spreads` 从 `PriorSpec` 导出，不是 raw parameters。** `normal` 的中心是 `mean`、尺度是 `std`；`lognormal` 的中心是 `exp(log_mean)`（几何中心，物理空间）、尺度要用 `exp(log_mean)·log_std` 近似（对数正态在物理空间无对称 σ，用一阶展开）；`soft_range` 的中心是 `(low+high)/2`、尺度是 `(high-low)/2 + std`；`uniform` **不参与冲突判定**（无中心可言，跳过）。这套映射必须在 `evaluation.py` 里作 `prior_center_and_spread(spec) -> tuple[float, float] | None`（`uniform` 返回 `None`），**不能散在三个调用点各写一份**。

**取值来源的落点问题**：三处调用点都需要拿到「参数名 → `PriorSpec`」的映射。这依赖修正 11 未定的先验存储位置——所以 **Task 4（存储决策）排在 Task 5（`prior_conflicts` 接线）之前**，否则 Task 5 的三个调用点不知道从哪读先验。这是两个 Task 之间的**顺序约束**，已写进 Task 编号。

**为什么 `report.py` 用点估计不算降级**：单候选路径本就没有后验样本（bootstrap 只产区间不产完整样本矩阵），点估计偏离先验 `k·σ` 是「先验与数据打架」的合法信号，只是比 MCMC 的中位数判据弱。文案上两条路径都用「与先验冲突」，不额外区分——用户看到的是同一类信息，统计强度差异属实现细节，记入剩余风险即可。

以下核对后与代码一致或判断正确，照 spec 执行：

- `PriorSpec` 只放 `kind` + `parameters` 元组，数学全在 `evaluation.py`——`MODEL_ALLOWED["parameters"] = set()`（`tests/architecture/test_dependency_rules.py:116`）确实禁止 `model/parameters.py` import 任何东西，核实无误。
- `ParameterDefinition` 加 `prior: PriorSpec | None = None`，默认 `None` 零迁移。字段末尾是 `sharing_key: str | None = None`（`model/parameters.py:52`），`prior` 追加在它之后。
- 先验与硬边界共存不替代，`prior_bounds` 就是 `(lower, upper)`，差分进化路径不改——`fit/` 完全不动。
- 不改标度先验（`scale_prior_*`）的实现或语义。核实 `scale_prior_penalty`（`evaluation.py:251`）与 `problem_log_probability` 里的 `scale_prior_center` 分支是独立机制，两者不交互。
- 构造期报错四条：元数与 kind 不匹配、先验中心落在 `[lower, upper]` 外、截断后归一化常数下溢、`lognormal` 用于可能取零或负的参数。注意最后一条的判据是 `definition.lower <= 0.0`（不是「参数名看起来像角度」）。
- 参数表加第七列「先验」，渲染摘要文本，列不可编辑，编辑走右键菜单。`ParameterTable.HEADERS`（`gui/parameters/table.py:11`）与 `ParametersPanel._show_row_context_menu`（`panel.py:212`）的位置照 spec 核对。
- `PriorDialog` 照 `gui/structure/dialogs.py` 的模式：`QComboBox` 选 kind、参数字段随 kind 切换显隐、`_accept_fields` 一次性构造并调回调、失败写 `priorDialogError` 且不关闭。
- nm 换算走 `table.py:14 _uses_nm` 的同一套（`.thickness_a` / `.roughness_a`），提交前折回 Å。
- 非目标：不实现 nested sampling 本体、不做多参数联合先验。

---

## Global Constraints

- **`prior=None` 时全链路逐位相同。** 这是核心安全网。修正 1 的 `+ 0.0` 结构性保证 + 一条 `==` 断言的测例，两者都要有。
- `fit/` 目录**完全不改**。差分进化与局部搜索都在 `prior_bounds` 给出的箱内，先验不进优化器。
- `analysis/mcmc.py` **完全不改**（修正 2）。
- `boundary_hits` 的判据与两套实现**一字不动**（修正 4）。
- 遵守 `tests/architecture/test_dependency_rules.py` 的 `ALLOWED`：`model:{model}`、`io:{io,model}`、`analysis:{analysis,model,physics}`、`api:{model,services}`、`gui:{gui,api}`。实施前用 `rg -n 'ALLOWED = ' -A 20 tests/architecture/test_dependency_rules.py` 核对 `analysis` 那一行的确切内容。
- `MODEL_ALLOWED["parameters"] = set()`：`model/parameters.py` 不能 import 任何 model 模块，`PriorSpec` 的全部校验必须在该文件内自洽。
- `MODEL_ALLOWED["analysis"] = {"data", "parameters", "fitting"}`：`model/analysis.py` 加 `prior_conflicts` 字段可行（不需要新 import，字段类型是 `tuple[str, ...]`）。
- `schema_version` **不升版**。条件发键 + `optional` 读取使新旧双向兼容，与仓库既有 `dock_state` 处理一致。
- 禁止 `pytest.skip` / `xfail` / 条件收集：`tests/outcome_gate.py` 会因 `skipped`/`xfailed`/`xpassed`/`deselected` 让整轮失败。
- 新增测试文件落在既有整目录注册下（`tests/unit/model`、`tests/unit/analysis`、`tests/unit/io`、`tests/gui`），**本计划不需要改 `tools/verify_registry.py`**。`tests/unit/test_evaluation.py` 是根目录单文件，已注册。
- 测试模块名不得以父目录名开头（`tests/architecture/test_naming_rules.py`）。
- `tests/conftest.py` 必须保持为仓库唯一 `conftest.py`。
- 新代码必须过 `tools/check_radon.py`（单块 CC ≤ 10、单文件平均 CC ≤ 5.0、MI 级别 A、仓库均值 ≤ 5.0）。四 kind × 四函数 = 16 个分支组合，必须用分派表而非 `if/elif` 链，见 Task 2。
- 跑测试必须带 `--import-mode=importlib`，解释器用 `.venv/bin/python`（只有它能 `import xrr_fitter`）。`tools/verify.py` 在本地会因 repo 内 `.venv` 被 `check_hygiene.py` 判 "generated directory inside repository" 而失败，直接调 pytest。
- 用户可见文案用中文。长度单位在界面上 nm、数据层 Å；`roughness_fraction` 的先验是无量纲分数，两侧都不换算（修正 9）。
- 不 stage 或修改 `.claude/` 与仓库根的 probe 文件。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `src/xrr_fitter/model/parameters.py` | 新增 `PRIOR_KINDS`、`PriorSpec`；`ParameterDefinition` 加 `prior` 字段与交叉校验。 |
| `src/xrr_fitter/model/fitting.py` | `ConfidenceThresholds`（`:166`，非 `ConfidenceConfig`）加 `prior_conflict_sigmas: float = 3.0` + `> 0` 校验（修正 5、13）。 |
| `src/xrr_fitter/model/analysis.py` | `UncertaintyReport` 与 `McmcReport` 各加 `prior_conflicts: tuple[str, ...]`。 |
| `src/xrr_fitter/evaluation.py` | `prior_log_density` / `prior_cdf` / `prior_inverse_cdf` / `prior_bounds`；`_parameter_prior_log_density` 接入 `problem_log_probability` 末尾一行；`prior_center_and_spread` 数学辅助。 |
| `src/xrr_fitter/analysis/mcmc.py` | `prior_conflicts(problem, representative_unit)` 共用点估计判据；MCMC 路径对物理样本取中位数，`roughness_fraction` 对 unit 分数取中位数。 |
| `src/xrr_fitter/analysis/report.py` | 计算点估计 `prior_conflicts` 并填进 `UncertaintyReport`。 |
| `src/xrr_fitter/services/fitting_phases/joint_analysis.py` | 将各 dataset 的局部冲突映射、去重为有序 global-variable union。 |
| `src/xrr_fitter/io/codec_candidates.py` | `ParameterDefinition` 编解码（`:41`/`:45`，**自动派生式**）条件发 `prior` 键、`optional` 读取（修正 8）。 |
| `src/xrr_fitter/io/codec_declarations.py` | `FitConfig` 编解码（`:103`/`:128`，**显式列键式**）为 `ConfidenceThresholds.prior_conflict_sigmas` 手工加键、按 `optional` 读取（修正 5、8 末段）。 |
| `src/xrr_fitter/io/codec_results.py` | 两个报告的 `prior_conflicts` 条件发键/optional 读取：`_mcmc_to_dict`(`:60`)/`_mcmc_from_dict`(`:82`，required 集 `:85`) 与 `_uncertainty_to_dict`(`:126`)/`_uncertainty_from_dict`(`:152`，required 集 `:155`)。 |
| `src/xrr_fitter/api.py` | re-export `PriorSpec`；新增 `validate_parameter_priors` / `set_parameter_priors`（修正 11）。 |
| `src/xrr_fitter/gui/parameters/table.py` | `HEADERS` 加「先验」列，渲染摘要文本，不可编辑。 |
| `src/xrr_fitter/gui/parameters/panel.py` | 右键菜单加「编辑先验」「清除先验」。 |
| `src/xrr_fitter/gui/parameters/dialogs.py` | `PriorDialog`（若该文件不存在则新建，先 `ls src/xrr_fitter/gui/parameters/` 确认）。 |
| `src/xrr_fitter/gui/results/uncertainty.py` | `:115` 与 `:211` 两处并列显示 `prior_conflicts`（修正 10）。 |
| `tests/unit/model/test_parameters.py` | `PriorSpec` 构造期校验、`ParameterDefinition` 交叉校验（既有文件）。 |
| `tests/unit/test_evaluation.py` | 四 kind 的密度/CDF/逆 CDF 往返、截断归一化、逐位不变（既有文件）。 |
| `tests/unit/analysis/` | `prior_conflicts` 判据（既有目录，文件名按既有命名核对）。 |
| `tests/unit/io/test_project_codec.py` | 往返与旧文件兼容（既有文件）。 |
| `tests/gui/test_parameter_table.py` | 先验列、右键菜单、对话框、nm 换算（既有文件）。 |
| `tests/gui/test_results.py` 或 `test_uncertainty_dialog.py` | 两个判据并列呈现（按既有归属核对）。 |
| `docs/algorithm.md` | 四 kind 的密度公式与元数语义、截断归一化、`roughness_fraction` 的分数空间、`prior_conflict_sigmas` 默认值理由。 |

---

## Tasks

**Task 编号即执行顺序**，依赖已排平：数学叶子（Task 1、2）→ 阈值配置（Task 3）→ 先验存储（Task 4）→ `prior_conflicts` 接线（Task 5，同时依赖 Task 3 的 `k` 与 Task 4 的存储）→ GUI（Task 6）。修正 14 的硬约束「存储决策先于接线」由 Task 4 排在 Task 5 之前满足。

### Task 1: `PriorSpec` 值类型与 `ParameterDefinition.prior` 字段

**Files:**
- Modify: `src/xrr_fitter/model/parameters.py`（`TRANSFORMS` 常量区之后加 `PRIOR_KINDS` 与 `PriorSpec`；`ParameterDefinition` 追加 `prior` 字段与交叉校验）
- Modify: `tests/unit/model/test_parameters.py`

**Interfaces:**
- Produces: `PRIOR_KINDS: frozenset[str]`、`PriorSpec(kind: str, parameters: tuple[float, ...])`、`ParameterDefinition.prior: PriorSpec | None = None`。
- Preserves: `ParameterDefinition` 既有字段的**顺序与默认值**一字不动（`parameters.py:40-52`）；`prior` 只能追加在末尾，否则所有位置构造的调用点全崩（修正 12 同类风险）。`ParameterSetting`、`ParameterCoordinate` 等其余类型不动。
- Removes: 无。

**为什么先做这一层且不 import 任何东西：** `MODEL_ALLOWED["parameters"] == set()`（`test_dependency_rules.py:111`）禁止 `model/parameters.py` import 任何模块——所以 `PriorSpec` 只能做**纯声明与构造期校验**，四个 kind 的密度数学一律落 Task 2 的 `evaluation.py`。这一层错了后面全错，但它能单独验证的只有「参数个数与 kind 匹配」「数值有限」这类结构契约。

**复杂度约束：** kind → 期望参数个数的校验若写成 `if/elif` 链会顶到 CC 5+。用模块级 `_PRIOR_ARITY: dict[str, int]` 表（`uniform:0, normal:2, lognormal:2, soft_range:3`），`PriorSpec.__post_init__` 查表，CC 保持 2。`PRIOR_KINDS = frozenset(_PRIOR_ARITY)` 从表派生，不手写第二份。

 - [x] **Step 1: 写失败的构造契约**

  在 `tests/unit/model/test_parameters.py` 追加，全部必须失败（`PriorSpec` 不存在）：

  - `test_prior_spec_accepts_each_kind_with_correct_arity`：四个 kind 各用正确参数个数构造成功，`kind`/`parameters` 原样存下（`parameters` 是 `tuple`）。
  - `test_prior_spec_rejects_unknown_kind`：`PriorSpec("gaussian", (0.0, 1.0))` 抛 `ValueError`，消息含 `"gaussian"`。
  - `test_prior_spec_rejects_wrong_arity`：`PriorSpec("normal", (0.0,))`、`PriorSpec("uniform", (1.0,))` 各抛 `ValueError`。这条钉住修正 6 的元数契约。
  - `test_prior_spec_rejects_nonfinite_parameters`：`normal` 的参数含 `nan`/`inf` 抛 `ValueError`。
  - `test_prior_spec_rejects_nonpositive_scale`：`normal` 的 `std=0.0`/负值、`lognormal` 的 `log_std<=0`、`soft_range` 的 `std<0` 抛 `ValueError`（尺度参数必须正；`soft_range` 的 `std` 允许 0 意为硬边界——按修正 6 的定义核对后决定 `>=0` 还是 `>0`，测例以修正 6 为准）。
  - `test_parameter_definition_defaults_prior_to_none`：既有五参构造的 `ParameterDefinition`，`prior is None`。这条守零迁移。
  - `test_parameter_definition_accepts_a_prior`：带 `prior=PriorSpec("normal", (1.0, 0.2))` 构造成功。
  - `test_parameter_definition_rejects_prior_center_outside_bounds`：`lower=0, upper=1` 但 `normal` 中心 `mean=5.0` 时抛 `ValueError`（交叉校验：先验中心须落在 `[lower, upper]` 内，否则先验与边界自相矛盾）。`uniform` 无中心，跳过此校验。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model/test_parameters.py -q
  ```

  预期 `ImportError: cannot import name 'PriorSpec'` 导致整个文件收集失败（含既有测例）。这是正常 RED，不要拆文件让旧测例先绿。

 - [x] **Step 3: 实现 `PriorSpec` 与字段**

  `src/xrr_fitter/model/parameters.py`：

  - `_PRIOR_ARITY: dict[str, int] = {"uniform": 0, "normal": 2, "lognormal": 2, "soft_range": 3}` 放 `TRANSFORMS` 之后；`PRIOR_KINDS = frozenset(_PRIOR_ARITY)`。
  - `PriorSpec`（`frozen=True, slots=True`）：`kind: str`、`parameters: tuple[float, ...]`。`__post_init__` 顺序：tupleize `parameters`（照既有防御式写法回写 `object.__setattr__`）→ kind 在 `PRIOR_KINDS` 内否则 `ValueError` → 个数等于 `_PRIOR_ARITY[kind]` 否则 `ValueError` → 全部有限 → 尺度参数按 kind 取对应下标校验正性。尺度校验用一个模块级 `_prior_scale_indices: dict[str, tuple[int, ...]]` 表避免 `if/elif`。
  - `ParameterDefinition` 追加 `prior: PriorSpec | None = None`；`__post_init__` 末尾追加 `_definition_prior(self.name, self.lower, self.upper, self.prior)` 助手调用（类型校验 + 中心落界校验）。**别塞进 `__post_init__` 正文**——它当前 CC 已到 5（`parameters.py:54-62`），加分支会超。助手内的「先验中心」需要一个只看 kind 的小映射（`normal`/`lognormal` 用第一个参数、`soft_range` 用 `(low+high)/2`、`uniform` 返回 `None` 跳过），这个映射与修正 14 的 `prior_center_and_spread` 语义一致但**不能 import evaluation**（依赖表禁止），所以这里放一份轻量版只算中心，Task 2 的完整版放 evaluation。两份用 Task 2 的一条等值测例钉住（同 plan 2 Task 2 的 kind 集合双写手法）。

 - [x] **Step 4: 确认 GREEN 与门禁**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model/test_parameters.py tests/architecture -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  `tests/architecture` 必须跟跑——它守 `model/parameters.py` 的零 import 约束。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/model/parameters.py tests/unit/model/test_parameters.py
  git commit --no-verify -m "model: declare PriorSpec value type with construction-time validation"
  ```

### Task 2: 四个先验数学函数 + 接入 `problem_log_probability`

**Files:**
- Modify: `src/xrr_fitter/evaluation.py`（四个 `prior_*` 函数、`prior_center_and_spread`、`_parameter_prior_log_density`、`problem_log_probability` 末尾一行）
- Modify: `tests/unit/test_evaluation.py`

**Interfaces:**
- Produces: `prior_log_density(spec, x)`、`prior_cdf(spec, x)`、`prior_inverse_cdf(spec, u)`、`prior_bounds(spec)`、`prior_center_and_spread(spec) -> tuple[float, float] | None`。四者均按 kind 分派。
- Consumes: `model.parameters.PriorSpec`（`ALLOWED["analysis"]` 及 evaluation 自身可 import model）。
- Preserves: `problem_log_probability`（`evaluation.py:2135`）在**所有参数无先验时的逐位输出**——这是全计划的核心安全网（修正 1）。
- Removes: 无。

**逐位不变是本任务的成败判据（修正 1）：** `:2174` 的注释禁止改动求和分组。先验项只能作为**最后一行** `log_probability += _parameter_prior_log_density(problem, unit)` 追加，且 `_parameter_prior_log_density` 在无任何先验时返回**精确 `0.0`**（不是近似 0），靠 IEEE 754 `x + 0.0 == x` 保证旧行为逐位重放。这条比任何先验密度的正确性都重要——它错了，所有既有拟合结果的检查点重放全部漂移。

**复杂度约束（修正 3、7）：** 4 kind × 4 函数 = 16 分支，全部用**分派表**而非 `if/elif`：每个 `prior_*` 一个 `dict[str, Callable]`，每个 kind 一个独立小函数。`lognormal` 与 `soft_range` 的截断重归一化常数必须在 `PriorSpec` 首次求值时算一次并缓存（修正 7）——但 `PriorSpec` 是 frozen，缓存放 evaluation 侧的 `functools.lru_cache` 或模块级 `dict` 键上 `(spec, lower, upper)`，不回写值对象。

 - [x] **Step 1: 写失败的数学契约**

  在 `tests/unit/test_evaluation.py` 追加。**第一条是核心安全网，必须在任何密度测例之前写**：

  - `test_problem_log_probability_is_bitwise_unchanged_without_priors`：取一个既有的无先验 `problem` 与 `unit`，`assert problem_log_probability(problem, unit) == _baseline`，其中 `_baseline` 用当前 HEAD 的同一调用现算并硬编码进测试（或用同一函数在测试内算两次比对）。用 `==` 不用 `approx`。这条现在（Task 2 未接入时）就应通过，Step 3 接入后**必须仍逐位通过**。
  - `test_prior_log_density_matches_closed_form`：四 kind 各取解析式手算一点比对（`uniform→0.0`；`normal` 在中心处 `→0.0`、偏移 `1σ` 处 `→-0.5`；`lognormal` 含 `-log(x)` 项；`soft_range` 区间内 `→0.0`、区间外 Gaussian 衰减）。
  - `test_prior_cdf_is_monotone_and_spans_zero_to_one`：`u = prior_cdf(spec, linspace(lower, upper, 257))` 单调不减、`u[0]` 近 0、`u[-1]` 近 1（截断重归一化后端点精确到 0/1，用 `approx` 容浮点）。
  - `test_prior_inverse_cdf_round_trips`：`prior_inverse_cdf(spec, prior_cdf(spec, x)) ≈ x`（修正 3：本轮无消费者但仍验证）。
  - `test_prior_bounds_are_respected`：`prior_bounds(spec)` 返回的区间与 spec 参数一致。
  - `test_prior_center_and_spread_maps_each_kind`：按修正 14 的映射逐 kind 断言；`uniform → None`。
  - `test_prior_log_density_normalization_constant_is_cached`：同一 `(spec, lower, upper)` 多次调用不重算归一化常数（用 `monkeypatch` 计数或断言缓存命中）。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/test_evaluation.py -q
  ```

  预期除逐位不变那条外全 RED（`prior_log_density` 不存在）。逐位不变那条应已 GREEN——它是接入前的基线锚。

 - [x] **Step 3: 实现四函数并接入**

  `src/xrr_fitter/evaluation.py`：

  - 四个分派表 `_PRIOR_LOG_DENSITY`、`_PRIOR_CDF`、`_PRIOR_INVERSE_CDF`、`_PRIOR_BOUNDS`，键是 kind，值是接收 `(params, x, lower, upper)` 的小函数。密度按修正 6/9 的解析式逐字实现；`lognormal` 的参数在 log 空间、密度含 `1/x` 雅可比。
  - `prior_center_and_spread(spec)` 按修正 14 的映射；`uniform` 返回 `None`。
  - 截断重归一化：`_prior_norm(spec, lower, upper)` 用 `prior_cdf` 的未归一化版在端点求值算分母，`functools.lru_cache` 缓存。
  - `_parameter_prior_log_density(problem, unit)`：遍历 `problem.variables`，对有 `prior` 的变量把 unit 坐标反变换回物理值、累加 `prior_log_density`。**无任何先验时提前返回 `0.0`**（遍历到底也没有先验则返回字面 `0.0`）。
  - `problem_log_probability` 在 `scale_prior` 分支之后追加**唯一一行** `log_probability += _parameter_prior_log_density(problem, unit)`。不动任何既有求和。

 - [x] **Step 4: 确认 GREEN 且逐位不变仍成立**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/test_evaluation.py tests/unit/analysis/test_mcmc.py -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  `test_mcmc.py` 跟跑是因为它重放持久化的 `log_probability`——逐位漂移会在这里炸。radon 16 分支若某个 `prior_*` 超 CC 10，说明分派没拆干净。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/evaluation.py tests/unit/test_evaluation.py
  git commit --no-verify -m "evaluation: add prior densities and wire into problem_log_probability"
  ```

### Task 3: `ConfidenceThresholds.prior_conflict_sigmas` 与 `FitConfig` 编解码

**Files:**
- Modify: `src/xrr_fitter/model/fitting.py`（`ConfidenceThresholds` 加字段 + 显式 `> 0` 校验）
- Modify: `src/xrr_fitter/io/codec_declarations.py`（`_fit_config_to_dict:103` / `_from_dict:128` 手工加键）
- Modify: `tests/unit/model/test_fitting.py`（或既有的 fitting 配置测试文件，按归属核对）
- Modify: `tests/unit/io/test_project_codec.py`

**Interfaces:**
- Produces: `ConfidenceThresholds.prior_conflict_sigmas: float = 3.0`。
- Preserves: `ConfidenceThresholds` 既有六字段顺序与默认值（`fitting.py:166-183`）；`FitConfig.confidence` 的默认实例构造（`:196`）不受影响（新字段带默认值）。
- Removes: 无。

**为什么独立成 Task 且排在接线前：** `prior_conflict_sigmas` 是 Task 5 判据的 `k`，必须先存在。它是纯配置叶子，不依赖 Task 1/2 的任何产物，本可与它们并行，但排在接线（Task 5）前即可。**两个致命细节（修正 13）**：① `__post_init__` 的 `any(...)` 循环遍历 `__dataclass_fields__`，新字段**自动**获得「有限非负」校验，不改循环；② 但 `0.0` 会通过该循环而使判据失效（0σ 意味着每个有先验参数永远报冲突），必须在 `strong_correlation` 校验之后**追加一条** `if self.prior_conflict_sigmas <= 0.0: raise ValueError(...)`，不能并进 `any(...)` 循环——`boundary_fraction = 0.0` 是合法的，语义不同。

**编解码的第三处同类风险（修正 8 末段）：** `FitConfig` 的 codec 在 `codec_declarations.py:103/:128`，是**显式列键式**（不是 `codec_candidates.py` 那种自动派生）。要在 `_fit_config_to_dict` 手工加 `prior_conflict_sigmas` 键、在 `_from_dict` 按 `optional` 读取（旧文件缺键落默认 `3.0`）。实施前先 `sed -n '103,160p' src/xrr_fitter/io/codec_declarations.py` 看清 `confidence` 子对象的确切构造方式。

 - [x] **Step 1: 写失败的配置与往返契约**

  - `test_confidence_thresholds_default_prior_conflict_sigmas_is_three`：默认实例 `.prior_conflict_sigmas == 3.0`。
  - `test_confidence_thresholds_rejects_nonpositive_sigmas`：`prior_conflict_sigmas=0.0` 与负值各抛 `ValueError`；同时断言 `boundary_fraction=0.0` **仍合法**（守住修正 13 的语义边界，别把 0 一刀切禁掉）。
  - `test_confidence_thresholds_rejects_nonfinite_sigmas`：`nan`/`inf` 抛 `ValueError`（由既有 `any(...)` 循环覆盖，这条确认它确实生效）。
  - 在 `test_project_codec.py` 加 `test_fit_config_roundtrips_prior_conflict_sigmas`：非默认值（如 `2.5`）经编解码后逐位相等。
  - 在 `test_project_codec.py` 加 `test_old_fit_config_without_sigmas_decodes_to_default`：手造一个缺 `prior_conflict_sigmas` 键的 payload，解码得 `3.0`（旧文件兼容）。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model tests/unit/io/test_project_codec.py -q -k "sigmas or prior_conflict"
  ```

 - [x] **Step 3: 实现字段、校验与 codec**

  按上文两个修正实施。`ConfidenceThresholds` 加字段在 `strong_correlation` **之后**（末尾），加显式 `> 0` 校验；`codec_declarations.py` 两处显式加键/optional 读取。

 - [x] **Step 4: 确认 GREEN**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model tests/unit/io -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  全 `tests/unit/io` 跟跑确认没碰坏其它 codec 的往返。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/model/fitting.py src/xrr_fitter/io/codec_declarations.py tests/unit/model tests/unit/io/test_project_codec.py
  git commit --no-verify -m "model: add configurable prior-conflict sigma threshold"
  ```

### Task 4: 先验存储与服务层读写（存储决策在此落地）

**Files:**
- Modify: `src/xrr_fitter/model/project.py`（`DatasetProject` 追加 `parameter_priors` 字段）
- Modify: `src/xrr_fitter/io/project_codec.py`（`_dataset_to_dict:262` / `_from_dict:313` / `_dataset_fields:300` 三处同步）
- Modify: `src/xrr_fitter/services/parameters.py`（`set_parameter_priors` / `validate_parameter_priors`；把先验合进 `compiled_parameter_definitions` 路径）
- Modify: `src/xrr_fitter/api.py`（import 块 + `__all__` 各加两行）
- Modify: `tests/unit/model/test_project.py`、`tests/unit/services/test_parameters.py`、`tests/unit/io/test_project_codec.py`

**Interfaces:**
- Produces: `DatasetProject.parameter_priors: tuple[ParameterSetting-like, ...]`（结构见下）、`services.parameters.set_parameter_priors(project, dataset_id, priors) -> XrrProject`、`validate_parameter_priors(...) -> tuple[...]`。
- Preserves: `DatasetProject` 既有字段顺序（新字段追加在 `automation` 之前还是之后需核对——见下）；`set_parameter_settings` 的三条行为契约（修正 11）。
- Removes: 无。

**存储决策：加 `DatasetProject.parameter_priors`，不加 `ParameterSetting.prior`（修正 11 倾向已被代码证据坐实）。** 核对 HEAD 后证据比计划阶段更硬：

1. `parameter_settings` 本就是 `DatasetProject` 的带默认字段（`project.py:192`），`parameter_priors` 与它**完全同构**——`_dataset_to_dict`/`_from_dict`/`_dataset_fields` 三处照 `parameter_settings` 抄一份，codec 风险是已知且已被现有代码验证可控的那一类，不是 `codec_candidates.py` 的自动派生陷阱。
2. `_default_definitions`（`services/parameters.py:37`）**根本不碰 `parameter_settings`**——它纯从 `data/structure/instrument/fit_config` 派生。带 `parameter_settings` 语义的是 `describe_parameters` 走的 `compiled_parameter_definitions`（`:64`）。所以**先验注入点是 `compiled_parameter_definitions` 这条路径**，与 `parameter_settings` 的注入点同源。若改走 `ParameterSetting.prior`，则要动 `compiled_parameter_definitions` 内部把 setting 的 prior 拆出来，反而更绕。
3. 「恢复默认值」语义（修正 11）：先验独立存储 → 该菜单项清 setting 不清先验，答案明确。

**未定的一个小点（实施第一步核对）：** `parameter_priors` 的元素类型。用一个新的轻量 `ParameterPrior(name: str, prior: PriorSpec)`（放 `model/parameters.py`）比复用 `ParameterSetting` 干净——`ParameterSetting` 带 `initial/lower/upper/locked`，先验只需要 `name → PriorSpec`。新增 `ParameterPrior` 会多一处 codec（`project_codec.py` 加 `_parameter_prior_to_dict`/`_from_dict`，显式列键式，非自动派生），但类型清晰。**这是本 Task 唯一的类型选择，实施第一步定，别拖到写 service 时再回头改。**

**字段追加位置：** `DatasetProject` 末尾字段是 `automation: DatasetAutomation = DatasetAutomation()`（`:196`）。新字段追加在**它之后**（带默认值），并同步 `_dataset_fields()` 集合与 `_dataset_from_dict` 的 optional 处理（旧文件无此键 → 空元组）。**必须核对**：`_dataset_from_dict` 用 `_mapping(value, _dataset_fields(), "dataset", {"display_name"})`——`parameter_priors` 要么进 `_dataset_fields()` 且旧文件补默认，要么进 optional 集 `{"display_name", "parameter_priors"}`。选后者：旧项目文件无此键，落空元组，语义正确。

 - [x] **Step 1: 定类型 + 写失败的服务契约**

  先在 `model/parameters.py` 加 `ParameterPrior`（`frozen, slots`；`name` 非空、`prior` 是 `PriorSpec`）。然后测试：

  - `test_dataset_defaults_parameter_priors_to_empty`：既有构造的 `DatasetProject`，`parameter_priors == ()`。守零迁移。
  - `test_set_parameter_priors_returns_new_project`：设一个先验后返回的是新 `XrrProject`，且 `parameter_priors` 生效。
  - `test_set_parameter_priors_returns_same_object_when_unchanged`：设与现有相等的先验，`set_parameter_priors(p, id, same) is p`（照 `set_parameter_settings` 的 `if validated == ...: return project` 契约，修正 11 第 2 点）。
  - `test_set_parameter_priors_invalidates_fit_state`：设先验后 `last_valid_result`/`checkpoint` 被清（`_replace_invalidated` 走 `clear_evidence=False`，先验不影响结构证据但使拟合结果失效——实施时读 `_replace_invalidated` 确认它清 `uncertainty`/`mcmc`，修正 11 第 1 点）。
  - `test_validate_parameter_priors_rejects_unknown_parameter`：先验指向不存在的参数名抛 `ValueError`。
  - `test_validate_parameter_priors_rejects_center_outside_bounds`：先验中心落在该参数 `[lower, upper]` 外抛 `ValueError`（复用 Task 1 的交叉校验，但这里边界来自 `compiled_parameter_definitions` 的实际定义）。
  - `test_compiled_definitions_carry_stored_priors`：设先验后 `describe_parameters` 返回的定义里对应参数 `.prior` 非 `None`。这条钉住「注入 `compiled_parameter_definitions` 路径」。
  - 在 `test_project_codec.py` 加 `test_project_roundtrips_parameter_priors` 与 `test_old_project_without_priors_decodes_to_empty`。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model tests/unit/services/test_parameters.py tests/unit/io/test_project_codec.py -q -k "prior"
  ```

 - [x] **Step 3: 实现存储、codec 与服务函数**

  照 `set_parameter_settings`（`services/parameters.py:172`）逐行仿写 `set_parameter_priors`：`dataset_index` → `_prepared_current` → 取 `compiled_parameter_definitions` 的实际定义做校验 → `validate_parameter_priors` 返回已校验元组 → `if validated == dataset.parameter_priors: return project` → `replace(dataset, parameter_priors=validated)` → `_replace_invalidated(..., clear_evidence=False)`。把先验合进定义：在 `compiled_parameter_definitions` 返回后、或在其内部，按 `name` 把 `parameter_priors` 的 `PriorSpec` 贴到对应 `ParameterDefinition.prior`（用 `replace`）。`api.py` 在 `from xrr_fitter.services.parameters import (...)`（`:79`）与 `__all__` 各加两行。

 - [x] **Step 4: 确认 GREEN 与架构门禁**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit tests/architecture -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  全 `tests/unit` 跟跑——存储字段动了 `project.py` 与 `project_codec.py`，影响面广，跑窄了会漏。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/model/project.py src/xrr_fitter/model/parameters.py src/xrr_fitter/io/project_codec.py src/xrr_fitter/services/parameters.py src/xrr_fitter/api.py tests/unit/model tests/unit/services/test_parameters.py tests/unit/io/test_project_codec.py
  git commit --no-verify -m "services: store and manage per-parameter priors"
  ```

### Task 5: `prior_conflicts` 字段与三条路径接线

**Files:**
- Modify: `src/xrr_fitter/model/analysis.py`（`UncertaintyReport` 与 `McmcReport` 各加 `prior_conflicts` 字段）
- Modify: `src/xrr_fitter/analysis/report.py:269`（单候选路径喂点估计）
- Modify: `src/xrr_fitter/analysis/mcmc.py:410`（MCMC 路径喂中位数，完整判据）
- Modify: `src/xrr_fitter/services/fitting_phases/joint_analysis.py`（联合路径把局部冲突映射为 global-variable union）
- Modify: `src/xrr_fitter/io/codec_results.py`（两个报告的条件发键/optional 读取）
- Modify: `tests/unit/analysis/test_report.py`、`test_mcmc.py`、`tests/unit/io/test_project_codec.py`

**Interfaces:**
- Produces: `UncertaintyReport.prior_conflicts: tuple[str, ...] = ()`、`McmcReport.prior_conflicts: tuple[str, ...] = ()`。
- Consumes: Task 2 的先验数学与 `prior_center_and_spread`；Task 3 的 `thresholds.prior_conflict_sigmas`；Task 4 存储的先验（经 `problem.parameter_definitions[...].prior` 到达）。共用判据实际位于 `analysis/mcmc.py`，避免 `evaluation.py` 同时承担分布数学和报告编排。
- Preserves: 两个报告的既有字段顺序与默认值；三处构造点全为关键字构造，追加带默认字段安全（修正 12）；`prior_conflicts` **不进入任何决策集合**——尤其 `profiles.py:1096` 的 `_reported_profile_names` 只吃 `boundary_hits`，绝不能顺手把 `prior_conflicts` 并进去（会改剖面选择→改分级，破坏修正 4 的零影响承诺）。
- Removes: 无。

**三条路径按信息量分层接线（修正 14 的落地）：**

- **MCMC（`mcmc.py`）信息最全，完整判据。** 非 `roughness_fraction` 参数直接对 `McmcReport.samples_physical` 的每列取中位数；分数粗糙度的先验声明在 unit 坐标，故对 retained unit samples 取中位数。不能先取 `median(unit)` 再映射：偶数样本时 NumPy 会平均两个中心值，非线性变换不保该平均。
- **单候选（`report.py`）只有点估计。** 有 `problem.variables` 能取 prior，但只有 `best.unit_vector`——共用 `analysis.mcmc.prior_conflicts` 把该点转换到先验声明坐标。文案上不区分「点估计」与「中位数」，统计强度差异记入剩余风险。
- **联合（`services/fitting_phases/joint_analysis.py`）使用局部判据的有序并集。** 在 global candidate 决定后，对每个 dataset 的 local problem 覆盖自己的 sidecar prior，计算 local conflict，再用 `JointVariable.members` 映射成 global name；shared 变量去重并按 `global_variables` 顺序输出。这样不需要把 prior 注入 joint search objective，也能正确处理 shared roughness 的局部坐标。

 - [x] **Step 1: 写失败的接线契约**

  最重要的一条**先写**——它挡的是「加了字段但忘接线」的静默失效（修正 12）：

  - `test_mcmc_report_flags_a_parameter_pulled_from_its_prior`：构造一个带强先验、且 MCMC 后验中位数明显偏离先验中心（> `k·spread`）的 `problem`，跑（或用小型定死的样本）得 `McmcReport`，断言 `prior_conflicts` 含该参数名、不含无冲突参数。
  - `test_mcmc_report_no_conflict_when_posterior_agrees_with_prior`：中位数落在先验中心 `k·spread` 内 → `prior_conflicts == ()`。
  - `test_uncertainty_report_point_estimate_conflict`：`report.py` 路径，点估计偏离先验 → 对应名字进 `prior_conflicts`。
  - `test_joint_analysis_unions_local_prior_conflicts_as_global_names`：联合路径将局部冲突映射成 global name。
  - `test_joint_analysis_deduplicates_shared_prior_conflicts`：共享变量冲突去重并保持 global order。
  - `test_reports_default_prior_conflicts_to_empty`：既有构造的两个报告 `prior_conflicts == ()`。
  - `test_prior_conflicts_do_not_enter_profile_selection`：带先验冲突的参数**不**因此被 `_reported_profile_names` 选中跑剖面（守 `profiles.py:1096` 的零影响）。
  - 在 `test_project_codec.py` 加两个报告各一条 `prior_conflicts` 往返 + 旧文件缺键落 `()` 的测例。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/analysis tests/unit/io/test_project_codec.py -q -k "prior_conflict"
  ```

 - [x] **Step 3: 加字段、三处接线、codec**

  - `model/analysis.py`：两个报告类各在字段末尾加 `prior_conflicts: tuple[str, ...] = ()`（`UncertaintyReport` 在 `bootstrap_performed` 后、`McmcReport` 在 `candidate_id` 后）。`__post_init__` 若有类型校验循环，按既有风格加一句 `tuple(...)` 化。
  - 三条路径按上文分层接线；MCMC 与 report 复用 `analysis.mcmc.prior_conflicts` 的坐标判据，joint 在 service composition 层做局部到全局的 union。
  - `codec_results.py`：`_mcmc_to_dict`/`_uncertainty_to_dict` 在 `prior_conflicts != ()` 时发 `list(...)`、为空时**不发键**（避免污染旧格式往返）；`_mcmc_from_dict`/`_uncertainty_from_dict` 把 `"prior_conflicts"` 加进第四参数的 optional 集，读侧 `tuple(_sequence(payload.get("prior_conflicts", ()), ...))`。

 - [x] **Step 4: 确认 GREEN 与全域回归**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit tests/architecture -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  全 `tests/unit` 必跑：动了 `model/analysis.py` 的报告类型，影响面覆盖 analysis 全域与 io。`test_mcmc.py` 的既有逐位重放测例必须仍绿（字段加在末尾不动既有序列化顺序）。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/model/analysis.py src/xrr_fitter/analysis/report.py src/xrr_fitter/analysis/mcmc.py src/xrr_fitter/services/fitting_phases/joint_analysis.py src/xrr_fitter/io/codec_results.py tests/unit/analysis tests/unit/services/test_fitting_prior_sidecar.py tests/unit/io/test_project_codec.py
  git commit --no-verify -m "analysis: flag posterior parameters that conflict with their prior"
  ```

### Task 6: GUI 呈现——先验列、编辑对话框、冲突并列显示

**Files:**
- Modify: `src/xrr_fitter/gui/parameters/table.py`（`HEADERS:11` 加「先验」列，渲染摘要，只读）
- Modify: `src/xrr_fitter/gui/parameters/panel.py`（右键菜单 `:221` 加「编辑先验」「清除先验」）
- Create: `src/xrr_fitter/gui/parameters/dialogs.py`（`PriorDialog`——目录现无此文件）
- Modify: `src/xrr_fitter/gui/results/uncertainty.py`（`_report_lines:114` 与 `_mcmc_lines:197` 各加一行 `prior_conflicts`）
- Modify: `docs/algorithm.md`
- Modify: `tests/gui/test_parameter_table.py`、`tests/gui/test_uncertainty_dialog.py`

**Interfaces:**
- Consumes: `api.set_parameter_priors` / `validate_parameter_priors` / `PriorSpec`（Task 4）；`report.prior_conflicts` / `mcmc.prior_conflicts`（Task 5）。
- Produces: `PriorDialog`（收集 kind + 参数，返回 `PriorSpec | None`）。
- Preserves: `HEADERS` 前六列顺序不动，新列追加在末尾（第 7 列，索引 6）；`_table_setting_changed` 的可编辑列 `(1, 2, 3, 5)`（`panel.py:236`）**不含 6**——新列天然只读，无需额外拦截。
- Removes: 无。

**新列只读，走对话框而非就地编辑：** 先验是结构化值（kind + 2~3 参数），塞不进单元格文本。第 7 列只渲染摘要（如 `normal(μ=1.0, σ=0.2)` 或空），编辑入口在右键菜单。`_table_setting_changed` 已按列号白名单 `(1,2,3,5)` 过滤，第 6 列自动落在只读侧，**这是既有机制的自然延伸，不要新增拦截逻辑**。

**冲突显示与 `boundary_hits` 并列、文案区分（修正 10）：** 两处各加一行——`_report_lines`（`:123` 的「边界命中」旁）加「先验冲突：…」；`_mcmc_lines`（`:211` 的「MCMC 边界命中」旁）加「MCMC 先验冲突：…」。文案要让用户能区分两类信号的性质：边界命中是**可疑**（贴硬边界），先验冲突是**信息**（后验与先验不一致，未必是坏事）。空时显示「无」，与既有 `_joined(...) or "无"` 一致。

 - [x] **Step 1: 写失败的 GUI 契约**

  在 `tests/gui/test_parameter_table.py`：
  - `test_prior_column_header_and_readonly`：`HEADERS` 第 7 列是「先验」；该列单元格 `flags()` 不含 `ItemIsEditable`。
  - `test_prior_column_renders_summary`：一个带 `prior=PriorSpec("normal", (1.0, 0.2))` 的定义，该列文本含 `normal`；无先验时该列为空串。
  - `test_prior_column_respects_nm_toggle`：`_uses_nm` 为真的参数（`.thickness_a`/`.roughness_a`），先验摘要里的中心值按 nm 换算显示（与既有第 1~3 列的 `scale` 一致，`table.py:93`）。**这条防止先验显示的单位与初值列打架。**

  在 `tests/gui/test_uncertainty_dialog.py`：
  - `test_report_lines_show_prior_conflicts`：`prior_conflicts=("slab1.thickness",)` 的报告，`_report_lines` 输出含「先验冲突」且含该名字。
  - `test_report_lines_show_no_conflict_when_empty`：空 `prior_conflicts` → 显示「无」。
  - `test_mcmc_lines_show_prior_conflicts`：MCMC 侧同理。
  - `test_boundary_and_prior_conflict_are_distinct_lines`：断言两行文案不同且都存在（守修正 10 的「分开呈现」）。

  `PriorDialog` 的测试（若 GUI 测试基建支持实例化对话框，照 `test_parameter_sharing.py` 的既有姿势）：
  - `test_prior_dialog_builds_spec_from_selection`：选 kind=normal、填两参 → `PriorSpec("normal", (...))`。
  - `test_prior_dialog_rejects_invalid_and_stays_open`：非法参数（如 σ≤0）→ 不返回 spec、提示错误（复用 Task 1 的构造校验，`PriorSpec` 抛 `ValueError` 时对话框捕获并提示，不崩）。

 - [x] **Step 2: 确认 RED**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/gui/test_parameter_table.py tests/gui/test_uncertainty_dialog.py -q
  ```

 - [x] **Step 3: 实现列、菜单、对话框、显示行**

  - `table.py`：`HEADERS` 追加 `"先验"`；渲染方法加第 7 列，无先验空串，有先验用一个 `_prior_summary(definition)` 生成 `kind(参数)` 文本，中心值走既有 `scale` 做 nm 换算。设该列 `flags` 去掉 `ItemIsEditable`。
  - `panel.py`：`_show_row_context_menu` 在 `reset` 之后加 `menu.addAction("编辑先验")`（objectName `editPriorAction`）与 `menu.addAction("清除先验")`（`clearPriorAction`）。触发分别开 `PriorDialog`（确定后调 `api.set_parameter_priors`）与直接调 `api.set_parameter_priors` 移除该参数先验。**「恢复默认值」不清先验**（修正 11 第 3 点的语义在此兑现——若既有 reset 会连带清先验，需在此保持先验不动）。
  - `dialogs.py`：新建 `PriorDialog(QDialog)`——kind 下拉 + 动态参数输入框，`accept` 时 `try: PriorSpec(...) except ValueError` 提示并留在对话框。照 `sharing.py` 的既有对话框风格（若 `gui/parameters/` 里 `sharing.py` 含对话框）。
  - `uncertainty.py`：`_report_lines` 加 `f"先验冲突：{_joined(report.prior_conflicts) or '无'}"`；`_mcmc_lines` 加 `f"MCMC 先验冲突：{_joined(mcmc.prior_conflicts) or '无'}"`。
  - `docs/algorithm.md`：补四 kind 的密度公式与元数语义、截断重归一化、`roughness_fraction` 先验的分数空间（修正 9）、`prior_conflict_sigmas` 默认 3.0 的理由、以及联合路径本轮不报冲突的说明。

 - [x] **Step 4: 确认 GREEN 与全域回归**

  ```bash
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/gui tests/unit tests/architecture -q
  env -u PYTHONPATH .venv/bin/python tools/check_radon.py
  ```

  `tests/architecture` 守 `gui → api` 单向依赖（新 `dialogs.py` 只能 import `api`，不能碰 `services`/`model` 内部）。全 `tests/unit` 收尾确认前五个 Task 无回归。

 - [x] **Step 5: 提交**

  ```bash
  git add src/xrr_fitter/gui/parameters/table.py src/xrr_fitter/gui/parameters/panel.py src/xrr_fitter/gui/parameters/dialogs.py src/xrr_fitter/gui/results/uncertainty.py docs/algorithm.md tests/gui/test_parameter_table.py tests/gui/test_uncertainty_dialog.py
  git commit --no-verify -m "gui: edit parameter priors and surface prior conflicts"
  ```

## 最终验收记录

六个 Task 全绿后，按下列命令跑一次全域收尾，把输出摘要贴回本节（谁执行谁填）：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit tests/gui tests/architecture -q
env -u PYTHONPATH .venv/bin/python tools/check_radon.py
```

逐项确认（缺一不可）：

- [x] `test_problem_log_probability_is_bitwise_unchanged_without_priors` 用 `==` 通过——无先验时对数概率**逐位不变**，这是"零影响"承诺的地基。
- [x] `test_prior_conflicts_do_not_enter_profile_selection` 通过——`profiles.py:1096` `_reported_profile_names` 只消费 `boundary_hits`，`prior_conflicts` 没渗进 profile 选择，因而没改 `ConfidenceClass`。
- [x] `tests/architecture` 全绿——`MODEL_ALLOWED["parameters"] == set()` 仍成立（先验数学没漏进 `model/parameters.py`）；`gui/parameters/dialogs.py` 只 import `api`。
- [x] 项目文件往返：带先验的工程存盘再读回，`parameter_priors` 逐位一致；**旧工程文件（无 `parameter_priors` 键）仍能读入**（`_dataset_from_dict` 的 optional-key 路径，修正 8）。
- [x] `tools/check_radon.py` 通过——`evaluation.py` 的 16 分支靠 dispatch 表拆开，单块 CC ≤ 10。

### 本轮执行摘要（2026-08-12；历史快照）

- 提交：`a954a9e` `gui: edit parameter priors and surface prior conflicts`（分支 `feat/interface-transitions`，7 files changed, 447 insertions(+), 61 deletions(-)）。走完整 pre-commit（未用 `--no-verify`），ruff/ruff-format 首轮改动后重新 stage 同一批显式路径二次提交，第二轮全 Passed。
- 全域收尾（命令同上，GUI 加 `QT_QPA_PLATFORM=offscreen`）：`2061 passed, 48 warnings in 135.39s`。48 warnings 全为 `export_plots.py` 既有 CJK 缺字，与本次改动无关。
- `tools/check_radon.py`：通过。`_report_lines` 因新增"先验冲突"行一度到 CC 11，提取 `_joined_or` helper 后回落 ≤ 10，输出字符串逐字节不变。
- Task 6 期间修复两处门禁红灯：(1) `dialogs.py` 模块常量 `_MAX_FIELDS`→`MAX_PRIOR_FIELDS` 并改 `max(map(len, ...))` 消除推导式泄漏的 `labels`，满足 R23 命名；(2) 上述 radon 提取。
- 5 项逐项确认对应的具名测试均以 `--collect-only` 确认真实存在（非 skip/deselect）并在 2061 passed 中通过：`test_problem_log_probability_is_bitwise_unchanged_without_priors`、`test_prior_conflicts_do_not_enter_profile_selection`、`test_project_roundtrips_parameter_priors`、`test_old_project_without_priors_decodes_to_empty`、`test_project_roundtrip_preserves_prior_conflicts`；`test_dependency_rules.py:119` 经 Read 核实为 `"parameters": set(),`。

### 当前工作区复核（2026-08-13）

- 当前 `HEAD = df7c826` 加未提交审计修复的 unit、GUI、architecture、integration 联跑为 `2179 passed in 194.34s (0:03:14)`，无 warning；实时预览重建中文图例后会回退到 DejaVu Sans 的遗漏已由回归测试覆盖并修复。
- 后续生命周期修复保证 source/structure/settings 更新均按 effective bounds 协调 prior；聚焦 services/GUI 套件曾为 `95 passed`。
- 新增 RED→GREEN 回归 `test_mcmc_prior_conflicts_use_the_physical_sample_median_for_log_parameters`：修复偶数 retained sample 在 log transform 下由 `median(unit)` 产生的误报；MCMC 现在用物理样本中位数，`roughness_fraction` 仍用 unit 分数中位数。
- 联合拟合不再是下述“显式留空”的历史范围：`services/fitting_phases/joint_analysis.py` 已实现 local-to-global conflict union，并有 shared-variable 去重测试。

## 剩余风险

1. **联合冲突是 winning-candidate 注释，不是 joint posterior 统计。** 当前 union 对每个 dataset 的 winning local `unit_vector` 计算点估计冲突后映射到 global variable；它不会把 sidecar prior 注入 joint search objective，也不是 joint ensemble 的后验中位数判据。这样保持拟合排名零影响，但统计强度弱于 MCMC 报告。

2. **`report.py` 点估计判据在统计上弱于 MCMC 中位数。** `report.py:269` 手上只有 `best.unit_vector`（单点），`prior_conflicts` 用点估计对比先验中心；`mcmc.py:410` 有完整后验样本，用中位数。同一参数在两条路径下的冲突判定可能不一致——点估计恰好落在先验中心附近、但后验整体偏移时，report 路径会漏报。这是信息量差异的必然结果，非 bug；用户看 MCMC 结果时得到的是更强的判据。

3. **修正 4 的保守方向失真：强先验且贴边界的参数仍被 `boundary_hits` 标记。** 因为 `boundary_hits` 与 `prior_conflicts` 是两套独立判据、从不合并，一个"先验强烈拉向边界、后验也就该贴边界"的参数会**同时**触发「边界命中（可疑）」和不触发先验冲突。用户可能把这种合理的贴边界误读成拟合触底。缓解靠修正 10 的文案区分（边界命中=可疑、先验冲突=信息）让用户自行判断，但两信号不会自动互相解释。

4. **截断重归一化的缓存键假设 `PriorSpec` 可哈希且 `(lower, upper)` 稳定。** `functools.lru_cache` 键为 `(spec, lower, upper)`（修正 7）。`PriorSpec` 是 frozen dataclass、参数为 `tuple[float, ...]`，可哈希成立；但若日后某参数的 bounds 在一次拟合内动态变化（当前不会），缓存会返回过期归一化常数。本轮 bounds 在编译期固定，风险为零，仅作为未来改动的警示留档。

<!-- PLAN-COMPLETE -->
