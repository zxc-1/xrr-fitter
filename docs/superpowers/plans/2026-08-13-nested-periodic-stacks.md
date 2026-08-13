# 嵌套周期堆栈（nested periodic stacks）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `PeriodicBlock.layers` 能容纳子 `PeriodicBlock`，从而声明 `[A/B]×N` 外面再包一层 `[…]×M` 的超晶格与非对称多层镜，且展开出的 `SlabStack` 与手工摊平的等价声明**逐位相等**。

**Architecture:** 声明层放宽类型并新增两条自包含守卫（深度、展开规模）；展开层把 `_append_periodic` 的内循环从"只认 `LayerSpec`"改为递归调用 `_append_component`，只为**最外层**注册 `PeriodicSpan`；`physics/geometry.py` 的三处平铺遍历与 `fit/`、`analysis/`、`services/` 的九处消费方同步递归；序列化与 GUI 树同批交付。`physics/derivatives.py` 零改动。

**Tech Stack:** Python 3.12、NumPy 2.x、SciPy 1.14+、PySide6、pytest 8.3+（`--import-mode=importlib`）。

**Design source:** `docs/superpowers/specs/2026-08-09-nested-periodic-stacks-design.md`
**Fact-check baseline:** HEAD `df7c826`（分支 `feat/interface-transitions`，工作区含 interface-transitions / parameter-priors / sld-uncertainty-bands 的落地改动）。

---

## 对 spec 的修正

spec 写于 interface-transitions 落地之前，以下偏差按既定方法逐条对着 HEAD 核对，**执行时以本节为准，不以 spec 为准**。

### 1. spec 的全部行号锚点已失效（轻微，但会误导定位）

实测位置：`StructureComponent` 联合 `model/structure.py:290`（spec 记 `:207`）、`PeriodicBlock` `:273`（spec 记 `:194`）、`_periodic_layers` `:202`（spec 记 `:123`）、`_append_periodic` `physics/stack.py:228`（spec 记 `:198`）、`_periodic_definitions` `fit/parameters.py:154`（spec 记 `:152`）、`_component_item` `gui/structure/editor.py:244`（spec 记 `:241`）、`PeriodicDialog` `gui/structure/dialogs.py:233`（spec 记 `:190`）。联合类型的**成员没变**，只是位置移动。

### 2. 周期块内部现在**禁止** transition，嵌套不得静默解锁（致命）

`model/structure.py:208-209`：

```python
if any(layer.transition is not None for layer in layers):
    raise ValueError(f"{name}.layers must not declare a transition inside a periodic block")
```

这是 interface-transitions 计划的修正 11 主动加的门禁，其"剩余风险"明确写道：解锁需要同时改 `PeriodicSpan.layer_count` 的语义、`_validate_roughness_repetition` 与 `top_roughness_a` 的覆盖条件，**"与 nested-periodic-stacks 计划有写集重叠——两个计划不要并行推进这一块"**。

本计划的政策：**保持禁止，并让它对嵌套递归生效**——子块内的 `LayerSpec` 同样不得带 transition。放宽 `layers` 类型时若只放宽 `isinstance` 检查而漏掉这条，子块里的 transition 会绕过校验直达 `_append_transition_layer`，而 `PeriodicSpan` 的等距假设不成立，`_validate_roughness_repetition` 会给出误导性的 `"periodic span roughness does not repeat"`。解锁内层 transition 属于**本计划范围之外**，留给后续计划。

### 3. `_append_component` 存在且语义未变，spec 的递归目标有效（确认项）

`physics/stack.py:249` 的 `_append_component(state, component, override)` 已按 `LayerSpec` / `PeriodicBlock` / `GradientLayerSpec` 分派。spec 要求把 `_append_periodic` 的内层 `_append_layer` 调用换成 `_append_component`，该目标可用。

### 4. `_validate_roughness` 已改为针对独立的 `limit_thickness` 数组校验（确认项）

`physics/stack.py:267` 现在读 `state.limit_thickness`（`_Expansion:175` 的独立字段，transition 微切片不进入动态粗糙度上限），抛 `PhysicalValueError`。嵌套原样复用，不改。

### 5. 结构序列化在 `io/codec_declarations.py`，不在 `io/codec_common.py`（spec 误标）

`_periodic_to_dict:267` / `_periodic_from_dict:277`。后者的必需键集是 `{"kind","name","layers","repeats","top_roughness_a"}` 且**没有 optional 集合**。递归解码要照抄同文件 `_layer_to_dict:231` / `_layer_from_dict:247` 的可选键模式（注释原文：*"Emitting the key only when present keeps files written before transitions existed byte-identical"*）。嵌套子项的判别子只接受 `layer` 与 `periodic_block`，**不接受 `gradient_layer`**（spec 已声明周期块内不放梯度层）。

### 6. `PeriodicSpan.layer_count` 的语义变成承重的（致命，主要风险点）

`physics/stack.py:235` 现在传 `len(block.layers)`：

```python
if block.repeats > 1:
    state.spans.append(PeriodicSpan(start, len(block.layers), block.repeats))
```

只有在无嵌套时 `len(block.layers)` 才等于"一个周期展开后的 slab 数"。而 `model/structure.py:380-453` 的 `_validate_media_repetition` / `_validate_roughness_repetition` 是按 `layer_count` 切**展开后**数组的：

```python
stop = start + span.layer_count
first = roughness[start - 1 : stop - 1]
normal_start = stop - 1
normal = roughness[normal_start : normal_start + span.layer_count]
```

因此嵌套时最外层 span 必须注册**展开后的每周期 slab 数**，即 `(len(state.thickness) - start) // block.repeats`。同一个数还必须与 `physics/geometry.py:34` `_periodic_interface_names` 每周期发出的粗糙度名个数完全一致，否则 `expand_geometry` 在 `geometry.py:239` 抛 `RuntimeError("expanded geometry interface mapping mismatch")`。

为让声明层与展开层用**同一个**数，在 `model/structure.py` 增一个模块级纯函数 `periodic_cell_slabs(block) -> int`（递归求和：`LayerSpec` 记 1，子 `PeriodicBlock` 记 `repeats * periodic_cell_slabs(child)`）。`physics/stack.py` 与 `physics/geometry.py` import 它——`ALLOWED` 允许 `physics → model`，而 `MODEL_ALLOWED["structure"] = set()` 要求这个函数不得 import 任何包内模块，纯 `int` 运算满足。

### 7. 展开规模没有任何上限，深度限制不足以约束它（致命）

全仓 `MAX_` 常量只有 `MAX_QUERY_VALUES = 4096`（`physics/resolution.py:19`）、`MAX_TRANSITION_SLABS = 512`（`model/structure.py:137`，per-transition 微切片）、`MAX_REPLAY_SAMPLES` / `MAX_FAILURE_RATE`（`analysis/sld_bands.py`）、`MAX_PRIOR_FIELDS`。**没有**"展开后总 slab 数"的守卫。

spec 只限了深度 ≤ 3。三层嵌套每层 `repeats=20`、叶子 2 层，展开就是 `20×20×20×2 = 16000` 个 slab；`repeats` 是可优化坐标（`fit/parameters.py` 里 locked，但项目文件可任意写），一个手写 JSON 就能让 `_Expansion` 的 Python list 循环与后续 `n×n` 矩阵传播卡死界面。本计划新增 `MAX_PERIODIC_SLABS = 2048` 在 `PeriodicBlock.__post_init__` 校验 `repeats * periodic_cell_slabs(self) `——自包含、不需要全局上下文。2048 的取值是我的判断（Mo/Si 20 周期 = 40 slab，留两个数量级余量），如需别的阈值在执行时改这一个常量即可。

### 8. 写集是 **17 个生产文件**，不是 spec 列的 6 个（致命）

spec 声称"`derivatives.py` 零改动"是对的，但它漏掉了九处平铺遍历 `block.layers` 的消费方。逐个核实过：

| 文件:行 | 现状 | 嵌套下的失效方式 |
| --- | --- | --- |
| `physics/geometry.py:34` `_periodic_interface_names` | `range(len(block.layers))` 生成粗糙度名 | 名数与展开 slab 数不符 → `RuntimeError` |
| `physics/geometry.py:146` `_append_periodic_geometry` | `for layer_index, layer in enumerate(...)` 取 `layer.thickness_a` | 子块无 `.thickness_a` → AttributeError |
| `physics/geometry.py:373` `_StackJacobianBuilder.append_periodic` | `divmod(flat_index, layer_count)` 平铺遍历 | Jacobian 行错位 → `RuntimeError("expanded structure Jacobian mapping mismatch")` |
| `physics/stack.py:78` `_replace_periodic` | `map(_replace_indexed_layer, enumerate(block.layers))` | 子块被当 `LayerSpec` 重建 |
| `fit/parameters.py:154` `_periodic_definitions` | `_layer_definitions(f"{prefix}.layer.{index}", layer, …)` | 子块无参数定义 |
| `fit/candidates.py:156/290` | `_periodic_interface_values` / `_baseline_component_values` 平铺 | 基线值缺键 → KeyError |
| `fit/global_search.py:236` `_periodic_geometry_group` | `[layer.thickness_a for layer in block.layers]` | AttributeError |
| `fit/problem.py:83` | 构 `(prefix, layer)` 对后读 `layer.density_scale` | AttributeError |
| `fit/initialization.py:260/409/594` | `layer.material.sld_override_a2`、`first.layers[:1]`、`_independent_thickness_dof` | AttributeError / 自由度算错 |
| `analysis/sld_bands.py:73` `_periodic_values` | 平铺 `{prefix}.layer.{index}` | 采样值缺键 |
| `analysis/binary_profiles.py:85` | `len(component.layers) == 2` 判定双层 | 见修正 9 |
| `services/structures.py:78/89` | `component.layers[0].material` / `layers[-1].material` | AttributeError（末尾已有 `raise TypeError`） |
| `services/batch.py:92` `_component_signature` | `(layer.name, *_material_signature(layer.material))` | AttributeError → 批任务签名崩 |

加上 `model/structure.py`、`io/codec_declarations.py`、`io/examples.py:67`、`api.py`、`gui/structure/editor.py`、`gui/structure/dialogs.py`。**`physics/geometry.py` 的三处必须与 `physics/stack.py` 的展开顺序锁步**——两个硬对齐 `RuntimeError` 是唯一的安全网，任何一侧单独改都会红。

### 9. `analysis/binary_profiles.py:85` 的双层判据在嵌套下语义错误（推翻 spec 的"零改动"清单）

```python
if isinstance(component, PeriodicBlock) and len(component.layers) == 2:
```

一个"一个 `LayerSpec` + 一个子块"的周期块 `len(layers) == 2` 成立，却不是双层膜，会被当二元剖面处理。判据要改成"两个成员都是 `LayerSpec`"。

### 10. `PeriodicDialog` 根本没有 `top_roughness_a` 字段（推翻 spec 的一整节）

`gui/structure/dialogs.py:233` 是 `QTableWidget(2, 5)`（`periodicLayerTable`），字段只有名称/重复数/表格，`block` property `:311` 构造 `api.PeriodicBlock(...)` 时不传 `top_roughness_a`。spec 要求"`top_roughness_a` 只在最外层对话框显示"——前提是先把这个字段**加上**。所以 GUI 任务是"新增字段 + 按层级条件显示"，不是"隐藏既有字段"。

### 11. `editor.py:_selected_index` 对子项返回 `None`，与"选中子块加子周期"冲突（致命）

`gui/structure/editor.py:283`：

```python
item = self.tree.currentItem()
if item is None or item.parent() is not None:
    return None
```

顶层索引是 `int`，子项一律不可选。spec 要求"添加子周期块"按钮在达深度上限时 disabled，隐含子块必须可选中并能定位到嵌套路径。本计划把选中模型从"顶层 `int`"扩为"路径 `tuple[int, ...]`"：`_selected_index()` 保持只认顶层（`remove`/`up`/`down` 依赖它，语义不变），新增 `_selected_path()` 供嵌套操作使用。这样既不动既有三个按钮的行为，也让新按钮有正确的定位依据。

### 12. 嵌套对 checkpoint 指纹安全，但**不得**为展开量加 dataclass 字段（确认项 + 陷阱）

`fit/checkpoint.py` 的 `_canonical_dataclass` 遍历 `fields(value)`。放宽 `layers` 的**类型注解**不新增字段，非嵌套块的规范化输出逐字节不变，因此 `POST_FREEZE_OMITTED_DEFAULTS` 不需要新条目，历史 checkpoint 照旧 resume。

反过来说：修正 6/7 需要的"每周期展开 slab 数"**必须**是模块级函数或 `property`，一旦写成 dataclass field（即使带默认值）指纹立刻漂移，旧 checkpoint 全部失效，而只有 `test_frozen_stage_search` 抓得到。

---

## Global Constraints

- 遵守 `tests/architecture/test_dependency_rules.py` 的 `ALLOWED`；`MODEL_ALLOWED["structure"] = set()`（`model/structure.py` 不得 import 包内任何模块）；不新增 `PACKAGE_EDGE_EXCEPTIONS`。
- 禁止 `pytest.skip` / `xfail` / 条件收集：`tests/outcome_gate.py` 会让任何 skip/xfail/xpass/deselect 直接失败。
- 新增测试落在既有整目录注册下，**不需要改 `tools/verify_registry.py`**；测试模块名不得以父目录名开头；`tests/conftest.py` 必须保持为仓库唯一 `conftest.py`。
- 过 `tools/check_radon.py`（单块 CC ≤ 10、文件平均 CC ≤ 5.0、MI 等级 A、仓库平均 ≤ 5.0）。
- 跑测试必须带 `--import-mode=importlib`，解释器用 `.venv/bin/python`（只有它能 `import xrr_fitter`）；`tools/verify.py` 在本地会因 repo 内 `.venv` 被判 "generated directory inside repository" 而失败，直接调 pytest。
- 界面 nm、数据层 Å，换算 `* 10.0`；不 stage `.claude/`。
- **无嵌套时全套测试逐位不变**：所有等价性断言用 `==` / `np.array_equal`，不用 `approx`。
- 周期块内禁止 transition 的门禁递归生效（修正 2）；周期块内不放梯度层。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `src/xrr_fitter/model/structure.py` | 放宽 `PeriodicBlock.layers` 类型；新增 `periodic_cell_slabs`、`MAX_PERIODIC_DEPTH`、`MAX_PERIODIC_SLABS`；递归 transition 门禁 |
| `src/xrr_fitter/physics/stack.py` | `_append_periodic` 递归到 `_append_component`；按展开每周期 slab 数注册最外层 `PeriodicSpan`；`_replace_periodic` 递归 |
| `src/xrr_fitter/physics/geometry.py` | 三处平铺遍历递归，与 stack 展开顺序锁步 |
| `src/xrr_fitter/fit/parameters.py` | `_periodic_definitions` 递归生成嵌套前缀参数 |
| `src/xrr_fitter/fit/candidates.py` | 基线值与界面值递归 |
| `src/xrr_fitter/fit/global_search.py` | 周期几何分组用展开厚度而非 `layer.thickness_a` |
| `src/xrr_fitter/fit/problem.py` | 层前缀收集递归 |
| `src/xrr_fitter/fit/initialization.py` | 直接 SLD 路径、首层探测、独立厚度自由度递归 |
| `src/xrr_fitter/analysis/sld_bands.py` | `_periodic_values` 递归 |
| `src/xrr_fitter/analysis/binary_profiles.py` | 双层判据改为"两成员均为 `LayerSpec`" |
| `src/xrr_fitter/services/structures.py` | 首/末材料取值下钻到叶子 `LayerSpec` |
| `src/xrr_fitter/services/batch.py` | `_component_signature` 递归 |
| `src/xrr_fitter/io/codec_declarations.py` | `_periodic_to_dict` / `_from_dict` 递归 + 可选键保旧文件字节等价 |
| `src/xrr_fitter/io/examples.py` | 新增一个嵌套超晶格示例 |
| `src/xrr_fitter/api.py` | 导出新增常量/函数（若需） |
| `src/xrr_fitter/gui/structure/editor.py` | `_component_item` 递归渲染；新增 `_selected_path`、添加子周期块按钮 |
| `src/xrr_fitter/gui/structure/dialogs.py` | `QTableWidget` → `QTreeWidget`；新增 `top_roughness_a` 字段，仅最外层显示 |
| `tests/unit/model/test_structures.py` | 类型放宽、深度上限、展开规模上限、递归 transition 门禁 |
| `tests/unit/physics/test_stack_expansion.py` | **主验收门**：嵌套 vs 手工摊平逐位相等；span 注册的每周期 slab 数 |
| `tests/unit/physics/test_derivatives.py` | 嵌套 Jacobian 与摊平逐位相等 |
| `tests/unit/io/test_project_codec.py` | 嵌套往返；旧文件缺新键仍可读 |
| `tests/gui/…` | 嵌套树层级渲染、深度上限按钮禁用、子对话框无 `top_roughness_a` |

---

## Tasks

### Task 1: 声明层放宽类型并加两条自包含守卫

**Files:**
- Modify: `src/xrr_fitter/model/structure.py`
- Modify: `tests/unit/model/test_structures.py`

**Interfaces:**
- Produces: `periodic_cell_slabs(block) -> int`、`MAX_PERIODIC_DEPTH = 3`、`MAX_PERIODIC_SLABS = 2048`
- Preserves: `PeriodicBlock` 的四个字段与顺序（`name`、`layers`、`repeats`、`top_roughness_a`）、`StructureComponent` 联合成员、周期块内禁止 transition
- Removes: 无

**为什么先做这一层：** `MODEL_ALLOWED["structure"] = set()` 让这个模块零依赖，可以独立测；后面每一层都要 import `periodic_cell_slabs`。

**复杂度约束：** `periodic_cell_slabs` 递归求和 CC ≤ 3；守卫拆成独立私有函数，别堆进 `__post_init__`。

- [ ] **Step 1**：写 `test_periodic_block_accepts_nested_block`、`test_nested_block_rejects_transition`、`test_periodic_depth_limit`、`test_periodic_expanded_slab_limit`、`test_periodic_cell_slabs_counts_expanded_members`。RED：`.venv/bin/python -m pytest tests/unit/model/test_structures.py --import-mode=importlib -q`
- [ ] **Step 2**：`_periodic_layers` 的 `isinstance` 放宽为 `(LayerSpec, PeriodicBlock)`；transition 检查只对 `LayerSpec` 成员生效，子块靠自身 `__post_init__` 递归保证（子块构造时已校验过，不重复走）。新增 `periodic_cell_slabs`、深度与展开量守卫。GREEN：同上 + `.venv/bin/python tools/check_radon.py`
- [ ] **Step 3**：`git add src/xrr_fitter/model/structure.py tests/unit/model/test_structures.py && git commit -m "model: allow nested periodic blocks with depth and size guards"`

### Task 2: 展开层递归 + 主验收门

**Files:**
- Modify: `src/xrr_fitter/physics/stack.py`
- Modify: `tests/unit/physics/test_stack_expansion.py`

**Interfaces:**
- Produces: 嵌套声明的 `SlabStack` 展开
- Preserves: `_append_layer` / `_append_transition_layer` / `_append_gradient` / `_validate_roughness` 全部不变；无嵌套时展开结果逐位不变；只为最外层注册 `PeriodicSpan`
- Removes: 无

**为什么在这一层放主验收门：** 等价性是 spec 的主验收门，且 `PeriodicSpan` 的 `layer_count` 语义（修正 6）在这里定型，`_validate_*_repetition` 会立刻验证它对不对。

**复杂度约束：** `_append_periodic` 改后 CC ≤ 6；`top_roughness_a` 的覆盖判定（只在最外层第一个 repeat 的第一个 slab）抽成私有谓词。

- [ ] **Step 1**：写 `test_nested_periodic_expansion_matches_flattened`（thickness、sld、roughness 三数组 `np.array_equal` 全等，**主验收门**）、`test_nested_span_records_expanded_cell_slabs`、`test_nested_top_roughness_overrides_outermost_first_interface`、`test_non_nested_expansion_unchanged`。RED：`.venv/bin/python -m pytest tests/unit/physics/test_stack_expansion.py --import-mode=importlib -q`
- [ ] **Step 2**：`_append_periodic` 内循环改调 `_append_component`；span 用 `PeriodicSpan(start, periodic_cell_slabs(block), block.repeats)`（等价于 `(len(state.thickness) - start) // block.repeats`，用前者保证与 geometry 同源）；子块递归时不传 `top_roughness_a` 覆盖。`_replace_periodic` 的成员重建改走 `_replace_component`，并保留"缺失 top 覆盖是语义继承而非零值"的判定。GREEN：`.venv/bin/python -m pytest tests/unit/physics --import-mode=importlib -q` + `check_radon.py`
- [ ] **Step 3**：`git add src/xrr_fitter/physics/stack.py tests/unit/physics/test_stack_expansion.py && git commit -m "physics: expand nested periodic blocks into equivalent slabs"`

### Task 3: 几何与 Jacobian 三处遍历锁步

**Files:**
- Modify: `src/xrr_fitter/physics/geometry.py`
- Modify: `tests/unit/physics/test_derivatives.py`

**Interfaces:**
- Produces: 嵌套结构的界面名序列、几何厚度展开、`_StackJacobianBuilder` 映射
- Preserves: `expand_geometry:239` 与 `finish()` 两个对齐 `RuntimeError` 保持为硬失败；`GRADIENT_INTERNAL_INTERFACE` 语义不变
- Removes: 无

**为什么紧跟 Task 2：** 两个硬对齐检查是这次改动唯一的安全网，Task 2 落地后它们必然红，必须同批修好。

**复杂度约束：** 三个函数各自递归，禁止把它们合并成一个通用遍历器（会推高 CC 并模糊 Jacobian 的行序）；`append_periodic` 的 `divmod` 平铺遍历改为按成员递归，CC ≤ 8。

- [ ] **Step 1**：写 `test_nested_jacobian_matches_flattened`（Jacobian 矩阵逐位相等）、`test_nested_interface_names_align_with_expansion`。RED：`.venv/bin/python -m pytest tests/unit/physics/test_derivatives.py --import-mode=importlib -q`
- [ ] **Step 2**：`_periodic_interface_names` 递归发名（子块整体展开后的名序列重复 `repeats` 次，最外层第一个位置按 `top_roughness_a` 决定用覆盖名）；`_append_periodic_geometry` 与 `_StackJacobianBuilder.append_periodic` 改按成员递归，嵌套前缀与 Task 4 的参数名保持一致。GREEN：`.venv/bin/python -m pytest tests/unit/physics --import-mode=importlib -q` + `check_radon.py`
- [ ] **Step 3**：`git add src/xrr_fitter/physics/geometry.py tests/unit/physics/test_derivatives.py && git commit -m "physics: keep nested geometry and Jacobian mappings aligned"`

### Task 4: 参数定义与候选基线值递归

**Files:**
- Modify: `src/xrr_fitter/fit/parameters.py`
- Modify: `src/xrr_fitter/fit/candidates.py`
- Modify: `tests/unit/model/test_parameters.py`

**Interfaces:**
- Produces: 嵌套前缀参数（`component.{i}.block.{j}.layer.{k}.…` 形式，命名在 Step 2 定死后三处必须一致）
- Preserves: 非嵌套参数名与顺序逐位不变；`repeats` 仍 locked + `integer=True`；`top_roughness_a` 在无显式覆盖时 locked
- Removes: 无

**复杂度约束：** `_periodic_definitions` 递归后 CC ≤ 8。

- [ ] **Step 1**：写 `test_nested_definitions_cover_every_expanded_interface`、`test_non_nested_definitions_unchanged`（与摊平声明的名序列 `==`）。RED：`.venv/bin/python -m pytest tests/unit/model/test_parameters.py --import-mode=importlib -q`
- [ ] **Step 2**：`_periodic_definitions` 对成员分派递归；`fit/candidates.py:156/290` 同步递归，`top_roughness_a` 仍只在非 `None` 时发。GREEN：`.venv/bin/python -m pytest tests/unit --import-mode=importlib -q` + `check_radon.py`
- [ ] **Step 3**：`git add src/xrr_fitter/fit/parameters.py src/xrr_fitter/fit/candidates.py tests/unit/model/test_parameters.py && git commit -m "fit: define parameters for nested periodic members"`

### Task 5: 其余 fit 消费方

**Files:**
- Modify: `src/xrr_fitter/fit/global_search.py`、`src/xrr_fitter/fit/problem.py`、`src/xrr_fitter/fit/initialization.py`
- Modify: 对应既有测试模块

**Interfaces:**
- Preserves: 非嵌套时全局搜索的周期分组、`_independent_thickness_dof`、直接 SLD 路径逐位不变
- Removes: 无

- [ ] **Step 1**：写 `test_nested_period_group_uses_expanded_thickness`、`test_nested_independent_thickness_dof`。RED：`.venv/bin/python -m pytest tests/unit/fit --import-mode=importlib -q`
- [ ] **Step 2**：`_periodic_geometry_group` 的 `declared` 改用递归收集的叶子厚度；`problem.py:83` 与 `initialization.py:260/409/594` 递归下钻到叶子 `LayerSpec`。GREEN：同上 + `check_radon.py`
- [ ] **Step 3**：`git commit -m "fit: walk nested periodic members in search and initialization"`

### Task 6: analysis 与 services 消费方

**Files:**
- Modify: `src/xrr_fitter/analysis/sld_bands.py`、`src/xrr_fitter/analysis/binary_profiles.py`、`src/xrr_fitter/services/structures.py`、`src/xrr_fitter/services/batch.py`
- Modify: `tests/unit/model/test_sld_bands.py` 及对应 services 测试

**Interfaces:**
- Produces: 嵌套结构的 SLD 采样值、结构摘要、批任务签名
- Preserves: 二元剖面只认"两成员均为 `LayerSpec`"（修正 9）；`services/structures.py` 末尾的 `raise TypeError` 保留
- Removes: 无

- [ ] **Step 1**：写 `test_nested_block_is_not_binary_profile`、`test_nested_sld_values_cover_members`、`test_nested_batch_signature_distinguishes_nesting`。RED：`.venv/bin/python -m pytest tests/unit --import-mode=importlib -q`
- [ ] **Step 2**：四处递归改造。GREEN：同上 + `check_radon.py`
- [ ] **Step 3**：`git commit -m "analysis: handle nested periodic blocks in profiles and summaries"`

### Task 7: 序列化递归与旧文件兼容

**Files:**
- Modify: `src/xrr_fitter/io/codec_declarations.py`、`src/xrr_fitter/io/examples.py`
- Modify: `tests/unit/io/test_project_codec.py`、`tests/unit/io/test_examples.py`

**Interfaces:**
- Produces: 嵌套结构往返
- Preserves: 无嵌套项目文件**字节等价**；`_periodic_from_dict` 的必需键集不变，嵌套成员靠判别子区分
- Removes: 无

**复杂度约束：** `_component_from_dict` 的 `decoders` 表复用，嵌套成员用一个只含 `layer` / `periodic_block` 的受限表。

- [ ] **Step 1**：写 `test_nested_structure_round_trip`、`test_legacy_periodic_file_still_loads`、`test_non_nested_serialization_byte_identical`。RED：`.venv/bin/python -m pytest tests/unit/io --import-mode=importlib -q`
- [ ] **Step 2**：`_periodic_to_dict` / `_periodic_from_dict` 递归；成员写入沿用可选键模式；`io/examples.py` 加一个嵌套超晶格示例。GREEN：同上 + `check_radon.py`
- [ ] **Step 3**：`git commit -m "io: serialize nested periodic blocks without touching legacy files"`

### Task 8: GUI 同批交付

**Files:**
- Modify: `src/xrr_fitter/gui/structure/editor.py`、`src/xrr_fitter/gui/structure/dialogs.py`、`src/xrr_fitter/gui/accessibility.py`
- Modify: `tests/gui/` 下对应模块

**Interfaces:**
- Produces: 嵌套树渲染、"添加子周期块"按钮、`PeriodicDialog` 的 `top_roughness_a` 字段（修正 10）、`_selected_path`（修正 11）
- Preserves: `_selected_index` 只认顶层，`remove`/`up`/`down` 行为不变；`TREE_HEADERS` 六列不变；错误标签 `periodicDialogError` 不变
- Removes: `periodicLayerTable`（`QTableWidget` → `QTreeWidget`）

**为什么最后做：** borrowing-roadmap 的"界面同批交付"要求 UI 与后端同一批落地；后端命名（嵌套前缀、深度上限）必须先定死。

**复杂度约束：** `_component_item` 递归后 CC ≤ 6；对话框构建拆函数，别让 `block` property 变成大块。

- [ ] **Step 1**：写三个 spec 指定的 GUI 测例——嵌套树按层级渲染且子块行显示"重复"值、达深度上限时"添加子周期块" disabled、子块对话框**没有** `top_roughness_a` 字段。RED：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
- [ ] **Step 2**：`_component_item` 递归；新增 `_selected_path`；`PeriodicDialog` 换 `QTreeWidget` 并加 `top_roughness_a`（仅最外层）；`accessibility.py:68` 注册新按钮的可访问名。GREEN：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q` + `check_radon.py`
- [ ] **Step 3**：`git commit -m "gui: edit nested periodic blocks in the structure tree"`

### Task 9: 全量验收

- [ ] **Step 1**：`.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`
- [ ] **Step 2**：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
- [ ] **Step 3**：`.venv/bin/python tools/verify_registry.py quality`
- [ ] **Step 4**：确认 `test_frozen_stage_search` 通过（checkpoint 指纹未漂移，修正 12）
- [ ] **Step 5**：填写下方"实际输出"表

---

## 验收清单

- [ ] **主验收门**：嵌套声明展开出的 `SlabStack` 与手工摊平的等价声明逐位相等（thickness、sld、roughness 三数组全部，用 `np.array_equal`）
- [ ] 嵌套结构的 Jacobian 与摊平声明逐位相等
- [ ] `top_roughness_a` 只覆盖**最外层**第一个 repeat 的第一个界面
- [ ] 深度 > 3 被拒；展开 slab 数 > `MAX_PERIODIC_SLABS` 被拒
- [ ] 子块内声明 transition 被拒（修正 2）
- [ ] 无嵌套时：展开结果、参数名序列、项目文件字节全部不变
- [ ] 旧项目文件（无嵌套键）仍可读
- [ ] 三个 GUI 测例通过
- [ ] `physics/derivatives.py` 零改动（`git diff --stat` 确认）
- [ ] checkpoint 指纹未漂移

### 实际输出（HEAD = ）

| 批次 | commit | 测试结果 |
| --- | --- | --- |
| Task 1 | | |
| Task 2 | | |
| Task 3 | | |
| Task 4 | | |
| Task 5 | | |
| Task 6 | | |
| Task 7 | | |
| Task 8 | | |
| Task 9 | | |

---

## 剩余风险

- **`PeriodicSpan` 的快速路径收益随嵌套下降。** 只注册最外层 span，内层块按普通层展开，所以 `[[A/B]×10]×10` 的矩阵幂加速只作用在最外层的 10 次重复上，内层 10 次是实打实的 slab。这是 spec 的选择（换来 `derivatives.py` 零改动），但用户若期望"嵌套=更快"会失望；`_apply_span_tangent` 的游标模型假设 span 不重叠，注册内层 span 会破坏它。
- **周期块内仍不支持 transition。** 修正 2 保持了 interface-transitions 立的门禁。解锁需要同时改 `PeriodicSpan.layer_count` 语义、`_validate_roughness_repetition` 与 `top_roughness_a` 覆盖条件，与本计划写集重叠——**不要与本计划并行推进**。
- **`MAX_PERIODIC_SLABS = 2048` 是我定的阈值。** 没有物理依据支撑这个具体数，只保证挡住手写 JSON 造出的病态展开。若真实需求（如深紫外多层镜 500+ 周期）撞上它，改常量而非删守卫。
- **17 个文件的写集里有 9 处是"隐式契约"消费方。** 它们没有类型约束保护，只靠两个对齐 `RuntimeError` 和测试覆盖。若后续有人新增一个平铺遍历 `block.layers` 的消费方，不会有任何门禁提醒。考虑在 Task 6 后补一条架构测试：遍历所有 `PeriodicBlock` 消费点，断言它们都走递归辅助函数——但这需要 AST 分析，成本可能高于收益，本轮不做。
- **`fit/global_search.py` 的周期先验语义变模糊。** `periods = initial.period_a or (float(declared.sum()),)` 在嵌套下"一个周期"指哪一层不唯一（外层周期 vs 内层周期）。Task 5 取最外层展开总厚度，但 Bragg 峰实际对应内层周期，全局搜索的先验可能给错量级。这是**已知的物理层不完备**，需要真实嵌套数据验证后再调。
- **工作区当前不干净。** HEAD `df7c826` 之上有 interface-transitions / priors / sld-bands 的未提交改动。执行本计划前必须先确认这些改动的归属（提交或 stash 由用户决定），否则 Task 各批次的 `git add` 会混入无关文件。
