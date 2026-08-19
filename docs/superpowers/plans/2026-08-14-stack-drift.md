# 堆栈厚度与粗糙度漂移（stack-drift）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让周期块（`PeriodicBlock`）的层厚/粗糙度沿重复方向按 linear/sine/random 三种规律漂移，仅引入 1 个自由标量（drift_scale），非漂移块逐位不变。

**Architecture:** 采用用户已选的 A2「编译期脱糖为显式层+约束」。声明期 `PeriodicBlock` 携带一个持久化的 `DriftSpec`（只含原始标量）；编译期 `_periodic_definitions` 为漂移块额外发射 1 个自由 `drift_scale` 与逐副本派生参数名，`compile_fit_problem` 从 `structure` 再生漂移 `ConstraintRule`（不持久化）绑定每个副本 = 基元 × (1 + drift_scale·c_k)，其中 c_k 是编译期常量；物理层用一个绝不持久化的 `_ExpandedDriftBlock` 逐副本展开为显式板，且**不发 `PeriodicSpan`**（漂移块无法走精确重复的矩阵幂快路径）。

**Tech Stack:** Python 3.12, NumPy 2.x, PySide6 6.8+, pytest 8.3+。

**Design source:** `docs/superpowers/specs/2026-08-09-stack-drift-design.md`

## Global Constraints

- 不新增第三方依赖（sin/cos 用 `math`/`numpy`，随机用 `numpy.random.default_rng`）。
- 前置：本计划依赖 `2026-08-13-expression-constraints.md` 已合并进 `repo/`（提供 `ConstraintRule`/`ConstraintNode`/`ParameterReference`/`validate_constraint_stage_split` 与运行期约束求值）。截至成文，`repo/src/xrr_fitter/model/parameters.py` 尚无 `ConstraintRule` → **Task 0 在执行前复核全部锚点行号**（本计划锚点取自 worktree `wt-expr-constraints`，合并后行号会漂移）。
- 无漂移（`drift=None`）时：参数声明、序列化字节、checkpoint 指纹、物理展开必须与本特性合入前**逐位一致**。
- 运行期零 RNG：随机漂移的偏移在编译期一次性物化为常量；自由参数只有 drift_scale（幅度/速率标量）。
- 中文回复；代码/命令/路径/API 字段保持原文。测试从 `repo/` 根运行：`.venv/bin/python -m pytest <path> --import-mode=importlib -v`（worktree 必须用各自独立 `.venv`）。
- 不 stage `.claude/`；不做破坏性 git 操作；每个 Task 末尾按需 commit。

---
## 对 spec 的修正

1. **（硬阻塞）spec「只做两件事」低估了范围。** 除「补 sin/cos 节点」与「漂移展开助手」外，还必须：给漂移块新增逐副本可寻址参数名与 1 个自由 `drift_scale`（改 `fit/parameters.py:_periodic_definitions`）、编译期再生并合并漂移约束（改 `fit/problem.py`）、新增 ephemeral `_ExpandedDriftBlock` 与 `physics/stack.py`+`physics/geometry.py` 的逐副本分支、序列化/codec/checkpoint 的 `drift` 可选字段接线。计划据此拆成 13 个 Task。

2. **（硬阻塞）三种 kind 统一为编译期常量 c_k，运行期不发 sin 节点。** period/phase 固定、唯一自由量是 drift_scale（幅度）时，`sin(2πk/period+phase)` 是逐副本编译常量。故 `t_k = base·(1 + drift_scale·c_k)`，c_k：linear=k、sine=`sin(2πk/period+phase)`、random=种子派生偏移。运行期约束树三种 kind 同形（`mul/add/const/ref`），**不含 sin 节点**。spec 要求的 sin/cos 节点仍按 Task 1–3 补齐并**独立验证**（通用 expr-constraints 扩展 + 未来可拟合 phase），但漂移生成器不依赖它。

3. **（硬阻塞）随机种子自包含于 `DriftSpec.seed`。** `compile_fit_problem` 拿不到 `project`/`dataset_id`，无法用 `project.master_seed`；且 spec 要求种子「可见且可编辑」。故随机偏移 = `numpy.random.default_rng(DriftSpec.seed).uniform(-1, 1, repeats-1)` 逐副本采样，**只依赖 `DriftSpec.seed`**（`drift_coefficients` 不接收块索引/spawn_key）；两个块要相互独立时，由用户为各块选不同的 `seed` 保证。**不触碰 `SERVICE_SEED_TREE_VERSION`**（早期假设作废）。此处偏离 spec line 27、82-84「派生自 `SERVICE_SEED_TREE_VERSION`、显示为『工程种子+块偏移量』」——A2 架构下编译期无 `project`，故以自包含 `DriftSpec.seed` 取代。

4. **copy 0 取自由基元、不调制；k≥1 相对 base 调制（c_0≡0）。** spec 纯公式在 i=0 处含 `sin(φ)` 调制，本计划让第 0 副本 = 既有自由基元（保命名、保 settings/priors），k=1..R-1 才派生。差异记录在案：更简、天然保号、下游命名零移位。

5. **无需新增 api 校验/设置对；漂移随 `set_structure` 持久化。** 漂移只能挂在 `PeriodicBlock` 上 → spec 的「作用于非连续层报错」由类型系统天然保证（无此字段可挂）。标量合法性落在 `DriftSpec.__post_init__`。**逐副本几何合法性由运行期约束域检查兜底**：派生副本定义的 `initial` 取 `base.initial`（base 已过 bounds → 逐副本 `ParameterDefinition.__post_init__` 天然通过，不做几何判定），漂移后的物理厚度由约束规则求值产生，越界时 `_apply_constraint_values`（evaluation.py）抛 `EvaluationConstraintError("constraint_out_of_bounds:{target}")` → 该候选向量被拒（不是崩溃）。api 仅**再导出 `DriftSpec`**（GUI 需构造），无新函数。

6. **spec 行号已过期，Task 0 复核。** `_append_periodic` spec:205→worktree stack.py:228；`_validate_roughness` spec:237→worktree:267；`derivatives.py:360/371` 快路径判据合并后重验。

7. **漂移约束绕过 service 校验，须专门测试。** 编译期再生的漂移规则不经 `validate_constraint_rules`/`validate_constraint_stage_split`；运行期相位归属由 `_ordered_phase_rules` 按 target transform 自动分（thickness→非粗糙度相位，roughness→粗糙度相位）。Task 8/13 专测。

8. **漂移规则去重键 = target 名含 `.repeat.`。** 只有 `compile_stage_problem` / `compile_fixed_parameter_problem` 会带着 `problem.constraint_rules`（可能已含上一轮再生的漂移规则）重新进入 `compile_fit_problem` → 合并前先剔除 incoming 中 target 含 `.repeat.` 者，再注入本轮再生，保证幂等。**`compile_joint_problem` 不重入 `compile_fit_problem`**（它用 `replace(problem, constraint_rules=…)` 组装，见 joint_problem.py:313/340），故联合编译不走本去重；漂移 × 联合的隐患是另一条路径——`_compiled_local_constraints`（joint_problem.py:182）会拿漂移规则的哨兵 dataset_id `__drift__` 去比对成员 dataset_id 并抛错，须由 Task 8B 的 `rebind_drift_dataset` 在该检查前归一。

9. **嵌套周期 × 漂移：本期不支持，显式报错。** `2026-08-13-nested-periodic-stacks.md` 已在，若漂移块的 `layers` 含嵌套 `PeriodicBlock`，`DriftSpec.__post_init__` 无法看到（只有标量），故在 `_periodic_definitions`/展开处遇「漂移块含非 `LayerSpec` 子层」抛 `ValueError`。

10. **`_ExpandedDriftBlock` 绝不持久化。** 仅存在于 `rebuild_structure` 之后的物理展开链；不进 codec、不进 checkpoint、不进 `problem.structure`（后者仍是声明式 `PeriodicBlock+drift`，指纹只见声明）。

## File Structure

| 文件 | 责任 | 动作 |
| --- | --- | --- |
| `src/xrr_fitter/model/parameters.py` | 约束节点定义 | 加 `CONSTRAINT_UNARY_OPS` + `ConstraintNode` 一元分支 |
| `src/xrr_fitter/evaluation.py` | 约束求值+导数 | sin/cos 值与解析导数 |
| `src/xrr_fitter/io/project_codec.py` | 约束节点读端 | 放行 sin/cos op |
| `src/xrr_fitter/model/structure.py` | 结构声明 | 加 `DriftSpec` + `PeriodicBlock.drift` + ephemeral `_ExpandedDriftBlock` |
| `src/xrr_fitter/io/codec_declarations.py` | 结构序列化 | `drift` 可选字段（镜像 `transition`） |
| `src/xrr_fitter/fit/checkpoint.py` | 指纹稳定 | 注册 `("PeriodicBlock","drift"):None` |
| `src/xrr_fitter/fit/parameters.py` | 参数声明 | `_periodic_definitions` 逐副本 + drift_scale |
| `src/xrr_fitter/fit/drift.py` | 漂移规则生成 | **新建**：`drift_constraint_rules(structure)` + c_k |
| `src/xrr_fitter/fit/problem.py` | 编译入口 | 编译期再生+合并+去重漂移规则 |
| `src/xrr_fitter/fit/joint_problem.py` | 联合约束合并 | `_compiled_local_constraints` 遇 `__drift__` 先 `rebind_drift_dataset` 归一到成员 dataset_id |
| `src/xrr_fitter/physics/stack.py` | 物理展开 | `_replace_component`/`_append_component` 加漂移分支 |
| `src/xrr_fitter/physics/geometry.py` | 雅可比展开 | `append_component`/切向 漂移分支 |
| `src/xrr_fitter/api.py` | 公共门面 | 再导出 `DriftSpec` |
| `src/xrr_fitter/gui/structure/dialogs.py` | 周期块对话框 | 漂移下拉+目标+种子+实时性能警告 |
| `src/xrr_fitter/gui/structure/editor.py` | 结构树 | 漂移 tooltip（不加列） |

---
### Task 0: 合并后锚点复核（gate，只读）

**Files:** 只读，无改动。

**Interfaces:**
- Produces: 一份「本计划全部 `file:line` 锚点在 `repo/` 合并 expr-constraints 后的实际行号」映射，供后续 Task 定位。

- [ ] **Step 1: 确认前置已合并**

Run: `grep -n "class ConstraintRule\|def validate_constraint_stage_split" src/xrr_fitter/model/parameters.py`
Expected: 两者都命中。若为空 → **停止**，先合并 `2026-08-13-expression-constraints.md`。

- [ ] **Step 2: 复核每个锚点**

Run: 逐一 `grep -n` 定位并记下当前行号（本计划所有 `worktree:NNN` 均替换为复核值）：
```bash
grep -n "CONSTRAINT_BINARY_OPS\|CONSTRAINT_LEAF\|class ConstraintNode" src/xrr_fitter/model/parameters.py
grep -n "_evaluate_constraint_value\|_constraint_value_and_grad\|_ordered_phase_rules" src/xrr_fitter/evaluation.py
grep -n "_constraint_node_from_dict\|MAX_CONSTRAINT_DEPTH" src/xrr_fitter/io/project_codec.py
grep -n "class PeriodicBlock" src/xrr_fitter/model/structure.py
grep -n "_periodic_to_dict\|_periodic_from_dict" src/xrr_fitter/io/codec_declarations.py
grep -n "POST_FREEZE_OMITTED_DEFAULTS" src/xrr_fitter/fit/checkpoint.py
grep -n "_periodic_definitions" src/xrr_fitter/fit/parameters.py
grep -n "rules = tuple(constraint_rules)\|def _mark_constrained\|def compile_fit_problem" src/xrr_fitter/fit/problem.py
grep -n "_append_periodic\|_append_component\|_validate_roughness\|_replace_periodic\|def rebuild_structure" src/xrr_fitter/physics/stack.py
grep -n "_append_periodic_geometry\|def expand_structure_with_jacobian\|def append_component" src/xrr_fitter/physics/geometry.py
```
Expected: 全部命中；无命中项记为「合并改名」并在对应 Task 就地修正。

- [ ] **Step 3: 复核快路径判据**

Run: `grep -n "array_equal\|PeriodicSpan" src/xrr_fitter/physics/derivatives.py`
Expected: 找到逐位相等判据（spec:360/371）。记下行号，Task 10 断言「漂移块不进此路径」。

（Task 0 无 commit，产出为行号映射。）

---

### Task 1: sin/cos 约束节点（一元算子）

**Files:**
- Modify: `src/xrr_fitter/model/parameters.py`（`CONSTRAINT_BINARY_OPS`/`CONSTRAINT_OPS` 附近，worktree:358-360；`ConstraintNode.__post_init__` worktree:367-399）
- Test: `tests/unit/model/test_constraint_node.py`

**Interfaces:**
- Produces: `CONSTRAINT_UNARY_OPS: frozenset[str] = frozenset({"sin", "cos"})`；`ConstraintNode(op, operands, reference=None, value=None)` 支持 `op ∈ CONSTRAINT_UNARY_OPS` 时恰 1 个 operand、无 reference/value。
- Consumes: 无（纯扩展）。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/model/test_constraint_node.py
import pytest
from xrr_fitter.model.parameters import (
    CONSTRAINT_UNARY_OPS, ConstraintNode, ParameterReference,
)

def _leaf(name="a"):
    return ConstraintNode(op="ref", operands=(), reference=ParameterReference("d", name))

def test_unary_ops_are_sin_cos():
    assert CONSTRAINT_UNARY_OPS == frozenset({"sin", "cos"})

def test_sin_accepts_single_operand():
    node = ConstraintNode(op="sin", operands=(_leaf(),))
    assert node.op == "sin" and len(node.operands) == 1

def test_sin_rejects_two_operands():
    with pytest.raises(ValueError):
        ConstraintNode(op="sin", operands=(_leaf("a"), _leaf("b")))

def test_sin_rejects_reference_or_value():
    with pytest.raises(ValueError):
        ConstraintNode(op="cos", operands=(_leaf(),), value=1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/model/test_constraint_node.py -v --import-mode=importlib`
Expected: FAIL（`CONSTRAINT_UNARY_OPS` 未定义 / 一元分支未校验）。

- [ ] **Step 3: 最小实现**

在 `CONSTRAINT_BINARY_OPS` 定义之后加：
```python
CONSTRAINT_UNARY_OPS = frozenset({"sin", "cos"})
```
把 `CONSTRAINT_OPS` 并集扩为：
```python
CONSTRAINT_OPS = CONSTRAINT_BINARY_OPS | CONSTRAINT_UNARY_OPS | CONSTRAINT_LEAF
```
在 `ConstraintNode.__post_init__` 的算子分支里，`binary` 校验之后加一元分支：
```python
elif self.op in CONSTRAINT_UNARY_OPS:
    if len(self.operands) != 1:
        raise ValueError(f"unary op {self.op!r} needs exactly one operand")
    if self.reference is not None or self.value is not None:
        raise ValueError(f"unary op {self.op!r} takes no reference/value")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/model/test_constraint_node.py -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/model/parameters.py tests/unit/model/test_constraint_node.py
git commit -m "feat(constraints): add sin/cos unary constraint nodes"
```

---
### Task 2: sin/cos 求值与解析导数

**Files:**
- Modify: `src/xrr_fitter/evaluation.py`（`_evaluate_constraint_value` worktree:612-631，ref 解析在 617；`_constraint_value_and_grad` worktree:648-686）
- Test: `tests/unit/test_evaluation.py`（或既有约束求值测试文件，Task 0 确认路径）

**Interfaces:**
- Consumes: `CONSTRAINT_UNARY_OPS`、`ConstraintNode`（Task 1）。
- Produces: `_evaluate_constraint_value` 对 sin/cos 返回 `math.sin/os(operand_value)`；`_constraint_value_and_grad` 对 sin/cos 返回 `(sin(u), cos(u)·∂u)` / `(cos(u), −sin(u)·∂u)`，梯度容器与现有 pow 分支一致。

- [ ] **Step 1: 读 pow 分支确认梯度容器形态**

Run: `grep -n '"pow"' src/xrr_fitter/evaluation.py`，Read `_constraint_value_and_grad` 里 pow 的单操作数梯度缩放写法（dict 推导还是数组逐元素乘），一元分支照此写。

- [ ] **Step 2: 写失败测试**

```python
# 解析导数 vs 有限差分；纯 sin 树（无自由 period/phase，仍验证节点能力）
import math
import numpy as np
from xrr_fitter.model.parameters import ConstraintNode, ParameterReference, ConstraintRule
from xrr_fitter.evaluation import _evaluate_constraint_value, _constraint_value_and_grad

def _sin_rule():
    ref = ConstraintNode(op="ref", operands=(), reference=ParameterReference("d", "x"))
    return ConstraintNode(op="sin", operands=(ref,))

def test_sin_value():
    node = _sin_rule()
    assert _evaluate_constraint_value(node, {"x": 0.3}) == math.sin(0.3)

def test_sin_grad_matches_finite_difference():
    node = _sin_rule()
    x0, h = 0.3, 1e-6
    _, grad = _constraint_value_and_grad(node, {"x": x0})
    fd = (_evaluate_constraint_value(node, {"x": x0 + h})
          - _evaluate_constraint_value(node, {"x": x0 - h})) / (2 * h)
    assert np.isclose(grad["x"], fd, atol=1e-6)  # 若容器为数组则改索引
```
（`_evaluate_constraint_value`/`_constraint_value_and_grad` 的确切签名以 Task 1/0 复核为准；上面按「rule 表达式节点 + 变量名→值 dict」的既有约定书写。）

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_evaluation.py -k "sin" -v --import-mode=importlib`
Expected: FAIL（一元 op 未处理，落到未知 op 分支报错）。

- [ ] **Step 4: 最小实现**

`_evaluate_constraint_value` 在 ref 解析（worktree:617）与 binary 之间插一元分支：
```python
if node.op in CONSTRAINT_UNARY_OPS:
    inner = _evaluate_constraint_value(node.operands[0], values)
    return math.sin(inner) if node.op == "sin" else math.cos(inner)
```
`_constraint_value_and_grad`（worktree:648-686，在 binary 组合前）插：
```python
if node.op in CONSTRAINT_UNARY_OPS:
    u, du = _constraint_value_and_grad(node.operands[0], values)
    if node.op == "sin":
        value, mult = math.sin(u), math.cos(u)
    else:
        value, mult = math.cos(u), -math.sin(u)
    grad = {name: mult * partial for name, partial in du.items()}  # 镜像 pow 单操作数缩放
    return value, grad
```
确保文件顶部已 `import math`（缺则补）。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/test_evaluation.py -k "sin" -v --import-mode=importlib`
Expected: PASS。补一个 cos 版本测试同样通过。

- [ ] **Step 6: Commit**

```bash
git add src/xrr_fitter/evaluation.py tests/unit/test_evaluation.py
git commit -m "feat(constraints): evaluate sin/cos with analytic derivatives"
```

---
### Task 3: 约束节点读端放行 sin/cos

**Files:**
- Modify: `src/xrr_fitter/io/project_codec.py`（`_constraint_node_from_dict` 的 op 白名单闸门 worktree:188）
- Test: `tests/unit/io/test_project_codec.py`

**Interfaces:**
- Consumes: `CONSTRAINT_UNARY_OPS`（Task 1）。
- Produces: 写端已对任意 op emit（结构对称），读端接受 sin/cos → sin/cos 节点可完整往返；`MAX_CONSTRAINT_DEPTH`（worktree=64）不变。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/io/test_project_codec.py
from xrr_fitter.model.parameters import ConstraintNode, ParameterReference
from xrr_fitter.io.project_codec import _constraint_node_to_dict, _constraint_node_from_dict

def test_sin_node_round_trips():
    ref = ConstraintNode(op="ref", operands=(), reference=ParameterReference("d", "x"))
    node = ConstraintNode(op="sin", operands=(ref,))
    restored = _constraint_node_from_dict(_constraint_node_to_dict(node))
    assert restored == node
```
（写/读函数名以 Task 0 复核为准；若写端非对称也需同步放行。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/io/test_project_codec.py -k "sin_node" -v --import-mode=importlib`
Expected: FAIL（读端闸门只认 `CONSTRAINT_BINARY_OPS`，sin 落入未知 op 报错）。

- [ ] **Step 3: 最小实现**

worktree:188 处：
```python
if op in CONSTRAINT_BINARY_OPS | CONSTRAINT_UNARY_OPS:
```
（若 import 未含 `CONSTRAINT_UNARY_OPS` 则补进 import。）

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/io/test_project_codec.py -k "sin_node" -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/io/project_codec.py tests/unit/io/test_project_codec.py
git commit -m "feat(io): accept sin/cos constraint nodes on decode"
```

---
### Task 4: `DriftSpec` + `PeriodicBlock.drift` 字段

**Files:**
- Modify: `src/xrr_fitter/model/structure.py`（`PeriodicBlock` 定义 repo:272-287；`DriftSpec` 新增于其前）
- Test: `tests/unit/model/test_structures.py`

**Interfaces:**
- Produces:
  - `DriftSpec(kind, target, amount=0.0, period=0.0, phase=0.0, seed=0)`，frozen+slots；`kind ∈ {"linear","sine","random"}`，`target ∈ {"thickness","roughness"}`。
  - `PeriodicBlock.drift: DriftSpec | None = None`（在 `top_roughness_a` 之后，保持既有位置参数顺序）。
  - `tests/support/drift_cases.py`：漂移测试共享构造器模块（模块级 builder 函数，沿用 `model_cases.py` 惯用法），供 Task 4-13 各单测显式调用复用（本任务 Step 3b 创建）。
- Consumes: 无。
- 备注：`_periodic_layers`（repo:206-207）已强制块内层皆 `LayerSpec` → 漂移块天然不含嵌套周期，无需额外守卫（对 spec 修正 9 已被既有校验覆盖）。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/model/test_structures.py（追加）
import pytest
from xrr_fitter.model.structure import DriftSpec, LayerSpec, MaterialSpec, PeriodicBlock

_FILM = LayerSpec("film", MaterialSpec("SiO2", "SiO2", 2.2), 20.0, roughness_a=2.0)

def test_drift_spec_defaults_and_kind_validation():
    d = DriftSpec(kind="linear", target="thickness", amount=0.1)
    assert (d.period, d.phase, d.seed) == (0.0, 0.0, 0)
    with pytest.raises(ValueError):
        DriftSpec(kind="bogus", target="thickness")

def test_sine_requires_positive_period():
    with pytest.raises(ValueError):
        DriftSpec(kind="sine", target="thickness", amount=0.1, period=0.0)

def test_random_requires_nonneg_int_seed():
    with pytest.raises(ValueError):
        DriftSpec(kind="random", target="roughness", amount=0.1, seed=-1)

def test_periodic_block_accepts_drift_and_requires_two_repeats():
    block = PeriodicBlock(name="p", layers=(_FILM,), repeats=3,
                          drift=DriftSpec(kind="linear", target="thickness", amount=0.05))
    assert block.drift.kind == "linear"
    with pytest.raises(ValueError):
        PeriodicBlock(name="p", layers=(_FILM,), repeats=1,
                      drift=DriftSpec(kind="linear", target="thickness", amount=0.05))

def test_periodic_block_without_drift_defaults_none():
    assert PeriodicBlock(name="p", layers=(_FILM,), repeats=2).drift is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/model/test_structures.py -k "drift" -v --import-mode=importlib`
Expected: FAIL（`ImportError: cannot import name 'DriftSpec'`；实现后 `PeriodicBlock` 尚无 `drift` 字段亦为 FAIL——RED 仅因待实现行为缺失，本步不依赖 `tests.support.drift_cases`）。

- [ ] **Step 3: 最小实现**

在 `PeriodicBlock` 之前插入：
```python
_DRIFT_KINDS = frozenset({"linear", "sine", "random"})
_DRIFT_TARGETS = frozenset({"thickness", "roughness"})


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """Per-repeat drift law for a periodic block (primitive scalars only)."""

    kind: str
    target: str
    amount: float = 0.0
    period: float = 0.0
    phase: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _DRIFT_KINDS:
            raise ValueError(f"drift.kind must be one of {sorted(_DRIFT_KINDS)}")
        if self.target not in _DRIFT_TARGETS:
            raise ValueError(f"drift.target must be one of {sorted(_DRIFT_TARGETS)}")
        if not isfinite(self.amount):
            raise ValueError("drift.amount must be finite")
        if self.kind == "sine":
            if not isfinite(self.period) or self.period <= 0.0:
                raise ValueError("drift.period must be finite and positive for sine drift")
            if not isfinite(self.phase):
                raise ValueError("drift.phase must be finite")
        if self.kind == "random":
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
                raise ValueError("drift.seed must be a non-negative integer for random drift")
```
`PeriodicBlock` 加字段与校验：
```python
    top_roughness_a: float | None = None
    drift: DriftSpec | None = None
```
`__post_init__` 末尾加：
```python
        if self.drift is not None:
            if not isinstance(self.drift, DriftSpec):
                raise TypeError(f"{self.name}.drift must be a DriftSpec")
            if self.repeats < 2:
                raise ValueError(f"{self.name}.drift requires repeats >= 2")
```

- [ ] **Step 3b: 新建共享测试构造器 `tests/support/drift_cases.py`**

漂移测试的全部构造器集中于此模块（沿用 `model_cases.py` 的“模块级 builder 函数 + 显式调用”惯用法：无 `@pytest.fixture`、无下划线私有别名）。`DriftSpec` 已在 Step 3 定义，故本步可安全在模块顶层 import；`drift_values` 对 `drift_coefficients`（Task 7）采用函数体内惰性 import，模块顶层不引入前向依赖。各 builder 的函数体只在其依赖实现落地后的任务里被调用，故早期任务仅做模块 import 是安全的。

```python
# tests/support/drift_cases.py（新建）
"""Shared stack-drift test builders (module-level functions; model_cases idiom)."""
from __future__ import annotations

from dataclasses import replace

from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    DriftSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from tests.support.model_cases import prepared_data

_AIR = MaterialSpec("Air", None, None, 0.0j)
_SILICON = MaterialSpec("Si", "Si", 2.329)
_SILICA = MaterialSpec("SiO2", "SiO2", 2.2)


def media() -> MaterialSpec:
    return _AIR


def make_layer(name="film", thickness_a=20.0, roughness_a=2.0) -> LayerSpec:
    return LayerSpec(name, _SILICA, thickness_a, roughness_a=roughness_a)
def two_layer_block(repeats=3) -> PeriodicBlock:
    return PeriodicBlock(
        name="p",
        layers=(
            LayerSpec("a", _SILICA, 20.0, roughness_a=2.0),
            LayerSpec("b", _SILICON, 500.0, roughness_a=3.0),
        ),
        repeats=repeats,
    )


def two_layer_block_with_thickness_drift(repeats=3, amount=0.1) -> PeriodicBlock:
    return replace(
        two_layer_block(repeats),
        drift=DriftSpec(kind="linear", target="thickness", amount=amount),
    )


def drift_block() -> PeriodicBlock:
    return two_layer_block_with_thickness_drift()


def one_drift_block_structure() -> StructureSpec:
    return StructureSpec(
        fronting=_AIR,
        components=(drift_block(),),
        backing=_SILICON,
        backing_roughness_a=3.0,
    )


def plain_periodic_structure() -> StructureSpec:
    return StructureSpec(
        fronting=_AIR,
        components=(two_layer_block(),),
        backing=_SILICON,
        backing_roughness_a=3.0,
    )


def drift_structure() -> StructureSpec:
    return one_drift_block_structure()


def wavelength() -> float:
    return 1.5406
def drift_case() -> tuple:
    return (
        prepared_data(),
        drift_structure(),
        InstrumentSpec(instrument_id="lab"),
        FitConfig.standard(11),
    )


def drift_values(structure) -> dict[str, float]:
    from xrr_fitter.fit.drift import drift_coefficients

    block = structure.components[0]
    drift = block.drift
    coeffs = drift_coefficients(drift, block.repeats)
    prefix = "component.0"
    values = {f"{prefix}.drift_scale": drift.amount}
    for index, layer in enumerate(block.layers):
        base = f"{prefix}.layer.{index}"
        values[f"{base}.thickness_a"] = layer.thickness_a
        values[f"{base}.roughness_a"] = layer.roughness_a
        values[f"{base}.density_scale"] = layer.density_scale
        for k in range(1, block.repeats):
            values[f"{prefix}.repeat.{k}.layer.{index}.thickness_a"] = (
                layer.thickness_a * (1.0 + drift.amount * coeffs[k])
            )
    return values
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/model/test_structures.py -k "drift" -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/model/structure.py tests/unit/model/test_structures.py tests/support/drift_cases.py
git commit -m "feat(structure): add DriftSpec and PeriodicBlock.drift field"
```

---
### Task 5: `DriftSpec` 序列化（可选字段，缺省不写）

**Files:**
- Modify: `src/xrr_fitter/io/codec_declarations.py`（`_periodic_to_dict` repo:267-274，`_periodic_from_dict` repo:277-288；新增 `_drift_to_dict`/`_drift_from_dict`）
- Test: `tests/unit/io/test_codec_declarations.py`

**Interfaces:**
- Consumes: `DriftSpec`、`PeriodicBlock.drift`（Task 4）。
- Produces: 有 drift → 往返相等；无 drift → payload **不含** `"drift"` 键（镜像 `transition`，保旧文件逐位不变，无 `SCHEMA_VERSION` bump）。
- 备注：**不**把 `drift` 加进 `codec_common.OPTIONAL_FIELDS`（该表用于其它自动机制）。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/io/test_codec_declarations.py（新建）
from xrr_fitter.model.structure import DriftSpec, PeriodicBlock
from xrr_fitter.io.codec_declarations import _periodic_to_dict, _periodic_from_dict
from tests.support.drift_cases import make_layer

def test_periodic_without_drift_omits_key():
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=2)
    assert "drift" not in _periodic_to_dict(block)

def test_periodic_drift_round_trips():
    drift = DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.5)
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=5, drift=drift)
    assert _periodic_from_dict(_periodic_to_dict(block)) == block
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/io/test_codec_declarations.py -k "drift" -v --import-mode=importlib`
Expected: FAIL（`drift` 未编码/解码）。

- [ ] **Step 3: 最小实现**

新增两个 helper（放在 `_periodic_to_dict` 前）：
```python
def _drift_to_dict(value: DriftSpec) -> dict[str, object]:
    return {
        "kind": value.kind,
        "target": value.target,
        "amount": value.amount,
        "period": value.period,
        "phase": value.phase,
        "seed": value.seed,
    }


def _drift_from_dict(value: object) -> DriftSpec:
    payload = _mapping(value, {"kind", "target", "amount", "period", "phase", "seed"}, "drift")
    return DriftSpec(
        kind=payload["kind"],
        target=payload["target"],
        amount=payload["amount"],
        period=payload["period"],
        phase=payload["phase"],
        seed=payload["seed"],
    )
```
`_periodic_to_dict` 末尾（return 前改为构造 payload 后按需加键）：
```python
def _periodic_to_dict(value: PeriodicBlock) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "periodic_block",
        "name": value.name,
        "layers": [_layer_to_dict(layer) for layer in value.layers],
        "repeats": value.repeats,
        "top_roughness_a": value.top_roughness_a,
    }
    if value.drift is not None:
        payload["drift"] = _drift_to_dict(value.drift)
    return payload
```
`_periodic_from_dict` 加 `optional={"drift"}` 并解码：
```python
    payload = _mapping(
        value,
        {"kind", "name", "layers", "repeats", "top_roughness_a"},
        "periodic block",
        optional={"drift"},
    )
    drift = payload.get("drift")
    return PeriodicBlock(
        name=payload["name"],
        layers=tuple(_layer_from_dict(item) for item in _sequence(payload["layers"], "layers")),
        repeats=payload["repeats"],
        top_roughness_a=payload["top_roughness_a"],
        drift=None if drift is None else _drift_from_dict(drift),
    )
```
确保 `DriftSpec` 已在本文件 import。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/io/test_codec_declarations.py -k "drift" -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/io/codec_declarations.py tests/unit/io/test_codec_declarations.py
git commit -m "feat(io): serialize PeriodicBlock.drift as omitted-when-absent field"
```

---
### Task 6: checkpoint 指纹稳定（注册 `("PeriodicBlock","drift"):None`）

**Files:**
- Modify: `src/xrr_fitter/fit/checkpoint.py`（`POST_FREEZE_OMITTED_DEFAULTS` repo:24-28）
- Test: 既有 `tests/unit/fit/test_frozen_stage_search.py`（字面 hash）与 `tests/unit/fit/test_checkpoint.py::test_prior_defaults_do_not_perturb_frozen_fingerprints`（Task 0 用 `grep -rn "POST_FREEZE\|1f0681cf" tests` 确认精确路径）；新增聚焦测试。

**Interfaces:**
- Consumes: `PeriodicBlock.drift`（Task 4）。
- Produces: `drift=None` 的块规范化时**省略** `drift` 键 → 无漂移结构指纹逐位不变、旧 checkpoint 仍可 resume；漂移块产生不同指纹（各得其身份）。
- **顺序警告**：Task 4 加字段后、本 Task 注册前，`_canonical_dataclass` 会带上 `drift=None` → 冻结指纹测试变 RED。本 Task 即修复它。

- [ ] **Step 1: 确认冻结测试已 RED**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_frozen_stage_search.py tests/unit/fit/test_checkpoint.py -k "frozen or fingerprint" -v --import-mode=importlib`
Expected: FAIL（无漂移块因新增 `drift=None` 字段导致 `structure_fingerprint` 漂移，字面 hash 不匹配）。

- [ ] **Step 2: 写聚焦失败测试**

```python
# tests/unit/fit/test_checkpoint.py（追加）
from xrr_fitter.fit.checkpoint import _canonical
from xrr_fitter.model.structure import DriftSpec, PeriodicBlock
from tests.support.drift_cases import make_layer

def test_no_drift_block_omits_drift_key():
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=2)
    assert "drift" not in _canonical(block)

def test_drifted_block_includes_drift_key():
    block = PeriodicBlock(name="p", layers=(make_layer(),), repeats=2,
                          drift=DriftSpec(kind="linear", target="thickness", amount=0.1))
    assert "drift" in _canonical(block)
```

- [ ] **Step 3: 最小实现**

`POST_FREEZE_OMITTED_DEFAULTS` 加一行：
```python
    ("PeriodicBlock", "drift"): None,
```

- [ ] **Step 4: 跑全部相关测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_frozen_stage_search.py tests/unit/fit/test_checkpoint.py -v --import-mode=importlib`
Expected: PASS（冻结字面 hash 恢复；聚焦测试通过）。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/fit/checkpoint.py tests/unit/fit/test_checkpoint.py
git commit -m "fix(checkpoint): omit PeriodicBlock.drift default to preserve frozen fingerprints"
```

---
### Task 7: 逐副本参数声明 + `drift_scale`

**Files:**
- Create: `src/xrr_fitter/fit/drift.py`（`drift_coefficients`）
- Modify: `src/xrr_fitter/fit/parameters.py`（`_periodic_definitions` repo:154-190；顶部补 `from dataclasses import replace`；`drift_coefficients` 只在 Task 8 的 `drift.py` 内部使用，parameters.py 不再导入它）
- Test: `tests/unit/fit/test_drift.py`、`tests/unit/fit/test_parameters.py`

**Interfaces:**
- Produces:
  - `drift_coefficients(drift: DriftSpec, repeats: int) -> tuple[float, ...]`：长度 `repeats`，`c[0]==0.0`（copy 0 不调制）；linear `c_k=k`，sine `c_k=sin(2π·k/period+phase)`，random `c_k=default_rng(seed).uniform(-1,1)` 第 k-1 抽样。
  - `_periodic_definitions` 在 `block.drift is not None` 时追加：1 个自由 `{prefix}.drift_scale`（transform="linear"，initial=`drift.amount`）与逐副本派生名 `{prefix}.repeat.{k}.layer.{i}.<family>`（k=1..R-1，family=`thickness_a`/`roughness_a`，用 `replace` 克隆 base 定义→保 bounds/transform 一致，**initial=`base.initial`**，不预调制物理值，见备注）。
- Consumes: `DriftSpec`（Task 4）、`_definition`/`_layer_definitions`（既有）。
- 备注：派生定义此刻 `constrained=False`；Task 8 的规则会把它们标为 constrained → 移出自由变量。派生 `initial` **必须取 `base.initial` 原值**：base 已过自己的 bounds 检查，故逐副本 `ParameterDefinition.__post_init__` 的 `initial∈[lower,upper]` 天然通过——它不是几何判定，若在此处预乘 `(1+drift.amount·c_k)` 反而会因负漂移压破 base 下界而在编译期误抛。真正的「非正/越界厚度报错」发生在**运行期**：漂移后的物理厚度由 Task 8 约束规则求值，越界时 `_apply_constraint_values` 抛 `EvaluationConstraintError("constraint_out_of_bounds:{target}")`，该候选向量被拒（非崩溃），对应 spec 的几何门。

- [ ] **Step 1: 写 `drift_coefficients` 失败测试**

```python
# tests/unit/fit/test_drift.py
import math
import numpy as np
from xrr_fitter.model.structure import DriftSpec
from xrr_fitter.fit.drift import drift_coefficients

def test_linear_coefficients():
    d = DriftSpec(kind="linear", target="thickness", amount=0.1)
    assert drift_coefficients(d, 4) == (0.0, 1.0, 2.0, 3.0)

def test_sine_coefficients_zero_at_copy0():
    d = DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.0)
    c = drift_coefficients(d, 3)
    assert c[0] == 0.0 and math.isclose(c[1], math.sin(2*math.pi/4))

def test_random_is_deterministic_bitwise():
    d = DriftSpec(kind="random", target="roughness", amount=0.2, seed=7)
    a = drift_coefficients(d, 6)
    b = drift_coefficients(d, 6)
    assert a == b and a[0] == 0.0 and all(-1.0 <= v <= 1.0 for v in a[1:])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -v --import-mode=importlib`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `fit/drift.py`（本 Task 只放 `drift_coefficients`）**

```python
"""Compile-time drift desugaring: per-copy coefficients and constraint rules."""

from __future__ import annotations

import math

import numpy as np

from xrr_fitter.model.structure import DriftSpec


def drift_coefficients(drift: DriftSpec, repeats: int) -> tuple[float, ...]:
    """Per-copy modulation constants c_k (c_0=0; copy 0 is the free base cell)."""
    coeffs: list[float] = [0.0]
    if drift.kind == "linear":
        coeffs.extend(float(k) for k in range(1, repeats))
    elif drift.kind == "sine":
        coeffs.extend(
            math.sin(2.0 * math.pi * k / drift.period + drift.phase)
            for k in range(1, repeats)
        )
    else:  # random — deterministic, self-contained in drift.seed
        rng = np.random.default_rng(drift.seed)
        coeffs.extend(float(v) for v in rng.uniform(-1.0, 1.0, size=max(0, repeats - 1)))
    return tuple(coeffs)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: 写 `_periodic_definitions` 失败测试**

```python
# tests/unit/fit/test_parameters.py（新建）
from xrr_fitter.fit.parameters import _periodic_definitions
from tests.support.drift_cases import two_layer_block, two_layer_block_with_thickness_drift

def _names(defs):
    return [d.name for d in defs]

def test_no_drift_definitions_unchanged():
    defs = _periodic_definitions("component.0", two_layer_block(), (2.0, 500.0))
    assert not any(".drift_scale" in n or ".repeat." in n for n in _names(defs))

def test_drift_adds_scale_and_percopy():  # repeats=3, 2 layers
    block = two_layer_block_with_thickness_drift()
    defs = _periodic_definitions("component.0", block, (2.0, 500.0))
    names = _names(defs)
    assert "component.0.drift_scale" in names
    for k in (1, 2):
        for i in (0, 1):
            assert f"component.0.repeat.{k}.layer.{i}.thickness_a" in names
    # 非目标族（roughness）不发逐副本
    assert not any(".repeat." in n and n.endswith("roughness_a") for n in names)
```

- [ ] **Step 6: 跑测试确认失败 → 实现 → 通过**

`_periodic_definitions` 改造（保留原 top_roughness/repeats 段），末尾追加漂移段：
```python
    if block.drift is not None:
        drift = block.drift
        family = "thickness_a" if drift.target == "thickness" else "roughness_a"
        definitions.append(
            _definition(
                f"{prefix}.drift_scale", f"{block.name} 漂移标度", "", "structure",
                drift.amount, min(-0.5, drift.amount), max(0.5, drift.amount), "linear", False,
            )
        )
        by_name = {d.name: d for d in definitions}
        for index in range(len(block.layers)):
            base = by_name[f"{prefix}.layer.{index}.{family}"]
            for k in range(1, block.repeats):
                definitions.append(
                    replace(
                        base,
                        name=f"{prefix}.repeat.{k}.layer.{index}.{family}",
                        display_name=f"{base.display_name} 副本{k}",
                        initial=base.initial,
                    )
                )
    return definitions
```
Run: `.venv/bin/python -m pytest tests/unit/fit/test_parameters.py -k "drift" -v --import-mode=importlib` → 先 FAIL 后 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/xrr_fitter/fit/drift.py src/xrr_fitter/fit/parameters.py tests/unit/fit/test_drift.py tests/unit/fit/test_parameters.py
git commit -m "feat(fit): emit per-copy drift definitions and free drift_scale"
```

---
### Task 8: 漂移约束规则生成 + 编译期注入与去重

**Files:**
- Modify: `src/xrr_fitter/fit/drift.py`（追加 `_DRIFT_DATASET`、`drift_constraint_rules`）
- Modify: `src/xrr_fitter/fit/problem.py`（`compile_fit_problem` worktree:209 注入点；顶部 `from xrr_fitter.fit.drift import drift_constraint_rules`）
- Test: `tests/unit/fit/test_drift.py`、`tests/unit/fit/test_problem_compilation.py`

**Interfaces:**
- Produces:
  - `drift_constraint_rules(structure: StructureSpec) -> tuple[ConstraintRule, ...]`：对每个 `drift is not None` 的 `PeriodicBlock`（组件下标 `c`），对每层 `i`、每副本 `k=1..R-1` 生成一条规则，target=`ParameterReference(_DRIFT_DATASET, f"component.{c}.repeat.{k}.layer.{i}.<family>")`，expression=`base·(1+drift_scale·c_k)`，其中 `base=ref(component.{c}.layer.{i}.<family>)`、`drift_scale=ref(component.{c}.drift_scale)`、`c_k=const`（来自 `drift_coefficients`）。表达式仅用 `mul`/`add`/`const`/`ref`——**不发 sin 节点**（period/phase 固定→c_k 是编译常量）。
  - `_DRIFT_DATASET = "__drift__"`：哨兵 dataset_id（evaluation 只按 `parameter_name` 解析，见 evaluation.py:617/661/743；单数据集编译从不消费 dataset_id）。
- Consumes: `drift_coefficients`（Task 7）、`ConstraintRule`/`ConstraintNode`/`ParameterReference`（model/parameters.py:337-425）。
- 关键顺序：`compile_fit_problem` 先建 definitions（含 Task 7 的 `.repeat.` 名，problem.py:211-214）再 `_mark_constrained`（215）→ `.repeat.` target 全部标 constrained、移出自由变量；`base`/`drift_scale` 仍自由 → 解析梯度经 `∂/∂base=1+drift_scale·c_k`、`∂/∂drift_scale=base·c_k` 双通道回流（drift_scale 可拟合）。
- 去重契约：注入点先剔除**入参**里 target 名含 `.repeat.` 的规则（上一轮编译回灌的旧漂移规则，来自 `compile_stage_problem`/`compile_fixed_parameter_problem` 回传 `problem.constraint_rules`），再拼接本轮从 structure 重新生成的 → stage/fixed 重编译不累积。用户不应手写 `.repeat.` 目标（漂移独占该命名空间）。
- 命名 schema（**刻意内联、不抽 fit 侧公共 helper**）：`component.{c}.repeat.{k}.layer.{i}.{family}` 在 Task 7（`_periodic_definitions` 发定义名）、Task 8（`drift_constraint_rules` 发 target 名）、Task 9（物理逐副本展开）多处以 f-string 内联复写。不抽共享构造器有硬约束——`ALLOWED["physics"]={"physics","model"}`，physics 无法 `import fit`；若把 schema 放 fit 侧再让 physics 消费即违反导入分层（放 model 侧又与漂移规则生成解耦、收益低）。house style 亦为就地 f-string。**跨点一致性靠三道运行/测试门兜底，而非靠单一真源**：① `_mark_constrained`（problem.py:215）对未在 definitions 中的 target 抛 `ValueError(f"constraint target not in dataset: {sorted(unknown)}")`——Task 7 定义名与 Task 8 target 名一旦漂开即编译期炸出；② 数值门 `test_drift_rule_values_compose_base_scale_coeff`（本 Task Step 5）；③ 去重门 `test_stage_recompile_does_not_accumulate_drift_rules` 依赖 target 名含 `.repeat.` 的过滤契约；④ Task 13 端到端逐副本比对。

- [ ] **Step 1: 写 `drift_constraint_rules` 失败测试**

```python
# tests/unit/fit/test_drift.py（新建）
from xrr_fitter.fit.drift import drift_constraint_rules, _DRIFT_DATASET
from tests.support.drift_cases import one_drift_block_structure, plain_periodic_structure

def test_rules_cover_every_copy_and_layer():  # 2 层, repeats=3
    rules = drift_constraint_rules(one_drift_block_structure())
    targets = {r.target.parameter_name for r in rules}
    assert targets == {
        f"component.0.repeat.{k}.layer.{i}.thickness_a" for k in (1, 2) for i in (0, 1)
    }
    assert all(r.target.dataset_id == _DRIFT_DATASET for r in rules)

def test_no_drift_yields_no_rules():
    assert drift_constraint_rules(plain_periodic_structure()) == ()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -k "rules or no_drift" -v --import-mode=importlib`
Expected: FAIL（`drift_constraint_rules` 不存在）。

- [ ] **Step 3: 实现 `drift_constraint_rules`（追加到 fit/drift.py）**

```python
from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference
from xrr_fitter.model.structure import PeriodicBlock, StructureSpec

_DRIFT_DATASET = "__drift__"  # sentinel; evaluation resolves by parameter_name only


def _ref(name: str) -> ConstraintNode:
    return ConstraintNode("ref", reference=ParameterReference(_DRIFT_DATASET, name))


def drift_constraint_rules(structure: StructureSpec) -> tuple[ConstraintRule, ...]:
    """Desugar every drifted block into per-copy ``target = base·(1+scale·c_k)`` rules."""
    rules: list[ConstraintRule] = []
    for index, component in enumerate(structure.components):
        if not isinstance(component, PeriodicBlock) or component.drift is None:
            continue
        prefix = f"component.{index}"
        drift = component.drift
        coeffs = drift_coefficients(drift, component.repeats)
        family = "thickness_a" if drift.target == "thickness" else "roughness_a"
        scale = _ref(f"{prefix}.drift_scale")
        for layer_index in range(len(component.layers)):
            base = _ref(f"{prefix}.layer.{layer_index}.{family}")
            for k in range(1, component.repeats):
                factor = ConstraintNode(
                    "add",
                    operands=(
                        ConstraintNode("const", value=1.0),
                        ConstraintNode("mul", operands=(scale, ConstraintNode("const", value=coeffs[k]))),
                    ),
                )
                target = ParameterReference(
                    _DRIFT_DATASET, f"{prefix}.repeat.{k}.layer.{layer_index}.{family}"
                )
                rules.append(ConstraintRule(target=target, expression=ConstraintNode("mul", operands=(base, factor))))
    return tuple(rules)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: 写编译期注入 + 去重失败测试**

```python
# tests/unit/fit/test_problem_compilation.py（追加）
import pytest

from xrr_fitter.evaluation import encode_physical_vector, values_by_name
from xrr_fitter.fit.problem import compile_fit_problem, compile_stage_problem
from tests.support.drift_cases import drift_case

def test_compile_marks_repeat_targets_constrained():  # data,structure,instrument,config
    problem = compile_fit_problem(*drift_case())
    constrained = {d.name for d in problem.parameter_definitions if d.constrained}
    assert "component.0.repeat.1.layer.0.thickness_a" in constrained
    assert "component.0.drift_scale" not in constrained          # 自由
    assert not any(v.name.startswith("component.0.repeat.") for v in problem.variables)

def test_stage_recompile_does_not_accumulate_drift_rules():
    problem = compile_fit_problem(*drift_case())
    n = len(problem.constraint_rules)
    values = {v.name: problem.parameter_definitions[v.parameter_index].initial for v in problem.variables}
    staged = compile_stage_problem(problem, "coarse", values)
    assert len(staged.constraint_rules) == n   # 去重生效，不翻倍

def test_drift_rule_values_compose_base_scale_coeff():
    # 数值闭环：约束规则求值必须真正产出 base·(1+drift_scale·c_k)，而不是只对齐名字。
    # 注入前 .repeat. 仍是自由变量、取各自 initial(=base.initial) → 断言失败(RED)；
    # 注入后被标 constrained、经规则解析 → GREEN。
    problem = compile_fit_problem(*drift_case())
    base_initial = {d.name: d.initial for d in problem.parameter_definitions}[
        "component.0.layer.0.thickness_a"
    ]
    unit = encode_physical_vector(problem, {"component.0.drift_scale": 0.1})  # 0.1 ∈ [-0.5, 0.5]
    values = values_by_name(problem, unit)
    # 线性 kind：c_k = k → repeat.k 厚度 = base·(1+0.1·k)
    assert values["component.0.repeat.1.layer.0.thickness_a"] == pytest.approx(base_initial * (1 + 0.1 * 1.0))
    assert values["component.0.repeat.2.layer.0.thickness_a"] == pytest.approx(base_initial * (1 + 0.1 * 2.0))
```

- [ ] **Step 6: 跑测试确认失败 → 实现 → 通过**

`compile_fit_problem` 把 `rules = tuple(constraint_rules)`（problem.py:209）替换为：
```python
    incoming = tuple(rule for rule in constraint_rules if ".repeat." not in rule.target.parameter_name)
    rules = incoming + drift_constraint_rules(structure)
```
Run: `.venv/bin/python -m pytest tests/unit/fit/test_problem_compilation.py -k "drift or repeat or accumulate" -v --import-mode=importlib` → 先 FAIL 后 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/xrr_fitter/fit/drift.py src/xrr_fitter/fit/problem.py tests/unit/fit/test_drift.py tests/unit/fit/test_problem_compilation.py
git commit -m "feat(fit): regenerate drift constraint rules at compile with .repeat. dedup"
```

---
### Task 8B: 漂移规则的联合编译归一（`rebind_drift_dataset`）

**背景（为什么需要）：** Task 8 以哨兵 `dataset_id=_DRIFT_DATASET`（`"__drift__"`）把漂移规则注入每个单数据集 `problem.constraint_rules`。`compile_joint_problem` **不重入** `compile_fit_problem`（它用 `replace(problem, constraint_rules=…)` 组装，joint_problem.py:313/340），而是先经 `_merged_constraints`→`_compiled_local_constraints`（joint_problem.py:182）逐成员核对「target 与所有 ref 的 dataset_id 必须等于成员 id」。`__drift__` 不等于任何真实成员 id → **抛 `ValueError("compiled local constraint dataset identity does not match the joint member: {dataset_id}")`**：任何含漂移块的数据集进入 joint 批处理即崩溃。根因＝哨兵 id 未在 joint 侧归一。

**Files:**
- Modify: `src/xrr_fitter/fit/drift.py`（新增 `rebind_drift_dataset`、`_rebind_node`；既有 import 追加 `_iter_references`）
- Modify: `src/xrr_fitter/fit/joint_problem.py`（顶部 `from xrr_fitter.fit.drift import rebind_drift_dataset`；`_compiled_local_constraints` 循环体 worktree:188-189 身份检查前先 rebind）
- Test: `tests/unit/fit/test_drift.py`、`tests/unit/fit/test_joint_problem.py`

**Interfaces:**
- Produces `rebind_drift_dataset(rule: ConstraintRule, dataset_id: str) -> ConstraintRule`：当 target 与 expression 中**无任何** `_DRIFT_DATASET` 坐标时**返回原对象（`is` 恒等）**——非漂移本地规则走恒等快路径、绝不重建；否则只把 `_DRIFT_DATASET` 的 target/ref 改写为 `dataset_id`，非哨兵坐标原样保留。
- Consumes: `_DRIFT_DATASET`（Task 8）、`ConstraintRule`/`ConstraintNode`/`ParameterReference`/`_iter_references`（model/parameters.py）。
- 关键正确性点：
  - **rebind 必须在 per-dataset 循环内、用循环的 `dataset_id`**：两套结构相同的漂移块在 dsA/dsB 编译出的规则在 rebind 前逐字段相同（都 `__drift__::component.0.repeat.1…`）；循环内 rebind 后各自变 `dsA::…`/`dsB::…`，互不折叠。
  - **恒等快路径保证非漂移路径零改动**：`ConstraintRule` 是 `frozen slots` 值相等类型，`_merged_constraints`（joint_problem.py:201）按值 `set`/`not in` 判等；对无 `__drift__` 的规则返回原对象既不动值等价、也杜绝重建引入的等价漂移，把新建对象限制在漂移一条路径。
  - rebind 后漂移规则 target 与所有 ref 同属成员 id → 仍 dataset-LOCAL（不进 `_cross_dataset_rules`/`_joint_constraint_closure`），随 `local_constraints` 经 `replace(problem, constraint_rules=…)` 落回该成员；evaluation 只按 `parameter_name` 解析 → 改写 dataset_id 对求值零影响。

- [ ] **Step 1: 写 `rebind_drift_dataset` 失败测试**

```python
# tests/unit/fit/test_drift.py（追加）
from xrr_fitter.fit.drift import rebind_drift_dataset, drift_constraint_rules, _DRIFT_DATASET
from xrr_fitter.model.parameters import (
    ConstraintNode, ConstraintRule, ParameterReference, _iter_references,
)
from tests.support.drift_cases import one_drift_block_structure

def test_rebind_rewrites_sentinel_target_and_refs():
    rule = drift_constraint_rules(one_drift_block_structure())[0]
    assert rule.target.dataset_id == _DRIFT_DATASET             # 前置：哨兵
    bound = rebind_drift_dataset(rule, "sampleA")
    assert bound.target.dataset_id == "sampleA"
    assert all(ref.dataset_id == "sampleA" for ref in _iter_references(bound.expression))

def test_rebind_returns_same_object_for_local_rule():
    # 真实本地规则（无 __drift__）→ 恒等返回，绝不重建
    rule = ConstraintRule(
        target=ParameterReference("sampleA", "component.0.layer.0.thickness_a"),
        expression=ConstraintNode(
            "ref", reference=ParameterReference("sampleA", "component.0.layer.1.thickness_a")
        ),
    )
    assert rebind_drift_dataset(rule, "sampleA") is rule
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -k "rebind" -v --import-mode=importlib`
Expected: FAIL（`rebind_drift_dataset` 不存在）。

- [ ] **Step 3: 实现 `rebind_drift_dataset`（追加到 fit/drift.py）**

```python
# 既有 import 行合并追加 _iter_references：
# from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference, _iter_references


def _rebind_node(node: ConstraintNode, dataset_id: str) -> ConstraintNode:
    if node.op == "ref":
        if node.reference.dataset_id != _DRIFT_DATASET:
            return node
        return ConstraintNode(
            "ref", reference=ParameterReference(dataset_id, node.reference.parameter_name)
        )
    if node.op == "const":
        return node
    return ConstraintNode(node.op, operands=tuple(_rebind_node(o, dataset_id) for o in node.operands))


def rebind_drift_dataset(rule: ConstraintRule, dataset_id: str) -> ConstraintRule:
    """Rewrite a compiled drift rule's ``__drift__`` sentinel to a real member id.

    Returns the SAME object untouched when neither the target nor any reference
    uses ``_DRIFT_DATASET`` — non-drift local rules take an identity-preserving
    fast path; only drift rules are reconstructed.
    """
    touches_target = rule.target.dataset_id == _DRIFT_DATASET
    touches_ref = any(
        reference.dataset_id == _DRIFT_DATASET for reference in _iter_references(rule.expression)
    )
    if not touches_target and not touches_ref:
        return rule
    target = (
        ParameterReference(dataset_id, rule.target.parameter_name)
        if touches_target
        else rule.target
    )
    return ConstraintRule(target=target, expression=_rebind_node(rule.expression, dataset_id))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_drift.py -k "rebind" -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: 写联合编译回归测试（先失败）**

```python
# tests/unit/fit/test_joint_problem.py（追加）
from xrr_fitter.fit.drift import _DRIFT_DATASET
from tests.support.drift_cases import drift_case

def test_joint_compile_accepts_drifted_members():
    # 回归：漂移块编入的 __drift__ 本地约束进入 compile_joint_problem 时，
    # 必须先被 rebind 成成员 id，否则 _compiled_local_constraints 身份门直接抛 ValueError。
    api = import_module("xrr_fitter.fit.joint_problem")   # 与本文件既有 idiom 一致：动态取 joint API
    pa = compile_fit_problem(*drift_case())
    pb = compile_fit_problem(*drift_case())
    joint = api.compile_joint_problem(("sampleA", "sampleB"), (pa, pb), (), ())
    for dataset_id, problem in zip(joint.dataset_ids, joint.problems, strict=True):
        for rule in problem.constraint_rules:
            assert rule.target.dataset_id == dataset_id
            assert rule.target.dataset_id != _DRIFT_DATASET
```

- [ ] **Step 6: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/fit/test_joint_problem.py -k "drifted_members" -v --import-mode=importlib`
Expected: FAIL，报 `ValueError: compiled local constraint dataset identity does not match the joint member: sampleA`（哨兵 `__drift__` 未归一，撞身份门）。

- [ ] **Step 7: 接线 `_compiled_local_constraints`（joint_problem.py），跑测试确认通过并提交**

```python
# joint_problem.py 顶部 import 追加：
# from xrr_fitter.fit.drift import rebind_drift_dataset

# _compiled_local_constraints 的 per-dataset 循环体改为（rebind 必须在循环内，
# 使两份逐位相同的 __drift__ 规则分别锚到各自 dataset_id、不在集合里塌缩）：
    for dataset_id, problem in zip(dataset_ids, problems, strict=True):
        for rule in problem.constraint_rules:
            rule = rebind_drift_dataset(rule, dataset_id)   # __drift__ → 成员 id（须在循环内）
            references = tuple(_iter_references(rule.expression))
            if rule.target.dataset_id != dataset_id or any(
                reference.dataset_id != dataset_id for reference in references
            ):
                raise ValueError(
                    "compiled local constraint dataset identity does not match "
                    f"the joint member: {dataset_id}"
                )
            rules.append(rule)
```

Run: `.venv/bin/python -m pytest tests/unit/fit/test_joint_problem.py tests/unit/fit/test_drift.py --import-mode=importlib`
Expected: PASS。

```bash
git add src/xrr_fitter/fit/drift.py src/xrr_fitter/fit/joint_problem.py \
        tests/unit/fit/test_drift.py tests/unit/fit/test_joint_problem.py
git commit -m "fix(fit): rebind __drift__ sentinel to member id in joint compilation"
```

---
### Task 9: 物理逐副本展开（`_ExpandedDriftBlock`，不发 PeriodicSpan）

**Files:**
- Modify: `src/xrr_fitter/model/structure.py`（新增 `_ExpandedDriftBlock`；`_structure_components` repo:218-223 的 `allowed` 元组加入它）
- Modify: `src/xrr_fitter/physics/stack.py`（`_replace_component` repo:132-141 加漂移分支→`_expand_drift`；`_append_component` repo:249-255 加分支→`_append_drift_block`；从 structure 导入 `_ExpandedDriftBlock`）
- Test: `tests/unit/physics/test_stack_expansion.py`

**Interfaces:**
- Produces:
  - `_ExpandedDriftBlock`（frozen slots，**仅 ephemeral，绝不 persist/codec/checkpoint**）：`layers: tuple[LayerSpec, ...]`（扁平化、copy-major、共 `repeats*len(base.layers)` 项）、`layer_count: int`（=len(base.layers)，供 geometry 从扁平位置还原 (k,i)）、`top_roughness_a: float|None`、`target: str`（"thickness"|"roughness"）。
  - `_expand_drift(block, prefix, values) -> _ExpandedDriftBlock`：copy 0 全取 base 名值；copy k≥1 目标族取 `{prefix}.repeat.{k}.layer.{i}.<family>`、非目标族取 base（Task 7/8 只对目标族发 `.repeat.` 名与规则）；density/material 恒取 base。
  - `_append_drift_block(state, block)`：逐副本 `_append_layer`，top override 仅落扁平首层 `position==0`；**故意不 append `PeriodicSpan`**（逐副本值不等→无精确重复→derivatives.py:360/371 矩阵幂快路径不得命中）。
- Consumes: `_replace_material`/`_replace_layer`/`_append_layer`（既有）、`PeriodicBlock.drift`（Task 4）。
- 命名不移位：漂移块占 1 个组件槽（`_ExpandedDriftBlock` 替换该位 `PeriodicBlock`），下游 `component.{index}` 不变；`rebuild_structure` 的 `replace(structure, components=...)` 会过 `_structure_components` 校验 → 故必须把它加入 `allowed`。**codec/checkpoint 对它无分支即 fail-fast**，保证「绝不持久化」。

- [ ] **Step 1: 在 structure.py 新增 `_ExpandedDriftBlock` 并放开校验（写失败测试）**

```python
# tests/unit/physics/test_stack_expansion.py（追加）
from xrr_fitter.model.structure import _ExpandedDriftBlock, StructureSpec
from xrr_fitter.physics.stack import rebuild_structure, expand_structure
from tests.support.drift_cases import drift_block, media, drift_structure, drift_values, wavelength

def test_expanded_drift_block_allowed_in_structure():
    # 直接把 ephemeral 块放进 components 应通过校验（rebuild 内部即如此）
    block = drift_block()
    media_spec = media()
    exp = _ExpandedDriftBlock(layers=block.layers, layer_count=len(block.layers),
                              top_roughness_a=None, target="thickness")
    StructureSpec(fronting=media_spec, components=(exp,), backing=media_spec)  # 不抛
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/physics/test_stack_expansion.py -k "allowed_in_structure" -v --import-mode=importlib`
Expected: FAIL（`_ExpandedDriftBlock` 不存在 / TypeError unsupported value）。

- [ ] **Step 3: 实现 structure.py 侧**

在 `PeriodicBlock` 定义后、`StructureComponent` 之前插入：
```python
@dataclass(frozen=True, slots=True)
class _ExpandedDriftBlock:
    """Ephemeral per-copy expansion of a drifted block. NEVER persisted/codec/checkpoint."""

    layers: tuple[LayerSpec, ...]
    layer_count: int
    top_roughness_a: float | None
    target: str
```
`_structure_components`（repo:220）`allowed` 改为：
```python
    allowed = (LayerSpec, PeriodicBlock, GradientLayerSpec, _ExpandedDriftBlock)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/physics/test_stack_expansion.py -k "allowed_in_structure" -v --import-mode=importlib`
Expected: PASS。

- [ ] **Step 5: 写逐副本展开失败测试**

```python
# tests/unit/physics/test_stack_expansion.py（追加）
def test_drift_expands_percopy_thickness_no_span():
    # drift_structure(): 单组件 PeriodicBlock(2 层, repeats=3, drift thickness amount=0.1 linear)
    structure = drift_structure()
    values = drift_values(structure)          # 手工构造含 .repeat. 的 values dict
    rebuilt = rebuild_structure(structure, values)
    stack = expand_structure(rebuilt, wavelength())
    assert stack.periodic_spans == ()                     # 关键：不发 span
    # 3 副本 × 2 层 = 6 有限层 → thickness 数组含 6 个内层 + 首尾 0
    body = stack.thickness_a[1:-1]
    assert body.size == 6
    # copy k 层 0 厚度 = base*(1+0.1*k)（drift_values 已按此填入 .repeat. 名）
```

- [ ] **Step 6: 跑测试确认失败 → 实现 → 通过**

stack.py 顶部导入加 `_ExpandedDriftBlock`。`_replace_component`（repo:132-141）改：
```python
    if isinstance(component, PeriodicBlock):
        if component.drift is not None:
            return _expand_drift(component, prefix, values)
        return _replace_periodic(component, prefix, values)
```
新增 `_expand_drift`（放在 `_replace_periodic` 之后）：
```python
def _expand_drift(block, prefix, values):
    target = block.drift.target
    flat = []
    for k in range(block.repeats):
        for index, layer in enumerate(block.layers):
            base = f"{prefix}.layer.{index}"
            repeat = f"{prefix}.repeat.{k}.layer.{index}"
            thickness = values[f"{repeat}.thickness_a"] if k and target == "thickness" else values[f"{base}.thickness_a"]
            roughness = values[f"{repeat}.roughness_a"] if k and target == "roughness" else values[f"{base}.roughness_a"]
            flat.append(replace(
                layer,
                material=_replace_material(layer.material, base, values),
                thickness_a=thickness,
                density_scale=values[f"{base}.density_scale"],
                roughness_a=roughness,
            ))
    top = None if block.top_roughness_a is None else values[f"{prefix}.top_roughness_a"]
    return _ExpandedDriftBlock(tuple(flat), len(block.layers), top, target)
```
`_append_component`（repo:249-255）加分支 + 新 `_append_drift_block`：
```python
    elif isinstance(component, _ExpandedDriftBlock):
        _append_drift_block(state, component)
```
```python
def _append_drift_block(state: _Expansion, block: _ExpandedDriftBlock) -> None:
    for position, layer in enumerate(block.layers):
        _append_layer(state, layer, block.top_roughness_a if position == 0 else None)
    # No PeriodicSpan: per-copy values differ, so the matrix-power fast path must not apply.
```
Run: `.venv/bin/python -m pytest tests/unit/physics/test_stack_expansion.py -k "drift" -v --import-mode=importlib` → 先 FAIL 后 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/xrr_fitter/model/structure.py src/xrr_fitter/physics/stack.py tests/unit/physics/test_stack_expansion.py
git commit -m "feat(physics): expand drifted blocks per-copy without a PeriodicSpan"
```

---
### Task 10: 几何逐副本切向（Jacobian）+ 快路径丧失核验

**Files:**
- Modify: `src/xrr_fitter/physics/geometry.py`（`append_component` repo:469-508 dispatch 加 `_ExpandedDriftBlock` 分支→新 `append_drift_block`；顶部导入 `_ExpandedDriftBlock`；repo:471 类型注解补该类型）
- Test: `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes（既有，精确签名）：`append_layer(layer, prefix, roughness_name, values, value_jacobians, wavelength_a)`；`_layer_sld_jacobian(layer, prefix, values, value_jacobians, wavelength_a)`；`self.sld_cache`（按 prefix 缓存复用）；`expand_structure_with_jacobian(structure, values, value_jacobians, wavelength_a, parameter_count)` repo:544-567 迭代 `rebuilt.components` 用 `f"component.{index}"`；`builder.finish` repo:510-541 的轴数对齐断言（每根 stack 轴恰好加 1 条自由参数轴，否则 `RuntimeError`）。
- Produces：`append_drift_block(block, prefix, values, value_jacobians, wavelength_a)`——从 `_ExpandedDriftBlock` 还原 `repeats = len(block.layers)//block.layer_count`，逐 (k,i) 取切向名：厚度 `k and target=="thickness"` → `{prefix}.repeat.{k}.layer.{i}.thickness_a` 否则 base；粗糙度 `k==0 and i==0 and top` → `{prefix}.top_roughness_a`，`k and target=="roughness"` → repeat 名，否则 base；SLD 用 copy-0 层 `block.layers[i]` + base prefix（material/density 副本不变）经 `sld_cache` 复用。
- **快路径丧失**：漂移栈无 `PeriodicSpan`（Task 9）→ derivatives.py 的 `if stack.periodic_spans:` 矩阵幂分支（repo:352/413/436，`_apply_span_tangent` repo:336-345 用 `_power(...,repeats-1)`）不命中，逐 slab 切向路径生效。核验用「漂移展开 == 手工逐层展开」等价断言坐实。

- [ ] **Step 1: 写等价失败测试（漂移 Jacobian == 手工逐层 Jacobian）**

```python
# tests/unit/test_evaluation.py（追加）
import numpy as np
from xrr_fitter.fit.drift import drift_coefficients
from xrr_fitter.model.structure import LayerSpec, StructureSpec
from xrr_fitter.physics.geometry import expand_structure_with_jacobian
from tests.support.drift_cases import drift_structure, drift_values, wavelength


def _drift_jacobian_case():
    """构造「漂移展开」与「手工逐层展开」两套对齐输入（唯一自由参数 = drift_scale）。

    等价关系按构造成立：厚度切向 = base·c_k、粗糙度/SLD 切向恒 0，两条路径逐槽同序。
    """
    structure = drift_structure()                 # 单组件漂移块（pre-rebuild；内部会 rebuild）
    block = structure.components[0]
    drift = block.drift
    coeffs = drift_coefficients(drift, block.repeats)   # c_0=0, linear c_k=k
    base_layers = block.layers
    param_count = 1

    # 漂移路径：values 复用共享构造器（与 Task 9 同源），value_jacobians 单独按读取名对齐
    drift_values_d = drift_values(structure)
    drift_jacobians_d = {"backing.roughness_a": np.array([0.0])}
    for i, layer in enumerate(base_layers):
        base = f"component.0.layer.{i}"
        drift_jacobians_d[f"{base}.thickness_a"] = np.array([0.0])     # 基副本厚度与 scale 无关
        drift_jacobians_d[f"{base}.roughness_a"] = np.array([0.0])
        drift_jacobians_d[f"{base}.density_scale"] = np.array([0.0])   # thickness 漂移 → SLD 切向恒 0
        for k in range(1, block.repeats):
            repeat = f"component.0.repeat.{k}.layer.{i}"
            drift_jacobians_d[f"{repeat}.thickness_a"] = np.array([layer.thickness_a * coeffs[k]])
    # 手工路径：R×L 独立 LayerSpec（copy-major：k0i0,k0i1,k1i0,...），无 PeriodicBlock/DriftSpec
    plain_layers = []
    plain_values_d = {}
    plain_jacobians_d = {"backing.roughness_a": np.array([0.0])}
    for k in range(block.repeats):
        for i, layer in enumerate(base_layers):
            index = len(plain_layers)
            prefix = f"component.{index}"
            thickness = layer.thickness_a * (1.0 + drift.amount * coeffs[k])
            plain_layers.append(
                LayerSpec(f"copy{k}_layer{i}", layer.material, thickness,
                          roughness_a=layer.roughness_a, density_scale=layer.density_scale)
            )
            plain_values_d[f"{prefix}.thickness_a"] = thickness
            plain_values_d[f"{prefix}.roughness_a"] = layer.roughness_a
            plain_values_d[f"{prefix}.density_scale"] = layer.density_scale
            plain_jacobians_d[f"{prefix}.thickness_a"] = np.array([layer.thickness_a * coeffs[k]])
            plain_jacobians_d[f"{prefix}.roughness_a"] = np.array([0.0])
            plain_jacobians_d[f"{prefix}.density_scale"] = np.array([0.0])
    plain_structure = StructureSpec(
        fronting=structure.fronting, components=tuple(plain_layers),
        backing=structure.backing, backing_roughness_a=structure.backing_roughness_a,
    )
    return (structure, plain_structure, drift_values_d, drift_jacobians_d,
            plain_values_d, plain_jacobians_d, wavelength(), param_count)


def test_drift_jacobian_equals_hand_expanded():
    (drift_structure_exp, plain_structure_exp, drift_values_d, drift_jacobians_d,
     plain_values_d, plain_jacobians_d, wavelength_a, param_count) = _drift_jacobian_case()
    drifted = expand_structure_with_jacobian(
        drift_structure_exp, drift_values_d, drift_jacobians_d, wavelength_a, param_count)
    plain = expand_structure_with_jacobian(
        plain_structure_exp, plain_values_d, plain_jacobians_d, wavelength_a, param_count)
    assert drifted.stack.periodic_spans == ()          # 快路径已丧失（漂移栈不发 span）
    np.testing.assert_allclose(drifted.thickness_jacobian, plain.thickness_jacobian)
    np.testing.assert_allclose(drifted.sld_jacobian, plain.sld_jacobian)
    np.testing.assert_allclose(drifted.roughness_jacobian, plain.roughness_jacobian)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/unit/test_evaluation.py -k "drift_jacobian" -v --import-mode=importlib`
Expected: FAIL（`append_component` 落入 `else: append_gradient` 或 `_ExpandedDriftBlock` 无分支 → KeyError/AttributeError）。

- [ ] **Step 3: 实现 `append_drift_block` 并接线 dispatch**

`append_component`（repo:507）`else` 之前插入：
```python
        elif isinstance(component, _ExpandedDriftBlock):
            self.append_drift_block(component, prefix, values, value_jacobians, wavelength_a)
```
新增方法（放在 `append_periodic` 之后）：
```python
    def append_drift_block(
        self,
        block: _ExpandedDriftBlock,
        prefix: str,
        values: dict[str, float],
        value_jacobians: dict[str, np.ndarray],
        wavelength_a: float,
    ) -> None:
        """Per-copy tangents for a drifted block: no span, no matrix-power path."""
        repeats = len(block.layers) // block.layer_count
        for repeat_index in range(repeats):
            for layer_index in range(block.layer_count):
                base = f"{prefix}.layer.{layer_index}"
                repeat = f"{prefix}.repeat.{repeat_index}.layer.{layer_index}"
                thickness_name = (
                    f"{repeat}.thickness_a" if repeat_index and block.target == "thickness" else f"{base}.thickness_a"
                )
                if repeat_index == 0 and layer_index == 0 and block.top_roughness_a is not None:
                    roughness_name = f"{prefix}.top_roughness_a"
                elif repeat_index and block.target == "roughness":
                    roughness_name = f"{repeat}.roughness_a"
                else:
                    roughness_name = f"{base}.roughness_a"
                self.thickness.append(np.asarray(value_jacobians[thickness_name], dtype=float))
                if base not in self.sld_cache:
                    self.sld_cache[base] = np.asarray(
                        _layer_sld_jacobian(block.layers[layer_index], base, values, value_jacobians, wavelength_a),
                        dtype=np.complex128,
                    )
                self.sld.append(self.sld_cache[base])
                self.roughness.append(np.asarray(value_jacobians[roughness_name], dtype=float))
```
顶部 import 加 `_ExpandedDriftBlock`；`append_component` 形参注解补 `| _ExpandedDriftBlock`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/unit/test_evaluation.py -k "drift_jacobian" -v --import-mode=importlib`
Expected: PASS（含 `finish` 轴对齐断言不抛）。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/physics/geometry.py tests/unit/test_evaluation.py
git commit -m "feat(physics): per-copy Jacobian tangents for drifted blocks"
```

---
### Task 11: 公共 API 重导出 `DriftSpec`

**Files:**
- Modify: `src/xrr_fitter/api.py`（`from xrr_fitter.model.structure import (...)` repo:62-71 加 `DriftSpec`；`__all__` repo:160 按字母序加 `"DriftSpec"`）
- Test: `tests/architecture/test_public_api.py`（`PUBLIC_NAMES` repo:9 加 `"DriftSpec"`）

**Interfaces:**
- Consumes：`DriftSpec`（Task 4，`model/structure.py`）。
- Produces：`xrr_fitter.api.DriftSpec` 可导入；`tuple(api.__all__) == PUBLIC_NAMES`（test repo:215）维持逐名相等。**不进 `GUI_USE_CASES`**——它只映射操作函数（test repo:223 `inspect.isfunction`），`DriftSpec` 是数据类型，随 `set_structure` 走既有 structure 用例，不新增 operation（对应「对 spec 的修正」第 5 条：不新增 api 对）。

- [ ] **Step 1: 先给测试加名（RED）**

`PUBLIC_NAMES`（repo:21，`"DatasetProject"` 与 `"ExportManifest"` 之间）插入：
```python
    "DriftSpec",
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/architecture/test_public_api.py -k "exports" -v --import-mode=importlib`
Expected: FAIL（`tuple(api.__all__) == PUBLIC_NAMES` 不等 / `hasattr(api,"DriftSpec")` 假）。

- [ ] **Step 3: api.py 重导出**

`from xrr_fitter.model.structure import (` 块（repo:62）首行加 `DriftSpec,`（字母序在 `GradientLayerSpec` 前）：
```python
from xrr_fitter.model.structure import (
    DriftSpec,
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    OxideSuggestion,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
)
```
`__all__`（repo:160）在 `"DatasetProject",` 与 `"ExportManifest",` 之间插入 `"DriftSpec",`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/architecture/test_public_api.py -v --import-mode=importlib`
Expected: PASS（全量，含 `__all__` 逐名相等、id 唯一、签名快照）。

- [ ] **Step 5: Commit**

```bash
git add src/xrr_fitter/api.py tests/architecture/test_public_api.py
git commit -m "feat(api): re-export DriftSpec on the public surface"
```

---
### Task 12: GUI 周期块漂移分区 + 实时性能警告 + 结构树 tooltip

**Files:**
- Modify: `src/xrr_fitter/gui/structure/dialogs.py`（`PeriodicDialog` repo:233-315 加漂移控件；imports 加 `QComboBox`）
- Modify: `src/xrr_fitter/gui/structure/editor.py`（`_component_item` repo:244-256 给漂移块设 tooltip；新增模块级 `_drift_text`）
- Test: `tests/gui/test_structure_editor.py`

**Interfaces:**
- Consumes：`api.DriftSpec`（Task 11 再导出）；`api.PeriodicBlock(name, layers, repeats, top_roughness_a=None, drift=None)`（Task 4）；既有 `_number(name, minimum, value)` repo:25、`_buttons` repo:34、`qtbot`/`findChild(Type, objectName)` GUI 测试范式（repo tests/gui/test_structure_editor.py:313-332）。
- Produces（objectName 固定，供测试与无障碍）：`driftKindInput`(QComboBox: 无=""/线性="linear"/正弦="sine"/随机="random")、`driftTargetInput`(QComboBox: 厚度="thickness"/粗糙度="roughness")、`driftAmountInput`(QDoubleSpinBox，min=-1.0)、`driftPeriodInput`(仅 sine 可见)、`driftPhaseInput`(仅 sine 可见)、`driftSeedInput`(QSpinBox，仅 random 可见，**可见可编辑**)、`driftPerformanceWarning`(QLabel，默认隐藏)。**无 randomize 按钮**（种子只手填，对 spec 修正：确定性自包含于 `DriftSpec.seed`）。
- 约束：结构树**不加列**（tooltip 挂在既有「周期」类型格 column 1）；kind="" 时 `PeriodicBlock(drift=None)`，与无漂移逐字节一致；sine 缺 period / random 负 seed 由 `DriftSpec.__post_init__` 抛 `ValueError` → 复用既有 `except (TypeError, ValueError)` 错误标签路径。

- [ ] **Step 1: 写失败测试（对话框构造 DriftSpec + 种子可编辑无按钮 + 性能警告 + 树 tooltip）**

```python
# tests/gui/test_structure_editor.py（追加）
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel

def _set_two_layers(dialog):
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    for row, fields in enumerate((("Mo","Mo","10.28","2.5","0.2"), ("Si","Si","2.329","4","0.3"))):
        for column, value in enumerate(fields):
            table.setItem(row, column, QTableWidgetItem(value))

def test_periodic_dialog_builds_sine_drift_and_seed_is_editable_without_button(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog
    dialog = PeriodicDialog(); qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "periodicNameInput").setText("Mo/Si")
    _set_two_layers(dialog)
    kind = dialog.findChild(QComboBox, "driftKindInput")
    kind.setCurrentIndex(kind.findData("sine"))
    target = dialog.findChild(QComboBox, "driftTargetInput")
    target.setCurrentIndex(target.findData("thickness"))
    dialog.findChild(QDoubleSpinBox, "driftAmountInput").setValue(0.1)
    dialog.findChild(QDoubleSpinBox, "driftPeriodInput").setValue(4.0)
    dialog.findChild(QDoubleSpinBox, "driftPhaseInput").setValue(0.5)
    seed = dialog.findChild(QSpinBox, "driftSeedInput")
    assert seed is not None and seed.isEnabled()                       # 可见可编辑
    assert dialog.findChild(QPushButton, "driftRandomizeButton") is None  # 无随机按钮
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.block().drift == api.DriftSpec(kind="sine", target="thickness", amount=0.1, period=4.0, phase=0.5)

def test_periodic_dialog_no_drift_yields_none(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog
    dialog = PeriodicDialog(); qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "periodicNameInput").setText("plain")
    _set_two_layers(dialog)
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)
    assert dialog.block().drift is None

def test_periodic_dialog_warns_whenever_drift_active(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog
    dialog = PeriodicDialog(); qtbot.addWidget(dialog)
    warning = dialog.findChild(QLabel, "driftPerformanceWarning")
    assert warning.isHidden()                                          # 默认隐藏
    kind = dialog.findChild(QComboBox, "driftKindInput")
    kind.setCurrentIndex(kind.findData("linear"))
    # 关键：即便重复数很小(丧失快路径与副本数无关)，只要漂移激活就必须提示
    dialog.findChild(QSpinBox, "periodicRepeatsInput").setValue(2)
    assert not warning.isHidden()                                      # 漂移激活→提示(与重复数无关)
    kind.setCurrentIndex(kind.findData(""))
    assert warning.isHidden()                                          # 关闭漂移→隐藏

def test_structure_tree_shows_drift_tooltip_without_new_column(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.editor import StructureEditor  # 经 _panel 装配
    panel = _panel(qtbot, tmp_path); panel.set_structure(_bare())
    layer = api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0, roughness_a=2.0)
    block = api.PeriodicBlock("Mo/Si", (layer, layer), 3,
                              drift=api.DriftSpec(kind="linear", target="thickness", amount=0.1))
    panel.add_periodic_block(block)
    tree = panel.findChild(QTreeWidget, "structureTree")
    row = tree.topLevelItem(tree.topLevelItemCount() - 1)
    assert tree.columnCount() == _bare_column_count(tree)              # 不新增列
    assert "漂移" in row.toolTip(1) and "线性" in row.toolTip(1)
```
（`_panel`/`_bare` 沿用文件内既有夹具；`_bare_column_count` = 载入前记录的 `tree.columnCount()`，断言载入漂移块后列数不变。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/gui/test_structure_editor.py -k "drift" -v --import-mode=importlib`
Expected: FAIL（`driftKindInput` 等控件不存在 → `findChild` 返回 None）。

- [ ] **Step 3a: 实现 — `dialogs.py` 漂移分区**

导入行（repo:7-20）加 `QComboBox`；`_number` 之上加组合框工厂：

```python
def _drift_combo(name: str, options: tuple[tuple[str, str], ...]) -> QComboBox:
    combo = QComboBox()
    combo.setObjectName(name)
    for label, data in options:
        combo.addItem(label, data)
    return combo
```

`PeriodicDialog.__init__`：在 `self.buttons = _buttons(...)` 之前插入漂移控件，并在 `self._arrange()` 之后追加 `self._sync_drift()`：

```python
        self.drift_kind = _drift_combo(
            "driftKindInput",
            (("无", ""), ("线性", "linear"), ("正弦", "sine"), ("随机", "random")),
        )
        self.drift_target = _drift_combo(
            "driftTargetInput", (("厚度", "thickness"), ("粗糙度", "roughness")),
        )
        self.drift_amount = _number("driftAmountInput", -1.0, 0.1)
        self.drift_period = _number("driftPeriodInput", 0.0, 4.0)
        self.drift_phase = _number("driftPhaseInput", -1_000_000.0, 0.0)
        self.drift_seed = QSpinBox()
        self.drift_seed.setObjectName("driftSeedInput")
        self.drift_seed.setRange(0, 2_147_483_647)
        self.drift_warning = QLabel("漂移块逐副本展开，将丧失矩阵幂快路径；重复数越大越慢。")
        self.drift_warning.setObjectName("driftPerformanceWarning")
        self.drift_warning.setWordWrap(True)
        self.drift_warning.hide()
        self.drift_kind.currentIndexChanged.connect(self._sync_drift)
```

`_arrange` 追加漂移行并存 `self._form`；把 `drift_warning` 加到 table 与 error 之间：

```python
        form.addRow("漂移类型", self.drift_kind)
        form.addRow("漂移目标", self.drift_target)
        form.addRow("漂移幅度", self.drift_amount)
        form.addRow("正弦周期", self.drift_period)
        form.addRow("正弦相位", self.drift_phase)
        form.addRow("随机种子", self.drift_seed)
        self._form = form
        ...
        layout.addWidget(self.table)
        layout.addWidget(self.drift_warning)
        layout.addWidget(self.error_label)
```

新增 `_sync_drift` + `_drift_spec`，并在 `_accept_fields` 的 `api.PeriodicBlock(...)` 调用加 `drift=self._drift_spec()`：

```python
    def _sync_drift(self) -> None:
        kind = self.drift_kind.currentData()
        active = bool(kind)
        self.drift_target.setEnabled(active)
        self.drift_amount.setEnabled(active)
        self._form.setRowVisible(self.drift_period, kind == "sine")
        self._form.setRowVisible(self.drift_phase, kind == "sine")
        self._form.setRowVisible(self.drift_seed, kind == "random")
        self.drift_warning.setVisible(active)   # 丧失快路径与副本数无关，漂移一激活即提示

    def _drift_spec(self) -> api.DriftSpec | None:
        kind = self.drift_kind.currentData()
        if not kind:
            return None
        return api.DriftSpec(
            kind=kind,
            target=self.drift_target.currentData(),
            amount=self.drift_amount.value(),
            period=self.drift_period.value(),
            phase=self.drift_phase.value(),
            seed=self.drift_seed.value(),
        )
```

- [ ] **Step 3b: 实现 — `editor.py` 结构树 tooltip（不加列）**

在模块级 helper 区（repo:27-46，`_transition_text` 附近）加标签映射与 `_drift_text`：

```python
_DRIFT_KIND_LABELS = {"linear": "线性", "sine": "正弦", "random": "随机"}
_DRIFT_TARGET_LABELS = {"thickness": "厚度", "roughness": "粗糙度"}


def _drift_text(drift: api.DriftSpec) -> str:
    kind = _DRIFT_KIND_LABELS.get(drift.kind, drift.kind)
    target = _DRIFT_TARGET_LABELS.get(drift.target, drift.target)
    return f"漂移：{kind}·{target} 幅度 {drift.amount:g}"
```

`_component_item` 的 `PeriodicBlock` 分支（repo:245-248），在 `for layer ...` 之后、`elif` 之前挂 tooltip 到「周期」类型格（column 1，与 `_layer_item` 的 `setToolTip(4, ...)` 同范式）：

```python
        if isinstance(component, api.PeriodicBlock):
            item = QTreeWidgetItem((component.name, "周期", "", "", "", str(component.repeats)))
            for layer in component.layers:
                item.addChild(self._layer_item(layer))
            if component.drift is not None:
                item.setToolTip(1, _drift_text(component.drift))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/gui/test_structure_editor.py -k "drift" -v --import-mode=importlib`
Expected: PASS（4 条 drift 测试全绿；种子可编辑无按钮、无漂移→None、性能警告随漂移激活而非重复数、tooltip 不加列）。

- [ ] **Step 5: 提交**

```bash
git add src/xrr_fitter/gui/structure/dialogs.py src/xrr_fitter/gui/structure/editor.py tests/gui/test_structure_editor.py
git commit -m "feat(gui): periodic drift controls with editable seed and slow-repeat warning"
```

---

### Task 13: 验收回归（无漂移逐位不变 · 漂移↔手工展开同一答案 · 粗糙度超限报错）

**Files:**
- Test: `tests/acceptance/test_stack_drift.py`（新建）

**Interfaces:**
- Consumes：`rebuild_structure(structure, values)` repo physics/stack.py:152、`expand_structure(structure, wavelength_a)` repo:276、`parratt_reflectivity(qz_a_inv, stack)` repo physics/parratt.py:204、`SlabStack.periodic_spans`（model/structure.py:433）、`PhysicalValueError`（model/parameters.py）、`DriftSpec`（Task 4）、Task 9 的 `_expand_drift`/`_ExpandedDriftBlock` 与 Task 10 已建立的逐副本命名。
- Produces：无（终末验收 gate；断言分别定位到 Task 5 / Task 9 / Task 10 的行为）。
- 说明：本任务校验的行为在 Task 5/9/10 已实现，故按「写测试 → 跑 → 绿」节奏；任一断言失败即定位到对应上游任务的缺陷（不是本任务新写实现）。spec 其余验证项映射：确定性(line 100)→Task 7；线性展开求值(line 98)→Task 8；`sin/cos` 解析导数(line 99)→Task 2；GUI(line 104-106)→Task 12。

- [ ] **Step 1: 写验收测试**

```python
# tests/acceptance/test_stack_drift.py
from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import (
    DriftSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.stack import expand_structure, rebuild_structure

_VAC = MaterialSpec("vacuum", None, None, sld_override_a2=0j)
_MO = MaterialSpec("Mo", "Mo", 10.28)
_SI = MaterialSpec("Si", "Si", 2.329)
_WL = 1.5406
_QZ = np.linspace(0.01, 0.30, 64)


def test_non_drift_periodic_keeps_matrix_power_fast_path() -> None:
    """无漂移块仍登记 PeriodicSpan（逐位不变的物理侧代理，spec line 103）。"""
    block = PeriodicBlock("mirror", (LayerSpec("Mo", _MO, 50.0, roughness_a=3.0),), 3)
    stack = expand_structure(StructureSpec(_VAC, (block,), _SI, backing_roughness_a=2.0), _WL)
    assert block.drift is None
    assert len(stack.periodic_spans) == 1 and stack.periodic_spans[0].repeats == 3

def test_drift_thickness_expansion_equals_hand_written_layers() -> None:
    """线性厚度漂移(amount=0.1) 逐副本厚度 50/55/60，与手工三层结构逐位一致，
    且 periodic_spans 为空——慢路径与手工展开必须给同一答案（spec line 101-102）。"""
    drifted = PeriodicBlock(
        "grade",
        (LayerSpec("Mo", _MO, 50.0, roughness_a=3.0),),
        3,
        drift=DriftSpec(kind="linear", target="thickness", amount=0.1),
    )
    values = {
        "component.0.layer.0.thickness_a": 50.0,
        "component.0.layer.0.density_scale": 1.0,
        "component.0.layer.0.roughness_a": 3.0,
        "component.0.repeat.1.layer.0.thickness_a": 55.0,
        "component.0.repeat.2.layer.0.thickness_a": 60.0,
        "backing.roughness_a": 2.0,
    }
    rebuilt = rebuild_structure(StructureSpec(_VAC, (drifted,), _SI, backing_roughness_a=2.0), values)
    drift_stack = expand_structure(rebuilt, _WL)

    hand = StructureSpec(
        _VAC,
        tuple(
            LayerSpec("Mo", _MO, thickness, roughness_a=3.0)
            for thickness in (50.0, 55.0, 60.0)
        ),
        _SI,
        backing_roughness_a=2.0,
    )
    hand_stack = expand_structure(hand, _WL)

    assert drift_stack.periodic_spans == ()
    assert np.array_equal(drift_stack.thickness_a, hand_stack.thickness_a)
    assert np.array_equal(drift_stack.roughness_a, hand_stack.roughness_a)
    assert np.array_equal(drift_stack.sld_a2, hand_stack.sld_a2)
    assert np.array_equal(
        parratt_reflectivity(_QZ, drift_stack),
        parratt_reflectivity(_QZ, hand_stack),
    )


def test_roughness_drift_over_neighbor_limit_raises_physical_value_error() -> None:
    """粗糙度目标漂移把某副本推过 _validate_roughness 邻层上限时报错（spec line 90-91）。"""
    drifted = PeriodicBlock(
        "grade",
        (LayerSpec("Mo", _MO, 50.0, roughness_a=20.0),),
        3,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.5),
    )
    values = {
        "component.0.layer.0.thickness_a": 50.0,
        "component.0.layer.0.density_scale": 1.0,
        "component.0.layer.0.roughness_a": 20.0,
        "component.0.repeat.1.layer.0.roughness_a": 30.0,
        "component.0.repeat.2.layer.0.roughness_a": 40.0,
        "backing.roughness_a": 2.0,
    }
    rebuilt = rebuild_structure(StructureSpec(_VAC, (drifted,), _SI, backing_roughness_a=2.0), values)
    with pytest.raises(PhysicalValueError):
        expand_structure(rebuilt, _WL)
```

- [ ] **Step 2: 跑测试，确认全绿**

Run: `.venv/bin/python -m pytest tests/acceptance/test_stack_drift.py -v --import-mode=importlib`
Expected: 3 passed。若 `test_drift_thickness_expansion_equals_hand_written_layers` 失败 → 定位 Task 9/10 的逐副本命名或 span 抑制；若 `test_roughness_drift_over_neighbor_limit_raises_physical_value_error` 失败 → 定位 Task 9 未复用 `_validate_roughness`；若 `test_non_drift_periodic_keeps_matrix_power_fast_path` 失败 → 回归，非漂移路径被 Task 9 误伤。

- [ ] **Step 3: 提交**

```bash
git add tests/acceptance/test_stack_drift.py
git commit -m "test(acceptance): drift expansion equals hand-written layers and no-drift keeps fast path"
```

---
