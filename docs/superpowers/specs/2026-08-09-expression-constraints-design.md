# 表达式参数约束设计

## 目标

把参数耦合从"只能相等"升级为"可用表达式关联"。这是吸收 Multifitting 唯一真正胜过本
项目的能力（跨结构、跨样品的主从耦合），但采 refnx 的惰性表达式 DAG 实现，明确不采
Multifitting 的 ExprTk 字符串求值。

## 为什么现有机制不能直接扩展

`fit/joint_sharing.py` 的等式共享是**索引投影，不是求值**：

```python
def _raw_scatter(problem: object, unit: np.ndarray) -> list[np.ndarray]:
    local = []
    for scatter in problem.scatter_maps:
        vector = np.array(unit[np.asarray(scatter, dtype=int)], dtype=float, copy=True)
        local.append(vector)
    return local
```

同一个全局坐标被 gather 到多个局部位置，等式约束的导数天然正确（重复的单位向量），零
开销。任何非恒等关系（`t2 = 2*t1`、`rho_ox = 0.85*rho_bulk`）都无法用 gather 表达，必须
走真正的求值加链式法则。

## 为什么不用 ExprTk 路线

1. 运行期字符串解析的导数只能有限差分，会破坏本项目的解析 Jacobian
   （`physics/derivatives.py` 的 `parratt_reflectivity_jacobian`、
   `smear_with_widths_jacobian`、`_periodic_jacobian`）。
2. Multifitting 手册自己承认非物理值可能导致 "an incorrect result or a program crash"。
3. 字符串求值在序列化边界是任意代码执行面，而本项目的工程文件是可交换的。

## 数据类型

`MODEL_ALLOWED["parameters"] = set()` 允许自洽的节点类型，因为它只需 `ParameterReference`
与数值运算：

```python
@dataclass(frozen=True)
class ConstraintNode:
    op: str                                    # "ref" | "const" | "add" | "mul" | "pow" | ...
    reference: ParameterReference | None = None
    value: float | None = None
    operands: tuple["ConstraintNode", ...] = ()

@dataclass(frozen=True)
class ConstraintRule:
    target: ParameterReference
    expression: ConstraintNode
```

**`SharingRule` 保留不动。** 等式共享继续走索引投影快路径——它覆盖绝大多数实际用法且
零开销。`ConstraintRule` 是并列的新机制，只在需要非恒等关系时启用。这个决定避免了 17 个
消费文件的破坏性重写（`services/structures.py`、`services/fitting_phases/{joint_execution,
joint_selection,sharing}.py`、`services/parameters.py`、`model/{project,parameters}.py`、
`api.py`、`gui/parameters/{panel,sharing}.py`、`gui/fitting/progress.py`、
`io/{export_tables,codec_common,project_codec}.py`、`fit/{joint_sharing,joint_pipeline,
joint_problem}.py`），代价是需要一处统一的冲突检查：同一 target 不得同时被 `SharingRule`
与 `ConstraintRule` 驱动。

## 求值挂载点

挂在 `evaluation.py:590 values_and_jacobians()`。它返回
`tuple[dict[str, float], dict[str, np.ndarray]]`，每个声明——**包括 locked 的**——都有一条
与 `problem.variables` 对齐的切向量。约束节点的导数在这里按链式法则合成，**完全不碰**
`physics/derivatives.py` 的 Parratt 与卷积 Jacobian 层。

这是整个设计最关键的发现：解析梯度不需要重写。

## 两阶段解码

求值必须尊重现有顺序。`_decode_nonrough_values` 先解非 roughness 轴，
`transform == "roughness_fraction"` 被推迟到几何算出 `dynamic_upper` 之后：

```python
postponed, decoded = _decode_nonrough_values(problem, unit, values, continuous_only=True)
dynamic_values = _roughness_dynamic_uppers(problem, values)
dynamic_jacobians = _roughness_dynamic_upper_jacobians(problem, values, value_jacobians)
```

因此约束求值也分两趟：第一趟解 target 为非 roughness 的约束，第二趟在动态上界算出后解
roughness 约束。**跨阶段约束在绑定期直接拒绝**（非 roughness 参数依赖 roughness 参数）：
它会造成循环依赖，且没有真实用例。

`_decode_nonrough_values` 对 `(continuous_only, integer) == (True, True)` 抛
`"analytic Jacobian requires continuous free parameters"`。integer 参数代表离散拓扑：作为
自变量没有解析切向量，作为 target 又需要取整并产生不连续前向模型；两种位置都在绑定期拒绝。

## 绑定期检查

全部照 refnx：

- `id()` 基的环检测，**绑定时拒绝**而非求值时爆。
- target 的 free 标志**自动清除**，不留"既被约束又被优化"的矛盾状态。
- 约束结果越界时明确报错，**不静默 clip**。静默 clip 会让优化器看到平坦区域并误判收敛。

## 代码边界

- 改 `src/xrr_fitter/model/parameters.py`：新增两个类型。
- 改 `src/xrr_fitter/model/project.py`：项目持有 `ConstraintRule` 集合。
- 改 `src/xrr_fitter/evaluation.py`：求值与导数合成，绑定期校验。
- 改 `src/xrr_fitter/fit/problem.py`：编译期把约束 target 从自由变量中剔除。
- 改 `src/xrr_fitter/io/project_codec.py` 与 `io/codec_common.py`：序列化节点树。
- 序列化零改造：项目文档已是 JSON，`sort_keys=True` 保证同一棵树字节表示确定。
- 不改 `fit/joint_sharing.py` 的索引投影路径。
- 扩 `src/xrr_fitter/api.py`：新增 `validate_constraint_rules` / `set_constraint_rules`，
  与既有 `validate_sharing_rules` / `set_sharing_rules`（`api.py:82,84`）并列导出。
- 新增 `src/xrr_fitter/gui/parameters/constraints.py`。

## 界面

`ALLOWED["gui"] = {"gui", "api"}`，所以 GUI 不能碰 `model.parameters` 的节点类型内部，
只能经 `api` 的 validate/set 对提交。这决定了必须先有上面那对 API 函数，GUI 才有落点。

`ConstraintEditor(QWidget)` 照 `gui/parameters/sharing.py` 的 `SharingEditor` 复制模式，
逐条对齐：

- `constraints_changed = Signal(tuple)`，挂进 `ParametersPanel` 的 `QTabWidget`
  （`panel.py:69-72` 现有"参数"/"共享"两页）作第三页"约束"。
- `document.project_changed.connect(self._render)`，`_render` 从
  `document.project.constraint_rules` 重建，不持有可变副本。
- `apply_rules()` 走 `api.validate_constraint_rules` → `api.set_constraint_rules` →
  `document.replace_project` → `constraints_changed.emit`；`TypeError`/`ValueError` 写进
  名为 `constraintValidationError` 的 `QLabel` 后**重新抛出**，与 `SharingEditor.apply_rules`
  的行为一致（先 `setText`+`show` 再 `raise`）。
- `QTreeWidget` 表头 `("目标参数", "表达式")`，节点树以缩进子项展开，父项显示 op、叶子显示
  `param` 名或 `const` 值。

**构造方式：树形拾取，不是文本框。** 非目标里明确不做字符串解析器，界面必须与之一致，否则
GUI 会逼出一个解析器。`ConstraintDialog(QDialog)` 照 `gui/structure/dialogs.py` 的模式建
完整不可变对象：目标参数用下拉框（候选来自 `api.describe_parameters` 且 `not locked`，复用
`SharingEditor.eligible_names` 的过滤思路），表达式用 `QTreeWidget` 加"加运算符/加参数/加常数"
三个按钮增量搭建，`_accept_fields` 里一次性构造 `ConstraintRule` 并调 `commit_rule` 回调，
失败则错误落在 `constraintDialogError` 标签上且对话框不关闭。

**target 的 free 标志自动清除要在界面上看得见。** 后端会清除该标志，参数表必须同步反映，
否则用户会看到一个"勾着可优化但实际被约束"的行。做法：`ParameterTable._render_row`
（`table.py:96`）对被约束驱动的行把"锁定"列渲染为只读勾选并加 tooltip 指出驱动它的约束，
"初值/下限/上限"三列（`VALUE_COLUMNS`）置为不可编辑。这需要 `api.describe_parameters`
返回的 `ParameterDefinition` 能表达"被约束驱动"——加一个布尔字段，与既有 `locked`
分开，因为二者的成因和可恢复路径不同。

**绑定期错误直接呈现。** 环检测、跨阶段约束、与 `SharingRule` 冲突这三类错误都在
`api.validate_constraint_rules` 里抛出，界面只负责把消息原样显示。环路径要写进异常消息文本，
因为用户只能从消息里知道是哪几条约束成环。

## 失败与状态

- 环（自引用、两跳、三跳、跨数据集）在绑定期报错并指出环路径。
- 同一 target 被 `SharingRule` 与 `ConstraintRule` 同时驱动时报错。
- 约束引用不存在的参数时报错，复用既有 `EvaluationConstraintError`
  （`evaluation.py:726`）。
- 约束结果越界、非有限、target 为 integer、或引用 integer 参数时报错。
- `JointFitLayout` 的既有告警（其文档承认 "cannot assert that a shared parameter still
  exists or is free"）在约束下更易触发，需补告警测例。

## 验证

- **解析导数 vs 有限差分，对每种 op 与每种 transform 的组合。** 这是主验收门。
- 环检测四类测例。
- `ConstraintRule` 为空时全量测试逐位不变。
- 与 `SharingRule` 的冲突检测测例。
- 两阶段解码测例：roughness 约束在动态上界变化时导数仍正确。
- 跨阶段约束被拒绝的测例。
- 工程文件往返：含约束树的项目保存后加载，节点树逐位相等。
- GUI 测例照既有 sharing 编辑器的测法：合法序列提交后 `document.project` 更新且信号带上
  持久化后的值；非法序列使 `constraintValidationError` 可见且项目不变。
- GUI 测例：被约束驱动的参数行，三个数值列不可编辑、锁定列只读。

## 非目标

- 不做字符串表达式解析器。节点树是唯一构造方式，GUI 也照此约束（树形拾取而非文本框）。
- 不做不等式约束或惩罚型软约束。
- 不替换 `SharingRule`。
- 不支持 integer 参数作为约束 target 或自变量。
