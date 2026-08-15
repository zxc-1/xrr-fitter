# 表达式参数约束实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把参数耦合从「只能相等」升级为「可用表达式关联」。新增 `ConstraintNode` / `ConstraintRule` 节点树，在求值层按链式法则合成解析导数，**完全不碰** `physics/derivatives.py` 的 Parratt 与卷积 Jacobian。`SharingRule` 的索引投影快路径保留不动。

**Architecture:** 节点树是纯值类型，落在 `model/parameters.py`（`MODEL_ALLOWED["parameters"] = set()`，只需自模块的 `ParameterReference` 与浮点运算，自洽）。**所有绑定期检查也落在 `model/parameters.py`**，不落 `evaluation.py`——见修正 4，这是架构门禁决定的，不是风格选择。`evaluation.py` 只做运行期求值与链式法则，并从 `model.parameters` import 那些纯检查函数（`ALLOWED["evaluation"] = {"model", "physics"}`，合规）。

**Tech Stack:** Python 3.12、NumPy 2.x、PySide6 6.8+、pytest 8.3+。**不新增依赖。**

**Design source:** `docs/superpowers/specs/2026-08-09-expression-constraints-design.md`

---

## 对 spec 的修正

写这份 plan 时逐条对着 HEAD 核了 spec，有十二处不符。修正 1、3、4、5 是硬阻塞：照 spec 写会分别导致旧工程文件读不回来、前向模型用错值、架构门禁失败、以及所有既有工程文件失效。**执行时以本节为准，不以 spec 为准。**

### 修正 1（硬阻塞）：给 `ParameterDefinition` 加字段会让所有含候选的旧工程文件读不回来

spec 要求「加一个布尔字段，与既有 `locked` 分开」。但 `ParameterDefinition` 是**被持久化**的，而且必填字段集是**自动派生**的（`io/codec_candidates.py:53`）：

```python
# prior is omitted from the auto-derived field set: it is emitted only when
# present so projects saved before priors existed stay byte-identical, and it
# is read back as an optional key (a bare definition decodes prior to None).
DEFINITION_FIELDS = frozenset(ParameterDefinition.__dataclass_fields__) - {"prior"}
```

新字段会**自动**进 `DEFINITION_FIELDS`，于是：
- `_parameter_definition_to_dict:57` 无条件 emit 它 → 新写的文件多一个键；
- `_parameter_definition_from_dict:64` 的 `_mapping(value, set(DEFINITION_FIELDS), ...)` 把它当**必填** → 每个存过候选的旧工程文件都报 `missing=['constrained']`。

`prior` 那三行注释就是为这件事写的，照抄它的做法：`DEFINITION_FIELDS = frozenset(...) - {"prior", "constrained"}`；`_parameter_definition_to_dict` 里 `if value.constrained:` 才 emit；`_parameter_definition_from_dict` 的 `optional={"prior", "constrained"}`，并 `payload.pop("constrained", False)` 后显式传参。

### 修正 2：同一个字段还会漂移 checkpoint 指纹

`ParameterDefinition` 在 checkpoint 图里（`fit/checkpoint.py:96`）：

```python
parameter_settings_fingerprint=_fingerprint(problem.parameter_definitions),
```

所以加字段会改指纹，旧 checkpoint 无法 resume。补 `POST_FREEZE_OMITTED_DEFAULTS` 条目：

```python
("ParameterDefinition", "constrained"): False,
```

语义正好对：未约束（`False`）时字段被省略，历史指纹逐位复现；一旦被约束驱动（`True`）就保留字段，理应换一个 identity。

**只有 `test_frozen_stage_search` 抓得到这件事**，跑窄测试集会漏。Task 2 必须显式跑它。

### 修正 3（硬阻塞，spec 挂载点分析不完整）：求值必须同时挂到 `values_by_name`

spec 说「挂在 `evaluation.py:590 values_and_jacobians()`」，并称「这是整个设计最关键的发现：解析梯度不需要重写」。前半句是**不够的**。

`values_and_jacobians` 实际在 `:585`。问题在于它不是前向模型的取值路径。前向模型走 `values_by_name`（`evaluation.py:1041`）：

```python
try:
    values = values_by_name(problem, unit_vector)
    rebuilt = rebuild_structure(problem.structure, values)
```

而 `_declared_values:543` 只用**初值**播种，被剔出自由变量的 target 在 `values_by_name` 里**没有任何东西覆写它**：

```python
def _declared_values(problem: object) -> dict[str, float]:
    return dict(map(_initial_parameter_pair, problem.parameter_definitions))
```

照 spec 只改 `values_and_jacobians` 的后果：`t2 = 2*t1` 从不进入 `rebuild_structure`，物理模型永远用 `t2` 的初值。拟合会「成功收敛」到一个约束从未生效的结果——静默错误，没有任何异常。

`values_by_name` 在 `src/` 有十几个消费方：`evaluation.py:1041`（前向模型）、`analysis/derivatives.py:115-116`（`physical_parameter_jacobian` 的有限差分）、`analysis/bootstrap.py:286`、`analysis/mcmc.py:358`、`analysis/binary_profiles.py:190`、`analysis/profile_tasks.py:83`、`fit/stages.py:857`、`fit/joint_roughness.py:155`。

顺带一个更阴的后果：**spec 的主验收门会自己失效**。`analysis/derivatives.py:95-122` 的有限差分参考就是拿 `values_by_name` 的差商算的。只改解析侧，「解析导数 vs 有限差分」比的是「带约束的解析」对「不带约束的差分」，必然不等——而失败原因跟链式法则对不对毫无关系，会把人引向完全错误的排查方向。

**改成：** 抽一对共享 helper，在**两个**函数里按相同的两阶段次序调用：

- `_apply_constraint_values(problem, values)`——纯取值，`values_by_name` 与 `values_and_jacobians` 都调。
- `_apply_constraint_jacobians(problem, values, value_jacobians)`——只有 `values_and_jacobians` 调。

阶段次序（两个函数一致）：第一趟在 `_decode_nonrough_values` 之后、`_roughness_dynamic_uppers` 之前解非 roughness 约束（必须在此之前，因为动态上界由几何算出，几何依赖 thickness 一类可能被约束驱动的值）；第二趟在 postponed roughness 循环之后解 roughness 约束。

对应地，`values_and_jacobians` 里第一趟的 Jacobian 合成也必须在 `:618` 的 `_roughness_dynamic_upper_jacobians` **之前**完成，因为那个函数读的就是刚写好的 `value_jacobians`。

### 修正 4（硬阻塞）：绑定期检查不能落在 `evaluation.py`

spec 一边说「改 `src/xrr_fitter/evaluation.py`：求值与导数合成，**绑定期校验**」，一边说「环检测、跨阶段约束、与 `SharingRule` 冲突这三类错误都在 `api.validate_constraint_rules` 里抛出」。这两句合不起来：

```python
"services": {"services", "model", "io", "physics", "fit", "analysis"},
```

`ALLOWED["services"]` **不含 `evaluation`**，`rg -l "xrr_fitter.evaluation" src/xrr_fitter/services/` 也确实无输出——没有先例可循。`api.validate_constraint_rules` 的实现体落在 `services/parameters.py`（与 `validate_sharing_rules:355` 并列），它调不到 `evaluation.py` 里的任何东西。

**改成：** 纯检查函数落 `model/parameters.py`，与节点类型同模块：

- `constraint_cycle_path(rules) -> tuple[str, ...]`——环路径，空元组表示无环。
- `validate_constraint_stage_split(rules, definitions_by_reference)`——跨阶段与 integer 自变量。
- `constraint_sharing_conflicts(rules, sharing_rules)`——双驱动冲突。

它们只依赖 `ParameterReference` / `ConstraintNode` / `ParameterDefinition`（全在本模块）与浮点运算，`MODEL_ALLOWED["parameters"] = set()` 下合规。`services/parameters.py` 与 `evaluation.py` 都能 import——后者的 `ALLOWED["evaluation"] = {"model", "physics"}` 含 `model`。

### 修正 5（硬阻塞）：项目根字段集是精确集，直接加键会让**所有**既有工程文件失效

spec 称「序列化零改造：项目文档已是 JSON，`sort_keys=True` 保证同一棵树字节表示确定」。字节确定性成立，但**字段集兼容性不成立**。

`io/project_codec.py:440` 的解码入口：

```python
payload = _mapping(value, _project_fields(), "project")
```

`_mapping` 的第四参 `optional` 默认 `None`（`io/codec_common.py:84-97`），所以项目根传的是**精确字段集，没有 optional 通道**。把 `constraint_rules` 加进 `_project_fields():355` 会让每个不含该键的旧文件报 `missing=['constraint_rules']`。而 `_validate_version:387` 是 `value != SCHEMA_VERSION` 精确相等，旧文件也不能靠版本号绕过。

**改成：**

```python
payload = _mapping(value, _project_fields(), "project", {"constraint_rules"})
```

`_project_fields()` **不加**该键；`project_to_dict:369` 里非空才 emit；`project_from_dict:476` 用 `payload.get("constraint_rules", [])` 解码。这样无约束项目的编码字节与 HEAD 完全相同，旧文件照旧能读。

**不要 bump `SCHEMA_VERSION`**（`model/project.py:51` 现为 `2`）。bump 需要配一个 `_migrate_v2_document`，还会牵动 `model/project.py:481` 的版本相等判定与所有以 `schema_version: 2` 写死的 fixture——代价远大于一个 optional 键。

### 修正 6：`XrrProject` 的新字段位置有约束

尾字段是 `base_directory: str | None = field(default=None, compare=False, repr=False)`（`model/project.py:229`）。`constraint_rules` 语义上该紧跟 `sharing_rules:226`，但那样会把 `ui_state` / `measurement_preset` / `base_directory` 的位置参数序号全部后移。

**做法：** 插在 `sharing_rules` 之后（语义正确），但**先** `rg -n 'XrrProject\(' src tests | rg -v '='` 确认没有位置构造式依赖后三个字段的序号。若有，改成插在 `base_directory` **之前**、`measurement_preset` 之后，并在 plan 执行记录里写明。`dataclasses.replace` 是关键字调用，不受影响。

`__post_init__:231` 要加 `object.__setattr__(self, "constraint_rules", tuple(self.constraint_rules))`，与 `sharing_rules:233` 同型。`validate_project:513` 里加一次 `_validate_constraints(...)`，紧跟 `_validate_sharing:519`。

### 修正 7：「target 的 free 标志自动清除」不能实现成 `locked=True`

`fit/problem.py:65-71` 的自由变量过滤：

```python
def _variables(definitions) -> tuple[ParameterCoordinate, ...]:
    return tuple(
        ParameterCoordinate(index, definition.name, definition.transform)
        for index, definition in enumerate(definitions)
        if not definition.locked
    )
```

本项目里「free」就是 `not locked`，所以「自动清除 free 标志」最省事的写法是把 target 的 `locked` 置 `True`——**但那样做会违反 spec 自己的界面要求**：「加一个布尔字段，与既有 `locked` 分开，因为二者的成因和可恢复路径不同」，且参数表要能把「被约束驱动」与「用户锁定」渲染成不同状态。

**改成：** `locked` 保持用户语义不动，过滤条件放宽为 `if not (definition.locked or definition.constrained)`。这也正是 spec 代码边界那句「改 `fit/problem.py`：编译期把约束 target 从自由变量中剔除」的落点。

### 修正 8：`api` 扩面是三重精确门禁，不是「加两个导出」

spec 只说「扩 `api.py`：新增 `validate_constraint_rules` / `set_constraint_rules`」。实际 `tests/architecture/test_public_api.py` 有三处精确断言，漏一处即红：

1. `PUBLIC_NAMES` 是**有序元组**，`assert tuple(api.__all__) == PUBLIC_NAMES`（`:205-208`）。`set_constraint_rules` 插在 `set_batch_mode:90` 之后、`set_dock_state:91` 之前；`validate_constraint_rules` 插在 `validate_parameter_priors:106` 之前。

   **注意这两份列表并非严格字母序**——`start_fit_job` 排在 `start_automatic_fit_job` 之前（`test_public_api.py:101-102` 与 `api.py:248-249` 一致地这样排）。所以判据不是「按字母序」而是「两份列表必须逐位一致」。上面给的两个位置在两份列表里都成立，照它插即可，别顺手去「修正」既有顺序——那会让 `api.py` 与测试同时要改，且不属于本轮范围。
2. `SIGNATURES`（`:113`）要求**精确签名字符串**，照 `:139` / `:145` 的既有格式写：
   `"set_constraint_rules": "(project: 'XrrProject', rules: 'Sequence[ConstraintRule]') -> 'XrrProject'"`
   `"validate_constraint_rules": "(project: 'XrrProject', rules: 'Sequence[ConstraintRule]') -> 'tuple[ConstraintRule, ...]'"`
3. `USE_CASE_GROUPS`（`:171` 有 `"sharing": ("validate_sharing_rules", "set_sharing_rules")`）要加 `"constraints": ("validate_constraint_rules", "set_constraint_rules")`。

同时 `api.py` 自身要改两处：import 块（`:99`/`:102` 附近）与 `__all__`（`:244`/`:255` 附近），两处都按字母序。

`ConstraintRule` 若要出现在 GUI 的类型标注里，还得跟 `SharingRule` 一样从 `api` 重导出。

### 修正 9：`set_constraint_rules` 必须复刻结果失效逻辑

spec 只说「与既有 `validate_sharing_rules` / `set_sharing_rules` 并列导出」，没说要复刻语义。`set_sharing_rules`（`services/parameters.py:377-398`）做了四件事，缺一件就会留下与新约束不符的陈旧结果：

```python
validated = validate_sharing_rules(project, rules)
if validated == project.sharing_rules:
    return project
affected = _rule_dataset_ids((*project.sharing_rules, *validated))
if project.batch_mode == "joint" and affected:
    affected = {dataset.dataset_id for dataset in project.datasets}
datasets = tuple(
    _cleared(dataset, clear_evidence=False) if dataset.dataset_id in affected else dataset
    for dataset in project.datasets
)
selected = tuple(item for item in project.ui_state.selected_candidate_ids if item[0] not in affected)
```

逐条照做：未变则**返回原对象**（GUI 靠 `updated is current` 判断要不要 `replace_project`）、joint 模式扩散到全部数据集、`_cleared(..., clear_evidence=False)`、剔除失效的 `selected_candidate_ids`。

`affected` 的口径要覆盖约束的**两侧**：target 的 dataset 与表达式里每个 `ParameterReference` 的 dataset。

另外照 `validate_sharing_rules:363` 的手法，用 `replace(project, constraint_rules=values)` 触发一次 `validate_project`，白拿跨数据集引用校验。

### 修正 10：绑定期错误不要复用 `EvaluationConstraintError`

spec 说「约束引用不存在的参数时报错，复用既有 `EvaluationConstraintError`（`evaluation.py:726`）」。实际类在 `:721`，其 docstring 写明了用途：

> This explicit type is the only boundary fit objectives may convert into an **invalid result**; unexpected implementation errors continue to propagate.

它会被 `:1837`、`:1914`、`:2361` 的 `except EvaluationConstraintError:` **静默吞成「无效候选」**。用它报「引用了不存在的参数」，后果是一个拼错参数名的约束不报错，而是让每个候选静默变成无效——用户看到的是「拟合全部失败」，看不到拼写错误。

**改成：** 绑定期的引用/环/跨阶段/integer 四类错误抛 `ValueError`（类型错误抛 `TypeError`），与 `api.validate_*` 家族一致，GUI 的 `except (TypeError, ValueError)` 正好接住。`EvaluationConstraintError` 只留给**运行期**的越界与非有限结果——那才是「这个候选无效」的正确语义。

### 修正 11：`VALUE_COLUMNS` 不存在，`_render_row` 也不在 spec 说的行号

spec 要求「`ParameterTable._render_row`（`table.py:96`）……『初值/下限/上限』三列（`VALUE_COLUMNS`）置为不可编辑」。实际 `_render_row` 在 `table.py:118`，且**没有 `VALUE_COLUMNS` 常量**。真实布局是 7 列硬编码：

```python
for column, value in enumerate(values):
    item = QTableWidgetItem(value)
    if column in (0, 4):
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
```

0=显示名、1=初值、2=下限、3=上限、4=单位、5=锁定、6=先验。0 与 4 已不可编辑，1/2/3 可编辑。

**改成：** 被约束驱动的行把 `column in (0, 4)` 放宽成 `column in (0, 1, 2, 3, 4)`。锁定列（`:135-138`）当前是：

```python
locked.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
```

被约束驱动时**去掉 `ItemIsUserCheckable`**（保留 Enabled/Selectable），即 spec 要的「只读勾选」，并 `setToolTip` 指出驱动它的约束。若同期要引入 `VALUE_COLUMNS` 常量，那是顺手的可读性改动，可以做，但别在 plan 之外扩到别的行为。

### 修正 12：其余锚点漂移

| spec 的标注 | HEAD 实际位置 |
| --- | --- |
| `evaluation.py:590 values_and_jacobians()` | `:585` |
| `EvaluationConstraintError`（`evaluation.py:726`） | `:721` |
| `api.py:82,84` 的 validate/set sharing | `api.py:99`（import `set_sharing_rules`）、`:102`（import `validate_sharing_rules`）、`:244`/`:255`（`__all__`）；实现体在 `services/parameters.py:355`/`:377` |
| `panel.py:69-72` 现有两页 | 准确：`:69` `QTabWidget()`、`:71` `"参数"`、`:72` `"共享"`。新页 `addTab(self.constraint_editor, "约束")` 接在 `:72` 之后 |
| `table.py:96` 是 `_render_row` | `:118`（`:96` 在 `definition()` 里） |

`SharingEditor` 的可复制细节（`gui/parameters/sharing.py`，供 `ConstraintEditor` 逐条对齐）：`rules_changed = Signal(tuple)` 在 `:15`；`setObjectName("sharingEditor")` `:20`；树 `"sharingTree"` `:22`；错误标签 `"sharingValidationError"` `:25`；`document.project_changed.connect(self._render)` `:32`；`apply_rules` `:39` 返回 `bool`，先 `setText`+`show` 再 `raise`，成功后 `hide()`，`updated is current` 时返回 `False` 且**不** emit。spec 对这段的描述准确，照抄即可。

---

## Global Constraints

- **约束为空时全量测试逐位不变**（spec 明列的验收项）。三条保障：`constraint_rules` 为空时不 emit 项目键；`constrained=False` 时不 emit definition 键；checkpoint 指纹靠 `POST_FREEZE_OMITTED_DEFAULTS` 省略。断言用 `==`。
- **不动 `fit/joint_sharing.py` 的索引投影路径**（spec 非目标）。等式共享继续走 gather。
- **不做字符串表达式解析器**（spec 非目标）。节点树是唯一构造方式，GUI 用树形拾取而非文本框——界面必须与非目标一致，否则会从 GUI 侧逼出一个解析器。
- **不碰 `physics/derivatives.py`。** 链式法则只在求值层合成。这条是 spec 的核心论点，也是本设计成立的前提。
- **不做不等式或惩罚型软约束；不替换 `SharingRule`；不支持 integer 参数作自变量**（spec 非目标）。
- **约束结果越界时明确报错，不静默 clip。** spec 给了理由且成立：clip 会让优化器看到平坦区域并误判收敛。
- `ConstraintNode` / `ConstraintRule` 用 `@dataclass(frozen=True, slots=True)`，与 `model/parameters.py` 全模块一致（spec 的示例只写了 `frozen=True`，补上 `slots=True`）。`ConstraintNode.operands` 默认 `()` 是不可变的，`slots=True` 允许。
- 测试命令必须带 `--import-mode=importlib`：
  `.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`
  GUI：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
  `tools/verify.py` 在本地会因仓库内的 `.venv` 触发 "generated directory inside repository" 而失败，不作为本地门禁。
- 不 stage `.claude/`。

---

## File Structure

| 文件 | 动作 |
| --- | --- |
| `src/xrr_fitter/model/parameters.py` | 新增 `ConstraintNode` / `ConstraintRule`；`ParameterDefinition` 末尾加 `constrained: bool = False`；新增三个纯检查函数（修正 4） |
| `src/xrr_fitter/model/project.py` | `XrrProject` 加 `constraint_rules`（修正 6）；`validate_project` 加 `_validate_constraints` |
| `src/xrr_fitter/fit/checkpoint.py` | `POST_FREEZE_OMITTED_DEFAULTS` 加 `("ParameterDefinition", "constrained"): False` |
| `src/xrr_fitter/io/codec_candidates.py` | `DEFINITION_FIELDS` 排除 `constrained`；条件 emit + optional 读（修正 1） |
| `src/xrr_fitter/io/project_codec.py` | 节点树 to/from dict；根键走 optional 通道（修正 5） |
| `src/xrr_fitter/evaluation.py` | 两个共享 applier，在 `values_by_name` 与 `values_and_jacobians` 里按同一两阶段次序调用（修正 3） |
| `src/xrr_fitter/fit/problem.py` | `_variables` 过滤放宽为 `not (locked or constrained)`（修正 7） |
| `src/xrr_fitter/services/parameters.py` | `validate_constraint_rules` / `set_constraint_rules`（修正 9） |
| `src/xrr_fitter/api.py` | import 块 + `__all__` 各加两个名字（修正 8） |
| `tests/architecture/test_public_api.py` | `PUBLIC_NAMES` + `SIGNATURES` + `USE_CASE_GROUPS` 三处（修正 8） |
| `src/xrr_fitter/gui/parameters/constraints.py` | **新建** `ConstraintEditor` + `ConstraintDialog` |
| `src/xrr_fitter/gui/parameters/panel.py` | 第三页 `"约束"` |
| `src/xrr_fitter/gui/parameters/table.py` | 被约束驱动行的列渲染（修正 11） |
| `tests/unit/test_evaluation.py` | 主验收门：解析导数 vs 有限差分，op × transform 全组合 |
| `tests/unit/model/test_parameters.py` | 节点类型、环检测四类、跨阶段拒绝、integer 拒绝 |
| `tests/unit/io/test_project_codec.py` | 节点树往返 + 旧文件缺键 |
| `tests/unit/io/test_export_values.py` 或 `tests/unit/model/test_export_values.py` | `ParameterDefinition` 往返（先定位 `_parameter_definition_from_dict` 的现有测例） |
| `tests/unit/services/test_parameters.py` | 冲突检测 + 结果失效 |
| `tests/unit/fit/test_problem_compilation.py` | target 不在自由变量里 |
| `tests/gui/test_parameter_sharing.py` | `ConstraintEditor` 与面板第三页（照 `SharingEditor` 的现有测法） |
| `tests/gui/test_parameter_table.py` | 被约束驱动行的列渲染 |
| **不改** | `physics/derivatives.py`、`fit/joint_sharing.py`、`SCHEMA_VERSION` |

---

## Tasks

### Task 0：只读核验前置事实（不改任何文件）

- [ ] 确认 `XrrProject` 无位置构造式依赖尾部字段序号：`rg -n 'XrrProject\(' src tests | rg -v '='`（修正 6 的前提）
- [ ] 确认 `ParameterDefinition` 往返的现有测例落在哪个文件：`rg -ln '_parameter_definition_from_dict|parameter definition' tests/`，Task 2 的 RED 加在那里而不是新建文件。
- [ ] 确认 `values_by_name` 消费方清单仍是修正 3 列的那些：`rg -n 'values_by_name' src/`
- [ ] 记录基线：`.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`

### Task 1：节点类型与纯检查函数

- [ ] RED：`tests/unit/model/test_parameters.py` 加测例——`ConstraintNode` 各 op 的构造校验（`ref` 必须带 `reference`、`const` 必须带 `value`、二元 op 的 `operands` 元数正确）；`ConstraintRule` 拒绝 target 出现在自己的表达式里（自引用是一跳环）；环检测四类（自引用、两跳、三跳、跨数据集）都返回非空环路径；跨阶段（非 roughness target 依赖 roughness 自变量）被拒；integer target 与自变量都被拒。
- [ ] GREEN：`model/parameters.py` 加 `ConstraintNode` / `ConstraintRule`，均 `@dataclass(frozen=True, slots=True)`，`__post_init__` 里 `object.__setattr__(self, "operands", tuple(self.operands))` 后校验，与 `SharingRule:350-359` 同型。
- [ ] GREEN：加三个纯检查函数（修正 4）：`constraint_cycle_path`、`validate_constraint_stage_split`、`constraint_sharing_conflicts`。**环路径必须写进异常消息文本**——用户只能从消息里知道是哪几条约束成环（spec 界面节的明确要求）。
- [ ] 错误类型按修正 10：`ValueError` / `TypeError`，**不用** `EvaluationConstraintError`。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/model/test_parameters.py tests/architecture --import-mode=importlib -q`

### Task 2：`ParameterDefinition.constrained` + 两处兼容性

- [ ] RED：三个测例——`constrained=False` 时 `_parameter_definition_to_dict` 的输出**不含**该键（`assert "constrained" not in payload`）；手工构造缺该键的旧 payload 能解回且字段为 `False`；`constrained=True` 时往返保值。
- [ ] RED：checkpoint 指纹测例——未约束的 problem 指纹与 HEAD 相同（先在 Task 0 基线上记下一个已知指纹值），约束后指纹改变。
- [ ] GREEN：`ParameterDefinition` **末尾**（`prior` 之后）加 `constrained: bool = False`；`__post_init__` 的 bool 校验元组从 `(self.locked, self.integer, self.expert_only)` 扩到含 `self.constrained`。
- [ ] GREEN：`fit/checkpoint.py:24` 加 `("ParameterDefinition", "constrained"): False`（修正 2）。
- [ ] GREEN：`io/codec_candidates.py:53` 的 `DEFINITION_FIELDS` 排除 `"constrained"`；`_to_dict` 条件 emit；`_from_dict` 的 `optional` 加该键并 `pop` 后显式传参（修正 1）。扩一下 `:50-52` 的注释，把新字段一并说明。
- [ ] **必跑**（只有它抓指纹漂移）：`.venv/bin/python -m pytest tests/regression -k frozen_stage_search --import-mode=importlib -q`
- [ ] 验证：`.venv/bin/python -m pytest tests/unit tests/regression --import-mode=importlib -q`

### Task 3：`XrrProject.constraint_rules` + 项目校验

- [ ] RED：项目持有约束时 `validate_project` 通过；约束引用不存在的 dataset 时报错；同一 target 被 `SharingRule` 与 `ConstraintRule` 同时驱动时报错。
- [ ] GREEN：按修正 6 定位置加字段，`__post_init__` 加 tuple 归一化，`validate_project:513` 里 `_validate_sharing:519` 之后加 `_validate_constraints(project.constraint_rules, project.sharing_rules, dataset_ids)`，内部调 Task 1 的三个检查函数。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit --import-mode=importlib -q`

### Task 4：序列化

- [ ] RED：含多层节点树的项目保存后加载，节点树**逐位相等**（spec 明列）；`constraint_rules` 为空时编码输出**不含**该键；缺该键的旧 v2 文件正常加载且字段为空元组。
- [ ] GREEN：`io/project_codec.py` 加 `_constraint_node_to_dict` / `_constraint_node_from_dict`（递归）与 `_constraint_rule_to_dict` / `_from_dict`，照 `_sharing_to_dict:126` / `_sharing_from_dict:139` 的风格。
- [ ] GREEN：按修正 5 走 optional 通道——`_validated_document:440` 的 `_mapping` 传第四参 `{"constraint_rules"}`，`_project_fields():355` **不加**该键，`project_to_dict:369` 非空才 emit，`project_from_dict:476` 用 `payload.get(..., [])`。
- [ ] **不 bump `SCHEMA_VERSION`**，不加迁移函数。
- [ ] 递归解码要有深度上限，避免恶意/损坏文件造成栈溢出；超限抛 `ProjectSchemaError`。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/io --import-mode=importlib -q`

### Task 5：求值与链式法则（主验收门）

- [ ] RED（**主验收门**）：`tests/unit/test_evaluation.py` 加参数化测例，**每种 op × 每种 transform 的组合**都比解析导数与有限差分。`TRANSFORMS` 的成员先从 `model/parameters.py` 读出来，别猜。
- [ ] RED：两阶段测例——roughness target 的约束在动态上界变化时导数仍正确。
- [ ] RED（修正 3 的回归护栏）：**同一个约束项目，`values_by_name` 与 `values_and_jacobians` 返回的 `values` 字典必须逐键相等**。这条测例是防止只改一侧的唯一保障，缺了它修正 3 描述的静默错误会重新溜回来。
- [ ] RED：前向模型测例——`t2 = 2*t1` 的项目，`rebuild_structure` 看到的 `t2` 确实是 `2*t1` 而不是初值。
- [ ] RED：运行期越界抛 `EvaluationConstraintError`（这是它的正确用法），非有限结果同理。
- [ ] GREEN：加 `_apply_constraint_values` 与 `_apply_constraint_jacobians`，按修正 3 的次序在**两个**函数里调用：第一趟在 `_decode_nonrough_values` 之后、`:617` 的 `_roughness_dynamic_uppers` 之前；第二趟在 postponed 循环之后。Jacobian 合成同样必须在 `:618` 之前完成。
- [ ] GREEN：链式法则按 op 分派，每个 op 的偏导显式写出。目标 Jacobian 是各自变量切向量的线性组合，天然与 `problem.variables` 对齐。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/test_evaluation.py tests/unit/analysis --import-mode=importlib -q`

### Task 6：编译期剔除自由变量

- [ ] RED：`tests/unit/fit/test_problem_compilation.py` 断言被约束驱动的 target 不出现在 `problem.variables`，且 `len(problem.variables)` 相应减少；`locked` 与 `constrained` 各自独立生效（两个都为真、只有一个为真的三种组合）。
- [ ] GREEN：`fit/problem.py:65-71` 的过滤改 `if not (definition.locked or definition.constrained)`（修正 7）。**不要**把 target 的 `locked` 置 `True`。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/fit --import-mode=importlib -q`

### Task 7：services 层

- [ ] RED：`validate_constraint_rules` 对非 `ConstraintRule` 抛 `TypeError`；环/跨阶段/integer target/integer 自变量/冲突各抛 `ValueError` 且消息含环路径；`set_constraint_rules` 未变时**返回原对象**；变更时清掉受影响 dataset 的结果与 `selected_candidate_ids`；joint 模式扩散到全部数据集。
- [ ] GREEN：照 `validate_sharing_rules:355` / `set_sharing_rules:377` 复刻，含 `replace(project, constraint_rules=values)` 的校验副作用与修正 9 列的四件事。`affected` 覆盖 target 与全部自变量的 dataset。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/services --import-mode=importlib -q`

### Task 8：api 三重注册

- [ ] RED：先改 `tests/architecture/test_public_api.py` 的三处（修正 8），此时应因 `api.__all__` 不匹配而红。
- [ ] GREEN：`api.py` 的 import 块与 `__all__` 各按字母序加两个名字；若 GUI 需要类型标注，`ConstraintRule` 也照 `SharingRule` 重导出。
- [ ] 验证：`.venv/bin/python -m pytest tests/architecture --import-mode=importlib -q`

### Task 9：GUI 编辑器与对话框

- [ ] RED（落在 `tests/gui/test_parameter_sharing.py`，照 `SharingEditor` 的现有测法）：合法序列提交后 `document.project` 更新且信号带上持久化后的值；非法序列使 `constraintValidationError` 可见且项目不变；对话框内构造失败时错误落 `constraintDialogError` 且**对话框不关闭**。
- [ ] GREEN：新建 `gui/parameters/constraints.py`。`ConstraintEditor(QWidget)` 逐条对齐 `SharingEditor`：`constraints_changed = Signal(tuple)`、`setObjectName("constraintEditor")`、树 `"constraintTree"`、错误标签 `"constraintValidationError"`、`document.project_changed.connect(self._render)`、`apply_rules` 先 `setText`+`show` 再 `raise`、`updated is current` 返回 `False`。
- [ ] GREEN：`QTreeWidget` 表头 `("目标参数", "表达式")`，父项显示 op、叶子显示 `param` 名或 `const` 值。
- [ ] GREEN：`ConstraintDialog(QDialog)` 照 `gui/structure/dialogs.py` 的模式，一次性构造完整不可变对象。目标下拉框候选复用 `SharingEditor.eligible_names:62` 的过滤思路（`api.describe_parameters` 且 `not locked`），**再排除 integer 与已被其他约束驱动的参数**。表达式用「加运算符/加参数/加常数」三按钮增量搭建——**不是文本框**（spec 非目标）。
- [ ] GREEN：`panel.py:72` 之后 `tabs.addTab(self.constraint_editor, "约束")`。
- [ ] 控件都要 `objectName` / `setAccessibleName` / `setToolTip`，与既有编辑器的无障碍风格一致。
- [ ] 验证：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`

### Task 10：参数表渲染

- [ ] RED（落在 `tests/gui/test_parameter_table.py`）：被约束驱动的行，1/2/3 三列不可编辑、锁定列无 `ItemIsUserCheckable`、tooltip 指出驱动它的约束；未被驱动的行渲染与 HEAD 完全一致。
- [ ] GREEN：`table.py:118` 的 `_render_row` 按修正 11 改——不可编辑列集合放宽，锁定列去掉 `ItemIsUserCheckable`。
- [ ] 验证：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`

### Task 11：JointFitLayout 假设核验（无需实现）

- [x] `JointFitLayout` 没有告警字段或告警生成路径；docstring 只是声明该只读布局不编译参数定义。
- [x] constraint target 与 sharing member 的双重所有权已在项目校验和 joint 编译期直接拒绝，因此“约束先把共享成员变成非自由，再由布局发告警”的状态不可达。
- [x] 不为不存在的告警契约新增生产代码或伪造测例。

### Task 12：全量回归

- [ ] `.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`
- [ ] `.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
- [ ] `.venv/bin/python -m pytest tests/integration tests/regression tests/acceptance --import-mode=importlib -q`
- [ ] 确认「约束为空时逐位不变」：无约束项目的编码字节、checkpoint 指纹、`_dataset_artifacts` 清单与 Task 0 基线一致。
- [ ] 与 Task 0 基线比对：新增测例之外无失败变化。仓库既有的两个失败（见 `mem:xrr-test-suite-commands`）不算本轮回归，但要在汇报里点名。
- [ ] 清理临时产物。

---

## 剩余风险

- **修正 3 是这份 plan 里最容易被实现者忽略的一条。** 只改 `values_and_jacobians` 的代码能通过大部分测试——因为约束在解析梯度里生效了，看起来「work」——但前向模型用的是初值，结果全错且无异常。Task 5 里那条「两个函数的 `values` 必须逐键相等」的测例是唯一护栏，不要跳过。
- **主验收门的组合数可能很大。** op 数 × `TRANSFORMS` 数，且每个组合都要跑有限差分。若 `tests/unit/test_evaluation.py` 因此显著变慢，用 `pytest.mark.parametrize` 的 id 让失败可定位，但**不要**为了提速把组合裁剪掉——那正是这个门禁的价值所在。
- **`ConstraintNode` 的 op 集合本 plan 没有定死。** spec 只给了 `"ref" | "const" | "add" | "mul" | "pow" | ...` 的示意。Task 1 开工时先定死一个**最小闭集**（建议 `ref`/`const`/`add`/`sub`/`mul`/`div`/`pow`），每个 op 的偏导都要能解析写出。`pow` 的指数若允许是子表达式而非常数，偏导会牵进 `log`，定义域约束（底数必须为正）也随之而来——**建议第一版把 `pow` 的指数限定为 `const`**，把这个决定写进 Task 1 的执行记录。
- **序列化格式是单向承诺。** 节点树的 JSON 形状一旦发布就成了工程文件契约。Task 4 一次定死键名（`op` / `reference` / `value` / `operands`），不要留「以后再改」的余地。
- **修正 6 的字段位置依赖 Task 0 的 grep 结果。** 若发现存在依赖尾部序号的位置构造式，位置决定要改，且 Task 3 的 `__post_init__` 与 Task 4 的 codec 都不受影响——只有构造点受影响。
- **GUI 树形拾取的可用性没人验证过。** 非目标里排除了文本框，这在安全性与「不做解析器」上是对的，但多层嵌套表达式用三个按钮搭建会不好用。这是产品取舍，不是实现缺陷；若 Task 9 做完发现难用，记成新 spec，别在本轮偷偷加一个解析器。
