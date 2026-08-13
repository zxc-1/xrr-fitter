# 阶段二四份计划的依赖清单

四份尚未写 plan 的 spec：`nested-periodic-stacks`、`orso-export`、`expression-constraints`、
`stack-drift`。它们全部撞在阶段一三份计划的核心写集上，依赖是**结构性的**（设计形状由阶段一
产出决定），不是表层的行号漂移。本文件为每份列出「需要从上游冻结哪些接口事实」的核对清单。

**用法：** 上游落地后，对着对应小节逐条核对——把「现状锚点」替换为落地后的真实字段名/签名/
键名姿势，全部勾掉即可直接开写正式 plan，无需重新推依赖。

## 依赖拓扑

```
阶段一（正在做）                    阶段二（本文件）
─────────────────                  ─────────────────
interface-transitions  ──────────▶ nested-periodic-stacks
sld-uncertainty-bands  ──┐
                         ├───────▶ orso-export
parameter-priors       ──┤
                         └───────▶ expression-constraints ──▶ stack-drift
```

**关键：`stack-drift` 是二级依赖**——它不直接依赖阶段一，而是依赖 `expression-constraints`
（另一份阶段二计划）。必须等 `expression-constraints` 全绿后才能写。

**可开写顺序：**
1. `interface-transitions` 落地 → 写 `nested-periodic-stacks`
2. `sld-uncertainty-bands` + `parameter-priors` 落地 → 写 `orso-export`
3. `parameter-priors` 落地 → 写 `expression-constraints`
4. `expression-constraints` 落地 → 写 `stack-drift`

---

## 1. nested-periodic-stacks ← interface-transitions

**依赖性质：** 两者同改 `physics/stack.py` 的层展开循环与 `model/structure.py` 的组件类型。
若界面过渡改了层的粗糙度剖面表示，嵌套展开的「逐位相等」主验收门必须对着**新的**层构建逻辑
验证，否则等价性测例会拿旧基线对新代码，假绿。

**落地后需确认（核对清单）：**

- [ ] `model/structure.py` `StructureComponent` 联合类型（现状锚点 `:207`
      `LayerSpec | PeriodicBlock | GradientLayerSpec`）在界面过渡后是否新增成员。嵌套要把
      `PeriodicBlock.layers` 放宽为 `tuple[LayerSpec | PeriodicBlock, ...]`，得先知道联合的
      最终形状。
- [ ] `physics/stack.py` `_append_periodic`（现状锚点 `:198`）落地后的签名与它调用的
      `_append_layer`/`_append_component` 分派——嵌套要把内层调用改成 `_append_component`
      递归，前提是这个分派函数在过渡改动后仍存在且语义不变。
- [ ] `PeriodicSpan` 登记条件（现状锚点 `:205` `repeats > 1`）与 `top_roughness_a` 覆盖点
      （现状锚点 `:202` `repeat_index == layer_index == 0`）是否被过渡剖面改写。嵌套的「只登记
      最外层 span」方案建立在这两处的当前语义上。
- [ ] `_validate_roughness`（现状锚点 `:237`）的邻层约束在过渡剖面下的最终形式——嵌套展开后
      要复用它，不能绕过。
- [ ] `io/codec_common.py` 的结构序列化姿势（过渡剖面若给层加了字段，嵌套的递归序列化要沿用
      同一套 optional-key 处理）。
- [ ] GUI：`gui/structure/editor.py` `_component_item`（现状锚点 `:241`）与
      `gui/structure/dialogs.py` `PeriodicDialog`（现状锚点 `:190`）在过渡剖面加了字段后的表单
      布局——嵌套要把 `QTableWidget` 改 `QTreeWidget` 并递归，得先知道过渡给对话框加了什么。

**解锁信号：** `interface-transitions` 全绿且 `physics/stack.py` 的层构建、`model/structure.py`
的组件类型定型（其计划的「最终验收记录」已勾完）。

---

## 2. orso-export ← sld-uncertainty-bands + parameter-priors

**依赖性质：** 三份都往同一个 `UncertaintyReport`（`model/analysis.py:461`）追加字段、往同一对
`codec_results.py` 的 `_uncertainty_to_dict`/`_from_dict` 加可选键。这是合并冲突高发区，且三者
都涉及旧工程文件兼容。orso 的 `parameter_sigma` 必须照前两者确立的 optional-key 范式加，不能
自立一套。

**落地后需确认（核对清单）：**

- [ ] `UncertaintyReport` 的最终字段序。SLD带加 `sld_bands`、先验加 `prior_conflicts`、orso 加
      `parameter_sigma`，三者都必须**追加在类末尾且带默认值**（该 dataclass 多处构造点，插中间
      会静默重解读位置参数）。orso 开写时要接在前两者字段**之后**。
- [ ] `codec_results.py` `_uncertainty_to_dict`/`_from_dict`（现状锚点 `:126`/`:152`）的
      optional-key 集合最终长什么样——`parameter_sigma` 要并进同一个「有值才发、读取端当可选键」
      的处理，`_mapping(value, required, "uncertainty report", {...})` 的第四参集合要把它算进去。
- [ ] `io/codec_common.py` `NULLABLE_ARRAY_FIELDS`（现状锚点 `:54`）是否已被 SLD带的
      `imaginary` 占用。`parameter_sigma` 也是可空数组，要登记进同一处，否则 `_validate_nulls`
      对未登记的 `null` 抛 `ProjectSchemaError`。
- [ ] `analysis/report.py` 填充点（现状锚点 `:166` `physical_covariance` 已算出、`:167`
      归一化成 correlation）。orso 的方案 A 要在 `:166` 之后填 `parameter_sigma`（对角标准差），
      得确认先验/SLD带落地后这段没被挪动。
- [ ] `services/exports.py` 的 artifact 列表落点（现状锚点 `export_result:165`、
      `_dataset_artifacts:132`、`_root_artifacts:146`）。
- [ ] `pyproject.toml` 依赖：`orsopy` + `jsonschema` 进生产依赖，然后 `tools/lock_environment.py`
      与 `tools/lock_windows_environment.py` 两平台重生成 lock。**这条要先单独确认取舍并获授权**
      （新增生产依赖 + 扩 CI 门禁，属规则要求先说明的项）。

**解锁信号：** `sld-uncertainty-bands` 与 `parameter-priors` 双双全绿，`UncertaintyReport` 字段
与 `codec_results` 可选键定型。

---

## 3. expression-constraints ← parameter-priors

**依赖性质：** 两者是**同一批落点、同一套范式**——都改 `model/parameters.py`（加类型/字段）、
`evaluation.py`（求值，且都受「不破坏逐位重放」约束）、`model/project.py` + `io/project_codec.py`
（存储）。先验落地会确立三个范式，约束表达式直接照抄；现在写等于凭猜想设计一遍，先验落地后
整份重写。

**落地后需确认（核对清单）：**

- [ ] `MODEL_ALLOWED["parameters"] == set()` 仍成立——先验的 `PriorSpec` 靠自洽类型不 import
      任何东西，约束的 `ConstraintNode`/`ConstraintRule`（全新类型）必须沿用同一纪律，数学放
      `evaluation.py`。这是架构门禁，先确认先验没破例。
- [ ] `ParameterDefinition` 的最终字段序。先验加 `prior` 字段后，约束要再加一个「被约束驱动」
      布尔字段（与 `locked` 分开，成因和可恢复路径不同），追加在 `prior` 之后。
- [ ] `DatasetProject` 存参数级附加数据的姿势。先验确立 `parameter_priors` + 新 `ParameterPrior`
      类型（不塞进 `ParameterSetting`）。约束的 `ConstraintRule` 集合照同一模式挂在
      `model/project.py`，`project_codec.py` 照同一 optional-key 处理序列化。
- [ ] `evaluation.py` 求值挂载点。先验接在 `problem_log_probability`（现状锚点 `:2135`，逐位
      重放约束）；约束挂在 `values_and_jacobians`（现状锚点 `:590`）合成链式导数。两者是**不同
      函数但同一逐位不变纪律**——要确认先验落地后 `values_and_jacobians` 签名与两阶段解码
      （`_decode_nonrough_values` → `_roughness_dynamic_uppers`）未变。
- [ ] `api.py` 导出块位置（现状锚点 `:83`/`:85` `set_sharing_rules`/`validate_sharing_rules`，
      `:82`/`:84` 是 `set_parameter_settings`/`validate_parameter_settings`；`__all__` 侧在
      `:199`/`:208`。先验又在此加了 `set_parameter_priors`/`validate_parameter_priors`）。约束的
      `validate_constraint_rules`/`set_constraint_rules` 追加进同一块，import 与 `__all__` 两处都要加。
- [ ] `services/parameters.py` `set_parameter_settings` 契约（现状锚点 `:172`，先验照它写了
      `set_parameter_priors`）——约束的 set 函数照同一「validate 返回验证值 / 不变则返回原对象 /
      `_replace_invalidated(..., clear_evidence=?)`」契约。

**解锁信号：** `parameter-priors` 全绿，`ParameterDefinition` 字段、`DatasetProject` 存储范式、
`evaluation.py` 逐位不变接入范式、`api.py` validate/set 对全部定型。

---

## 4. stack-drift ← expression-constraints（二级依赖）

**依赖性质：** stack-drift 的全部机制建立在 `ConstraintNode` 之上——spec 原文「本设计只做两件
事：给节点集合补 `sin` 与 `cos`，以及提供一个把漂移意图展开成约束树的构造辅助，不新增独立的
漂移机制」。`ConstraintNode` 不存在就无从加 op。**必须等 `expression-constraints` 全绿**，不是
等阶段一。

**落地后需确认（核对清单）：**

- [ ] `ConstraintNode.op` 的分派结构（`expression-constraints` 落地后的真实形态）。漂移要加
      `"sin"`/`"cos"` 两个 op，得先知道 op 是字符串分派还是别的、加在哪。
- [ ] `evaluation.py` 里 `ConstraintNode` 的求值与导数合成挂载点——`sin`/`cos` 的解析导数
      （`cos`/`-sin`）要挂在这里，与其它 op 同一处。验收门是「解析导数 vs 有限差分」，得对着
      约束求值的真实入口写。
- [ ] 漂移展开辅助的落点。spec 说新增「`model/` 下一个漂移声明类型 + 展开辅助（纯数据到约束树
      的转换）」，展开在 `services` 侧完成，`api` 暴露漂移声明本身。要确认约束落地后
      `ConstraintRule` 的构造入口，好让展开辅助产出合法的规则。
- [ ] `SERVICE_SEED_TREE_VERSION`（现状锚点 `services/datasets.py:60`，`spawn_key=(...,)`
      派生子种子）——随机漂移的偏移序列由它派生的确定性子种子在**编译期**物化为 `const` 节点，
      不在求值期调随机数。要确认这个派生接口的调用姿势。
- [ ] `physics/stack.py` `PeriodicSpan` 登记点（现状锚点 `:205`）——漂移周期块要跳过登记退化为
      普通层（漂移与矩阵幂快路径数学上互斥）。若 `nested-periodic-stacks` 也已落地并改了这里，
      要确认两者对同一登记点的改动不冲突。
- [ ] GUI：`gui/structure/dialogs.py` `PeriodicDialog` 增加漂移段——要在
      `nested-periodic-stacks` 把它改成 `QTreeWidget`（若那份先落地）之后接线。

**解锁信号：** `expression-constraints` 全绿，`ConstraintNode` 的 op 分派与
`evaluation.py` 求值/导数挂载点定型。

---

## 全局注意

- 四份的验收门都含「上游机制为空时全量测试逐位不变」——这是每份 plan 的地基测例，写 plan 时
  照阶段一三份的同款 `==` 断言写死。
- 四份都改 codec，全部要有「旧工程文件缺新键也能读」的往返测例（`parameter-priors` 修正 8 的
  教训：codec 从 `__dataclass_fields__` 自动派生键会静默破坏旧文件）。
- 写正式 plan 前，仍按既定方法：对着**落地后的 HEAD** 逐条 fact-check spec，把偏差记成带代码
  证据的编号修正。本清单只降低推依赖的成本，不替代 fact-check。
