# 嵌套周期堆栈设计

## 目标

支持周期块内再含周期块。吸收自 Multifitting。对多层镜类样品（例如"20 × [10 × (Mo/Si)
+ 间隔层]"）有用，当前必须手工摊平成 200 层声明。

## 现有限制

`model/structure.py:194` 的 `PeriodicBlock.layers` 类型是 `tuple[LayerSpec, ...]`，
`:123 _periodic_layers` 断言 `"periodic block layers must be LayerSpec values"`。因此嵌套
在类型层面被明确禁止，不是遗漏。

`StructureComponent = LayerSpec | PeriodicBlock | GradientLayerSpec`（`:207`）已是联合类型，
把 `PeriodicBlock.layers` 放宽到接受 `PeriodicBlock` 即可递归，`physics/stack.py:198
_append_periodic` 的展开循环天然递归（它调用 `_append_layer`，改为调用 `_append_component`）。

## 快路径的处理

`physics/stack.py:205` 只在 `repeats > 1` 时登记 `PeriodicSpan`，而
`physics/derivatives.py:352` 反向遍历 `stack.periodic_spans` 逐个套用矩阵幂切向量。

嵌套后内层 span 完全落在外层 span 内部，两者是包含关系而非并列关系。`_apply_span_tangent`
的 `cursor` 递减逻辑假定 span 之间不重叠，嵌套会破坏这个假定。

**本轮方案：只登记最外层 span，内层展开为普通层。** 外层仍享受 `O(log repeats)` 的矩阵幂
优化，内层退化为线性。理由：外层重复数通常远大于内层（"20 × [10 × ...]"中优化 20 比优化
10 更划算），且这个方案对 `derivatives.py` 零改动，风险最低。

嵌套 span 的完整矩阵幂（内外层各自取幂）是可行的数学优化，但它需要重写 `_apply_span_tangent`
的游标模型，**明确不在本轮范围**。

## 代码边界

- 改 `src/xrr_fitter/model/structure.py`：`PeriodicBlock.layers` 放宽为
  `tuple[LayerSpec | PeriodicBlock, ...]`，`_periodic_layers` 相应放宽，增加嵌套深度上限。
- 改 `src/xrr_fitter/physics/stack.py`：`_append_periodic` 内层调用改为 `_append_component`，
  仅最外层登记 `PeriodicSpan`。
- 改 `src/xrr_fitter/io/codec_common.py`：递归序列化。
- 不改 `physics/derivatives.py`。
- 不改 `GradientLayerSpec` 的嵌套能力（梯度层不得含周期块）。
- 改 `src/xrr_fitter/gui/structure/{dialogs,editor}.py`。
- `api` 侧无需新函数：嵌套块随 `api.set_structure` 整体提交。

## 界面

`StructureEditor._component_item`（`editor.py:241`）已经为 `PeriodicBlock` 生成带子项的树项，
子项由 `_layer_item` 生成。改成递归：子项若是 `PeriodicBlock` 就再调 `_component_item` 的
同一分支，于是树天然显示嵌套层级，"重复"列在每一层都有值。这是本项唯一必需的渲染改动。

`PeriodicDialog`（`dialogs.py:190`）现在用固定 5 列的 `QTableWidget` 收集平铺层，表达不了
嵌套。改为 `QTreeWidget` 加两个按钮："添加层"、"添加子周期块"。后者递归打开一个新的
`PeriodicDialog`，其结果作为子项插入。递归深度在对话框层就拦住（超过上限时按钮禁用并给出
提示），不让用户搭出一个只能在提交时被拒的结构。

`_accept_fields` 仍是一次性构造完整不可变 `PeriodicBlock`，失败落在既有 `periodicDialogError`
标签。既有 `_cell` 的"行不完整"校验（`dialogs.py:271`）迁到树项上，消息里带层级路径而不只是
行号，否则嵌套下"第 2 行不完整"无法定位。

**`top_roughness_a` 的覆盖点要在界面上唯一。** 后端语义是只有整个嵌套结构最顶部的界面能被
覆盖。因此该字段只在最外层 `PeriodicDialog` 出现，子块对话框里不显示——显示了就等于允许用户
设一个会被静默忽略的值。

`top_roughness_a` 的覆盖语义需要明确：`stack.py:202` 现在的条件是
`repeat_index == layer_index == 0`。嵌套后"第一个 cell 的第一层"要递归定义为"最外层第一个
repeat 的最内层第一层"，即只有整个嵌套结构最顶部的那个界面能被覆盖。

## 失败与状态

- 嵌套深度超过上限（建议 3）时报错。深度上限的目的是防止声明爆炸而非技术限制。
- 展开后总层数超过既有上限时报错，复用现有校验。
- 梯度层内出现周期块时报错。
- 嵌套结构的 `_validate_roughness`（`stack.py:237`）邻层约束在展开后照常执行，不因嵌套
  跳过。

## 验证

- 等价性测例：嵌套声明展开出的 `SlabStack` 与手工摊平的等价声明**逐位相等**（thickness、
  sld、roughness 三个数组全部）。这是主验收门。
- Jacobian 等价性测例：嵌套结构的 Jacobian 与手工摊平结构的 Jacobian 逐位相等。这验证
  "只登记最外层 span"没有引入数值差异。
- `top_roughness_a` 覆盖位置测例：嵌套下只有最顶界面被覆盖。
- 深度上限测例。
- 无嵌套时全量测试逐位不变。
- GUI 测例：两层嵌套声明在树中渲染出对应层级，且每层"重复"列有值。
- GUI 测例：达到深度上限时"添加子周期块"按钮禁用。
- GUI 测例：子块对话框不含 `top_roughness_a` 字段。

## 非目标

- 不做嵌套 span 的完整矩阵幂优化（内外层各自取幂）。
- 不支持梯度层内嵌周期块。
- 不改变 `PeriodicSpan` 的字段或 `derivatives.py` 的游标模型。
