# 堆栈厚度与粗糙度漂移设计

## 目标

用 2 到 3 个自由度表达多层膜沿堆栈方向的系统性厚度或粗糙度漂移（线性、随机、正弦）。
吸收自 Multifitting。镀制过程中厚度沿堆栈漂移是真实现象；当前要表达它必须给每层单独开
自由参数，参数量随层数线性膨胀，而漂移本身只有 2 到 3 个自由度。

## 依赖表达式约束

做完 `2026-08-09-expression-constraints-design.md` 后，漂移可以直接用 `ConstraintRule`
表达，几乎零额外成本：

- 线性漂移 `t_i = t_0 * (1 + k*i)` 是 `mul` 与 `add` 节点的组合。
- 正弦漂移 `t_i = t_0 * (1 + a*sin(2*pi*i/p + phi))` 需要 `sin` 节点。

因此本设计只做两件事：给节点集合补 `sin` 与 `cos`，以及提供一个把漂移意图展开成约束树的
构造辅助（避免用户手工搭 N 层的树）。**不新增独立的漂移机制。**

这是把它排在表达式约束之后的理由。

## 随机漂移的特殊处理

线性与正弦是确定性函数，可直接进约束树。随机漂移不是——它需要一个每层不同但可复现的
偏移序列。

方案：随机漂移的偏移序列由 `SERVICE_SEED_TREE_VERSION` 派生的确定性子种子生成，序列在
结构编译期一次性物化为 `const` 节点，之后完全确定。**不在求值期调用随机数生成器。**

这保证同一工程文件每次运行得到逐位相同的漂移序列，与项目的确定性承诺一致。对比：
Multifitting 的种子在三处硬编码为时钟值（`launcher.cpp:17-19`、`Fitting::randomize_Position()`、
`fitting_swarmops.cpp:11`），且 `gsl_rng_env_setup()` 在 `gsl_rng_alloc()` 之后调用又被显式
`gsl_rng_set()` 覆盖，`GSL_RNG_SEED` 也不生效，其随机行为不可复现。

随机漂移的自由参数是**幅度**（一个标量），不是序列本身。

## 关键取舍：周期快路径失效

`physics/derivatives.py:360 _normal_cells_repeat` 要求重复 cell 的参数值逐位相同才走矩阵
幂快路径：

```python
cells = values[start:stop].reshape(count, span.layer_count, values.shape[1])
return np.array_equal(cells, np.broadcast_to(cells[0], cells.shape))
```

`:371 _span_normal_tangents_repeat` 对切向量有同样要求。漂移的本质就是让每个 cell 不同，
因此**漂移与周期块的矩阵幂优化互斥**。

处理方式：漂移作用于周期块时，`_append_periodic` 不再登记 `PeriodicSpan`（`stack.py:205`），
展开为普通层序列。代价是 Jacobian 从 `O(log repeats)` 退化为 `O(repeats)`。

这个代价必须在文档与 GUI 中明示，让用户知道给 50 周期的多层镜加漂移会显著变慢。不隐藏、
不自动放弃漂移。

## 代码边界

- 改 `src/xrr_fitter/model/parameters.py`：`ConstraintNode.op` 增加 `"sin"`、`"cos"`。
- 改 `src/xrr_fitter/evaluation.py`：两个新 op 的求值与导数。
- 新增 `src/xrr_fitter/model/` 下一个漂移声明类型与展开辅助（纯数据到约束树的转换）。
- 改 `src/xrr_fitter/physics/stack.py`：漂移周期块跳过 `PeriodicSpan` 登记。
- 不新增求值期随机数调用。
- 不改 `physics/derivatives.py` 的快路径判据本身。
- 扩 `src/xrr_fitter/api.py`：漂移声明的 validate/set 对。漂移最终落成 `ConstraintRule`，
  但界面不该逼用户手搭那棵树，所以 API 暴露的是漂移声明本身，展开在 `services` 侧完成。
- 改 `src/xrr_fitter/gui/structure/dialogs.py`：`PeriodicDialog` 增加漂移段。

## 界面

漂移是周期块的属性，编辑入口在 `PeriodicDialog` 里，不在参数面板的约束页。理由：用户的心智
模型是"这个多层镜的厚度逐周期递增"，而不是"给第 k 层写一条表达式"。约束树是实现手段。

对话框加一段：`QComboBox` 选漂移类型（无 / 线性 / 正弦 / 随机），选"无"时不展开任何字段且
不产生 `ConstraintRule`。选中后展开该类型的字段（线性：每周期增量；正弦：幅度与周期数；
随机：标准差与种子偏移），并用另一个下拉选作用目标（厚度 / 粗糙度）。

**性能代价必须在勾选时就说。** 设计正文已定"不隐藏、不自动放弃漂移"，界面据此做：选中非"无"
的漂移类型后，对话框内立即显示当前 `repeats` 值下的告知文本，说明该块将退出矩阵幂快路径、
Jacobian 复杂度从 `O(log repeats)` 变为 `O(repeats)`。文本随 `repeats` 编辑实时更新，因为
50 周期和 5 周期的实际代价差一个量级。

**随机漂移的种子要可见且可复现。** 偏移在编译期由确定性子种子物化为 `const` 节点。界面必须
显示实际使用的种子来源（工程种子 + 该块的偏移量），否则用户无法解释两个工程为何得到不同
偏移序列。种子字段本身可编辑，但不提供"随机生成"按钮——那会重新引入不可复现性。

`StructureEditor` 的树在周期块行的 tooltip 里标注漂移类型与作用目标，不加新列。

## 失败与状态

- 漂移导致任一层厚度非正时报错，不静默 clip。
- 漂移导致粗糙度超过 `_validate_roughness`（`stack.py:237`）的邻层约束时报错，复用既有
  校验而非绕过。
- 漂移作用于非周期、非连续的层集合时报错（漂移只对有序连续层有意义）。
- 随机漂移缺少确定性种子上下文时报错，不退化为不可复现的随机。

## 验证

- 线性漂移展开测例：展开出的约束树求值结果与手写公式逐位相等。
- `sin`/`cos` 的解析导数 vs 有限差分。
- 随机漂移确定性测例：同一工程文件两次编译得到逐位相同的偏移序列。
- 周期快路径失效测例：漂移周期块的 `SlabStack.periodic_spans` 为空，且其 Jacobian 与逐层
  展开的等价结构逐位相等。这是正确性的关键断言——快路径与慢路径必须给同一答案。
- 无漂移时全量测试逐位不变。
- GUI 测例：漂移类型选"无"时提交出的项目不含 `ConstraintRule`。
- GUI 测例：选中漂移后性能告知文本可见，且 `repeats` 从 5 改到 50 后文本随之更新。
- GUI 测例：随机漂移对话框显示种子来源且不含"随机生成"按钮。

## 非目标

- 不新增独立于表达式约束的漂移机制。
- 不做漂移与周期矩阵幂优化的共存（数学上不可能）。
- 不做漂移参数的自动识别。
- 不支持二维漂移或横向不均匀性。
