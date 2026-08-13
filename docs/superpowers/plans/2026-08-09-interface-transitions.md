# 界面过渡函数族实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把界面过渡从只有 erf 扩展为六种可归一化线性组合的过渡核（erf、linear、exponential、tanh、sine、step），每个分支有独立宽度与权重；非 erf 过渡通过微切片参与反射率计算并保有完整解析导数；`transition=None` 的既有行为与 HEAD 逐位相同。

**Architecture:** 过渡是 `LayerSpec` 的一个可选字段。带 `transition` 的层在 `physics/stack.py` 展开期把入射界面切成微切片，每片 SLD 由过渡核在该片中心求值——这与 `GradientLayerSpec` 走 `_append_gradient` 的既有机制同型。展开后的 `SlabStack` 里过渡形状已编码进 SLD 序列，因此 Parratt 反射率与 `sld_depth_profile` 剖面**共用同一个真值源**，不存在图上画一种、拟合算另一种的可能。

过渡的形状坐标（kind、权重、宽度、切片粒度）**不产生可拟合参数**，因此切片数在整轮拟合中恒定，每片的切线是既有母层切线的常系数线性组合，解析 Jacobian 完整可用，不需要任何降级或报错（见修正 2、3）。

**Tech Stack:** Python 3.12、NumPy 2.x、SciPy 1.14+、PySide6 6.8+、pytest 8.3+。不新增依赖。

**Design source:** `docs/superpowers/specs/2026-08-09-interface-transitions-design.md`

---

## 对 spec 的修正

写这份 plan 时逐条核对了 spec 与 HEAD 代码，有十六处不符，其中修正 1 是自相矛盾、修正 2/3 推翻了 spec 的一整节要求、修正 15 是一条无法成立的验收标准。**执行时以本节为准，不以 spec 为准。**

### 修正 1（致命）：spec 的两条代码边界互斥，只能取微切片那条

spec 同时要求「改 `physics/sld_profile.py`：`_transition` 改为按 `kind` 分派」和「改 `physics/stack.py`：非 erf 过渡的层走微切片展开」。这两条不能同时成立。

`sld_depth_profile(stack: SlabStack)`（`physics/sld_profile.py:53`）消费的是**已经展开完的** `SlabStack`，其字段只有 `thickness_a` / `sld_a2` / `roughness_a` / `periodic_spans`（`model/structure.py:344`）——**没有任何 kind 信息**。调用点是 `fit/candidates.py:743`，传的是 `evaluation.expanded_stack`。层一旦在 `stack.py` 里展开成微切片，`sld_profile.py` 拿到的就是一串锐利小台阶，它无从得知原始 `kind`，按 kind 分派的代码永远不会被触发。

**取微切片那条，`physics/sld_profile.py` 完全不改。** 理由：微切片让剖面与反射率共用一个真值源。只改 `sld_profile.py` 的另一条路会得到「剖面图画 tanh、拟合仍用 erf」——这比不做这个功能更糟，因为它填不了 spec 自己声明的动机（“扩散界面用 erf 在物理上就是错的”），却让用户以为填上了。

六个核函数因此落在新模块 `physics/transitions.py`，由 `stack.py` 与 `physics/geometry.py` 调用，不动 `sld_profile.py:23` 的 `_transition`。

### 修正 2（推翻 spec 一整节）：不存在"关闭解析梯度"这个动作

spec 的失败清单写「非 erf 过渡与解析 Jacobian 同时启用时报错……提示改用 erf 或**关闭解析梯度**」。仓库没有关闭解析梯度的开关：`fit/local_search.py:113` 与 `fit/joint_pipeline.py:137` 都是无条件 `jac=jacobian`。`fit/global_search.py:552` 的 `differential_evolution` 本就不用梯度，但局部精修必然用。这条提示指向一个用户无法执行的动作。

`model/fitting.py:201` 的 `jacobian_version: str = "analytic-v1"` 看着像开关，其实只是写进导出与工程文件的溯源字符串（消费点仅 `io/export_tables.py:308,618` 与 `io/codec_declarations.py:124,143`），改它不影响任何数值路径。

### 修正 3（推翻 spec 一整节）：过渡的解析导数完整可用，不需要报错

spec 说「微切片路径的解析导数不在本轮范围」，并把「非 erf 与解析 Jacobian 并用时报错」列为验收项。核对后这个前提不成立，两条都删掉。

首先，微切片**已经有**解析导数：`physics/geometry.py:373` 的 `append_gradient` 为 `GradientLayerSpec` 的微切片提供完整切线（厚度均分、SLD 线性插值、只有第一界面带粗糙度切线）。

其次，过渡的切线是既有量的常系数线性组合。关键是**过渡的形状坐标不产生可拟合参数**（kind、权重、分支宽度、切片粒度都不进 `_layer_definitions`），所以：

- 切片数 `N = ceil(W / microslab_max_a)`，其中 `W = max(branch.thickness_a)`，两者都是常数 → 整轮拟合中 N 恒定，Jacobian 的行数不变。
- 第 k 片厚度 `W / N` 是常数 → 切线是零向量。
- 第 k 片 SLD 是 `sld_prev + f(t_k) * (sld_layer - sld_prev)`，`f(t_k)` 是常数 → 切线 `(1 - f(t_k)) * J[sld_prev] + f(t_k) * J[sld_layer]`。`J[sld_layer]` 就是既有的 `_layer_sld_jacobian(...)`，`J[sld_prev]` 就是 builder 手里的 `self.sld[-1]`（primal 侧对称地是 `state.sld[-1]`）。
- 层体剩余厚度 `T - W` 的切线等于 `value_jacobians[f"{prefix}.thickness_a"]`，与 HEAD 逐位相同。
- 所有过渡界面的粗糙度锁定在 0（修正 4），锁定坐标的 `value_jacobians` 本就是零向量，直接取用即可，不必特判。

结论：`physics/geometry.py` 加一个与 `append_gradient` 同型的 `append_transition_layer`，解析路径就完整了，**没有任何降级、没有任何报错**。spec 那条报错以及它在 GUI 里的提前告知（"该层将走微切片且不能与解析梯度并用"）一并删除——照 spec 实现会在界面上写一句不实的警告。

真正需要防的是形状对齐：`geometry.py:405` 的 `append_component` 若仍对带过渡的层走 `append_layer`，primal 侧 N+1 行、切线侧 1 行，`finish()`（`geometry.py:449`）会抛 `RuntimeError("expanded structure Jacobian mapping mismatch")`。这个断言已存在，是 Task 3 的天然 RED 条件。

### 修正 4：过渡与 `roughness_a` 会把同一界面展宽两次，spec 未处理

`LayerSpec.roughness_a` 是该层**入射界面**的粗糙度（`geometry.py:303` 的 `append_layer` docstring：“the roughness of its incident interface”），在 Parratt 里通过 Névot-Croce 因子 `exp(-2*upper*lower*roughness**2)`（`physics/parratt.py:48`）展宽这个界面。若同一层再带 `transition`，微切片斜坡又展宽同一个界面一次，总宽度既不是两者之和也不是任一者，物理上无意义。

**处理：过渡是界面宽度模型的替代而非叠加，`roughness_a` 必须恒为 0。但落点不能是构造期。**

`roughness_a` 是自由参数（`fit/parameters.py:125` 无条件以 `locked=False` 发出），而 `rebuild_structure` 每轮用 `_replace_layer`（`physics/stack.py:56`）以 `roughness_a=values[f"{prefix}.roughness_a"]` **重新构造** `LayerSpec`。若把「`transition is not None` 时 `roughness_a == 0.0`」写进 `LayerSpec.__post_init__`，优化器第一次把该坐标推离 0 就会在结构重建时抛 `ValueError`，整轮拟合崩在残差函数里。**这是 spec 与我初稿共有的一个致命错误，实施者不得按构造期校验实现。**

正确落点是两处，缺一不可：

1. **参数发出端**：`_layer_definitions` 在 `layer.transition is not None` 时，把 `roughness_a` 发成 `initial=0.0, lower=0.0, upper=0.0, locked=True`。锁定坐标不进 `_variables`（`fit/problem.py:78` 按 `not definition.locked` 过滤），于是 `values[...]` 恒为 0.0、该行 Jacobian 恒为零向量，`rebuild_structure` 重建出的 `LayerSpec` 永远合法。
2. **编译期兜底**：`apply_parameter_settings`（`fit/parameters.py:509`）允许外部 `ParameterSetting` 覆写 `locked`，所以发出端的锁可以被解开。新增 `validate_transition_modes(definitions, structure)`，位置与写法照 `validate_instrument_modes`（`:540`）**逐条对齐**——复用同一个 `_require_locked_value`（`:529`）、同样在 `compile_fit_problem` 里紧跟 `validate_instrument_modes` 之后调用（`fit/problem.py:174` 之后）。必须在 `apply_parameter_settings` 之后，否则覆写绕过校验。

`stage_parameter_settings`（`:642`）对 `definition.locked` 的分支原样回写 `locked=True` 且不读 `current_values`，所以这把锁能贯穿 `compile_stage_problem` 的分阶段拟合，不需要额外处理。

构造期只保留一条**不涉及自由坐标**的断言：`transition is not None` 时 `roughness_a` 的**声明值**为 0.0。它只拦住用户手写的非法初始声明，不会被 `rebuild_structure` 触发——因为锁定后重建传入的就是 0.0。这条放 `LayerSpec.__post_init__`，满足 `MODEL_ALLOWED["structure"] = set()`。

副作用（对修正 3 有利）：过渡界面的粗糙度切线恒为零向量，Névot-Croce 因子恒为 1，非 erf 过渡自动不叠加 Névot-Croce 展宽，无需在 `parratt.py` 里加任何分支。

### 修正 5：核签名不能是 `(depth, interface, sigma)`

spec 的 `TransitionBranch` 有 `thickness_a` 而无 `sigma`，但它给的核签名沿用 `_transition(depth, interface, sigma)`，并把「每个核在 `sigma == 0.0` 时退化为阶跃」列为验收。`sigma` 在新数据类型里根本不存在，这条验收无从落地。

**处理：核定义在归一化坐标 `t ∈ [0, 1]` 上，契约是 `f(0) == 0`、`f(1) == 1`、单调不减。** 零宽界面由 `step` 核显式表达，不再靠 `sigma == 0.0` 退化。六个核的形状常数必须具名并写进 `docs/algorithm.md`，否则「erf 过渡」是一句没有数值含义的话：

| kind | `f(t)` | 形状常数 |
| --- | --- | --- |
| `erf` | `0.5 * (1 + erf(c*(2t-1)/√2) / erf(c/√2))` | `ERF_HALF_WIDTH_SIGMAS = 2.0`（分支宽度对应 ±2σ） |
| `linear` | `t` | 无 |
| `tanh` | `0.5 * (1 + tanh(c*(2t-1)) / tanh(c))` | `TANH_HALF_WIDTH = 2.0` |
| `sine` | `0.5 * (1 - cos(πt))` | 无 |
| `exponential` | `(1 - exp(-k*t)) / (1 - exp(-k))` | `EXPONENTIAL_RATE = 4.0` |
| `step` | `where(t < 0.5, 0.0, 1.0)` | 无 |

`erf` 与 `tanh` 的分母是归一化项，保证端点精确取 0 与 1，而不是近似取。这是修正 6 的合成规则能精确收敛的前提。

### 修正 6：多分支不同宽度的合成规则 spec 未定义

spec 给每个分支独立 `thickness_a`，却没说宽度不同的分支怎么合成一条剖面。

**处理：过渡区总宽 `W = max(branch.thickness_a)`。** 每个分支在自己的宽度上按 `t_b = clip(z / branch.thickness_a, 0, 1)` 求值，再按归一化权重加权求和。因为每个分支在自己末端已精确到 1（修正 5），加权和在 `z = W` 处精确为 `Σw_b = 1`，在 `z = 0` 处精确为 0——spec 要求的「两端仍精确趋 0 与 1」由此成为恒等式而非数值巧合。单调性由各分支单调性直接给出。

### 修正 7：`InterfaceTransition` 缺切片粒度字段

微切片展开需要切片宽度上限，spec 的 `InterfaceTransition` 只有 `branches`。

**处理：加 `microslab_max_a: float = 1.0`，与 `GradientLayerSpec.microslab_max_a`（`model/structure.py:174`）同名同默认值，校验区间取 `(0, W]`（`W` 见修正 6），与 `model/structure.py:186` 的 `(0,thickness]` 同型。**

### 修正 8：spec 要求的「微切片数量上限」在仓库里不存在

spec 说「微切片数量超过上限时报错」。仓库**没有**任何切片数量上限：`GradientLayerSpec` 只校验 `microslab_max_a in (0, thickness]`（`model/structure.py:186`），这只保证 `N ≥ 1`，上界完全不受限。

**处理：新增常量 `MAX_TRANSITION_SLABS = 512`，在 `InterfaceTransition.__post_init__` 里校验 `ceil(W / microslab_max_a) <= 512`。** 取 512 的理由：过渡可挂在每一层上，风险是各层切片数的累加式膨胀而非单层；512 片对应 25 Å 过渡区切到 0.05 Å，远超任何物理需要，同时把单层最坏情况钉死。这个上限只约束过渡，不改 `GradientLayerSpec` 的现状。

### 修正 9：`limit_thickness` 必须同步追加，否则薄切片会被粗糙度上限误杀

`stack.py:262` 的 `_validate_roughness` 拿的是 `state.limit_thickness` 而不是 `state.thickness`，上限是 `0.49 * min(相邻厚度)`，且 `sigma >= limit` 即判非法。`_append_gradient`（`stack.py:214`）为每片追加的是**母层总厚**而非切片厚度，正是为了让微切片不受这个上限约束。spec 通篇未提 `limit_thickness`。

**处理：过渡微切片同样为每片追加母层 `thickness_a` 到 `limit_thickness`，追加次数与 `thickness` 完全一致。** 两个症状各自独立：追加次数不一致会让两个列表错位，`_validate_roughness` 读到别的界面的邻居厚度；追加切片厚度而非母层厚度则会让紧邻过渡区的下一个界面的粗糙度上限缩到 `0.49 * 切片厚度`，把本来合法的粗糙度判非法。

### 修正 10：过渡总宽超过层厚会让层体厚度变负，spec 未校验

过渡区占用的是层的入射侧，层体剩余厚度是 `T - W`（`T = layer.thickness_a`，`W` 见修正 6）。spec 没有任何约束阻止 `W > T`。此时 `_validate_thickness`（`model/structure.py:261`）只查非负，会抛 `slab thickness must be finite and nonnegative`——一条完全指不到病因的报错，而且要等到展开期才抛。

**处理：`LayerSpec.__post_init__` 里校验 `W <= thickness_a`，报错文案点名层名与两个数值。** 允许 `W == T`（过渡吃掉整层，层体退化为零厚切片，与 `_validate_thickness` 的非负契约相容）。

**由此定死一条贯穿三条展开路径的不变量，实施者必须逐字遵守：** 带过渡的层**无条件**展开成 `N + 1` 行，`N = transition_slab_count(W, microslab_max_a)` 个过渡切片加**恰好一个**层体切片，层体厚度 `T - W`（`W == T` 时为 0.0）。不要写「`T - W` 为 0 时省略层体切片」这类优化——那会让展开行数依赖数值比较，三处实现各自判断一次，任何一处的浮点边界判断不同就直接触发两条 `RuntimeError` 之一，而且是只在特定参数值下才复现的间歇故障。行数只由 `N` 决定，`N` 只由构造期已冻结的 `W` 与 `microslab_max_a` 决定。

### 修正 11：`PeriodicBlock` 内的层本轮不能带过渡

`_append_periodic`（`stack.py:200`）记录的 `PeriodicSpan.layer_count = len(block.layers)`，`model/structure.py:340` 的 `_validate_spans` 按这个计数逐段比对重复性。层若展开成 N+1 行，`layer_count` 就错了，`_validate_media_repetition` 会抛 `periodic span thickness does not repeat`。spec 完全没覆盖这个交互。

**处理：`_periodic_layers`（`model/structure.py:123`）里拒绝带 `transition` 的层，报错说明本轮限制。** 让 span 计数跟随展开是可做的，但要同时改 `_validate_roughness_repetition`（`:312`）与 `top_roughness_a` 的覆盖语义（`stack.py:203` 只在 `repeat_index == layer_index == 0` 时覆盖），超出本轮范围。

### 修正 12：序列化落点不是 `codec_common.py`

spec 说「改 `src/xrr_fitter/io/codec_common.py`：序列化」。`LayerSpec` 的实际编解码在 **`io/codec_declarations.py:183` 的 `_layer_to_dict` 与 `:194` 的 `_layer_from_dict`**。`codec_common.py` 只提供 `_mapping`（`:80`）与 `OPTIONAL_FIELDS`（`:19`）白名单。

`_mapping` 对 `extra` 键**严格报错**（`codec_common.py:90-94`），所以 `transition` 必须作为 `optional` 集合传入 `_layer_from_dict` 的 `_mapping` 调用，不能加进 required——否则旧工程文件（无该键）直接因 `missing` 报错。仓库里已有先例：`project_codec.py:173` 的 `optional={"dock_state"}`。

**`OPTIONAL_FIELDS` 的语义与 spec 的理解相反，实施者必须注意：** 它不是「字段可缺省」白名单，而是 `_allows_null`（`codec_common.py:117`）用的「该键的值允许为 JSON `null`」白名单，由 `_validate_nulls`（`:127`）消费。关键在于 `_validate_nulls` 在**读路径**（`project_codec.py:446`）与**写路径**（`:519`，`project_to_bytes` 内）都会跑。

由此定死写法：**`_layer_to_dict` 在 `value.transition is None` 时完全不发 `transition` 键，只在非 `None` 时发。** 不要发 `"transition": None`——那会在 `project_to_bytes` 里被 `_validate_nulls` 判 `required project value is null: ...transition` 而直接拒绝写盘，且因为绝大多数层都无过渡，这个 bug 会在第一次保存任何工程时就触发。采取条件发键后，`OPTIONAL_FIELDS` **不需要**登记 `transition`；`branches` 是嵌套数组也不涉及 `null`，同样不需要登记。也就是说 `codec_common.py` 这个文件本轮**只需**改 `_layer_from_dict` 侧对 `optional` 的传参方式，不需要动任何白名单常量。

### 修正 13：kind 名字集合必须在 `model` 与 `physics` 各存一份

`MODEL_ALLOWED["structure"] = set()`，`model/structure.py` 不能 import `physics.transitions`。所以合法 kind 集合在 `model/structure.py`（构造期校验用）与 `physics/transitions.py`（分派用）里各有一份字面量。

**处理：加一条测例断言两份集合相等**，放在能同时 import 两侧的 `tests/unit/physics/test_transitions.py`。不加这条断言，两份集合迟早分叉，症状是构造期放过一个 kind、展开期 `KeyError`。

### 修正 14：多分支表格需要三列

spec 说小表格列为 `("过渡类型", "权重")`。`TransitionBranch` 有三个字段，两列填不进 `thickness_a`。

**处理：列为 `("过渡类型", "权重", "宽度 (nm)")`。** 照 `PeriodicDialog.table = QTableWidget(2, 5)`（`dialogs.py:212`）的用法建 `QTableWidget(2, 3)`，nm→Å 的 `* 10.0` 换算照 `LayerDialog._accept_fields` 既有写法。

### 修正 15：`LayerDialog` 没有回显路径，spec 的回显验收无从落地

spec 要求 GUI 测例「权重 `(2, 2)` 提交后回显 `(0.5, 0.5)`」。`LayerDialog.__init__` 的签名是 `(parent, *, commit_layer)`，**不接受待编辑的层**；`_accept_fields` 成功后直接 `accept()` 关闭对话框。这是纯新增对话框，不存在"回显"这回事。

**处理：改为断言 `commit_layer` 收到的 `LayerSpec.transition.branches` 权重已归一化为 `(0.5, 0.5)`。** 构造期归一化的行为照 spec 保留，只是验收落在提交出的数据上。

### 修正 16：`api.py` 必须改

spec 说「`api` 侧无需新函数」。函数确实无需新增，但 `TransitionBranch` 与 `InterfaceTransition` 必须 re-export——`gui` 只能 import `{gui, api}`（`ALLOWED`），`LayerDialog` 通过 `api.LayerSpec` 构造层，同理需要 `api.InterfaceTransition` 才能构造过渡。漏掉这一步的症状是 `gui/structure/dialogs.py` 无法通过架构测试。

### 修正 17：过渡内部界面必须复用 `GRADIENT_INTERNAL_INTERFACE` 哨兵

微切片有三条平行的展开路径，都要处理过渡，不止 primal 与 Jacobian 两条：

1. `physics/stack.py` 的 `_append_*`——primal SLD 序列（修正 1/9）。
2. `physics/geometry.py` 的 `_StackJacobianBuilder`——解析切线（修正 3）。
3. `physics/geometry.py` 的 `expand_geometry` / `_expanded_interface_names`——**动态粗糙度上限**，被 `evaluation.py:369` 的 `_roughness_dynamic_uppers` 消费。

第 3 条最容易漏。`_expanded_interface_names`（`geometry.py:32`）为每个展开界面回填一个源坐标名，`GradientLayerSpec` 的内部数值界面回填 `GRADIENT_INTERNAL_INTERFACE` 哨兵（`geometry.py:40`），`evaluation.py:349` 的 `_is_public_interface` 据此把它们**排除**在动态上限计算之外。过渡层展开出的 N 个内部界面同理是数值细分而非物理界面。

**处理：`_expanded_interface_names` 对带过渡的层回填 `[f"{prefix}.roughness_a", *(GRADIENT_INTERNAL_INTERFACE,) * N]`——共 `N + 1` 个标签，对应修正 10 定死的 `N + 1` 个展开行；`_append_layer_geometry` 分流出过渡分支，追加同样的 `N + 1` 个厚度行。** 第一个标签是该层真实的入射界面（虽然其粗糙度已被锁 0，它仍拥有一个公共参数定义，必须保留在动态上限计算内，否则 `dynamic[name]` 缺键会让 `values_and_jacobians` 抛 `KeyError`）；其后 `N` 个是切片之间的内部数值界面。

漏掉第 3 条的症状：`expand_geometry` 末尾 `len(names) != len(thickness) - 1` 的对齐校验抛 `RuntimeError("expanded geometry interface mapping mismatch")`。这是 Task 3 的第二个天然 RED 条件——两条 `RuntimeError` 各自钉住一条展开路径。

### 保留不动的 spec 判断

以下核对后与代码一致，照 spec 执行：不改 `physics/parratt.py` 与 `physics/derivatives.py` 的 Névot-Croce 实现；非 erf 过渡不套用 Névot-Croce（过渡界面粗糙度锁 0，Névot-Croce 因子退化为 1，自动成立）；`transition` 默认 `None` 使既有工程零迁移；权重非正或全零、`thickness_a` 非正在构造期报错；归一化在构造期完成；`StructureEditor` 树不加新列，在“粗糙度 (nm)”列（`TREE_HEADERS` 索引 4，`editor.py:25`）的 tooltip 里标注过渡类型。

**一个可接受的后果要写进文档：** `sld_depth_profile` 在 `np.all(roughness_a == 0.0)` 时走 `_sharp_profile`（`sld_profile.py:57`）。过渡层的界面粗糙度锁 0（修正 4），若整个结构其余界面也都是 0，剖面图会画成 N 级台阶而非平滑曲线。这是微切片的真实离散化，不是缺陷——台阶细到 `microslab_max_a` 就是模型的真实分辨率。`docs/algorithm.md` 要写明这一点，免得被当成 bug 修。

---

## Global Constraints

- **不改 `src/xrr_fitter/physics/sld_profile.py`。** 见修正 1。
- 不改 `physics/parratt.py` 与 `physics/derivatives.py` 的 Névot-Croce 实现与任何既有数值行为。
- `transition=None` 的路径必须与 HEAD **逐位相同**。这是本计划的核心兼容断言，每个 Task 都要维持。
- 遵守 `tests/architecture/test_dependency_rules.py` 的 `ALLOWED`：`model:{model}`、`physics:{physics,model}`、`io:{io,model}`、`api:{model,services}`、`gui:{gui,api}`。
- `MODEL_ALLOWED["structure"] = set()`：`model/structure.py` **不能** import `physics.transitions`，kind 集合与所有构造期校验必须在该文件内自洽。
- 不新增 `PACKAGE_EDGE_EXCEPTIONS` 条目。`physics/transitions.py` 只 import `numpy`/`scipy`，最省事也最稳。
- 禁止 `pytest.skip` / `xfail` / 条件收集：`tests/outcome_gate.py` 会因 `skipped`/`xfailed`/`xpassed`/`deselected` 让整轮失败。
- 新增测试文件落在既有整目录注册下（`tests/unit/physics`、`tests/unit/model`、`tests/unit/io`、`tests/gui`），**本计划不需要改 `tools/verify_registry.py`**。
- 测试模块名不得以父目录名开头（`tests/architecture/test_naming_rules.py`）：`tests/unit/physics/` 下用 `test_transitions.py` 而非 `test_physics_transitions.py`。
- `tests/conftest.py` 必须保持为仓库唯一 `conftest.py`。
- 新代码必须过 `tools/check_radon.py`（单块 CC ≤ 10、单文件平均 CC ≤ 5.0、MI 级别 A、仓库均值 ≤ 5.0）。六核分派与展开逻辑容易超标，按各 Task 给出的函数边界拆分。
- 跑测试必须带 `--import-mode=importlib`，解释器用 `.venv/bin/python`（只有它能 `import xrr_fitter`）。`tools/verify.py` 在本地会因 repo 内 `.venv` 被 `check_hygiene.py` 判 “generated directory inside repository” 而失败，直接调 pytest。
- 用户可见文案用中文，长度单位在界面上一律 nm、在数据层一律 Å，换算照 `LayerDialog._accept_fields` 的 `* 10.0`。
- 不 stage 或修改 `.claude/` 与仓库根的 probe 文件。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `src/xrr_fitter/model/structure.py` | 新增 `TRANSITION_KINDS`、`MAX_TRANSITION_SLABS`、`TransitionBranch`、`InterfaceTransition`；`LayerSpec` 加 `transition` 字段与三条交叉校验；`_periodic_layers` 拒绝带过渡的层。 |
| `src/xrr_fitter/physics/transitions.py` | 六个归一化核、形状常数、`transition_fractions()` 切片中心求值、`transition_slab_count()`。 |
| `src/xrr_fitter/physics/stack.py` | `_append_layer` 分流出 `_append_transition_layer`，primal 侧微切片展开与 `limit_thickness` 同步。 |
| `src/xrr_fitter/physics/geometry.py` | `_StackJacobianBuilder.append_transition_layer`；`append_component` 分派；`_expanded_interface_names` 补齐过渡界面标签。 |
| `src/xrr_fitter/fit/parameters.py` | `_layer_definitions` 对带过渡的层把 `roughness_a` 发成锁定 0；新增 `validate_transition_modes`（修正 4）。 |
| `src/xrr_fitter/fit/problem.py` | `compile_fit_problem` 调用 `validate_transition_modes`（修正 4）。 |
| `src/xrr_fitter/io/codec_declarations.py` | `_layer_to_dict` 条件发 `transition` 键；`_layer_from_dict` 以 `optional={"transition"}` 读取。**不改 `codec_common.py`**，见修正 12。 |
| `src/xrr_fitter/api.py` | re-export `TransitionBranch`、`InterfaceTransition`。 |
| `src/xrr_fitter/gui/structure/dialogs.py` | `LayerDialog` 加 kind 下拉与三列分支表。 |
| `src/xrr_fitter/gui/structure/editor.py` | 粗糙度列 tooltip 标注过渡类型。 |
| `tests/unit/model/test_structures.py` | 构造期校验与归一化（既有文件，`LayerSpec`/`GradientLayerSpec` 校验测试已在此）。 |
| `tests/unit/physics/test_transitions.py` | 六核契约、两份 kind 集合一致、合成规则（新增文件）。 |
| `tests/unit/physics/test_stack_expansion.py` | 展开形状、`limit_thickness`、逐位不变（既有文件）。 |
| `tests/unit/test_evaluation.py` | 切线与 primal 对齐、有限差分交叉验证（既有文件）。 |
| `tests/unit/fit/test_problem_compilation.py` | 粗糙度锁定与解锁报错（既有文件，整目录注册）。 |
| `tests/unit/io/test_project_codec.py` | 往返与旧文件兼容（既有文件）。 |
| `tests/gui/test_structure_editor.py` | 对话框提交与归一化断言（既有文件，`LayerDialog` 测试已在 `:261`/`:286`）。 |
| `docs/algorithm.md` | 六核公式与形状常数、合成规则、粗糙度替代语义、`_sharp_profile` 台阶后果。 |

---

## Tasks

### Task 1: 六个归一化核与形状常数

**Files:**
- Create: `src/xrr_fitter/physics/transitions.py`
- Create: `tests/unit/physics/test_transitions.py`

**Interfaces:**
- Produces: `TRANSITION_KINDS: frozenset[str]`、六个形状常数、`transition_profile(kind: str, t: np.ndarray) -> np.ndarray`、`transition_slab_count(width_a: float, microslab_max_a: float) -> int`。
- Preserves: 本任务不 import `model`，也不被任何既有模块 import，纯新增叶子模块。
- Removes: 无。

**为什么先做这一层：** 六核是唯一有闭式契约可单独验证的部分（`f(0)==0`、`f(1)==1`、单调不减），先钉死数值行为，后面三条展开路径都只是搬运它的输出。这一层错了，后面所有对齐测试都会以错误的基线通过。

**复杂度约束：** 六核分派若写成一个 `if/elif` 链，单块 CC 直接 7 起、加上校验会逼近 10。改用模块级 `dict[str, Callable]` 分派表，每个核一个独立小函数，分派函数 CC 保持 2。

 - [x] **Step 1: 写失败的核契约**

  在 `tests/unit/physics/test_transitions.py` 写这些测例，全部必须失败（模块不存在）：

  - `test_every_kind_maps_zero_to_zero_and_one_to_one`：对 `TRANSITION_KINDS` 里每个 kind，`transition_profile(kind, np.array([0.0, 1.0]))` 精确等于 `[0.0, 1.0]`。用 `==` 而非 `pytest.approx`——`erf`/`tanh` 的归一化分母就是为了让端点精确，近似断言会放过漏掉分母的实现。
  - `test_every_kind_is_monotone_nondecreasing`：`t = np.linspace(0, 1, 257)`，断言 `np.all(np.diff(f) >= 0.0)`。
  - `test_step_kind_is_a_sharp_half_width_jump`：`transition_profile("step", np.array([0.0, 0.49, 0.5, 1.0]))` 等于 `[0.0, 0.0, 1.0, 1.0]`。这条钉住修正 5 的「零宽界面由 `step` 显式表达」而非靠 `sigma == 0.0` 退化。
  - `test_linear_kind_is_the_identity`：`transition_profile("linear", t)` 与 `t` 逐位相等。
  - `test_erf_kind_matches_the_documented_half_width_constant`：以 `ERF_HALF_WIDTH_SIGMAS` 手算 `t=0.25` 的期望值并断言。这条防止有人换了常数却没改文档。
  - `test_unknown_kind_is_rejected`：`transition_profile("gaussian", t)` 抛 `ValueError`，消息含 `"gaussian"`。
  - `test_slab_count_is_at_least_one_and_ceils`：`transition_slab_count(10.0, 3.0) == 4`、`transition_slab_count(2.0, 2.0) == 1`、`transition_slab_count(1.0, 4.0) == 1`。最后一条钉住宽度小于切片上限时不返回 0。
  - `test_slab_count_rejects_nonpositive_inputs`：`width_a` 或 `microslab_max_a` 为 0、负数、`nan`、`inf` 时抛 `ValueError`。

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/unit/physics/test_transitions.py --import-mode=importlib -q
  ```

  必须是收集期 `ModuleNotFoundError: xrr_fitter.physics.transitions`。若是 `ImportError` 指向别的名字，说明测试文件 import 写错了，先修测试。

 - [x] **Step 3: 实现六核**

  `src/xrr_fitter/physics/transitions.py`：

  - 模块 docstring 写清坐标约定：`t` 是**归一化深度**，`0` 在过渡上界（靠 fronting 一侧）、`1` 在下界，`f(t)` 是**下层材料的占比**。这个方向必须与 Task 2 的 `sld = (1 - f) * upper + f * lower` 一致，写反了两层材料会互换。
  - 六个私有核函数 `_erf_profile` / `_linear_profile` / `_tanh_profile` / `_sine_profile` / `_exponential_profile` / `_step_profile`，签名统一 `(t: np.ndarray) -> np.ndarray`。
  - 形状常数模块级具名：`ERF_HALF_WIDTH_SIGMAS = 2.0`、`TANH_HALF_WIDTH = 2.0`、`EXPONENTIAL_RATE = 4.0`。
  - `_PROFILES: dict[str, Callable[[np.ndarray], np.ndarray]]` 分派表；`TRANSITION_KINDS = frozenset(_PROFILES)`——集合从表派生，不手写第二份。
  - `transition_profile(kind, t)`：查表、未命中抛 `ValueError(f"unknown transition kind: {kind}")`、`np.asarray(t, dtype=float)` 后求值。
  - `transition_slab_count(width_a, microslab_max_a)`：先校验两个入参有限且为正，再 `max(1, int(ceil(width_a / microslab_max_a)))`。

  `erf` 与 `tanh` 的归一化形式照修正 5 的表逐字实现，分母是常数、可在模块加载时算一次。

 - [x] **Step 4: 确认 GREEN 并核复杂度**

  ```
  .venv/bin/python -m pytest tests/unit/physics/test_transitions.py --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  两条都必须干净。`check_radon.py` 是仓库级门禁，单文件平均 CC 超 5.0 会挡。

 - [x] **Step 5: 提交六核**

  ```
  git add src/xrr_fitter/physics/transitions.py tests/unit/physics/test_transitions.py
  git commit -m "physics: add normalized interface transition kernels"
  ```

### Task 2: 声明类型、归一化与构造期校验

**Files:**
- Modify: `src/xrr_fitter/model/structure.py`（在 `_periodic_layers` 之前插入两个新类型与校验助手；`LayerSpec` 追加字段；`_periodic_layers` 加拒绝分支）
- Modify: `src/xrr_fitter/api.py`
- Modify: `tests/unit/model/test_structures.py`

**Interfaces:**
- Produces: `TransitionBranch`（`kind`、`weight`、`thickness_a`）、`InterfaceTransition`（`branches`、`microslab_max_a=1.0`）、`MAX_TRANSITION_SLABS = 512`、`LayerSpec.transition: InterfaceTransition | None = None`。
- Preserves: `LayerSpec` 既有五个字段的**顺序与默认值**一个不动（`structure.py:150-153`）。新字段只能追加在 `roughness_a` 之后，否则所有位置构造的调用点全崩。`GradientLayerSpec`、`PeriodicBlock`、`SlabStack` 本任务完全不动。
- Removes: 无。

**kind 集合必须写两份（修正 13）：** `MODEL_ALLOWED["structure"] = set()` 禁止 `model/structure.py` import `physics.transitions`，所以这里要有一份独立的 kind 名字集合。两份不一致的症状极隐蔽——构造期放过一个 kind、展开期才抛 `ValueError`，而那时已经在拟合迭代里。用 `tests/unit/physics/test_transitions.py` 的一条等值测例把两份钉在一起（该文件同时能 import `model` 与 `physics`，`ALLOWED["physics"] == {"physics", "model"}` 不受影响，因为测试目录不在依赖表管辖内）。

**归一化在构造期完成：** `weight` 存归一化后的值，`InterfaceTransition.__post_init__` 用 `object.__setattr__` 回写（照 `OxideSpec` 对 `thickness_bounds_a` 的既有写法，`structure.py:~95`）。这样修正 6 的合成规则里 `Σw_b == 1` 是类型不变量而非调用点责任。

 - [x] **Step 1: 写失败的类型契约**

  在 `tests/unit/model/test_structures.py` 追加：

  - `test_transition_branch_rejects_invalid_declarations`：参数化，`kind="gaussian"`、`weight=0.0`、`weight=-1.0`、`weight=nan`、`thickness_a=0.0`、`thickness_a=-2.0`、`thickness_a=inf` 各自抛 `ValueError`。注意 `thickness_a` 这里**不**走 `_thickness`（它要求 ≥ 2 Å）——过渡分支宽度可以小于 2 Å，只要求有限且为正。
  - `test_interface_transition_normalizes_weights`：两分支权重 `(2.0, 2.0)` → 存下来是 `(0.5, 0.5)`；`(1.0, 3.0)` → `(0.25, 0.75)`。
  - `test_interface_transition_requires_branches`：空 `branches` 抛 `ValueError`。
  - `test_interface_transition_rejects_invalid_microslab_max`：`microslab_max_a` 为 0、负、`nan`、或大于总宽度 `W = max(branch.thickness_a)` 时抛 `ValueError`（修正 7）。
  - `test_interface_transition_rejects_excessive_slab_count`：`W = 4096.0`、`microslab_max_a = 1.0` → 4096 > `MAX_TRANSITION_SLABS` 抛 `ValueError`，消息含 `"512"`（修正 8）。同时断言恰好 512 片时**通过**，钉住边界是 `<=` 而非 `<`。
  - `test_layer_with_transition_requires_zero_declared_roughness`：`LayerSpec(..., roughness_a=3.0, transition=...)` 抛 `ValueError`（修正 4 的构造期一半）。
  - `test_layer_transition_width_must_not_exceed_thickness`：`thickness_a=20.0`、分支 `thickness_a=25.0` 抛 `ValueError`（修正 10）。
  - `test_periodic_block_rejects_layers_with_transitions`：把带 `transition` 的 `LayerSpec` 放进 `PeriodicBlock.layers` 抛 `ValueError`，消息说明周期块内不支持过渡（修正 11）。
  - `test_layer_without_transition_keeps_existing_construction`：`LayerSpec` 的既有五参位置构造仍然成立，且 `transition is None`。这条守住零迁移。

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/unit/model/test_structures.py --import-mode=importlib -q
  ```

  预期是 `ImportError`（`TransitionBranch` 不存在）导致整个文件收集失败——包括那 10 个既有测例。这是正常的 RED 形态，不要为了让既有测例先通过而拆文件。

 - [x] **Step 3: 实现类型与校验**

  `src/xrr_fitter/model/structure.py`：

  - `TRANSITION_KINDS = frozenset({"erf", "linear", "exponential", "tanh", "sine", "step"})`、`MAX_TRANSITION_SLABS = 512` 放模块级常量区。
  - `TransitionBranch`（`frozen=True, slots=True`）：`kind: str`、`weight: float = 1.0`、`thickness_a: float`。校验 kind 在集合内、`weight` 有限且 > 0、`thickness_a` 有限且 > 0。
  - `InterfaceTransition`（`frozen=True, slots=True`）：`branches: tuple[TransitionBranch, ...]`、`microslab_max_a: float = 1.0`。`__post_init__` 顺序是：tupleize `branches`（照 `_periodic_layers` 的防御式做法）→ 非空校验 → 类型校验 → 算 `W = max(b.thickness_a)` → 校验 `microslab_max_a ∈ (0, W]` → 校验 `ceil(W / microslab_max_a) <= MAX_TRANSITION_SLABS` → 归一化权重回写。
  - `LayerSpec` 追加 `transition: InterfaceTransition | None = None`，`__post_init__` 末尾追加一个 `_layer_transition(self.name, self.thickness_a, self.roughness_a, self.transition)` 调用。把三条交叉校验（类型、`roughness_a == 0.0`、`W <= thickness_a`）放进这个模块级助手，别塞进 `__post_init__` 正文——`LayerSpec.__post_init__` 当前 CC 已到 5，再加三个分支会超。
  - `_periodic_layers` 追加 `if any(layer.transition is not None for layer in layers): raise ValueError(...)`。

  `src/xrr_fitter/api.py` re-export `TransitionBranch`、`InterfaceTransition`（修正 16）。

 - [x] **Step 4: 确认 GREEN 与架构门禁**

  ```
  .venv/bin/python -m pytest tests/unit/model/test_structures.py tests/architecture --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  `tests/architecture` 必须跟着跑——`api.py` 的 re-export 和 `model/structure.py` 的零 import 约束都由它守。

 - [x] **Step 5: 补 kind 集合一致性测例**

  在 `tests/unit/physics/test_transitions.py` 加：

  ```python
  def test_kind_sets_agree_across_model_and_physics() -> None:
      from xrr_fitter.model.structure import TRANSITION_KINDS as model_kinds
      from xrr_fitter.physics.transitions import TRANSITION_KINDS as physics_kinds

      assert model_kinds == physics_kinds
  ```

  这条现在就应该通过——若不通过说明 Task 1 与 Task 2 的六个名字拼写不一致，立刻修，不要往后带。

 - [x] **Step 6: 提交声明层**

  ```
  git add src/xrr_fitter/model/structure.py src/xrr_fitter/api.py tests/unit/model/test_structures.py tests/unit/physics/test_transitions.py
  git commit -m "model: declare interface transition types with construction-time validation"
  ```

### Task 3: 三条展开路径同步微切片

**Files:**
- Modify: `src/xrr_fitter/physics/stack.py`（`_append_layer` 分流；新增 `_append_transition_layer`）
- Modify: `src/xrr_fitter/physics/geometry.py`（`_expanded_interface_names`、`_append_layer_geometry`、`_StackJacobianBuilder.append_transition_layer`、`append_component` 分派）
- Modify: `tests/unit/physics/test_stack_expansion.py`
- Modify: `tests/unit/test_evaluation.py`

**Interfaces:**
- Produces: 带过渡的层在 primal / 切线 / geometry 三条路径上都展开成 `N + 1` 行。
- Preserves: `expand_structure`、`expand_structure_with_jacobian`、`expand_geometry` 的签名与返回类型全不变；`transition=None` 的层走的仍是原来那三个函数体，逐位相同。
- Removes: 无。

**这是全计划风险最高的一步。** 三条路径的行数与顺序必须**同时**正确，两条 `RuntimeError` 会各自兜住一条，但兜不住第三条（primal 无对齐校验，错了会静默算出错误的反射率）。因此 Step 1 的测例里必须有一条直接比对 primal 展开行数与 `N + 1` 的断言，不能只靠 Jacobian 那两条 assert。

**三处必须一致的量，从同一个源头取：** `N = transition_slab_count(W, microslab_max_a)`，其中 `W = max(b.thickness_a for b in transition.branches)`。`physics/stack.py` 与 `physics/geometry.py` 都能 import `physics.transitions`（`ALLOWED["physics"] == {"physics", "model"}`），**不要在任何一处重新手算 `ceil`**。这是修正 10 那条不变量能成立的唯一方式。

**过渡剖面的 SLD 取值（primal）：**

```
上层 SLD = 该层入射侧的前一个介质 SLD（fronting 或前一个 component 的最后一个切片）
下层 SLD = 该层自身材料的 SLD
f_k = transition_profile(kind, (k + 0.5) / N)          # 切片中心，k = 0..N-1
sld_k = Σ_b w_b * [(1 - f_b,k) * 上层 + f_b,k * 下层]  # 多分支按归一化权重加权
```

多分支各自的 `t` 不同（修正 6）：分支 `b` 在切片 `k` 上取 `t_b,k = clip((k + 0.5) * (W / N) / b.thickness_a, 0, 1)`。窄分支在深处早已 `clip` 到 1，宽分支还在爬升——这正是不同宽度分支能合成一条剖面的机制。

**上层 SLD 的来源必须是展开状态而非声明：** `_Expansion.sld` 列表的**最后一个元素**就是上层 SLD（`expand_structure` 先 append fronting，再逐个 component 追加）。直接读 `state.sld[-1]`，不要试图从 `structure.components[index - 1]` 反推——那样处理不了前一个 component 是周期块或 gradient 的情形。

**切线路径的对应推导（修正 3 的落地）：** 过渡形状坐标不产生任何自由参数（`_layer_definitions` 只发 `.thickness_a` / `.density_scale` / `.roughness_a`），所以 `N`、`f_k`、`w_b` 全是常数，切线只是常系数线性组合：

```
thickness 切线：过渡切片各为 J[thickness_a] * 0.0（W 是常数，切片厚 W/N 不含自由参数）
                层体切片为 J[thickness_a]（层体厚 T - W，只有 T 是自由参数）
sld 切线：      sld_jac_k = (1 - F_k) * J[上层 SLD] + F_k * J[本层 SLD]，F_k = Σ_b w_b * f_b,k
roughness 切线：第一个界面取 J[roughness_a]（已锁定，恒为零向量），其后 N 个取 zero_real()
```

`J[上层 SLD]` 从 `self.sld[-1]` 取——与 primal 读 `state.sld[-1]` 严格对偶。这是本任务里最容易写错的一行：`_StackJacobianBuilder.sld` 的最后一个元素是上一个介质的 SLD 切线，`append_transition_layer` 必须在追加任何新行**之前**把它读出来存进局部变量。

**复杂度：** 三处各自的过渡分支都要独立成函数，且过渡剖面求值（多分支加权、`clip`、`t` 计算）要抽成 `physics/transitions.py` 里的一个 `transition_fractions(transition, count) -> np.ndarray`（返回长度 `count` 的 `F_k` 数组）。这样 primal 与切线两侧都只是消费一个数组，各自 CC 保持在 5 以内，加权逻辑只有一份实现——两份必然漂移。

 - [x] **Step 1: 写失败的展开契约**

  `tests/unit/physics/test_transitions.py` 追加：

  - `test_transition_fractions_are_slab_centered_and_monotone`：单分支 `linear`、`W=10`、`microslab_max_a=2.5` → `N=4`，`F` 等于 `[0.125, 0.375, 0.625, 0.875]`。这条同时钉住切片中心取样与 `N` 的算法。
  - `test_transition_fractions_weight_branches_of_different_widths`：两分支 `linear`，宽度 `(10.0, 5.0)`、权重各 0.5，`microslab_max_a=2.5` → `N=4`。窄分支在后两片已 `clip` 到 1.0，手算期望值并断言。这条钉住修正 6。
  - `test_transition_fractions_end_below_one_for_interior_centers`：任何 kind、任何 `N`，`F` 全部落在 `[0, 1]` 内且单调不减。

  `tests/unit/physics/test_stack_expansion.py` 追加：

  - `test_transition_layer_expands_to_slab_count_plus_body`：一层 `thickness_a=20.0`、单分支 `erf` `thickness_a=10.0`、`microslab_max_a=2.5` → `N=4`。断言 `stack.thickness_a.size` 比同结构无过渡时**多 4**，且过渡区四片厚度各为 `2.5`、层体片为 `10.0`。
  - `test_transition_body_slab_is_present_even_when_width_equals_thickness`：`thickness_a == W == 10.0` → 仍展开 5 行，层体片厚度精确 `0.0`。这条钉住修正 10 的无条件不变量。
  - `test_transition_slabs_interpolate_between_neighbor_and_own_sld`：过渡区第一片 SLD 更接近上层、最后一片更接近本层，且全部落在两者之间（复数分别比较实部与虚部）。
  - `test_transition_slabs_reuse_parent_thickness_for_roughness_limits`：紧跟过渡层之后再放一层，给它一个大于 `0.49 * 2.5` 但小于 `0.49 * 20.0` 的 `roughness_a`，断言**不**抛 `PhysicalValueError`。这条钉住修正 9，且是唯一能测到 `limit_thickness` 的角度。
  - `test_transition_internal_interfaces_are_zero_roughness`：过渡区内部 `N` 个界面粗糙度精确为 0.0。
  - `test_structure_without_transitions_expands_bit_identically`：构造一个含普通层、周期块、gradient 的结构，把展开结果与本次改动前的期望值逐位比对（`np.array_equal`，不用 `allclose`）。期望值直接内联在测试里，从当前 HEAD 跑一次取值。这条是零回归的守门员。

  `tests/unit/test_evaluation.py` 追加：

  - `test_transition_jacobian_matches_finite_differences`：编译一个含过渡层的问题，用 `values_and_jacobians` 取解析切线，对 `thickness_a` 与 `density_scale` 各做中心差分，`rtol=1e-6`。这是切线正确性的唯一硬证据，形状对齐 assert 通不过它。
  - `test_transition_expansion_aligns_across_all_three_paths`：同一结构分别调 `expand_structure`、`expand_structure_with_jacobian`、`expand_geometry`，断言三者的厚度数组逐位相等、`len(geometry.interface_names) == thickness.size - 1`。三条路径同时验一次。

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/unit/physics tests/unit/test_evaluation.py --import-mode=importlib -q
  ```

  预期至少三类失败：`transition_fractions` 不存在（`AttributeError`/`ImportError`）；primal 行数不足；`RuntimeError("expanded geometry interface mapping mismatch")` 或 `RuntimeError("expanded structure Jacobian mapping mismatch")`。**如果没看到后两条 `RuntimeError` 中的任何一条，说明测例没真正走到 Jacobian/geometry 路径，先修测例再往下。**

 - [x] **Step 3: 实现三条路径**

  按这个顺序改，每改一处立刻跑一次 pytest 看 RED 的形态变化——这样能确切知道是哪条路径还没通：

  1. `physics/transitions.py` 加 `transition_fractions(transition, count)`。入参用 `InterfaceTransition`（`ALLOWED["physics"]` 含 `model`，可以 import）。
  2. `physics/stack.py`：`_append_transition_layer(state, layer)`，读 `state.sld[-1]` 存局部、算 `N`、取 `F`、循环追加 `N` 个过渡切片（`thickness` 与 `limit_thickness` 各追加 `W/N` 与 `layer.thickness_a`、`roughness` 首片取 `layer.roughness_a` 其余 0.0）、最后追加层体切片（厚 `T - W`、SLD 为本层材料、roughness 0.0）。`_append_layer` 开头加 `if layer.transition is not None: return _append_transition_layer(...)`。
  3. `physics/geometry.py` 的 `_expanded_interface_names`：`LayerSpec` 分支里判 `component.transition is not None`，回填 `N + 1` 个标签（修正 17）。
  4. `physics/geometry.py` 的 `_append_layer_geometry`：过渡分支追加 `N` 个 `W/N` 加一个 `T - W`，切线照上面推导（过渡片用 `np.zeros`，层体片用 `J[thickness_a]`）。
  5. `_StackJacobianBuilder.append_transition_layer`，照 `append_gradient`（`geometry.py:371`）的结构写；`append_component` 的 `LayerSpec` 分支加分流。

  第 2 步与第 4 步的 `roughness` 首片取 `layer.roughness_a`：它已被锁 0（修正 4），这里写 `layer.roughness_a` 而不是硬编码 `0.0`，是为了让 Task 4 的锁若被误删时测试会炸而不是静默产出错误物理。

 - [x] **Step 4: 确认 GREEN 并跑全量物理与拟合**

  ```
  .venv/bin/python -m pytest tests/unit/physics tests/unit/test_evaluation.py tests/unit/fit tests/unit/analysis --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  `tests/unit/fit` 与 `tests/unit/analysis` 必须跟着跑——展开路径是它们的地基，逐位不变的断言在这里才真正被检验。

 - [x] **Step 5: 提交展开路径**

  ```
  git add src/xrr_fitter/physics/transitions.py src/xrr_fitter/physics/stack.py src/xrr_fitter/physics/geometry.py tests/unit/physics tests/unit/test_evaluation.py
  git commit -m "physics: expand interface transitions across primal, tangent, and geometry paths"
  ```

### Task 4: 过渡界面粗糙度的编译期锁定

**Files:**
- Modify: `src/xrr_fitter/fit/parameters.py`（`_layer_definitions` 锁定分支；新增 `validate_transition_modes`）
- Modify: `src/xrr_fitter/fit/problem.py`（`compile_fit_problem` 调用点）
- Modify: `tests/unit/fit/test_problem_compilation.py`

**Interfaces:**
- Produces: `validate_transition_modes(definitions, structure) -> None`。
- Preserves: `_layer_definitions` 对 `transition is None` 的层输出**完全不变**（三个定义的名字、顺序、bounds、transform 一个不动）；`validate_instrument_modes` 不动；`_require_locked_value`（`fit/parameters.py:529`）复用不改签名。
- Removes: 无。

**为什么这个 Task 排在 Task 3 之后：** Task 3 的 `roughness` 首片写的是 `layer.roughness_a`，此刻它还是自由参数，Task 3 的测试会在非零粗糙度下算出「过渡 + Névot-Croce」双重展宽的物理。这不影响 Task 3 的任何断言（那些断言都在 `roughness_a=0.0` 的声明下跑），但意味着**在 Task 4 完成前，带过渡的模型不可用于真实拟合**。不要在 Task 3 之后就去跑端到端拟合。

**顺序是硬约束：** `validate_transition_modes` 必须在 `apply_parameter_settings` **之后**调用，否则外部 `ParameterSetting` 的覆写绕过校验。照 `validate_instrument_modes` 的既有位置放（`fit/problem.py:174` 那一行之后）。

 - [x] **Step 1: 写失败的锁定契约**

  在 `tests/unit/fit/test_problem_compilation.py` 追加：

  - `test_transition_layer_roughness_is_compiled_as_locked_zero`：编译含过渡层的问题，取出 `component.0.roughness_a` 定义，断言 `locked is True` 且 `initial == lower == upper == 0.0`。
  - `test_transition_layer_roughness_is_not_a_free_variable`：断言该名字不在 `{v.name for v in problem.variables}` 里。这条比上一条更贴近真正要防的事——自由坐标才会被优化器推动。
  - `test_layer_without_transition_keeps_free_roughness`：无过渡的层其 `roughness_a` 仍 `locked is False`、`upper == max(50.0, 0.49 * thickness_a)`。守住零回归。
  - `test_unlocking_transition_roughness_is_rejected_at_compile_time`：用 `parameter_settings=(ParameterSetting("component.0.roughness_a", 3.0, 0.0, 10.0, locked=False),)` 编译，断言抛 `ValueError`，消息说明过渡界面粗糙度必须锁定在 0。这条是编译期兜底的唯一证据。
  - `test_locking_transition_roughness_at_nonzero_is_rejected`：`locked=True` 但 `initial=3.0` 同样抛 `ValueError`。`_require_locked_value` 两个条件都要被测到。
  - `test_stage_compilation_preserves_the_transition_lock`：对含过渡的问题调 `compile_stage_problem`（三个 stage 各一次），断言每次编译出的 `component.0.roughness_a` 仍 `locked is True` 且 `initial == 0.0`。这条钉住我核对过的 `stage_parameter_settings:656` 行为，防止后续有人改那个分支时静默破锁。

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/unit/fit/test_problem_compilation.py --import-mode=importlib -q
  ```

  前两条应报 `locked is False`（当前无条件自由）；两条拒绝测例应是「没抛异常」而非抛了别的异常。若看到 `ValueError: initial outside compiled bounds`，说明测例里 `ParameterSetting` 的 bounds 与 `_validate_setting` 冲突，是测例问题，先修测例。

 - [x] **Step 3: 实现锁定与校验**

  `fit/parameters.py`：

  - `_layer_definitions` 里 `roughness_a` 那个 `_definition(...)` 调用的四个参数改为条件取值。用一个模块级小助手 `_roughness_axis(layer)` 返回 `(initial, lower, upper, locked)` 四元组，避免在 `_layer_definitions` 里内联三元表达式把 CC 推高（它当前已有 `direct_sld` 分支）。`transform` 保持 `"roughness_fraction"` 不变——锁定坐标不进解码路径，transform 不被使用，改它反而会让 `_is_roughness_definition` 的既有过滤行为变化。
  - `validate_transition_modes(definitions, structure)`：`by_name = {d.name: d for d in definitions}`，遍历 `structure.components`，对 `isinstance(component, LayerSpec) and component.transition is not None` 的层调 `_require_locked_value(by_name, f"component.{index}.roughness_a", 0.0, "带过渡的界面粗糙度必须锁定在 0")`。周期块内的层不需要处理（修正 11 已在构造期拒绝）。

  `fit/problem.py`：`compile_fit_problem` 里 `validate_instrument_modes(definitions, instrument)` 之后加 `validate_transition_modes(definitions, structure)`。

 - [x] **Step 4: 确认 GREEN 并跑端到端拟合**

  ```
  .venv/bin/python -m pytest tests/unit/fit tests/unit/test_evaluation.py tests/unit/services --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  到这一步带过渡的模型才真正可拟合。`tests/unit/services` 跟着跑是为了确认服务层编排没被新校验挡住。

 - [x] **Step 5: 提交锁定**

  ```
  git add src/xrr_fitter/fit/parameters.py src/xrr_fitter/fit/problem.py tests/unit/fit/test_problem_compilation.py
  git commit -m "fit: lock transition interface roughness at zero"
  ```

### Task 5: 工程文件往返与旧文件兼容

**Files:**
- Modify: `src/xrr_fitter/io/codec_declarations.py`
- Modify: `tests/unit/io/test_project_codec.py`

**Interfaces:**
- Produces: `transition` 的 JSON 表示——`{"branches": [{"kind": ..., "weight": ..., "thickness_a": ...}, ...], "microslab_max_a": ...}`。
- Preserves: `_layer_to_dict` 对无过渡层的输出**逐键相同**（不多不少不改序）；`_periodic_to_dict` / `_periodic_from_dict` 不动（周期块内不允许过渡，修正 11）；`schema_version` **不升版**。
- Removes: 无。

**为什么不升 schema_version：** 条件发键 + `optional` 读取使新旧双向兼容——旧代码读新文件会因 `extra` 报错（这是可接受的单向不兼容，与仓库既有 `dock_state` 的处理一致），新代码读旧文件正常。升版会触发 `_migrate_v1_document` 那一整套迁移路径，代价远大于收益。

**权重归一化与往返的相互作用（必须测）：** `InterfaceTransition` 在构造期归一化权重（Task 2），所以写盘的是归一化后的值。`(2.0, 2.0)` 保存为 `(0.5, 0.5)`，再读回来归一化一次仍是 `(0.5, 0.5)`——幂等。这个性质要有测例，否则将来若有人把归一化挪到别处，往返会静默漂移。

 - [x] **Step 1: 写失败的往返契约**

  在 `tests/unit/io/test_project_codec.py` 追加：

  - `test_transition_layer_round_trips`：含单分支与含双分支（不同 kind、不同宽度）各一层，`project_to_bytes` → `project_from_bytes`，断言 `transition` 深度相等（`branches` 的 kind/weight/thickness_a 逐个比对，`microslab_max_a` 相等）。
  - `test_layer_without_transition_omits_the_key`：无过渡层的 `_layer_to_dict` 输出的键集合精确等于原来那 6 个，`"transition" not in payload`。这条钉住修正 12 的条件发键。
  - `test_saving_a_project_without_transitions_passes_null_validation`：整份工程（全无过渡）走 `project_to_bytes` 不抛异常。这条是防「发 `null` 导致所有保存失败」那个 bug 的直接守门员，必须有。
  - `test_legacy_layer_payload_without_transition_still_loads`：手写一个只含原 6 键的 layer dict，`_layer_from_dict` 成功且 `transition is None`。旧工程兼容。
  - `test_transition_payload_with_unknown_key_is_rejected`：`transition` 里多一个 `"sigma": 1.0`，抛 `ProjectSchemaError`。新嵌套结构也要走 `_mapping` 严格校验，不能用裸 dict 读取。
  - `test_transition_weights_round_trip_idempotently`：构造权重 `(2.0, 2.0)`、往返两次，四个权重值全为 `0.5`。
  - `test_transition_payload_with_invalid_kind_is_rejected`：`kind="gaussian"`，抛错。这里会由 `TransitionBranch.__post_init__` 抛 `ValueError`，需要确认它被 `codec` 层包装成 `ProjectSchemaError`——照 `project_codec.py:512` 的 `except (KeyError, TypeError, ValueError) as error: raise ProjectSchemaError` 既有处理，应当自动成立。**这条如果直接漏出 `ValueError`，说明 `_layer_from_dict` 的调用不在那个 `try` 覆盖范围内，需要在 codec 层补包装。**

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/unit/io/test_project_codec.py --import-mode=importlib -q
  ```

  预期 `_layer_to_dict` 不含 `transition`（往返丢字段）与 `_mapping` 因 `extra` 报错两类失败。

 - [x] **Step 3: 实现编解码**

  `io/codec_declarations.py`：

  - `_transition_branch_to_dict` / `_transition_branch_from_dict`、`_transition_to_dict` / `_transition_from_dict` 四个私有函数，照文件里 `_material_to_dict` / `_material_from_dict` 的既有形态写。两个 `from_dict` 都用 `_mapping` 传精确的 required 集合（分支是 `{"kind", "weight", "thickness_a"}`，过渡是 `{"branches", "microslab_max_a"}`），不加 `optional`——这是新结构，没有旧文件要兼容。
  - `_layer_to_dict` 末尾：
    ```python
    payload = { ... 原来的 6 键 ... }
    if value.transition is not None:
        payload["transition"] = _transition_to_dict(value.transition)
    return payload
    ```
    原来是直接 `return {...}`，改成先绑再条件加。**不要用字典推导或 `|` 合并把 `None` 也带进去。**
  - `_layer_from_dict`：`_mapping(value, {原 6 键}, "layer", optional={"transition"})`，构造时 `transition=_transition_from_dict(payload["transition"]) if "transition" in payload else None`。

 - [x] **Step 4: 确认 GREEN 并跑全量 io**

  ```
  .venv/bin/python -m pytest tests/unit/io tests/unit/model --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  `tests/unit/io` 全量跑是必要的——`test_examples.py` 会读仓库里的示例工程文件，是旧文件兼容的真实证据，比手写 payload 更有说服力。

 - [x] **Step 5: 提交编解码**

  ```
  git add src/xrr_fitter/io/codec_declarations.py tests/unit/io/test_project_codec.py
  git commit -m "io: serialize interface transitions with backward-compatible optional key"
  ```

### Task 6: 对话框录入、树提示与算法文档

**Files:**
- Modify: `src/xrr_fitter/gui/structure/dialogs.py`
- Modify: `src/xrr_fitter/gui/structure/editor.py`
- Modify: `tests/gui/test_structure_editor.py`
- Modify: `docs/algorithm.md`

**Interfaces:**
- Produces: `LayerDialog` 上的过渡启用开关、`microslab_max_a` 输入、三列分支表；`StructureEditor` 树在粗糙度列的 tooltip 标注过渡类型。
- Preserves: `LayerDialog.__init__` 签名不变（`(parent=None, *, commit_layer=None)`）；`layer()` 与 `_show_error` 行为不变；`TREE_HEADERS`（`editor.py:25`）**不加列**；`PeriodicDialog`、`BackingDialog` 完全不动。
- Removes: 无。

**粗糙度输入与过渡互斥的界面表达（修正 4 的用户可见后果）：** 启用过渡时 `roughness_editor` 必须 `setEnabled(False)` 并 `setValue(0.0)`。若只在提交时报错，用户会填了粗糙度再被拒绝，而错误消息来自 `model` 层（英文、指向 `roughness_a`），对着一个可编辑的中文输入框，无从理解。禁用是唯一说得通的表达。**不要**改成「提交时静默忽略粗糙度」——那是静默吞用户输入。

**分支表的列（修正 14）：** `("过渡类型", "权重", "宽度 (nm)")`，`QTableWidget(2, 3)`，照 `PeriodicDialog.table`（`dialogs.py:212`）的建法。读取照 `_layer_at` / `_cell`（`:255`/`:265`）的既有模式：`_cell` 空值抛 `ValueError`、`float()` 转换、宽度 `* 10.0` 转 Å。权重不换算（无量纲）。

**过渡类型这一列用文本单元格而非 `QComboBox`：** 与 `PeriodicDialog` 的表格形态一致，且 kind 校验已在 `TransitionBranch.__post_init__`（Task 2），非法值会走既有的 `except (TypeError, ValueError)` → `_show_error` 路径，用户看到含该 kind 名字的报错。`setCellWidget` 会让 `_cell` 的读取模式失效，得写第二套读取逻辑，不值当。

**启用开关的形态：** 一个 `QCheckBox`（objectName `layerTransitionToggle`）。未勾选时提交出 `transition=None`，与 HEAD 行为完全一致——这是「零迁移」在界面上的落点。勾选状态变化时联动 `roughness_editor.setEnabled` 与分支表的 `setEnabled`。

 - [x] **Step 1: 写失败的界面契约**

  在 `tests/gui/test_structure_editor.py` 追加（照 `:261`/`:286` 两条既有 `LayerDialog` 测例的写法，用 `qtbot`）：

  - `test_layer_dialog_without_transition_commits_none`：不勾选开关、正常填名称/化学式/密度/厚度/粗糙度，提交后 `commit_layer` 收到的 `LayerSpec.transition is None` 且 `roughness_a` 是填入值 `* 10.0`。零迁移守门员。
  - `test_layer_dialog_disables_roughness_when_transition_is_enabled`：勾选开关后 `dialog.roughness_editor.isEnabled() is False` 且 `value() == 0.0`；取消勾选后恢复 `isEnabled() is True`。
  - `test_layer_dialog_commits_normalized_branch_weights`：勾选开关，两行分支各填 `erf`/`2`/`1.0` 与 `linear`/`2`/`0.5`，提交后 `commit_layer` 收到的 `transition.branches` 权重为 `(0.5, 0.5)`、宽度为 `(10.0, 5.0)` Å。**这条替代 spec 的回显验收（修正 15）——断言落在提交出的数据上，不是对话框控件上。**
  - `test_layer_dialog_reports_unknown_transition_kind`：分支填 `gaussian`，提交后 `error_label.isVisible() is True`、文本含 `"gaussian"`，且对话框**未** `accept()`（`dialog.result() != QDialog.DialogCode.Accepted`）。
  - `test_layer_dialog_reports_incomplete_branch_row`：分支行留空一格，`error_label` 可见。走 `_cell` 的既有 `ValueError` 路径。
  - `test_structure_tree_marks_transition_in_the_roughness_column`：把带过渡的层放进 `StructureEditor`，断言粗糙度列（`TREE_HEADERS` 索引 4）的 tooltip 文本含过渡类型名，且 `len(TREE_HEADERS) == 6`（不加列）。

 - [x] **Step 2: 确认 RED**

  ```
  .venv/bin/python -m pytest tests/gui/test_structure_editor.py --import-mode=importlib -q
  ```

  预期 `AttributeError`（`transition_toggle` / `branch_table` 不存在）。GUI 测试需要 Qt 离屏环境，若报 `qt.qpa.plugin` 相关错误说明环境问题而非代码问题，先确认 `tests/conftest.py` 的既有 offscreen 设置生效。

 - [x] **Step 3: 实现界面**

  `gui/structure/dialogs.py`：

  - `LayerDialog.__init__` 加 `self.transition_toggle = QCheckBox()`（objectName `layerTransitionToggle`）、`self.microslab_editor = _number("layerMicroslabInput", 0.001, 0.1)`（默认 0.1 nm = 1 Å，与 `microslab_max_a` 默认值一致）、`self.branch_table = QTableWidget(2, 3)`（objectName `layerTransitionTable`）。`transition_toggle.toggled.connect(self._sync_transition_inputs)`，并在 `__init__` 末尾调一次 `_sync_transition_inputs(False)` 建立初始态。
  - `_arrange` 在粗糙度行之后加「界面过渡」勾选行、「切片上限 (nm)」行，然后 `layout.addWidget(self.branch_table)`（在 `error_label` 之前）。
  - `_sync_transition_inputs(enabled: bool)`：`roughness_editor.setEnabled(not enabled)`、`enabled` 时 `roughness_editor.setValue(0.0)`、`branch_table.setEnabled(enabled)`、`microslab_editor.setEnabled(enabled)`。
  - `_transition()` 返回 `api.InterfaceTransition | None`：未勾选返回 `None`；勾选时逐行读三格构造 `api.TransitionBranch`，用 `api.InterfaceTransition(branches, microslab_editor.value() * 10.0)`。复用 `_cell`——把 `PeriodicDialog._cell` 提成模块级 `_table_cell(table, row, column, label)` 供两个对话框共用，别复制一份。
  - `_accept_fields` 的 `api.LayerSpec(...)` 调用加 `transition=self._transition()`。它已在 `try` 内，`TransitionBranch` / `InterfaceTransition` 抛的 `ValueError` 自动走 `_show_error`。

  `gui/structure/editor.py`：树项构建处对 `layer.transition is not None` 的层，在索引 4 调 `setToolTip(4, ...)`，文案列出各分支 kind 与宽度。

  **复杂度提醒：** `_accept_fields` 与 `_transition` 都容易超 CC 10。分支行读取单独成 `_branch_at(row)`，照 `_layer_at` 的形态。

 - [x] **Step 4: 确认 GREEN 并跑全量 GUI 与架构门禁**

  ```
  .venv/bin/python -m pytest tests/gui tests/architecture --import-mode=importlib -q
  .venv/bin/python tools/check_radon.py
  ```

  `tests/architecture` 在这里第二次跑是必需的：`gui` 只能 import `{gui, api}`，Task 2 加的 re-export 是否够用在此刻才被真正检验（`test_accessibility.py` 也会跟着验新控件的可访问名）。

 - [x] **Step 5: 写算法文档**

  `docs/algorithm.md` 新增一节「界面过渡函数族」，必须包含：

  1. 六核在归一化坐标 `t ∈ [0,1]` 上的**闭式公式**与三个形状常数的数值（`ERF_HALF_WIDTH_SIGMAS = 2.0` 等）。没有常数的话「erf 过渡」是句没有数值含义的话。
  2. 多分支合成规则：`W = max(b.thickness_a)`、`t_b = clip(z / b.thickness_a, 0, 1)`、按归一化权重加权，端点精确 0 与 1。
  3. **粗糙度替代语义**：过渡取代 Névot-Croce 而非叠加，界面粗糙度锁定在 0，理由写清（同一界面不能被展宽两次）。
  4. **`_sharp_profile` 台阶后果**（保留不动的 spec 判断那一节里记的）：`sld_depth_profile` 在全结构粗糙度为 0 时走 `_sharp_profile`（`physics/sld_profile.py:57`），过渡区会画成 `N` 级台阶而非平滑曲线。这是微切片的真实离散化分辨率，**不是缺陷**。写明这一点，免得后来者当 bug 修掉。
  5. `MAX_TRANSITION_SLABS = 512` 的取值理由。

 - [x] **Step 6: 提交界面与文档**

  ```
  git add src/xrr_fitter/gui/structure/dialogs.py src/xrr_fitter/gui/structure/editor.py tests/gui/test_structure_editor.py docs/algorithm.md
  git commit -m "gui: enter interface transitions and document the kernel family"
  ```

---

## 最终验收记录

六个 Task 全部完成后跑这一组，把实际输出贴在下面：

```
.venv/bin/python -m pytest tests --import-mode=importlib -q
.venv/bin/python tools/check_radon.py
.venv/bin/python tools/verify_registry.py
```

- [x] 全量 pytest 通过，且**无** `skipped` / `xfailed` / `deselected`（`tests/outcome_gate.py` 会因此整轮失败，所以这一条由门禁自动保证，但仍要看一眼计数）。
- [x] `check_radon.py` 干净。
- [x] `verify_registry.py` 干净（本计划新增测试文件都落在整目录注册下，预期无需改动，若报错说明落错了目录）。
- [x] 记下 `tests/unit/physics/test_stack_expansion.py::test_structure_without_transitions_expands_bit_identically` 的通过状态——这是零回归的唯一硬证据。
- [x] 记下 `tests/unit/test_evaluation.py::test_transition_jacobian_matches_finite_differences` 的通过状态——这是解析导数正确性的唯一硬证据。

**已知的两个既有失败**：仓库在本计划开始前已有两条失败测例（见项目记忆），它们与过渡无关。验收时确认失败集合**没有变大**即可，不要试图在本计划里修它们。

### 历史验收输出（HEAD = `e1da5b6`）

全量套件按目录分批跑（`tests/acceptance` 单批需 2 小时以上，故独立执行）。每批命令形如：

```
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate <paths> -q
```

| 批次 | 结果 |
| --- | --- |
| `tests/unit/physics tests/unit/test_evaluation.py tests/unit/fit tests/unit/analysis` | `544 passed` |
| `tests/unit/io tests/unit/model` | `306 passed` |
| `tests/gui tests/architecture` | `573 passed in 84.73s` |
| `tests/integration` | `26 passed in 88.55s` |
| `tests/regression` | `47 passed in 50.05s` |
| `tests/acceptance` | `2 failed, 2 passed in 7900.61s (2:11:40)` |

门禁：

```
tools/check_radon.py    → exit 0
tools/verify_registry.py → exit 0
```

两条零回归硬证据均通过（含在上表第一批的 `544 passed` 内）：

- `tests/unit/physics/test_stack_expansion.py::test_structure_without_transitions_expands_bit_identically`
- `tests/unit/test_evaluation.py::test_transition_jacobian_matches_finite_differences`

### 关于 acceptance 的两个失败

失败集合**没有变大**，但成因与计划开始前记录的那两条不同（那两条已在本计划之前修好）。当前两条是环境缺失，不是回归：

```
tests/acceptance/test_gui_real_data_workflows.py::test_gui_real_data_workflows_round_trip_owner_projects
tests/acceptance/test_real_data_workflows.py::test_real_data_workflows_produce_four_run_candidate_records
E   KeyError: 'XRR_APPROVED_DATA_ROOT'
```

两者在 `f63ae5c`（早于本计划起点 `2fc4ea5`）就要求 `XRR_APPROVED_DATA_ROOT` 与
`XRR_APPROVED_REPORT_DIR` 指向获批的真实测量数据目录，本机未提供该数据集，故在读环境变量时即失败，
未进入任何过渡代码。不含真实数据的那两条 acceptance（合成恢复语料）通过。

### 当前工作区复核（HEAD = `df7c826`，2026-08-13）

当前 unit、GUI、architecture 与 integration 使用 outcome gate 联跑：

```text
2179 passed in 194.34s (0:03:14)
```

`tools/check_radon.py`、`tools/verify_registry.py` 与 `git diff --check` 均通过；
实时预览重建中文图例后的 CJK 字体遗漏已修复，全量联跑不再产生 warning；
`test_structure_without_transitions_expands_bit_identically` 与
`test_transition_jacobian_matches_finite_differences` 包含在全量通过集合中。
transition 对话框测试已拆到 `tests/gui/test_transition_dialog.py`，不再追加到
`test_structure_editor.py`；这是测试组织调整，不是覆盖缺口。

<!-- PLAN-COMPLETE -->

---

## 剩余风险

- **非 erf 过渡的解析导数是本计划的范围选择，不是永久限制。** 修正 3 推翻了 spec 的「非 erf 不支持解析导数」判断——六核全部有完整解析切线，因为形状坐标不产生自由参数（`N`、`f_k`、`w_b` 全是常数）。代价是**过渡宽度本身不可拟合**：`branch.thickness_a` 是声明值，不进 `_layer_definitions`。若将来要让宽度可拟合，需要为 `W` 发一个参数定义，并处理 `N = ceil(W / microslab_max_a)` 随之变成**参数依赖的整数**——那会让展开行数在迭代中变化，`finish()` 的形状对齐 assert 立刻失效。这是一个独立的、更大的改动，不要顺手做。
- **过渡宽度不可拟合意味着用户只能手动扫。** 这是相对 refnx/Refl1D 之类允许界面宽度自由拟合的软件的实际短板。本轮先把物理与序列化打通，宽度可拟合作为后续项。
- **`PeriodicBlock` 内不支持过渡（修正 11）。** 多层膜的界面过渡是真实需求，本轮拒绝掉了。解锁需要同时改 `PeriodicSpan.layer_count` 的语义、`_validate_roughness_repetition`（`model/structure.py:312`）与 `top_roughness_a` 的覆盖条件（`physics/stack.py:203`），且与 nested-periodic-stacks 计划有写集重叠——两个计划不要并行推进这一块。
- **微切片使总切片数上升，反射率计算成本随之上升。** `MAX_TRANSITION_SLABS = 512` 只约束单层最坏情况，多层各带过渡时是累加的。本计划没有做性能测量，也没有加总切片数的全局上限。若实测拟合明显变慢，先量再决定是否加全局上限——不要凭感觉调 `microslab_max_a` 默认值。
- **旧代码读新工程文件会因 `extra` 键报错**（修正 12）。这是有意的单向兼容，与仓库既有 `dock_state` 的处理一致，且不升 `schema_version`。若将来要支持降级读取，需要另设迁移路径。
- **`_sharp_profile` 台阶不是缺陷但会被误报。** 全零粗糙度结构的剖面图上过渡区呈 `N` 级台阶。已写进 `docs/algorithm.md`（Task 6 Step 5），但文档挡不住用户报 bug。若反馈多，可考虑让 `sld_depth_profile` 对含过渡的结构强制走平滑分支——那需要它感知 kind，而 kind 在展开后已丢失（修正 1），代价是重新设计 `SlabStack` 的信息量，本轮明确不做。
