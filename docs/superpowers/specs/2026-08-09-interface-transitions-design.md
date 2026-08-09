# 界面过渡函数族设计

## 目标

把界面过渡从只有 erf 扩展为六种可归一化线性组合的过渡函数（erf、线性、指数、tanh、
正弦、阶跃），每个分支有独立的厚度与权重。吸收自 Multifitting 手册 §5.1.3。

值得做的理由不是"多几个选项"：erf 假定界面粗糙度是高斯分布的，这对**扩散界面**（如热
处理后的硅化物层）在物理上就是错的，tanh 或指数才对。这是真实的模型能力缺口。

## 现有结构

`physics/sld_profile.py:23` 的 `_transition` 是唯一的过渡核：

```python
def _transition(depth: np.ndarray, interface: float, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.where(depth < interface, 0.0, np.where(depth > interface, 1.0, 0.5))
    return 0.5 * (1.0 + erf((depth - interface) / (np.sqrt(2.0) * sigma)))
```

`:29` 的 `_profile` 按界面累加，用 `np.minimum.accumulate` 保证单调、用相邻差构造权重。
新增核函数是替换 `_transition` 的分派，**不是新架构**。实现成本低。

## 关键边界：Névot-Croce 的适用性

Névot-Croce 修正是从高斯界面推导出来的。过渡函数换成 tanh 或指数后，解析修正不再对应
该过渡形状。这一点不能含混。

**本轮范围：过渡函数族只用于 SLD 剖面的可视化与切片计算，反射率的解析 Névot-Croce 修正
保持 erf。** 非 erf 过渡的反射率通过微切片（复用 `_append_gradient` 的既有机制）计算，
不套用 Névot-Croce。文档必须写清这个边界。

理由：把非高斯过渡塞进 Névot-Croce 会得到一个既非解析正确、也非数值正确的中间物。切片
路径慢但正确，而慢的代价只在用户显式选择非 erf 过渡时才付。

## 数据类型

`MODEL_ALLOWED["structure"] = set()`，所以过渡声明必须自洽：

```python
@dataclass(frozen=True)
class TransitionBranch:
    kind: str                    # "erf" | "linear" | "exponential" | "tanh" | "sine" | "step"
    weight: float
    thickness_a: float

@dataclass(frozen=True)
class InterfaceTransition:
    branches: tuple[TransitionBranch, ...]
```

`LayerSpec` 增加 `transition: InterfaceTransition | None = None`，默认 `None` 即当前纯 erf
行为，现存工程文件零迁移。

归一化在构造期做：权重和必须为正，构造后按和归一化并冻结。不在求值期归一化——那会让同
一份声明在不同调用点产生不同剖面。

## 代码边界

- 改 `src/xrr_fitter/model/structure.py`：两个新类型，`LayerSpec` 加字段。
- 改 `src/xrr_fitter/physics/sld_profile.py`：`_transition` 改为按 `kind` 分派，新增五个
  核函数。保持每个核在 `sigma == 0.0` 时退化为阶跃。
- 改 `src/xrr_fitter/physics/stack.py`：非 erf 过渡的层走微切片展开。
- 改 `src/xrr_fitter/io/codec_common.py`：序列化。
- 不改 `physics/parratt.py` 与 `physics/derivatives.py` 的 Névot-Croce 实现。
- 改 `src/xrr_fitter/gui/structure/{dialogs,editor}.py`。
- `api` 侧无需新函数：过渡是 `LayerSpec` 的字段，随 `api.set_structure`
  （`StructurePanel.set_structure`，`panel.py:58`）整体提交。

## 界面

过渡是层的属性，编辑入口就在层对话框里。`LayerDialog`（`dialogs.py:42`）在"粗糙度 (nm)"
之后加一行：`QComboBox` 选 kind，默认 "erf"。选 erf 时不再显示别的控件，`transition` 字段
留 `None`，保证既有工程的往返与逐位不变的断言不被界面破坏。选非 erf 时展开该核所需的参数
字段。多分支组合用一个小表格（照 `PeriodicDialog.table` 的 `QTableWidget` 用法），列为
`("过渡类型", "权重")`。

`_accept_fields` 里连同 `transition` 一起构造完整 `LayerSpec`；权重非正、全零、或 kind 元数
不匹配的构造期错误落在既有 `layerDialogError` 标签上，对话框不关闭。归一化在构造期完成，
所以对话框回显的是归一化后的权重，用户输入 `(2, 2)` 会看到 `(0.5, 0.5)` ——这是正确行为，
避免同一份声明有两种权重表示。

`StructureEditor` 的树（`TREE_HEADERS`，`editor.py:25`）在"粗糙度 (nm)"列的 tooltip 里标注
过渡类型，不加新列。理由：表头已有六列，绝大多数层是 erf，加一列会让常见情形多一列空白。

**解析 Jacobian 冲突要在提交时就说清。** 非 erf 过渡与解析梯度并用是显式报错（不静默降级），
这个错误如果只在开始拟合时才抛，用户已经编辑完整个结构了。做法：`LayerDialog` 选中非 erf
kind 时立即在对话框内显示一条提示，说明该层将走微切片且不能与解析梯度并用；真正的报错仍由
后端在拟合前抛出，界面只是提前告知，不复制后端判断逻辑。

## 失败与状态

- `weight` 非正或全部为零时构造期报错。
- `thickness_a` 非正时报错。
- 非 erf 过渡与解析 Jacobian 同时启用时报错并说明原因（微切片路径的解析导数不在本轮
  范围），提示改用 erf 或关闭解析梯度。这是显式失败而非静默降级为有限差分。
- 微切片数量超过上限时报错，不静默截断切片精度。

## 验证

- 每个核的单调性测例：`_transition` 输出在 `[0, 1]` 内且单调不减。
- 每个核的极限测例：`sigma == 0.0` 退化为阶跃；`depth -> -inf` 趋 0，`-> +inf` 趋 1。
- 归一化测例：多分支组合的过渡在两端仍精确趋 0 与 1。
- erf 单分支与 `transition=None` 结果**逐位相等**。这是增量兼容的核心断言。
- 微切片收敛测例：切片加密时非 erf 过渡的反射率收敛。
- 非 erf 与解析 Jacobian 并用时报错的测例。
- GUI 测例：kind 选 erf 时提交出的 `LayerSpec.transition is None`。这条挡的是界面把
  "erf 单分支"写成一个显式对象，从而绕过逐位不变断言。
- GUI 测例：权重 `(2, 2)` 提交后回显 `(0.5, 0.5)`。

## 非目标

- 不为非 erf 过渡实现解析 Jacobian。
- 不把非 erf 过渡接入 Névot-Croce 解析修正。
- 不做过渡参数的自动识别或推荐。
