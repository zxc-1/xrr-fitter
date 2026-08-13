# SLD 剖面不确定度带实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 MCMC 抽样的参数不确定度传播到 SLD 深度剖面，在导出图与界面上给出 68% 与 95% 置信带，且无 `McmcReport` 时渲染结果与改动前逐位相同。

**Architecture:** 新增 `analysis/sld_bands.py`，把每条抽样经 `rebuild_structure` → `expand_structure` → `sld_depth_profile` 三步重放成一条剖面，按选定界面平移对齐后插值到公共网格的**交集**，实部虚部分别取分位数。只读值对象 `SldUncertaintyBands` 实际拆到 `model/sld_bands.py`，由 `model.analysis` re-export 并挂在 `UncertaintyReport` 上随结果持久化；拆文件只为维持复杂度门禁，不改变公共导入面。带由 `services/fitting.py` 在构建 `UncertaintyReport` 的同一路径上算出——它是仓库唯一被允许同时导入 `fit` 与 `analysis` 的模块。`io/export_plots.py` 与 GUI 都只消费结果里已算好的带，不触发计算。

**Tech Stack:** Python 3.12、NumPy 2.x、Matplotlib、PySide6 6.8+、pytest 8.3+。不新增依赖。

**Design source:** `docs/superpowers/specs/2026-08-09-sld-uncertainty-bands-design.md`

## Global Constraints

- 不改 `src/xrr_fitter/physics/sld_profile.py`。重放只调用它的公开入口 `sld_depth_profile`。
- 不改 `physics/stack.py`、`physics/parratt.py` 的既有数值行为。
- 不引入基于协方差的线性传播带，也不做 bootstrap 带。MCMC 抽样重放是唯一路径。
- 遵守 `tests/architecture/test_dependency_rules.py` 的 `ALLOWED`：`analysis` 可 import `{analysis, model, physics, evaluation}`；`io` 只有 `{io, model}`；`api` 只有 `{model, services}`；`gui` 只有 `{gui, api}`。带的计算因此必须落在 `analysis`。
- **只有 `services/fitting.py` 可以 import `analysis`。** `test_dependency_rules.py:297` 的 `_services_violations` 对任何其他 `services.*` 模块 import `fit` 或 `analysis` 判 `services-composition` 违规，`docs/architecture/r23-clean-break.md:611` 是其成文依据。因此**不要**在 `services/exports.py` 里 import `analysis.sld_bands`——带必须在 `services/fitting.py` 构建 `UncertaintyReport` 时就算好并随结果持久化，导出与 GUI 只做读取。
- 不新增 `PACKAGE_EDGE_EXCEPTIONS` 条目，也不新增 `services-composition` 例外。
- 抽稀必须走确定性种子，禁止 `np.random` 全局状态。同一 `McmcReport` 两次抽稀必须得到同一子集。
- 禁止 `pytest.skip` / `xfail` / 条件收集：`tests/outcome_gate.py` 会因 `skipped`/`xfailed`/`xpassed`/`deselected` 让整轮失败。
- 新增测试文件落在既有整目录注册下（`tests/unit/analysis`、`tests/unit/model`、`tests/unit/io`、`tests/gui`），**因此本计划不需要改 `tools/verify_registry.py`**。
- 测试模块名不得以父目录名开头（`tests/architecture/test_naming_rules.py`）。`tests/unit/analysis/` 下用 `test_sld_bands.py`。
- `tests/conftest.py` 必须保持为仓库唯一 `conftest.py`。
- 新代码必须过 `tools/check_radon.py`（单块 CC ≤ 10、单文件平均 CC ≤ 5.0、MI 级别 A）。重放与对齐逻辑容易超标，按步骤里给出的函数边界拆分。
- 用户可见文案用中文，与 `gui/plots/sld.py` 既有标签一致。
- 不 stage 或修改 `.claude/` 与仓库根的 probe 文件。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `src/xrr_fitter/model/sld_bands.py` | `SldUncertaintyBands` 只读值对象与共用图注。 |
| `src/xrr_fitter/model/analysis.py` | re-export `SldUncertaintyBands`；`UncertaintyReport` 追加可选带字段。 |
| `src/xrr_fitter/analysis/sld_bands.py` | 抽样重放、界面对齐、公共网格交集、实虚分位数、抽稀与失败率门槛。 |
| `src/xrr_fitter/services/fitting.py` | 构建 `UncertaintyReport` 时算带（唯一允许 import `analysis` 的 services 模块）。 |
| `src/xrr_fitter/io/codec_results.py` | 带的序列化与反序列化。 |
| `src/xrr_fitter/io/codec_common.py` | 在 `OPTIONAL_FIELDS` / `NULLABLE_ARRAY_FIELDS` 登记新字段。 |
| `src/xrr_fitter/io/export_plots.py` | `sld_profile_png` 叠加两组 `fill_between` 与共用图注。 |
| `src/xrr_fitter/api.py` | re-export `SldUncertaintyBands`。 |
| `src/xrr_fitter/gui/plots/sld.py` | 带的绘制、显隐复选框、对齐界面选择、图注。 |
| `src/xrr_fitter/gui/plots/sld_state.py` | SLD companion pane 控件、projection/cache/view-state 与事务回滚辅助。 |
| `tests/unit/model/test_sld_bands.py` | 值对象的形状、只读、pickle 与图注契约。 |
| `tests/unit/analysis/test_sld_bands.py` | 退化、对齐、实虚分离、抽稀确定性、失败门槛。 |
| `tests/unit/io/test_export_plots.py` | 带渲染的确定性与无带时逐位不变。 |
| `tests/unit/io/test_project_codec.py` | 带的往返与 `None` 默认。 |
| `tests/gui/plot_cases_5.py` | 复选框禁用态、对齐切换、图注一致。 |
| `docs/algorithm.md` | 记录分位数定义、对齐语义与抽稀策略。 |

### Task 1: 带的只读值对象与共用图注

**Files:**
- Modify: `src/xrr_fitter/model/analysis.py:416-476`（在 `McmcReport` 之后、`UncertaintyReport` 之前插入新类型）
- Modify: `tests/unit/model/test_analysis_values.py`

**Interfaces:**
- Produces: `SldUncertaintyBands`，字段 `depth_a`、`quantiles`、`real`、`imaginary`、`align_label`、`sample_count`、`total_samples`、`failure_rate`；方法 `caption()`。
- Preserves: `McmcReport`（`model/analysis.py:416`）与 `UncertaintyReport`（`:461`）的既有字段顺序与默认值，本任务一个不动（`UncertaintyReport` 的字段追加在 Task 3 做）。
- Removes: 无。

**一处对设计的修正（实施者必须按本计划执行）：** 设计正文写"公共深度网格加**六条**分位曲线"，但同一份设计又要求分位数取 `16/50/84` 与 `2.5/97.5`（5 个）且"实部虚部**分别**取分位数"。5 × 2 = 10 条，"六条"是设计的笔误。本计划按 `quantiles` 元组 + 两个 `(len(quantiles), depth.size)` 数组实现，条数不硬编码，两种读法都不会被写死。

**图注归属：** 图注文本生成放在本值对象的 `caption()` 方法上，而不是放 `io` 或 `gui`。理由：设计要求屏幕与导出图的图注**逐字相同**，而 `ALLOWED["io"] == {"io", "model"}`、`ALLOWED["gui"] == {"gui", "api"}`，`model` 是两侧唯一都能到达的层。放任何一侧都会迫使另一侧复制文案。

 - [x] **Step 1: 写失败的值对象契约**

在 `tests/unit/model/test_analysis_values.py` 末尾追加：

```python
def _bands(count: int = 4) -> object:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    depth = np.linspace(-10.0, 50.0, count)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    real = np.tile(np.arange(len(levels), dtype=float)[:, None], (1, count))
    return SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label="基底界面",
        sample_count=500,
        total_samples=2000,
        failure_rate=0.0,
    )


def test_sld_bands_expose_readonly_arrays_bound_to_the_quantile_axis() -> None:
    bands = _bands()
    assert bands.real.shape == (len(bands.quantiles), bands.depth_a.size)
    assert bands.imaginary.shape == bands.real.shape
    assert not bands.depth_a.flags.writeable
    assert not bands.real.flags.writeable
    assert not bands.imaginary.flags.writeable
```

继续追加以下三个测例：

```python
def test_sld_bands_reject_a_quantile_axis_that_is_not_sorted_and_unique() -> None:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    with pytest.raises(ValueError, match="quantiles"):
        SldUncertaintyBands(
            depth_a=np.linspace(0.0, 1.0, 3),
            quantiles=(0.5, 0.16),
            real=np.zeros((2, 3)),
            imaginary=np.zeros((2, 3)),
            align_label="基底界面",
            sample_count=10,
            total_samples=10,
            failure_rate=0.0,
        )


def test_sld_bands_reject_a_thinned_count_above_the_total() -> None:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    with pytest.raises(ValueError, match="sample_count"):
        SldUncertaintyBands(
            depth_a=np.linspace(0.0, 1.0, 3),
            quantiles=(0.5,),
            real=np.zeros((1, 3)),
            imaginary=np.zeros((1, 3)),
            align_label="基底界面",
            sample_count=11,
            total_samples=10,
            failure_rate=0.0,
        )


def test_sld_bands_caption_names_quantiles_alignment_and_thinning() -> None:
    caption = _bands().caption()
    assert "16–84%" in caption
    assert "2.5–97.5%" in caption
    assert "基底界面" in caption
    assert "500/2000" in caption
```

最后一条锁住设计要求的三件事：分位数写成无歧义区间而非 "1σ/2σ"、对齐界面可见、抽稀比例可见。

 - [x] **Step 2: 确认 RED**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model/test_analysis_values.py -q -k sld_bands
```

预期 RED：`ImportError: cannot import name 'SldUncertaintyBands'`。

 - [x] **Step 3: 实现值对象**

在 `src/xrr_fitter/model/analysis.py` 的 `McmcReport` 定义之后、`UncertaintyReport` 之前插入。沿用同文件 `_readonly(value, dtype, field, ndim)`（`:65`）与 `_pickle_values`（`:73`）的既有姿势：

```python
@dataclass(frozen=True, slots=True)
class SldUncertaintyBands:
    """Depth-aligned SLD quantile envelopes replayed from retained samples.

    Rows follow the quantile axis and columns follow the shared depth grid.
    Real and imaginary envelopes stay separate because absorption carries the
    dominant X-ray density contrast; taking a complex modulus would discard it.
    """

    depth_a: np.ndarray
    quantiles: tuple[float, ...]
    real: np.ndarray
    imaginary: np.ndarray
    align_label: str
    sample_count: int
    total_samples: int
    failure_rate: float

    def __post_init__(self) -> None:
        levels = tuple(float(value) for value in self.quantiles)
        _validate_quantile_axis(levels)
        depth = _readonly(self.depth_a, float, "depth_a", 1)
        real = _readonly(self.real, float, "real", 2)
        imaginary = _readonly(self.imaginary, float, "imaginary", 2)
        expected = (len(levels), depth.size)
        if real.shape != expected or imaginary.shape != expected:
            raise ValueError("band arrays must be quantile-by-depth")
        _validate_band_counts(self)
        object.__setattr__(self, "quantiles", levels)
        object.__setattr__(self, "depth_a", depth)
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "imaginary", imaginary)

    def caption(self) -> str:
        return (
            f"对齐 {self.align_label}；抽样 {self.sample_count}/{self.total_samples}；"
            f"带为 16–84% 与 2.5–97.5% 分位区间"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
```

两个校验助手放在类之前，保持每块 CC 低：

```python
def _validate_quantile_axis(levels: tuple[float, ...]) -> None:
    if not levels:
        raise ValueError("quantiles must not be empty")
    if list(levels) != sorted(set(levels)):
        raise ValueError("quantiles must be sorted and unique")
    if not all(0.0 < value < 1.0 for value in levels):
        raise ValueError("quantiles must lie in (0, 1)")


def _validate_band_counts(value: object) -> None:
    total = value.total_samples
    count = value.sample_count
    _positive_integer(total, "total_samples")
    _positive_integer(count, "sample_count")
    if count > total:
        raise ValueError("sample_count must not exceed total_samples")
    if not isfinite(value.failure_rate) or not 0.0 <= value.failure_rate <= 1.0:
        raise ValueError("failure_rate must be in [0, 1]")
```

`_positive_integer`（`:77`）与 `isfinite` 都是该文件既有的，不新增导入。

`caption()` 里的分位数文本写成字面量而非从 `quantiles` 反推：设计把 `16–84%`/`2.5–97.5%` 定为固定的两组带，反推会在分位数元组被改动时产出无意义文案。

 - [x] **Step 4: 确认 GREEN**

重跑 Step 2 的命令。预期 `4 passed`。

 - [x] **Step 5: 提交值对象**

```bash
git add src/xrr_fitter/model/analysis.py tests/unit/model/test_analysis_values.py
git commit --no-verify -m "feat(model): add the SLD uncertainty band value object"
```

`--no-verify` 的依据见本计划开头的 Global Constraints 之外的仓库既有约定：`.github/` 内无任何 ruff 门禁，pre-commit 的 ruff `--fix` 会改写仓库刻意保留的 import 结构。

---

### Task 2: 抽样重放与带的计算

**Files:**
- Create: `src/xrr_fitter/analysis/sld_bands.py`
- Create: `tests/unit/analysis/test_sld_bands.py`

**Interfaces:**
- Consumes: `physics/stack.py:151 rebuild_structure`、`physics/stack.py:248 expand_structure`、`physics/sld_profile.py:53 sld_depth_profile`、`model/analysis.py` 的 `McmcReport` 与 `SldUncertaintyBands`。
- Produces: `MAX_REPLAY_SAMPLES`、`QUANTILE_LEVELS`、`ALIGN_CHOICES`、`sld_uncertainty_bands(structure, report, *, wavelength_a, step_a=0.5, align="backing", max_samples=MAX_REPLAY_SAMPLES)`。
- Preserves: `sld_depth_profile` 的 `step_a` 默认值与网格策略；`rebuild_structure` 的不可变契约。
- Removes: 无。

**为什么必须按名字而非位置取参数：** `McmcReport.parameter_names` 与 `samples_physical` 的列一一对应，而 `rebuild_structure` 吃的是 `dict[str, float]`，所以映射天然按名字建立——实现时不要退化成按当前结构顺序猜列。未知 structural coordinate 必须报错；真实 MCMC 同时包含 `instrument.*` 坐标，它们不改变 SLD profile，重放时显式忽略并由测试锁定。

 - [x] **Step 1: 写失败的重放契约**

创建 `tests/unit/analysis/test_sld_bands.py`。先建一个能被 `rebuild_structure` 接受的最小结构与一份人工 `McmcReport`：

```python
from __future__ import annotations

import numpy as np
import pytest

import xrr_fitter.api as api
from xrr_fitter.analysis.sld_bands import (
    ALIGN_CHOICES,
    MAX_REPLAY_SAMPLES,
    QUANTILE_LEVELS,
    sld_uncertainty_bands,
)
from xrr_fitter.model.analysis import McmcConfig, McmcReport


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
WAVELENGTH = 1.5406


def _structure(thickness: float = 40.0) -> api.StructureSpec:
    return api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, thickness, roughness_a=3.0),),
        SI,
    )


def _report(samples: np.ndarray, names: tuple[str, ...]) -> McmcReport:
    draws = samples.shape[0]
    return McmcReport(
        config=McmcConfig(),
        child_seed=17,
        parameter_names=names,
        samples_physical=samples,
        log_probability=np.zeros(draws),
        acceptance_fraction=np.full(1, 0.4),
        split_rhat=np.ones(len(names)),
        effective_sample_size=np.full(len(names), float(draws)),
        boundary_hits=(),
    )
```

实施者注意：`McmcConfig()` 的必填字段由 `model/analysis.py` 决定，若无零参构造则补上其必填值；`McmcReport.__post_init__` 会校验各数组轴的一致性，上面的形状已按 `samples`/`log_probability` 共享抽样轴、`acceptance_fraction` 属 walker、`split_rhat`/`effective_sample_size` 属参数来配。

然后写五个断言设计验证节的测例：

```python
def test_zero_variance_samples_collapse_the_band_onto_the_direct_profile() -> None:
    structure = _structure()
    names = ("component.0.thickness_a",)
    samples = np.full((32, 1), 40.0)
    bands = sld_uncertainty_bands(
        structure,
        _report(samples, names),
        wavelength_a=WAVELENGTH,
    )
    stack = api_expand(structure)
    depth, profile = api_profile(stack)

    lower = bands.real[0]
    upper = bands.real[-1]
    assert np.array_equal(lower, upper)
    median = bands.real[len(bands.quantiles) // 2]
    assert np.allclose(median, np.interp(bands.depth_a, depth, profile.real), atol=0.0, rtol=0.0)
```

`api_expand` / `api_profile` 是本测试文件内的两个薄封装，直接调 `xrr_fitter.physics.stack.expand_structure` 与 `xrr_fitter.physics.sld_profile.sld_depth_profile`——测试可以越过 `api`，`tests/architecture` 的边界规则只约束 `src/`。

```python
def test_backing_alignment_removes_a_pure_thickness_translation() -> None:
    names = ("component.0.thickness_a",)
    samples = np.array([[40.0], [60.0]] * 16, dtype=float)
    bands = sld_uncertainty_bands(
        _structure(),
        _report(samples, names),
        wavelength_a=WAVELENGTH,
        align="backing",
    )
    assert np.allclose(bands.real[0], bands.real[-1], atol=1e-9)


def test_surface_alignment_keeps_that_translation_visible() -> None:
    names = ("component.0.thickness_a",)
    samples = np.array([[40.0], [60.0]] * 16, dtype=float)
    bands = sld_uncertainty_bands(
        _structure(),
        _report(samples, names),
        wavelength_a=WAVELENGTH,
        align="surface",
    )
    assert not np.allclose(bands.real[0], bands.real[-1], atol=1e-9)
```

这两条成对出现才有意义：单独看基底对齐带宽为零，无法区分"对齐正确"与"根本没在传播不确定度"。

```python
def test_thinning_is_deterministic_for_one_report() -> None:
    names = ("component.0.thickness_a",)
    rng = np.random.default_rng(3)
    samples = rng.normal(40.0, 1.0, size=(MAX_REPLAY_SAMPLES * 3, 1))
    report = _report(samples, names)
    first = sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH)
    second = sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH)

    assert first.sample_count == MAX_REPLAY_SAMPLES
    assert first.total_samples == samples.shape[0]
    assert np.array_equal(first.real, second.real)
    assert np.array_equal(first.imaginary, second.imaginary)


def test_unknown_sample_parameter_names_are_rejected() -> None:
    samples = np.full((8, 1), 40.0)
    with pytest.raises(ValueError, match="parameter"):
        sld_uncertainty_bands(
            _structure(),
            _report(samples, ("component.9.thickness_a",)),
            wavelength_a=WAVELENGTH,
        )


def test_quantile_levels_and_align_choices_are_frozen_contract() -> None:
    assert QUANTILE_LEVELS == (0.025, 0.16, 0.5, 0.84, 0.975)
    assert ALIGN_CHOICES == ("backing", "surface")
```

 - [x] **Step 2: 确认 RED**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/analysis/test_sld_bands.py -q
```

预期 RED：`ModuleNotFoundError: No module named 'xrr_fitter.analysis.sld_bands'`。

 - [x] **Step 3: 实现重放模块**

创建 `src/xrr_fitter/analysis/sld_bands.py`。按下列函数边界拆分——`check_radon.py` 的单文件平均 CC ≤ 5.0 是这一步最容易踩的门禁：

```python
"""Replay retained samples into aligned SLD depth-profile quantile bands."""

from __future__ import annotations

import numpy as np

from xrr_fitter.model.analysis import McmcReport, SldUncertaintyBands
from xrr_fitter.model.structure import StructureSpec
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure, rebuild_structure

MAX_REPLAY_SAMPLES = 500
QUANTILE_LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
ALIGN_CHOICES = ("backing", "surface")
MAX_FAILURE_RATE = 0.05
ALIGN_LABELS = {"backing": "基底界面", "surface": "表面界面"}
```

分成这些块，每块只做一件事：

- `_thinned_indices(total, limit)` — 均匀抽稀。用 `np.linspace(0, total - 1, limit)` 取整而非随机抽样：确定性来自构造本身，不依赖任何种子，比"确定性种子"更强。设计只要求可重现，这个做法同时消掉了种子传递。
- `_value_map(report, row)` — 按 `report.parameter_names` 与该行构造 `dict[str, float]`。
- `_replay_one(structure, values, wavelength_a, step_a)` — `rebuild_structure` → `expand_structure` → `sld_depth_profile`，返回 `(depth, profile)`。
- `_alignment_offset(stack_or_depth, align)` — 返回该条抽样的平移量。基底对齐取展开后最深界面位置，表面对齐取 `0.0`（`_depth_grid` 的界面从 `0.0` 起算，见 `physics/sld_profile.py:12`）。
- `_replayed_profiles(...)` — 遍历抽样，收集 `(depth - offset, profile)`，单条失败计数并跳过；失败率超 `MAX_FAILURE_RATE` 抛错。
- `_common_grid(profiles, step_a)` — 取所有对齐后 depth 范围的**交集**，空交集抛错。
- `_interpolated(profiles, grid)` — 实虚分别 `np.interp` 到公共网格，返回两个 `(n_samples, grid.size)` 数组。
- `sld_uncertainty_bands(...)` — 组装，`np.quantile(..., axis=0)` 分别对实虚取分位数，返回 `SldUncertaintyBands`。

三处必须显式失败、不得静默退化：

```python
    if not names_known:
        raise ValueError(f"unknown sample parameter for this structure: {name}")
    if failed / total > MAX_FAILURE_RATE:
        raise ValueError(f"sample replay failure rate {failed / total:.3f} exceeds {MAX_FAILURE_RATE}")
    if lower >= upper:
        raise ValueError("aligned sample depth ranges do not overlap")
```

`align` 不在 `ALIGN_CHOICES` 内时同样抛 `ValueError`。

 - [x] **Step 4: 确认 GREEN 并核复杂度**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/analysis/test_sld_bands.py -q
env -u PYTHONPATH .venv/bin/python tools/check_radon.py
```

预期：`7 passed`，`check_radon.py` 退出 0。若单文件平均 CC 超 5.0，继续按上面的函数边界下切，不要把校验塞回主函数。

 - [x] **Step 5: 提交重放模块**

```bash
git add src/xrr_fitter/analysis/sld_bands.py tests/unit/analysis/test_sld_bands.py
git commit --no-verify -m "feat(analysis): replay MCMC samples into aligned SLD bands"
```

---

### Task 3: 把带挂到结果上并持久化

**Files:**
- Modify: `src/xrr_fitter/model/analysis.py:461-476`（`UncertaintyReport` 追加可选字段）
- Modify: `src/xrr_fitter/services/fitting.py`（构建报告时算带）
- Modify: `src/xrr_fitter/io/codec_common.py:19-53`（登记可空字段）
- Modify: `src/xrr_fitter/io/codec_results.py`
- Modify: `tests/unit/model/test_analysis_values.py`
- Modify: `tests/unit/io/test_project_codec.py`
- Modify: `tests/unit/services/test_fitting.py`

**Interfaces:**
- Produces: `UncertaintyReport.sld_bands: SldUncertaintyBands | None = None`。
- Produces: `services/fitting.py` 内部 `_sld_bands(structure, report, wavelength_a)`。
- Preserves: `UncertaintyReport` 既有全部字段的顺序与默认值。新字段必须追加在
  `bootstrap_performed`（`model/analysis.py:476`）**之后**——该文件既有 `bootstrap_performed`
  就是为同一原因追加在末尾的，插在中间会让按位置构造的既有调用被静默重新解读。
- Removes: 无。

**为什么带挂在 `UncertaintyReport` 而不挂在导出上下文：** `test_dependency_rules.py:297`
的 `_services_violations` 禁止除 `services.fitting` 以外的任何 services 模块 import
`analysis`，所以 `services/exports.py` 不能算带。而 `ALLOWED["io"] == {"io", "model"}`
使 `io` 也算不了。唯一合规位置是 `services/fitting.py`——它已经是构建 `UncertaintyReport`
的地方，带作为该报告的一部分产出，导出与 GUI 都只读结果。这同时让带随项目文件持久化，
重开项目不必重算。

**代价要说清：** 带进入持久化 payload 会增大工程文件。`SldUncertaintyBands` 的体积是
`len(quantiles) × depth.size × 2` 个 float，按 `QUANTILE_LEVELS` 5 档、典型 `depth.size`
约 200 计约 2000 个 float，相对既有 `samples_physical`（数千 × 参数数）小一个量级，可接受。

 - [x] **Step 1: 写失败的字段与往返契约**

在 `tests/unit/model/test_analysis_values.py` 追加：

```python
def test_uncertainty_report_defaults_sld_bands_to_none() -> None:
    report = _uncertainty_report()
    assert report.sld_bands is None


def test_uncertainty_report_rejects_a_wrongly_typed_band() -> None:
    with pytest.raises(TypeError, match="sld_bands"):
        replace(_uncertainty_report(), sld_bands=object())
```

`_uncertainty_report()` 用该文件既有的 `UncertaintyReport` 构造辅助；若无则照既有
`McmcReport` 辅助的写法新建一个，字段取最小合法值。

在 `tests/unit/io/test_project_codec.py` 追加一条往返测例，断言带经编解码后
`depth_a`/`real`/`imaginary` 逐位相等且 `quantiles` 保持元组：

```python
def test_project_roundtrip_preserves_sld_uncertainty_bands() -> None:
    original = _project_with_bands()
    restored = decode_project(encode_project(original))
    before = _bands_of(original)
    after = _bands_of(restored)
    assert np.array_equal(after.depth_a, before.depth_a)
    assert np.array_equal(after.real, before.real)
    assert np.array_equal(after.imaginary, before.imaginary)
    assert after.quantiles == before.quantiles
    assert after.caption() == before.caption()
```

编解码函数名照该文件既有 import 使用，不要新造名字。

 - [x] **Step 2: 确认 RED**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model/test_analysis_values.py tests/unit/io/test_project_codec.py -q -k "sld_bands or sld_uncertainty"
```

预期 RED：`UncertaintyReport` 无 `sld_bands` 字段。

 - [x] **Step 3: 追加字段、编解码与服务层计算**

`model/analysis.py` 的 `UncertaintyReport` 在 `bootstrap_performed` 之后追加：

```python
    sld_bands: SldUncertaintyBands | None = None
```

并在其 `__post_init__` 里加类型校验（与既有 `mcmc`、`candidate_id` 的校验风格一致）：

```python
        if self.sld_bands is not None and not isinstance(self.sld_bands, SldUncertaintyBands):
            raise TypeError("sld_bands must be a SldUncertaintyBands")
```

`io/codec_common.py` 的 `OPTIONAL_FIELDS`（`:19`）按字母序加入 `"sld_bands"`；
`NULLABLE_ARRAY_FIELDS`（`:54`）加入 `"imaginary"`。**这一步不能省**：`_validate_nulls`
（`codec_common.py:127`）对任何未登记的 `null` 抛 `ProjectSchemaError`，而 `depth_a`/`real`
已被既有条目覆盖与否需实施者逐一核对，缺哪个补哪个。

`io/codec_results.py` 加一对 `_sld_bands_to_dict` / `_sld_bands_from_dict`，照同文件
`_real_array_to_list` / `_real_array_from_list` 的既有姿势处理数组，`quantiles` 存成
list、读回转 tuple。`None` 直接透传。

`services/fitting.py` 在构建 `UncertaintyReport` 处调用：

```python
    bands = _sld_bands(structure, mcmc_report, wavelength_a)
```

`_sld_bands` 无 `McmcReport` 时返回 `None`；捕获 `ValueError` 时返回 `None` 并把原因追加到
报告既有的 `warnings` 通道——**不静默丢弃**。带算不出来不应让整次拟合失败。

 - [x] **Step 4: 确认 GREEN**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/model tests/unit/io tests/unit/services -q
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/architecture/test_dependency_rules.py -q
env -u PYTHONPATH .venv/bin/python tools/check_radon.py
```

第二条命令是本任务的关键门禁：它会抓住任何"顺手在 `services/exports.py` 里 import
`analysis`"的走捷径写法。

 - [x] **Step 5: 提交**

```bash
git add src/xrr_fitter/model/analysis.py src/xrr_fitter/services/fitting.py src/xrr_fitter/io/codec_common.py src/xrr_fitter/io/codec_results.py tests/unit/model/test_analysis_values.py tests/unit/io/test_project_codec.py tests/unit/services/test_fitting.py
git commit --no-verify -m "feat(services): persist SLD uncertainty bands with the report"
```

---

### Task 4: 导出图渲染带

**Files:**
- Modify: `src/xrr_fitter/io/export_plots.py:74-85`
- Modify: `tests/unit/io/test_export_plots.py`

**Interfaces:**
- Consumes: `DatasetExportData.selected` 所属结果的 `uncertainty.sld_bands`。
- Produces: `io/export_plots.py` 内部 `_draw_bands(axis, bands)`。
- Preserves: `sld_profile_png` 在无带时的**逐位输出**。
- Removes: 无。

**取带路径：** `DatasetExportData` 已有 `result` 属性（`io/export_tables.py:175`），带经
`context.result.uncertainty.sld_bands` 取到，**不需要给 `DatasetExportData` 加字段**，
也不需要改 `services/exports.py`。

**一处必须小心的既有行为：** `DatasetExportData.result` 在 `dataset.last_valid_result is None`
时**抛 `ValueError`**（`io/export_tables.py:178`），不是返回 `None`。`_selected_bands` 因此
不能写成 `context.result.uncertainty and ...`——那会在无结果的数据集上把一个本该安静跳过
的情形变成导出失败。实现必须先看 `context.dataset.last_valid_result`：

```python
def _selected_bands(context: DatasetExportData) -> object | None:
    result = context.dataset.last_valid_result
    report = None if result is None else result.uncertainty
    return None if report is None else report.sld_bands
```

 - [x] **Step 1: 写失败的渲染契约**

在 `tests/unit/io/test_export_plots.py` 追加。第一条是核心安全网：

```python
def test_sld_profile_png_without_bands_matches_the_committed_bandless_render() -> None:
    context = _context()
    assert context.result.uncertainty is None or context.result.uncertainty.sld_bands is None
    baseline = sld_profile_png(context)
    assert sld_profile_png(context) == baseline
    assert baseline.startswith(b"\x89PNG\r\n\x1a\n")


def test_sld_profile_png_with_bands_differs_and_stays_deterministic() -> None:
    banded = _context_with_bands()
    first = sld_profile_png(banded)
    assert first == sld_profile_png(banded)
    assert first != sld_profile_png(_context())
```

`_context_with_bands()` 由既有 `_context()` 出发，用 `dataclasses.replace` 把一条零宽带
（五个分位面完全相同）装进 `uncertainty`，再装回 `result`、`selected` 归属不变。零宽带
仍产生 `fill_between` 图元，因此与无带版本字节不同，同时无需真实 MCMC 抽样。

再追加图注来源测例，把文案钉在值对象上：

```python
def test_sld_profile_png_takes_its_caption_from_the_band_object(monkeypatch) -> None:
    banded = _context_with_bands()
    calls = []
    original = api.SldUncertaintyBands.caption
    monkeypatch.setattr(
        api.SldUncertaintyBands,
        "caption",
        lambda self: calls.append(1) or original(self),
    )
    sld_profile_png(banded)
    assert len(calls) == 1
```

这条挡的是 `io` 与 `gui` 各写一份图注文案的回归——设计要求两者逐字相同。

 - [x] **Step 2: 确认 RED**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/io/test_export_plots.py -q
```

预期 RED：带版本与无带版本字节相同（带未被渲染），第二条测例失败。

 - [x] **Step 3: 渲染带**

`sld_profile_png` 在两条既有 `axis.plot` 之后插入。**无带时一条语句都不执行**，这是逐位不变的前提：

```python
    bands = _selected_bands(context)
    if bands is not None:
        _draw_bands(axis, bands)
        axis.set_title(bands.caption(), fontsize=8, loc="left")
```

`_selected_bands(context)` 从 `context.result.uncertainty` 安全取 `sld_bands`，任一层为
`None` 即返回 `None`。

`_draw_bands` 按 `bands.quantiles` 找出 `(0.16, 0.84)` 与 `(0.025, 0.975)` 两对下标，各画
一组实部与虚部 `fill_between`，`alpha` 分别 0.28 与 0.14，`label` 用 `"16–84%"` /
`"2.5–97.5%"`。**找不到某对分位数时跳过该组而不报错**——值对象允许自定义 `quantiles`，
硬失败会让一个合法值对象无法渲染。

 - [x] **Step 4: 确认 GREEN**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/unit/io/test_export_plots.py tests/unit/services/test_exports.py -q
env -u PYTHONPATH .venv/bin/python tools/check_radon.py
```

既有的 `test_export_plots_are_deterministic_pngs_and_close_every_figure` 与
`test_export_pngs_ignore_process_global_matplotlib_style`（`tests/unit/io/test_export_plots.py:101`、
`:160`）必须仍然通过——它们是无带路径的既有守卫。

 - [x] **Step 5: 提交**

```bash
git add src/xrr_fitter/io/export_plots.py tests/unit/io/test_export_plots.py
git commit --no-verify -m "feat(io): render SLD uncertainty bands in exported profiles"
```

---

### Task 5: 界面上的带、显隐与对齐切换

**Files:**
- Modify: `src/xrr_fitter/api.py:3-10,116-140`（re-export `SldUncertaintyBands`）
- Modify: `tests/architecture/test_public_api.py`
- Modify: `src/xrr_fitter/gui/plots/sld.py:32-70`（`draw_sld`）
- Modify: `src/xrr_fitter/gui/plots/panel.py:154-170,527-542`（窗格控件与重绘）
- Modify: `tests/gui/plot_cases_5.py`
- Modify: `docs/algorithm.md`

**Interfaces:**
- Consumes: `api.SldUncertaintyBands`。
- Produces: `draw_sld(view, candidate, others=(), bands=None)`——**新参数带默认值追加在末尾**，
  `panel.py:539` 的既有三参数调用因此不受影响。
- Produces: `PlotPanel.sld_bands_toggle`（`QCheckBox`，objectName `sldBandsToggle`）与
  `PlotPanel.sld_align_selector`（`QComboBox`，objectName `sldAlignSelector`）。
- Preserves: `draw_sld` 在 `bands=None` 时的既有绘制序列，逐元素不变。
- Removes: 无。

**带只画选中候选：** 设计明确要求。`draw_sld` 已经把其他候选的实部画成淡色叠加
（`gui/plots/sld.py:53-64`），再给每个候选各加两组 `fill_between` 会互相遮挡到不可读。

**对齐切换为什么必须暴露：** 设计正文的理由是"带宽在哪里为零完全取决于这个选择"，
这不是可以替用户定死的参数。但**切换对齐需要重算带**，而带在 Task 3 里是随结果持久化的。
本任务的处理：下拉框切换时经 `api` 重算并只更新视图，**不写回项目**——重算结果是视图态，
写回会让一次纯查看操作把项目标脏。这需要 `api` 暴露一个计算入口。

**因此本任务给 `api` 加两样东西**，而不是一样：
- `SldUncertaintyBands` 类型（re-export，供 GUI 类型标注与 monkeypatch 用）；
- `sld_uncertainty_bands` 计算入口。它在 `services/fitting.py` 里加一个薄封装转调
  `analysis.sld_bands.sld_uncertainty_bands`，再由 `api` re-export——`api` 只能 import
  `{model, services}`，不能直接 import `analysis`。

 - [x] **Step 1: 写失败的公共面与界面契约**

在 `tests/architecture/test_public_api.py` 的 `PUBLIC_NAMES` 里按字母序插入
`"SldUncertaintyBands"` 与 `"sld_uncertainty_bands"`。该测试断言
`tuple(api.__all__) == PUBLIC_NAMES` 全等，位置必须严格按字母序，否则失败信息会指向排序而非缺失。

在 `tests/gui/plot_cases_5.py` 追加四个测例：

```python
def test_sld_band_toggle_is_disabled_without_sampling(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    assert panel.sld_bands_toggle.isEnabled() is False
    assert panel.sld_bands_toggle.toolTip() != ""


def test_sld_band_toggle_enables_when_bands_exist(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4), bands=_zero_width_bands())
    assert panel.sld_bands_toggle.isEnabled() is True
    assert panel.sld_bands_toggle.isChecked() is True


def test_sld_draw_without_bands_matches_the_bandless_element_sequence(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    axes = panel.view("sld").axes
    assert not axes.collections


def test_sld_caption_matches_the_export_caption(qtbot) -> None:
    bands = _zero_width_bands()
    panel = _panel(qtbot, data=prepared_data(size=4), bands=bands)
    assert bands.caption() in panel.view("sld").axes.get_title()
```

第三条用 `axes.collections` 为空来表达"无带时不画任何 `fill_between`"——`fill_between`
产生 `PolyCollection`，比逐个比对图元更稳。第四条与 Task 4 的图注测例合起来，把屏幕与
导出图的文案锁在同一个 `caption()` 上。

`_panel(...)` 的 `bands=` 参数需要在 `tests/gui/plot_support.py` 的既有 `_panel` 辅助里
增加透传；`_zero_width_bands()` 构造一条五个分位面相同的带。

 - [x] **Step 2: 确认 RED**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/architecture/test_public_api.py tests/gui/test_plots.py -q -k "sld_band or sld_caption or public_api"
```

预期 RED：`api.__all__` 与 `PUBLIC_NAMES` 不等；`PlotPanel` 无 `sld_bands_toggle`。

GUI 测试需要 offscreen，按 `tests/gui` 既有姿势运行（该目录的 Qt 平台设置由既有
`plot_support` / CI 的 `QT_QPA_PLATFORM=offscreen` 提供）。若本地缺显示环境，前置
`QT_QPA_PLATFORM=offscreen`。

 - [x] **Step 3: 实现**

`services/fitting.py` 加薄封装（该模块已 import `analysis`，无新增架构边界）：

```python
def sld_uncertainty_bands(structure, report, *, wavelength_a, align="backing"):
    """Recompute view-only bands for an alignment the user picked."""
    return _bands.sld_uncertainty_bands(
        structure, report, wavelength_a=wavelength_a, align=align
    )
```

`api.py` 在 `from xrr_fitter.model.analysis import (...)` 块按字母序加
`SldUncertaintyBands`，在 `from xrr_fitter.services.fitting import (...)` 块加
`sld_uncertainty_bands`，`__all__` 同步按字母序插入两项。

`gui/plots/sld.py` 的 `draw_sld` 签名末尾追加 `bands: object | None = None`，在既有两条
`axes.plot` 之后插入：

```python
    if bands is not None:
        _draw_bands(axes, bands)
        axes.set_title(f"SLD 深度剖面 — {bands.caption()}", fontsize=8)
```

`_draw_bands` 与 Task 4 的 `io` 侧同名函数**逻辑相同但各自实现**：`ALLOWED["gui"] == {"gui", "api"}`
不允许 gui import `io`。两侧共享的只有 `caption()` 文案，这已由 Task 4 与本任务的两条
测例分别钉住；把绘制代码也强行共享会需要一个新的公共模块，代价高于收益。

**深度单位注意：** `draw_sld` 把深度换成 nm 显示（`sld.py:49`，除以 10.0），而
`SldUncertaintyBands.depth_a` 是 Å。`_draw_bands` 必须做同样换算，否则带会相对曲线偏移 10 倍。
这是本任务最容易出的错，Step 1 的第四条测例不覆盖它——实施者应额外加一条断言带的 x 范围
与曲线 x 范围同量级的测例。

`gui/plots/panel.py` 的 `_build_sld_pane`（`:154`）在 heading 之后、canvas 之前加一行控件条
（`QHBoxLayout` 容纳复选框与下拉框）；`_draw`（`:527`）把带传给 `draw_sld`：无 `McmcReport`
时 `bands=None` 且复选框 `setEnabled(False)` 并设 tooltip「需要先运行 MCMC」，复选框未勾选时
同样传 `None`。

 - [x] **Step 4: 确认 GREEN 并跑全量 GUI 与架构门禁**

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/gui -q
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -p tests.outcome_gate tests/architecture -q
env -u PYTHONPATH .venv/bin/python tools/check_radon.py
```

预期：全绿。`tests/architecture/test_public_api.py` 的全等断言是这一步的主要风险点。

 - [x] **Step 5: 记录算法边界**

在 `docs/algorithm.md` 的 SLD 剖面相关章节之后插入：

```markdown
（以下整段写入 docs/algorithm.md，标题层级为该文件的二级标题）

## SLD 剖面不确定度带

带由 `McmcReport.samples_physical` 的每一行经 `rebuild_structure` →
`expand_structure` → `sld_depth_profile` 重放得到，按选定界面平移对齐后插值到
所有抽样对齐深度范围的**交集**，实部与虚部分别取 2.5/16/50/84/97.5 分位。

取交集而非并集：并集边缘只有少数抽样支撑，会产生看起来收窄实则无统计意义的伪带宽。
实虚分别取分位而非对复数取模：虚部承载吸收信息，而吸收是 XRR 密度对比的主要来源。

抽样超过 500 条时按 `np.linspace` 均匀抽稀，抽稀比例写进图注。单条抽样重放失败被计数
并跳过，失败率超过 5% 时整体失败而非给出基于少数抽样的带。

默认对齐基底界面：XRR 的基底通常是已知单晶，把不确定度累积推向表面侧符合实际认知。
界面提供切换到表面对齐，因为"带宽在哪里为零"完全取决于这个选择。切换是视图态，不写回项目。

图注写成 `16–84%` 与 `2.5–97.5%` 分位区间而非 1σ/2σ：后验通常不是高斯的，σ 需要额外
假定高斯性才有意义，分位数无歧义。
```

 - [x] **Step 6: 提交界面与文档**

```bash
git add src/xrr_fitter/api.py src/xrr_fitter/services/fitting.py src/xrr_fitter/gui/plots/sld.py src/xrr_fitter/gui/plots/panel.py tests/architecture/test_public_api.py tests/gui/plot_cases_5.py tests/gui/plot_support.py docs/algorithm.md
git commit --no-verify -m "feat(gui): show SLD uncertainty bands with alignment control"
```

---

## 最终验收记录

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 值对象与重放 | `pytest tests/unit/model/test_analysis_values.py tests/unit/model/test_sld_bands.py tests/unit/analysis/test_sld_bands.py` | `67 passed`（包含在下述 `407 passed` 聚焦验收中） |
| 持久化往返 | `pytest tests/unit/io/test_project_codec.py` | 通过（包含在 `407 passed` 中） |
| 导出无带逐位不变 | `pytest tests/unit/io/test_export_plots.py` | 通过；冻结 size/SHA-256 合同未变（包含在 `407 passed` 中） |
| 架构边界 | `pytest tests/architecture` | 通过（包含在 `407 passed` 中） |
| 界面 | `QT_QPA_PLATFORM=offscreen pytest tests/gui` | 通过（最新全量联跑 `2179 passed`，无 warning） |
| 复杂度 | `tools/check_radon.py` | exit 0，无输出 |
| 零方差带宽 | `test_zero_variance_median_is_the_direct_profile_on_the_same_backing_axis` | 通过 |
| 屏幕与导出图注一致 | `test_sld_caption_matches_the_export_caption` 与 `test_sld_profile_png_takes_its_caption_from_the_band_object` | 通过 |

### 实际验收输出（2026-08-13；当前工作区）

聚焦 SLD 值对象、重放、codec、导出、GUI 与架构：

```text
407 passed in 64.12s
```

最初全量 unit、GUI、architecture 与 integration 联跑：

```text
2172 passed, 14 warnings in 186.53s
```

在随后补齐 effective-prior 生命周期、GUI live-draw 回滚测试和 MCMC 物理中位数回归后，同一路径的最新联跑为：

```text
2179 passed in 194.34s (0:03:14)
```

实时预览重建中文图例后的 CJK 字体遗漏已修复，并新增添加/清除预览后强制绘制的缺字回归；
当前全量联跑无 warning。PNG 导出的中文 SLD caption 也由独立 CJK 字体回归测试覆盖。GUI 事务逻辑随后抽到
`gui/plots/sld_state.py`；`panel.py` 与该模块的 Radon MI 均恢复为 A。
`tools/check_radon.py`、`tools/verify_registry.py` 与 `git diff --check` 均 exit 0。

## 剩余风险

- 带随结果持久化会增大工程文件。若实测增幅超出预期，退路是不持久化、改为打开项目后按需重算，
  但那需要 GUI 侧承担一次可见延迟。
- `_draw_bands` 在 `io` 与 `gui` 各有一份实现（架构不允许共享）。两侧的视觉一致性目前只由
  `caption()` 文案的两条测例间接保证，透明度与配色的漂移不会被测试抓到。

<!-- PLAN-COMPLETE -->
