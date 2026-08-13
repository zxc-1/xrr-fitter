# ORSO 导出实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个数据集的导出批次可以额外产出一个自足的 `fit_result.ort`，携带数据、模型、参数、误差棒、协方差与 `ConfidenceClass` 证据；标准段严格通过 ORSO 的 JSON schema 校验，本项目独有的置信度分级放进 `xrr_fitter.confidence` 扩展命名空间。不开启 `.ort` 时，全套测试与磁盘产物与 HEAD 逐位相同。

**Architecture:** `.ort` 是导出批次里的一个 artifact，不是独立导出动作。`services/exports.py::_dataset_artifacts` 在 `ArtifactPayload` 元组里条件追加一项，之后完全复用 `publish_export_run` 的目录级原子发布——写入、fsync、rename 都不需要新代码（见修正 7）。

协方差标度按 spec 的方案 A 落在 `UncertaintyReport.parameter_sigma`，是纯增可选字段。重建协方差的数值组合放 `analysis/derivatives.py`，由 `services/exports.py` 调用后把矩阵传给 `io/orso.py`——因为 `ALLOWED["io"] = {"io", "model"}` 不含 `analysis`，`io/orso.py` 自己不能调重建函数（见修正 9）。

**Tech Stack:** Python 3.12、NumPy 2.x、orsopy 1.2.3、jsonschema 4.26.0、PySide6 6.8+、pytest 8.3+。**不新增依赖**——两个依赖已完整声明并入 lock（见修正 4）。

**Design source:** `docs/superpowers/specs/2026-08-09-orso-export-design.md`

---

## 对 spec 的修正

写这份 plan 时逐条核对了 spec 与 HEAD 代码，有十处不符。其中修正 1 是 spec 完全漏掉的硬阻塞，修正 2 是一条数学上不可能成立的验收标准，修正 3 是 spec 自相矛盾。**执行时以本节为准，不以 spec 为准。**

### 修正 1（硬阻塞，spec 完全未提）：架构门禁不认识 `orsopy`，新建 `io/orso.py` 会直接失败

`tests/architecture/test_dependency_rules.py` 的 docstring 写明第三方边界是**穷举白名单**：

> Third-party boundaries: - **the seven declared roots form an exhaustive allowlist**; - each root is restricted to its declared module owners

```python
THIRD_PARTY_ROOTS = {"numpy", "scipy", "periodictable", "pandas", "xlsxwriter", "matplotlib", "PySide6"}
```

`orsopy` 不在其中。`:313` 的判定是 `if root not in THIRD_PARTY_ROOTS or not _third_party_allowed(root, module):` → 违规。也就是说 `io/orso.py` 第一行 `import orsopy` 落地的瞬间，`tests/architecture` 就红，而 spec 里一个字都没提这件事。

`:90-113` 的注释给出了扩容契约，必须逐条满足：

- 例外是「exact directed module edges, never owner-wide grants」
- 「Any future exception requires its own exact mapping and **three-way fixture**」

因此 Task 1 是**注册**而不是绕过：`THIRD_PARTY_ROOTS` 加 `"orsopy"`，并用 `THIRD_PARTY_MODULE_ALLOWLIST["orsopy"] = {"io.orso"}` 精确到单模块。键格式是不带 `xrr_fitter.` 前缀的点分模块名，与 `:153` 的 `"periodictable": {"physics.materials"}` 同型。

**不要**加进 `THIRD_PARTY_OWNER_ALLOWLIST`（`:146`）——那是 owner-wide 授权，会让整个 `io` 包都能 import orsopy。也不需要 `THIRD_PARTY_PREFIX_ALLOWLIST`（`:158`），那是给 `gui.plots.` 那种子模块树用的，单模块用不上。

`jsonschema` **不需要** owner 条目：它只在 orsopy 的 `_validate_header_data` 函数体内被 import，本项目任何 `src/` 文件都不直接 import 它，静态扫描抓不到。

### 修正 2（推翻一条验收标准）：协方差逐位重建在数学上不可能

spec 的验证节要求：

> 协方差重建测例：`sigma[:,None] * correlation * sigma[None,:]` 与 `report.py:166` 的 `physical_covariance` 逐位相等。

这条无法成立。`analysis/derivatives.py:125` 的 `correlation_from_covariance` 是**有损**的：

```python
diagonal = np.clip(np.diag(values), 0.0, np.inf)
scale = np.sqrt(diagonal)
denominator = scale[:, None] * scale[None, :]
correlation = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0.0)
correlation = np.clip(correlation, -1.0, 1.0)
correlation[np.diag_indices_from(correlation)] = np.where(diagonal > 0.0, 1.0, 0.0)
```

四处不可逆：
1. `where=denominator > 0.0` 把非正对角对应的整行整列**零化**；
2. `np.clip(correlation, -1.0, 1.0)` 截断超界元素（pinv 数值噪声下会出现）；
3. 对角被**覆写**成精确 1.0/0.0，原对角信息只剩 `sigma` 里那份；
4. 一次浮点除法 + 后续一次浮点乘法，往返本身就带舍入。

**改成：** 断言只在正定子块上成立，且用显式容差 `np.allclose(..., rtol=0.0, atol=1e-12)`，并在测例里注释写清为什么不能用 `==`。非正对角的坐标（锁定参数、pinv 零模）在 `.ort` 里协方差元素写 0，与 correlation 的既有语义一致。

这不影响 spec 的其他逐位要求：往返测例（第 3 层验证）里数据列与参数值仍必须逐位相等，因为它们不经过任何归一化。

### 修正 3（spec 自相矛盾）：CLI 只能取一种读法

spec 同时写：

- 代码边界节：「CLI 增加 `--format ort`」
- 界面节：「`.ort` 是随导出批次一起产出的 artifact，**不是独立的导出动作**」

`--format ort` 的语义是「换一种导出格式」，与后者互斥。而且 `cli/main.py:53-55` 的 `export` 子命令目前没有任何 `--format`：

```python
export = subparsers.add_parser("export", help="发布已有结果")
export.add_argument("project")
export.add_argument("output_dir")
```

**取 opt-in flag：`export --ort`**，语义是「本批次额外产出 `.ort`」。理由：spec 的 GUI 一节已经假设了一个复选框（「导出前让用户选是否产出 `.ort`」「默认勾选」），复选框对应的就是布尔 opt-in，不是格式切换。

**默认值分两层，不要混：**
- API/CLI 层 `include_ort: bool = False`。不传时 `_dataset_artifacts` 的元组与 HEAD 逐位相同，既有 `tests/unit/services/test_exports.py` 与 `tests/unit/io/test_export_run.py` 的清单断言全部不动。
- GUI 复选框默认勾选（spec 的理由成立：零成本，忘勾要重跑）。这是 GUI 的呈现默认，不是 API 默认。

### 修正 4（降级为只读核验）：依赖已就位，没有新增依赖，不触发授权门禁

spec 要求「改 `pyproject.toml`」并「按 `tools/lock_environment.py` 与 `tools/lock_windows_environment.py` 重新生成两个平台的 lock」。**这些都已经做完了**，本轮一行都不用改：

- `pyproject.toml:20-21`：`orsopy>=1.2,<2`、`jsonschema>=4.0,<5`，均在生产 `dependencies` 里。
- 两个 lock 均已 pin：`orsopy==1.2.3`、`jsonschema==4.26.0`，连传递依赖 `attrs==26.1.0` / `jsonschema-specifications==2025.9.1` / `referencing==0.37.0` / `rpds-py==2026.6.3` 都在。
- `verification/release-spec.json` 的 `runtime_dependencies` 已含两者，顺序与 `pyproject` 一致（`tests/architecture/test_distribution.py:253` 断言 `tuple(spec["runtime_dependencies"]) == tuple(project["dependencies"])`）。
- `packaging/windows/xrr-fitter.spec:24` 已有 `*collect_data_files("orsopy", subdir="fileio/schema")`，`:34` 已 `excludes=["orsopy.slddb"]`。

因此：**不重新生成 lock**（会改 `lock_sha256: 577e7abb...` 并牵动 release-spec 门禁），全局规则里「新增生产依赖前必须先说明取舍并获得确认」这一条**不触发**——依赖不是本轮新增的。Task 0 只做只读核验，确认这些事实仍然为真。

当前状态是「已声明、已入 lock、零消费者」：`src/`、`tests/`、`tools/` 下没有任何文件 import orsopy 或 jsonschema，`src/xrr_fitter/io/orso.py` 不存在。

### 修正 5：`parameter_sigma` 是 checkpoint 安全的，但字段位置与默认值有硬约束

`fit/checkpoint.py:24` 的 `POST_FREEZE_OMITTED_DEFAULTS` 只覆盖三个类型：

```python
POST_FREEZE_OMITTED_DEFAULTS: dict[tuple[str, str], object] = {
    ("ParameterDefinition", "prior"): None,
    ("ConfidenceThresholds", "prior_conflict_sigmas"): 3.0,
    ("LayerSpec", "transition"): None,
}
```

`UncertaintyReport` **不在 checkpoint 图里**，所以给它加字段不会漂移指纹，不需要新增 `POST_FREEZE_OMITTED_DEFAULTS` 条目。（对比：`expression-constraints` 要动 `ParameterDefinition`，那个才需要。）

但仍有三条硬约束：
1. **位置必须在最后**，即 `prior_conflicts: tuple[str, ...] = ()` 之后。前面全是位置参数，插中间会静默改变所有 positional 构造点的语义。
2. **`slots=True` 禁止 mutable 默认值**，`np.array([])` 不能做 default。用 `parameter_sigma: np.ndarray | None = None`，与既有 `mcmc: McmcReport | None = None`、`sld_bands: SldUncertaintyBands | None = None` 同风格。spec 说的「默认空」按 `None` 实现。
3. `__post_init__` 里非 None 时走现成的 `_readonly(self.parameter_sigma, float, "parameter_sigma", 1)`（`model/analysis.py:71`）并校验 `len == len(names)`，然后 `object.__setattr__` 写回——与 `correlation_matrix` 的既有处理同型。

### 修正 6：`_correlation_evidence` 在内部丢弃了 `physical_covariance`，返回值要扩宽

`analysis/report.py:176` 的签名是三元组：

```python
def _correlation_evidence(problem, unit_vector, names) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str, float], ...]]:
    ...
    physical_covariance = physical_jacobian @ unit_covariance @ physical_jacobian.T
    correlation = correlation_from_covariance(physical_covariance)
    ...
    return correlation, boundary_hits, strong
```

`physical_covariance` 是局部变量，出了函数就没了；唯一调用点在 `:265`：

```python
correlation, boundary_hits, strong_correlations = _correlation_evidence(problem, unit, names)
```

要拿到 sigma，扩成四元组返回 `sigma`（而不是返回整个协方差矩阵——只需要标度，`sqrt(clip(diag, 0, inf))` 与 `correlation_from_covariance` 内部第 4-5 行完全一致，这样两者的对角处理天然自洽）。调用点同步解包。

### 修正 7（落盘机制与 spec 不同）：不走 `atomic_replace_bytes`，走 `publish_export_run`

spec 写「落盘走既有 `atomic_replace_bytes`，与 `project_codec.py:save_project` 同一路径」。**导出批次不用这个函数。** `io/project_codec.py:551` 的 `atomic_replace_bytes` 是单文件同目录 mkstemp+fsync+replace，只被 `save_project` 用。

导出走的是**目录级原子**：

```python
def publish_export_run(output_dir, datasets, root_files=(), *, run_timestamp=None) -> ExportManifest:
    """Atomically publish one collision-safe export run."""
    ...
    partial, final = _allocate_run(Path(output_dir), timestamp)
    try:
        staged, written = _stage_export(partial, dataset_values, root_values)
        _sync_tree(partial, written)
        _publish_directory(partial, final)
    except BaseException as error:
        _cleanup_after_failure(partial, error)
        raise
```

所以 `.ort` 的落盘**不需要任何新代码**——只要把它作为一个 `ArtifactPayload` 交给 `_dataset_artifacts`。spec 里「schema 校验失败时导出整体失败，不产出部分文件。原子发布保证不留半成品」这一条因此自动成立：`orso_bytes()` 抛异常发生在 `_stage_export` 之前，`partial` 目录都还没建。

**另一个硬约束：`ArtifactPayload` 拒绝空内容**（`io/export_run.py:80`，`"artifact content must not be empty"`）。所以「不产出 `.ort`」必须是**把它从元组里省掉**，绝不能塞一个 `b""`。

spec 的代码边界节还写了「扩 `io/export_run.py` 的 artifact 列表」——`export_run.py` 里没有 artifact 列表，它是通用发布器，元组由 `services/exports.py` 组装。**`io/export_run.py` 一行都不用改。**

### 修正 8（NaN 政策与既有 exporter 不同，且行过滤有现成掩码）

既有 JSON 导出把 NaN 写成 `null`（`io/export_tables.py`）：

```python
def _finite_scalar(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
```

`.ort` 的政策相反——spec 要求非正角度的行**不写入**。别自己重算掩码，`evaluation.py:780-788` 已经算好并且注释说明了它就是为发布保留的：

```python
theta = problem.data.two_theta_deg / 2.0 + values["instrument.angle_offset_deg"]
model_mask = np.isfinite(theta) & (theta > 0.0)
model_indices = np.flatnonzero(model_mask)
# Full source layout is retained for result publication and diagnostic rows.
# Only the compact positive-angle subset enters trigonometric physics.
qz = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
```

在导出上下文里没有 `problem`，等价掩码是对 `.ort` 要写的所有列取 `np.isfinite` 的交集（q、强度、误差、模型、残差），再记录 `len(mask) - mask.sum()` 进扩展段。测例按 spec 构造含非正角度的数据集验证计数。

### 修正 9（spec 自己的边界声明反过来约束了协方差重建的落点）

spec 说「新增 `io/orso.py`，只依赖 `{io, model}`，符合 `ALLOWED["io"]`。协方差重建这类数值组合放 `analysis/`，不放 `io/`」——这两句合起来意味着 `io/orso.py` **不能自己调重建函数**，因为 `ALLOWED["io"] = {"io", "model"}` 不含 `analysis`。

落点：
- `analysis/derivatives.py` 新增 `covariance_from_correlation(sigma, correlation) -> np.ndarray`，与 `correlation_from_covariance` 相邻，测试进 `tests/unit/analysis/test_derivatives.py`。
- `io/orso.py` 的写入函数签名接受**已重建好的**矩阵：`orso_bytes(context: DatasetExportData, *, covariance: np.ndarray | None) -> bytes`。
- `services/exports.py` 做组合（`ALLOWED["services"]` 含 `analysis` 与 `io`，合规）。

### 修正 10：spec 的行号锚点与两处来源标注已漂移

对着 HEAD 逐条核，spec 引用的位置多数已偏，实施时按下表找真实落点，不要照 spec 的行号跳转：

| spec 的标注 | HEAD 实际位置 |
| --- | --- |
| `analysis/report.py:166` 算出 `physical_covariance` | `analysis/report.py:186`（`:166` 在 `_pick_best_candidate` 里） |
| `model/analysis.py:464` 是 `UncertaintyReport` | `model/analysis.py:463` |
| `model/analysis.py:237` 是 `ConfidenceClass` | `model/analysis.py:234` |
| `io/codec_results.py:132` 是持久化点 | `_uncertainty_to_dict` 在 `io/codec_results.py:169`，`_uncertainty_from_dict` 在 `:201` |
| 数据段来源是「`io/xy.py` 的 `DataRecord`」 | **`DataRecord` 不存在。**`load_export_data`（`services/exports.py:36`）经 `_prepared_current` 给出的是 `model/data.py:217` 的 `PreparedData` |
| `SERVICE_SEED_TREE_VERSION` 来自 `model/fitting.py` | 定义在 `services/datasets.py:60`（值为 `1`）。**但 `io/orso.py` 不能 import `services`**，要从 `context.replay_identity.service_seed_tree_version` 读（`ExportReplayIdentity` 在 `io/export_tables.py:136`） |

顺带记录一个既有瑕疵，**本轮不要动**：`services/exports.py::_contexts` 构造 `ExportReplayIdentity(1, ...)` 用的是字面量 `1`，不是 `SERVICE_SEED_TREE_VERSION` 常量。两者当前同值，`.ort` 从 context 读到的就是 1，与常量一致。修它属于另一件事。

---

## Global Constraints

- **未开启时逐位相同。** `include_ort=False`（默认）下，`_dataset_artifacts` 返回的元组与 HEAD 完全一致，导出目录内容不变。断言用 `==`，不用 `approx`。
- **不改 `correlation_matrix` 的语义或任何消费方**（spec 非目标节）。`parameter_sigma` 是纯增。
- **新字段的 emit 走「非空才写」**：旧结果（`parameter_sigma is None`）重新编码后与 HEAD 字节相同，旧工程文件仍能读回。这是 `prior_conflicts` 已经确立的模式，`codec_results.py:191-193` 有现成注释可参照。
- **schema 校验的 orsopy 私有名调用点只允许一处**，在 `io/orso.py` 内，且必须带注释说明依赖私有 API、orsopy 升级时这里是首个检查点。
- **不做导入器**（spec 非目标）。第 3 层「往返」验证用 orsopy 自己的读取器读回做断言，不在 `src/` 里加读取路径。
- **不导出 MCMC 原始链**（spec 非目标），只导出派生区间与协方差。
- 测试命令必须带 `--import-mode=importlib`：
  `.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`
  GUI：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
  `tools/verify.py` 在本地会因仓库内的 `.venv` 触发 "generated directory inside repository" 而失败，不作为本地门禁。
- 不 stage `.claude/`。

---

## File Structure

| 文件 | 动作 |
| --- | --- |
| `tests/architecture/test_dependency_rules.py` | 注册 `orsopy` root + `io.orso` 精确 owner + 三向 fixture |
| `src/xrr_fitter/model/analysis.py` | `UncertaintyReport` 末尾加 `parameter_sigma: np.ndarray \| None = None` + `__post_init__` 校验 |
| `src/xrr_fitter/analysis/report.py` | `_correlation_evidence` 扩成四元组返回 sigma；调用点解包并填字段 |
| `src/xrr_fitter/analysis/derivatives.py` | 新增 `covariance_from_correlation` |
| `src/xrr_fitter/io/codec_common.py` | `NULLABLE_ARRAY_FIELDS` 加 `parameter_sigma`（字母序插在 `objectives` 后） |
| `src/xrr_fitter/io/codec_results.py` | `_uncertainty_to_dict` 条件 emit；`_uncertainty_from_dict` optional 集合 + 解码 |
| `src/xrr_fitter/io/orso.py` | **新建**：header 构造、单点 schema 校验、行过滤、扩展命名空间 |
| `src/xrr_fitter/services/exports.py` | `export_result(..., *, include_ort=False)`；`_dataset_artifacts` 条件追加；调重建函数 |
| `src/xrr_fitter/cli/main.py` | `export` 子命令加 `--ort` |
| `src/xrr_fitter/cli/commands.py` | `run_export` 透传 |
| `src/xrr_fitter/gui/export/dialog.py` | 目录选择前的 `.ort` 复选框；摘要追加扩展字段说明与协方差缺席原因 |
| `tests/unit/analysis/test_derivatives.py` | 重建函数测例（含修正 2 的容差与注释） |
| `tests/unit/analysis/test_report.py` | sigma 填充与自洽 |
| `tests/unit/io/test_project_codec.py` | 往返 + 旧文件缺字段 |
| `tests/unit/io/test_orso_export.py` | **新建**：三层验证 + NaN 行排除 + sigma 缺失 |
| `tests/unit/services/test_exports.py` | `include_ort` 开/关的清单断言 |
| `tests/gui/test_export_dialog.py` | 复选框与摘要披露 |
| **不改** | `io/export_run.py`（修正 7）、`pyproject.toml` 与两个 lock 与 `release-spec.json`（修正 4）、`physics/`、`fit/checkpoint.py`（修正 5） |

---

## Tasks

### Task 0：只读核验前置事实（不改任何文件）

- [ ] 确认两个依赖仍已声明且已入 lock：
      `rg -n 'orsopy|jsonschema' pyproject.toml verification/release-spec.json packaging/windows/xrr-fitter.spec`
      `rg -n 'orsopy|jsonschema|attrs|referencing|rpds' requirements*.txt` （两个平台 lock 都要看）
- [ ] 确认零消费者：`rg -n 'import orsopy|from orsopy|import jsonschema' src tests tools` 应无输出；`ls src/xrr_fitter/io/orso.py` 应不存在。
- [ ] 在 REPL 复核 spec 断言的三条 orsopy 接线事实（lock 里是 1.2.3，可核）：
      `.venv/bin/python -c "from orsopy.fileio import base; import inspect, os; print(base._validate_header_data); print('jsonschema' in inspect.getsource(base._validate_header_data)); print(os.path.exists(os.path.join(os.path.dirname(base.__file__), 'schema', 'refl_header.schema.json')))"`
- [ ] 顺带把 orsopy 的**公开**构造 API 摸清（`orsopy.fileio` 的 `Orso` / `OrsoDataset` / `save_orso` 或等价物的确切签名与必填字段），记进 Task 6 开工笔记。**不要凭记忆写 orsopy 调用**，本 plan 不预设这些签名。
- [ ] 记录本轮基线：`.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`

### Task 1：架构门禁注册 orsopy（必须最先做，否则 Task 6 一落地就红）

- [ ] RED：在 `tests/architecture/test_dependency_rules.py` 补 fixture 覆盖三个方向。**实际只需加两条 parametrize 条目**，不是三条新测例：
      「接受精确 owner」→ 往 `test_fixture_checker_accepts_exact_third_party_owners`（`:692`）的参数表加 `("io.orso", "import orsopy")`。
      「拒绝错误 owner」→ 往 `test_fixture_checker_rejects_unknown_or_wrong_third_party_owner`（`:708`）的参数表加 `("fit.search", "import orsopy")`。
      「拒绝未知 root」→ **已被覆盖**，`:704` 的 `("fit.search", "import refnx")` 在加了 `orsopy` 之后仍然有效，不用动。
- [ ] GREEN：`THIRD_PARTY_ROOTS` 加 `"orsopy"`；`THIRD_PARTY_MODULE_ALLOWLIST` 加 `"orsopy": {"io.orso"}`。
- [ ] **不要**加进 `THIRD_PARTY_OWNER_ALLOWLIST`（owner-wide 授权违反 `:90-113` 的「never owner-wide grants」契约）；**不要**给 `jsonschema` 加任何条目（修正 1）。
- [ ] 验证：`.venv/bin/python -m pytest tests/architecture --import-mode=importlib -q`

### Task 2：`UncertaintyReport.parameter_sigma`

- [ ] RED：`tests/unit/analysis/` 里加测例——构造带 sigma 的报告断言只读与形状校验；构造长度不匹配的 sigma 断言抛错；不传 sigma 时字段为 `None`。
- [ ] GREEN：`model/analysis.py` 的 `UncertaintyReport` **末尾**（`prior_conflicts` 之后）加 `parameter_sigma: np.ndarray | None = None`；`__post_init__` 里非 None 时 `_readonly(..., float, "parameter_sigma", 1)` + 长度等于 `len(names)` 校验 + `object.__setattr__` 写回。
- [ ] 核验 checkpoint 指纹未漂移（修正 5 说不该漂，但这是唯一能证明的测例）：
      `.venv/bin/python -m pytest tests/regression -k frozen_stage_search --import-mode=importlib -q`
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/analysis --import-mode=importlib -q`

### Task 3：`analysis/report.py` 填充 sigma

- [ ] RED：断言 `build_uncertainty_report` 产出的 `parameter_sigma` 非 None、长度等于 `correlation_names`、且与 `correlation_matrix` 的对角自洽（对角为 1 处 sigma > 0，对角为 0 处 sigma == 0）。
- [ ] GREEN：`_correlation_evidence`（`:176`）扩成四元组，返回 `sigma = np.sqrt(np.clip(np.diag(physical_covariance), 0.0, np.inf))`——与 `correlation_from_covariance` 内部第 4-5 行逐字一致，保证对角处理自洽。`:265` 的调用点同步解包，`UncertaintyReport(...)` 里传 `parameter_sigma=sigma`。
- [ ] **回归风险点**：这一步让所有跑过 `build_uncertainty_report` 的路径开始产出新字段。跑全套确认没有 golden 字节断言被打破：
      `.venv/bin/python -m pytest tests/unit tests/regression --import-mode=importlib -q`

### Task 4：`analysis/derivatives.py` 协方差重建

- [ ] RED：`tests/unit/analysis/test_derivatives.py` 加测例——正定协方差经 `correlation_from_covariance` 再经 `covariance_from_correlation` 往返，用 `np.allclose(rtol=0.0, atol=1e-12)` 断言，**注释写清为什么不能用 `==`**（引修正 2 的四条有损性）；含非正对角的协方差断言对应行列重建为 0。
- [ ] GREEN：`covariance_from_correlation(sigma, correlation)` 返回 `sigma[:, None] * correlation * sigma[None, :]`，校验形状匹配、返回只读数组（与模块内既有 `setflags(write=False)` 风格一致）。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/analysis/test_derivatives.py --import-mode=importlib -q`

### Task 5：codec 往返

- [ ] RED：`tests/unit/io/test_project_codec.py` 加三个测例——带 sigma 的结果存读往返后数组逐位相等；`parameter_sigma is None` 时**编码结果不含该键**（断言 `"parameter_sigma" not in payload`，这是向后兼容的证明）；手工构造缺该键的旧 payload 能读回且字段为 None。
- [ ] GREEN：
      `io/codec_common.py` 的 `NULLABLE_ARRAY_FIELDS` 加 `"parameter_sigma"`（字母序：插在 `"objectives"` 之后、`"qz_a_inv"` 之前）。
      `codec_results.py::_uncertainty_to_dict` 末尾按 `prior_conflicts` 的模式加：非 None 才 `payload["parameter_sigma"] = _real_array_to_list(...)`，并写一句同型注释说明为什么条件 emit。
      `_uncertainty_from_dict` 的 optional 集合（当前 `{"candidate_id", "sld_bands", "prior_conflicts"}`）加 `"parameter_sigma"`；构造时 `parameter_sigma=None if payload.get("parameter_sigma") is None else _real_array_from_list(payload["parameter_sigma"])`。
- [ ] 注意 `_mapping` 会拒绝未声明的 extra 键，optional 集合漏加会导致新写的文件读不回来。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/io --import-mode=importlib -q`

### Task 6：新建 `io/orso.py`

- [ ] RED：新建 `tests/unit/io/test_orso_export.py`，按 spec 的三层验证写：(1) orsopy 读回自洽；(2) `_validate_header_data` 校验通过；(3) 往返后数据列与参数值**逐位相等**（这一层的 `==` 是成立的，数据列不经归一化）。
- [ ] GREEN：`orso_bytes(context: DatasetExportData, *, covariance: np.ndarray | None) -> bytes`，四段按 spec：
      - 数据段：从 `context.data`（`PreparedData`）取 q/强度/误差，从 `context.selected` 取模型与残差；**按修正 8 过滤非有限行**。
      - `orso.data_source`：测量元数据 + 源文件 sha256 provenance（`context.data.source_path` / `source_sha256`；`io/source.py` 的 `validate_source` 是校验器，导出只需要已记录的哈希）。
      - `orso.reduction`：软件版本、`context.replay_identity.service_seed_tree_version`（**不 import `services`**，修正 10）、完整 `FitConfig`。
      - `xrr_fitter.confidence` 扩展段：参数表、误差棒（`bootstrap_intervals`）、协方差（`covariance` 为 None 时**整段省略**并写明原因）、`ConfidenceClass` 的 **enum 名 + 中文显示值**两份、reason codes（`classification_evidence`，`model/analysis.py:538`）、被排除的行数与原因。
- [ ] **取用路径（spec 与本 plan 初稿都没写清，这里定死）：** `DatasetExportData` 没有 `FitResult` 字段，只有 `selected: FitCandidate`。`FitResult` 经 property 拿：`io/export_tables.py:175-181` 的 `DatasetExportData.result`，它返回 `self.dataset.last_valid_result` 并在为 None 时抛 `ValueError("dataset has no fit result")`。所以：
      `UncertaintyReport` → `context.result.uncertainty`（可能为 None）
      `ConfidenceClass` 与 reason codes → `context.result.confidence` / `context.result.classification_evidence`
      模型与残差 → `context.selected`
      q / 强度 / 误差 → `context.data`（`PreparedData`）
      既有 `export_tables.py:545`/`:566`/`:809` 就是这么取的，照抄。**不要**自己从 `context.dataset.last_valid_result` 手取——那会绕开 property 的 None 检查。
- [ ] schema 校验就地做，唯一调用点带注释：依赖 `orsopy.fileio.base._validate_header_data` 私有名，orsopy 升级时首个检查点。校验失败直接抛，不降级（`publish_export_run` 尚未建 partial 目录，天然不留半成品）。
- [ ] NaN 行排除测例：构造含非正角度（`two_theta_deg <= 0`）的数据集，断言 `.ort` 行数等于有效行数且扩展段计数正确。
- [ ] `covariance is None` 时导出仍成功且协方差段缺席。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/io/test_orso_export.py tests/architecture --import-mode=importlib -q`

### Task 7：`services/exports.py` 接线

- [ ] RED：`tests/unit/services/test_exports.py` 加两个测例——`include_ort=False`（默认）时产物清单与 HEAD 逐位相同（`==`，列出完整期望元组）；`include_ort=True` 时每个数据集目录多且只多一个 `fit_result.ort`。
- [ ] GREEN：`export_result(result, output_dir, *, include_ort: bool = False)`；`_dataset_artifacts(context, *, include_ort)` 在 `include_ort` 且能构造时把 `ArtifactPayload("fit_result.ort", orso_bytes(...))` 追加进元组；协方差由 `covariance_from_correlation` 在此处重建后传入（修正 9）。
- [ ] 「不产出」必须是**从元组省略**，不是空 payload（`ArtifactPayload` 拒绝空内容，修正 7）。
- [ ] `api.py:84` 是直接 re-export `services.exports.export_result`，签名自动跟随，**`api.py` 不用改**。
- [ ] `io/export_run.py` **不改**（修正 7）。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit/services tests/unit/io --import-mode=importlib -q`

### Task 8：CLI `export --ort`

- [ ] RED：断言不带 `--ort` 时导出目录无 `.ort`；带 `--ort` 时有。退出码语义不变（`cli/exit_codes.py:16-19` 的 TRUSTED/CORRELATED→SUCCESS、MULTIPLE/UNTRUSTED→NOT_CONVERGED 不动）。
- [ ] GREEN：`cli/main.py` 的 `export` 子命令加 `export.add_argument("--ort", action="store_true", help="额外产出 ORSO .ort 文件")`；`cli/commands.py:95` 的 `run_export` 透传 `include_ort=arguments.ort`。
- [ ] **不加 `--format`**（修正 3）。
- [ ] 验证：`.venv/bin/python -m pytest tests/unit tests/acceptance --import-mode=importlib -q`

### Task 9：GUI 复选框与摘要披露

- [ ] RED：`tests/gui/test_export_dialog.py` 加测例——勾选后摘要含 `.ort` 路径**且含扩展字段说明**；不勾选时摘要无该行且磁盘无 `.ort`；`parameter_sigma` 缺失时摘要写出协方差段缺席的原因。
- [ ] GREEN：`gui/export/dialog.py` 在 `QFileDialog.getExistingDirectory`（`:94`）**之前**插一个带复选框的小对话框，默认勾选（修正 3 的 GUI 默认）。`ExportWorkflow.export_results(directory, *, include_ort)` 透传。
- [ ] `export_summary`（`:25`）遍历 `manifest.files`，`.ort` 自动进清单，**这段主体不用改**；要加的是产出 `.ort` 时在摘要末尾追加扩展命名空间说明，以及协方差缺席时的原因说明（spec 界面节两条要求）。
- [ ] 复选框要有 `objectName` / `setAccessibleName` / `setToolTip`，与 `ExportSummaryDialog` 里既有控件的无障碍风格一致。
- [ ] 验证：`.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`

### Task 10：全量回归

- [ ] `.venv/bin/python -m pytest tests/unit tests/architecture --import-mode=importlib -q`
- [ ] `.venv/bin/python -m pytest tests/gui --import-mode=importlib -q`
- [ ] `.venv/bin/python -m pytest tests/integration tests/regression tests/acceptance --import-mode=importlib -q`
- [ ] 与 Task 0 的基线比对：新增测例之外无失败变化。仓库既有的两个失败（见 `mem:xrr-test-suite-commands`）不算本轮回归，但要在汇报里点名。
- [ ] 确认 `include_ort` 默认关闭下导出产物与 HEAD 逐位相同。
- [ ] 清理临时产物（REPL 脚本、临时导出目录）。

---

## 剩余风险

- **orsopy 的公开构造 API 未在本 plan 中固定。** Task 0 要求先摸清签名再写 Task 6，本 plan 刻意不预设 `Orso` / `OrsoDataset` / `save_orso` 的字段名——凭记忆写这些会得到跑不起来的代码。若实测发现某个 ORSO 必填字段本项目没有对应数据，停下来记成新的修正，不要塞占位值。
- **私有名 `_validate_header_data` 是升级脆点。** 单点调用 + 注释是缓解，不是消除。orsopy 升到 1.3+ 时这里必炸，而且是 ImportError 那种响亮的炸法（可接受，好过静默跳过校验）。
- **修正 2 让 spec 的一条验收标准从「逐位」降为「容差 + 子块」。** 这是数学事实决定的，不是实现偷懒。若后续真要逐位可复现的协方差，只能改回 spec 的方案 B（直接持久化协方差矩阵，correlation 派生），那会动 GUI 与 `export_tables.py` 的既有消费方——属于另一个 spec。
- **Task 3 的回归面比看起来大。** 一旦 `report.py` 开始填 sigma，所有存过 `UncertaintyReport` 的工程文件都会多一个键。emit 条件保护的是**旧数据**的字节稳定，不保护**新算结果**的字节稳定。若 `tests/regression` 里有对新算结果的 golden 字节断言，Task 3 会红，届时要判断是更新 golden 还是重新审视字段位置。
- **`.ort` 的扩展命名空间是单向承诺。** 一旦发布出去，`xrr_fitter.confidence` 下的键名就成了对外契约。命名在 Task 6 一次定死，不要留「以后再改」的余地。
