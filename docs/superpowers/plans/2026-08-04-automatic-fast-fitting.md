# XRR 快速全自动拟合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把默认使用路径收敛为“选择文件 -> 自动建模 -> 单点或同批多点自动拟合 -> 查看逐点与均匀性结果”，常规单点目标 20 秒、2-4 点目标 45 秒，同时保留现有专家拟合入口。

**Architecture:** 新增持久化自动状态和测量预设，由 `services.datasets` 严格解析一次导入批次，由 `services.batch` 按物理签名路由单点或多点流程。`services.fitting` 继续作为唯一同时组合 `fit` 与 `analysis` 的模块：它执行快速质量门槛、有限搜索升级、吸收部试探和最多四个定向 profile；多点流程先并行预拟合，再用候选共识初始化联合精修并处理隔离点。GUI 只调用 `xrr_fitter.api`，现有 `fit_project()` / `start_fit_job()` 保持专家合同不变。

**Tech Stack:** Python 3.12、NumPy 2.x、SciPy 1.14+、periodictable 2.x、PySide6 6.8+、pytest 8.3+；不新增生产依赖。

## Global Constraints

- Python 必须保持 `>=3.12,<3.13`，代码继续使用 `src/` 布局。
- `xrr_fitter.api` 是唯一受支持 Python API；GUI 不得直接导入 model、services、fit 或 analysis。
- `services.fitting` 是唯一可同时导入 `fit` 与 `analysis` 的模块；`services.batch` 只通过注入的 fitting 函数组合二者。
- 保留 `fit_project()`、`start_fit_job()`、手动结构编辑、专家参数、MCMC 和导出能力；自动路径不得修改它们的默认物理结果。
- 文件名最后一个空格段按“基底侧膜层+...+表面侧膜层”解析，单层 token 也必须解析，内部只反转一次为表面到基底顺序。
- 默认基底为 Si；最左有限膜层为 Si 时按同结构组选择一次实际基底。
- Si 基底自动加入 SiO2；基底相邻已有精确 `SiO2` 时不得重复。自动氧化层固定为 2.20 g/cm3、初值 10 A、边界 2-50 A、密度锁定。
- 未知多元代号使用直接有效 SLD：实部自动拟合，吸收部默认锁定，`density_scale=1` 锁定；不得伪造配比或质量密度。
- 同一导入批次同物理签名的多点才联合；厚度和仪器参数局部，同名材料 SLD/相对密度共享，粗糙度先共享、证据冲突时释放。
- 每次自动结果始终运行快速质量门槛；完整 profile 不得成为默认完成条件，自动路径最多运行 4 个定向 profile。
- 自动路径不导出；点位坐标和空间分布图不进入本计划。
- 参考墙钟目标为常规单点 20 秒、2-4 点 45 秒、疑难升级 60 秒；求解结果由确定性工作预算决定，不按墙钟强杀进程。
- 现有完整合成恢复语料是 **220 例**，不得缩减；厚度/周期、相对密度和粗糙度门槛保持设计中给出的 2%/5%、3%/8%、1 A/3 A。
- 所有运行行为改动先取得本任务的 RED，再实现 GREEN；每个任务独立提交，且不得提交根目录的三个 probe 文件。

---

## 文件与职责总览

| 文件 | 职责 |
| --- | --- |
| `src/xrr_fitter/model/automation.py` | 自动状态、测量预设、导入预览、逐层结果和均匀性值对象 |
| `src/xrr_fitter/model/project.py` | 在项目/数据集根上持久化预设和自动状态，执行交叉引用校验 |
| `src/xrr_fitter/model/operations.py` | 持有引用 `XrrProject` / `FitResult` 的导入与自动拟合操作结果 |
| `src/xrr_fitter/io/project_codec.py` | schema v2 编解码和 v1 原位迁移入口 |
| `src/xrr_fitter/io/codec_results.py` | `bootstrap_performed` 的结果编解码 |
| `src/xrr_fitter/services/materials.py` | 已知 formula-density 与未知 direct-SLD 的唯一 token 分类 |
| `src/xrr_fitter/services/datasets.py` | 严格文件名预览、逐文件容错导入、导入批次身份 |
| `src/xrr_fitter/fit/initialization.py` / `fit/candidates.py` | 数据相关、确定性的直接 SLD 初始候选 |
| `src/xrr_fitter/analysis/automatic.py` | 快速质量证据、升级动作和最多四个 profile 的选择 |
| `src/xrr_fitter/fit/automatic.py` | 不依赖 analysis 的有界局部重拟合原语 |
| `src/xrr_fitter/services/fitting.py` | 单点自动拟合以及 fit/analysis 的唯一自动组合边界 |
| `src/xrr_fitter/services/parallel.py` | 保持返回输入顺序，同时按实际完成顺序发布回调 |
| `src/xrr_fitter/services/batch.py` | 物理签名分组、全局 CPU 预算、预拟合、联合精修和隔离重试 |
| `src/xrr_fitter/fit/joint_sharing.py` / `fit/joint_pipeline.py` | 同数据集多成员共享和预拟合共识初值 |
| `src/xrr_fitter/services/results.py` | 逐层物理结果与总体标准差/CV/相对极差汇总 |
| `src/xrr_fitter/services/workers.py` | 复用现有事件协议启动自动拟合进程 |
| `src/xrr_fitter/gui/data/*` | 首次测量预设、基底选择、逐文件错误恢复和导入即拟合 |
| `src/xrr_fitter/gui/fitting/*` | 默认自动入口与专家入口分层、即时 checkpoint 曲线 |
| `src/xrr_fitter/gui/results/*` | 逐点逐层表和批次均匀性表 |

### Task 1: 自动状态、测量预设与 schema v2

**Files:**
- Create: `src/xrr_fitter/model/automation.py`
- Create: `tests/unit/model/test_automation_values.py`
- Modify: `src/xrr_fitter/model/project.py:43-52,160-237,443-499`
- Modify: `src/xrr_fitter/model/analysis.py:447-484`
- Modify: `src/xrr_fitter/model/operations.py:28-60`
- Modify: `src/xrr_fitter/io/project_codec.py:45-52,187-398`
- Modify: `src/xrr_fitter/io/codec_results.py:126-190`
- Modify: `tests/unit/io/test_project_codec.py`
- Modify: `tests/integration/test_project_roundtrip.py`
- Modify: `tests/architecture/test_dependency_rules.py:110-121`

**Interfaces:**
- Produces: `AutomaticRole`, `AutomaticStatus`, `DatasetAutomation`, `MeasurementPreset`, `ImportFilePreview`, `ImportBatchPreview`, `ImportFailure`, `AutomaticLayerResult`, `AutomaticDatasetSummary`, `LayerUniformitySummary`, `AutomaticResultSummary`.
- Produces: `ProjectImportResult(updated_project, import_batch_id, imported_dataset_ids, failures)` in `model.operations`.
- Changes: `DatasetProject.automation: DatasetAutomation` with a manual/not-run default.
- Changes: `XrrProject.measurement_preset: MeasurementPreset | None` with a `None` default.
- Changes: append `UncertaintyReport.bootstrap_performed: bool = True` after the existing `candidate_id` field so old positional construction is not reinterpreted; v1 documents migrate this field to `True`.
- Changes: persisted `SCHEMA_VERSION` from 1 to 2; only schema 1 is migrated, all other unsupported versions still fail closed.

- [ ] **Step 1: Write failing value and migration tests**

Add the following focused cases. The codec test deliberately derives a v1 fixture from current output so it proves every newly required v2 field is restored without maintaining a second codec.

```python
# tests/unit/model/test_automation_values.py
from dataclasses import replace

import pytest

from tests.support.model_cases import dataset_project, project
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec


def test_measurement_preset_owns_beam_instrument_and_angle_offset() -> None:
    preset = MeasurementPreset(
        "lab-cu-kalpha",
        BeamSpec(kind="monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab", footprint_mode="fit"),
        0.012,
    )
    assert preset.preset_id == "lab-cu-kalpha"
    assert preset.import_angle_offset_deg == 0.012


def test_automatic_state_requires_group_identity_and_review_reason() -> None:
    with pytest.raises(ValueError, match="fit_group_id"):
        DatasetAutomation(role=AutomaticRole.JOINT, status=AutomaticStatus.PENDING)
    with pytest.raises(ValueError, match="reason"):
        DatasetAutomation(
            import_batch_id="batch-1",
            fit_group_id="group-1",
            role=AutomaticRole.SINGLE,
            status=AutomaticStatus.REVIEW,
        )


def test_only_passed_automatic_results_can_be_statistics_members() -> None:
    with pytest.raises(ValueError, match="statistics_member"):
        DatasetAutomation(
            import_batch_id="batch-1",
            fit_group_id="group-1",
            role=AutomaticRole.JOINT,
            status=AutomaticStatus.REFINING,
            statistics_member=True,
        )


def test_project_validates_dataset_automation_values() -> None:
    value = project(dataset_project("sample"))
    state = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=AutomaticRole.SINGLE,
        status=AutomaticStatus.PENDING,
    )
    updated = replace(value, datasets=(replace(value.datasets[0], automation=state),))
    assert updated.datasets[0].automation is state
```

```python
# append to tests/unit/io/test_project_codec.py
def _project_with_result():
    result, checkpoint = _manual_result_graph()
    dataset = replace(
        dataset_project("sample-1"),
        last_valid_result=result,
        checkpoint=checkpoint,
    )
    return project(dataset)


def test_schema_one_migrates_automation_preset_and_bootstrap_flag() -> None:
    value = _project_with_result()
    payload = project_to_dict(value)
    payload["schema_version"] = 1
    payload.pop("measurement_preset")
    for dataset in payload["datasets"]:
        dataset.pop("automation")
        result = dataset["last_valid_result"]
        if result is not None and result["uncertainty"] is not None:
            result["uncertainty"].pop("bootstrap_performed")

    migrated = project_from_dict(payload)

    assert migrated.schema_version == 2
    assert migrated.measurement_preset is None
    assert migrated.datasets[0].automation.status.value == "not_run"
    assert migrated.datasets[0].last_valid_result.uncertainty.bootstrap_performed is True


@pytest.mark.parametrize("version", (0, 3, 999))
def test_only_schema_one_has_a_migration_path(version: int) -> None:
    payload = project_to_dict(_project_with_result())
    payload["schema_version"] = version
    with pytest.raises(ProjectVersionError, match="unsupported project schema"):
        project_from_dict(payload)
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/model/test_automation_values.py tests/unit/io/test_project_codec.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'xrr_fitter.model.automation'` or construction fails because the new fields are absent.

- [ ] **Step 3: Add immutable model contracts**

Create enums and values with the following exact persisted strings and validation rules. Result values use plain tuples/floats so the model does not add a NumPy dependency.

```python
# src/xrr_fitter/model/automation.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec


class AutomaticRole(StrEnum):
    MANUAL = "manual"
    UNROUTED = "unrouted"
    SINGLE = "single"
    JOINT = "joint"
    ISOLATED_RETRY = "isolated_retry"


class AutomaticStatus(StrEnum):
    NOT_RUN = "not_run"
    PENDING = "pending"
    REFINING = "refining"
    PASSED = "passed"
    REVIEW = "review"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MeasurementPreset:
    preset_id: str
    beam: BeamSpec
    instrument: InstrumentSpec
    import_angle_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.preset_id.strip():
            raise ValueError("preset_id must not be empty")
        if not isinstance(self.beam, BeamSpec):
            raise TypeError("beam must be BeamSpec")
        if not isinstance(self.instrument, InstrumentSpec):
            raise TypeError("instrument must be InstrumentSpec")
        if not isfinite(self.import_angle_offset_deg):
            raise ValueError("import_angle_offset_deg must be finite")


@dataclass(frozen=True, slots=True)
class DatasetAutomation:
    import_batch_id: str | None = None
    fit_group_id: str | None = None
    role: AutomaticRole = AutomaticRole.MANUAL
    status: AutomaticStatus = AutomaticStatus.NOT_RUN
    statistics_member: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, AutomaticRole):
            raise TypeError("role must be AutomaticRole")
        if not isinstance(self.status, AutomaticStatus):
            raise TypeError("status must be AutomaticStatus")
        if not isinstance(self.statistics_member, bool):
            raise TypeError("statistics_member must be bool")
        automatic = self.role is not AutomaticRole.MANUAL
        if automatic and not self.import_batch_id:
            raise ValueError("automatic role requires import_batch_id")
        if self.role not in {AutomaticRole.MANUAL, AutomaticRole.UNROUTED} and not self.fit_group_id:
            raise ValueError("routed automatic role requires fit_group_id")
        if self.status in {AutomaticStatus.REVIEW, AutomaticStatus.FAILED} and not self.reason:
            raise ValueError("review or failed status requires reason")
        if self.statistics_member and self.status is not AutomaticStatus.PASSED:
            raise ValueError("statistics_member requires passed status")


@dataclass(frozen=True, slots=True)
class ImportFilePreview:
    source_path: str
    display_name: str
    dataset_id_stem: str | None
    layers_backing_to_surface: tuple[str, ...]
    substrate_group_id: str | None
    requires_substrate_choice: bool
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class ImportBatchPreview:
    import_batch_id: str
    preset: MeasurementPreset
    files: tuple[ImportFilePreview, ...]

    def __post_init__(self) -> None:
        if not self.import_batch_id.strip():
            raise ValueError("import_batch_id must not be empty")
        if not isinstance(self.preset, MeasurementPreset):
            raise TypeError("preset must be MeasurementPreset")
        files = tuple(self.files)
        if any(not isinstance(value, ImportFilePreview) for value in files):
            raise TypeError("files must contain ImportFilePreview values")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ImportFailure:
    source_path: str
    message: str
    recovery_action: str


@dataclass(frozen=True, slots=True)
class AutomaticLayerResult:
    dataset_id: str
    layer_index: int
    material_name: str
    thickness_a: float
    roughness_a: float
    sld_real_a2: float
    sld_imag_a2: float
    electron_density_a3: float
    nominal_density_g_cm3: float | None
    density_scale: float
    fitted_density_g_cm3: float | None
    density_note: str | None


@dataclass(frozen=True, slots=True)
class AutomaticDatasetSummary:
    dataset_id: str
    status: AutomaticStatus
    statistics_member: bool
    reason: str | None
    layers: tuple[AutomaticLayerResult, ...]


@dataclass(frozen=True, slots=True)
class LayerUniformitySummary:
    fit_group_id: str
    layer_index: int
    material_name: str
    count: int
    mean_thickness_a: float
    minimum_thickness_a: float
    maximum_thickness_a: float
    population_std_a: float
    cv_percent: float
    relative_range_percent: float


@dataclass(frozen=True, slots=True)
class AutomaticResultSummary:
    import_batch_id: str | None
    datasets: tuple[AutomaticDatasetSummary, ...]
    uniformity: tuple[LayerUniformitySummary, ...]
```

Add `ProjectImportResult` to `model.operations`, because that module already owns the allowed `operations -> project` edge:

```python
@dataclass(frozen=True, slots=True)
class ProjectImportResult:
    updated_project: XrrProject
    import_batch_id: str
    imported_dataset_ids: tuple[str, ...]
    failures: tuple[ImportFailure, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.updated_project, XrrProject):
            raise TypeError("updated_project must be XrrProject")
        if not self.import_batch_id.strip():
            raise ValueError("import_batch_id must not be empty")
        identifiers = tuple(self.imported_dataset_ids)
        if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
            raise ValueError("imported_dataset_ids must be unique and nonempty")
        failures = tuple(self.failures)
        if any(not isinstance(value, ImportFailure) for value in failures):
            raise TypeError("failures must contain ImportFailure values")
        object.__setattr__(self, "imported_dataset_ids", identifiers)
        object.__setattr__(self, "failures", failures)
```

Update `MODEL_ALLOWED` with these exact edges:

```python
MODEL_ALLOWED.update(
    {
        "automation": {"data", "instrument"},
        "project": {
            "automation",
            "data",
            "instrument",
            "structure",
            "parameters",
            "fitting",
            "analysis",
        },
        "operations": {"automation", "fitting", "analysis", "project"},
    }
)
```

- [ ] **Step 4: Persist v2 and migrate v1 in one codec**

Set `SCHEMA_VERSION = 2`, encode `measurement_preset`, `automation`, and `bootstrap_performed`, and run this migration before exact v2 field validation:

```python
# src/xrr_fitter/io/project_codec.py
from copy import deepcopy


def _migrate_v1_document(value: object) -> object:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return value
    payload = deepcopy(value)
    payload["schema_version"] = 2
    payload["measurement_preset"] = None
    for dataset in payload.get("datasets", ()):
        if not isinstance(dataset, dict):
            continue
        dataset["automation"] = {
            "import_batch_id": None,
            "fit_group_id": None,
            "role": "manual",
            "status": "not_run",
            "statistics_member": False,
            "reason": None,
        }
        result = dataset.get("last_valid_result")
        if not isinstance(result, dict):
            continue
        uncertainty = result.get("uncertainty")
        if isinstance(uncertainty, dict):
            uncertainty["bootstrap_performed"] = True
    return payload


def project_from_dict(value: object) -> XrrProject:
    try:
        payload = _validated_document(_migrate_v1_document(value))
        return _project_from_validated_payload(payload)
    except ProjectSchemaError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectSchemaError(str(error)) from error
```

Do not make `_validate_version()` accept a range. It must still require the single current version after migration.

- [ ] **Step 5: Run GREEN and architecture checks**

Run:

```bash
python -m pytest -o addopts= tests/unit/model/test_automation_values.py tests/unit/model/test_analysis_values.py tests/unit/io/test_project_codec.py tests/integration/test_project_roundtrip.py tests/architecture/test_dependency_rules.py -q
```

Expected: all selected tests pass; v2 round trips byte-stably and the derived v1 fixture loads with `bootstrap_performed=True`.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/xrr_fitter/model/automation.py src/xrr_fitter/model/project.py src/xrr_fitter/model/analysis.py src/xrr_fitter/model/operations.py src/xrr_fitter/io/project_codec.py src/xrr_fitter/io/codec_results.py tests/unit/model/test_automation_values.py tests/unit/model/test_analysis_values.py tests/unit/io/test_project_codec.py tests/integration/test_project_roundtrip.py tests/architecture/test_dependency_rules.py && git commit -m "feat: persist automatic fitting state"
```

### Task 2: 已知材料、未知 direct-SLD 与自动 SiO2

**Files:**
- Modify: `src/xrr_fitter/services/materials.py`
- Modify: `tests/unit/services/test_datasets.py`
- Create: `tests/unit/services/test_automatic_materials.py`

**Interfaces:**
- Produces: `material_from_token(token: str) -> MaterialSpec`.
- Produces: `automatic_structure(formulas_surface_to_backing: tuple[str, ...], backing_token: str) -> tuple[StructureSpec, tuple[ParameterSetting, ...]]`.
- Changes: filename-driven `initial_structure()` classifies every layer through `material_from_token()` but retains its existing no-auto-oxide behavior.
- Keeps: `material_from_initial_density()` as the strict known-material helper for callers that explicitly require a table-backed formula and density.
- Known table retains only audited formula+density entries; `CrSiC` and `SiCMo` are no longer formula-density declarations.

- [ ] **Step 1: Write RED material/oxide tests**

```python
# tests/unit/services/test_automatic_materials.py
from xrr_fitter.services.materials import automatic_structure, material_from_token


def test_unknown_compound_tokens_are_direct_sld_not_fake_formulas() -> None:
    for token in ("CrSiC", "SiCMo", "AlScN", "custom-4element"):
        material = material_from_token(token)
        assert material.name == token
        assert material.formula is None
        assert material.bulk_density_g_cm3 is None
        assert material.sld_override_a2 == 20e-6 + 0j


def test_known_material_retains_formula_and_nominal_density() -> None:
    material = material_from_token("Si3N4")
    assert (material.formula, material.bulk_density_g_cm3) == ("Si3N4", 3.17)


def test_si_backing_inserts_one_locked_native_oxide() -> None:
    structure, settings = automatic_structure(("Zr", "Si3N4"), "Si")
    assert tuple(layer.name for layer in structure.components) == ("Zr", "Si3N4", "SiO2 native oxide")
    assert structure.components[-1].material.bulk_density_g_cm3 == 2.20
    by_name = {setting.name: setting for setting in settings}
    assert by_name["component.2.thickness_a"].initial == 10.0
    assert (by_name["component.2.thickness_a"].lower, by_name["component.2.thickness_a"].upper) == (2.0, 50.0)
    assert by_name["component.2.density_scale"].locked is True


def test_existing_backing_adjacent_exact_sio2_is_not_duplicated() -> None:
    structure, _settings = automatic_structure(("Zr", "SiO2"), "Si")
    assert tuple(layer.name for layer in structure.components) == ("Zr", "SiO2")


def test_unknown_direct_sld_density_scale_is_locked_to_one() -> None:
    _structure, settings = automatic_structure(("CrSiC",), "sapphire")
    density = next(value for value in settings if value.name == "component.0.density_scale")
    assert (density.initial, density.lower, density.upper, density.locked) == (1.0, 1.0, 1.0, True)
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_materials.py tests/unit/services/test_structures.py -q
```

Expected: import fails because `automatic_structure` and `material_from_token` do not exist; the old table would also make the unknown-token assertions fail.

- [ ] **Step 3: Implement the single material classification boundary**

Use these exact constants and policies in `services.materials`:

```python
INITIAL_DENSITY_TABLE_VERSION = "initial-density-v2"
INITIAL_DENSITIES_G_CM3: Mapping[str, float] = MappingProxyType(
    {
        "Si": 2.329,
        "SiO2": 2.20,
        "Si3N4": 3.17,
        "TaN": 14.30,
        "Zr": 6.52,
    }
)
DEFAULT_DIRECT_SLD_A2 = 20e-6 + 0.0j


def material_from_token(token: str) -> MaterialSpec:
    name = token.strip()
    if not name:
        raise ValueError("material token must not be empty")
    density = INITIAL_DENSITIES_G_CM3.get(name)
    if density is None:
        return MaterialSpec(name, None, None, DEFAULT_DIRECT_SLD_A2)
    return MaterialSpec(name, name, density)


def initial_structure(formulas: tuple[str, ...]) -> StructureSpec:
    if not formulas:
        raise ValueError("filename material stack must not be empty")
    components = tuple(
        LayerSpec(
            token,
            material_from_token(token),
            DEFAULT_LAYER_THICKNESS_A,
            roughness_a=DEFAULT_LAYER_ROUGHNESS_A,
        )
        for token in formulas
    )
    return StructureSpec(
        MaterialSpec("Air", None, None, 0.0j),
        components,
        material_from_initial_density("Si"),
    )
```

Replace the second stack's old all-formula density assertion in `tests/unit/services/test_datasets.py` with:

```python
materials = project.datasets[1].structure.components
assert tuple(layer.material.formula for layer in materials) == ("TaN", None, None)
assert materials[0].material.bulk_density_g_cm3 == 14.30
assert all(layer.material.bulk_density_g_cm3 is None for layer in materials[1:])
assert all(layer.material.sld_override_a2 is not None for layer in materials[1:])
```

Keep the first all-known stack's current formula/density assertions unchanged.

Build the automatic structure and partial parameter settings without calling project-level oxide services, which avoids a `datasets <-> structures` import cycle:

```python
def automatic_structure(
    formulas_surface_to_backing: tuple[str, ...],
    backing_token: str,
) -> tuple[StructureSpec, tuple[ParameterSetting, ...]]:
    if not formulas_surface_to_backing:
        raise ValueError("filename material stack must not be empty")
    components = [
        LayerSpec(
            token,
            material_from_token(token),
            DEFAULT_LAYER_THICKNESS_A,
            roughness_a=DEFAULT_LAYER_ROUGHNESS_A,
        )
        for token in formulas_surface_to_backing
    ]
    backing = material_from_token(backing_token)
    backing_adjacent = components[-1].material.formula if components else None
    if backing.formula == "Si" and backing_adjacent != "SiO2":
        components.append(
            LayerSpec(
                "SiO2 native oxide",
                MaterialSpec("SiO2", "SiO2", 2.20),
                10.0,
                roughness_a=3.0,
            )
        )
    settings: list[ParameterSetting] = []
    for index, layer in enumerate(components):
        prefix = f"component.{index}"
        if layer.material.sld_override_a2 is not None:
            settings.append(ParameterSetting(f"{prefix}.density_scale", 1.0, 1.0, 1.0, locked=True))
        if layer.name == "SiO2 native oxide":
            settings.extend(
                (
                    ParameterSetting(f"{prefix}.thickness_a", 10.0, 2.0, 50.0),
                    ParameterSetting(f"{prefix}.density_scale", 1.0, 1.0, 1.0, locked=True),
                )
            )
    deduplicated = {setting.name: setting for setting in settings}
    structure = StructureSpec(
        MaterialSpec("Air", None, None, 0.0j),
        tuple(components),
        backing,
    )
    return structure, tuple(deduplicated.values())
```

The exact `SiO2` comparison remains case-sensitive and formula-based. Do not broaden it to fuzzy name matching.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_materials.py tests/unit/services/test_structures.py tests/unit/services/test_datasets.py tests/unit/physics/test_material_sld.py -q
```

Expected: all selected tests pass; existing manual oxide suggestions remain unchanged and unknown aliases have no fake density.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/xrr_fitter/services/materials.py tests/unit/services/test_automatic_materials.py tests/unit/services/test_datasets.py && git commit -m "feat: model unknown materials with direct SLD"
```

### Task 3: 严格批次预览、组级基底选择与逐文件容错导入

**Files:**
- Modify: `src/xrr_fitter/services/datasets.py:79-205`
- Modify: `src/xrr_fitter/api.py`
- Modify: `tests/unit/services/test_datasets.py`
- Create: `tests/unit/services/test_automatic_import.py`
- Modify: `tests/architecture/test_public_api.py`

**Interfaces:**
- Produces: `preview_import_batch(paths, preset, import_batch_id=None) -> ImportBatchPreview`.
- Produces: `import_dataset_batch(project, preview, substrate_choices=None, column_mappings=None) -> ProjectImportResult`.
- Consumes: `automatic_structure()` and Task 1 import values.
- A preview never mutates a project; import commits each valid file independently in preview order.

- [ ] **Step 1: Write strict parsing and partial-success tests**

```python
# tests/unit/services/test_automatic_import.py
from pathlib import Path

import numpy as np

from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.automation import MeasurementPreset
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services.datasets import import_dataset_batch, preview_import_batch
from xrr_fitter.services.projects import new_project


def _curve(path: Path) -> Path:
    path.write_bytes(xy_bytes(np.linspace(0.1, 3.0, 32), np.geomspace(1.0, 1e-5, 32)))
    return path


def _preset() -> MeasurementPreset:
    return MeasurementPreset(
        "cu-kalpha",
        BeamSpec(kind="monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab"),
    )


def test_preview_accepts_single_layer_and_reverses_multilayer_once(tmp_path: Path) -> None:
    paths = (
        _curve(tmp_path / "P1 Zr.xy"),
        _curve(tmp_path / "P2 Si3N4+Si+Zr.xy"),
    )
    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-1")
    assert preview.files[0].layers_backing_to_surface == ("Zr",)
    assert preview.files[1].layers_backing_to_surface == ("Si3N4", "Si", "Zr")
    result = import_dataset_batch(new_project(), preview)
    assert tuple(layer.name for layer in result.updated_project.datasets[1].structure.components[:3]) == (
        "Zr",
        "Si",
        "Si3N4",
    )


def test_leftmost_si_requests_one_substrate_choice_per_structure_group(tmp_path: Path) -> None:
    paths = (
        _curve(tmp_path / "P1 Si+Zr.xy"),
        _curve(tmp_path / "P2 Si+Zr.xy"),
    )
    preview = preview_import_batch(paths, _preset(), import_batch_id="batch-2")
    groups = {item.substrate_group_id for item in preview.files}
    assert len(groups) == 1
    assert all(item.requires_substrate_choice for item in preview.files)
    group_id = next(iter(groups))
    result = import_dataset_batch(new_project(), preview, {group_id: "Al2O3"})
    assert all(dataset.structure.backing.name == "Al2O3" for dataset in result.updated_project.datasets)


def test_bad_filename_and_bad_data_do_not_block_valid_files(tmp_path: Path) -> None:
    valid = _curve(tmp_path / "good Zr.xy")
    missing_stack = _curve(tmp_path / "missing-stack.xy")
    unreadable = tmp_path / "bad Zr.xy"
    unreadable.write_text("not numeric\n", encoding="utf-8")
    preview = preview_import_batch(
        (missing_stack, valid, unreadable),
        _preset(),
        import_batch_id="batch-3",
    )
    result = import_dataset_batch(new_project(), preview)
    assert result.imported_dataset_ids == ("good",)
    assert len(result.failures) == 2
    assert {Path(item.source_path).name for item in result.failures} == {
        "missing-stack.xy",
        "bad Zr.xy",
    }
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_import.py -q
```

Expected: collection fails because both batch functions are absent.

- [ ] **Step 3: Implement strict filename preview**

Use `rsplit(maxsplit=1)` for the last segment, but do not require a plus sign:

```python
def _strict_filename_materials(stem: str) -> tuple[str, tuple[str, ...]]:
    parts = stem.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("filename must end with a space-separated material stack")
    sample_id, material_segment = parts
    tokens = tuple(value.strip() for value in material_segment.split("+"))
    if not sample_id.strip() or not tokens or any(not value for value in tokens):
        raise ValueError("filename material stack contains an empty token")
    return sample_id, tokens


def _substrate_group_id(tokens: tuple[str, ...]) -> str:
    encoded = json.dumps(tokens, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return sha256(encoded).hexdigest()[:20]


def preview_import_batch(
    paths: Sequence[str | Path],
    preset: MeasurementPreset,
    import_batch_id: str | None = None,
) -> ImportBatchPreview:
    if not isinstance(preset, MeasurementPreset):
        raise TypeError("preset must be MeasurementPreset")
    batch_id = secrets.token_hex(16) if import_batch_id is None else import_batch_id
    if not batch_id.strip():
        raise ValueError("import_batch_id must not be empty")
    files = []
    for declaration in paths:
        path = Path(declaration)
        try:
            dataset_id_stem, tokens = _strict_filename_materials(path.stem)
        except ValueError as error:
            files.append(ImportFilePreview(str(path), path.stem, None, (), None, False, str(error)))
            continue
        group_id = _substrate_group_id(tokens)
        files.append(
            ImportFilePreview(
                str(path),
                path.stem,
                dataset_id_stem,
                tokens,
                group_id,
                tokens[0] == "Si",
            )
        )
    return ImportBatchPreview(batch_id, preset, tuple(files))
```

- [ ] **Step 4: Implement per-file transactional import**

For every valid preview row, choose `Si` unless the row requires an explicit group choice, reverse tokens exactly once, call `automatic_structure`, read the source with that file's optional mapping, and append one immutable dataset. A caught file error becomes:

```python
ImportFailure(
    source_path=row.source_path,
    message=f"{type(error).__name__}: {error}",
    recovery_action=(
        "rename the file and retry or open manual structure editing"
        if row.error is not None
        else "choose the data columns for this file and retry"
    ),
)
```

The successful dataset must be created with:

```python
automation=DatasetAutomation(
    import_batch_id=preview.import_batch_id,
    role=AutomaticRole.UNROUTED,
    status=AutomaticStatus.PENDING,
),
parameter_settings=automatic_settings,
```

Return `ProjectImportResult` even when all rows fail. Set `updated_project.measurement_preset` to the preview preset only after at least one successful import, preserve input order, allocate duplicate IDs with `_dataset_id()`, and select the first imported dataset only when there was no active dataset before the call.

- [ ] **Step 5: Add exact public API signatures and run GREEN**

Add these strings to `tests/architecture/test_public_api.py` and export their types/functions from `api.py`:

```python
SIGNATURES.update(
    {
        "preview_import_batch": "(paths: 'Sequence[str | Path]', preset: 'MeasurementPreset', import_batch_id: 'str | None' = None) -> 'ImportBatchPreview'",
        "import_dataset_batch": "(project: 'XrrProject', preview: 'ImportBatchPreview', substrate_choices: 'Mapping[str, str] | None' = None, column_mappings: 'Mapping[str, DataColumnMapping] | None' = None) -> 'ProjectImportResult'",
    }
)
```

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_import.py tests/unit/services/test_datasets.py tests/architecture/test_public_api.py -q
```

Expected: all selected tests pass, including a mixed good/bad batch with one committed dataset.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/xrr_fitter/services/datasets.py src/xrr_fitter/api.py tests/unit/services/test_automatic_import.py tests/unit/services/test_datasets.py tests/architecture/test_public_api.py && git commit -m "feat: import filename batches transactionally"
```

### Task 4: direct-SLD 参数编译与确定性候选

**Files:**
- Modify: `src/xrr_fitter/fit/parameters.py:62-140`
- Modify: `src/xrr_fitter/fit/initialization.py:21-40,467-507`
- Modify: `src/xrr_fitter/fit/candidates.py:100-244,247-352`
- Modify: `tests/unit/fit/test_problem_compilation.py`
- Modify: `tests/unit/fit/test_candidate_initialization.py`
- Create: `tests/unit/fit/test_direct_sld_initialization.py`

**Interfaces:**
- Changes: direct SLD real definition is free in `[-150e-6, 150e-6] A^-2`; imag remains locked in `[0, 20e-6] A^-2`.
- Produces: `critical_sld_candidates(data, structure) -> tuple[float, ...]` and `direct_sld_start_rows(structure, candidates) -> tuple[tuple[tuple[str, float], ...], ...]`.
- Keeps: formula-density definitions and all expert parameter overrides unchanged.

- [ ] **Step 1: Write RED compilation and determinism tests**

```python
# tests/unit/fit/test_direct_sld_initialization.py
import numpy as np

from tests.support.model_cases import prepared_data
from xrr_fitter.fit.candidates import build_candidate_pool
from xrr_fitter.fit.parameters import parameter_definitions
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec


AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", "Si", 2.329)


def _unknown_structure() -> StructureSpec:
    return StructureSpec(
        AIR,
        (
            LayerSpec("CrSiC", MaterialSpec("CrSiC", None, None, 20e-6 + 0j), 80.0),
            LayerSpec("SiCMo", MaterialSpec("SiCMo", None, None, 20e-6 + 0j), 100.0),
        ),
        SI,
    )


def test_direct_sld_real_is_free_but_absorption_starts_locked() -> None:
    definitions = parameter_definitions(
        prepared_data(),
        _unknown_structure(),
        InstrumentSpec(),
        FitConfig.fast(41),
    )
    by_name = {value.name: value for value in definitions}
    assert by_name["component.0.sld_real_a2"].locked is False
    assert (by_name["component.0.sld_real_a2"].lower, by_name["component.0.sld_real_a2"].upper) == (-150e-6, 150e-6)
    assert by_name["component.0.sld_imag_a2"].locked is True


def test_direct_sld_candidate_rows_are_seed_independent_and_layer_distinct() -> None:
    data = prepared_data(size=96)
    first = build_candidate_pool(data, _unknown_structure(), InstrumentSpec(), np.random.default_rng(1), limit=64)
    second = build_candidate_pool(data, _unknown_structure(), InstrumentSpec(), np.random.default_rng(99), limit=64)
    first_sld = tuple(
        tuple((name, value) for name, value in start.values if "sld_real_a2" in name)
        for start in first
    )
    second_sld = tuple(
        tuple((name, value) for name, value in start.values if "sld_real_a2" in name)
        for start in second
    )
    assert first_sld[:6] == second_sld[:6]
    assert any(len({value for _name, value in row}) > 1 for row in first_sld if row)
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/fit/test_direct_sld_initialization.py tests/unit/fit/test_problem_compilation.py -q
```

Expected: the real SLD assertion fails because both direct components are currently locked, and candidate starts contain no `sld_real_a2` values.

- [ ] **Step 3: Compile physically fixed bounds**

Replace `_material_definitions()` direct-SLD policy with:

```python
def _material_definitions(prefix: str, material: MaterialSpec) -> list[ParameterDefinition]:
    if material.sld_override_a2 is None:
        return []
    return [
        _definition(
            f"{prefix}.sld_real_a2",
            f"{prefix} SLD 实部",
            "Å⁻²",
            "material",
            material.sld_override_a2.real,
            -150e-6,
            150e-6,
            "linear",
            False,
        ),
        _definition(
            f"{prefix}.sld_imag_a2",
            f"{prefix} SLD 吸收部",
            "Å⁻²",
            "material",
            material.sld_override_a2.imag,
            0.0,
            20e-6,
            "linear",
            True,
            expert_only=True,
        ),
    ]
```

- [ ] **Step 4: Add deterministic SLD rows to Stage A**

Derive one critical-edge estimate using `rho = qc**2 / (16*pi)`, clamp it to the compiled bounds, merge it with fixed anchors `(-20, 0, 10, 20, 40, 80, 120) * 1e-6`, and deduplicate in numeric order. Enumerate every direct-SLD path in stable structure order, including ordinary layers, periodic-cell layers, and a direct-SLD backing, but never the Air fronting. Build at most eight rows: declared values first, the data estimate applied to all direct materials second, then cyclic rotations of the fixed anchors across material paths. This is deterministic and grows linearly rather than taking a Cartesian product.

Add the rows to `InitialCandidates` as:

```python
direct_sld_rows: tuple[tuple[tuple[str, float], ...], ...]
```

Protect direct-SLD coverage before the RNG-capped product is sampled. The pool order is exact:

1. `_declared_baseline_start()` first; it includes every direct material's declared real and imaginary SLD.
2. One candidate for each `direct_sld_rows` entry, produced by overlaying that row on the complete declared baseline and deduplicating without changing row order.
3. The remaining `limit` slots from `_selected_combinations()` using the caller's RNG.

Add the selected direct-SLD row as one dimension immediately after geometry in the generated combinations. Reserve the protected rows before computing `generated_limit`; never put them inside the randomly truncated Cartesian product. In `_material_and_interface_values()`, emit `density_scale=1.0` for a direct-SLD layer and use the density hypothesis only for formula-density layers. This makes the first six SLD-bearing starts identical for different RNG seeds while later mixed hypotheses remain random but replayable.

- [ ] **Step 5: Run GREEN and numerical unit tests**

Run:

```bash
python -m pytest -o addopts= tests/unit/fit/test_direct_sld_initialization.py tests/unit/fit/test_candidate_initialization.py tests/unit/fit/test_problem_compilation.py tests/unit/fit/test_stage_search.py tests/unit/test_evaluation.py -q
```

Expected: all selected tests pass; the direct real SLD appears in deterministic starts and formula-density reference tests are unchanged.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/xrr_fitter/fit/parameters.py src/xrr_fitter/fit/initialization.py src/xrr_fitter/fit/candidates.py tests/unit/fit/test_direct_sld_initialization.py tests/unit/fit/test_candidate_initialization.py tests/unit/fit/test_problem_compilation.py && git commit -m "feat: seed direct SLD fitting deterministically"
```

### Task 5: 快速质量门槛、搜索升级、吸收试探与定向 profile

**Files:**
- Create: `src/xrr_fitter/analysis/automatic.py`
- Create: `src/xrr_fitter/fit/automatic.py`
- Modify: `src/xrr_fitter/analysis/report.py:92-117,227-272,336-359,491-535`
- Modify: `src/xrr_fitter/model/analysis.py:447-484`
- Modify: `src/xrr_fitter/services/fitting.py:158-268`
- Create: `tests/unit/analysis/test_automatic_quality.py`
- Create: `tests/unit/fit/test_automatic_refit.py`
- Modify: `tests/unit/analysis/test_report.py`
- Modify: `tests/unit/services/test_fitting.py`

**Interfaces:**
- Produces: `AutomaticQualityDecision(passed, search_upgrade, absorption_names, profile_names, reasons)` in `analysis.automatic`.
- Produces: `assess_automatic_quality(problem, result, profile_limit=4) -> AutomaticQualityDecision`.
- Produces: `refit_from_physical_values(problem, starts, max_nfev, cancelled=None) -> FitSearchResult` in `fit.automatic`.
- Produces: `AutomaticPreparedResult(prepared, fit_result, passed, reason)` in `services.fitting`.
- Produces: `fit_automatic_prepared_dataset(prepared, *, progress=None, cancelled=None, checkpoint=None, local_workers=None) -> AutomaticPreparedResult` in `services.fitting`.
- Changes: `AnalysisRequest.bootstrap_enabled: bool = True`; expert construction keeps `True`, automatic construction passes `False`.

- [ ] **Step 1: Write RED quality-policy tests**

```python
# tests/unit/analysis/test_automatic_quality.py
from types import SimpleNamespace

import pytest

from xrr_fitter.analysis.automatic import assess_automatic_quality


def _problem(
    names: tuple[str, ...],
    definition_names: tuple[str, ...] | None = None,
):
    variables = tuple(SimpleNamespace(name=name) for name in names)
    definitions = tuple(
        SimpleNamespace(name=name)
        for name in (names if definition_names is None else definition_names)
    )
    thresholds = SimpleNamespace(
        equivalent_cost_fraction=0.02,
        equivalent_cost_floor=1e-5,
    )
    return SimpleNamespace(
        variables=variables,
        parameter_definitions=definitions,
        config=SimpleNamespace(confidence=thresholds),
    )


def _result(*, boundaries=(), correlations=(), systematic=False, autocorrelation=False, diagnostics=(), evidence=()):
    uncertainty = SimpleNamespace(
        boundary_hits=boundaries,
        strong_correlations=correlations,
        systematic_residual=systematic,
        residual_autocorrelation=autocorrelation,
        diagnostics=diagnostics,
    )
    candidate = SimpleNamespace(valid=True, objective=0.01, stop_reason="converged")
    return SimpleNamespace(best_candidate=candidate, uncertainty=uncertainty, classification_evidence=evidence)


def test_clean_fast_evidence_passes_without_profiles() -> None:
    decision = assess_automatic_quality(_problem(("component.0.thickness_a",)), _result())
    assert decision.passed is True
    assert decision.profile_names == ()
    assert decision.search_upgrade is False


def test_evidence_selects_at_most_four_relevant_profiles_in_parameter_order() -> None:
    names = (
        "component.0.thickness_a",
        "component.0.sld_real_a2",
        "component.0.roughness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
    )
    result = _result(
        boundaries=(names[0], names[1], names[2]),
        correlations=((names[0], names[3], 0.98),),
        systematic=True,
    )
    decision = assess_automatic_quality(_problem(names), result)
    assert decision.passed is False
    assert decision.profile_names == names[:4]


def test_systematic_residual_can_request_only_direct_sld_absorption() -> None:
    variables = ("component.1.thickness_a",)
    definitions = ("component.0.sld_imag_a2", *variables)
    decision = assess_automatic_quality(
        _problem(variables, definitions),
        _result(systematic=True),
    )
    assert decision.absorption_names == ("component.0.sld_imag_a2",)


@pytest.mark.parametrize(
    "code",
    (
        "distinct_equivalent_clusters",
        "profile_path_merge_failed",
        "insufficient_cluster_support",
    ),
)
def test_existing_candidate_evidence_codes_request_one_search_upgrade(code: str) -> None:
    decision = assess_automatic_quality(
        _problem(("component.0.thickness_a",)),
        _result(evidence=(code,)),
    )
    assert decision.search_upgrade is True
    assert code in decision.reasons


def test_existing_open_primary_profile_code_requests_review_not_search_replay() -> None:
    decision = assess_automatic_quality(
        _problem(("component.0.thickness_a",)),
        _result(evidence=("primary_profile_open",)),
    )
    assert decision.passed is False
    assert decision.search_upgrade is False
    assert decision.profile_names == ("component.0.thickness_a",)
```

Add to `tests/unit/analysis/test_report.py` a spy proving `AnalysisRequest(..., profile_names=(), bootstrap_enabled=False)` calls neither `bootstrap_problem_local` nor profile construction and returns `bootstrap_performed=False`.

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/analysis/test_automatic_quality.py tests/unit/analysis/test_report.py -q
```

Expected: the new module/request field is missing; current `run_analysis()` would call bootstrap unconditionally.

- [ ] **Step 3: Implement the evidence decision without running solvers**

Use existing report fields and preserve compiled parameter order:

```python
# src/xrr_fitter/analysis/automatic.py
from __future__ import annotations

from dataclasses import dataclass


SEARCH_UPGRADE_EVIDENCE = frozenset(
    {
        "distinct_equivalent_clusters",
        "profile_path_merge_failed",
        "insufficient_cluster_support",
    }
)
PROFILE_REVIEW_EVIDENCE = frozenset({"primary_profile_open"})


@dataclass(frozen=True, slots=True)
class AutomaticQualityDecision:
    passed: bool
    search_upgrade: bool
    absorption_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    reasons: tuple[str, ...]


def assess_automatic_quality(problem, result, profile_limit: int = 4) -> AutomaticQualityDecision:
    if profile_limit < 0:
        raise ValueError("profile_limit must be nonnegative")
    best = result.best_candidate
    if best is None or not best.valid:
        return AutomaticQualityDecision(False, True, (), (), ("no valid candidate",))
    report = result.uncertainty
    reasons: list[str] = []
    implicated: set[str] = set()
    if report is None:
        return AutomaticQualityDecision(False, True, (), (), ("missing quality report",))
    if report.boundary_hits:
        reasons.append("parameter boundary hit")
        implicated.update(report.boundary_hits)
    if report.strong_correlations:
        reasons.append("strong parameter correlation")
        for first, second, _value in report.strong_correlations:
            implicated.update((first, second))
    if report.systematic_residual or report.residual_autocorrelation:
        reasons.append("systematic residual")
    if report.diagnostics:
        reasons.append("physical diagnostic")
    evidence = tuple(result.classification_evidence)
    reasons.extend(
        code
        for code in evidence
        if code in SEARCH_UPGRADE_EVIDENCE or code in PROFILE_REVIEW_EVIDENCE
    )
    names = tuple(variable.name for variable in problem.variables)
    definition_names = tuple(
        definition.name for definition in problem.parameter_definitions
    )
    structural = tuple(
        name
        for name in names
        if any(fragment in name for fragment in ("thickness", "sld_", "roughness"))
        or name.startswith("instrument.")
    )
    if reasons and not implicated:
        implicated.update(structural)
    profiles = tuple(name for name in names if name in implicated)[:profile_limit]
    absorption = tuple(
        name
        for name in definition_names
        if name.endswith(".sld_imag_a2")
        and (
            report.systematic_residual
            or any(code in SEARCH_UPGRADE_EVIDENCE for code in evidence)
        )
    )
    search_upgrade = (
        any(code in SEARCH_UPGRADE_EVIDENCE for code in evidence)
        or best.stop_reason == "max_nfev"
    )
    return AutomaticQualityDecision(not reasons, search_upgrade, absorption, profiles, tuple(reasons))
```

Do not invent aliases for classification evidence. The automatic policy consumes the stable codes already emitted by `analysis.classification`: `distinct_equivalent_clusters`, `profile_path_merge_failed`, `insufficient_cluster_support`, and `primary_profile_open`.

- [ ] **Step 4: Make bootstrap optional but auditable**

Add `bootstrap_enabled` to `AnalysisRequest`, and change the branch in `run_analysis()` to:

```python
bootstrap = request.bootstrap
if bootstrap is None and request.bootstrap_enabled:
    bootstrap = _run_bootstrap_with_progress(
        problem,
        best,
        search_result,
        publish,
        cancelled,
        task_runner,
    )
```

Set `bootstrap_performed=bootstrap is not None` in `build_uncertainty_report()`. An explicitly empty `profile_names=()` continues to mean no profile; `None` retains expert automatic profile selection.

- [ ] **Step 5: Add fit-owned bounded local refit**

`fit.automatic` must import only `evaluation`, `fit`, and `model` modules. It encodes each supplied physical mapping, runs analytic `solve_local`, publishes candidates with IDs `automatic-refit-0`, `automatic-refit-1`, selects the minimum valid objective, creates one `FitStageSummary("automatic-refit", ...)`, and binds `fit_search_provenance_sha256`. Reject an empty start list and cap `max_nfev` to a positive integer.

Use two absorption starts per released parameter: current value and `min(2e-6, upper)`. Retain a released-absorption result only when:

```python
gain = baseline.objective - trial.objective
required = max(
    abs(baseline.objective) * problem.config.confidence.equivalent_cost_fraction,
    problem.config.confidence.equivalent_cost_floor,
)
accepted = trial.valid and gain > required
```

- [ ] **Step 6: Compose the single-point automatic path in services.fitting**

Add the service-owned handoff value before composing the stages:

```python
@dataclass(frozen=True, slots=True)
class AutomaticPreparedResult:
    prepared: PreparedDatasetFit
    fit_result: FitResult
    passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be FitResult")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if self.passed and self.reason is not None:
            raise ValueError("passed result must not have a reason")
        if not self.passed and not self.reason:
            raise ValueError("failed quality decision requires a reason")
```

Implement this fixed order:

1. Run existing A-E search without `_search_with_profile_recovery()`.
2. Run `AnalysisRequest(..., profile_names=(), bootstrap_enabled=False)` for covariance/residual quality.
3. If `search_upgrade`, call existing `recover_profile_basin()` and `continue_profile_basin()` once; never add more than its four bounded paths.
4. Recompute the fast report.
5. If absorption is requested, compile a temporary problem with only implicated `.sld_imag_a2` settings unlocked, run the bounded local refit, and retain it only under the gain rule above.
6. Select at most four profile names from the latest evidence, then run a final no-bootstrap analysis with exactly those names.
7. Return `AutomaticPreparedResult(prepared, fit_result, passed, reason)`, where `reason` is `None` on pass and `"; ".join(decision.reasons)` otherwise.

Do not call the expert `_search_with_profile_recovery()` or expert `run_analysis()` defaults from this new function.

- [ ] **Step 7: Run GREEN**

Run:

```bash
python -m pytest -o addopts= tests/unit/analysis/test_automatic_quality.py tests/unit/analysis/test_report.py tests/unit/fit/test_automatic_refit.py tests/unit/services/test_fitting.py tests/regression/test_profile_basin_regressions.py -q
```

Expected: clean evidence performs zero bootstrap and zero profile work; abnormal evidence runs no more than four profiles; expert report/profile-basin regressions remain unchanged.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/xrr_fitter/analysis/automatic.py src/xrr_fitter/fit/automatic.py src/xrr_fitter/analysis/report.py src/xrr_fitter/model/analysis.py src/xrr_fitter/services/fitting.py tests/unit/analysis/test_automatic_quality.py tests/unit/analysis/test_report.py tests/unit/fit/test_automatic_refit.py tests/unit/services/test_fitting.py && git commit -m "feat: add adaptive automatic quality analysis"
```

### Task 6: 物理签名分组、全局 CPU 预算与完成即发布预拟合

**Files:**
- Modify: `src/xrr_fitter/services/parallel.py`
- Modify: `src/xrr_fitter/services/batch.py`
- Modify: `src/xrr_fitter/services/fitting.py`
- Modify: `src/xrr_fitter/model/operations.py`
- Modify: `tests/unit/model/test_operations.py`
- Modify: `tests/unit/services/test_parallel.py`
- Create: `tests/unit/services/test_automatic_batch.py`
- Modify: `tests/unit/services/test_independent_batch.py`

**Interfaces:**
- Changes: `OrderedTaskRunner.run(tasks, completed=None) -> tuple[T, ...]`; return order remains input order, callback order is actual completion order and callback runs on the calling thread.
- Changes: `ProjectFitResult.mode` accepts exactly `{"independent", "joint", "automatic"}`.
- Produces: `automatic_physical_signature(dataset, preset) -> str` in `services.batch`.
- Produces: `fit_automatic_transaction(project, import_batch_id, progress, checkpoint_callback, cancelled, *, seed_branches, prepare_dataset, fit_dataset) -> ProjectFitResult`.
- Produces: `preflight_automatic_fit()` and `fit_automatically()` in `services.fitting`.

- [ ] **Step 1: Write RED completion-order and routing tests**

```python
# append to tests/unit/services/test_parallel.py
def test_completed_callback_observes_finish_order_but_return_stays_input_order() -> None:
    release = threading.Event()
    completed = []

    def slow() -> str:
        release.wait(timeout=2.0)
        return "slow"

    def fast() -> str:
        release.set()
        return "fast"

    with OrderedTaskRunner(2) as runner:
        values = runner.run((slow, fast), completed=lambda index, value: completed.append((index, value)))

    assert values == ("slow", "fast")
    assert completed[0] == (1, "fast")
```

```python
# tests/unit/services/test_automatic_batch.py
from dataclasses import replace
from types import SimpleNamespace

from tests.support.model_cases import dataset_project, final_fit_result, project
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services import batch
from xrr_fitter.services.fitting import AutomaticPreparedResult
from xrr_fitter.services.materials import automatic_structure


def _preset() -> MeasurementPreset:
    return MeasurementPreset(
        "lab",
        BeamSpec("monochromatic", wavelength_a=1.5406),
        InstrumentSpec(instrument_id="lab"),
    )


def _automatic_dataset(dataset_id: str, layers: tuple[str, ...]):
    structure, settings = automatic_structure(layers, "Si")
    return replace(
        dataset_project(dataset_id),
        beam=_preset().beam,
        instrument=_preset().instrument,
        structure=structure,
        parameter_settings=settings,
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )


class RecordingAutomaticFits:
    def __init__(self) -> None:
        self.prefit_dataset_ids: set[str] = set()
        self.checkpoints = []

    def seeds(self, value):
        return (
            {dataset.dataset_id: index + 1 for index, dataset in enumerate(value.datasets)},
            101,
            102,
        )

    def prepare(self, value, dataset_id, _seed):
        index = next(
            index
            for index, dataset in enumerate(value.datasets)
            if dataset.dataset_id == dataset_id
        )
        self.prefit_dataset_ids.add(dataset_id)
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_index=index,
            updated_dataset=value.datasets[index],
        )

    def fit_dataset(self, prepared, **_kwargs):
        return AutomaticPreparedResult(prepared, final_fit_result(), True, None)

    def checkpoint(self, value) -> None:
        self.checkpoints.append(value)


def test_mixed_import_batch_routes_singletons_and_matching_points_separately(monkeypatch) -> None:
    other_batch = replace(
        _automatic_dataset("d", ("Zr",)),
        automation=replace(
            _automatic_dataset("d", ("Zr",)).automation,
            import_batch_id="batch-2",
        ),
    )
    value = replace(
        project(
            _automatic_dataset("a", ("Zr",)),
            _automatic_dataset("b", ("Zr",)),
            _automatic_dataset("c", ("TaN",)),
            other_batch,
        ),
        measurement_preset=_preset(),
    )
    records = tuple(
        SimpleNamespace(dataset_id=item.dataset_id, status=SimpleNamespace(value="ok"))
        for item in value.datasets
    )
    monkeypatch.setattr(
        batch,
        "inspect_sources",
        lambda _value: SimpleNamespace(valid=True, issues=(), datasets=records),
    )
    calls = RecordingAutomaticFits()

    result = batch.fit_automatic_transaction(
        value,
        None,
        None,
        calls.checkpoint,
        None,
        seed_branches=calls.seeds,
        prepare_dataset=calls.prepare,
        fit_dataset=calls.fit_dataset,
    )

    by_id = {item.dataset_id: item.automation for item in result.updated_project.datasets}
    assert calls.prefit_dataset_ids == {"a", "b", "c", "d"}
    assert by_id["a"].role is AutomaticRole.JOINT
    assert by_id["b"].fit_group_id == by_id["a"].fit_group_id
    assert by_id["a"].status is AutomaticStatus.REFINING
    assert by_id["c"].role is AutomaticRole.SINGLE
    assert by_id["c"].status is AutomaticStatus.PASSED
    assert by_id["d"].role is AutomaticRole.SINGLE
    assert by_id["d"].fit_group_id != by_id["a"].fit_group_id
    assert result.mode == "automatic"


def test_physical_signature_separates_backing_and_beam() -> None:
    preset = _preset()
    first = _automatic_dataset("a", ("Zr",))
    different_backing = replace(
        first,
        structure=replace(
            first.structure,
            backing=first.structure.components[0].material,
        ),
    )
    different_beam = replace(
        first,
        beam=BeamSpec("monochromatic", wavelength_a=0.7093),
    )
    signatures = {
        batch.automatic_physical_signature(value, preset)
        for value in (first, different_backing, different_beam)
    }
    assert len(signatures) == 3
```

Append this value test in `tests/unit/model/test_operations.py`:

```python
def test_project_fit_result_accepts_only_the_three_declared_modes() -> None:
    current = project()

    assert ProjectFitResult("automatic", (), (), current).mode == "automatic"
    with pytest.raises(ValueError, match="mode"):
        ProjectFitResult("parallel", (), (), current)
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/model/test_operations.py tests/unit/services/test_parallel.py tests/unit/services/test_automatic_batch.py -q
```

Expected: `OrderedTaskRunner.run()` rejects `completed`, and the automatic transaction is absent.

- [ ] **Step 3: Implement caller-thread completion callbacks**

For threaded execution, submit all futures, map each future to its input index, iterate `concurrent.futures.as_completed`, call `future.result()` on the caller thread, store it into a pre-sized list, and invoke `completed(index, value)` immediately. On the first exception, cancel every unfinished future and re-raise. For `max_workers=1`, invoke the callback after each task. Return `tuple(results)` in input order.

- [ ] **Step 4: Implement canonical physical grouping**

Hash canonical JSON containing only behavior-changing values:

```python
payload = {
    "layers": tuple(
        (
            component.name,
            component.material.name,
            component.material.formula,
            component.material.sld_override_a2 is not None,
        )
        for component in dataset.structure.components
    ),
    "backing": (
        dataset.structure.backing.name,
        dataset.structure.backing.formula,
        dataset.structure.backing.sld_override_a2 is not None,
    ),
    "beam": tuple(getattr(dataset.beam, field) for field in dataset.beam.__dataclass_fields__),
    "import_angle_offset_deg": dataset.import_angle_offset_deg,
    "instrument": tuple(
        getattr(dataset.instrument, field) for field in dataset.instrument.__dataclass_fields__
    ),
    "preset": (
        tuple(getattr(preset.beam, field) for field in preset.beam.__dataclass_fields__),
        tuple(
            getattr(preset.instrument, field)
            for field in preset.instrument.__dataclass_fields__
        ),
        preset.import_angle_offset_deg,
    ),
    "structure_modes": tuple(type(component).__name__ for component in dataset.structure.components),
}
```

Do not include source path, dataset ID, display name, thickness values, roughness values, import order, or prior fit results in the physical signature. In `fit_automatic_transaction()`, group by `(dataset.automation.import_batch_id, automatic_physical_signature(...))`, never by the physical signature alone. Derive `fit_group_id` by hashing canonical JSON containing both values so identical structures from different import batches cannot share parameters.

- [ ] **Step 5: Run every point under one CPU budget and publish completions**

Use existing `_worker_allocations(total_workers, count)`. Before starting, assign singleton roles to `SINGLE`, multi-member roles to `JOINT`, `fit_group_id` to the signature hash, and status `REFINING`. Each completed prefit must immediately update that dataset's `last_valid_result`, checkpoint, parameter settings, and automation reason, then call `checkpoint_callback(working)` before another input-order result is awaited.

For singleton groups, convert the final status to `PASSED`, `REVIEW`, or `FAILED`. Multi-member groups remain `REFINING` until Task 7's joint stage. A preparation/read/source failure affects only its file and produces a `DatasetFitResult` with `FAILED`; it must not cancel another group.

Change `ProjectFitResult.__post_init__()` at the same time; its exact validation becomes:

```python
if self.mode not in {"independent", "joint", "automatic"}:
    raise ValueError("mode must be independent, joint, or automatic")
```

- [ ] **Step 6: Add synchronous service entrypoints**

Use exact signatures:

```python
_AUTOMATIC_RUNNABLE = frozenset(
    {AutomaticStatus.PENDING, AutomaticStatus.REFINING, AutomaticStatus.REVIEW}
)


def _automatic_dataset_ids(
    project: XrrProject,
    import_batch_id: str | None,
) -> tuple[str, ...]:
    return tuple(
        dataset.dataset_id
        for dataset in project.datasets
        if dataset.automation.role is not AutomaticRole.MANUAL
        and dataset.automation.status in _AUTOMATIC_RUNNABLE
        and (
            import_batch_id is None
            or dataset.automation.import_batch_id == import_batch_id
        )
    )


def preflight_automatic_fit(
    project: XrrProject,
    import_batch_id: str | None = None,
) -> FitReadiness:
    if project.measurement_preset is None:
        return FitReadiness(False, "automatic fit requires a measurement preset")
    dataset_ids = _automatic_dataset_ids(project, import_batch_id)
    if not dataset_ids:
        return FitReadiness(False, "no runnable automatic datasets")
    try:
        records = {
            record.dataset_id: record
            for record in inspect_sources(project).datasets
        }
        seeds, _joint_seed, _mcmc_seed = service_seed_branches(project)
        for dataset_id in dataset_ids:
            record = records[dataset_id]
            if record.status.value != "ok":
                return FitReadiness(False, record.message)
            prepare_dataset_fit(project, dataset_id, seeds[dataset_id])
    except Exception as error:
        return FitReadiness(False, str(error) or type(error).__name__)
    return FitReadiness(True, "ready")


def fit_automatically(
    project: XrrProject,
    import_batch_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> ProjectFitResult:
    readiness = preflight_automatic_fit(project, import_batch_id)
    if not readiness.ready:
        raise ValueError(readiness.message)
    from xrr_fitter.services.batch import fit_automatic_transaction

    return fit_automatic_transaction(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        None,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_automatic_prepared_dataset,
    )
```

The implementation represented by these signatures must select only automatic datasets with status `PENDING`, `REFINING`, or `REVIEW`; when `import_batch_id` is provided it must additionally require an exact batch match. It delegates to `fit_automatic_transaction` with top-level pickle-safe fitting functions.

- [ ] **Step 7: Run GREEN**

Run:

```bash
python -m pytest -o addopts= tests/unit/model/test_operations.py tests/unit/services/test_parallel.py tests/unit/services/test_automatic_batch.py tests/unit/services/test_independent_batch.py tests/unit/services/test_fitting.py -q
```

Expected: callback order follows completion order, returned results remain ordered, and a mixed batch creates one joint group plus one singleton without cross-group failure.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/xrr_fitter/services/parallel.py src/xrr_fitter/services/batch.py src/xrr_fitter/services/fitting.py src/xrr_fitter/model/operations.py tests/unit/model/test_operations.py tests/unit/services/test_parallel.py tests/unit/services/test_automatic_batch.py tests/unit/services/test_independent_batch.py && git commit -m "feat: route automatic fit groups in parallel"
```

### Task 7: 预拟合共识联合初始化、粗糙度释放与隔离点重试

**Files:**
- Modify: `src/xrr_fitter/services/parameters.py:197-215`
- Modify: `src/xrr_fitter/fit/joint_sharing.py:19-36,105-155`
- Modify: `src/xrr_fitter/fit/joint_pipeline.py:51-62,549-601`
- Modify: `src/xrr_fitter/services/fitting.py:270-421`
- Modify: `src/xrr_fitter/services/batch.py`
- Modify: `tests/unit/services/test_parameters.py`
- Modify: `tests/unit/fit/test_joint_problem.py`
- Modify: `tests/unit/fit/test_joint_pipeline.py`
- Modify: `tests/unit/services/test_automatic_batch.py`
- Create: `tests/unit/services/test_automatic_joint.py`
- Modify: `tests/integration/test_joint_fit_workflow.py`

**Interfaces:**
- Changes: one `SharingRule` may contain multiple distinct parameter references from the same dataset.
- Produces: `consensus_joint_vector(problem, candidates_by_dataset) -> np.ndarray`.
- Changes: `JointFitRequest.initial_unit_vector: np.ndarray | None = None`.
- Produces: `automatic_sharing_rules(prepared, fit_group_id, *, share_roughness) -> tuple[SharingRule, ...]` in `services.fitting`.
- Produces: `fit_automatic_joint_group(prepared, prefits, fit_group_id, *, progress=None, cancelled=None, checkpoint=None) -> tuple[AutomaticPreparedResult, ...]` in `services.fitting`.
- Changes: `fit_automatic_transaction(...)` adds the injected keyword-only `fit_joint` callable and invokes it only for qualified multi-member groups.
- Consumes: Task 6 prefit results and returns final joint projections plus isolated retries.

- [ ] **Step 1: Write RED sharing and consensus tests**

First extend Task 6's `RecordingAutomaticFits`, replace the transaction call, and replace its `REFINING` assertion so the final transaction exercises its now-required joint callback:

```python
# tests/unit/services/test_automatic_batch.py
def fit_joint(self, prepared, prefits, fit_group_id, **_kwargs):
    self.joint_groups.append(tuple(item.dataset_id for item in prepared))
    return tuple(prefits)


# in test_mixed_import_batch_routes_singletons_and_matching_points_separately
result = batch.fit_automatic_transaction(
    value,
    None,
    None,
    calls.checkpoint,
    None,
    seed_branches=calls.seeds,
    prepare_dataset=calls.prepare,
    fit_dataset=calls.fit_dataset,
    fit_joint=calls.fit_joint,
)
assert calls.joint_groups == [("a", "b")]
by_id = {item.dataset_id: item.automation for item in result.updated_project.datasets}
assert by_id["a"].status is AutomaticStatus.PASSED
assert by_id["b"].status is AutomaticStatus.PASSED
```

Initialize `self.joint_groups = []` in `RecordingAutomaticFits.__init__()`. Pass `import_batch_id=None` here so the same test continues proving that batch 2 dataset `d` is not merged into batch 1.

```python
# append to tests/unit/fit/test_joint_problem.py
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.model.structure import LayerSpec, MaterialSpec


def _problem_with_two_direct_sld_layers():
    base = _problem(seed=809)
    material = MaterialSpec("CrSiC", None, None, 20e-6 + 0j)
    structure = replace(
        base.structure,
        components=(
            LayerSpec("CrSiC lower", material, 80.0),
            LayerSpec("CrSiC upper", material, 100.0),
        ),
    )
    return compile_fit_problem(base.data, structure, base.instrument, base.config)


def test_one_material_group_can_share_repeated_members_from_the_same_dataset() -> None:
    problems = (_problem_with_two_direct_sld_layers(), _problem_with_two_direct_sld_layers())
    rule = SharingRule(
        "material:CrSiC:sld_real",
        (
            ParameterReference("a", "component.0.sld_real_a2"),
            ParameterReference("a", "component.1.sld_real_a2"),
            ParameterReference("b", "component.0.sld_real_a2"),
            ParameterReference("b", "component.1.sld_real_a2"),
        ),
    )
    joint = compile_joint_problem(("a", "b"), problems, (rule,))
    shared = next(value for value in joint.global_variables if value.sharing_key == rule.sharing_key)
    first_index = _coordinate_index(problems[0], "component.0.sld_real_a2")
    second_index = _coordinate_index(problems[0], "component.1.sld_real_a2")
    assert len(shared.members) == 4
    assert joint.scatter_maps[0][first_index] == joint.scatter_maps[0][second_index]
```

```python
# append to tests/unit/fit/test_joint_pipeline.py
import xrr_fitter.fit.joint_pipeline as joint_pipeline


def test_joint_request_uses_prefit_consensus_instead_of_declared_initial(monkeypatch) -> None:
    joint = _joint_problem()
    consensus = np.full(len(joint.global_variables), 0.73)
    observed = []
    evaluate = joint_pipeline.evaluate_joint_vector

    def capture(problem, unit):
        observed.append(unit.copy())
        return evaluate(problem, unit)

    monkeypatch.setattr(joint_pipeline, "evaluate_joint_vector", capture)
    joint_pipeline.run_joint_fit(joint_pipeline.JointFitRequest(joint, initial_unit_vector=consensus))
    assert np.array_equal(observed[0], consensus)


def test_joint_request_schema_includes_optional_initial_vector() -> None:
    assert [field.name for field in fields(joint_pipeline.JointFitRequest)] == [
        "problem",
        "resume_checkpoints",
        "initial_unit_vector",
    ]
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/fit/test_joint_problem.py tests/unit/fit/test_joint_pipeline.py tests/unit/services/test_automatic_batch.py tests/unit/services/test_automatic_joint.py -q
```

Expected: same-dataset sharing raises `sharing group may contain at most one member per dataset`, and `JointFitRequest` rejects `initial_unit_vector`.

- [ ] **Step 3: Allow distinct same-dataset references and build consensus**

Remove only the duplicate dataset-ID rejection in both `services.parameters.validate_sharing_rules()` and `fit.joint_sharing._rule_definitions()`. Keep duplicate `ParameterReference`, multiple ownership, family/bounds compatibility and missing/free-coordinate checks unchanged.

Implement consensus by reading every `JointVariable.members` reference from its dataset's best prefit candidate, converting the corresponding local unit coordinate, and taking `float(np.median(values))`. Local variables have one member and therefore retain that point's prefit value. Reject missing/invalid candidates and non-finite consensus values.

Validate `JointFitRequest.initial_unit_vector` as a copied, read-only vector with shape `(len(problem.global_variables),)`, finite values and `[0,1]` bounds. `_fresh_state()` uses it when provided; resume requests must reject a simultaneous explicit initial vector.

- [ ] **Step 4: Generate automatic sharing rules**

Use these rule families:

- Formula-density `density_scale`: group by exact `MaterialSpec.name` across all occurrences and points.
- Direct SLD real: group by exact material name; direct SLD imaginary joins only when it was evidence-released in every member.
- Roughness: group by exact component path across points when `share_roughness=True`.
- Thickness, background, scale, angle offset, resolution, footprint and oxide thickness: no rule, always local.

Rule keys must include `fit_group_id`, family and material/path so they cannot collide with expert rules.

- [ ] **Step 5: Isolate outliers before joint consensus**

Exclude a prefit when it has no valid candidate, non-finite objective, physical diagnostics that make the candidate invalid, or failed automatic quality. For groups of at least three valid prefit points, also isolate objective outliers above `median + 3 * max(MAD, equivalent_cost_floor)`; do not apply this objective-outlier rule to a two-point group. Persist role `ISOLATED_RETRY`, status `REFINING`, `statistics_member=False`, and an auditable reason.

If fewer than two points remain, do not claim joint success: keep each available independent result, mark it `REVIEW` with reason `insufficient qualified points for joint refinement`, and continue other groups.

- [ ] **Step 6: Joint refine, release roughness once, then retry isolated points**

Run the first joint fit with material and roughness rules and prefit consensus. Compare each projection's objective to its prefit objective using:

```python
allowed = max(
    abs(prefit.objective) * thresholds.equivalent_cost_fraction,
    thresholds.equivalent_cost_floor,
)
conflict = joint.objective > prefit.objective + allowed or joint_report.systematic_residual
```

If any point conflicts, remove only the automatic roughness rules and rerun joint refinement once, initialized from the first joint result. Never silently switch to independent results while labeling the result joint.

After the accepted joint result, lock the group's material SLD/density values into each isolated point's settings, call `fit_automatic_prepared_dataset()` independently, and set:

- `PASSED` + `statistics_member=True` when retry quality passes;
- `REVIEW` + `statistics_member=False` when it has a valid best result but evidence remains;
- `FAILED` when it has no publishable candidate.

Qualified joint points become `PASSED` and statistics members only after the accepted joint projection passes its local quality checks.

Complete the Task 6 dispatcher by passing the new top-level, spawn-safe callable:

```python
return fit_automatic_transaction(
    project,
    import_batch_id,
    progress_callback,
    checkpoint_callback,
    cancelled,
    seed_branches=service_seed_branches,
    prepare_dataset=prepare_dataset_fit,
    fit_dataset=fit_automatic_prepared_dataset,
    fit_joint=fit_automatic_joint_group,
)
```

Use the same keyword in `fit_automatically()` and `automatic_worker_handler()`; do not close over a local function because the worker uses multiprocessing `spawn`.

- [ ] **Step 7: Run GREEN and integration tests**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_parameters.py tests/unit/fit/test_joint_problem.py tests/unit/fit/test_joint_pipeline.py tests/unit/services/test_automatic_batch.py tests/unit/services/test_automatic_joint.py tests/integration/test_joint_fit_workflow.py -q
```

Expected: repeated same-material coordinates share one global coordinate, the first evaluated joint vector equals the prefit consensus, roughness releases at most once, and isolated retry membership is explicit.

- [ ] **Step 8: Commit Task 7**

```bash
git add src/xrr_fitter/services/parameters.py src/xrr_fitter/fit/joint_sharing.py src/xrr_fitter/fit/joint_pipeline.py src/xrr_fitter/services/fitting.py src/xrr_fitter/services/batch.py tests/unit/services/test_parameters.py tests/unit/fit/test_joint_problem.py tests/unit/fit/test_joint_pipeline.py tests/unit/services/test_automatic_batch.py tests/unit/services/test_automatic_joint.py tests/integration/test_joint_fit_workflow.py && git commit -m "feat: refine matching points with joint consensus"
```

### Task 8: 逐层结果与批次均匀性汇总

**Files:**
- Create: `src/xrr_fitter/services/results.py`
- Create: `tests/unit/services/test_automatic_results.py`
- Modify: `src/xrr_fitter/physics/materials.py`
- Modify: `tests/unit/physics/test_material_sld.py`

**Interfaces:**
- Produces: `summarize_automatic_results(project, import_batch_id=None) -> AutomaticResultSummary`.
- Consumes: Task 1 result values and persisted best candidates.
- Formula-density layers expose nominal density, relative density and fitted g/cm3; direct-SLD layers expose `fitted_density_g_cm3=None` and the exact note `配比未知，无法换算`.

- [ ] **Step 1: Write RED result and statistics tests**

```python
# tests/unit/services/test_automatic_results.py
from dataclasses import replace

import pytest

from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    project,
)
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)
from xrr_fitter.model.parameters import ParameterValue
from xrr_fitter.services.materials import automatic_structure
from xrr_fitter.services.results import summarize_automatic_results


def _parameter(name: str, value: float) -> ParameterValue:
    if name.endswith("sld_real_a2"):
        return ParameterValue(name, value, -150e-6, 150e-6)
    if name.endswith("sld_imag_a2"):
        return ParameterValue(name, value, 0.0, 20e-6)
    if name.endswith("density_scale"):
        return ParameterValue(name, value, 0.2, 1.2)
    if name.endswith("roughness_a"):
        return ParameterValue(name, value, 0.0, 100.0)
    return ParameterValue(name, value, 2.0, 2e5)


def _point(
    dataset_id: str,
    *,
    thicknesses: tuple[float, ...],
    direct_sld: float | None = None,
    passed: bool,
):
    tokens = ("Si3N4", "CrSiC") if len(thicknesses) == 2 else ("Zr",)
    structure, settings = automatic_structure(tokens, "Al2O3")
    parameters = []
    for index, (layer, thickness) in enumerate(
        zip(structure.components, thicknesses, strict=True)
    ):
        prefix = f"component.{index}"
        parameters.extend(
            (
                _parameter(f"{prefix}.thickness_a", thickness),
                _parameter(f"{prefix}.roughness_a", 3.0),
                _parameter(
                    f"{prefix}.density_scale",
                    1.0 if layer.material.sld_override_a2 is not None else 0.9,
                ),
            )
        )
        if layer.material.sld_override_a2 is not None:
            parameters.extend(
                (
                    _parameter(
                        f"{prefix}.sld_real_a2",
                        20e-6 if direct_sld is None else direct_sld,
                    ),
                    _parameter(f"{prefix}.sld_imag_a2", 0.0),
                )
            )
    candidate = replace(
        fit_candidate(f"{dataset_id}-candidate"),
        parameters=tuple(parameters),
    )
    status = AutomaticStatus.PASSED if passed else AutomaticStatus.REVIEW
    automation = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=AutomaticRole.JOINT,
        status=status,
        statistics_member=passed,
        reason=None if passed else "synthetic quality failure",
    )
    return replace(
        dataset_project(dataset_id, result=final_fit_result(candidate)),
        structure=structure,
        parameter_settings=settings,
        automation=automation,
    )


def _fitted_project(*points):
    return project(*points)


def test_known_and_unknown_material_results_do_not_confuse_mass_density() -> None:
    value = _fitted_project(
        _point("p1", thicknesses=(90.0, 100.0), direct_sld=24e-6, passed=True),
    )
    summary = summarize_automatic_results(value, "batch-1")
    known, unknown = summary.datasets[0].layers
    assert known.fitted_density_g_cm3 == pytest.approx(known.nominal_density_g_cm3 * known.density_scale)
    assert unknown.nominal_density_g_cm3 is None
    assert unknown.fitted_density_g_cm3 is None
    assert unknown.density_note == "配比未知，无法换算"
    assert unknown.electron_density_a3 == pytest.approx(unknown.sld_real_a2 / 2.8179403262e-5)


def test_uniformity_uses_only_passed_members_and_population_standard_deviation() -> None:
    value = _fitted_project(
        _point("p1", thicknesses=(90.0,), passed=True),
        _point("p2", thicknesses=(100.0,), passed=True),
        _point("p3", thicknesses=(500.0,), passed=False),
    )
    item = summarize_automatic_results(value, "batch-1").uniformity[0]
    assert item.count == 2
    assert item.mean_thickness_a == 95.0
    assert item.population_std_a == 5.0
    assert item.cv_percent == pytest.approx(5.0 / 95.0 * 100.0)
    assert item.relative_range_percent == pytest.approx(10.0 / 95.0 * 100.0)
```

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_results.py -q
```

Expected: `xrr_fitter.services.results` is absent.

- [ ] **Step 3: Implement layer projection**

Add `CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5` to `physics.materials`. For each ordinary filename-derived `LayerSpec`, read `component.{index}.thickness_a`, `.roughness_a`, `.density_scale`, `.sld_real_a2`, and `.sld_imag_a2` from the best candidate's parameter map, falling back only to that layer's declared locked value. Formula materials calculate SLD with `material_sld()` at the dataset beam's primary wavelength; direct materials use fitted real/imag values and force density scale 1.

Reject periodic/gradient components in this automatic summary with a clear `ValueError`; they remain supported by the expert result views but are not produced by the filename contract.

- [ ] **Step 4: Implement statistics with the standard library**

Group rows by `(fit_group_id, layer_index, material_name)`, include only `automation.statistics_member`, and calculate:

```python
mean = sum(values) / len(values)
population_std = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
cv_percent = 0.0 if mean == 0.0 else population_std / mean * 100.0
relative_range_percent = 0.0 if mean == 0.0 else (max(values) - min(values)) / mean * 100.0
```

Keep all review/failed datasets in `AutomaticResultSummary.datasets`; exclusion affects only `uniformity`.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_automatic_results.py tests/unit/physics/test_material_sld.py -q
```

Expected: all selected tests pass and no service-level NumPy import is introduced.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/xrr_fitter/services/results.py src/xrr_fitter/physics/materials.py tests/unit/services/test_automatic_results.py tests/unit/physics/test_material_sld.py && git commit -m "feat: summarize automatic layer uniformity"
```

### Task 9: 自动 worker/API 与 GUI 默认路径

**Files:**
- Modify: `src/xrr_fitter/services/workers.py`
- Modify: `src/xrr_fitter/services/fitting.py`
- Modify: `src/xrr_fitter/api.py`
- Modify: `tests/architecture/test_public_api.py`
- Modify: `src/xrr_fitter/gui/data/import_dialog.py`
- Create: `src/xrr_fitter/gui/data/substrate_dialog.py`
- Modify: `src/xrr_fitter/gui/data/panel.py`
- Modify: `src/xrr_fitter/gui/fitting/controller.py`
- Modify: `src/xrr_fitter/gui/fitting/panel.py`
- Modify: `src/xrr_fitter/gui/results/panel.py`
- Create: `src/xrr_fitter/gui/results/automatic.py`
- Modify: `src/xrr_fitter/gui/workspace.py`
- Modify: `tests/unit/services/test_workers.py`
- Modify: `tests/gui/test_data_import.py`
- Modify: `tests/gui/test_fit_controller.py`
- Modify: `tests/gui/test_fit_progress.py`
- Modify: `tests/gui/test_results.py`
- Create: `tests/integration/test_gui_automatic_workflow.py`

**Interfaces:**
- Produces: `start_automatic_fit_job(project, import_batch_id=None, checkpoint_path=None) -> OperationJob`.
- Reuses: existing `progress`, `checkpoint`, `fit_result`, `cancelled`, `error`, `stopped` event kinds; no new IPC payload kind.
- GUI default: successful import emits its `import_batch_id` and immediately calls `FitPanel.start_automatic_fit()`.
- Expert mode: existing batch selector and `start_fit()` remain available and keep calling `start_fit_job()`.

- [ ] **Step 1: Write RED worker and GUI-flow tests**

Add a worker unit test that monkeypatches the process owner and asserts the request contains `import_batch_id`. Add GUI tests with these exact outcomes:

```python
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog

import xrr_fitter.api as api


def _write_automatic_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _saved_preset() -> api.MeasurementPreset:
    return api.MeasurementPreset(
        "gui-lab",
        api.BeamSpec("monochromatic", wavelength_a=1.5406),
        api.InstrumentSpec(instrument_id="gui-lab"),
    )


def project_with_saved_preset() -> api.XrrProject:
    return replace(api.new_project(), measurement_preset=_saved_preset())


class FakeJob:
    is_running = True

    def poll(self):
        return ()

    def cancel(self) -> None:
        self.is_running = False

    def force_stop(self) -> None:
        self.is_running = False

    def close(self) -> None:
        pass


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, ...]:
    return (_write_automatic_curve(tmp_path / "P1 Zr.xy"),)


@pytest.fixture
def si_left_sources(tmp_path: Path) -> tuple[Path, ...]:
    return (
        _write_automatic_curve(tmp_path / "P1 Si+Zr.xy"),
        _write_automatic_curve(tmp_path / "P2 Si+Zr.xy"),
    )


@pytest.fixture
def automatic_window(qtbot):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project_with_saved_preset()))
    qtbot.addWidget(window)
    return window


def test_reused_measurement_preset_skips_repeated_import_configuration(
    qtbot,
    automatic_window,
    sources,
    monkeypatch,
):
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_args, **_kwargs: FakeJob())
    automatic_window.document.replace_project(project_with_saved_preset())
    automatic_window.data_panel.import_paths(sources)
    assert automatic_window.findChild(QDialog, "importDialog") is None


def test_ambiguous_si_left_layer_prompts_once_per_structure_group(
    qtbot,
    automatic_window,
    si_left_sources,
    monkeypatch,
):
    from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog

    dialogs = []

    def accept_substrate(dialog):
        dialogs.append(dialog)
        dialog.substrate_editor.setText("Al2O3")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SubstrateDialog, "exec", accept_substrate)
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_args, **_kwargs: FakeJob())
    automatic_window.data_panel.import_paths(si_left_sources)
    assert len(dialogs) == 1


def test_successful_import_starts_automatic_fit_without_mode_selection(qtbot, automatic_window, sources, monkeypatch):
    starts = []
    monkeypatch.setattr(
        api,
        "start_automatic_fit_job",
        lambda project, import_batch_id=None, checkpoint_path=None: (
            starts.append((project, import_batch_id)) or FakeJob()
        ),
    )
    automatic_window.data_panel.import_paths(sources)
    assert len(starts) == 1
    assert automatic_window.document.project.batch_mode == "independent"
```

The integration test must import one valid and one bad file, prove the valid file reaches a published curve, and prove the bad row exposes its filename and recovery action without blocking the worker.

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_workers.py tests/gui/test_data_import.py tests/gui/test_fit_controller.py tests/gui/test_fit_progress.py tests/gui/test_results.py tests/integration/test_gui_automatic_workflow.py -q
```

Expected: automatic worker/API and GUI entrypoints are absent.

- [ ] **Step 3: Add the process-safe automatic worker**

Create a frozen request containing project, optional batch ID and optional checkpoint path. Its top-level handler calls `services.fitting.automatic_worker_handler`. Publish prefit snapshots through the existing `checkpoint` callback and the terminal `ProjectFitResult` through `fit_result`. Preserve cancellation and force-stop behavior by using the existing `OperationJob` owner without adding an executor.

Add these exact top-level definitions so the `spawn` target and request are pickle-safe:

```python
@dataclass(frozen=True, slots=True)
class _AutomaticFitJobRequest:
    project: XrrProject
    import_batch_id: str | None
    checkpoint_path: str | None


def _run_automatic_fit_worker(
    request: _AutomaticFitJobRequest,
    queue,
    cancellation,
) -> None:
    try:
        def progress(value) -> None:
            _put(queue, "progress", value)

        def checkpoint(value) -> None:
            if request.checkpoint_path is not None:
                save_project(value, request.checkpoint_path)
            _put(queue, "checkpoint", value)

        result = automatic_worker_handler(
            request.project,
            request.import_batch_id,
            progress,
            checkpoint,
            cancellation.is_set,
        )
        if result.cancelled:
            _put(queue, "cancelled", "requested")
        else:
            _put(queue, "fit_result", result)
    except BaseException as error:
        _put(queue, "error", _operation_error(error))
    finally:
        _put(queue, "stopped", None)


def start_automatic_fit_job(
    project: XrrProject,
    import_batch_id: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> OperationJob:
    path = None if checkpoint_path is None else str(checkpoint_path)
    request = _AutomaticFitJobRequest(project, import_batch_id, path)
    return _start(_run_automatic_fit_worker, request)
```

Add the matching top-level service handler in `services.fitting`:

```python
def automatic_worker_handler(
    project: XrrProject,
    import_batch_id: str | None,
    progress_callback: ProgressCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    cancelled: CancellationProbe | None,
) -> ProjectFitResult:
    from xrr_fitter.services.batch import fit_automatic_transaction

    return fit_automatic_transaction(
        project,
        import_batch_id,
        progress_callback,
        checkpoint_callback,
        cancelled,
        seed_branches=service_seed_branches,
        prepare_dataset=prepare_dataset_fit,
        fit_dataset=fit_automatic_prepared_dataset,
        fit_joint=fit_automatic_joint_group,
    )
```

- [ ] **Step 4: Complete the public API surface**

Export the Task 1 values and these operations in `api.__all__` and the exact-signature test:

```python
preview_import_batch
import_dataset_batch
preflight_automatic_fit
fit_automatically
start_automatic_fit_job
summarize_automatic_results
```

Add these exact entries to `SIGNATURES` (the two import signatures were added in Task 3):

```python
SIGNATURES.update(
    {
        "preflight_automatic_fit": "(project: 'XrrProject', import_batch_id: 'str | None' = None) -> 'FitReadiness'",
        "fit_automatically": "(project: 'XrrProject', import_batch_id: 'str | None' = None, progress_callback: 'ProgressCallback | None' = None, checkpoint_callback: 'CheckpointCallback | None' = None) -> 'ProjectFitResult'",
        "start_automatic_fit_job": "(project: 'XrrProject', import_batch_id: 'str | None' = None, checkpoint_path: 'str | Path | None' = None) -> 'OperationJob'",
        "summarize_automatic_results": "(project: 'XrrProject', import_batch_id: 'str | None' = None) -> 'AutomaticResultSummary'",
    }
)
```

Add these GUI use-case groups and keep every GUI domain call in the form `api.<name>`:

```python
GUI_USE_CASES.update(
    {
        "automatic_import": ("preview_import_batch", "import_dataset_batch"),
        "automatic_fit": (
            "preflight_automatic_fit",
            "fit_automatically",
            "start_automatic_fit_job",
        ),
        "automatic_results": ("summarize_automatic_results",),
    }
)
```

- [ ] **Step 5: Make import configuration one-time and substrate selection group-scoped**

On the first import, reuse `ImportDialog` to construct and persist:

```python
api.MeasurementPreset(
    preset_id=dialog.instrument_spec().instrument_id or "default-measurement",
    beam=dialog.beam_spec(),
    instrument=dialog.instrument_spec(),
)
```

On later imports, reuse `project.measurement_preset` without opening the full dialog. Provide an explicit “更换测量预设” expert action to clear/change it.

`SubstrateDialog` displays the shared layer stack, uses a `QLineEdit` for arbitrary substrate material token, defaults to an empty value rather than guessing, validates nonempty input, and returns one token for the group's `substrate_group_id`. It appears once for every unique required group.

After `import_dataset_batch`, show every `ImportFailure` in a single table with filename, message, and recovery action. Emit `automatic_fit_requested(import_batch_id)` when at least one dataset imported.

- [ ] **Step 6: Make automatic fitting the default control**

Add `FitController.start_automatic_fit()` calling `api.start_automatic_fit_job`. `FitPanel` shows one primary `自动拟合` command in normal mode; batch selector and existing `开始拟合` command are visible only when `ui_state.expert_mode` is true. `start_automatic_fit()` runs `api.preflight_automatic_fit`, resets progress, and starts the job. Checkpoint events replace the document immediately so each finished point's curve becomes selectable during `REFINING`.

Cancellation keeps the latest checkpoint project. A terminal error never replaces it with the pre-operation snapshot.

- [ ] **Step 7: Add automatic result tables**

Create two un-nested table views:

- Point/layer table columns: point, status, statistics membership, material, thickness, roughness, SLD real, SLD absorption, effective electron density, density result/note.
- Uniformity table columns: group, layer, count, mean, min, max, population standard deviation, CV, relative range.

Use `api.summarize_automatic_results()` on project change. Dataset selection continues to drive existing curve/residual plots, so users can scan every point without duplicating plot canvases. Status text maps exactly: `PASSED=通过`, `REFINING=精修中`, `REVIEW=需复核`, `FAILED=失败`.

- [ ] **Step 8: Run GREEN and spawn integration**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_workers.py tests/architecture/test_public_api.py tests/gui/test_data_import.py tests/gui/test_fit_controller.py tests/gui/test_fit_progress.py tests/gui/test_results.py tests/integration/test_gui_automatic_workflow.py -q
```

Expected: first-use preset and ambiguous substrate are the only normal interactions; import starts fitting; partial checkpoints render; expert controls remain available.

- [ ] **Step 9: Commit Task 9**

```bash
git add src/xrr_fitter/services/workers.py src/xrr_fitter/services/fitting.py src/xrr_fitter/api.py src/xrr_fitter/gui/data/import_dialog.py src/xrr_fitter/gui/data/substrate_dialog.py src/xrr_fitter/gui/data/panel.py src/xrr_fitter/gui/fitting/controller.py src/xrr_fitter/gui/fitting/panel.py src/xrr_fitter/gui/results/automatic.py src/xrr_fitter/gui/results/panel.py src/xrr_fitter/gui/workspace.py tests/unit/services/test_workers.py tests/architecture/test_public_api.py tests/gui/test_data_import.py tests/gui/test_fit_controller.py tests/gui/test_fit_progress.py tests/gui/test_results.py tests/integration/test_gui_automatic_workflow.py && git commit -m "feat: make automatic fitting the default GUI flow"
```

### Task 10: 精确失效、合成恢复、spawn/GUI/性能验收与文档

**Files:**
- Modify: `src/xrr_fitter/services/projects.py:78-115,233-307`
- Modify: `src/xrr_fitter/services/datasets.py:208-244`
- Modify: `tests/unit/services/test_projects.py`
- Modify: `tests/unit/services/test_datasets.py`
- Modify: `tests/gui/test_source_recovery.py`
- Modify: `tests/integration/test_process_workers.py`
- Create: `tests/regression/test_automatic_recovery.py`
- Create: `tests/support/automatic_recovery.py`
- Create: `tools/benchmark_automatic_fit.py`
- Create: `tests/unit/tools/test_benchmark_automatic_fit.py`
- Modify: `tools/verify_registry.py`
- Modify: `docs/user-guide.md`
- Modify: `README.md`

**Interfaces:**
- Changes: a source hash change invalidates the changed dataset and only datasets sharing its non-null automatic `fit_group_id`; expert joint mode retains existing all-dataset invalidation.
- Adds: deterministic automatic recovery cases for direct SLD, shared material/local thickness, uniformity and isolated retry.
- Adds: non-CI Apple Silicon wall-clock benchmark plus CI-stable work-count assertions.

- [ ] **Step 1: Write RED invalidation tests**

```python
from dataclasses import replace

from tests.support.model_cases import dataset_project, final_fit_result, project
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)
from xrr_fitter.services.projects import clear_fit_results


def _fitted_automatic_dataset(dataset_id: str, group_id: str):
    return replace(
        dataset_project(dataset_id, result=final_fit_result()),
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            fit_group_id=group_id,
            role=AutomaticRole.JOINT,
            status=AutomaticStatus.PASSED,
            statistics_member=True,
        ),
    )


def test_automatic_result_clear_invalidates_only_its_fit_group() -> None:
    value = project(
        _fitted_automatic_dataset("a", "g1"),
        _fitted_automatic_dataset("b", "g1"),
        _fitted_automatic_dataset("c", "g2"),
        _fitted_automatic_dataset("d", "g2"),
    )
    changed = clear_fit_results(value, ("a",))
    by_id = {dataset.dataset_id: dataset for dataset in changed.datasets}
    assert by_id["a"].last_valid_result is None
    assert by_id["b"].last_valid_result is None
    assert by_id["c"].last_valid_result is not None
    assert by_id["d"].last_valid_result is not None
    assert by_id["a"].automation.status is AutomaticStatus.PENDING
    assert by_id["b"].automation.statistics_member is False


def test_single_automatic_result_clear_does_not_clear_unrelated_points() -> None:
    value = project(
        replace(
            _fitted_automatic_dataset("a", "single-a"),
            automation=replace(
                _fitted_automatic_dataset("a", "single-a").automation,
                role=AutomaticRole.SINGLE,
            ),
        ),
        replace(
            _fitted_automatic_dataset("b", "single-b"),
            automation=replace(
                _fitted_automatic_dataset("b", "single-b").automation,
                role=AutomaticRole.SINGLE,
            ),
        ),
    )
    changed = clear_fit_results(value, ("a",))
    assert changed.datasets[0].last_valid_result is None
    assert changed.datasets[1].last_valid_result is not None
```

Append a source-acceptance case to `tests/unit/services/test_datasets.py` using that file's existing `_write_curve()`, `_instrument()`, `preview_source_update()`, and `accept_source_update()` helpers:

```python
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)


def test_automatic_source_acceptance_clears_the_matching_fit_group(tmp_path: Path) -> None:
    value = new_project()
    sources = (
        _write_curve(tmp_path / "a.xy"),
        _write_curve(tmp_path / "b.xy", scale=2.0),
        _write_curve(tmp_path / "c.xy", scale=3.0),
    )
    for source in sources:
        value = add_dataset(value, source, _instrument())
    groups = ("g1", "g1", "g2")
    value = replace(
        value,
        datasets=tuple(
            replace(
                dataset,
                last_valid_result=final_fit_result(),
                automation=DatasetAutomation(
                    import_batch_id="batch-1",
                    fit_group_id=group_id,
                    role=AutomaticRole.JOINT,
                    status=AutomaticStatus.PASSED,
                    statistics_member=True,
                ),
            )
            for dataset, group_id in zip(value.datasets, groups, strict=True)
        ),
    )
    sources[0].write_bytes(sources[0].read_bytes() + b"# changed\n")

    changed = accept_source_update(value, preview_source_update(value, "a"))

    by_id = {dataset.dataset_id: dataset for dataset in changed.datasets}
    assert by_id["a"].last_valid_result is None
    assert by_id["b"].last_valid_result is None
    assert by_id["c"].last_valid_result is not None
```

- [ ] **Step 2: Run invalidation tests and record RED**

Run:

```bash
python -m pytest -o addopts= tests/unit/services/test_projects.py tests/unit/services/test_datasets.py tests/gui/test_source_recovery.py -q
```

Expected: current invalidation knows only `batch_mode`: an automatic project in independent mode clears just the changed point, while joint mode can clear every point. Neither branch follows `fit_group_id`, so the new group assertions fail.

- [ ] **Step 3: Implement fit-group-aware invalidation**

Centralize affected IDs:

```python
def _dependent_fit_ids(project: XrrProject, changed_ids: set[str]) -> set[str]:
    groups = {
        dataset.automation.fit_group_id
        for dataset in project.datasets
        if dataset.dataset_id in changed_ids and dataset.automation.fit_group_id is not None
    }
    automatic = {
        dataset.dataset_id
        for dataset in project.datasets
        if dataset.automation.fit_group_id in groups
    }
    if automatic:
        return changed_ids | automatic
    if project.batch_mode == "joint" and changed_ids:
        return {dataset.dataset_id for dataset in project.datasets}
    return changed_ids
```

Use it for source restore, source acceptance, fit-mask changes, structure changes and result clearing. Reset affected automatic rows while retaining import and group identity:

```python
def _reset_automatic_state(dataset: DatasetProject) -> DatasetAutomation:
    state = dataset.automation
    if state.role is AutomaticRole.MANUAL:
        return state
    return replace(
        state,
        status=AutomaticStatus.PENDING,
        statistics_member=False,
        reason=None,
    )


def _cleared(dataset: DatasetProject, *, clear_evidence: bool) -> DatasetProject:
    return replace(
        dataset,
        structure_evidence=None if clear_evidence else dataset.structure_evidence,
        scale_prior=ScalePriorState(enabled=False),
        last_valid_result=None,
        checkpoint=None,
        automation=_reset_automatic_state(dataset),
    )
```

Do not invalidate another automatic group in the same project. `services.projects` imports and uses `_dependent_fit_ids()` alongside `_cleared()`; `services.parameters.accept_source_update()` continues to route through the updated `_replace_invalidated()` rather than duplicating the rule.

- [ ] **Step 4: Add deterministic recovery cases without changing the 220-case corpus**

Create a separate automatic corpus with at least these four fixtures:

1. One unknown direct-SLD layer with known generating SLD and no mass density.
2. Four same-structure points sharing generating SLD and having controlled local thicknesses.
3. Four points with one deliberately corrupted outlier that is isolated and excluded from first-pass statistics.
4. A roughness-mismatch group that triggers release and recovers local roughness.

Assert the existing thresholds for applicable parameters, exact population statistics for controlled thicknesses, and no false `PASSED` on model-error/ambiguous cases. Do not append these cases to `build_corpus()` and do not change its required count of 220.

Add `tests/regression/test_automatic_recovery.py` to the `regression` verify mode.

- [ ] **Step 5: Add deterministic work-count and manual wall-clock benchmarks**

`tools/benchmark_automatic_fit.py` requires exactly one mode from `--single`, `--batch-size {2,3,4}`, and `--adaptive`; it also accepts `--repeat` and `--json`. The adaptive mode runs each deterministic upgrade fixture once per repeat. The tool generates fixed synthetic sources in a temporary directory, calls only public API functions, records total wall time, per-stage `total_nfev`, bootstrap count, profile count, status and recovery error, and deletes its temporary directory through `TemporaryDirectory`.

Unit tests assert argument validation and JSON schema. Regression tests assert clean cases have `bootstrap_count == 0`, `profile_count <= 4`, and deterministic nfev/profile counts across two runs. Wall-clock assertions are printed but not used as CI pass/fail conditions.

Record the reference command in the user guide:

```bash
python tools/benchmark_automatic_fit.py --single --repeat 3 --json
python tools/benchmark_automatic_fit.py --batch-size 4 --repeat 3 --json
python tools/benchmark_automatic_fit.py --adaptive --repeat 1 --json
```

Acceptance on the reference Apple Silicon machine is median `<=20 s` for single, median `<=45 s` for 2-4 points, and every adaptive case `<=60 s`. If the numerical gates pass but wall time misses, report the timing miss rather than weakening search/recovery thresholds.

- [ ] **Step 6: Run focused GREEN, real spawn, and GUI tests**

Run:

```bash
python -m pytest -o addopts= tests/regression/test_automatic_recovery.py tests/integration/test_process_workers.py tests/integration/test_gui_automatic_workflow.py tests/unit/tools/test_benchmark_automatic_fit.py -q
```

Expected: direct SLD and joint/local recovery pass, spawn emits partial checkpoints before the terminal result, and benchmark schema/work counts are deterministic.

- [ ] **Step 7: Update stable user documentation**

Document:

- filename layer order and single-layer example;
- default Si and the exact case that prompts for another substrate;
- automatic SiO2 policy;
- why unknown compounds show effective SLD/electron density but no g/cm3;
- singleton versus same-batch same-signature joint routing;
- status and statistics membership meanings;
- how to enter expert fitting/profile/MCMC/export;
- the three benchmark commands and 20/45/60 second reference targets.

Do not describe point coordinates, spatial maps or automatic export as implemented.

- [ ] **Step 8: Run all repository gates**

Run each command separately so a failing domain is attributable:

```bash
python tools/verify.py unit
python tools/verify.py integration
python tools/verify.py gui
python tools/verify.py spawn
python tools/verify.py regression
python tools/verify.py statistical
python tools/verify.py quality
python tools/check_radon.py
```

Expected: every command exits 0; statistical verification still reports exactly 220 existing cases. Clean temporary benchmark/source directories and any generated report directories not tracked by the repository before committing.

- [ ] **Step 9: Commit Task 10**

```bash
git add src/xrr_fitter/services/projects.py src/xrr_fitter/services/datasets.py tests/unit/services/test_projects.py tests/unit/services/test_datasets.py tests/gui/test_source_recovery.py tests/integration/test_process_workers.py tests/regression/test_automatic_recovery.py tests/support/automatic_recovery.py tools/benchmark_automatic_fit.py tests/unit/tools/test_benchmark_automatic_fit.py tools/verify_registry.py docs/user-guide.md README.md && git commit -m "test: verify automatic fitting end to end"
```

## 最终验收记录

实施者在最后一次提交后把以下证据追加到 PR/交付说明，不写入项目持久化格式：

```text
unit: PASS
integration: PASS
gui: PASS
spawn: PASS
regression: PASS
statistical: PASS (220 cases)
quality: PASS
radon: PASS
single median seconds: numeric `median_seconds` from the `--single --repeat 3 --json` result
four-point median seconds: numeric `median_seconds` from the `--batch-size 4 --repeat 3 --json` result
adaptive maximum seconds: maximum numeric `elapsed_seconds` from the `--adaptive --repeat 1 --json` result
temporary artifacts removed: yes
```

若参考机器墙钟未达到目标但数值/质量门禁通过，保留最佳有效结果和测得数字，继续定位实际耗时阶段；不得通过关闭快速质量门槛、减少 220 例语料或把 `需复核` 改成 `通过` 来制造达标。
