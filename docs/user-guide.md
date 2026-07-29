# XRR Fitter 用户指南

本指南覆盖公共 Python API 与 PySide6 图形工作台。所有集成代码只应通过
`xrr_fitter.api` 调用项目、拟合、批处理和导出能力；`xrr_fitter` 的其他模块是
内部实现，不是受支持的应用边界。

## 1. 坐标、单位和输入

- 输入列 `two_theta_deg` 表示仪器记录的 `2θ`，单位为 degree；物理计算使用
  `θ = two_theta_deg / 2 + import_angle_offset_deg`。
- 长度统一使用 Å，`qz` 使用 `Å⁻¹`，复 X-ray SLD 使用 `Å⁻²`。
- 只有角度从 `2θ` 转成入射角时除以二。原始强度、归一化强度和模型强度
  **never divided by two**。
- 每次导入必须给出显式 `BeamSpec`。单色束使用 `wavelength_a`；混合 Kα
  保存 `wavelength_1_a`、`wavelength_2_a` 和 `intensity_ratio_21`，两条波长
  的材料光学常数和层栈分别展开。

最小导入示例：

```python
from pathlib import Path
import xrr_fitter.api as api

beam = api.BeamSpec(kind="monochromatic", wavelength_a=1.5406)
data = api.import_data(
    Path("sample.xy"),
    beam,
    import_angle_offset_deg=0.0,
    column_mapping=None,
)
```

`DataColumnMapping` 固定 `two_theta`、`intensity`、可选误差/分辨率列及其语义。
导入会保留原始行、解析状态、重复角合并来源、原始字节 SHA-256、归一化、
`r_floor` 和初始拟合掩码；不会用解析后的文本重新计算“等价”哈希。

## 2. 项目格式与版本

当前持久格式为：

```text
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "xrr-fit-v1"
```

加载器会拒绝不支持的 schema 或 algorithm version，不静默迁移成当前算法。
项目根保存 `fit_config`、`input_angle_kind="two_theta_deg"`、`batch_mode`、
ordered datasets、sharing rules 和 `ProjectUiState`。`base_directory` 仅在
`load_project()` 后由项目文件所在目录赋值，是 runtime-only 字段，绝不写入
JSON。

每个 `DatasetProject` 至少保存：

- 稳定且唯一的 `dataset_id`；
- 相对或绝对 `source_path` 与原始字节 `source_sha256`；
- 必需的 `BeamSpec`、`DataColumnMapping`、导入角度偏移；
- `fit_mask` 和 `fit_range_two_theta_deg`；
- `StructureSpec` 与完整 `InstrumentSpec`；
- 参数设置、结构证据、尺度先验、氧化层决定；
- 最近有效结果、检查点和项目级 UI candidate selection。

完整 `InstrumentSpec` 包含 `instrument_id`、`footprint_mode`、
`footprint_spill_angle_deg`、可选样品/光束几何尺寸、`background_kind` 和
`resolution_domain`。项目加载时不为缺少的 beam 或 instrument 猜默认值。

### 创建、保存和加载

```python
from pathlib import Path
import xrr_fitter.api as api

source = Path("sample.xy").resolve()
beam = api.BeamSpec(kind="monochromatic", wavelength_a=1.5406)
instrument = api.InstrumentSpec(footprint_mode="none")

project = api.new_project()
project = api.add_dataset(
    project,
    source,
    instrument,
    display_name="sample",
    beam=beam,
)

air = api.MaterialSpec("Air", None, None, 0.0j)
si = api.MaterialSpec("Si", "Si", 2.329)
film = api.MaterialSpec("SiO2", "SiO2", 2.20)
structure = api.StructureSpec(
    air,
    (api.LayerSpec("film", film, 173.0, roughness_a=3.0),),
    si,
    backing_roughness_a=4.0,
)
dataset_id = project.datasets[0].dataset_id
project = api.set_structure(project, dataset_id, structure)

target = source.parent / "sample.xrrproj.json"
api.save_project(project, target)
loaded = api.load_project(target)
```

保存使用同目录临时文件、写满、`fsync` 和原子替换。写入失败会传播原异常并
清理本次临时文件，不把旧项目伪装成新版本。

### 标准 JSON 与 `null`

项目和 `fit_result.json` 均使用 standards-compliant JSON，编码时
`allow_nan=False`：

- 需要保持数组对齐的非有限数值槽写为 JSON `null`，加载后恢复为只读数组中
  的 `NaN`；
- 只有 `valid=false` 的无效 candidate 非有限 objective，以及没有可选
  candidate 的 stage `best_objective`，允许使用 documented `null` sentinel；
- valid candidate objective、普通必需标量和任意未列出的对象位置不得为
  `null`；
- 禁止写入非标准 `NaN` 或 `Infinity` 文本。

## 3. 源文件校验与失效

公开工作流中的：

```text
api.inspect_sources(project) -> ProjectValidation
```

只观察当前 source/hash，不修改项目。schema/object-graph 校验由项目构造和加载
内部执行，应用代码不应导入其内部实现。source status 精确为：

- `ok`：文件可读且原始字节 SHA-256 与声明一致；
- `missing`：路径不存在；
- `unreadable`：操作系统拒绝读取或发生 I/O 错误；
- `hash_mismatch`：文件可读，但原始字节与声明不同。

相对 source 只在 I/O 时相对 runtime `base_directory` 解析。校验绝不重写
`source_path` 或 expected hash。拟合和导出会在读取前校验，并在解析后再次比较
真实字节哈希，以关闭 validate/read 之间的 TOCTOU 窗口。

`load_project()` 会在恢复 workspace 时重新校验 source，并按 `independent` 或
`joint` 影响范围清除不再可信的派生状态。`save_project()` 在写入前也会重新校验，
不会把已知 stale 的结果持久化。接受当前路径的新字节或重新链接到另一文件必须走
两阶段事务：

```python
validation = api.inspect_sources(project)
if not validation.valid:
    preview = api.preview_source_update(project, dataset_id, Path("replacement.xy"))
    # 在应用中显示 old/new path 与 expected/observed SHA-256，再取得用户确认。
    project = api.accept_source_update(project, preview)
```

- `independent` 模式只清理受影响 dataset 的 `structure_evidence`、已解析 scale
  prior、`last_valid_result`、checkpoint 和 candidate selection。
- `joint` 模式中任一 source 变更会清理所有 datasets 的共享派生状态。
- structure、mask、instrument、parameter 和 sharing 修改必须调用对应的
  `api.set_*` operation；这些 operation 在同一 immutable transaction 中处理失效。
- `accept_source_update()` 会再次核对 preview 对应的路径和字节。preview 之后发生
  变化会硬失败，不会接受未展示给用户的 source identity。

## 4. 结构证据、尺度先验与氧化层

`api.analyze_structure(project, dataset_id)` 返回观察性的
`structure_evidence`，不会自动附加到项目。它记录数据可支持的结构复杂度、峰位
等 provenance，供调用者决定是否持久化。

`ScalePriorState` 只有三种合法状态：

1. unresolved：`enabled=False, s_hat=None, tau_s_decades=None, reason=None`；
2. inactive：`enabled=False, s_hat=None, tau_s_decades>0, reason=非空`；
3. active：`enabled=True, s_hat>0, tau_s_decades>0, reason=None`。

inactive 表示已明确关闭并保留原因；unresolved 表示尚未由真实 compiled problem
解析。source 失效会回到 unresolved，不能保留旧平台估计。

`api.suggest_oxide_layers(structure)` 返回 immutable `OxideSuggestion`
tuple，不修改输入结构。接受或拒绝另存为 `OxideDecision`，字段为
`base_material`、`oxide_material`、`location`、`accepted` 和
`oxide_table_version`。决定记录 provenance；真正改变 layer structure 仍是单独、
显式的项目编辑事务。

## 5. 参数描述与校验

只从公共 API 获取稳定参数 namespace，并通过 service operation 提交设置：

```python
import xrr_fitter.api as api

definitions = api.describe_parameters(project, dataset_id)
settings = tuple(
    api.ParameterSetting(
        item.name,
        item.initial,
        item.lower,
        item.upper,
        locked=item.locked,
    )
    for item in definitions
)
settings = api.validate_parameter_settings(definitions, settings)
project = api.set_parameter_settings(project, dataset_id, settings)
```

名称、单位、bounds、transform、integer/expert flags 和 sharing identity 都是
接口的一部分。未知名称、倒置 bounds 或不兼容设置必须抛错，不能忽略。

## 6. 可复现种子树

当前 `SERVICE_SEED_TREE_VERSION = 1`。服务根为：

```text
SeedSequence(project.master_seed,
             spawn_key=(SERVICE_SEED_TREE_VERSION,)).spawn(3)
```

三个固定分支顺序是 `independent -> joint -> MCMC`：

- independent 先按 sorted dataset ID 派生 dataset root；
- joint 使用单一 joint root；
- MCMC 先按 sorted dataset ID，再按 sorted candidate ID 派生 child seed；
- dataset/candidate 在项目或 UI 中重排不会改变对应稳定 child seed。

每个拟合分支把 child seed 作为运行时 `FitConfig.master_seed`，优化器内部 Stage
B/E 等 generation seeds 再由 `SeedSequence(runtime_master_seed).spawn(count)`
派生。项目持久 master seed 不被替换。

当前 bootstrap/uncertainty 是明确例外：它使用
`SeedSequence([config.master_seed, 0x554E434552544149]).generate_state(...)`
生成独立确定性 seed，不属于上述 MCMC branch，也不能声称 bootstrap 本身通过
`.spawn()` 或已写入 `FitResult.child_seeds`。

还需区分：

- project master seed：项目/拟合服务的持久根；
- optimizer generation seed：某次 Stage B/E 等搜索的 child；
- MCMC seed：选定 dataset/candidate 的 expert sampling branch；
- example generation seed：`EXAMPLE_SEED_TREE_VERSION = 1` 的独立 fixture
  domain，不等于 example project 的 master seed。

## 7. 独立与联合拟合

### 独立拟合

`batch_mode="independent"` 是默认值。每个 dataset 独立编译参数、搜索、生成
confidence/uncertainty；一个 dataset 的失败不会伪装成成功，也不会覆盖另一个
dataset 的有效结果。

### 联合拟合

`batch_mode="joint"` 使用显式 `SharingRule`。允许共享的业务 family 包括兼容
的 material density/SLD，以及明确允许的 instrument resolution/footprint。
共享 resolution 或 footprint 时，所有成员必须保存相同且非空的
`instrument_id`，且 resolution domain、active footprint mode 等定义兼容。

下列参数保持 dataset-local，禁止共享：

- thickness、period/repeats 和 roughness；
- angle offset 与 scale；
- constant/linear/power-law background 的全部参数。

每条规则需要不同 dataset 的唯一成员；同一 local parameter 不能进入两条规则。
调用 `api.validate_sharing_rules(project, rules)` 校验声明，再通过
`api.set_sharing_rules(project, rules)` 持久化并使受影响结果失效。真实 joint compile
仍由 `api.preflight_fit()` 在启动 solver 前完成；joint 不合法或 resume 失败时不得
静默 fallback 到 independent。

### 单 dataset 与 joint objective

对 dataset `d`：

```text
Delta_di = log10(Rmodel_di + Rfloor_d) - log10(Robs_di + Rfloor_d)
L_C(Delta) = 2 C^2 (sqrt(1 + (Delta/C)^2) - 1)
z_d = (log10(S_d) - log10(S_hat_d)) / tau_S,d       # only when active
J_d = (sum_i w_di^2 L_C(Delta_di) + z_d^2) / N_d
```

scale prior inactive 时省略 `z_d^2`。region weight 位于 robust argument 外，
因此所有数据点共享一个物理 saturation threshold `C`。

对 `K` 个 datasets：

```text
J_joint = (1/K) sum_d J_d
alpha_d = N_total / (K N_d)
```

joint local solver 保持 residual row 为未缩放的 `Delta_di`，外部 row weight 为
`sqrt(alpha_d) * w_di`。不能先缩放 Delta、把 `alpha_d` 放进 soft-L1 argument，
或继续使用一个 global `f_scale`；这些形式会改变 saturation threshold，不等价。

## 8. 进度、取消、检查点和恢复

同步拟合接收显式 progress 与 checkpoint callback：

```python
progress = []
checkpoints = []
fit_output = api.fit_project(
    project,
    progress_callback=progress.append,
    checkpoint_callback=checkpoints.append,
)
```

`FitProgress` 记录 `dataset_id`、stage、completed/total、当前 best objective 和
message。checkpoint callback 收到的是包含最新 checkpoint 的 immutable
whole-project snapshot；服务先保存 snapshot 再调用用户 callback。因此 callback
失败不会抹掉最新可恢复证据，而会产生显式 `checkpoint_failed` 不可信结果。

需要协作取消或独立进程隔离时，使用 `api.start_fit_job(project,
checkpoint_path=None)`。调用方轮询 `OperationJob.poll()` 的有序 `OperationEvent`，
通过 `cancel()` 请求协作取消；超时后才可显式 `force_stop()`，并最终调用 `close()`
回收进程和队列。GUI 使用的也是这条公共边界，不在界面线程执行 optimizer。

取消或拟合失败：

- 不覆盖 previous `last_valid_result`；
- 保留已成功发布的最新 checkpoint；
- 返回显式 cancelled/untrusted/warnings，而不是 clean-fit 假成功；
- independent 会在完成当前边界后停止准备后续 dataset；
- joint checkpoint 必须在所有 datasets 间保持同一 stage、candidate graph、seed
  ledger 和 layout identity。

恢复会精确核对 data hash、structure、instrument、parameter settings、config、
stage graph、candidate order、seed ledger 和 joint layout fingerprints。任何不匹配
都会显式拒绝，并保留 previous result/checkpoint；不会在背后重新开始一次“干净”
拟合。

`run_mcmc()` 只接受当前 source、parameter definitions 与有效 candidate 完全一致
的项目。同步调用成功时只为选定结果附加新的 MCMC report；需要取消时使用
`start_mcmc_job()` 并通过同一 `OperationJob` 协议处理事件。取消或失败不会修改输入
项目和 source bytes。

## 9. Confidence 与不确定度

持久 confidence labels 为：

- `不可信`：无有效解、候选非有限/物理违规、最佳簇支持不足、bootstrap 失败率
  超过 20%，或运行/检查点失败；
- `多解`：存在目标值近等价且物理上分离的 candidate clusters，profile path
  无法合并，或关键 thickness/period profile 双侧开放；
- `可用但相关`：没有触发多解，但存在 boundary hit、强相关、开放 profile、
  systematic residual/ACF、Nevot-Croce 适用性告警，或最佳簇支持较弱；
- `可信`：给定当前 structure model 与数据范围，多个标准 seeds 汇聚且关键
  profiles 闭合，没有上述降级证据。

`可信` 不表示材料顺序已被实验唯一证明。报告应联合解释：

- profile likelihood 的 values/objectives 和两侧 closed/open；
- bootstrap intervals 与 failure rate；
- correlation matrix、strong correlation 和 boundary hits；
- unweighted log residual、weighted residual 与 residual ACF；
- structured physics diagnostics；
- 可选 expert MCMC 的 acceptance fraction、split-Rhat、ESS 和 boundary hits。

MCMC 输出固定标为“目标函数伪后验”，不是有已知计数似然时的严格贝叶斯覆盖；
一键 `fit_project()` 不自动运行 MCMC。

## 10. Example projects

仓库提供两套未拟合、可整体搬迁的输入：

```text
examples/single-layer.xy
examples/single-layer.xrrproj.json
examples/mo-si-periodic.xy
examples/mo-si-periodic.xrrproj.json
```

- single layer：1200 点、`2θ=[0.08, 8.0]`、173 Å SiO2 film；
- Mo/Si：1800 点、`2θ=[0.08, 12.0]`、20 repeats；
- project 只保存 `.xy` basename，没有 candidate、fit result、checkpoint 或预埋
  recovery answer。

四个文件是受版本控制和发行身份校验约束的 canonical input。仓库维护者可通过
`xrr_fitter.io.examples.write_examples()` 在受控目标目录重建它们；该内部 writer
不是应用集成 API。使用任一 example 的公共 API workflow：

```python
from pathlib import Path
import xrr_fitter.api as api

path = Path("examples/single-layer.xrrproj.json")
project = api.load_project(path)
validation = api.inspect_sources(project)
if not validation.valid:
    raise RuntimeError(
        tuple(item.message for item in validation.datasets if item.status.value != "ok")
    )

for dataset in project.datasets:
    definitions = api.describe_parameters(project, dataset.dataset_id)
    settings = tuple(
        api.ParameterSetting(
            item.name,
            item.initial,
            item.lower,
            item.upper,
            locked=item.locked,
        )
        for item in definitions
    )
    settings = api.validate_parameter_settings(definitions, settings)
    project = api.set_parameter_settings(project, dataset.dataset_id, settings)

readiness = api.preflight_fit(project)
if not readiness.ready:
    raise RuntimeError(readiness.message)

fit_output = api.fit_project(project)
if fit_output.cancelled:
    raise RuntimeError(fit_output.warnings)

updated = fit_output.updated_project
for dataset in fit_output.updated_project.datasets:
    result = dataset.last_valid_result
    candidate = None if result is None else result.best_candidate
    if candidate is None:
        raise RuntimeError(f"no valid result: {dataset.dataset_id}")
    updated = api.select_candidate(
        updated,
        dataset.dataset_id,
        candidate.candidate_id,
    )

api.save_project(updated, path.with_name("single-layer-fitted.xrrproj.json"))
manifest = api.export_result(updated, Path("exports"))
print(manifest.run_directory)
```

把四个 example 文件一起复制到另一目录后，JSON 无需编辑；`load_project()` 会把
新目录设为 runtime `base_directory`，source status 仍应为 `ok`。

## 11. 导出文件和语义

`api.export_result(result, output_dir)` 接受 fitted `XrrProject` 或
`ProjectFitResult`；后者先归一为 `updated_project`，持久
`dataset.last_valid_result` 是 artifact source of truth。导出先完成 schema、
result、source/TOCTOU 和 selected candidate preflight，才分配输出目录。

每个 dataset 固定输出：

```text
fit_result.xlsx
fit_result.json
fit_overview.png
sld_profile.png
residuals.png
run_log.txt
```

`fit_result.xlsx` sheets 固定为：

```text
Parameters, Candidates, RawData, ModelResiduals,
Correlation, Profiles, RunInfo
```

Root 总有 `compatibility_summary.xlsx`。多 dataset 另有
`batch_summary.xlsx` 和 `parameter_trends.png`；单 dataset 不伪造 batch 文件。

兼容汇总的语义：

- `two_theta_deg`、`intensity_raw`、`intensity_normalized` 逐行保持 source data；
- `qz_a_inv` 和 `model_normalized` 使用 selected candidate 的 fitted-offset
  reporting grid；
- fit mask 显式保存，raw 与 normalized intensity 不混淆；
- `.thickness_a`、`.roughness_a`、`.microslab_max_a` 参数保留 Å，同时仅以
  `value_nm = value / 10` 增加 nm 兼容列；
- intensity/model values 永远不除以二。

`fit_result.json` 包含完整 persisted project、source/beam/instrument、raw data
provenance、model/residual arrays、所有 candidates、convergence、scale prior、
structure evidence、oxide decisions、confidence/warnings、版本和 seed identity。
`run_log.txt` 记录 stages、seeds、warnings、停止原因和 structured diagnostics。

导出先在 output root 下的 private `.partial-*` sibling 完整写文件，验证每个
manifest path 位于 run tree 且非空，`fsync` 文件/目录，再以同 filesystem 的
exclusive rename 一次发布 `<UTC timestamp>-<8hex>` 目录。已有 run 不覆盖；任意
异常只删除本次 owned partial tree 并原样传播。

## 12. 公共 API

唯一受支持的导入方式是：

```python
import xrr_fitter.api as api
```

R23 的公共 operation 按职责分组如下；完整类型和值对象集合由 `api.__all__` 固定，
包根 `xrr_fitter` 不做 re-export：

```text
new_project()
load_project(path)
save_project(project, path)
inspect_sources(project)
select_active_dataset(project, dataset_id)
select_candidate(project, dataset_id, candidate_id)
set_batch_mode(project, mode)
set_expert_mode(project, enabled)
set_workspace_state(project, state)
clear_fit_results(project, dataset_ids)

import_data(path, beam, import_angle_offset_deg, column_mapping)
add_dataset(project, source_path, instrument, display_name, column_mapping,
            import_angle_offset_deg, beam)
remove_dataset(project, dataset_id)
set_fit_mask(project, dataset_id, mask)
set_instrument(project, dataset_id, instrument)
preview_source_update(project, dataset_id, new_path)
accept_source_update(project, preview)

set_structure(project, dataset_id, structure)
describe_parameters(project, dataset_id)
analyze_structure(project, dataset_id)
validate_parameter_settings(definitions, settings)
set_parameter_settings(project, dataset_id, settings)
validate_structure(structure, beam)
validate_sharing_rules(project, rules)
set_sharing_rules(project, rules)
suggest_oxide_layers(structure)
accept_oxide_suggestion(project, dataset_id, suggestion)
record_oxide_decision(project, dataset_id, decision)

preflight_fit(project)
fit_project(project, progress_callback, checkpoint_callback)
run_mcmc(project, dataset_id, candidate_id, config, progress_callback)
start_fit_job(project, checkpoint_path)
start_mcmc_job(project, dataset_id, candidate_id, config)
export_result(result, output_dir)
```

`OperationJob` 提供只读 `pid`/`is_running` 以及 `poll()`、`cancel()`、
`force_stop()`、`close()`。调用者应显式处理异常、`ProjectValidation`、`FitReadiness`、
cancelled/untrusted result、`OperationError` 和 export 错误。服务不提供吞异常、
静默 fallback 或“部分成功即成功”的替代合同。

## 13. PySide6 图形工作台

启动桌面程序：

```bash
python -m xrr_fitter
# 安装 wheel 后也可使用：
xrr-fitter
```

主窗口最低尺寸为 `1280×760`，固定为数据/结构、图形诊断、参数/结果三列。
项目命令、导入、结构、图形交互、参数、拟合、候选、保存和导出均可通过键盘
聚焦；置信度同时使用文字和视觉标记，不以颜色作为唯一信息。

### 13.1 普通一键流程

1. 选择“导入文件”或“导入文件夹”。导入确认框不会预选光路；必须显式选择
   “单色”或“混合 Kα”，并确认可见的 `InstrumentSpec`。文件夹模式可显式启用
   递归。额外列只有在“列映射”中声明后才解释。
2. 新数据集没有隐式结构。选择“初始化结构”建立 `Air / 空层栈 / Si 基底`，
   再用“添加普通层”或“添加周期块”声明实际模型。厚度、粗糙度和周期在界面中
   使用 nm，项目和计算仍保存 Å，换算恒为 `1 nm = 10 Å`。
3. 若出现“添加自然氧化层”，必须明确接受或选择“忽略建议”。接受会添加普通
   层；拒绝会把版本化 `OxideDecision` 写入当前 dataset。拟合不会自动添加氧化层。
4. 数据树可选择 dataset；图形工具可选择拟合范围或排除/恢复单点。操作只更新
   项目 mask，原始强度、源文件和 expected SHA-256 不会被改写。
5. 三种足迹模式精确对应 `geometry|fit|none`。`geometry` 使用样品长度和束宽换算，
   `fit` 释放 `θ_fp`，`none` 固定无足迹；模式和数值都属于当前 dataset。
6. 在“拟合模式”中明确选择“独立拟合”或“联合拟合”，再点击“一键拟合”。
   启动前工作台重新计算每个已声明结构的
   `StructureEvidence`；复杂度警告不阻断拟合。进度显示 dataset、stage、完成数和
   best objective，GUI 线程不执行优化。
7. 点击“取消拟合”只请求协作取消。最近有效结果继续显示，已发布 checkpoint
   保留；关闭窗口会等待协作停止，超时后必须再次确认才会强制终止进程。
8. 在结果列表切换候选会同步模型、残差、SLD、参数和证据。“早期淘汰”候选仍
   可检查，但只有显式确认后才持久化为选择；推荐和自动 confidence 不被改写。
9. “保存”写入项目、active dataset、候选 ID、expert mode、splitter 尺寸和图形
   tab。再次“打开”会恢复这些状态，并从声明 source 重新读取数据。
10. “导出”每次创建新的 run 目录，包含每个 dataset 子目录和根
    `compatibility_summary.xlsx`。成功摘要列出所有实际文件；失败保留以前的 run。

### 13.2 拟合模式、readiness 与 checkpoint

- `独立拟合` 是默认模式；每个 dataset 独立产生结果和 confidence。
- `联合拟合` 至少需要两个 source-valid、已声明有效结构的 datasets，并要求所有
  sharing rules 可由真实 joint compile 接受。模式不合法时界面保留用户选择并显示
  第一个阻断 dataset/字段及恢复动作，不会静默回退到独立拟合。
- 切换模式会清除所有 datasets 的 scale prior、checkpoint、结果和 candidate
  selection，但保留 source、beam/instrument、结构及其 provenance。修改 mask、
  结构、参数、sharing 或 instrument 时也按实际影响范围失效旧结果。
- “一键拟合”只有在所有计划 datasets 的 source 为 `ok`、数据可拟合、结构已声明、
  参数/sharing 有效且当前没有 worker 时才启用。相邻 readiness 文本给出第一个
  未满足条件，不以禁用按钮代替错误原因。
- worker 发布 checkpoint 时，工作台先接收包含 checkpoint 的 immutable project
  snapshot，再刷新内存状态；若调用方给出 checkpoint path，则子进程先通过
  `save_project()` 原子保存，再向 GUI 发布。普通按钮不会暗中创建未声明的外部文件。
- 保存过的项目若含兼容 checkpoint，再次点击拟合会由 service 按 stage、candidate
  graph、seed ledger、data hash、structure、instrument、parameters 和 config 校验后
  恢复。任一身份不匹配会明确拒绝该 checkpoint，绝不伪装为“从头恢复成功”。

### 13.3 Source hash 警告

打开项目时会重新计算原始字节 SHA-256。`missing`、`unreadable` 或
`hash_mismatch` 会持续显示 dataset、expected hash 和 actual hash（若可得），
清除 stale result/checkpoint，并阻止拟合和导出。工作台不会静默接受新字节。

- “重新加载数据源”表示明确接受声明路径当前字节，并更新 expected hash；
- “重新链接数据源”表示选择另一文件，再明确接受其路径和 hash；
- 两种操作都会清除依赖旧数据的结构证据、scale-prior 状态、结果和 checkpoint。
- 确认框显示 old/new path 与 SHA-256；确认后的 hash 与实际提交字节绑定。preview
  或 commit 失败会保留完整旧 workspace，并显示异常类型、原始消息和恢复动作。
- 新 source 不再兼容的 parameter settings 会被过滤，并在成功后列出具体名称，
  不会静默丢弃。

### 13.4 Expert 模式与候选状态

Expert mode 显示完整 bounds/lock/sharing、背景模型
`constant|linear|powerlaw`、分辨率域 `q|theta`、scale weak prior 和 MCMC。
普通流程不要求打开 expert mode。修改参数、仪器模型、共享或项目级 scale prior
会按影响范围清除旧结果；不会以 fallback 继续使用不兼容状态。

标准模式隐藏 expert-only optimizer rows、背景/分辨率、MCMC 和 SLD 深度剖面 tab，
但不删除其持久状态；重新打开 Expert mode 会恢复相同控件、tab canvas 和可用选择。

- `invalid` candidate 表示物理/数值校验失败，不可持久化、导出或运行 MCMC；仍保留
  在列表中作为失败证据。
- `archived/早期淘汰` candidate 曾参与搜索但已被阶段淘汰，只能检查；选择它作为
  持久 candidate 前必须额外确认，且它不能替代自动推荐或 confidence。
- 普通 valid candidate 可被选择用于图形、参数、保存和导出；导出始终核对持久的
  selected candidate ID，而不是假定 `best_index`。

MCMC 只对当前有效候选运行，使用与自动拟合同一个 spawn controller。完成后只
附加 candidate-owned report，自动 confidence 保持不变；取消则恢复运行前项目。

### 13.5 项目命令与快捷键

- “新建”或“打开”遇到 dirty project 时先要求确认；拒绝后当前 workspace 不变。
- “另存为”会以新项目目录重新计算相对 source 声明，同时证明解析后的 source
  identity 未变化。保存、rebasing、UI 投影或 I/O 任一步失败都会恢复旧 project、
  result projection、validation、路径和 dirty 状态。
- `Open`：平台 `StandardKey.Open`；`Save`：`StandardKey.Save`；`Save As`：
  `StandardKey.SaveAs`。
- 导出：`Ctrl+Shift+E`；一键拟合：`Ctrl+Return`；取消：`Escape`。
- DataPanel 获得焦点时，导入文件为 `Ctrl+I`，导入文件夹为 `Ctrl+Shift+I`；这两个
  shortcut 只作用于 DataPanel 及其子控件，不抢占其他文本编辑器。
- 拟合运行时 `Escape` 只路由到 cooperative cancel；空闲时 plot 内的 `Escape`
  取消未完成的 range/mask interaction，不存在双重触发。

### 13.6 故障定位

- “数据不足”：检查 validation mask、拟合范围和排除点，不能强制启用校验失败点。
- “源文件已变化”：比较界面显示的 expected/actual SHA-256，再决定 reload/relink。
- “厚度参数预期不可辨识”：减少结构自由度或扩展有效 q 范围；警告不是成功证据。
- “联合拟合未就绪”：至少准备两个 source-valid datasets，逐项检查 structure、
  `instrument_id`、resolution/footprint domain 和 sharing member identity。
- “参数设置不兼容”：根据 readiness 中的 dataset/parameter 名称修正 bounds、lock、
  integer value 或 sharing；不要通过切换模式规避校验。
- stage 长时间不前进：记录界面中的 dataset、stage、completed/total、best objective
  和 checkpoint 状态；先用“取消拟合”请求协作停止。关闭窗口的 5 秒 deadline 到期
  后，只有明确确认才会进入 TERM -> KILL -> asynchronous reap。
- “checkpoint 不兼容”：核对 source hash、结构、instrument、parameter settings、
  fit config、candidate order 和 seed identity；保留旧结果，重新拟合生成新 checkpoint。
- worker technical failure：保留完整异常类型/消息和最近有效结果；dead-without-stopped
  错误同时报告 exit code/signal。不要把失败状态当作 clean completion。
- `多解`/`不可信`：检查候选簇、profile openness、boundary hits、相关性、残差和
  structured diagnostics，不只看曲线重合。
- 导出失败：错误消息包含失败路径和原始异常；修复目录权限或源 hash 后重试，
  不要删除以前已经发布的 run。

### 13.7 界面截图

以下截图使用仓库 canonical `single-layer` synthetic example 与固定 seed
`20260726` 的快速拟合结果，不包含用户 source。平台为 macOS/PySide6 Fusion；
覆盖 minimum/normal window、light/dark palette 与 standard/expert 状态：

- [Light 1280×760](images/gui-light-1280x760.png)
- [Light 1600×900 Expert](images/gui-light-1600x900-expert.png)
- [Dark 1280×760](images/gui-dark-1280x760.png)
- [Dark 1600×900 Expert](images/gui-dark-1600x900-expert.png)
