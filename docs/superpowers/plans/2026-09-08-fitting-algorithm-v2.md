# Fitting Algorithm V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实施已批准的拟合 V2 设计，闭环 C1–C6 与 O1–O5，并给出可复现的正确性、统计及性能证据。

**Architecture:** 保留物理内核。`model` 定义不可变证据，`evaluation` 统一数值，`fit` 搜索，`analysis` 推断，`services.fitting` 组合；共享接口串行锁定，不能建立 `fit <-> analysis` 导入。

**Tech Stack:** Python 3.12、NumPy、SciPy、PySide6、pytest，以及项目当前锁定依赖。

## 当前后续状态（2026-09-14）：RMS v4 诊断验收通过，joint 区间校准仍打开

当前身份为 schema4 / `xrr-fit-v2-poisson-4` / objective 字符串2 / `poisson-refit-null-v4`，数值 refit policy 保持 v3。`2026-09-13-poisson-rms-calibration.md` 的实现、同快照七 MODE **4420 passed**、Radon **602 files/0 issues/PASS**、五例B999，以及固定 **4400新holdout+200开发重放**均已完成。
完整独立 audit003 PASS：4600病例、6600观测流、0逐例证据缺口、最终hash复核通过；八个预登记 diagnostics 端点全部 PASS。原低计数200种子重放 **188/200=94%**，新single **1908/2000=95.4%**；joint bootstrap **1868/2000=93.4%** 仍欠覆盖，不能写成一般95%校准。原81%与旧4/8 FAIL保持历史记录。
当前唯一待关闭的覆盖条件与 joint bootstrap 额外 provenance 限制见 `2026-09-12-poisson-diagnostic-calibration-progress.md`、`docs/acceptance/fitting-algorithm-v2.md`；不是重做原 Task1–8 或重复科学 shard。冻结源未变，收尾只同步文档；保留外置证据、分支和未提交改动，不提交/合并/发布。下方版本、数字和旧执行步骤属于各自历史合同，不作为当前待执行命令。

## 历史后续状态（2026-09-13）：两阶段 v3 工程子计划完成

原 Task 1–8 的历史完成状态不替代后续 Poisson 科学验收。最新 `2026-09-13-poisson-two-stage-refit.md` 的 T1–T4 已闭环：固定 L-BFGS-B → 完整计费交接 → TRF、不可变工作证据和严格 v3 身份；不增加预算、不放宽门槛、不处理旧格式。
冻结 003 的七门禁 **4340 passed**、完整 Radon **597 文件 / 0 issues / PASS**；四组开发 B999 均 **999/999 null**，原 119/148 不再发生该次重放的 null-refit failure，两个注入目标 adjusted p=.001。独立审查发现的 R6 已修复并关闭，最终收据/数值等价审查 Approved。
当前身份 `xrr-fit-v2-poisson-3` / `poisson-refit-null-v3`、schema4、objective 字符串2；完整证据与保留历史见 `2026-09-12-poisson-diagnostic-calibration-progress.md` 及 `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001/two-stage-engineering-final.json`。正式验证后仅同步状态文档，保留分支、未提交改动和外置复现原件，不提交/发布或删除用户文件。
ACF 尺度竞争及新种子独立科学协议仍未完成；旧 81% 与旧 4400 + 200 实验的 4/8 FAIL 不改写。下列原任务的身份和验收数字为各历史快照，不作为 v3 总体科学 PASS。

## 原 Task 1–8 执行状态（2026-09-12 历史核对）

Task 1–8 的实现和本计划约定验收已完成。以下勾选依据进度账中的 RED/GREEN、
历史提交、任务审查和 Task 8 最终封存证据同步，不代表本次重新执行或补造历史验证。
Task 1–5 已有历史提交；Task 6–8 按用户要求保留未提交改动，不执行原模板中的提交动作。

| Task | 完成证据入口 | 提交状态 |
| --- | --- | --- |
| 1 | 进度账 Task 1；历史提交 `0ce68cb`、`08aab7f` | 已有历史提交 |
| 2 | 进度账 Task 2；历史提交 `533f382`、`29d87f5` | 已有历史提交 |
| 3 | 进度账 Task 3；历史提交 `25c90fc` | 已有历史提交 |
| 4 | 进度账 Task 4；历史提交 `3a28f53`、`691f609` | 已有历史提交 |
| 5 | 进度账 Task 5；历史提交 `687f7cf` | 已有历史提交 |
| 6 | 进度账 Task 6 的 RED/GREEN、独立审查与快照门禁；最终 Task 8 | 保留未提交 |
| 7 | 进度账 Task 7 的入口、GUI、导出独立审查；最终 Task 8 | 保留未提交 |
| 8 | 验收文档、3699 passed、完整 Radon、最终独立复核与封存 | 保留未提交 |

进度账：`docs/superpowers/plans/2026-09-08-fitting-algorithm-v2-progress.md`；
验收入口：`docs/acceptance/fitting-algorithm-v2.md`。
早期批次的独立审查服务曾不可用；最终整分支独立复核已完成，不倒写为早期已有独立审查。
Poisson low 全 seed 覆盖 81% 及真实测量数据未验仍是已声明限制，不因清单完成而消失。

## Global Constraints

- 已批准设计：`docs/superpowers/specs/2026-09-08-fitting-algorithm-v2-design.md`。
- 新算法身份：`xrr-fit-v2`；目标函数版本：`2`；项目结构版本：`3`。
- 旧算法身份、旧项目结构和旧检查点明确拒绝，错误信息说明版本不支持，不尝试静默转换。
- 不新增生产依赖，不改部署方式，不扩大现有 CI 门禁类型。
- 保留 Parratt、现有解析导数、材料、分辨率、混合波长、足迹和界面物理定义。
- 工作树 `/Users/dala/Desktop/XRR-Fitter/fit-algorithm-v2`，分支 `feat/fitting-algorithm-v2`；不修改其他工作树。
- 唯一公开 Python API 为 `xrr_fitter.api`；服务数值组合遵守架构 allowlist，不在禁止的服务模块引入 NumPy。
- 不自动 commit、push、合并其他分支、创建 release 或删除用户文件；当前用户要求保留未提交工作。
- 每个实施任务必须先取得针对性 RED，再完成 GREEN 与相关回归；不通过放宽容差、跳过测试或吞异常制造成功。
- 新测试放入已登记的 suite 目录；独立覆盖率实验为显式工具，不新建默认 CI mode。
- 测试解释器 `/Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python`；每条命令必须显式设置 `PYTHONPATH=src`，防止 editable 安装误测主仓库。
- 测试/报告/临时产物放仓库外；直接 pytest 使用 `PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`。
- hygiene 要求普通 `.git` 目录；官方验证在当前提交的临时普通 Git clone 中执行，不放宽规则。
- 任务报告记录 RED 断言及输出、GREEN 命令及输出、变更文件、提交、未验项和风险；审查通过后记入进度账。

## 工作流与验收命令

各任务使用精确 suite 路径，基础命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/fit/test_objective.py
```

官方门禁从相同提交加完整未提交文件快照创建临时普通 checkout，`PYTHONPATH` 使用 clone 的绝对 `src`。通过已登记的 `tools/verify.py quality|tools|unit|integration|gui|spawn|regression|statistical` 和 `tools/check_radon.py`，不跳过已登记的测试。注册表没有 `r22-reference` mode，物理参照使用已有 regression。不需要测试发行/推送流程来改变应用版本。新行为对应的旧算法数值断言应重写为 V2 恒等式，物理数值参照容差不变。

## 文件职责与共享接口

- `model/fitting.py`：配置、冻结的完整点数及粗网格积分质量、模式残差单位、搜索证据。
- `evaluation_objective.py` / `evaluation_solver.py`：统一 `Q`、`J`、局部损失和完整尺度链式导数。
- `evaluation_statistics.py`（新增）：模式统计输入、score/bread/meat、期望信息，不依赖 fit/analysis。
- `analysis/covariance.py`（新增）：SVD 秩判定、统计协方差与物理坐标变换。
- `model/analysis.py`：协方差/区间方法、可用状态、逐成员诊断，不能用零填不可用误差。
- `model/project_joint_ranking.py`：仅用保存的 mask、模式与 prior 重建 V2 联合排名，不导入 fit/analysis。
- `fit/adaptive_grid.py`（新增）：网格层级、完整目标复核、可重放决策；已有 stages/pipeline 接线。
- `services.fitting`：单/联合数值证据与分析的组合根，phase 模块只接收 callable。
- `io` 与 GUI：同一模型证据的序列化/显示，不另算统计量。

后续任务消费前一任务已经实现的接口；实现者须在报告中明确最终签名。若细节命名为遵循现有风格需要调整，同时更新本计划，不保留同义 API 或兼容字段。

---

### Task 1: V2 身份、统一 robust 目标、完整尺度导数和冻结粗网格语义

**Files:**
- Modify: `src/xrr_fitter/model/project.py`, `src/xrr_fitter/model/fitting.py`, `src/xrr_fitter/io/project_codec.py`
- Modify: `src/xrr_fitter/evaluation.py`, `src/xrr_fitter/evaluation_objective.py`, `src/xrr_fitter/evaluation_solver.py`, `src/xrr_fitter/evaluation_priors.py`, `src/xrr_fitter/evaluation_model.py`
- Modify: `src/xrr_fitter/analysis/derivatives.py`, `src/xrr_fitter/fit/problem.py`, `src/xrr_fitter/fit/stages.py`, `src/xrr_fitter/fit/joint_evaluation.py`
- Test: `tests/unit/fit/test_objective_priors.py`, `tests/unit/fit/test_objective_jacobian.py`, `tests/unit/fit/test_problem_compilation_stages.py`, `tests/unit/model/test_project_state.py`, `tests/unit/io/test_project_codec.py`, `tests/unit/test_evaluation_priors.py`
- Create test: `tests/unit/fit/test_objective_contract.py`

**Interfaces:**
- Consumes: `FitEvaluationContext`, `values_by_name`, 现有完整物理/约束 Jacobian 和 `compile_fit_problem`。
- Produces: `problem_objective_total(problem: FitEvaluationContext, unit_vector: np.ndarray) -> float` 经 `evaluation` 暴露；`FitEvaluationContext.objective_point_count: int` 为完整点数；`sampling_multipliers: np.ndarray` 为数据项积分质量，完整数据为 1。
- 保留 `ModelEvaluation.objective` 为 `J = Q / N`。`objective_gradient` 是 J 的梯度，`objective_information` 明确为统计总量而非均值的曲率。

- [x] **1. RED：新增公式、版本和真实约束回归。** 基本核对代码：

```python
r = np.array([0.0, 0.04, -0.2])
w = np.array([1.0, 2.0, 0.5])
c = 0.05
expected = np.mean(2.0 * w**2 * (np.sqrt(1.0 + (r / c)**2) - 1.0))
assert robust_log_cost(r, w, c) == pytest.approx(expected)
assert problem_log_probability(problem, u) == pytest.approx(-problem_objective_total(problem, u) / 2)
```

建立 1200 点真实模型平台，约束 `instrument.scale = layer.film.thickness_a / 20`（使用实际 parameter 定义名称），厚度 30；h=1e-6 中心差分核对先验最后一行与完整 J 梯度。原实现先验行错误为 0。粗编译保留 prior center/reason/tau、完整 N 和原区域 label，不再关闭平台先验。对 schema 1、2 和 objective v1 均显式拒绝。

- [x] **2. 运行 RED。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/fit/test_objective_contract.py`，记录旧损失多乘 c²、约束导数遗漏及粗网格 prior 变化。
- [x] **3. GREEN：统一数值与身份。** 所有求解、scalar、Jacobian、MCMC 只消费以下定义；近零/极值使用稳定等价表达，不牺牲已有极端输入测试。

```python
Q_data = np.sum(sampling_multipliers * 2 * weights**2 * (np.hypot(1, r / c) - 1))
z = np.log10(scale / center) / tau
Q = Q_data + z*z
J = Q / objective_point_count
prior_jacobian = scale_jacobian / (scale * np.log(10) * tau)
log_probability = -0.5 * Q + explicit_parameter_prior_log_density
```

粗网格保留完整 labels/weights 的选中行，采样质量总量校准到完整拟合 N，先验不乘采样放大系数。阶段重编译保留冻结上下文。删除 `_migrate_v1_document` 及其调用；不增加 V1 分支。联合稳健等数据集偏好只作用数据项，先验每成员按声明计数并共同归一到总 N，不能重复按数据集平均改变先验强度。

- [x] **4. 回归。** 运行 `tests/unit/fit tests/unit/test_evaluation.py tests/unit/test_evaluation_priors.py tests/unit/analysis tests/unit/model tests/unit/io`；迁移明确因 V2 定义改变的断言并记录原因，不改物理容差。验证总目标、rho 三行、gradient 链一致，约束包括共享与多级。
- [x] **5. 历史提交、自审及正式门禁。** 已有 `0ce68cb`、`08aab7f`，精确接口和验证见进度账 Task 1；当前源码的最终独立整合复核见 Task 8。不重复提交。

### Task 2: 显式 Gaussian / Poisson 模式贯穿数值及数据预检

**Files:**
- Modify: `src/xrr_fitter/model/fitting.py`, `src/xrr_fitter/model/data.py`, `src/xrr_fitter/io/codec_declarations.py`
- Modify: `src/xrr_fitter/fit/problem.py`, `src/xrr_fitter/fit/joint_evaluation.py`, `src/xrr_fitter/evaluation_model.py`, `src/xrr_fitter/evaluation_solver.py`, `src/xrr_fitter/evaluation_instrument_jacobian.py`
- Create: `src/xrr_fitter/evaluation_statistics.py`
- Modify: `src/xrr_fitter/io/xy.py`, `src/xrr_fitter/services/datasets.py`
- Test: `tests/unit/fit/test_noise_modes.py`（新增）, `tests/unit/model/test_fit_masks.py`, `tests/unit/io/test_xy.py`, `tests/unit/fit/test_joint_evaluation_loss.py`

**Interfaces:**
- Consumes: Task 1 的总 Q、N、sampling_multipliers 及完整约束导数。
- Produces: `FitConfig.noise_model` 值域 `robust_log`, `gaussian`, `poisson`；内部残差明确叫 `fit_residuals` 并有单位字段，log 绘图残差不能冒充模式残差。
- `evaluation_statistics` 输出模式残差、d residual/d model、数据信息与 score 所需数组；`evaluation` 是 fit/analysis 的共享入口。
- 最终身份为 `xrr_noise_model` / `2`，具体模式由 `noise_model` 声明；内部 `ModelEvaluation`
  归入 `model/evaluation.py`，公开候选保留单独的 `residuals` 与 log 绘图字段。
- Poisson 选择是用户原始整数计数语义的显式声明；无自动探测或降级。

- [x] **1. RED：真实模型已知 sigma、Poisson 零计数及非法输入。**

```python
np.testing.assert_allclose(residual_gaussian, (model - observed) / sigma)
assert Q_gaussian == pytest.approx(np.sum(((model - observed) / sigma)**2))
assert Q_gaussian_sigma_doubled == pytest.approx(Q_gaussian / 4)
expected_zero_count = 2.0 * mu
np.testing.assert_allclose(poisson_deviance(np.zeros_like(mu), mu), expected_zero_count)
```

非整数/负 counts、缺失/非正 sigma 在 compile 阶段失败。Gaussian 允许有限零/负观测；切换模式不重新启用用户排除点。测试拟合 mask 外非法观测不影响预检；正 count/非正 μ 候选无效。联合 80+800 点信息相加，不等权稀释。

- [x] **2. RED 命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/fit/test_noise_modes.py`。
- [x] **3. GREEN：残差/损失/导数同时按模式接线。** Gaussian 用标准化线性强度，Poisson 用 `sign(mu-k)*sqrt(2*(mu-k+k*log(k/mu)))`；k=0 与 mu≈k 的值和导数用极限/稳定表达。Poisson `mu=normalization*M`，数据归一化不改变计数语义。Gaussian/Poisson 禁止经验区域/等成员再权重；采样积分质量不是经验权重。诊断 log 变换只在定义域显示，不能为负观测造 log。
- [x] **4. GREEN 与回归。** 三模式中心差分检验 Jacobian、scalar gradient、rho 恒等式以及 `log_probability=-Q/2`；运行 fit/evaluation/model/io 相关 suite 并报告精确统计 evidence 接口。
- [x] **5. 历史提交、自审及正式门禁。** 已有 `533f382`、`29d87f5`，验证见进度账 Task 2；当前源码的最终独立整合复核见 Task 8。未混入入口视觉重设计，不重复提交。

### Task 3: 正确的单/联合协方差及逐成员残差诊断

**Files:**
- Create: `src/xrr_fitter/analysis/covariance.py`
- Modify: `src/xrr_fitter/evaluation_statistics.py`, `src/xrr_fitter/evaluation.py`, `src/xrr_fitter/analysis/derivatives.py`, `src/xrr_fitter/analysis/report.py`, `src/xrr_fitter/analysis/joint.py`, `src/xrr_fitter/analysis/classification.py`
- Modify: `src/xrr_fitter/model/analysis.py`, `src/xrr_fitter/services/fitting.py`, `src/xrr_fitter/services/fitting_phases/joint_analysis.py`, `src/xrr_fitter/fit/joint_evaluation.py`
- Modify: `src/xrr_fitter/io/codec_results.py`（模型证据同批可 round-trip）
- Create test: `tests/unit/analysis/test_covariance_calibration.py`; expand `tests/unit/analysis` / `tests/unit/services` 联合报告测试。

**Interfaces:**
- Consumes: 三模式统计输入、全局 scatter Jacobian、物理参数 Jacobian。
- Produces: 不可变 `CovarianceEvidence`（model.analysis），含 names、matrix 或 None、method、rank、不可辨识 names、unavailable_reason；逐成员 residual evidence 含 dataset_id、executed、systematic/autocorrelation。
- 组合根提供联合数值 evidence/callable，分析层不接触 `fit` 的实现类型。
- `UncertaintyReport` 持有相同证据；优化起点 spread 另叫搜索稳定性，不能填 parameter_sigma。
- 最终实现：不可变值在 `model/inference.py`，由 `model.analysis` 暴露；严格的
  `statistical_information(problem, unit)` 组合在 `evaluation_inference.py`，经 `evaluation`
  暴露 `(data_bread, prior_bread, meat)`，模式数学仍在 `evaluation_statistics.py`，避免导入环。
  `UncertaintyReport.covariance_evidence/member_residuals/search_parameter_spread` 为持久化证据；
  `joint_inference_layout` 经服务 callable 接入 `analysis.joint.analyze_joint_point`。

- [x] **1. RED：统计比例与虚假可信度。** 对已知 Gaussian sigma 且只自由 scale 的模型，80 和 800 个相同独立观测应满足：

```python
assert sigma_800 == pytest.approx(sigma_80 / np.sqrt(10), rel=1e-6)
assert sigma_scale == pytest.approx(1 / np.sqrt(np.sum((prediction / sigma_y)**2)))
assert singular_evidence.matrix is None
assert singular_evidence.unavailable_reason
```

用 seed17/18 两个80点、4% log 噪声，仅共享 scale，四个初始值 .46/.49/.51/.54；sigma 不能等于起点优化收敛 spread。加入 `.18*sin(linspace(0,5*pi,80))` 后两个成员诊断为 True，联合不能可信。未执行诊断必须区别于 False。

- [x] **2. RED 命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/analysis/test_covariance_calibration.py`。
- [x] **3. GREEN：统计总信息及 SVD。** Gaussian `I=J_sigma.T@J_sigma`；Poisson `I=J_mu.T@((1/mu)[:,None]*J_mu)`。Robust 用独立 score 的 sandwich `bread^-1@meat@bread^-T`，其中权重和 c 导数完整进入 bread/meat；prior 可进 bread，但数据生成先验不是独立观测，不向 meat 捏造信息。按奇异值与参数尺度检查秩，不能伪逆后把缺信息写为零方差。强自相关时 iid sandwich 明确不可校准/需块 bootstrap。
- [x] **4. 回归。** 运行 analysis、joint fit/service、model/io；检验共享约束映射、秩亏、锁定/边界、低计数及 prior 开关。所有方法与原因能项目 round-trip；旧 synthetic reports 显式声明未执行状态。
- [x] **5. 历史提交、自审及正式门禁。** 已有 `25c90fc`，验证见进度账 Task 3；当前源码的最终独立整合复核见 Task 8。不重复提交。

### Task 4: 区间分类、profile 精度与模式/联合 bootstrap

**Files:**
- Modify: `src/xrr_fitter/analysis/profiles.py`, `src/xrr_fitter/analysis/profile_tasks.py`, `src/xrr_fitter/analysis/report.py`, `src/xrr_fitter/analysis/bootstrap.py`, `src/xrr_fitter/analysis/bootstrap_samples.py`, `src/xrr_fitter/analysis/residual_resampling.py`, `src/xrr_fitter/analysis/joint.py`
- Modify: `src/xrr_fitter/model/fitting.py`, `src/xrr_fitter/model/analysis.py`, `src/xrr_fitter/io/codec_declarations.py`, `src/xrr_fitter/io/codec_results.py`, `src/xrr_fitter/services/fitting.py`, `src/xrr_fitter/services/fitting_phases/joint_analysis.py`
- Create: `src/xrr_fitter/analysis/joint_bootstrap.py`
- Create test: `tests/unit/analysis/test_interval_semantics.py`; expand bootstrap/profile suites。

**Interfaces:**
- Consumes: Task 3 covariance availability/diagnostic evidence；共享完整重新编译与求解 callable。
- Produces: `FitConfig.profile_steps: int` 独立于 bootstrap_samples；profile 和 bootstrap 均记录 interval_kind、confidence_level（可 None）、method、unavailable_reason、成功样本数。
- 正式 percentile CI 的 `MIN_BOOTSTRAP_SUCCESS = 200`；fast 的8次为探索性数据，不产生正式95%区间。
- 最终模型分别位于 `model/profile.py` 和 `model/bootstrap.py`，由 `model.analysis` 暴露；
  profile 保存总阈值与完整 N，bootstrap 保存 `attempted_count/failure_reasons`，成功数派生。
  `UncertaintyReport.bootstrap_evidence` 保存完整采样证据，performed 默认 False。
- `analysis/bootstrap_generation.py` 共用模式生成；`bootstrap_problem_local(..., recompile=...)`
  与 `bootstrap_joint_local(..., recompile=..., refit=...)` 消费服务注入的 compiler/refit。
  服务 `run_analysis` 注入 `fit.problem.recompile_resampled_problem`；请求不保存 callable。
- profile 校准位于 `analysis/profile_calibration.py`，路径与阈值消费位于
  `analysis/profile_paths.py`；正规 likelihood 使用总量 chi-square 阈值，其余显式 support。

- [x] **1. RED：正式区间边界。**

```python
assert result_199.confidence_level is None
assert result_199.intervals == ()
assert result_200.confidence_level == 0.95
assert robust_profile.interval_kind == "loss_support"
assert robust_profile.confidence_level is None
assert likelihood_profile.delta_total == pytest.approx(3.841458820694124)
```

固定 profile_steps 后改变 bootstrap_samples 不改变 profile 网格。无 prior、无经验权重、满秩、内点且诊断通过的 Gaussian/Poisson 才能标 `likelihood_ratio` 95%；其他情形标 support 或 unavailable。

- [x] **2. RED 命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/analysis/test_interval_semantics.py`。
- [x] **3. GREEN：阈值和重采样。** 使用 `scipy.stats.chi2.ppf(.95,1)/N` 转换到 J profile；robust 阈值保持经验语义。Gaussian 参数 bootstrap 按 sigma，Poisson 从 μ 抽整数，robust 用有序移动块 log residual；每次按对应生成目标重新估计平台。联合每个 replicate 先生成全部成员，再拟合整个 shared problem，不能拼各自单独的 sigma。成功计数与失败原因精确保存。
- [x] **4. 回归。** 跑 analysis/profile/bootstrap、joint services、项目 round-trip。块抽样保持用户排除点、identity、normalization 和 counts；快速自动拟合 bootstrap 未请求时不加昂贵阶段，不假报 performed。
- [x] **5. 提交/自审及正式门禁。** `3a28f53` 和集成快照补充 `691f609`；详情见进度文档。
  该批次当时独立审查服务不可用，只有自审；当前源码后来完成 Task 8 的最终独立整合复核。

### Task 5: 拟合点内环和线程隔离的联合系统缓存

**Files:**
- Modify: `src/xrr_fitter/evaluation_model.py`, `src/xrr_fitter/evaluation_instrument_jacobian.py`, `src/xrr_fitter/evaluation_solver.py`, `src/xrr_fitter/evaluation.py`, `src/xrr_fitter/fit/objective.py`, `src/xrr_fitter/fit/joint_evaluation.py`, `src/xrr_fitter/fit/joint_solvers.py`
- Create test: `tests/unit/fit/test_evaluation_workload.py`, `tests/unit/fit/test_joint_solver_cache.py`

**Interfaces:**
- Consumes: 三模式统一 system 数值接口。
- Produces: scalar/residual/Jacobian 内环只计算 fit_mask；`evaluate_model` 的最终发布默认保持完整轴。`cached_joint_least_squares_callbacks(problem, cancelled)` 返回线程隔离 residual/Jacobian callable，同一点只求一次系统。
- 最终签名：`evaluate_model`、单/联合 `evaluate_vector` 使用 keyword-only `fit_only=False`；
  联合 `joint_least_squares_system(problem, global_unit)` 组合成员 system；缓存 cancelled 默认 None。
  profile、MCMC、标量导数内环同步选择 fit_only，分辨率 helper 在计算前选择 row_mask。

- [x] **1. RED：点数与调用次数。** 同一1200点模型 mask 每10点取1点，与仅存120点对照：

```python
np.testing.assert_allclose(masked_residual, compact_residual)
np.testing.assert_allclose(masked_jacobian, compact_jacobian)
assert inner_model_point_count == 120
assert published_model.size == 1200
assert calls_after_residual_and_jacobian_at_same_point == 1
```

真实系统点数通过窄 spy 记录，不能只mock数值。多线程交错不同 u，无串值；输入改变、返回数组被SciPy修改均不能污染缓存。

- [x] **2. RED 命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/fit/test_evaluation_workload.py tests/unit/fit/test_joint_solver_cache.py`。
- [x] **3. GREEN：分离 publication 与 inner layout。** 角度、resolution、波长混合、约束导数使用同一选择；返回完整曲线的路径不复用稀疏数组。联合 scatter 后的每成员 system 同次求残差与 Jacobian，保存线程local拥有的数组，返回独立副本。
- [x] **4. 回归及性能。** 三模式、resolution、共享roughness、cross-dataset constraints、无自由参数都测试。物理Jacobian参照不放宽。记录5次墙钟中位数、物理点数与系统调用数，不承诺通用9倍。
- [x] **5. 提交/自审及正式门禁。** `687f7cf`：unit/quality/integration/regression/gui 全通过，
  固定提交性能复验通过。该批次当时独立审查服务不可用；当前源码后来完成 Task 8 的最终独立整合复核。

### Task 6: 渐进网格、完整目标复核与确定性预算

**状态：已实施、独立审查及官方快照门禁通过；按用户要求未提交。精确证据见进度文档。**

**Files:**
- Create: `src/xrr_fitter/fit/adaptive_grid.py`, `src/xrr_fitter/fit/adaptive_review.py`, `src/xrr_fitter/fit/feature_grid.py`, `src/xrr_fitter/fit/local_budget.py`, `src/xrr_fitter/fit/profile_rescue.py`
- Create: `src/xrr_fitter/model/search.py`, `src/xrr_fitter/io/codec_search.py`
- Modify: `src/xrr_fitter/fit/global_search.py`, `src/xrr_fitter/fit/stages.py`, `src/xrr_fitter/fit/candidates.py`, `src/xrr_fitter/fit/screening.py`, `src/xrr_fitter/fit/pipeline.py`, `src/xrr_fitter/fit/joint_pipeline.py`, `src/xrr_fitter/fit/checkpoint.py`, `src/xrr_fitter/fit/resume.py`
- Modify: `src/xrr_fitter/model/fitting.py`, `src/xrr_fitter/io/codec_candidates.py`
- Create test: `tests/unit/fit/test_adaptive_grid.py`; expand stage/global/resume/parallel tests。

**Interfaces:**
- Consumes: 冻结 prior/N/region 的粗编译、统一Q/J、候选ID与谱系。
- Produces: 冻结 `SearchEvidence` 含 grid_points、full_review_evaluations、budget_allocations、stop_reason、seed、candidate_origin，入 checkpoint。`GridReview.grid_evaluations` 记录当前网格（可为 full）上的初始/重筛工作，`full_evaluations` 仅记录额外完整复核 cache miss；不保留旧同义计数字段。`feature_grid_indices` 和 `reconverge_profile_basin` 分别只由 `feature_grid.py` 与 `profile_rescue.py` 实现。

- [x] **1. RED：混叠与确定性。**

```python
assert grid_levels == (128, 256, 512, full_count)
assert points_per_detected_fringe >= 8 or selected_count == full_count
assert should_promote(relative_error=0.050001, winner_changed=False)
assert full_review_count >= min(8, candidate_count)
assert resumed_evidence == uninterrupted_evidence
```

构造粗网格排序颠倒且完整成本差>2%的候选，必须提升再筛；端点/尖峰保留。小于层级数据不造观测。三代改善<1e-4且无单位距离>=.05新候选才提前结束。

- [x] **2. RED 命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/fit/test_adaptive_grid.py`。
- [x] **3. GREEN：纯决策与现有stage接线。** 相对误差 `abs(J_coarse-J_full)/max(abs(J_full),1e-12)`；换第一名且完整差 `>max(.02*abs(best),1e-12)` 同样promote。每次至少复核前8个成本/结构多样性候选再淘汰。每条保留谱系先给1次局部精修，回收预算给完整成本较优且本轮未追加的谱系；tie按ID。总nfev有上限，复核额外物理评估单独计数。
- [x] **4. 回归。** 相同seed串并行、暂停恢复重放一致；所有停止路径、短预算、无自由参数、联合/自动均正确。运行fit stage/resume以及integration batch resume，不改物理kernel。
- [x] **5. 审查完成，保留未提交。** 任务独立 Spec / Quality 审查及最终 Task 8 整合复核通过；按全局约束保留当前改动，不执行模板提交动作。

### Task 7: API/CLI/GUI 模式入口与报告/文件语义闭环

**状态：实施与入口、GUI、导出独立审查已通过；最终整合门禁已由 Task 8 闭环，按用户要求未提交。**

**Files:**
- Modify: `src/xrr_fitter/api.py`, `src/xrr_fitter/services/projects.py`, `src/xrr_fitter/cli/main.py`, `src/xrr_fitter/cli/commands.py`
- Create: `src/xrr_fitter/gui/noise.py`, `src/xrr_fitter/gui/results/inference_text.py`, `src/xrr_fitter/io/export_evidence.py`
- Modify: `src/xrr_fitter/gui/fitting/panel.py`, `src/xrr_fitter/gui/data/import_dialog.py`, `src/xrr_fitter/gui/data/panel.py`, `src/xrr_fitter/gui/data/mask_editor.py`, `src/xrr_fitter/gui/results/uncertainty.py`, `src/xrr_fitter/gui/plots/diagnostics.py`, `src/xrr_fitter/gui/plots/heatmaps.py`, `src/xrr_fitter/gui/plots/live.py`, `src/xrr_fitter/gui/plots/panel.py`, `src/xrr_fitter/gui/plots/reflectivity.py`, `src/xrr_fitter/gui/plots/sld.py`
- Modify: `src/xrr_fitter/io/export_tables.py`, `src/xrr_fitter/io/export_log.py`, `src/xrr_fitter/io/export_plots.py`, `src/xrr_fitter/io/orso.py`, `src/xrr_fitter/services/exports.py`
- Modify: `src/xrr_fitter/model/analysis.py`, `src/xrr_fitter/io/codec_common.py`, `src/xrr_fitter/io/codec_inference.py`, `src/xrr_fitter/io/codec_results.py`, `src/xrr_fitter/services/fitting_phases/joint_analysis.py`（联合统计轴的纯身份元数据，不改变数值/搜索 provenance）
- Test: unit API/services/cli/io, `tests/gui`, `tests/integration/test_project_roundtrip.py`, `tests/integration/test_export_workflow.py`, `tests/integration/test_cli_workflow.py`

**Interfaces:**
- Consumes: 已冻结的三模式FitConfig、covariance evidence、interval evidence及search evidence。
- Produces: 公开 `set_fit_config(project, config)`，CLI `--noise-model`（省略沿用保存模式；有效模式提示到 stderr），GUI三模式选择及原始counts声明。不新增重复 raw-count bool。
- `api.import_data` 保留 reader/preview 职责；统计输入由优化前 preflight 校验。所有配置改变清理旧计算状态，仅模式变化额外清 structure evidence，保留用户 fit_mask。
- 导出真实发布每数据集 `parameters.csv`；内部 ORSO 入口为 `orso_bytes(context)`，只消费选定候选的保存证据，不保留外传 covariance 兼容参数。
- `UncertaintyReport.parameter_members` 为 None（本地轴）或与全局 correlation_names 对齐的不可变 ParameterReference 成员组；联合发布直接记录真实编译布局，codec 严格必需。导出用 dataset/member 身份映射保存 sigma，不按名字后缀猜测，不重算统计。

- [x] **1. RED：端到端选择与语义。**

```python
assert loaded.fit_config.noise_model == "gaussian"
assert loaded.algorithm_version == "xrr-fit-v2"
assert loaded.schema_version == 3
assert stale_result_invalidated_after_mode_change
assert exported_uncertainty["confidence_level"] is None
assert "95%" not in exploratory_interval_label
```

验证切模式使旧结果/checkpoint不可复用，保留用户fit_mask；缺sigma/非整数counts在启动优化前给清楚错误。不可估计sigma在表格/ORSO为空/不可用，不是0。

- [x] **2. 运行相应新增节点取得RED。** 分别使用已登记unit/gui/integration具体文件，不用全suite的偶然失败代替本功能RED。
- [x] **3. GREEN：入口、codec和显示只消费同一证据。** 中文标签分别为“稳健对数（探索）”“Gaussian（已知标准差）”“Poisson（原始整数计数）”；显示实际诊断状态、方法、区间类型及不可用原因。不改独立GUI重设计的主题/视觉框架。项目schema3必需字段严格读取，不为旧格式增加default迁移。
- [x] **4. 回归。** unit/io/cli/services；`QT_QPA_PLATFORM=offscreen` GUI；单/联合保存-加载-导出-重新拟合。外部ORso验证与物理参照通过。GUI选项必须实际改变优化输入，而不是仅改变文案。
- [x] **5. 审查完成，保留未提交。** 入口、GUI、导出独立审查及最终 Task 8 整合复核通过；按全局约束保留当前改动，不执行模板提交动作。

### Task 8: 全量验收、区间覆盖率与性能证据

**状态：已完成。double-12019 的 A/B 连续回归及 oxide-cap-14017 的 profile 解析恢复已修复并完整重放通过；当前源码的覆盖率、性能、普通 clone 八门禁（3699 passed，含完整 220-case）、完整 Radon 及最终独立 Spec / Quality 复核均通过。源码/索引、科学证据、文档绑定与临时产物清理已闭环；低 counts 与真实数据未验等限制保留，不自动提交、合并、推送或打 tag。**

**Files:**
- Create: `tools/check_interval_coverage.py`（显式实验入口，不注册CI mode）
- Create: `tools/interval_coverage_cases.py`（物理生成及公共 API 拟合/保存加载）, `tools/interval_coverage_evidence.py`（纯保存证据消费）
- Create test: `tests/unit/tools/test_interval_coverage.py`
- Create test: `tests/unit/tools/test_interval_coverage_cli.py`（CLI 输出预检、独占输出及 strict JSON 边界）
- Create test: `tests/unit/tools/test_interval_coverage_cases.py`, `tests/unit/tools/test_interval_coverage_evidence.py`
- Modify: `README.md`, `docs/acceptance/fitting-algorithm-v2.md`（新增）
- 真实验收补回归：`src/xrr_fitter/fit/candidates.py` 的完全重复起点、`src/xrr_fitter/model/project.py` 与新增 `project_joint_ranking.py` 的联合排名验证、`src/xrr_fitter/analysis/classification.py` 的零维候选分类；对应独立 unit/integration 文件及精确 model 架构声明。只修复实测输入路径，不改物理、seed 或统计阈值。

**Interfaces:**
- Consumes: V2正式Gaussian/Poisson区间、单/联合API、模式bootstrap和性能计数。
- Produces: 外部JSON报告含seed列表、每例估计/区间/覆盖、失败、经验覆盖率、二项检验；软件验收记录精确提交与命令。

- [x] **1. RED：实验报告不能挑选种子。**

```python
assert report["seed_count"] >= 200
assert len(report["cases"]) == report["seed_count"]
assert report["failed_count"] == sum(not row["success"] for row in report["cases"])
assert report["nominal_coverage"] == 0.95
```

工具单测注入有失败的确定性案例以验证失败不被删掉；真实实验使用真实模型拟合，不用mock。
- [x] **2. RED命令。** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider -q tests/unit/tools/test_interval_coverage.py`。
- [x] **3. GREEN：固定生成seeds 0..199，名义95%区间用二项检验报告。** 常规Gaussian、低/常规counts Poisson和共享参数病例分组报告；无法满足正则条件的病例保留unavailable/failure，不挑seed。另列robust块bootstrap代表实验，不把loss_support参与95%覆盖率统计。
- [x] **4. 官方验收。** 同提交加完整未提交快照的普通clone运行quality、tools、unit、integration、gui、spawn、regression、statistical和完整Radon；参照测试使用已登记regression，不新增不存在的r22-reference mode。额外显式覆盖率>=200固定seed实验，记录求值点数/次数/nfev/墙钟中位数。若旧R22“算法值”不再适用，明确区分已批准V2变化与未变物理reference；不能放宽物理容差。
- [x] **5. 整理及最终审查。** 文档列每项C/O的实现与证据、真实数据未输入/未运行的限制。清理自身临时clone/cache，保留外部验收报告直到交付。整体分支review后才报已完成；不自动merge/push/tag。

## 自审

C1/C2/C3 -> Task 3；C4/C5/C6 -> Task 1；O3 -> Task 2；O4 -> Task 4；O1/O2 -> Task 5；O5 -> Task 6；所有入口与文件 -> Task 7；慢速、覆盖率及性能验收 -> Task 8。统计依赖先统一目标再模式再推断，界面最后接线；共享写集不会并行。任务实施中依据实际路径校正文档，不削减已批准范围。
