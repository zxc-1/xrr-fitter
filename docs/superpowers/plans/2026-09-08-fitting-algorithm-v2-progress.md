# 拟合 V2 实施进度

## 当前续接：2026-09-14 Joint Bootstrap v5 验收通过

批准范围内的剩余 joint bootstrap 来源/内容校验与有限 B 区间修复已通过实现审查、完整工程和新科学验收。当前 schema5 / `xrr-fit-v2-poisson-5`，objective字符串2；diagnostic v4、refit policy v3不变，无旧格式兼容。执行入口为同日 `2026-09-14-joint-bootstrap-provenance.md` 与 `2026-09-14-joint-bootstrap-coverage.md`，完整结论见 `docs/acceptance/fitting-algorithm-v2.md`。

新独立 seeds7010000..7011999 固定2000例全部完成：formal1995、覆盖 **1954/2000=97.7%**，单侧95% CP下界约 **.9706866254≥.95**，达到预登记K≥1917；R5/U0，所有拒绝仍保留分母。审计002实际 exit0：2000病例观测重放/公共回读、无逐例证据缺口、末态身份校验通过。普通clone709文件同快照七MODE **4483 passed**、完整Radon **611 files/0 issues/PASS**；本轮重核原始回执与文件一致，没有重跑工程或科学实验。

证据根 J=`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/joint-bootstrap-20260914`；汇总 `joint-bootstrap-outcome-001.json` SHA `35d54da704b9b02ae506f5c7c591de46e79f6ec16ae2b57fdc15273f57e23856`，审计 `scientific-audit-002.json` SHA `01f1ab1c5bf94ee2c775bd2be8796b9a3872bd1a4c3d381f091b1c0b72d12249`。审计器哈希域修复105 toy通过并独立复审，旧audit001 FAIL保留。

RMS父计划的已批准场景覆盖关闭项由本轮新证据关闭；不声称任意模型/边界/真实测量的普适95%校准。原单曲线81%→开发重放94%、旧joint93.4%与本轮新joint97.7%属于不同实验，保留原记录；新区间更宽是代价。`status-joint-001/` 记录文档-only收尾，不把新文档冒称原冻结源。无工作树临时探针需删，外置原件与失败保留；不提交、合并或发布。下方均为历史检查点，所有旧shard/summary/audit已结束，不重启。

## 历史续接：2026-09-14 RMS v4 全量诊断验收通过

RMS实现、五例B999、同快照七 MODE **4420 passed** 与完整 Radon **602 files/0 issues/PASS** 已完成。固定新4400+开发200全部封存，独立 audit003 **PASS**：4600 cases / 6600 observation streams、0逐例证据缺口、末尾hash复核通过；八个预登记 diagnostics 端点全部 PASS。
原200种子重放正式profile198/覆盖188，**全seed94%**（原81%为历史值）；新single正式1992/覆盖1908，**95.4%**。joint正式bootstrap1990/覆盖1868，**93.4%**，一般95%校准仍打开；另无独立joint bootstrap context/content seal。事后type6敏感性不充当新验收，也不回填原件。
当前身份 schema4 / `xrr-fit-v2-poisson-4` / objective字符串2 / `poisson-refit-null-v4`，refit policy仍v3。唯一当前执行账为 `2026-09-12-poisson-diagnostic-calibration-progress.md`，验收入口 `docs/acceptance/fitting-algorithm-v2.md`；`2026-09-13-poisson-rms-calibration.md` 保留总体覆盖关闭项。旧81%、旧4/8 FAIL与证据丢失事故保留；所有科学shard已结束，禁止重启。冻结源未变，本次仅同步状态文档并保留外置证据，无提交/发布或用户文件删除。

## 历史续接：2026-09-13 Poisson 两阶段 v3 工程闭环

`2026-09-13-poisson-two-stage-refit.md` 的 T1–T4 已完成。固定 L-BFGS-B → 完整计费交接 → TRF 在不增预算、不放宽成功门槛的条件下修复暴露的低计数坏路径，并补齐严格工作证据。003 同快照七门禁 **4340 passed**、完整 Radon **597 文件 / 0 issues / PASS**；四组 B999 全部 **999/999 null**，最大单路径 nfev 48/46/42/41 ≤ 80，两个注入目标 adjusted p=.001。R6 RED→GREEN、独立关闭及最终运行绑定审查均已完成。
当前身份为 `xrr-fit-v2-poisson-3` / `poisson-refit-null-v3`，schema4、objective 字符串2；完整记录见 `2026-09-12-poisson-diagnostic-calibration-progress.md`。工程汇总：`/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001/two-stage-engineering-final.json`。其后仅有 `status-update-two-stage-003/` 记录的文档同步；保留外置原件、失败记录、分支与未提交改动，无提交/发布或用户文件删除。
**ACF 尺度竞争与新种子科学验收仍未完成。** 旧 81% 和旧 4400 + 200 实验的 4/8 FAIL 不改写，不以四个暴露病例估计新覆盖率，不重跑旧 holdout/corpus。

**历史精度 v2 阶段（2026-09-13）：** Poisson 诊断窄精度修订的实现/审查/七项相关工程验证已闭环（同快照 4189 passed）；低计数收敛与 ACF 功效尚未修完，旧独立实验仍为 FAIL。该阶段执行入口为 `2026-09-12-poisson-diagnostic-calibration-progress.md` 与 `2026-09-13-poisson-local-precision.md`。下列原 Task 1–8 已完成记录不代表当前 Poisson 修订的科学验收。

**原 Task 1–8 历史状态（2026-09-12）：Task 8 已闭环，当前源码八门禁 3699 passed，完整 220-case、
覆盖率/性能证据与最终独立复核通过；按要求保留分支和未提交改动。**
下文按时间保留各轮过程，早期“未完成/失败”及封存记录属于各自历史快照；当前结论与统计限制以
本页顶部和 `docs/acceptance/fitting-algorithm-v2.md` 为准。

## Task 1：统一目标与尺度先验

已实现：V2 / schema 3 身份、旧版本拒绝、无量纲 robust 总目标、
完整约束尺度 Jacobian、单/联合先验归一化、完整数据先验及采样质量的阶段继承。
`objective_information` 表示总 Q/2 曲率；统计协方差校准仍属于 Task 3，不能把本批视为全部完成。

### RED → GREEN

- 原会话的 8 条核心失败及联合目标恒等式失败已接续保留。
- 本次新增复现：`c=1e308, r=1e154` 的可表示损失错误变零；
  V2 不可表示曲率未显式报错；schema 3 能夹带 V1 目标；导数接受错列 Jacobian。
- 本批核心 61 项、迁移后的聚焦回归 234 项通过。
- 完整 `unit` 注册表（包括严格结果插件）：**1703 passed，35.44 s**。
  官方普通 clone 验证将在本批提交后执行，不能以直接 pytest 冒充该入口结果。
- 同种子完整 A–E 搜索重复两次，候选参数、顺序、nfev、阶段、进度及检查点逐项一致。
- 改动文件的 Ruff 与 Radon 指标预检通过；全仓库 Radon 随官方 quality 执行。

### 旧断言迁移说明

- robust 数据项移除旧 `c²` 因子，rho 一、二阶导数同步采用 V2 公式。
- 总信息的先验增量改为 `dz/du` 外积，不再乘 `2/N`。
- 联合数据集平衡不再乘到先验行；MCMC 严格消费同一 Q，独立参数先验除外。
- `c=1e-200, r²=1e-320` 的 V2 rho 导数超出 double 表示范围，改为明确
  `FloatingPointError`，不能沿用 V1 被 `c²` 缩小后的有限数值断言。
- schema 1/2 的迁移测试改为无副作用拒绝；两个仓库示例仅更新版本字段，原始曲线不变。
- V2 配置改变检查点指纹。冻结 Stage-A 目标改为 `5.807945232222954`，
  不再产生粗网格关闭先验的错误警告；冻结候选参数、顺序和 nfev 未改变。
- 未改物理参照、未放宽物理容差、未新增生产依赖或 CI mode。

### Task 1 官方门禁及补充修复

- `0ce68cb` 普通 clone：unit **1703 passed**，quality **188 passed**，
  integration **14 passed**；regression **46 passed / 4 failed**。
- 三条 profile 失败源于测试伪造候选仅差 `0.10`，而 V2 的真实中心目标为
  `103.85452521124125`，2% 改善门槛为 `2.079090504224825`。测试改用
  至少 10% 的相对改善；未降低生产 profile 门槛。
- 自动联合拟合将正常 P1 隔离：`0.043335325545766058 > 0.040248192988975376`。
  原因是 robust 默认 `c=0.05` 的损失转换后，绝对成本容差未转换到 V2 单位。
  新默认为 `0.004`（旧 `1e-5 / 0.05**2`），显式用户设置保持原值。
  新配置测试取得 RED；修复后上述 4 项回归通过，相关测试 **298 passed**，
  余下 2 项为预期配置指纹变化，已更新。候选物理参数及精度容差未动。
- 官方原始日志：`/tmp/xrr-v2-0ce68cb-v8tisd7z/`；普通 clone 已自动清理。
  原因复现：`/tmp/xrr-v2-task1-followup/probe.log`；合成源文件自动清理。
  补充提交 `08aab7f` 官方普通 clone：unit **1704 passed**、quality **188 passed**、
  integration **14 passed**、regression **50 passed**，全部通过。
  报告 `/tmp/xrr-v2-08aab7f-ot6gn8er/`，临时 clone 已清理。

## Task 2：显式噪声模式与共享统计边界

已实现 `FitConfig.noise_model` 三种模式，目标身份为 `xrr_noise_model` / `2`。
模式及数据语义无自动探测或兼容回退。Gaussian/Poisson 不施加经验区域权重或成员等权。

### RED → GREEN

- 新增 26 项初始 RED：缺少模式、Gaussian 仍调用 log、缺失 sigma/非法 counts 未拒绝、
  Poisson 零计数及稳定 deviance 未实现、联合成本仍按成员等权。
- 追加 RED：候选缺少真实模式残差；真实 API 导入排除负观测；联合恢复校验使用旧成员均值；
  Gaussian 诊断丢掉 16 个不可取 log 的点；导入单行 counts 被加权运算舍入成非整数。
- 修复后单元/回归/漂移验收组合 **2240 passed**（含严格 outcome 插件，111.08 s）。
  本轮新增的粗网格与边界 4 项、单/联合真实搜索与恢复 4 项均通过。
- 全仓库 Ruff 与 `tools/check_radon.py` 通过，Radon 报告 `/tmp/xrr-v2-task2-radon.json`。
- 本批官方普通 clone 门禁和独立审查待提交后执行。Task 3–8 未完成，不能把当前
  Gaussian/Poisson 数值目标当作全部不确定度与导出语义已经完成。

### 最终接口

- `model/evaluation.py:ModelEvaluation`：`fit_residuals`、`fit_weighted_residuals`、`noise_model`，
  `residual_name` / `residual_unit` 从模式派生；删除旧 `fit_log_residuals_decades` 字段。
- `model/fitting.py:FitCandidate`：新增全轴 `residuals` 与 `noise_model`，完整 codec/pickle；
  `log_residuals_decades` 只保存正观测与正预测的 log 绘图残差，其他点 NaN。
- `evaluation_statistics.py`：模式预检、残差及模型导数、总 Q 的 rho、score/曲率。
  当前曲率是 Q/2 Gauss-Newton，不冒充 Task 3 的统计协方差。
- `evaluation` 暴露 `validate_noise_data`、`poisson_deviance`、`data_loss_rho`、
  `data_score_information`。新增模块已登记精确 evaluation boundary。
- `read_xy` / `read_xy_bytes` / `api.import_data` 消费显式 `noise_model`；项目导入、
  重载和源更新传播项目配置。编译时保留 fit_mask，不重新启用排除点。
- 联合恢复使用实际联合目标复核排名，沿用 Task 1 的先验独立计数。
- 两个示例仅改目标名和噪声字段；原始数据、物理容差、输出图像基准不变。

### 后续

Task 3–8 尚未实施完成。当前注册表没有 `r22-reference` mode；物理参照按现有
`regression` 中的数值/ORSO 测试执行，不另建 CI mode。
子代理服务重复返回 HTTP 429，本轮改由主代理继续实施和审查，未取得独立子代理审查结论。

本轮外部日志：`/tmp/xrr-v2-resume.O7yj3A/`。临时文件没有写入生产目录。

### Task 2 官方门禁补充

- `533f382` 普通 clone：unit **1750 passed**、integration **14 passed**、
  regression **50 passed**；quality **186 passed / 2 failed**、GUI **646 passed / 1 failed**。
- quality 两项为 Task 2 接口声明未同步：新增 `model.evaluation` 未登记精确模型依赖，
  `import_data` 签名断言缺少 `noise_model`。按实际 DAG 与已批准接口更新，未扩大层级权限。
  聚焦架构及 API 检验 **98 passed**。
- GUI 两点曲线夹具未同步新增的全轴 `residuals`，触发正确的模型轴长拒绝；
  更新夹具数组，保持绘图生产逻辑及所有物理容差不变。
- 原始日志 `/tmp/xrr-v2-533f382-2kpmt2n9/`，临时 clone 已清理。
  本次独立审查服务返回 `Model "gpt-5.5" is not supported`，未获得独立审查结论。

- 补充提交 `29d87f5` 官方普通 clone 重验：quality **188 passed**、GUI **647 passed**。
  报告 `/tmp/xrr-v2-29d87f5-pznr4qvn/`，临时 clone 已清理。
  GUI 保留已有的隐藏画布 `constrained_layout` 警告，未改绘图布局。

## Task 3：统计协方差、秩与成员诊断

已实现单/联合 Gaussian 已知 sigma 信息、Poisson 期望计数信息、robust 独立 score
sandwich。采用总量而非展示均值；数据产生的尺度 prior 只进入 bread，不作为独立 score
进入 meat。联合 robust 成员权重在 bread 中计一次、meat 中计两次。

### RED → GREEN

- 首批 **8 项 RED**：缺少统计方法/秩证据；Poisson 使用 deviance 曲率；robust
  直接逆曲率；强自相关和边界仍返回 sigma；联合 spread 被当作 sigma；缺失诊断被当成 False；
  没有严格的零均值 Poisson 推断边界。Gaussian 同时暴露物理参数有限差分映射的数值偏差。
- robust seed 17 的旧 sigma `0.008262241914726945`，score sandwich 应为
  `0.005875513133346553`。改用解析物理参数 Jacobian 与对应统计矩阵后通过。
- 新增 **3 项真实联合 RED**：四起点 `.46/.49/.51/.54` 实际求解后的全局统计 sigma、
  两个含 `.18*sin(0..5*pi)` 成员诊断，以及 80+800 个 Gaussian 观测信息相加。
- 新增 **5 项证据 RED**：项目 codec 丢失协方差与成员身份、允许 sigma 与矩阵矛盾、
  允许成员诊断与汇总矛盾、缺少 V2 字段仍解码。现均通过。
- 追加 RED：全锁定问题的 `0x0` 矩阵经 JSON `[]` 解码后形状错误；
  联合导数不可用时整份报告失败。现分别恢复正确空轴、保留真实诊断并标注协方差不可用。
- 当前专项 **27 passed**；此前完整 `tests/unit tests/regression` **2265 passed**，
  相关 analysis/model/services/io 回归 **1015 passed**。本批提交后仍须官方普通 clone 验证。
- 全仓库 Ruff lint/format 与 Radon 通过；Radon 报告 `/tmp/xrr-v2-task3-radon.json`。
  不改变物理内核、物理参照容差、生产依赖或 CI mode。

### 最终接口与语义

- `model/inference.py` 的不可变 `CovarianceEvidence`、`ResidualEvidence` 由
  `model.analysis` 暴露。前者含 names/matrix/method/rank/unidentifiable_names/unavailable_reason；
  后者保留 dataset_id/executed/systematic/autocorrelation/point_count/diagnostics/reason。
- `UncertaintyReport` 保存 `covariance_evidence`、`member_residuals`、
  `search_parameter_spread`。`parameter_sigma` 仅来自统计矩阵，搜索 spread 单独保留。
  `covariance` 只返回带方法证据的矩阵；未知相关矩阵为 NaN，未知诊断为 None。
- `evaluation_inference.py:statistical_information(problem, unit)` 经 `evaluation` 暴露，返回
  `(data_bread, prior_bread, meat)`。推断要求完整拟合观测，不消费优化器的零 Jacobian 哨兵。
  `evaluation_statistics.py` 只做模式 score/信息数学，避免与物理/Jacobian 导入成环。
- `analysis/covariance.py` 用列尺度归一化后的 SVD 检查数据秩，秩亏时不给伪逆矩阵。
  边界解、非正 Poisson 均值及强相关 robust 残差不发布局部统计 sigma，保存原因。
- `fit.joint_evaluation.joint_inference_layout` 提供真实全局 scatter 与物理参数映射。
  `services.fitting` 通过 callable 组合到 `analysis.joint.analyze_joint_point`，没有
  `fit <-> analysis` 导入。每个成员先完成实际残差诊断，导数故障不抹掉已执行证据。
- codec 同批严格保存新增字段，不为缺失 V2 协方差字段建立兼容回退。
  旧测试中把起点 spread 当 sigma、对未执行诊断默认可信的断言已按 V2 含义更新。

Task 4–8 仍待实施：区间语义与重采样、内环/缓存、搜索调度、入口/导出和最终统计验收。
本批独立审查尚不可用，不能称独立审查通过。工作树外保留验收日志，未产生仓库内临时产物。

### Task 3 官方门禁

- 固定提交 `25c90fc` 的普通 clone：unit **1777 passed**、quality **188 passed**、
  integration **14 passed**、regression **50 passed**、GUI **647 passed**。
- GUI 仅保留已有隐藏画布 `constrained_layout` 警告。
- 外部报告 `/tmp/xrr-v2-25c90fc-4psplh8d/`，临时 clone 已自动清理。
- Task 4 开始前追加单曲线导数故障自审测试，验证诊断不会因协方差不可用而整体丢失。

## Task 4：区间分类、profile 精度与模式/联合 bootstrap

实现、自审与官方 clone 门禁已完成。正式 percentile CI 要求至少
200 个成功样本且失败率不超过 20%；fast 的 8 次只保存探索性样本，不能标 95%。

### RED → GREEN

- 初始 RED 覆盖 199/200 成功数边界、失败索引、独立 profile 网格、likelihood 阈值和
  robust support 标签，以及模式生成、共享问题整体 refit 和逐次平台先验重估。
- Gaussian 保留负观测，Poisson 保留整数原始 counts、零值及非整数 normalization；
  robust 按 q 排序、中心化 log residual 后做移动块抽样，以 log 反变换生成观测，
  不截断数学上有效的 subfloor 样本。用户 mask、原始行身份和未选点保持不变。
- 非收敛、目标增大、不可表示的生成值分别记录失败；最后一次 refit 内取消也不能发布结果。
- 自审追加 **2 项 RED**：空参数轴被模型接受；联合采样未拒绝噪声模式错配的 residual。
  修复后连同全部失败证据 round-trip、有效平台重估和末次联合取消，专项 **41 passed**。
- 真实 **200 次联合 Gaussian refit** 通过，经验 spread 与总信息 sigma 的核对通过。
  这不代替 Task 8 的 200 个独立数据集覆盖率实验。
- 本轮完整直接 `tests/unit tests/regression`：**2309 passed，327.57 s**。
  先前相关 GUI **247 passed**；正式 GUI 门禁仍须固定提交后运行。
- Ruff lint/format、Radon、`git diff --check HEAD` 通过；提交钩子再次通过。
  Radon 报告 `/tmp/xrr-v2-task4-radon.json`。

### 最终接口与语义

- `model/profile.py:ParameterProfile` 和 `model/bootstrap.py:BootstrapResult` 由
  `model.analysis` 暴露。不保留中间的 `model/intervals.py`。
- `FitConfig.profile_steps` 独立于 bootstrap 数量，standard 为 41、fast 为 11；
  standard bootstrap 为 200、fast 为 8。
- profile 保存 `interval_kind/confidence_level/method/unavailable_reason/delta_total/`
  `objective_point_count`；成功求值数和展示目标阈值由证据派生。正规 Gaussian/Poisson
  使用 `chi2.ppf(.95, 1)/N`，有 prior、秩亏、边界、诊断失败或 robust 模式只标 support。
- `BootstrapResult.attempted_count` 必需，保存逐失败 `(index, reason)`、方法及不可用原因；
  成功数取 samples 行数。模型与入口均拒绝空参数轴。
- `UncertaintyReport.bootstrap_evidence` 保留完整样本，汇总必须一致；
  `bootstrap_performed` 默认 False，无证据不能宣称已执行。
- `analysis/bootstrap_generation.py` 是单/联合共用生成边界；
  `fit.problem.recompile_resampled_problem(problem, data)` 由服务以 callable 注入分析。
  `bootstrap_problem_local(..., recompile=...)` 必需 compiler；`run_analysis` 请求
  bootstrap 时同样必需。`services.fitting.run_analysis` 负责组合，不在请求里存 callable。
- `analysis/joint_bootstrap.bootstrap_joint_local` 先生成全部成员，再整体 refit；
  `fit.joint_solvers.refit_resampled_joint` 检查真实 SciPy success。普通联合拟合启用，
  快速自动联合默认关闭。服务 `_analyze_joint_searches` 接收 bootstrap 开关、取消与进度。
- `analysis/profile_calibration.py` 生成阈值元数据，`analysis/profile_paths.py` 消费保存的
  profile 阈值。codec 严格保存新字段，全部失败的 `(0, n)` 样本轴能 JSON 往返。
- 单曲线协方差导数失败仍保留已执行诊断，补齐 Task 3 的对应边界。

未改变物理参照容差、生产依赖、CI mode 或其他工作树。临时产物在工作树外。
独立审查服务此前返回 429 / unsupported model，尚无独立审查结论。
Task 5–8 仍未实施完成，不把本批统计接口视为整个 V2 已完成。

### Task 4 官方门禁及快照补充

- 实现提交 `3a28f53` 普通 clone：unit **1819 passed**、quality **188 passed**、
  regression **50 passed**、GUI **647 passed**；integration **13 passed / 1 failed**。
- 集成失败来自普通联合拟合旧进度快照未包含新增的 bootstrap 起止事件；生产路径和
  共享结果均一致。更新完整进度快照，并断言真实 attempted_count=1、无正式置信区间，
  聚焦联合集成测试 **2 passed，18.43 s**。未修改生产代码或物理容差。
- 补充提交 `691f609` 官方 integration **14 passed**。GUI 仅保留已有隐藏画布
  `constrained_layout` 警告。
- 原始日志 `/tmp/xrr-v2-3a28f53-nymtapo1/`、`/tmp/xrr-v2-691f609-28mzoc6d/`；
  临时普通 clone 均已自动清理。

## Task 5：拟合点内环与联合系统缓存

实现、自审和固定提交的官方 clone 门禁已完成。所有物理路径共用同一实现，
最终发布默认计算完整曲线；内环仅选择 fit_mask，不改变目标、采样质量或物理容差。

### RED → GREEN

- 初始 RED：1200 点源数据每 10 点拟合 1 点，实际物理工作量仍为 1200；联合缓存缺失。
- 补充 **2 项 RED**：每点角度分辨率仍计算完整轴；补充 **4 项 RED**：profile、
  binary profile、目标导数和 profile path 标量求值仍计算未拟合点。
- 真实先验导数溢出回归取得 **1 failed**：有效的 prior residual=0 被联合 system 的
  Jacobian 溢出打断。仅捕获已知 FloatingPointError，重算 fit-only 真实残差并保留原有
  求解器零 Jacobian 哨兵；意外 RuntimeError 继续抛出，统计推断不消费该哨兵。
- 修复后专项 **41 passed**，覆盖三模式、角度/每点分辨率、混合波长、共享 roughness、
  跨成员约束、线程隔离、缓存输入/返回数组所有权、缓存命中取消与无自由参数。
- 正角度自审最初怀疑的 mask 检查短路并不存在；保留真实回归，未修改该生产判断。
- 最终完整直接 `tests/unit tests/regression` **2355 passed，117.08 s**。
  纳入新增测试后全仓库 Ruff lint/format、Radon 和 `git diff --check HEAD` 通过；
  Radon 报告 `/tmp/xrr-v2-task5-final-radon.json`。

### 最终接口

- `evaluate_model(problem, unit, *, fit_only=False)`、`fit.objective.evaluate_vector` 和
  `fit.joint_evaluation.evaluate_joint_vector` 默认保留完整发布轴；显式 fit_only 时未计算行 NaN。
- scalar Q、MCMC、单/联合 DE、local residual、profile 与导数内环显式选择 fit_only。
  `_model_residual_jacobian` 仅计算拟合物理行，移除 full-model/full-Jacobian 临时数组。
- `_point_resolution_for_wavelength` 与 `_point_resolution_with_jacobian` 接收可选 row_mask，
  在角度分辨率及其导数计算前选择同一行。
- `joint_least_squares_system(problem, global_unit)` scatter 后每成员求一次联合 residual/Jacobian，
  再按真实约束组装全局列；旧独立 Jacobian 入口复用同一组装逻辑。
- `cached_joint_least_squares_callbacks(problem, cancelled=None)` 复用现有线程 local 缓存，
  返回拥有的副本，每次调用（含缓存命中）前后检查取消。`solve_joint` 已接入。

### 性能证据

固定 baseline `691f609` 与未提交实现串行比较，每项 5 次中位数，1200 源点/120 拟合点，
混合波长及每点角度分辨率；数值比较通过。

- residual：0.12005 → 0.01658 s（7.24x）；Parratt 点数 1,224,000 → 122,400。
- system：0.18582 → 0.02326 s（7.99x）；primal+tangent 点数 1,200,000 → 120,000。
- joint callback pair：0.62171 → 0.04958 s（12.54x）；20 次独立回调变为 10 次联合 system。
- joint local fit：0.54642 → 0.08955 s（6.10x）；前后 nfev=8，目标及参数一致。
- 可重放脚本和原始报告 `/tmp/xrr-v2-task5-benchmark.cCgtbC/`。提交后另行复跑，绑定最终提交。
- 该工作量夹具的观测为线性强度，拟合落在 scale 上边界；它证明数值等价及工作量减少，
  不是统计覆盖率或通用速度保证。真实合成病例仍由 Task 8 验收。

本批不改变生产依赖、CI mode 或其他工作树；临时产物保存在工作树外。
独立审查服务此前持续不可用，只有主代理自审，不能声称独立审查通过。
Task 6–8 尚未实施。

### Task 5 官方门禁与固定提交性能复验

- 实现提交 `687f7cf` 普通 clone：unit **1860 passed**、quality **188 passed**、
  integration **14 passed**、regression **50 passed**、GUI **647 passed**。
  GUI 仅保留已有隐藏画布 `constrained_layout` 警告。
- 官方报告 `/tmp/xrr-v2-687f7cf-axepg4tl/`，临时普通 clone 已清理。
- 固定提交 `687f7cf` 对比 `691f609`，无并行验收负载，5 次中位数复测：residual
  **7.67x**、system **7.95x**、joint callback pair **12.33x**、joint local fit **6.04x**。
  数值等价检查全部通过，局部拟合仍为 nfev=8；点数/系统次数与初测一致。
- 新脚本及报告 `/tmp/xrr-v2-task5-final-benchmark.Tzeqdc/`，保留初测记录不覆盖。
  baseline clone 自动清理，无仓库内临时产物；夹具适用范围与上节限制不变。

## Task 6：渐进网格、完整目标复核与确定性预算

实现、独立审查及相关官方门禁已完成，**尚未提交**。验收对象是 `1a8c71d` 加未提交
源码快照，不是新的固定提交；快照 SHA-256 为
`0181d560fda976c9c631a2a8862ae758bd9b21647904156f54505935daff3e17`。

### RED → GREEN 与独立审查

- 真实 solver 包装核对单/联合 B/C/D/E 的每次预算、nfev、seed 与停止原因，覆盖 locked
  路径。B 几何合并不再丢失被合并启动的工作，E 同时保留 DE 与全部 local/restart 证据。
- C/D 每条谱系先精修一次，再按当前完整成本与稳定 ID 逐轮分配有界重启池；一轮内每条
  谱系最多一次，下一轮等待本轮真实结果。串行/并行结果和实际分配可精确重放。
- 网格边界 **13 项有效 RED → 66 passed**：实际采样逐条纹至少 8 行，否则升级/full；
  联合结构坐标按真实 sharing/cross-constraint 映射选取；严格计数字段与不可变成本。
  独立重放关闭全部 4 项发现；没有重复或兼容 sampler。
- 联合末次求解/完整发布取消先取得 **6 failed / 5 passed**，再补 B 发布窗口 RED；
  最终相关 **72 passed**。取消不会多发未完成 checkpoint，已完成 seed 的回调语义保留。
- resume 重建候选保留搜索证据，相关 **37 passed**；profile rescue 漏记追加工作取得
  **1 failed → 19 passed**，职责拆分后相关 **53 passed**。
- 流程独立审查发现 B checkpoint 将发布顺序改为成本顺序，进而交换 C 的 seed/谱系。
  真实 `seed=730, size=160` 和不均预算均取得 RED；统一按 candidate ID 对齐发布顺序，
  相关 **47 passed**。最终快照经真实 strict JSON checkpoint 恢复，所有候选、模型曲线、
  nfev、seed、summary、evidence 与 provenance 精确一致，最佳单位向量差为 0。
- 原冻结输入在所有粗级别仍有欠采样条纹，因此现在使用全部 1200 个真实点；只迁移启动项、
  审计警告、工作量与进度快照。B 合并工作量由 33 补齐为 66；参数数组及最终目标逐位不变。
  冻结/阶段/batch resume 回归 **26 passed**，未修改物理参照或容差。

### 最终接口与证据含义

- `fit/feature_grid.py` 唯一拥有 `feature_grid_indices`；`adaptive_grid.py` 做网格/预算纯决策，
  `adaptive_review.py` 执行完整成本复核。单/联合 B/E 均真实使用所记录的初始网格。
- `model/search.py` 定义不可变 `GridReview`、`SearchAllocation`、`SearchEvidence`，由
  `model.fitting` 暴露；候选、summary、checkpoint 和严格 codec 同批保存。
  `grid_evaluations` 计当前网格上的初始/重筛工作（该网格可能是 full），`full_evaluations`
  仅计额外完整复核 cache miss；两者不冒充优化器 nfev，也不将联合工作乘成员数。
- C/D 调度在 `fit/local_budget.py`；`fit/profile_rescue.py` 唯一拥有四路径原子继续搜索，
  成功时保留历史证据并追加实际 P namespace seed、预算与下一轮编号。
- 删除已被完整复核替代的旧 coarse-only population selector 和无效替换分支，不保留垫片。

### 官方未提交快照门禁

在普通临时 clone 应用上述完整快照，使用现有 `tools/verify.py`，所有 mode 无 skip/deselect：

| Mode | 本轮结果 |
| --- | ---: |
| quality | 188 passed；完整 Radon 529 文件 PASS |
| tools | 448 passed |
| unit | 1955 passed |
| integration | 14 passed |
| spawn | 4 passed |
| regression | 50 passed（含现有数值参照及 ORSO） |
| gui | 647 passed；仅已有隐藏画布 `constrained_layout` 警告 |

全仓 Ruff lint/format、`git diff --check HEAD` 通过。实际注册表没有 `r22-reference` mode，
使用已登记 regression 的参照测试，没有新增或绕过 mode。

可复核命令、源码清单、RED/GREEN、两份独立审查及官方输出保存在
`/tmp/xrr-v2-task6-resume-riede3o8/`。普通 clone 和本次显式测试缓存已清理，保留报告；
仓库未生成缓存，未删除已有 `.superpowers`，未改索引、HEAD、其他工作树、依赖或 CI。
Task 7/8 尚待实施；完整慢速语料、200 固定种子覆盖率和最终性能实验留在 Task 8，
真实用户数据未运行，本批不能称整个 V2 已完成。

## Task 7：模式入口、图形及导出证据（实施与独立审查完成）

API/CLI、GUI 和导出已实施；入口独立审查 PASS。初次冻结快照普通 clone 七门禁
全部通过：quality **189**、tools **448**、unit **2080**、integration **30**、spawn **4**、
regression **50**、GUI **685 passed**（仅已有隐藏画布 warning）。恢复时核对全部
626 个文件与该快照逐字节一致，临时普通 clone 已清理。

这轮绿色门禁没有覆盖下列独立审查发现，**不能用它宣称修复后已验收**：

- 多参数推断图裁切：8 profiles / 8 Bootstrap / 组合图均有真实 bbox RED；有界摘要修复
  已取得 **15 failed / 2 passed → 214 passed / 1 既有 warning**。曲线保留；完整逐参数
  证据由结果正文的 UncertaintyView 展示。入口提示误指 MCMC dialog 的二次发现另取
  **5 项 RED → 44 passed**，真实普通/专家 Qt 接线与六类图形边界均通过；独立窄复审
  **11 passed，Spec / Quality PASS**，两项 P2 关闭。
- 联合 sigma 轴身份缺失：真实 Gaussian/Poisson joint 的 sigma 已保存但 CSV/XLSX/ORSO
  参数行错误为空。已取得真实 **2 failed / 4 passed**，以及四出口/错误同名匹配的
  **25 failed / 4 passed**。不可变 `parameter_members` 已由真实联合布局发布并 strict
  codec 保存；包含自动 joint（项目 sharing_rules 为空）与非共享同名参数。最终
  **60 + 176 passed**，旧消费者隔离重放 **28 failed / 4 passed**；9 个真实成员全部
  既有结果字段/checkpoint 精确不变。独立核对 15 份导出物后 **Spec / Quality PASS**，
  原 P2 关闭；CSV/ORSO 取原 sigma，XLSX 按既有 writer 16G 编码精确回读。

证据目录 `/tmp/xrr-v2-task7-k41he35k/`：入口 `main-review.md`；GUI `gui-review.md`、
`gui-report.md`、`gui-layout-fix.diff`；导出 `exports-review.md`、`exports-report.md`、
`exports-sigma-proposal.md`；初次官方门禁 `official/`。所有 Task 6/7 改动仍未提交。

## Task 8：完整验收（已完成；下列为逐轮历史）

独立的显式 coverage 工具固定 0..199 seeds、四个正式模式/共享组与独立 robust
block-bootstrap 代表实验。报告/证据边界初轮 **41 failed → 41 passed**，按 CLI、
纯证据、物理执行职责拆分后 **55 passed**，Ruff 与原 Radon 政策通过，未删除覆盖或
放宽阈值。独立审查另发现输出位置应在昂贵实验前验证，以及 normalization 文案应为
reader 的前段正值 95th percentile 而非 maximum；均已修复。输出独占预留并持有句柄，
异常时只清理本次 inode；后续复审又补出成功返回前必须验证路径身份的缺口，新增
**2 项有效 RED → CLI 16 passed / 四文件 64 passed**。旧 62 节点全部保留，Ruff、
scoped Radon 与最终独立复审 PASS；外部删除/替换不再假成功，也不覆盖外部文件。
这几项仅影响工具输出与说明，实际科学数值链路不变。

### 真实 pilot 暴露的核心路径

- `poisson_regular` 固定 seed 0：512 个生成起点仅 505 个完整身份，direct-SLD 的密度轴
  归一造成完全重复。生成端按完整身份稳定去重，不补抽、不合并 feature、不放松 selector。
  **3 failed / 1 passed → 68 passed**；原输入两次无插桩完整保存结果/checkpoint 精确一致。
- Gaussian/Poisson 60/30 点联合计算正确，却被项目旧 arithmetic-mean 校验拒绝。
  保存层现在以真实 mask 点数计算 likelihood 权重，并对 robust prior 保留 total-N 单次
  贡献。**20 项有效 RED → 195 passed**；为保持 MI A 将纯标量校验拆到 model helper，
  最终含架构回归 **242 passed**。原三模式 60/30 输入及 400/200 点真实 active-prior 输入
  均两次完整 JSON 精确重放；保留原 `rel_tol=1e-12, abs_tol=1e-15`。
- 上述两项独立复审 **Spec / Quality PASS**：16 组候选池 baseline 对照保持身份/顺序/
  cap/RNG，216 组联合标量/拒绝路径与 runtime 一致，32 个真实保存候选独立核对通过。
- 全锁定独立拟合在 Task 5 就已存在零维分类错误：合法 `(4,0)` 被误当空候选集。
  当前允许零列但仍拒绝零行，并定义空坐标距离为 0；**12 failed / 9 passed → 47 passed**，
  测试职责拆分后 **21 passed**。三模式单/联合保存加载、空统计轴及真实诊断均验证；
  原 SHA 输入两次保存 JSON 精确一致。独立复审 **Spec / Quality PASS**：27 组非空
  baseline 对照及 14 组置信门禁通过，原始系统残差仍使结果保持“可用但相关”。

### 工具链 pilot

原五组 seed-0、`--workers 2` 均实际完成公共 fit/save/load，未替换 seed、物理数据或预算。
Gaussian、低/常规 Poisson 均得到保存的 likelihood 区间；shared Gaussian 和 robust
分别完成正确生成模式的 **200 attempts / 200 successes / 0 failures**。
运行前后 216 个生产源码和六个工具/测试文件 hash 相同；生产源码集合 SHA 为
`a8f5f2b29e1e8fdd877e9669f715fd8a831deab0dcb1d671a2579457bc07f8dc`。
`coverage-pilot-after-core.json` 明确 `protocol_complete=false (pilot_only)`；单次 pilot
不代表 200 个独立数据集的覆盖率。初始失败证据均保留，未用成功报告覆盖。

### 首轮完整覆盖率与整体审查

固定四组 × seeds 0..199 已实际运行，耗时 **189.83 s**；800 次公共拟合全部成功。
运行前后完整源码和索引相同，非文档源码快照 SHA 为
`ae21d9e47b322bda4ce0b4de17721259907fff7a08c813f60efe878d8e35195f`。
独立审计重建 1000 份观测的 RNG、原始文件字节与 normalization，逐一 exact；
重新核对所有区间分母与双侧精确二项检验，未挑选 seed。

| 分组 | 可用区间 | 覆盖真值 | 可用条件覆盖 | 全 seed 覆盖 |
| --- | ---: | ---: | ---: | ---: |
| Gaussian | 198/200 | 191 | 96.46% | 95.5% |
| Poisson low | 168/200 | 162 | 96.43% | 81.0% |
| Poisson regular | 200/200 | 191 | 95.5% | 95.5% |
| Shared Gaussian | 200/200 | 184 | 92.0% | 92.0% |

Gaussian 2 例、低 counts 32 例因 `residual_diagnostics_failed` 无正式区间，原样保留。
低 counts 的全 seed 口径拒绝 p=0.95（双侧 p=1.40209e-12）；不能用条件覆盖率遮掉
不可用病例。其余不拒绝原假设也不等于证明通用校准。Robust 独立代表实验完成
200/200 moving-block bootstrap，不将单例称作覆盖率校准。

首轮完整证据为 `/tmp/xrr-v2-final-_rb817em/coverage.json`、`coverage-audit.json`。
整体分支独立审查已启动，确认联合 bootstrap 在分析后失败时没有重新应用置信门禁；
真实服务组合的纯控制流探针可使 200/200 次失败仍发布“可信”。该项已在后续恢复轮
补齐修复和真实路径回归；当前首轮覆盖率仍是修复前证据。

README 与 `docs/acceptance/fitting-algorithm-v2.md` 区分实现和未完成验收。完整
220-case corpus、最终性能复跑、最终普通 clone 门禁与整体独立审查仍未完成；
当前不能声称整个 V2 已完成。历史任务证据保留于 `/tmp/xrr-v2-task8-t718qpds/`。


### 2026-09-11 恢复：联合 bootstrap 分类门禁

采样证据现在由 `analysis/joint.py` 在最终分类前合并；服务传入既有采样 callable。
只有真实执行且失败率严格大于 20% 才以 `bootstrap_failure_rate` 标为“不可信”。
三模式真实共享问题验证 0/40/41/200 次失败（总计 200 次），保留每次失败原因，成功
replicate 仍调用真实 refitter；未启用采样保持原状。有效 **6 failed / 7 passed RED**，
修复扩大回归 **247 passed / 1 个既有清洁目录失败**：工作树 `.superpowers` 被原架构
门禁禁止，保留该用户已有目录，最终普通 clone 按原门禁复验。

公共 `fit_project` → save → load 注入 200 次明确 refit 失败，两次输出 JSON 精确一致，
全部联合成员均“不可信”；采样生成、分析、持久化仍走真实实现。新回归 strict codec
往返、scoped Radon 和全仓 Ruff lint/format 均通过，未改物理或统计容差。

本轮证据目录：
`/var/folders/4b/3j6m_kld2klgm7dxq___yd2m0000gn/T/xrr-v2-resume-final-m12xh1q8/`。
修复包 `bootstrap-confidence.diff`、`bootstrap-confidence-report.md`、
`bootstrap-public-replay/report.json`；本轮仅新增三个源码/测试改动，其余既有工作保留。
冻结的非文档源码 SHA 为
`6f8fcac82edcc885fba9f8c25196dd5ec0e5d1a9e4e618947a14a5595d8fce85`。
最终覆盖率、性能、完整语料和普通 clone 门禁正在执行；独立分支审查仍在进行。


### 最终实验与测试归属补齐

修复后四组各 200 固定 seeds 重跑 **190.95 s**，800 次公共 fit 成功，1000 份成员观测
的 RNG、原始字节、normalization、区间分母和二项检验审计通过。区间/covered/失败原因
与修复前 800 例逐一 exact；低 counts 的 32 个 unavailable 完整保留。

Micro 五次中位数在数值等价前提下，residual/system/joint callback/joint local fit
加速分别 **7.27×/8.01×/12.43×/6.12×**；完整 automatic 四类各五次非时间证据完全
确定，异常 P4 保持 review，其余 passed。补充两文件集成 **9 passed**。

首轮普通 clone quality **189 passed**，完整 Radon PASS；tools **511 passed / 1 failed**。
失败来自仓库既有“每份测试恰属一个 mode”规则：Task 7/8 两份新增集成未登记，单独
补跑不能满足合同。先同步精确注册期望，取得 **3 failed / 22 passed RED**，再将
`test_export_parameter_members.py`、`test_locked_fit_workflow.py` 补入现有 integration。
相关 **52 passed**，独立窄复审 PASS；只增加两个显式测试路径与对应断言，原 ownership
检查、门禁种类、workflow jobs 和 outcome plugin 保留。

最终官方验收正在新普通 clone 重跑，证据根为本轮报告目录 `final-verification/`。
非文档源码 SHA 更新为
`1832838f6bee3ec80b2c5983fd1089ba374b65898b30dd2572524f5291f49d93`；
生产源码集合 SHA 始终为
`17a87630de739530cd45ca86649f48e24df8c0a34294ae00317ea6cee8d9c695`。
覆盖率/性能与最终状态的逐文件关联在 `final-verification/rebind-evidence.json`，
只发生测试注册/断言和文档变化，因此不重复相同科学实验。原失败验收报告保留，首轮
clone 已自动清理。完整慢速语料和最后八门禁结果尚待回收。

### 最终 statistical 返回失败，继续原病例定位

最终冻结快照的前七项官方门禁合计 **3673 passed**；statistical 为
**1 passed / 1 failed，7150.46 s**。完整语料的定义与确定性检查通过，实际拟合检查
在 `double-12019` 的 `confidence != UNTRUSTED` 断言失败，伴随三条 Stage A 候选
拒绝警告。完整失败栈位于本轮 `final-verification/official/statistical.log`。
这不是八门禁通过；保留原 seed、数据、预算及阈值，正在用同一病例和
`local_workers=2` 无插桩重放以获取完整分类/候选证据。

官方运行前后 HEAD、非文档源码与索引均未改变；普通 clone 已自动清理。
失败后的调查与重放输入输出保存在本轮 `statistical-failure/`，不覆盖原验收证据。

### 原病例确认 Stage A/B 声明起点连续回归

`double-12019` 的原始输入 SHA 为
`74d4a8c494053e812dd1352408c4f75e890f737eb743836d1d9d4455c8a25e05`，
data seed 12019、fit seed 22019、无噪声、520 行（518 个 fit 点），`local_workers=2`。
真实分类原因是 `insufficient_cluster_support`；bootstrap 8/8 成功，不能归因于采样。
Task 5 HEAD 在相同原始字节上通过，四条 E 均达到约 `2.521114435166e-06`。

Stage A 的新 8 候选复核提前排除 declared baseline；按 `feature_key` 将其加入每轮
真实复核并集，仍执行物理检查、完整成本提升判断和条纹筛选。原 Stage A RED 及边界
RED 后专项 67 passed。仅此修复的完整病例重放仍失败（110.27 s），证明还存在后继断点。

Stage B 的 declared 成本 11.090882410319985，几何距两个 DE 代表均超过 0.05，且在
原 10 倍成本范围内，却被按 ID 排除于 C/D continuation。恢复所有候选一起参与现有
几何代表选择、实际工作量保留、归档和发布顺序预算映射；没有改成本阈值或预算设置。
有效 RED **6 failed**；修复及原场景回归 **136 passed，46.78 s**。独立窄复审未发现
实质阻断。归档测试替身现在完整分区所有输入，不给生产代码加缺项 fallback。

冻结输入仅合并重复 B audit：B-0 的 nfev 66→67、best_index 4→3；保留候选参数、
C/D/E、阶段总 nfev、进度和检查点身份精确不变。全仓 Ruff lint/format 通过。
完整 Radon 发现 Stage A 复核函数 CC11，抽取声明起点索引选择后全仓通过；补验
**50 passed，6.99 s**。原 RED、GREEN、冻结真实输出与 Radon 均保存在
`statistical-failure/`。当前正在无插桩重放原病例，随后冻结新源码复验科学实验和八门禁。

### A/B 修复后双重放与新快照实验

原病例两次完整无插桩重放均通过，分别 **107.92/106.95 s**。完整 fit-result JSON
逐字节 SHA 为 `a82297761aa629875d3b1aa8f62765005690375ed6ffeafbf7f21638adce1127`，
strict 保存加载、原输出与恢复指标断言通过。独立复核确认声明 B→C→D 的厚度及成本
与 Task 5 精确一致；4 条 E 各有独立 seed/真实预算，目标逐项与 Task 5 精确一致。
分类恢复“可用但相关”；bootstrap 8/8 仍为探索性。第一跑只发生三份文档更新，
第二跑全树未变；两跑 HEAD、非文档源码、生产源码与索引均未变。

新冻结源码 SHA 为 `157304e16cedf8e7d558e5ed0a42ad6c1fab8f76a1172bc3ce8b5e977ae46d74`，
生产源码 SHA 为 `462317173980165924f1accded89743c1c0d8f31536a40c4115cfb3100fee7d9`。
新证据位于本轮报告根 `final-verification-ab/`。性能顺序复验：micro 数值等价，五次
中位加速 **7.24×/8.01×/12.47×/6.08×**；automatic 四类各五次，非时间证据精确一致，
只有预期的异常 P4 保持 review。覆盖率 **212.83 s、800 fits、1000 份原始观测审计 PASS**，
可用/覆盖数仍为 198/191、168/162、200/191、200/184。18 个低 counts 区间详情变化，
其中8个边界最大差 `7.0344e-12 Å`；可用性、covered、失败原因全部不变，低 counts
全 seed 81% 和 p=1.40209e-12 的限制继续保留。

官方八门禁已在新普通 clone 启动，quality **189 passed**、完整 Radon **559 文件 PASS**；
其余结果仍待回收。旧失败轮7个测试缓存及 Task 5 对照 clone 已按归属记录清理，
正式输入、日志、JSON、补丁及审查报告保留，清理记录 `historical-scratch-cleanup-ab.json`。

### 生成夹具候选身份迁移与最终整合重跑

第一轮 A/B 快照 quality **189**、tools **512**、unit **2190 passed**，integration
**38 passed / 1 failed**。`_automatic_sigma_design` 固定寻找 `B-declared-start` 来生成
观测；正常几何合并后该审计 ID 已不存在。仅将 design 参数锁在声明初值，通过真实
公开 API 取其前向曲线；实际 automatic 项目仍只放开密度，seeds17/18和导出断言原样。

原官方旧快照在普通 clone 复原并逐文件核 hash：改前/后80点 prediction exact，
`design.xy`、`auto-17.xy`、`auto-18.xy` 三份原始字节 exact，实际拟合设置 exact。
原集成文件 **3 passed，12.24 s**；Ruff lint/format 通过。比较脚本、JSON和原数据保留
在 `statistical-failure/sigma-*`，两份对照 clone 已自动清理，没有新增生产修复。

新的非文档源码 SHA 为
`a6a6dcf298c20382518fadcf2e7941f619d376aedd8bf75a899d78ed7fc36a40`，生产源码 SHA
仍为 `462317173980165924f1accded89743c1c0d8f31536a40c4115cfb3100fee7d9`，索引未变。
科学实验、双重放和统计生成协议未受测试辅助文件影响；
`final-verification-ab-final/rebind-evidence.json` 保存逐文件差异及科学报告哈希。
新普通 clone 的完整八门禁已启动，旧失败报告不覆盖，最终结果尚待回收。


### 2026-09-12 恢复：修复 profile basin 的数值路径分叉

上一轮 `final-verification-ab-final` 已结束：前七项 3688 passed；statistical 为
1 passed / 1 failed（7868.12 s），失败是 `oxide-cap-14017` 粗糙度 profile 不覆盖真值，
不是仍在运行。原病例本轮无插桩重放再次失败，输入 SHA 为
`e004114563fe099d6c68c792f2a00e6d98f6fc050ec88221908a687921c26421`。

真实 profile 已有更优区域，但 basin 恢复仍使用有限差分 L-BFGS-B。单一变量对照中，
标量路径 12 次 ABNORMAL、85.61 s、无 decision；已有解析残差/Jacobian 路径 8.24 s，
发现目标 1.430287814。恢复入口现复用相同解析系统、模式损失和配置派生的有界预算。
不改 seed、输入、profile_steps、物理定义、统计阈值或默认配置。

有效 RED 5 failed / 4 passed → GREEN 45 passed；扩大回归 347 passed。
完整 Radon 560 文件 PASS，全仓 Ruff lint/format PASS。独立只读窄复审按用户要求
继承主模型及推理强度，未发现实质阻断；独立另验 solver/约束 9 passed、真实中心 1 passed。

原病例两次完整重放 72.93 / 72.02 s，strict codec 和原指标断言通过；输出 JSON
字节一致，SHA 为 `38e2d364c6a482d7bfe02375df588fd7ef2b7dc984663c9238b07125d04ecac6`。
最终最优目标 3.05284e-26，粗糙度 2.8073378735704506 Å，分类“可用但相关”。
A–D 候选及阶段摘要精确不变，E 历史工作与新增四路 nfev 逐项核对；旧
`double-12019` 新重放与先前成功结果逐字节一致。

本轮证据根 `/tmp/xrr-v2-profile-resume-qms2ukkz/`，审计 `replay-audit.json`。
非文档源码 SHA 为 `91d2a711cfebf479a2061401ab354d64664529aa5de18cfca66a07c744c66e1b`，
生产源码 SHA 为 `592fa0600aefe3a0b1edb7fd34dc14090e2c861ecc4dc845bdbc5956187f87fc`；
仅新增 1 个生产文件改动和 3 个测试文件改动，HEAD 与索引未变。
新科学实验与普通 clone 八门禁正在 `final-verification/` 顺序复验，尚不声明整体验收完成。


### 2026-09-12 当前冻结源码的科学复验与七门禁

当前证据为 `/tmp/xrr-v2-profile-resume-qms2ukkz/final-verification/`。
Micro 五次中位加速为 **7.48× / 8.12× / 12.54× / 6.11×**，保存数值和 local fit
记录 exact；每项五次工作计数一致。Automatic 四类各五次非时间证据 exact，
中位时间 **0.6532 / 4.3108 / 3.4838 / 5.3956 s**；isolated-outlier P4 仍为 review。
阶段严格顺序执行，没有重叠本任务验收负载；未监测整机外部负载，不声称整机空闲。

覆盖率耗时 **193.19 s**；四组完整 seeds 0..199、800 次 public fit 均有有效结果，
1000 份成员观测的 RNG、原始字节及 normalization 独立补核通过。可用/覆盖数为
198/191、168/162、200/191、200/184；Gaussian 2 例、低 counts 32 例 unavailable
原样保留。低 counts 全 seed 81%、p=1.402092333154397e-12 的限制不变。
本轮 800 例完整 interval、非时间字段和全部 observations 与上一 A+B 科学报告 exact；
旧文档的“18 个区间变化”只属于更早轮历史。Robust 单例 bootstrap 200/200 不是覆盖校准。

继承主模型和推理强度的只读代理完成独立补核：重建 566 个 LR 区间、200 个共享
bootstrap 区间，以 65 位 Decimal PMF 枚举复核双分母二项 p 值；0 次额外拟合或物理求值。
记录见 `scientific-review.md`。仍未运行真实用户测量数据。

当前官方 quality/tools/unit/integration/gui/spawn/regression 分别为
**189 / 512 / 2198 / 39 / 704 / 4 / 51 passed**，合计 **3697**；完整 Radon
**560 文件 PASS、零违规**。statistical 仍运行完整 220-case corpus，不能提前报八门禁通过。
当前 647 文件与执行前冻结快照精确一致后，只更新验收说明、计划状态和本进度文档；
非文档源码、生产源码、HEAD 与索引未改变。最终整合审查及清理须在官方结束后补核。


### 2026-09-12 八门禁完整通过

当前官方阶段已于 **2026-09-12 03:00:53 UTC** 完整结束，退出码 0，总耗时
**8911.32 s**。最后 statistical 为 **2 passed in 8382.91 s**，原测试实际检验
完整 220-case 定义、220 次拟合、原恢复与降级阈值及空失败集合，不是单病例替代。
八门禁合计 **3699 passed**，各 mode 无失败、skip、deselect、xfail 或 xpass；
GUI 仅一条既有隐藏画布布局 warning。完整 Radon 为 560 文件、零违规、平均 CC 3.17184。

运行前后 HEAD、非文档源码、生产源码和索引精确一致，12 份当前科学证据 hash 也一致。
普通 clone 已自动清理；已按预审过的归属脚本清理七个 pytest-tmp、十六个 mpl/xdg
缓存目录，statistical 未生成 pytest-tmp；micro baseline clone 及审查遗留空目录均不存在。
保留所有原失败、完整重放、报告、补丁及 `.superpowers`。清理回执位于当前证据根。
Task 8 第 4 项已关闭；最终独立整合复核、文档 hash 绑定及第 5 项正在收尾。


### 2026-09-12 最终独立复核与封存

最终独立整合复核 **Spec / Quality PASS**，108 项只读检查通过；当前 85 个生产变更
路径与历史整分支审查集合一致，后续生产差异均有专项复审，没有遗漏审查的新变更。
报告为 `/tmp/xrr-v2-profile-resume-qms2ukkz/whole-branch-final-review.md` 和同名 JSON。
本轮只读复核不重复拟合，不把旧静态结论替代完整八门禁；完整统计、Radon、来源身份、
12 份科学证据和实际清理均已核实。最终状态文案更新后仅三份文档变化，生产源码、
测试、工具、HEAD 与索引不变。

Task 8 第 4、5 项均已关闭；最终绑定为当前报告根
`final-verification/rebind-evidence.json`，最终审计为 `final-verification/final-audit.json`，
独立封存窄核为 `final-verification/final-seal-review.json`。已清理本轮普通 clone、
七个 pytest-tmp、十六个 mpl/xdg 缓存及审查遗留空目录；保留 `.superpowers`、失败输入、
重放结果、原始日志、补丁和独立复审材料。没有提交、合并、推送、打 tag 或发布。

科学边界不变：低 counts 全 seed 覆盖仅 81%，拒绝名义 95%；Gaussian 的 2 例和
低 counts 的 32 例 unavailable 未删除。实验仅覆盖其他参数已知的合成单层厚度；
robust 单例 bootstrap 不代表覆盖率校准；真实用户测量数据未验，性能不承诺通用倍率。

### 2026-09-12 计划清单与恢复账本同步

按用户要求回到现有计划核对，发现 Task 1–3、6–7 共 25 个步骤未同步勾选，
`.superpowers/sdd/progress.md` 仍保留初始 Task 1 in progress。依据各批 RED/GREEN、
历史提交、最终八门禁和独立整合复核，同步清单与恢复入口；没有重做已完成的实现。

Task 1–5 的历史提交和早期独立审查记录保留；Task 6–7 的提交模板
改为“审查完成，保留未提交”，Task 6–8 继续保留未提交改动。
不将未发生的提交或早期未获证实的独立审查写成已完成。
原计划 Task 8 要求保留并报告 unavailable/failure，未要求通过修改阈值消除它们；
Poisson low 全 seed 81% 与真实数据未验限制保持不变，没有启动新的统计定义修改。

本次只同步计划、本文、验收入口及恢复账本。原封存报告不覆盖，改前文件已备份；
新鲜窄回归、文件差分、源码/索引与旧验收证据的重新绑定见
`/tmp/xrr-v2-plan-closeout-z49kl5k9/verification.json`。
当前实现没有原计划内待执行的任务；分支和全部既有未提交改动继续保留。

本轮原计划 10 个关键测试文件 **206 passed，13.67 s**；计划每个 Task 均为 5/5，
数值示例代码块原样保留，`git diff --check` 通过。旧初始临时报告
`/tmp/xrr-v2-sdd/task-1-report.md` 已不可用，其历史引用保留并标注；
当前完成度由各批进度、已存在的历史提交和最终整合验收交叉核对，不补造缺失报告。
