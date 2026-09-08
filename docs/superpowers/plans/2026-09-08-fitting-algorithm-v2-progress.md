# 拟合 V2 实施进度

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
