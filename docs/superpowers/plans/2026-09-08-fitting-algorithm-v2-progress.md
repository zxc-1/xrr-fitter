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
