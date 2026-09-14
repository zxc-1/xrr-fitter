# Poisson 低计数诊断校准：单实现修复规格

## 1. 范围与用户裁决

用户要求查明低计数全 seed 覆盖 81% 的根因并修复，随后确认“没有旧格式文件，不用管旧的”及“继续”。
按已呈现的模拟计数＋有界同估计器重拟合方案执行；不做迁移、旧结果阅读器、兼容层或双实现。
原 V2 Task 1–8 保持已完成。本规格是新增修复，不改变 Poisson 似然、signed deviance、零计数、物理模型或已封存结果。
不新增依赖、CI 门禁类型，不自动 commit、merge、push、tag 或发布；保留全部既有未提交改动。

## 2. 已确认的缺陷与目标

200 个正确 Poisson 模型的样例均有效拟合；168 个正式区间、162 个覆盖真值。
32 个 unavailable 分别由 28 个 background、1 个 footprint、3 个 ACF 筛查触发。
零计数残差为 +sqrt(2*mu)，不满足旧趋势和白噪声 ACF 启发式所暗含的交换性。
修复的是诊断误拒机制，不通过删样本、调宽旧阈值或重新命名旧区间提高数字。
原 81% 为全 seed 覆盖产出，96.43% 为条件覆盖；二者在新实验中继续分开报告。

## 3. 状态语义

- 原始筛查照常执行，raw_systematic、raw_autocorrelation、advisories 永久可见。
- Poisson 筛查未触发：不运行 MC，calibration=None；明确是“未触发校准”，不是已验证模型正确。
- Poisson 筛查触发：必须获得有效校准证据；缺 refitter、预算不足、生成/拟合失败或不匹配均不可用。
- available 且拒绝：effective systematic/ACF 来自同族 max 校准后的成员/检测器决定。
- available 且不拒绝：保留原 advisory，但不再把它当作已证实物理无效或正式区间阻断。
- unavailable：effective executed=False、systematic/autocorrelation=None，并保留原始筛查与明确原因。
- 真物理诊断、非有限/不完整输入、先验、边界、秩亏、零均值非正则等门禁独立保留。
- Gaussian/robust 原诊断结论不改变；其原 advisory 仍构成原有阻断。
- automatic、classification、profile、joint 的不可用状态不得因 None 的布尔值为假而被误当通过。

## 4. 固定统计规则

默认诊断模拟数 B=999、族 alpha=0.01，单次完整单/联合拟合一个检测族。
SearchBudget.diagnostic_samples 显式持久化，默认 999；小于 99 时返回 diagnostic_budget_insufficient，不能假校准。
FitConfig.diagnostic_version 固定为 poisson-refit-null-v1；校准方法 poisson_refit_null_v1，refit_policy=declared_sobol4_v1。
独立随机流 domain=0x504F495344494147，从原有效 master_seed 派生；不消耗搜索或 bootstrap 流。
固定 B，不根据中途 p 提前结束；失败后停止是明确 unavailable，不发布 p、不丢失败补样。

### 4.1 连续统计量

保持旧排序、窗口、FFT 和 signed deviance；D 为首末三分位 median drop：

```
U_footprint = max(0, -rho) * D / (0.75 * 0.05)
U_background = max(0, -rho) * D / (0.70 * 0.05)
U_surface = (peak_2_to_50_A - median_spectrum - 5 * 1.4826 * MAD_spectrum) / sqrt(n_surface)
U_acf = sqrt(n) / 3 * second_largest(abs(ACF_lags))
```

保留 D 与 surface contrast 的符号，不截顶、不加 jitter。常量相关量依原规则取零；非有限值不置零。
footprint/background 是否适用由仪器声明决定；surface 是否存在频带由固定几何决定。
ACF 的 lag 范围保持 1..min(20,n//5)。旧 advisory 仍使用旧双条件，不改成上述乘积判定。
所有适用检测器入族，包括 observed 未触发的列；joint 不只选异常成员。

### 4.2 对称 max 校准

令矩阵 U 有 observed 加 B 个 null 共 B+1 行：

```
c_j = median_i(U_ij)
s_j = max(1, 1.4826 * median_i(abs(U_ij - c_j)))
z_ij = (U_ij - c_j) / s_j
S_i = max(0, max_j(z_ij))
p = count_i(S_i >= S_0) / (B + 1)
p_j = count_i(S_i >= max(0, z_0j)) / (B + 1)
```

下限 1 是预声明统计量单位。ties 用 >=；p_j 是同族定位证据，不宣称因果归因或部分原假设下强 FWER。
成员 systematic = 该成员任一 p_j <= alpha，autocorrelation = 该成员 acf 的 p_j <= alpha。
至少一个成员拒绝与 omnibus 拒绝一致。记录 B、tail/tie count、p、分辨率、每列 observed/center/scale/p_j。
已知 mu 且每行映射相同有行置换等变的保守 rank 性质；未知参数 plug-in 只称近似校准。
全矩阵标准化后的 tail count 不直接套普通独立二项 Monte Carlo 误差条；分辨率不是误差条。

## 5. 一致、有界的重拟合

诊断 estimator 与产品 A–E 搜索明确区分。observed 和每个 null 都从以下同一规则重跑：

1. 声明初值（encode_physical_vector(problem,{})；joint 使用 initial_joint_vector）。
2. scipy.stats.qmc.Sobol(d, scramble=False).random_base2(2)[1:] 的三个内点。
3. 按顺序去掉逐位重复起点；d=0 只求值一次，不优化。
4. 每个起点同样 bounded analytic least_squares，bounds=(0,1)、trf、x_scale=jac、三容差 1e-10。
5. 每路径 max_nfev=max(local_min_nfev,local_nfev_per_parameter*max(1,d))。
6. 所有声明路径都必须有效且收敛；任一路径失败返回失败记录，不能只选成功路径来假装完整。
7. 完整成功时取最低原目标的解，等值按起点顺序。记录 optimizer nfev，不把它冒充全部物理求值次数。

observed refit 必须同时满足：unit L∞ <= 1e-4、总目标 Q 差绝对值 <= 1e-6、
拟合计数均值的对称 Poisson KL 总和 <= 1e-6。
该总和明确定义为全部成员拟合点的 `sum((mu - nu) * log(mu / nu))`（不乘 1/2）；
两均值同为零贡献零，一零一正不匹配。
若不满足，diagnostic_refit_mismatch；不更换产品 winner。统计量用 observed refit 的真实输出，不用近似相等的产品残差代替。
所有 null 从产品 winner 的原计数均值生成。保留 q、fit mask、normalization、锁定参数/约束；每次重编译 data-derived prior。
joint 每次重编译整个共享问题并一起拟合；初版混合噪声 joint 的 Poisson 校准明确 unavailable（mixed_noise_diagnostic_calibration_unsupported），不冒充全 Poisson 校准。
按最多 16 个 replicate 的有界批次生成并运行，输出按 index 收集；refitter 内不调用 report/profile/API、不嵌套提交线程池任务。
取消沿既有异常边界传播，不发布校准通过或半份正式区间。生成/优化的预期数值失败记入 unavailable；编程错误不吞掉。

## 6. 不可变模型与接口

新增 model/diagnostic_calibration.py：

```python
@dataclass(frozen=True, slots=True)
class DiagnosticStatistic:
    dataset_id: str | None
    kind: str  # footprint/background/surface/acf
    observed: float
    center: float | None = None
    scale: float | None = None
    adjusted_p_value: float | None = None

@dataclass(frozen=True, slots=True)
class DiagnosticCalibration:
    status: str  # available/unavailable
    sample_count: int
    child_seed: int
    owner_sha256: str
    method: str = 'poisson_refit_null_v1'
    refit_policy: str = 'declared_sobol4_v1'
    alpha: float = 0.01
    attempted_count: int = 0
    successful_count: int = 0
    statistics: tuple[DiagnosticStatistic, ...] = ()
    tail_count: int | None = None
    tie_count: int | None = None
    observed_score: float | None = None
    null_statistics_sha256: str | None = None
    failure_reasons: tuple[tuple[int, str], ...] = ()
    unavailable_reason: str | None = None
    refit_nfev: int = 0
    refit_discrepancy: tuple[float, float, float] | None = None
    provenance_sha256: str | None = None
    # p_value, rejected, resolution are derived read-only properties.

@dataclass(frozen=True, slots=True)
class DiagnosticRefit:
    unit_vector: np.ndarray | None
    evaluations: tuple[ModelEvaluation, ...]
    nfev: int
    attempted_paths: int
    failure_reason: str | None = None
```

所有数值有限、序列冻结、数组自有只读，pickle 后重建验证。available 必须完整 B 成功、无失败原因、
所有列校准字段齐全且 scale>=1；tail/ties 在合法范围，min(p_j)=p。unavailable 不发布 tail/ties/score/p_j。
失败 replicate 索引唯一；-1 仅表示 observed refit，null 索引小于 attempted_count。
DiagnosticRefit 成功必须有合法 unit 和有效 evaluations；拟合 residual/weighted/objective 必须有限，
完整轴允许现有 ModelEvaluation 契约规定的 mask 外缺测 NaN，但不允许 Inf。
失败必须 unit=None/evaluations=()，保留实际已用 nfev/路径数。

ResidualEvidence 在现有七个字段之后增加 raw_systematic、raw_autocorrelation、advisories、calibration、owner_sha256（均有空默认以支持明确的未执行对象，不是文件兼容解析）。
有校准时有效布尔结论必须与对应 dataset_id 的 p_j 一致。advisories 和 diagnostics 分开：后者仅包含仍阻断的提示/物理告警。

model/provenance.py 增加三个纯 hash 接口：

```python
residual_owner_sha256(problem, candidate, dataset_id=None) -> str
joint_residual_owner_sha256(problems, dataset_ids, unit_vector, local_evaluations, layout_fingerprint) -> str
diagnostic_calibration_sha256(evidence) -> str
```

owner 包含完整 context、objective_point_count、sampling_multipliers、数值 winner 的 unit/mean/residual/objective/noise、
成员顺序及 joint layout；排除 source_path、analysis-derived advisory、warning 与分析阶段标签。
证据 seal 绑定所有校准字段（排除 seal 自身）；构造只校验语法，消费者与 codec 重算 seal，不运行物理。

## 7. 分层组合与复用

fit/diagnostic_refit.py 提供 refit_diagnostic_single(problem,*,cancelled=None) 和
refit_diagnostic_joint(template,members,*,cancelled=None)，返回 DiagnosticRefit。
fit 不导入 analysis，analysis 不导入 fit；service 注入 refit/recompile callable。
analysis/residual_statistics.py 只计算固定列和 symmetric max；analysis/residual_calibration.py 负责 draws/refits/失败/证据。
AnalysisRequest 可携带已存在的不可变 ResidualEvidence；构造/unpickle 只验证 owner，不执行任何拟合。
run_analysis 每个精确 winner 只校准一次，preliminary/report/profile/binary 共用；automatic 快报到最终报传回同一证据。
joint 在 service 所有的单次 operation 内以精确 owner 复用；换数据/mask/normalization/winner/layout/config 必须失效；不加全局缓存。
Profile 的 covariance 与残差资格在 coordinator 计算一次，profile tasks 直接消费已定 options，不在 worker 重启 MC。
Poisson 的 product Bootstrap 正式置信资格也消费 effective 残差结论：拒绝/不可用时保留已算出的
samples、失败率和 quantile，但以 `exploratory_bootstrap`、confidence_level=None 发布。
BootstrapResult 追加 `diagnostic_unavailable_reason: str | None = None` 记录此发布门禁；
原样本数门禁保持，unavailable_reason 先报告样本数/失败率原因，否则报告该诊断原因。
不改变 bootstrap 数值算法、样本数或一般覆盖主张；未触发筛查及校准不拒绝不降级。
直接低层入口若无校准 callable/证据，遇到 Poisson 阳性必须 unavailable，不偷偷使用旧启发式或放行。
_enrich_search_result 不把已否定 advisory 带回下一次物理输入；所有已知 derived code 从物理输入剥离后按当前残差重建。

## 8. 文件、导出和界面

schema_version=4，algorithm_version=xrr-fit-v2-poisson-1，objective_version=2。使用现有顶层版本检查，不新增兼容模块。
FitConfig/budget 和 ResidualEvidence 的新字段在 JSON 中必填；未知/缺失字段、矛盾状态、损坏 seal 明确拒绝。
保存/加载/导出仅展示已保存证据，不触发拟合。JSON/CSV/Excel/log/ORSO 共用 evidence projection。
GUI 显示原 advisory、校准状态/方法/失败原因/MC 计数；“不拒绝”不能写成“模型正确”，不可用不能写成“无系统误差”。

## 9. 验证与科学边界

- 每个实现改动先 RED，再 GREEN；保留真实日志和逐批 source diff，不用预期输出冒充执行。
- 原 seed 10/8 与全 200 cases 完整回归，不删 unavailable；使用当前代码重建输入，旧 JSON 不迁移。
- 独立 null 和 background/footprint/surface/相关计数强阳性规则，在运行前写入外部预登记 manifest 并 hash。
- 检验全零尾、ties、频带缺失、mask/排序、边界/弱识别/多参数、单/联合、失败/预算/取消、serial/parallel、严格 codec/展示。
- 误拒、unavailable、功效、区间 availability、条件覆盖和全 seed 覆盖分别报告；不能只报告数字上涨。
- 固定 0.01 族诊断误拒与区间失覆盖是两笔不同误差预算；不宣称新筛查保证全 seed 95%。
- 最后运行项目 quality/tools/unit/integration/gui/spawn/regression 和完整 Radon；新生产修改独立验收，不借旧 Task 8 的 3699 passed。
- 多参数、联合和新统计覆盖在执行前均未验证。原小样例成本 128/128 暖启动成功；18 组四起点对比 72/72 成功，仅为成本与数值一致性证据。

## 10. 当前实施证据

外部根：/tmp/xrr-v2-poisson-fix-fww0fik3/implementation。
前置备份 pre-fix-source.tar.gz 对应 647 文件 snapshot a9a6e921ffbe4007a6ed09bb4148506377518f6b685c730af57438c84defa2b0。
本轮初始窄回归 42 passed；未提交，旧封存证据/用户改动/.superpowers 全部保留。
