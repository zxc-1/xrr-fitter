# Joint bootstrap provenance repair

用户要求“全部修完”，本规格实现剩余的来源校验项；覆盖欠校准在同阶段独立统计规格中处理，不能因为本项完成而关闭总体目标。

## Design

采用两层、不同域的 SHA-256，而不是首成员 single seal 或 residual owner 的别名。

- `joint_owner_sha256` 绑定全部有序成员的完整 `FitEvaluationContext`（仅排除可搬迁 `source_path`）、全部共享/约束声明、global variables、scatter maps、layout fingerprint、candidate ID、联合 unit vector 和全部保存 winner candidates 的数值快照与实际 bootstrap child seed。包含 objective_point_count 和 sampling_multipliers。
- `provenance_sha256` 绑定 candidate ID、joint owner 和 BootstrapResult 的全部内容（samples、intervals、失败索引/原因、attempt count、method、sampling 与 diagnostics 资格）。只排除自身 seal。
- 生成后先验证来源再改变资格；资格变化后重封印；analysis 在独立重算预期 owner 后验证，不信任 callback 交来的同轴/同 E-0 标签。
- pure codec 重算内容 seal；joint project 发布要求所有成员的全局 uncertainty 完全一致（明确排除合法的成员局部 mcmc/sld_bands 附件），并强制有 joint owner，不能删除字段后降格。
- 受支持的 public save/load 用 service_seed_branches(project)[1]、原成员顺序、原 preparation 和 compile_joint_problem 路径重建 context/layout，重建 winner vector，用保存 candidates 重算 owner；不执行 forward、refit 或新 sampling。自动联合流程当前不生成 bootstrap，不从最终改写的 automatic settings 猜历史上下文；此类不支持的 joint evidence 显式拒绝。
- 合法搬迁只忽略 source_path，不忽略源字节、数值数据或布局；不新增依赖、不修改诊断阈值、不处理旧格式。

## Alternatives

1. 仅让 member reports 相同：只能发现分叉，不能发现整体移植，拒绝。
2. 持久化整个编译 context：数据冗余、扩大 schema、还需要验证 context 与实际 source，拒绝。
3. 两层 seal + public 重建：数据最少、真实绑定，采用。SHA 提供确定性来源/完整性检查，不是抵抗可重新计算 hash 的攻击者的数字签名。

## Acceptance

同标签/同轴不同 context、任一成员 data/weight/sampling/config、layout、winner vector/evaluation、样本矩阵/失败/资格任一变动均导致验证失败；成员副本分叉、字段剥除、损坏 pickle/JSON 在相应边界拒绝。真实 joint Gaussian 小样本可 roundtrip，Poisson 资格变更后可 roundtrip；不启动旧4600实验。全部新增行为先 RED 后 GREEN，独立审查后新普通 clone 工程门禁。
