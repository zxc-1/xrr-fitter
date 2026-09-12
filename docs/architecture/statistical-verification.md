# 完整统计验收与分片证据

统计验收始终覆盖 canonical 220-case corpus。分片只改变计算调度，不改变案例、
随机种子、拟合预算、指标或阈值。默认 `python tools/verify.py statistical` 仍重新拟合
完整 corpus；没有环境变量开关或历史结果缓存可以绕过执行。

## 计算与判定分离

- `tools/statistical_partition.py` 按既有 optical-work estimate 排序、轮转分配到固定
  8 片，每片保留 canonical 案例顺序。全部分片构成无遗漏、无重复的精确覆盖。
- `tools/statistical_shards.py` 调用原 worker initializer 和 case fitter，沿用原 CPU
  预算与 `spawn` 多进程策略。每完成一例就写入结果和耗时，并输出进度。
- 只有整片完成且源码、运行时身份未变时，才写入 `result.json`，其状态为
  `SHARD_COMPLETE`，不是全局 `PASS`。失败或中断产生的部分记录不能替代完整分片。
- `tools/statistical_results.py` 收齐全部结果后，通过原 `validate_corpus_outcomes`
  执行三个全局统计断言；原两个 statistical pytest node ID 和完整 report 断言保留。

## 证据边界

每片绑定 clean commit/tree、锁及包清单等 6 个输入文件的 SHA-256、实际 Python
版本、macOS ARM64 平台、普通锁定包版本和 refnx 版本。依赖安装仍经过既有锁定 setup
与包字节验证；统计记录本身不声明 OS/interpreter/native 组成完整性。

GitHub Actions 记录还必须绑定当前 repository、run ID、run attempt，且 `GITHUB_SHA`
与实际 HEAD 相等。不同源码、run 或 attempt 的结果不能混合；重跑时应重跑整个
workflow，不能拿上一 attempt 的成功分片补齐。

消费者要求恰好 8 个普通分片目录、8 个 manifest 和 220 个 case record，校验顺序、
类别字段、数值、计数和每例字节哈希。缺失、额外、重复、符号链接、重复 JSON key、
超限输入或读取期间变化均失败。发布 summary 前再次实际求值并复验输入；summary
绑定全部 228 个输入文件及每例耗时，而不是只引用上游 job 的成功标记。

## 运行方式

正式门禁在普通 clean clone 和项目锁定环境中执行，继续遵守 checkout hygiene。
报告和输入均放在仓库外。单片输出必须是尚不存在的目录，例如：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python tools/statistical_shards.py --shard-index 0 --report-dir /external/statistical-inputs/shard-0
```

独立计算索引 `0` 到 `7` 后，用原 statistical 门禁消费全部分片：

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py statistical --statistical-results /external/statistical-inputs --report-dir /external/statistical-report
```

`--statistical-results` 仅适用于 `statistical` 和 `release`。release 保持原阶段顺序，
在 statistical 阶段重新验证原始分片；不接受 aggregate summary 作为输入替代物。

`.github/workflows/statistical.yml` 是只读 reusable workflow：8 个标准 `macos-15`
计算 job，全部成功后执行 aggregate。正式 tag pipeline 和 hosted rehearsal 共用它；
artifact 名称包含 SHA、run ID 和 attempt，结果与日志分开保存。发布权限、tag/version
校验、Windows 前置条件和 owner 手动公开 Release 的边界不变。
