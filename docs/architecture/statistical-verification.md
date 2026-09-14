# 完整统计验收与分片证据

统计验收始终覆盖 canonical 220-case corpus。计算调度与证据控制不改变案例、
随机种子、拟合预算、指标或阈值；**没有显式授权就不启动全量拟合**。

## 默认拒绝计算，先做快速门禁

- `python tools/verify.py statistical` 和无统计输入的 `release` 默认报错，
  在创建报告、执行门禁或分配 worker 前停止。环境变量不能代替授权。
- 有完整分片时，使用 `--statistical-results` 重验原有 outcome，不重新拟合。
- 新计算必须显式传 `--compute-statistical`；单片脚本必须传 `--compute`。
  直接运行 statistical pytest 也必须加载 `tests.statistical_gate` 并明确选择计算。
  计算开关与结果输入互斥，坏输入绝不回退成新拟合。
- `preflight` 执行 `RELEASE_ORDER` 中除 statistical 以外的全部软件门禁，包含
  GUI、distribution、identity。它不是 statistical PASS，也不包含 owner approved-data。

在普通 clean clone 和既有锁定 Python 环境中，先运行：

```sh
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen python tools/verify.py preflight --report-dir /external/preflight --artifact-dir /external/preflight/artifacts
```

## 冻结发行依赖，而不是在长跑末尾追最新版

`distribution` 使用 `tools/locked_closure.py` 验证既有 Windows lock、包清单及完整
依赖闭包。校验真实 wheel SHA-256、METADATA、Windows marker、extras、版本约束和
从 build/runtime/packaging 声明出发的可达性；缺传递依赖、冲突、重复、无关包和输入
竞态都拒绝。下载只使用清单中的固定 URL/hash，`--no-index --no-deps --require-hashes`，
不执行最新版解析或安装应用依赖。

已有 wheel 可离线验证：

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/locked_closure.py --repo-root . --wheel-dir /external/windows-wheels
```

这是明确的控制规则调整：发行验证冻结闭包，不再以“结束时的实时最新版”为通过条件。
`lock_windows_environment.py --check` 仍只是较弱的声明/pin 检查，不能替代闭包证明；
`lock_windows_environment.py --verify requirements-windows-x64-py312.lock` 保留为**显式
实时更新检查**，仍会报告新版漂移。旧漂移失败不因此变成历史 PASS，锁也不会被更新。

## 原始统计证据不变

- `statistical_partition.py` 按既有 optical-work estimate 排序、轮转分配固定 8 片，
  每片保持 canonical 顺序，合计无遗漏、无重复。
- `statistical_shards.py` 使用原 worker initializer、case fitter、CPU 预算与 `spawn`。
  整片完成且源码/运行时未变，才发布 `SHARD_COMPLETE`，它不是全局 PASS。
- 消费者要求恰好 8 个普通分片目录、8 个 manifest 和 220 个 case record，严格检查
  顺序、字段、数值、计数、schema 和逐例哈希。缺失、额外、重复、符号链接、重复
  JSON key、超限或读取期间变化均失败。
- 原 `validate_corpus_outcomes` 全局断言、两个 statistical pytest node ID 和完整
  report 断言保持不变。发布 summary 前再次求值并复验全部 228 个输入文件。

同一源码、运行上下文下重验：

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py statistical --statistical-results /external/statistical-inputs --report-dir /external/statistical-report
```

只有确实需要并获准新计算时，才使用 `--compute-statistical` 或在单片脚本上使用
`--compute --shard-index N`；不要把它们加入默认脚本、重试或缓存未命中的兜底路径。

## 跨 commit/run/attempt：显式 producer 与独立 consumer

没有 `--statistical-producer` 时，保留严格同源码、同 runtime、同 run/attempt 校验。
跨上下文重验必须另给仓库外的 producer descriptor，字段为：

```json
{
  "schema": "xrr-statistical-producer-v1",
  "repository": "owner/repository",
  "run_id": "123456",
  "run_attempt": "1",
  "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_tree": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

以上仅为格式示例，必须替换为真实、明确的 producer；不支持 `latest-success`。

校验器通过只读 GitHub API 验证指定 run/attempt、同仓库来源、受支持的 workflow、
8 个成功计算 job 和成功 aggregate、artifact 所属 run/上传时窗与 GitHub 原始 ZIP
摘要。随后将下载的真实 artifact 字节与全部本地原始分片逐一绑定；descriptor 自报
`PASS` 或拷贝来的 metadata 不能代替上游核验。原 run 的 release 可以失败，但计算
和 aggregate 必须成功。过期、不可访问或来源不明的 artifact 拒绝使用。

计算兼容性另行检查：

- producer 的 Git commit/tree 必须真实存在，6 个原始锁/包输入哈希必须匹配它自己的
  commit；consumer 必须是当前 clean HEAD。
- 全量 `src/**`、synthetic recovery 实现、corpus/断言、相关测试入口、partition、
  record/provenance 协议保持一致。未知变更返回 `NEEDS_NEW_FIT`，但**不会启动拟合**。
- 对 shard 脚本只归一化已审核的显式权限参数/拒绝保护；其余完整 AST 仍比较，不能
  忽略 worker、预算、record 逻辑或未知入口行为。其他 workflow/控制工具变化不冒充
  计算变化，也不把所有 tools/tests 的整体 SHA 当计算指纹。
- 比较全部原始 macOS 锁定包与 refnx 的实际版本、平台、架构和 Python patch，
  不只比较 NumPy/SciPy，也不忽略 pytest、tzdata 或其他包。

重验已下载的原始分片：

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py statistical --statistical-results /external/original-shards --statistical-producer /external/producer.json --report-dir /external/revalidation
```

需要从指定 producer 获取原始分片时：

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/statistical_handoff.py --producer-json "$(cat /external/producer.json)" --report-dir /external/handoff
```

该步骤只声明 `DOWNLOADED`，不是统计 PASS。随后将 `/external/handoff/shards` 与
`/external/handoff/producer.json` 传给上述重验命令。

原始文件的 identity、SHA、run 和 attempt 不改写。跨上下文 summary 使用 v2，分别
记录 `producer_identity`、`consumer_identity`、兼容性/上游证据，以及
`execution=revalidated-existing-outcomes`、`new_fit_count=0`，不声称 consumer 重新算过。
`release` 支持同样两个输入选项，保持原门禁顺序，不接受 aggregate summary 替代原始分片。

## 工作流边界

`statistical.yml` 是只读 reusable workflow。`compute` 默认 false，`producer` 默认空；
两者都未提供时明确 `NOT_READY`，不会跳过后判绿。只有首次 `workflow_dispatch`
明确 `compute=true`，并通过完整 preflight 后，才允许 8 片计算。自动 push/tag 和
rerun 不授权新计算；重试应显式选择先前 producer，不重新拟合或混用多个 attempt。

`hosted-release-verify.yml` 仅接受手动调用，不再有 push 触发器。正式 tag pipeline
缺 producer 时不能通过 statistical/checkpoint；可通过 tag 上的手动调用明确提供证据。
版本/tag 校验、readiness、Windows 前置条件、draft-only 发布权限与 owner 人工公开
Release 边界不变。SBOM composition 仍不能声称 complete，owner approved-data 仍是独立验收。
