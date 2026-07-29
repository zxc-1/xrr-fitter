# XRR R23 干净切断架构实施方案

## 0. 2026-07-26 执行覆盖：软件先交付，真实数据由用户交付后验收

本节记录用户在实施开始后的明确裁决，并在与本文后续条款冲突时优先。R23 的当前交付目标
是完成软件架构迁移、自动化回归、统计验收、GUI 功能、打包、CI 和发行；三类真实数据的
最终领域验收由用户在软件交付完成后自行执行，不是创建 R23 repository、任务 0-14、
GitHub Actions、`R23-final` tag 或 GitHub Release 的前置条件。

- 不得用 synthetic fixture 冒充真实数据，也不得把未执行的真实数据验收写成 `PASS`。
- R22 冻结保留已经验证的 canonical 220-case `PASS` 和 GUI Task 10
  `blocked: missing approved dataset` 历史状态；不再要求为了创建 R23 把它改写为
  `accepted: approved real-data complete`。
- R23 必须交付真实数据导入、运行、项目/导出/图表、候选报告和签核冻结工具及其自动化测试；
  当前 release 对真实数据状态只允许如实记录为 `NOT_RUN: owner post-delivery acceptance`。
- 缺少 `/Users/dala/Desktop/xrr-approved-data-r22-final`、candidate report、domain sign-off、
  committed approved-data manifest/records 或 approved-visible runner 时，真实数据专属验证
  不运行；这不是 pytest skip/xfail，也不得产生空成功报告。standard/statistical/GUI/
  distribution/identity/release 门禁不得依赖这些交付后输入。
- 当前 `R23-final` 表示软件发行完成，不表示真实数据领域验收完成。用户后续验收产生的原始
  数据、candidate report 和外部 sign-off 仍不得上传 GitHub；是否形成后续签核提交或发行
  由用户验收时另行决定。
- 本覆盖替代后文所有“真实数据必须在 Task 0/Task 13/Task 14 或 release 前通过”、
  “缺少 approved data 必须阻塞普通 CI/release”以及要求在当前发行身份中绑定尚不存在的
  approved evidence/source tree 的条款。其余架构、TDD、Radon、Git、Actions、禁止兼容层和
  禁止伪造成功的约束继续有效。

执行流程按用户确认的推荐解释固定如下：

1. 使用 `executing-plans`；任务 brief、审查包和进度台账放在仓库外，R23 不创建
   `.superpowers`。
2. 每个将 push 的 commit（包括 Task 11 十个 slice）都在 push 前完成聚焦审查；Task 14
   的任务审查和全分支审查必须在创建不可变 tag/Release 前完成。
3. 只能在 clean commit 上运行的门禁采用
   `commit -> clean-HEAD local gate -> fast-forward push -> exact-SHA Actions GREEN`；中间不做
   其他工作。
4. Task 1 root boundary 与 Task 14 deterministic manifest 等 metadata-only commit 不要求
   虚构行为 RED，改用对应结构测试、确定性生成器或 clean-tree 验证证据。
5. 计划命令中的 `PYTHONDWRITEBYTECODE` 视为拼写错误，执行时统一使用
   `PYTHONDONTWRITEBYTECODE=1`。

> **面向智能体执行者：** 必需的子 Skill：使用 `subagent-driven-development` 或 `executing-plans` 逐项执行本方案。R22 只在原仓库中完成一次本地不可变冻结；R23 从独立仓库的第一个 root commit 开始由 GitHub 控制。每个 R23 实施任务都遵循 RED -> GREEN -> 完整相关验证 -> 本地提交 -> GitHub fast-forward push；任务 2 起还必须等待对应 Actions GREEN。不得在 R22 仓库中实施 R23。

**目标：** R22 标准/统计执行和 GUI Task 10 完成后，创建独立的 R23 树；保留已批准的产品行为，同时用一个可维护的 `src/xrr_fitter` 包、一个公共 API、一棵测试树和覆盖整个仓库的 Radon 门禁替换旧有平铺实现。

**架构：** R23 是一次干净切断。唯一受支持的 Python API 是 `xrr_fitter.api`；唯一应用入口是 `python -m xrr_fitter` 和 `xrr-fitter`。不可变领域模型位于最底层，物理计算和 I/O 依赖模型，拟合与分析保持相互独立，服务层负责组合二者，GUI 调用公共 API。R22 代码仅通过只读的 `R22-final` 标签/归档和归一化参考制品保留。

**技术栈：** Python 3.12、NumPy、SciPy、periodictable、pandas、XlsxWriter、Matplotlib、PySide6、pytest、pytest-qt、Radon 6.0.1、setuptools。

## 1. 状态与执行边界

本文档是实施前方案。当前权威树为：

```text
/Users/dala/Desktop/xrr-rewrite-design-integration
branch: integration-r22
plan-audit snapshot: 4a048e5dbd26ac2624c17bd85ac0a26433849241
state: completed R22 product tree; immutable local freeze metadata is created by task 0
```

上述 commit 只记录本文最后一次只读核查时看到的状态，不是迁移基线。R22 产品工作已经
结束；任务 0 不是继续开发 R22，而是从执行时明确记录的最终 clean commit 生成本地 tag、
archive 和 receipt。不得把 `4a048e5`、`c410e79` 或聊天中的任何旧 HEAD 硬编码成
`R22-final`。

后续实施树固定为：

```text
/Users/dala/Desktop/xrr-rewrite-r23
repository: independent Git repository with a new root commit
branch: r23-clean-architecture
R22 source: local /Users/dala/Desktop/xrr-rewrite-design-integration at R22-final
approved data root: /Users/dala/Desktop/xrr-approved-data-r22-final
GitHub remote: origin
GitHub repository: zxc-1/xrr-fitter
```

只有在以下条件全部满足后，才能创建独立 R23 repository：

- R22 标准/统计运行器已成功完成。
- 220 个案例的验收语料库具备完整、经批准且绑定 hash 的证据。
- GUI Task 10 已通过全部三类经领域负责人批准的真实数据：
  - 已知单层曲线；
  - 当前可正常处理的 Mo/Si 多层曲线；
  - 当前失败或不稳定的多层曲线。
- 每个真实数据输入都有不可变 SHA-256、已记录的随机种子/配置、三次同种子运行、一次新种子运行、已验收项目、导出、图表，以及经批准且绑定 hash 的结论。
- `integration-r22` 不存在无法解释的已跟踪或未跟踪变更。
- 发行身份验证器已针对实际文件和证据通过验证，而不是只验证此前生成的清单。

本方案不得修改、停止、重命名、清理或复用活动中的 R22 运行器目录。
当前核查时 R22 仓库没有 Git remote，也没有 `.github/workflows/`。R22 已完成，但任务 0E
仍负责把该完成状态固化为本地不可变基线；R22 仓库始终不配置 remote，不 push，也不发布
GitHub Release。GitHub 接入只发生在任务 1 创建的独立 R23 repository 中，并且只发布 R23。

### 1.1 当前文件架构诊断

以下数字来自上述 `4a048e5` 快照，只用于说明重构原因，不作为 R23 的文件数量或
LOC 门禁：

- `xrr_fitter/xrr/` 当前有 659 个 Python module、38,114 行；其中 646 个普通 module
  basename 使用单前导下划线，占 98.0%。
- 在这 646 个普通前导下划线 module 中，按 basename 的首个职责词统计，
  `optimize` 232 个、`batch` 99 个、
  `uncertainty` 92 个、`export` 62 个、`project` 60 个、`auto_init` 44 个、
  `physics` 22 个。大量文件同时重复父职责前缀并只承载很窄的 helper。
- 单词之间的 `_` 是 Python 正常 `snake_case`，例如 `joint_problem.py`，不是问题。
  要清理的是把“包内可见性”编码成 646 个 `_module.py`、重复
  `_optimize_*`/`_batch_*` 前缀，以及为压低单函数复杂度形成的微文件链。
- R23 不机械地把这 646 个文件逐个改名。迁移单位是可观察产品合同：同一变更原因、
  同一数据流且能共同通过 Radon 的 helper 合并到一个职责模块；转发层、重复实现和
  只服务旧布局测试的 module 直接删除。目标树是职责地图，不是预先要求全部文件都存在。

因此不在 R22 原地整理。R22 完成本地冻结后，创建没有共享 `.git`、remote 或提交历史的
独立 R23 repository，按领域连同测试一起重写；R22 只通过本地 tag/archive 和绑定 hash 的
参考制品提供来源。这样失败时停止独立 R23 仓库即可，不需要兼容层回滚。

## 2. 不可协商的全局约束

1. 保留产品合同，而不是旧 Python 路径：
   - R22 项目/数据格式；
   - 已批准的数值行为和单位；
   - 确定性随机种子、阶段顺序、取消、检查点和恢复语义；
   - GUI Tasks 1-10 行为；
   - 220 个案例的统计验收；
   - 三类已批准真实数据的结果。
2. 从 R23 删除 `xrr_core.py`、`xrr_app.py`、顶层 `xrr/`、顶层 `gui/` 和 `tests_r21/`。
3. 不得创建 `compat.py`、旧导入垫片、`sys.modules` 别名、`__module__` 改写、通配符导出、条件式新旧导入或双实现。
4. R23 测试或运行时不得导入或启动 R22 代码。
5. R22 项目兼容性由 R23 主项目编解码器直接实现。这是产品要求，不是兜底解析器。
6. 不得捕获意外异常并将其转换为空结果、警告、跳过或表面成功。只在负责该异常的层捕获已有文档说明的领域失败。
7. 已批准的数值恢复阶段必须保留，因为它们属于算法行为。不得将其与禁止的架构/导入兼容兜底混为一谈。
8. 不得添加 DI 容器、服务定位器、通用 `ports` 包、通用 `contracts` 包，或只有一个实现的接口。
9. 不得强制任意文件长度限制。由内聚性、依赖方向、Radon CC 和 Radon MI 决定何时拆分文件。
10. 除有意设置在包边界的 `api.py`、`__init__.py` 和 `__main__.py` 外，不得创建唯一职责只是转发一个符号的文件。
11. 所有项目自有 Python 文件（包括测试和工具）都必须通过同一 Radon 策略，且不得设置永久允许列表。
12. 每个生产领域的迁移必须在同一可审查批次中包含其测试迁移。
13. 所有直接 pytest 调用都必须设置 `PYTHONDONTWRITEBYTECODE=1`、禁用 cacheprovider，并把 basetemp/cache 放到仓库外；优先通过 `tools/verify.py` 执行。
14. R23 不积累只存在于本地的实施提交。任务 1 的首个提交直接发布分支；任务 2 起每个提交
    必须逐个 fast-forward push，并以该精确 commit 的 GitHub Actions `success` 作为下一批
    工作的前置条件。禁止 force push、改写已推送提交，或用没有聚焦 RED/GREEN、未修复根因的
    无关提交掩盖失败 run。

### 2.1 GitHub 交付合同

R22 仓库永远不接入 GitHub。唯一目标固定为 private repository `zxc-1/xrr-fitter`；当前审计
确认该名称尚不存在。任务 1 的首个 R23 commit 和 Gitleaks GREEN 后由已认证账号 `zxc-1`
创建它，不接受调用方覆盖 owner/name，也不改用其他已有 repository。创建中断后的重跑只接受
三种连续状态：目标仍不存在、目标是没有 ref 的 exact private 空仓库，或目标的唯一 ref 已是
本任务的 R23 root commit；其他 owner、visibility、权限、ref 或 object 状态全部失败。

R23 是独立 Git repository；它不共享 R22 的 `.git`、remote、branch、tag 或提交历史。唯一
remote 名为 `origin`，唯一开发分支为 `r23-clean-architecture`。不得向该 remote push
`integration-r22`、`R22-final` 或任何 R22 Release；GitHub 凭据只由
`gh` credential helper 持有；Token、cookie、approved raw data、candidate report、外部
domain signoff 和运行时生成的本机路径配置不得进入 Git、Actions artifact、cache、log 或
GitHub Release。本文作为审计方案显式声明的 workspace 路径不是运行时配置，但不得由工具把
新的 cwd、venv、runner 或用户目录写入制品。
GitHub 只接收已审计的 repository 内容、绑定 hash 的非原始证据、发行包和 canonical receipt。
执行任何 `gh` 或 network Git 命令前都必须拒绝环境中的 `GH_TOKEN`、`GITHUB_TOKEN`、
`GH_ENTERPRISE_TOKEN` 和 `GITHUB_ENTERPRISE_TOKEN` 覆盖，固定 `GH_HOST=github.com`，并要求
`origin` 为精确 canonical HTTPS URL `https://github.com/zxc-1/xrr-fitter`；不接受 SSH、别名
host 或 URL 重写。

任务 2 创建真实 `.github/workflows/verify.yml` 后，从任务 2 到任务 14 的**每个**本地提交
（包括十个 GUI slice 和任何 CI 修复提交）都立即运行下面的唯一分支发布门禁。远端分支必须
精确等于当前提交的父提交，因而一次只能发布一个线性提交；push 后必须核对远端 SHA，并等待
该精确 SHA、`push` event、`r23-clean-architecture` branch 的 workflow run。run 未出现、
被取消、超时或结论不是 `success`，均不得开始下一任务。失败只能先定位原因，再用新的
RED -> GREEN 修复提交前进；不得 amend/rebase 已推送历史或使用任何 force 选项。

任务 1 首次 push 后 GitHub 默认分支就是 `r23-clean-architecture`，但其 root commit 尚未包含
workflow；任务 2 首次引入 workflow 的发布门禁因此不得依赖默认分支预先解析 workflow。下面
统一查询 repository-wide runs，并按
`head_sha + event + branch + path=.github/workflows/verify.yml` 精确筛选；workflow 出现等待上限
为 5 分钟，完成等待上限为 12 小时。网络中断后重跑同一门禁时，远端只允许仍等于 parent，
或已经精确等于本地 HEAD；任何第三种状态都失败。这只是同一 push 的可验证续跑，不接受分叉。

```bash
set -euo pipefail
test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"
test -z "${GH_HOST-}" || test "$GH_HOST" = github.com
export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0
ROOT=/Users/dala/Desktop/xrr-rewrite-r23
GITHUB_REPOSITORY=zxc-1/xrr-fitter
cd "$ROOT"
test "$(git branch --show-current)" = r23-clean-architecture
test "$(git rev-list --max-parents=0 --count HEAD)" -eq 1
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git remote get-url origin)" = "https://github.com/$GITHUB_REPOSITORY"
EXPECTED_REFS=refs/heads/r23-clean-architecture
REMOTE_REFS=$(git ls-remote --refs origin | awk '{print $2}' | LC_ALL=C sort)
test "$REMOTE_REFS" = "$EXPECTED_REFS"
HEAD_COMMIT=$(git rev-parse HEAD)
PARENT_COMMIT=$(git rev-parse HEAD^)
REMOTE_BEFORE=$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')
case "$REMOTE_BEFORE" in
  "$PARENT_COMMIT") git push origin HEAD:refs/heads/r23-clean-architecture ;;
  "$HEAD_COMMIT") : ;;
  *)
    printf 'unexpected remote branch: %s\n' "$REMOTE_BEFORE" >&2
    exit 1
    ;;
esac
REMOTE_COMMIT=$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')
REMOTE_REFS=$(git ls-remote --refs origin | awk '{print $2}' | LC_ALL=C sort)
test "$REMOTE_COMMIT" = "$HEAD_COMMIT"
test "$REMOTE_REFS" = "$EXPECTED_REFS"
RUN_REF=r23-clean-architecture
RUN_ID=
for ATTEMPT in $(seq 1 60); do
  RUN_TOTAL=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq .total_count)
  test "$RUN_TOTAL" -le 100
  RUN_ID=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq '[.workflow_runs[] | select(.path == ".github/workflows/verify.yml")] | if length == 1 then .[0].id elif length == 0 then empty else error("ambiguous verify.yml run") end')
  test -n "$RUN_ID" && break
  sleep 5
done
test -n "$RUN_ID"
STATUS=
for ATTEMPT in $(seq 1 720); do
  STATUS=$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json status --jq .status)
  test "$STATUS" = completed && break
  case "$STATUS" in
    queued|in_progress|requested|waiting|pending) sleep 60 ;;
    *)
      printf 'unexpected run status: %s\n' "$STATUS" >&2
      exit 1
      ;;
  esac
done
test "$STATUS" = completed
test "$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .path)" = .github/workflows/verify.yml
test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json conclusion --jq .conclusion)" = success
test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headSha --jq .headSha)" = "$HEAD_COMMIT"
test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json event --jq .event)" = push
test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headBranch --jq .headBranch)" = "$RUN_REF"
CHECKPOINT=$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json jobs --jq '[.jobs[] | select(.name == "checkpoint") | .conclusion] | if length == 1 then .[0] else "invalid" end')
test "$CHECKPOINT" = success
test "$(git rev-parse HEAD)" = "$HEAD_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

任务 1 是唯一例外：此时 workflow 尚未创建，所以首个 R23 commit 仍必须立即 push 并核对
remote SHA，但不能伪造 Actions GREEN。任务 2 在真实 verifier、测试和 workflow 同批落地后，
第一次执行上述完整发布门禁。GitHub Actions 的跳过仅允许来自 workflow 中明确的事件级 job
或 capability 条件，并必须由唯一 `checkpoint` job 验证“本次哪些 job 必须成功”；pytest 内的
skip/xfail/deselect 仍由第 15 节硬失败。

### 2.2 实施期间的 GitHub 执行节奏

任务 1 的 repository/权限/凭据/泄密扫描 preflight 一旦通过，即视为本次 R23 实施已经取得
按本节合同向该唯一 repository 发布的授权。执行者不得在每个提交后再次停下来询问是否 push，
也不得把一批已经本地 GREEN 的提交留到任务末尾集中上传。固定节奏只有一条：
`focused RED -> focused GREEN -> 本批完整本地门禁 -> 单一 commit -> 立即 fast-forward push ->
精确 SHA Actions/checkpoint GREEN -> 下一批`。push 或 Actions 未完成时，本任务状态仍是
`in progress`，不能标记为完成。

| 阶段 | GitHub 动作 | 继续条件 |
|---|---|---|
| 任务 0A-0F | R22 仓库不配置 remote，不上传 | R22 final commit/tag/archive/receipt 全部仅在本地冻结并回读通过 |
| 任务 1 | 创建独立 R23 repository；首个 root commit 后完成 preflight，只 push `r23-clean-architecture` | 远端只有该 R23 branch，SHA 精确等于本地 HEAD；此时尚无 workflow，不伪造 CI 结果 |
| 任务 2 | verifier、测试和首版 workflow 同一 commit 后立即 push | 该 exact SHA 的唯一 `verify.yml` push run 和 `checkpoint` 为 `success` |
| 任务 3-10、12-13 | 每个任务的单一实施 commit 立即 push | 对应 exact SHA Actions GREEN；失败只用新的聚焦修复 commit 前进 |
| 任务 11 | 十个 GUI slice 各自 commit、各自立即 push | 前一个 slice 的 exact SHA Actions GREEN 后才开始下一个 slice |
| 任务 14 | final-manifest commit 立即 push；随后发布并核验 `R23-final` tag/run/Release | branch/tag runs 均 GREEN，Release 五项资产回读相等，默认分支回读为 `r23-clean-architecture` |

每个访问项目 repository 的 GitHub 命令块都在 shell 内固定
`GITHUB_REPOSITORY=zxc-1/xrr-fitter`；不得从环境、`.env`、
Git config、Actions secret 或参数覆盖。该名称不是凭据，可以进入本方案和仓库文档。GitHub
是受审计交付目标，不是跳过本地 RED/GREEN、Radon、架构测试或真实数据门禁的备份通道。

## 3. 命名规则

- Python 模块、函数、变量和测试文件使用常规 `snake_case`。
- 对真正私有的函数或属性，前导下划线仍然有效。
- 不要仅为了表示包内部模块，就在 R23 模块文件名上使用前导下划线。只有 `xrr_fitter.api` 是受支持的外部 API；其他所有模块按合同均为内部模块。
- 文件名中不得重复父目录名称：
  - 使用 `fit/resume.py`，不用 `_optimize_fit_resume_flow.py`；
  - 使用 `analysis/profiles.py`，不用 `_uncertainty_problem_profile.py`；
  - 使用 `gui/structure/editor.py`，不用 `main_window_structure_editor_tree.py`。
- 永久测试文件名中不得保留开发阶段编号。例如，`test_gui_task8_export.py` 应改为体现行为的 `test_export_dialog.py`。
- 算法 Stage A-E 名称可以保留，因为它们是领域概念，不是开发任务标签。
- 所有 Python package 的 `__init__.py` 必须为 0 bytes；不放重导出、version、
  package docstring 或任何执行逻辑。

## 4. 目标仓库布局

```text
xrr-rewrite-r23/
├── AGENTS.md
├── .gitignore
├── README.md
├── MANIFEST.in
├── pyproject.toml
├── requirements-macos-arm64-py312.lock
├── src/
│   └── xrr_fitter/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── evaluation.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── data.py
│       │   ├── instrument.py
│       │   ├── structure.py
│       │   ├── parameters.py
│       │   ├── fitting.py
│       │   ├── provenance.py
│       │   ├── analysis.py
│       │   ├── project.py
│       │   ├── operations.py
│       │   └── export.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── xy.py
│       │   ├── source.py
│       │   ├── project_codec.py
│       │   ├── export_run.py
│       │   ├── export_tables.py
│       │   ├── export_plots.py
│       │   └── examples.py
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── materials.py
│       │   ├── stack.py
│       │   ├── parratt.py
│       │   ├── resolution.py
│       │   ├── footprint.py
│       │   ├── reflectivity.py
│       │   ├── derivatives.py
│       │   └── sld_profile.py
│       ├── fit/
│       │   ├── __init__.py
│       │   ├── objective.py
│       │   ├── parameters.py
│       │   ├── problem.py
│       │   ├── initialization.py
│       │   ├── screening.py
│       │   ├── candidates.py
│       │   ├── local_search.py
│       │   ├── global_search.py
│       │   ├── stages.py
│       │   ├── pipeline.py
│       │   ├── checkpoint.py
│       │   ├── resume.py
│       │   ├── joint_problem.py
│       │   ├── joint_sharing.py
│       │   ├── joint_evaluation.py
│       │   └── joint_pipeline.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   ├── profiles.py
│       │   ├── binary_profiles.py
│       │   ├── derivatives.py
│       │   ├── bootstrap.py
│       │   ├── mcmc.py
│       │   ├── diagnostics.py
│       │   └── report.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── structures.py
│       │   ├── parameters.py
│       │   ├── projects.py
│       │   ├── fitting.py
│       │   ├── batch.py
│       │   ├── exports.py
│       │   └── workers.py
│       └── gui/
│           ├── __init__.py
│           ├── application.py
│           ├── main_window.py
│           ├── document.py
│           ├── workspace.py
│           ├── accessibility.py
│           ├── project/
│           │   ├── __init__.py
│           │   ├── actions.py
│           │   └── dialogs.py
│           ├── data/
│           │   ├── __init__.py
│           │   ├── panel.py
│           │   ├── import_dialog.py
│           │   └── mask_editor.py
│           ├── structure/
│           │   ├── __init__.py
│           │   ├── panel.py
│           │   ├── editor.py
│           │   └── dialogs.py
│           ├── parameters/
│           │   ├── __init__.py
│           │   ├── panel.py
│           │   ├── table.py
│           │   └── sharing.py
│           ├── fitting/
│           │   ├── __init__.py
│           │   ├── panel.py
│           │   ├── controller.py
│           │   └── progress.py
│           ├── results/
│           │   ├── __init__.py
│           │   ├── panel.py
│           │   ├── candidates.py
│           │   └── uncertainty.py
│           ├── export/
│           │   ├── __init__.py
│           │   └── dialog.py
│           └── plots/
│               ├── __init__.py
│               ├── panel.py
│               ├── reflectivity.py
│               ├── diagnostics.py
│               ├── sld.py
│               └── interactions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── outcome_gate.py
│   ├── unit/test_evaluation.py
│   ├── unit/{model,io,physics,fit,analysis,services,tools}/
│   ├── gui/
│   ├── integration/
│   ├── regression/
│   ├── acceptance/
│   ├── architecture/
│   ├── support/
│   │   ├── __init__.py
│   │   ├── model_cases.py
│   │   ├── recovery_cases.py
│   │   └── processes/{__init__.py,run_fit_worker.py,run_analysis_worker.py}
│   └── fixtures/source/
├── tools/
│   ├── check_radon.py
│   ├── check_hygiene.py
│   ├── collect_test_manifest.py
│   ├── validate_test_ledger.py
│   ├── build_r22_reference.py
│   ├── compare_r22_reference.py
│   ├── freeze_approved_data.py
│   ├── lock_environment.py
│   ├── build_release_spec.py
│   ├── verify.py
│   ├── verify_distribution.py
│   ├── release_identity.py
│   └── reference_groups/
│       ├── __init__.py
│       ├── registry.py
│       ├── model_project.py
│       ├── io.py
│       ├── physics.py
│       ├── fit_compile.py
│       ├── fit_search.py
│       ├── analysis.py
│       ├── services.py
│       └── gui.py
├── verification/
│   ├── release-spec.json
│   ├── r22/
│   │   ├── collections/{tests-active.json,tests-r21.json}
│   │   └── reference/{manifest.json,golden/}
│   ├── r23/tests.json
│   └── approved-data/{manifest.json,records/}
├── examples/
│   ├── single-layer.xy
│   ├── single-layer.xrrproj.json
│   ├── mo-si-periodic.xy
│   └── mo-si-periodic.xrrproj.json
├── .github/workflows/verify.yml
└── docs/
    ├── architecture/{r23-clean-break.md,r22-r23-test-ledger.csv}
    ├── acceptance/
    ├── images/
    ├── algorithm.md
    └── user-guide.md
```

这是职责映射，不是数量配额。当规划文件与相邻文件具有相同的变更原因且合并后仍能通过 Radon 时，可以合并。只有真实职责或复杂度边界有需要时，才新增文件。

由于 setuptools 明确设置 `namespaces = false`，`src/xrr_fitter/**` 下每个作为 Python package 的目录都必须包含空的 `__init__.py`；不得依赖隐式 namespace package。`tests/__init__.py` 同样必须存在，以便验证器通过模块名稳定加载 `tests.outcome_gate`。

## 5. 模块职责

### 5.1 `model`

`model` 包含不可变值和状态转换；它们不执行文件 I/O、不启动 worker、不调用 Qt，也不运行优化器。

- `data.py`：beam、列映射、预处理数据、不可变 mask 和数据源身份值。
- `instrument.py`：几何、波长、分辨率、光斑、缩放、偏移和背景规格。
- `structure.py`：材料、层、梯度、周期块、结构声明和 slab-stack 值。
- `parameters.py`：定义、坐标、边界、锁定、设置、引用和共享规则。
- `fitting.py`：拟合配置/进度、候选项、检查点和单数据集拟合结果；不引用 project 或 operation 值。
- `provenance.py`：对完整不可变 fitting/analysis evidence 进行 canonical identity 编码并绑定 evaluation context；不运行 physics 或优化器。
- `analysis.py`：置信度类别、剖面、bootstrap/MCMC/报告值和分析配置。
- `project.py`：数据集、数据源状态、失效状态、项目根目录和已持久化工作区状态。
- `operations.py`：跨数据集 `ProjectFitResult`、不可变 `OperationEvent` 和
  `OperationError`；可依赖 project/fitting/analysis 值，任何反向依赖都禁止。
- `export.py`：导出清单、文件记录和发布结果值。

除下述显式 model DAG 外，任何模型模块都不得导入其他 R23 包或
反向导入高层 model。`TYPE_CHECKING` 也遵守该图：

```text
fitting -> data + instrument + structure + parameters
provenance -> fitting
analysis -> data + parameters + fitting
project -> data + instrument + structure + parameters + fitting + analysis
operations -> fitting + analysis + project
export -> data + fitting + analysis + project + operations
```

箭头表示左侧可导入右侧；`fitting` 不依赖 `analysis`，
`analysis` 可依赖 fitting-only result，`project` 可依赖 fitting/analysis，
`operations` 才可同时持有 `XrrProject` 和 `ProjectFitResult`。若某字段会要求
任意反向 import，就把该字段移到更高层的值，不用字符串 annotation、
local import 或 `TYPE_CHECKING` 绕环。

### 5.2 `io`

- `xy.py`：字节哈希、列解码、重复角度合并、q 转换和不可变预处理数据构造。
- `source.py`：数据集查找、相对路径解析、数据源状态检查和显式数据源接受。
- `project_codec.py`：唯一权威的 R22 兼容 JSON 编解码器，以及原子项目保存/加载。
- `export_run.py`：防冲突运行命名、部分目录生命周期、原子发布和清单哈希。
- `export_tables.py`：JSON/CSV/XLSX 序列化。
- `export_plots.py`：确定性导出图表。
- `examples.py`：确定性示例生成和重定位验证。

I/O 模块不得导入 `fit`、`analysis`、`services`、`api` 或 `gui`。
唯一例外是 `io.examples` 可调用 `physics.stack` 和 `physics.reflectivity`
生成受审的确定性预测曲线；其他 `io -> physics` 依赖仍禁止。

### 5.3 `physics`

- `materials.py`：化学式验证和依赖波长的 SLD。
- `stack.py`：确定性结构展开和粗糙度验证。
- `parratt.py`：稳定的 Parratt 递推和已批准分支约定。
- `resolution.py`：17 -> 33 -> 65 自适应 Gauss-Hermite convolution。
- `footprint.py`：已批准的光斑模型。
- `reflectivity.py`：完整的缩放/光斑/分辨率/背景组合。
- `derivatives.py`：拟合和分析使用的解析模型导数。
- `sld_profile.py`：物理 SLD 剖面生成。

物理计算只能导入 `model`、NumPy、SciPy 和 periodictable。

### 5.4 `evaluation.py`

`evaluation.py` 是 fit 和 analysis 共用的唯一数值求值边界：负责参数
coordinate 变换、residual/loss/likelihood 组装、共享的权重/惩罚和对 physics
kernel 导数的 Jacobian 链式组合。它只导入 `model`、`physics`、NumPy 和
SciPy，不导入 `fit`、`analysis`、`services`、I/O 或 GUI。只保留真正被两个
领域共享的纯函数；search/stage/profile 策略仍归各自包，不把此模块做成
generic utility bag。

### 5.5 `fit`

- 负责参数编译、目标函数求值、解析 Jacobian、初始化、候选项排序、确定性局部/全局搜索、阶段、取消、检查点、恢复和联合拟合。
- 不构建不确定性报告、不运行 MCMC，也不导入 `analysis`。
- 返回不可变 `model.fitting` 值，以及足以供后续分析使用的不可变求值上下文。
- 用直接的类型化函数和具体数据值替换包含 49 个 callable 的 `FitDatasetOperations` bundle。不得引入另一种操作包替代物。
- 提供纯函数、可安全 pickle 的计算处理器；worker 子进程只向它传入一个具体的 cancellation probe，不引入 operation bag。fit 不负责进程创建、队列或进程间消息封装。

### 5.6 `analysis`

- 负责分类、剖面似然、派生/二元剖面、bootstrap、MCMC、残差诊断和最终不确定性报告。
- 不导入 `fit`。
- 使用不可变模型/求值值和共享物理函数。
- 将剖面盆地挽救决策作为分析结果返回。由 `services.fitting` 决定是否需要再次执行拟合搜索。
- 提供纯函数、可安全 pickle 的计算处理器，但不负责进程创建、队列或进程间消息封装。

### 5.7 `services`

- 负责用例和跨领域编排。
- `datasets.py`：导入、基于 source stem 的确定性 dataset ID 分配、重载、重新链接、
  mask 变更和精确失效。新增 dataset 的 ID 由该 service 从现有 project namespace 分配，
  GUI 不生成 ID；project codec 只保留已持久化 ID，不在 load 时重新编号。
- `structures.py`：结构编辑、氧化物建议和结构证据。
- `parameters.py`：参数验证、共享验证和协调。
- `projects.py`：新建/打开/保存、数据源检查和项目级转换。
- `fitting.py`：单数据集搜索与分析组合。
- `batch.py`：已声明的独立/联合调度和项目发布。
- `exports.py`：导出编排。
- `workers.py`：拟合和分析处理器共用的进程创建、spawn 上下文、队列、取消、进度、请求/结果封装、`OperationJob` 以及 `start_fit_job`/`start_mcmc_job` 的唯一负责人。

`load_project` 和 `save_project` 由 `services.projects` 在发布 workspace 前关联 source
validation 与实际解析快照：最多执行两次有界重验证，校验持久化 mask 的派生长度，且按
independent/joint 影响范围清除非 `ok` source 的 evidence、scale prior、result、checkpoint
和 candidate selection。`save_project` 在 Save As 时由 service 重定位相对 source
declaration；同一 base directory 的普通保存逐字节保留原 declaration。GUI 不复制这些
路径、竞态或失效规则。`accept_source_update` 在提交 preview 绑定的字节时只保留仍能通过
当前 parameter definitions 验证的首个同名 setting；GUI 只比较提交前后 setting 名称并展示
移除项。

`services.fitting` 是唯一组合拟合与分析的模块。`fit` 和 `analysis` 绝不相互导入，包括局部导入或 `TYPE_CHECKING` 导入。
module-level allowlist 进一步强制：只有 `services.fitting` 可同时导入
`fit` 和 `analysis`；`services.batch` 只通过 `services.fitting` 组合分析，不直接
导入 `analysis`；`services.workers` 只启动 `services.fitting` 提供的顶层
pickle-safe handler，不直接导入 `fit`/`analysis`；其他 services 都不导入这两包。

### 5.8 `gui`

- `MainWindow` 组合具体 widget/controller，不负责任何拟合/领域算法。
- `ProjectDocument` 以显式类型负责当前不可变项目、路径、脏状态、活动数据集/候选项和运行中作业状态。
- 不使用 mixin 层次结构。每个具体 widget 只继承其 Qt 基类。
- Qt 信号处理器调用服务/API 操作并渲染返回的状态。不得重复失效或协调逻辑。
- 多进程由 `services.workers` 负责；GUI controller 只负责启动/取消/进度/结果生命周期。
- GUI 生产模块只能从 `xrr_fitter.api` 导入领域功能。

## 6. 强制依赖图

```text
model
  ^
  +-- io
  +-- physics
  +-- evaluation -> physics
  +-- fit ------> physics + evaluation
  +-- analysis -> physics + evaluation

services -> model + io + physics + fit + analysis
api -> model + services
gui -> api
__main__ -> gui
```

内部 import 使用穷尽式 package-level allowlist，而不是只列几条已知禁止边：

```text
model -> model
io -> io + model
physics -> physics + model
evaluation -> model + physics
fit -> fit + model + physics + evaluation
analysis -> analysis + model + physics + evaluation
services -> services + model + io + physics + fit + analysis
api -> model + services
gui -> gui + api
__main__ -> gui
__init__ -> no internal imports
```

package-level `model -> model` 之内还必须满足 5.1 节的 module-level DAG；
`services` 还必须满足 5.7 节的 module-level composition allowlist。
`tests/architecture/test_dependency_rules.py` 同时对这三层规则做穷尽验证。

除以上边和标准库/已声明第三方依赖外，其他内部 import 一律失败。因此 `fit <-> analysis`、`fit|analysis -> io|services|api|gui`、`services -> api|gui`、`api -> io|physics|evaluation|fit|analysis|gui`、`__main__ -> model|io|physics|evaluation|fit|analysis|services|api` 均没有后门。任何 R23 module 导入 `xrr`、顶层 `gui`、`xrr_core` 或 `xrr_app` 同样失败。

生产树第三方 import 使用 exact-root allowlist。允许的 root **只有** `numpy`、
`scipy`、`periodictable`、`pandas`、`xlsxwriter`、`matplotlib`、`PySide6`；AST checker
用 `sys.stdlib_module_names` 区分标准库，用完整 root `xrr_fitter` 识别内部 import，
其余非标准库且非内部 root 一律失败，不能仅维护已知 denylist。模块归属进一步固定为：
`numpy` 仅允许 `model`、`io`、`physics`、顶层 `evaluation.py`、`fit`、`analysis`
和 `services.datasets`；`scipy` 仅允许 `physics`、顶层 `evaluation.py`、`fit`、
`analysis`；`periodictable` 仅允许 `physics.materials`；`pandas` 和 `xlsxwriter` 仅允许
`io.export_tables`；`matplotlib` 仅允许 `io.export_plots` 和 `gui.plots`；`PySide6`
仅允许 `gui`。因此 `pytest`、`pytestqt`、`refnx`、`openpyxl`、`radon`、`build`
以及任何以后意外出现的 root 在 `src/` 中都会失败；test/tool dependency 不获得
production 豁免。

package DAG 的唯一具名例外是 `io.examples -> physics.stack` 和
`io.examples -> physics.reflectivity`，只用于从固定 model 声明生成示例预测曲线；
checker 仍拒绝该 source module 的其他 physics target 以及任何其他 I/O module 的该依赖。

进程创建同样采用穷尽规则：除 `services.workers` 外，`src/` 中任何
`multiprocessing` import 都失败；唯一特例是 `__main__.py` 中 exact AST 形式
`from multiprocessing import freeze_support`，且该 module 不得导入或引用任何其他
`multiprocessing` 名称。`src/` 全局禁止 `subprocess` import、
`concurrent.futures.ProcessPoolExecutor`、`asyncio.create_subprocess_exec`、
`asyncio.create_subprocess_shell`，以及 `os.fork`、`os.forkpty`、`os.posix_spawn`、
`os.posix_spawnp`、`os.system`、`os.popen`、`os.spawnl`、`os.spawnle`、`os.spawnlp`、
`os.spawnlpe`、`os.spawnv`、`os.spawnve`、`os.spawnvp`、`os.spawnvpe`。所有 fit、
analysis 和 GUI 路径只能调用 `services.workers`，不能换用另一套 executor、shell 或
spawn 旁路。

`tests/architecture/test_dependency_rules.py` 解析 top-level import、local import 和
`TYPE_CHECKING` import，并解析 `import ... as ...`、`from ... import ... as ...` 后的
绑定名，使 alias 不能绕过 call/reference 检查。将 import 延迟到函数内部不能规避此规则；
测试必须为 package DAG、七个第三方 exact root、未知第三方 root、
`services.workers`/`__main__` 两个进程例外和上述每类进程旁路分别构造合法或非法 fixture，
并验证同 package import 不被误报。
所有 `src/` 还禁止通过 `__import__`、`importlib.import_module`、`exec` 或 `eval`
构造 dynamic import；这四类方式各有独立负向 fixture，不允许引入第二套
可绕过 AST allowlist 的 plugin/discovery 通道。

## 7. 公共 API

只支持以下 import 方式：

```python
from xrr_fitter.api import XrrProject, fit_project, load_project
```

`xrr_fitter.api.__all__` 必须显式定义，且只包含：

```text
BeamSpec
DataColumnMapping
DatasetProject
ExportManifest
FitConfig
FitProgress
FitReadiness
FitResult
GradientLayerSpec
InstrumentSpec
LayerSpec
MaterialSpec
McmcConfig
McmcReport
OperationError
OperationEvent
OperationJob
OxideDecision
OxideSuggestion
ParameterDefinition
ParameterProfile
ParameterReference
ParameterSetting
PeriodicBlock
PreparedData
ProjectFitResult
ProjectUiState
ProjectValidation
ScalePriorState
SharingRule
SourceUpdatePreview
StructureEvidence
StructureSpec
UncertaintyReport
ValidationIssue
XrrProject
accept_oxide_suggestion
accept_source_update
add_dataset
analyze_structure
clear_fit_results
describe_parameters
export_result
fit_project
import_data
inspect_sources
load_project
new_project
preflight_fit
preview_source_update
record_oxide_decision
remove_dataset
run_mcmc
save_project
select_active_dataset
select_candidate
set_batch_mode
set_expert_mode
set_fit_mask
set_instrument
set_parameter_settings
set_sharing_rules
set_structure
set_workspace_state
start_fit_job
start_mcmc_job
suggest_oxide_layers
validate_parameter_settings
validate_sharing_rules
validate_structure
```

关键 service signature 固定为以下接口声明：

```text
import_data(path: str | Path, beam: BeamSpec, import_angle_offset_deg: float = 0.0, column_mapping: DataColumnMapping | None = None) -> PreparedData
new_project() -> XrrProject
add_dataset(project: XrrProject, source_path: str | Path, instrument: InstrumentSpec, display_name: str | None = None, column_mapping: DataColumnMapping | None = None, import_angle_offset_deg: float = 0.0, beam: BeamSpec | None = None) -> XrrProject
remove_dataset(project: XrrProject, dataset_id: str) -> XrrProject
preview_source_update(project: XrrProject, dataset_id: str, new_path: str | Path | None = None) -> SourceUpdatePreview
accept_source_update(project: XrrProject, preview: SourceUpdatePreview) -> XrrProject
load_project(path: str | Path) -> XrrProject
save_project(project: XrrProject, path: str | Path) -> None
inspect_sources(project: XrrProject) -> ProjectValidation
validate_structure(structure: StructureSpec, beam: BeamSpec) -> None
analyze_structure(project: XrrProject, dataset_id: str) -> StructureEvidence
suggest_oxide_layers(structure: StructureSpec) -> tuple[OxideSuggestion, ...]
set_fit_mask(project: XrrProject, dataset_id: str, mask: numpy.ndarray) -> XrrProject
set_instrument(project: XrrProject, dataset_id: str, instrument: InstrumentSpec) -> XrrProject
set_structure(project: XrrProject, dataset_id: str, structure: StructureSpec) -> XrrProject
record_oxide_decision(project: XrrProject, dataset_id: str, decision: OxideDecision) -> XrrProject
accept_oxide_suggestion(project: XrrProject, dataset_id: str, suggestion: OxideSuggestion) -> XrrProject
describe_parameters(project: XrrProject, dataset_id: str) -> tuple[ParameterDefinition, ...]
validate_parameter_settings(definitions: Sequence[ParameterDefinition], settings: Sequence[ParameterSetting]) -> tuple[ParameterSetting, ...]
set_parameter_settings(project: XrrProject, dataset_id: str, settings: Sequence[ParameterSetting]) -> XrrProject
validate_sharing_rules(project: XrrProject, rules: Sequence[SharingRule]) -> tuple[SharingRule, ...]
set_sharing_rules(project: XrrProject, rules: Sequence[SharingRule]) -> XrrProject
set_batch_mode(project: XrrProject, mode: Literal["independent", "joint"]) -> XrrProject
select_active_dataset(project: XrrProject, dataset_id: str | None) -> XrrProject
select_candidate(project: XrrProject, dataset_id: str, candidate_id: str | None) -> XrrProject
set_expert_mode(project: XrrProject, enabled: bool) -> XrrProject
set_workspace_state(project: XrrProject, state: ProjectUiState) -> XrrProject
clear_fit_results(project: XrrProject, dataset_ids: Sequence[str]) -> XrrProject
preflight_fit(project: XrrProject) -> FitReadiness
fit_project(project: XrrProject, progress_callback: Callable[[FitProgress], None] | None = None, checkpoint_callback: Callable[[XrrProject], None] | None = None) -> ProjectFitResult
run_mcmc(project: XrrProject, dataset_id: str, candidate_id: str, config: McmcConfig, progress_callback: Callable[[FitProgress], None] | None = None) -> XrrProject
start_fit_job(project: XrrProject, checkpoint_path: str | Path | None = None) -> OperationJob
start_mcmc_job(project: XrrProject, dataset_id: str, candidate_id: str, config: McmcConfig) -> OperationJob
export_result(result: XrrProject | ProjectFitResult, output_dir: str | Path) -> ExportManifest
```

`OperationJob` 是 `services.workers` 中唯一的具体 worker/process 实现，不定义 Protocol、抽象基类或第二套 process wrapper。GUI controller 可以用 `QTimer` 调用其公共方法并投射为 Qt signal，但不得复制任何 worker 状态机。其稳定接口固定为：

```text
OperationJob.pid: int
OperationJob.is_running: bool
OperationJob.poll() -> tuple[OperationEvent, ...]
OperationJob.cancel() -> None
OperationJob.force_stop() -> None
OperationJob.close() -> None
```

`OperationEvent` 是不可变 tagged value，字段固定为：

```text
OperationEvent.sequence: int
OperationEvent.kind: Literal["progress", "checkpoint", "fit_result", "mcmc_result", "cancelled", "error", "stopped"]
OperationEvent.progress: FitProgress | None
OperationEvent.checkpoint: XrrProject | None
OperationEvent.fit_result: ProjectFitResult | None
OperationEvent.mcmc_result: XrrProject | None
OperationEvent.cancellation: str | None
OperationEvent.error: OperationError | None
```

`OperationError` 固定包含 `exception_type: str`、`message: str` 和 `traceback: str`。`sequence` 从 0 开始严格递增。除 `stopped` 的全部 payload 字段均为 `None` 外，每个 event 必须且只能有一个与 `kind` 对应的非 `None` payload；`cancelled` 使用非空 `cancellation` reason。不得把 exception object、process handle 或 queue 放进 event。

`poll()` 必须非阻塞。每个 job 恰好产生一个 terminal event（`fit_result`、`mcmc_result`、`cancelled` 或 `error`）和一个后续 `stopped` event；只有收到合法 `stopped` 且子进程已回收后，`is_running` 才变为 false。spawn 失败直接抛出，cooperative cancellation 产生 `cancelled`，queue/protocol/计算失败产生明确的 `error`，三者不得互相伪装。`force_stop()` 启动由 job 自己管理的 terminate/kill 升级流程，后续 `poll()` 推进状态；GUI 不得操作底层 process、queue 或 event。`close()` 只释放已停止 job 的句柄，对仍运行的 job 调用必须失败。

不保留 `validate_project`，因为旧 facade 仅将该名称用于 source inspection，而 project 层将其用于 schema validation。R23 采用无歧义名称 `inspect_sources`。

`validate_sharing_rules` 执行纯 declaration validation，绝不读取 source file。`preflight_fit` 执行显式 source loading 和 compilation check。GUI 绝不从 sharing-combo change handler 执行昂贵的 preflight。

`add_dataset` 使用 source stem 作为首个 ID；发生碰撞时依次分配 `stem-2`、`stem-3`。
同一 project 内已加载和本次会话新增的 dataset 共享一个 namespace。`display_name=None`
时使用 source stem 作为显示名，但显示名不参与 ID 唯一性；GUI 只提交 source/instrument，
不得调用或复制 ID allocator。GUI 的显式 beam 选择通过 `add_dataset(..., beam=...)`
提交；`beam=None` 只为非 GUI 调用保留 monochromatic 默认值，service 不得覆盖显式的
mixed Kα `BeamSpec`。

每个 GUI mutation 都映射到一个显式 API use case：

| GUI 操作 | API operation |
|---|---|
| 新建/打开/保存/另存为 | `new_project`, `load_project`, `save_project` |
| 导入/移除 dataset | `add_dataset`, `remove_dataset` |
| 重载/relink/接受已变更 source | `preview_source_update`，然后 `accept_source_update` |
| 编辑 mask | `set_fit_mask` |
| 编辑 instrument/geometry/resolution/footprint | `set_instrument` |
| 编辑 structure | `set_structure` |
| 接受/拒绝 oxide proposal | `accept_oxide_suggestion`, `record_oxide_decision` |
| 编辑 parameter bound/lock | `set_parameter_settings` |
| 编辑 sharing | `validate_sharing_rules`，然后 `set_sharing_rules` |
| independent/joint mode | `set_batch_mode` |
| active dataset/candidate | `select_active_dataset`, `select_candidate` |
| expert mode/workspace layout | `set_expert_mode`, `set_workspace_state` |
| 显式清除 result | `clear_fit_results` |
| readiness/同步无界面拟合 | `preflight_fit`, `fit_project` |
| GUI 启动/轮询/取消/checkpoint/强制停止 | `start_fit_job`, `OperationJob.poll`, `OperationJob.cancel`, `OperationJob.force_stop` |
| 同步 MCMC/GUI MCMC/export | `run_mcmc`, `start_mcmc_job`, `export_result` |

`tests/architecture/test_public_api.py` 固化此 use-case map，并验证 GUI domain call 解析到 `api.__all__` 中的名称。任何 GUI signal handler 都不得直接替换 project dataclass。

## 8. 打包与入口

`pyproject.toml` 的 **R23 最终状态** 使用标准 src layout；其中
`[project.gui-scripts]` 按任务 11 的时序加入，任务 2 不预先写入：

```toml
[build-system]
requires = ["setuptools==75.8.2", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"

[project]
name = "xrr-fitter"
version = "0.2.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.0,<3",
  "scipy>=1.14,<2",
  "periodictable>=2.0,<3",
  "pandas>=2.2,<3",
  "xlsxwriter>=3.2,<4",
  "matplotlib>=3.9,<4",
  "PySide6>=6.8,<7",
]

[project.optional-dependencies]
test = [
  "pytest>=8.3,<9",
  "pytest-qt>=4.4,<5",
  "openpyxl>=3.1,<4",
  "radon==6.0.1",
  "build>=1.2,<2",
  "refnx @ git+https://github.com/refnx/refnx.git@3d3808f66a14a8200eba020f8dff53f4d1e059bc",
]

[project.gui-scripts]
xrr-fitter = "xrr_fitter.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]
include = ["xrr_fitter*"]
namespaces = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-config --strict-markers -ra"
markers = [
  "slow: long-running statistical or recovery verification",
  "approved_data: requires the approved, hash-bound approved-data manifest",
  "visible_gui: requires a visible desktop session",
  "spawn: starts real multiprocessing spawn workers",
]
```

`refnx` 仍仅用于测试。R23 继续使用 `pip`；不引入或切换 package manager。

发行包内容采用两份精确 allowlist：

- wheel 只包含 `xrr_fitter/**`、明确声明的 package resource 和 distribution metadata；
  不得包含 `tests/`、`tools/`、root `examples/`、`docs/`、`verification/`、lock、CI 或
  acceptance 原始数据。当前 R22 没有 license 文件，本架构迁移不得擅自选择或生成
  license；后续如确定授权条款，必须作为单独审阅的产品元数据变更。
- sdist 通过审计过的 `MANIFEST.in` 显式包含 `src/`、`tests/`、`tools/`、`examples/`、
  `docs/`、`verification/`、`requirements-macos-arm64-py312.lock`、`pyproject.toml`、
  `README.md` 和 `MANIFEST.in`；staging **输入**显式拒绝 cache、venv、build output、
  预存 egg-info、外部 approved raw data 和临时 report。setuptools 生成的 sdist **输出**
必然包含 root `PKG-INFO`、root `setup.cfg` 以及
  `src/xrr_fitter.egg-info/{PKG-INFO,SOURCES.txt,dependency_links.txt,entry_points.txt,requires.txt,top_level.txt}`；
  这组 pinned-build metadata 是 sdist allowlist 的显式组成部分，不等于允许工作树残留
  egg-info。若实际 pinned setuptools 输出集合变化，member-list test 失败并要求审阅策略，
  不能用 `*.egg-info/**` 宽泛放行。

Task 2 用与正式构建完全相同的 `setuptools==75.8.2`、`wheel==0.45.1`
和最小 fixture project 实际构建两次 sdist，将上述生成 metadata 的精确 path 集合
固化到 `verification/release-spec.json`并要求两次相同。正式 sdist 的其余成员由
已跟踪输入 manifest 精确派生；不允许凭记忆增加宽泛 metadata glob。

`tests/architecture/test_distribution.py` 和 `tools/verify_distribution.py` 分别枚举 wheel/sdist member 并与上述 allowlist 比对；不得只检查“能安装”。新增资源必须先更新显式配置、内容测试和 release identity，不能依赖 setuptools 自动猜测。

lock 工作流固定如下：

1. `tools/lock_environment.py` 在仓库外部的 `tempfile.TemporaryDirectory` 中创建 resolver venv。
2. 精确安装 `pip==26.1.2`。
3. 使用 TOML parser 从 `pyproject.toml` 读取 `[build-system].requires`、runtime 和 test dependency array，安装这些规格但不安装本地项目，并将 `pip freeze --exclude-editable` 写入 `requirements-macos-arm64-py312.lock`。
4. 同一工具提供 `--check LOCK_PATH`，以 UTF-8 严格读取并用 requirement parser 验证每个
   非空行；文件缺失、不是 regular file、不可读、解码/解析失败、重复或未按 canonical
   顺序排列，以及包含 `-e`、本地 `file:` URL、绝对路径或未固定 VCS revision 时均返回
   非零。普通项必须为 `name==version`；direct reference 必须使用 HTTPS Git URL 和 40 位
   lowercase hex commit，且 normalized name、URL 和 commit 必须与 `pyproject.toml` 中
   显式声明的唯一 direct VCS test dependency 逐值相等。当前唯一合法行是
   `refnx @ git+https://github.com/refnx/refnx.git@3d3808f66a14a8200eba020f8dff53f4d1e059bc`；
   任何额外 VCS/URL、branch/tag/短 commit 或不同 URL/commit 都
   失败。调用方不得用 `! rg` 代替，因为搜索工具自身错误不能被反转成成功。
5. 在 `verification/release-spec.json` 中记录 lock SHA-256。这是“精确版本，且
   lock 文件 SHA-256 受发行身份绑定”的锁，不声称它是 pip
   `--require-hashes` 所需的逐 wheel/sdist 下载哈希锁。
6. 每个迁移、测试和 CI 环境只安装 lock，不 editable-install 本地项目。`tools/verify.py` 用 `Path(__file__).resolve().parents[1]` 解析当前 repository root，清除调用方 `PYTHONPATH` 后精确设置为该 root 下的 `src/`；只有 distribution smoke 才用 `pip install --no-deps` 安装已构建 wheel。任何 wheel/sdist build 都在仓库外 staging copy 使用 `--no-isolation`，不得让 build isolation 在 lock 外解析或下载 dependency。

生成命令：

```bash
python3.12 tools/lock_environment.py --output requirements-macos-arm64-py312.lock
```

可复现环境安装命令：

```bash
python3.12 -m venv /Users/dala/Desktop/xrr-r23-venv && /Users/dala/Desktop/xrr-r23-venv/bin/python -m pip install pip==26.1.2 && /Users/dala/Desktop/xrr-r23-venv/bin/python -m pip install -r requirements-macos-arm64-py312.lock
```

`src/xrr_fitter/__main__.py` 在导入 PySide6 或任何 GUI module 之前调用 `multiprocessing.freeze_support()`。

### 8.1 最终发行身份

`identity`/`release` mode 在仓库外 `report-dir/release-identity.json` 写唯一 canonical
schema；仓库内不保存第二份 identity，也不把 identity 自身 hash 写回自身：

```text
ArtifactManifest
  schema: Literal["xrr-r23-artifact-manifest-v1"]
  status: Literal["PASS"]
  head_commit: FullGitCommit40
  head_tree: GitObjectId
  artifacts: tuple[ArtifactRecord, ArtifactRecord]

R23ReleaseIdentity
  schema: Literal["xrr-r23-release-identity-v1"]
  status: Literal["PASS"]
  head_commit: FullGitCommit40
  head_tree: GitObjectId
  release_spec: RepoFileRecord
  dependency_lock: RepoFileRecord
  r22_oracle_tree_sha256: SHA256Hex64
  test_manifest: TestManifestBinding
  approved_data: ApprovedDataBinding
  artifact_manifest: ExternalFileRecord
  artifacts: tuple[ArtifactRecord, ArtifactRecord]

RepoFileRecord
  path: normalized repository-relative POSIX path
  size: positive integer
  sha256: SHA256Hex64

TestManifestBinding
  file: RepoFileRecord(path="verification/r23/tests.json")
  source_commit: FullGitCommit40
  collection_hash: SHA256Hex64

ApprovedDataBinding
  manifest: RepoFileRecord(path="verification/approved-data/manifest.json")
  approved_evidence_tree_sha256: SHA256Hex64
  approved_source_tree_sha256: SHA256Hex64
  candidate_report_sha256: SHA256Hex64
  domain_signoff_sha256: SHA256Hex64

ExternalFileRecord
  path: Literal["artifact-manifest.json"]
  size: positive integer
  sha256: SHA256Hex64

ArtifactRecord
  kind: Literal["wheel", "sdist"]
  path: normalized string formed as `artifacts/` plus the exact filename
  filename: basename equal to path basename
  size: positive integer
  sha256: SHA256Hex64
```

`release_spec.path` 固定为 `verification/release-spec.json`，`dependency_lock.path` 固定为
`requirements-macos-arm64-py312.lock`；`artifacts` 按 `kind` 排序且 wheel/sdist 各恰好一项。
`artifact-manifest` 的物理 parent 必须与 `artifact-dir` 的 parent 相同，且
`artifact-dir.name == "artifacts"`；identity 的 `report-dir` 可以是另一个外部目录，但
manifest/artifacts bundle 本身不能 flatten、rename 或拆开。
Artifact manifest 与 release identity 中的 head commit/tree 和两个 `ArtifactRecord` 必须
逐值相等；两份 JSON 都遵守下述 canonical JSON 规则。
JSON 使用 UTF-8、sorted keys、无 NaN/Infinity、紧凑分隔符和单个末尾换行；duplicate key、
unknown/missing/extra field、非 canonical path/order/value 或自引用字段一律失败。

三个 owner 的边界固定且不可复制：

- `tools/verify_distribution.py` 唯一拥有 `ArtifactManifest` 的 strict parser、纯计算器和
  canonical/atomic writer，并从两个实际 artifact 重算记录；
- `tools/freeze_approved_data.py` 唯一拥有 approved manifest/record strict parser、
  candidate/signoff 无损投影重建、evidence/source tree hash 计算；
- `tools/release_identity.py` 只导入并组合上述 pure API，再加入 Git object、release spec、
  lock、R22 oracle 和 test manifest，拥有 `R23ReleaseIdentity` strict parser/calculator 以及
  `build`/`validate` 两个 CLI subcommand。它不得复制 artifact、approved-data 的 JSON
  parsing、canonicalization、path validation 或 hash 实现。

`tools/release_identity.py validate` 从当前 Git object/filesystem、两个 owner 的新鲜计算结果
和两个实际 artifact 重算 identity，再与待验 identity 逐字段比较；`tools/verify.py identity`
和 `release` 只能调用这个单一 CLI/module，不能各自拼另一份 schema。Task 11 先用
`tests/unit/tools/test_verify_distribution.py` 固化 `ArtifactManifest` owner；Task 13 在
`freeze_approved_data.py` GREEN 后，再用 fixture-repo `test_release_identity.py` 固化组合层，
对每个顶层/嵌套字段逐个做 missing/extra/tamper，并覆盖 duplicate key、artifact
多/少/换目录、source commit 到 HEAD 的 test-tree drift、approved evidence/raw source/
candidate/signoff drift、tag 类型/指向和 atomic identity/freeze-receipt success/failure。
Task 14 只调用已测试能力，不临时改变 schema 或 hash 规则。

## 9. Radon 策略

Radon 没有定义通用的项目通过线。R23 明确定义以下策略。

### 9.1 范围

`tools/check_radon.py` 首先遍历仓库文件系统，而不是 Git 的 non-ignored view。它枚举仓库下每个 `*.py`，并且只剪除生成内容或非源码根目录 `.git/`、`build/`、`dist/`、`*.egg-info/`、`.pytest_cache/` 和 `__pycache__/`。开发 virtual environment 位于仓库外部。

每个发现的 Python source 必须位于以下任一受管 root 之下；R23 不允许 root-level Python script：

```text
src/
tests/
tools/
examples/
```

上述 root 之外的 Python 文件无法通过 ownership validation。checker 随后对每个受管 Python path 运行 `git check-ignore`；被 ignore 的受管 Python 文件仍会被扫描，同时使门禁失败。最后，它将 filesystem set 与以下命令结果交叉核对：

```bash
git ls-files -co --exclude-standard -- '*.py'
```

从而确保 tracked、untracked、ignored 和位置错误的项目 Python 文件都不会从报告中消失。

范围包括 production module、测试、`conftest.py`、builder、subprocess runner，以及
`tools/`、`tests/`、`examples/` 受管 root 内的 quality/release/comparison script。
不得设置按文件 suppression 或历史 allowlist；repository root 本身仍不得出现 Python
script。

### 9.2 硬门禁

- 精确要求 Radon `6.0.1`。
- 使用 Radon 默认行为统计 assertion；测试不得使用 `--no-assert`。
- 包括 nested function 和 closure。
- 每个报告的 function、method、class 和 closure 均须满足 CC <= 10，rank 为 A 或 B。
- 每个文件的平均 CC <= 5.0，rank 为 A。
- 整个仓库的平均 CC <= 5.0，rank 为 A。
- 每个 module 使用 `mi_visit(source, multi=True)` 计算，MI rank 必须为 A。
- 任何文件不可读、decode failure、syntax error 或 Radon exception 都使门禁失败。
- discovery set 为空时，门禁失败。

“单 block 最高 B 且文件平均 A”规则允许偶尔存在内聚的 orchestration function，同时防止分支过多的巨型 method 和人为制造的单 function 微文件。

checker 使用完整 Radon block list，其中 class block、其 method 和暴露的 inner closure 都恰好计入一次。没有 callable/class block 的文件，其 CC 平均值为 `0.0`。只有 `mi_rank(value) == "A"` 时 MI 才通过。

### 9.3 报告

checker 输出简洁的失败信息，包含 path、symbol、line、numeric score 和 rank。它还可以写入包含 CC、MI、Halstead 和 raw metrics 的 CI JSON artifact。Halstead/raw value 只报告，不人为设定通过等级。

### 9.4 为什么必须使用包装检查器

即使输出 F-rank block，Radon CLI 仍返回 exit code `0`。因此 CI 运行：

```bash
/Users/dala/Desktop/xrr-r23-venv/bin/python tools/check_radon.py
```

并依赖 wrapper 显式返回的非零结果，而不是依赖对 `radon cc` 输出进行过滤。

### 9.5 检查器测试

`tests/unit/tools/test_check_radon.py` 创建小型临时 Git 仓库并验证：

- 平均 A、最高 B 的 sample 通过；
- CC 11 的 sample 失败；
- 单文件平均值高于 5 时失败；
- MI 为 B/C 的 sample 失败；
- nested closure 违规时失败；
- class block 与其 method 各恰好计入一次，不因 Radon 嵌套结构重复计数；
- assertion complexity 过高的测试文件失败；
- untracked 且未 ignored 的 Python 文件会被扫描；
- 受管 root 下被 ignored 的 Python 文件会被扫描，且 `ignore-policy` validation 失败；
- 受管 root 外的 Python 文件 ownership validation 失败；
- `src/`、`tests/`、`tools/`、`examples/` 各有至少一个 Python fixture，四者都出现在 JSON report；
- 多文件 fixture 的 repository aggregate CC 计算与 Radon block 算术平均精确一致，超过 5.0 时失败；
- 剪除生成目录后 discovery 为空时失败；
- syntax 和 decode failure 均失败；
- version mismatch 失败；
- JSON 输出包含每个发现的 path。

受管 root 内被 ignore 的 Python 文件报告失败原因为 `ignore-policy`；只有
位于受管 root 之外时才报 `ownership`，测试分别断言两个错误类别。

checker 本身也包含在其全树扫描范围内。

### 9.6 文件系统清洁门禁

`tools/check_hygiene.py` 直接遍历工作树文件系统，因此 ignored artifact 也可见。root
`/.git` 是唯一控制目录特例：它必须是本独立 repository 的普通 directory，并且不遍历其
内容；regular gitfile、symlink 或其他 file type 都失败。checker 还要核对
`git rev-parse --show-toplevel` 精确等于传入的 repository root，防止指向 R22 或父仓库。
除此之外，以下任一项始终失败：仓库内 venv、
`__pycache__/`、`.pytest_cache/`、
`*.pyc`/`*.pyo`、`*.egg-info/`、`build/`、`dist/`、`.coverage*`、任意层级的
`.DS_Store`、`Thumbs.db`、任何 symlink，以及 basename 匹配
`*.tmp`/`*.partial`/`*.part`/`*.bak`/`*~` 的 partial file。repository root 下
`artifacts/`、`reports/`、`exports/`、`output/`、`tmp/` 是明确禁止的外部
publication/report 误落目录；合法产品目录 `src/xrr_fitter/gui/export/` 不做 basename
子串误报。

除上述 root `/.git` 特例外，允许的 top-level 输入 root 精确为 `src/`、`tests/`、`tools/`、`examples/`、`docs/`、
`verification/`、`.github/`，以及 root regular file `AGENTS.md`、`.gitignore`、
`README.md`、`MANIFEST.in`、`pyproject.toml`、`requirements-macos-arm64-py312.lock`；
除此之外的新 top-level path 报 `ownership`。基础模式允许这些合法 root 中正在审阅、
尚未 stage 的 regular source/test/tool/example/doc/evidence 和 root metadata 变更，只按上述
生成物规则拒绝；因此任务 11 截图、任务 13 committed evidence 在提交前可以跑
RED/GREEN。它提供一个且只有一个附加严格开关
`--require-git-clean`，该开关再要求
`git status --porcelain=v1 --untracked-files=all` 为空。开发中的 RED/GREEN 可以包含正在
审阅的 source/test 改动，但不能包含生成垃圾；发行验证必须同时满足文件系统 clean 和
Git clean，两者不能互相替代。

`tools/verify.py` 在每个 mode 前后运行基础 hygiene checker，并为 pytest 子进程统一设置
`PYTHONDONTWRITEBYTECODE=1`、精确为当前 repository root 下 `src/` 的
`PYTHONPATH`、外部
`MPLCONFIGDIR`/`XDG_CACHE_HOME`/`--basetemp`，同时传入 `-p no:cacheprovider`。所有这些
外部目录都位于 `--report-dir` 或工具拥有的 `TemporaryDirectory`。`distribution`、
`identity` 和 `release` 无条件启用 `--require-git-clean`；CI 在每个 clean-checkout job
入口也显式运行严格开关。`tests/unit/tools/test_hygiene.py` 必须验证上述每个合法 root
中的 tracked 修改和普通 untracked regular file 在基础模式可验证、在严格模式失败；
root allowlist 外普通 path 在两种模式都以 `ownership` 失败。测试参数化覆盖上述每一
类硬失败项：venv、`__pycache__`、pytest cache、pyc/pyo、egg-info、build/dist、
coverage、`.DS_Store`、`Thumbs.db`、partial file、五个 root-level generated directory、
symlink 和 allowlist 外 top-level path；每类都各有普通 untracked 与被 `.gitignore`
命中两种 case，并单独
覆盖合法独立 `.git` directory、regular gitfile、`.git` symlink、错误 top-level 和真正
clean tree；不存在 linked-worktree 兼容分支。checker 自身也通过 Radon。

## 10. 目标测试架构

### 10.1 测试类别

- `unit/`：model、I/O unit、physics、fit、analysis、services 和 tools 的纯行为测试。
- `gui/`：pytest-qt widget/controller 行为测试，每个文件对应一个 feature workflow。
- `integration/`：跨多个 production domain、project round trip、worker 和 export 的测试。
- `regression/`：具名的历史数值缺陷和归一化 R22 对比测试。
- `acceptance/`：220-case 语料库、R22 等价性、真实数据和完整 GUI workflow 测试。
- `architecture/`：import 方向、API surface、旧模块缺失、distribution 内容和 quality gate 接线测试。
- `support/`：只保存具名的 `model_cases.py`、`recovery_cases.py` 和真实
  spawn entry script；不建第二套 production framework 或通用 GUI harness。
- `fixtures/`：只保存不可变的小型 source data。迁移 golden 统一属于
  `verification/r22/reference/`，不在 tests 下复制第二份。

### 10.2 测试设计规则

- 按稳定行为组织测试，不按开发任务或私有 production 文件名组织。
- 只使用一个 root `conftest.py`；它仅注册 marker/option 和真正跨 suite 的 fixture。
- Domain builder 是 `tests/support` 中的普通显式 function，不是 autouse fixture。
- 不得 monkeypatch facade global 或私有 module path。
- 仅向负责相应 nondeterministic boundary 的具体 function 注入 clock、executor 或 solver callable。不得引入覆盖整个仓库的 mock/DI framework。
- 非 GUI 的 acceptance 和 integration 测试只导入 `xrr_fitter.api`。GUI 及
  GUI-integration 测试可导入具体 `xrr_fitter.gui.*` widget/controller，但它们使用的
  领域类型和领域操作仍必须仅来自 `xrr_fitter.api`。
- 因此任务 10 完整 API 发布前不得创建任何 `tests/integration/**` 或 `tests/acceptance/**` target test；早期跨函数行为先在所属 domain 的 unit/regression test 中验证，不得临时放宽 import rule 或建立 façade stub。
- Unit 测试可以导入内部 capability module，但不得断言另一个 module 委托给它。
- 任何测试都不得断言 `__module__`、re-export identity、旧 import 可用性或私有文件数量。
- 不允许永久 `xfail`。
- 被选中的测试不得 skip。平台专属测试只出现在支持该平台的显式 mode/job path
  registry 中；其他 job 不收集这些路径，而不是收集后 skip。
- approved-data job 中缺少已批准数据属于失败，不是 skip。
- Test helper 不得仅用于隐藏 branch/assertion complexity；它必须表示可复用的 domain builder 或 comparison。

### 10.3 目标测试文件

```text
tests/architecture/
  test_dependency_rules.py
  test_naming_rules.py
  test_public_api.py
  test_removed_legacy_modules.py
  test_distribution.py
  test_quality_gate.py

tests/integration/
  test_entrypoints.py
  test_project_roundtrip.py
  test_single_fit_workflow.py
  test_joint_fit_workflow.py
  test_batch_resume.py
  test_export_workflow.py
  test_gui_project_workflow.py
  test_process_workers.py

tests/regression/
  test_numerical_reference.py
  test_profile_basin_regressions.py
  test_recovery_metrics.py

tests/acceptance/
  test_synthetic_recovery_corpus.py
  test_r22_reference_equivalence.py
  test_real_data_workflows.py
  test_gui_real_data_workflows.py

tests/gui/
  test_project_document.py
  test_project_actions.py
  test_source_recovery.py
  test_data_import.py
  test_data_masks.py
  test_structure_editor.py
  test_oxide_workflow.py
  test_parameter_table.py
  test_parameter_sharing.py
  test_fit_controller.py
  test_fit_progress.py
  test_results.py
  test_plots.py
  test_export_dialog.py
  test_workspace.py
  test_accessibility.py
  test_focus_navigation.py
  test_expert_views.py
```

Unit 文件按下方迁移映射创建；不得预先创建空文件。

## 11. 现有测试迁移映射

经审计的 Task-10 前快照，仅用于估算规模，绝不作为最终数量门禁：

- 活动 `tests/`：28 个 Python 文件、33,581 LOC、906 个 `test_*` definition；
- GUI 测试族：11 个文件、13,446 LOC、361 个测试；
- `tests_r21/`：12 个 Python 文件、13,451 LOC、376 个 definition；
- R21 oracle：428 个 frozen node、34 个 old-architecture node 被 deselect、394 个 behavior node 已执行。

`R22-final` 完成后，生成机器可读的最终 collection manifest，并创建 `docs/architecture/r22-r23-test-ledger.csv`，列必须精确为：

```text
source_tree,source_nodeid,contract_id,action,target_nodeids,reason
```

允许的 `action` value 为 `port`、`rewrite`、`merge` 和 `delete_layout_only`。`target_nodeids` 是一个 canonical JSON string array，至少包含一个 R23 node ID，从而同时支持 many-old -> one-new 和 one-old -> many-new；不得使用逗号拼接的非结构化字符串。`delete_layout_only` 只允许用于旧 import path、re-export/delegation、`__module__`、facade 存在性或私有 module layout assertion，其每个 target 都必须是新的 legacy-absence 或 API architecture test。最终带 hash 的 R22 collection manifest 中的每个 source node 必须恰好出现一行。

### 11.1 核心领域

| 旧测试文件 | R23 目标位置 |
|---|---|
| `tests/test_data.py` | `unit/io/test_xy_reader.py`, `unit/model/test_prepared_data.py`, `unit/model/test_instrument_values.py`, `unit/model/test_resolution_mapping.py`, `unit/model/test_fit_masks.py` |
| `tests/test_structure.py` | `unit/model/test_structures.py`, `unit/physics/test_stack_expansion.py` |
| `tests/test_materials.py` | `unit/model/test_materials.py`, `unit/physics/test_material_sld.py`, `unit/services/test_structures.py`（oxide suggestion service behavior） |
| `tests/test_auto_init.py` | `unit/fit/test_feature_detection.py`, `unit/fit/test_candidate_initialization.py` |
| `tests/test_objective.py` | `unit/model/test_parameters.py`, `unit/model/test_fitting_values.py`, `unit/fit/test_objective.py` |
| `tests/test_physics.py` | `unit/physics/test_resolution.py`, `test_instrument_model.py`, `test_reflectivity.py`, `test_periodic_reflectivity.py`, `test_derivatives.py`, `test_sld_profile.py`, `regression/test_numerical_reference.py` |

### 11.2 拟合与分析

| 旧测试文件 | R23 目标位置 |
|---|---|
| `tests/test_optimize.py` | `unit/test_evaluation.py`, `unit/fit/test_problem_compilation.py`, `unit/fit/test_objective.py`, `test_local_solver.py`, `test_global_solver.py`, `test_stage_search.py`, `test_resume.py`, `test_checkpoint.py`, `test_candidate_ranking.py`, `test_screening.py`, `regression/test_profile_basin_regressions.py`, `integration/test_process_workers.py` |
| `tests/test_uncertainty.py` | `unit/model/test_analysis_values.py`, `unit/analysis/test_classification.py`, `test_profiles.py`, `test_binary_profiles.py`, `test_derivatives.py`, `test_bootstrap.py`, `test_mcmc.py`, `test_diagnostics.py`, `test_report.py`, `integration/test_process_workers.py` |
| `tests/test_batch.py` | `unit/fit/test_joint_problem.py`, `test_joint_evaluation.py`, `test_joint_pipeline.py`, `unit/services/test_joint_registry.py`, `test_independent_batch.py`, `integration/test_joint_fit_workflow.py`, `integration/test_batch_resume.py` |
| `tests/test_synthetic_recovery.py` | `tests/support` 中的 builder/case、`regression/test_recovery_metrics.py` 中的 metric test、`acceptance/test_synthetic_recovery_corpus.py` 中的慢速 220-case 门禁 |

`test_synthetic_recovery.py` 前部的 support definition 迁移为具名 builder/case definition，且其自身必须通过 Radon。不保留相同的 R21 副本。

### 11.3 项目、外观层、导出与示例

| 旧测试文件 | R23 目标位置 |
|---|---|
| `tests/test_project.py` | `unit/model/test_project_state.py`, `unit/model/test_operations.py`, `unit/io/test_project_codec.py`, `unit/io/test_source_validation.py`, `integration/test_project_roundtrip.py` |
| `tests/test_export.py` | `unit/model/test_export_values.py`, `unit/io/test_export_run.py`, `test_export_tables.py`, `test_export_plots.py`, `unit/services/test_exports.py`, `integration/test_export_workflow.py` |
| `tests/test_examples.py` | `unit/io/test_examples.py`, `architecture/test_distribution.py` |
| `tests/test_core_api.py` | `architecture/test_public_api.py` 和 `integration/test_single_fit_workflow.py` 中的新 API 行为；所有 `xrr_core` 转发 assertion 使用 `delete_layout_only` |
| `tests/_spawn_smoke_runner.py` | `tests/support/processes/run_fit_worker.py` |
| `tests/_uncertainty_spawn_smoke_runner.py` | `tests/support/processes/run_analysis_worker.py` |
| `tests/conftest.py` | 最小 root registration，以及 `tests/support` 中的显式 builder |

### 11.4 GUI

11,007 行的 `tests/test_gui.py` 按可观察 workflow 拆分：

- project 新建/打开/保存/回滚 -> `test_project_document.py`, `test_project_actions.py`；
- source reload/relink/acceptance -> `test_source_recovery.py`；
- import/instrument/mask -> `test_data_import.py`, `test_data_masks.py`；
- structure tree/periodic/oxide transaction -> `test_structure_editor.py`, `test_oxide_workflow.py`；
- parameter editing/expert/sharing -> `test_parameter_table.py`, `test_parameter_sharing.py`, `test_expert_views.py`；
- process lifecycle/cancel/checkpoint/progress -> `test_fit_controller.py`, `test_fit_progress.py`；
- candidate/result/MCMC -> `test_results.py`；
- reflectivity/residual/SLD/diagnostic interaction -> `test_plots.py`；
- workspace layout/state -> `test_workspace.py`；
- 完整一键 workflow -> `integration/test_gui_project_workflow.py`；
- 真实数据 GUI workflow -> `acceptance/test_gui_real_data_workflows.py`。

现有按阶段命名的文件迁移如下：

| 旧文件族 | R23 目标位置 |
|---|---|
| `test_gui_task3_actions.py`, `test_gui_task3_review.py` | structure、oxide 和 project action 测试 |
| `test_gui_task8_source_dialogs.py`, `test_gui_task8_source_recovery.py` | `test_source_recovery.py` |
| `test_gui_task8_export.py` | `test_export_dialog.py`、integration export workflow |
| `test_gui_task9_accessibility*.py` | `test_accessibility.py` |
| `test_gui_task9_focus.py` | `test_focus_navigation.py` |
| `test_gui_task9_batch_readiness.py` | `test_fit_progress.py`、integration joint workflow |
| `test_gui_task9_expert_tabs.py` | `test_expert_views.py`, `test_plots.py` |

### 11.5 R21 与集成工具

- 从 R23 删除 `tests_r21/`。
- 将其代码、hash、node manifest 和 JUnit result 固化在 R22 archive 中。
- 将 R21 独有的 profile reconvergence、Stage-E 和 classification/profile-basin 行为按语义迁移到 R23 behavior test。
- 不迁移仅用于证明 modular delegation、facade 存在、re-export 或旧私有文件名的测试。
- 将 `.integration/tools/run_r21_behavior_oracle.py`、injection script 和 R21 node/hash manifest 随 R22 一并固化；R23 永不运行它们。
- 将通用的 release identity 测试迁移到 `tests/unit/tools/test_release_identity.py`，实现收敛到
  单文件 `tools/release_identity.py`；不保留 build/validate 转发模块。
- 不得将 R22 专属 run ID、硬编码路径或 task-state default 复制到 R23 tool。

## 12. R22 参考与已批准数据

现有 canonical/statistical summary 只证明 release acceptance，不自动包含 R23 逐域迁移所需的 compiled array、parameter order、analytic Jacobian、完整 stage history 和 candidate lineage。因此任务 0 必须在 `R22-final` tagging 前，用 R22 本身生成专用 data-only reference bundle；R23 不得从缺失字段的 summary 猜测这些值。

R22 bundle 固定包含以下非空 group：

- `model_project`：不可变 model、project state transition 和 serialization field；
- `io`：XY parse、column mapping、duplicate merge、mask/source identity 和 project round trip；
- `physics`：stack、SLD、Parratt、resolution、footprint、reflectivity、derivative 和 SLD profile array；
- `fit_compile`：parameter order、coordinate/bound、compiled array、objective、warning 和固定 vector 上的 analytic Jacobian；
- `fit_search`：固定 seed 的 stage history、candidate lineage/ranking、checkpoint 和 resume result；
- `analysis`：classification、profile、binary profile、bootstrap/MCMC stream、diagnostic 和 report；
- `services`：single/joint workflow、duplicate-stem dataset ID allocation、invalidation、
  project result 和 export manifest；
- `gui`：GUI Tasks 1-10 的规范化 action/state trace、已接受 project/export/plot artifact hash。

bundle builder 是 R22 evidence-only tool，不修改 production module。所有输入、seed、configuration、platform/dependency identity、builder hash 和输出 field policy 都写入 manifest；wall-clock duration、生成时间、PID、临时目录和绝对路径不得进入 golden payload。它在两个独立输出目录运行，canonical JSON 和采用固定 ZIP metadata 的 deterministic NPZ 必须逐字节一致；不一致则不得 tagging。bundle 及 builder/test 均进入 R22 release identity、独立逐文件 rehash 和最终 archive。

R23 存储归一化证据，而不是 R22 Python：

```text
verification/r22/collections/tests-active.json
verification/r22/collections/tests-r21.json
verification/r22/reference/manifest.json
verification/r22/reference/golden/*.json
verification/r22/reference/golden/*.npz
verification/approved-data/manifest.json
verification/approved-data/records/*.json
```

R22 manifest 包含：

- schema version；
- `R22-final` commit 和 tree SHA；
- release identity SHA-256；
- Python/dependency/platform identity；
- input ID/class 和 SHA-256 value；
- 每次运行的 seed 和 configuration hash；
- 相对归一化 output path 和 hash；
- 每个 artifact field 的 comparison policy。

对比规则：

- ID、label、ordering、warning、confidence class、stage history、candidate lineage、seed consumption 和 file manifest 必须精确一致。
- 在固定平台上生成的 deterministic array 使用精确 shape、dtype 和 `numpy.array_equal(actual, reference, equal_nan=True)` 对比。
- Physics reference parity 保留已批准边界：当 `R_ref >= 1e-12` 时为 `abs_error <= 1e-10 + 5e-7 * R_ref`；否则为 `abs_error <= 1e-12`。
- Statistical acceptance 使用已批准的 median/p95/open-profile/confidence threshold，绝不使用从单个 R22 输出推断的 threshold。
- 用于 non-bitwise solver scalar 的任何 tolerance 都记录为具名 manifest field，且不得在测试代码中放宽。
- 缺少 field、artifact、hash 或 manifest 时失败。不得使用 discovery fallback。

`tools/compare_r22_reference.py` 对 R22 一侧只解析这些归一化文件，绝不从 R22 tree 导入 module 或启动 R22 subprocess。对 actual 一侧，它使用显式、封闭的 group registry 在当前进程调用已经迁移完成的 R23 adapter；`--group physics` 等命令由该 adapter 以 manifest 中相同 input/config 生成 actual value。Task 2 的 `--self-check` 只验证 reference schema/hash，不加载尚不存在的 domain；普通 `--group` 在 adapter/module 尚未实现、字段缺失或 group 未注册时硬失败，不做 discovery、fallback 或跨版本调用。

adapter 与生产领域同批落地，生命周期固定为：Task 2 只创建 comparator
engine、空 registry 和 `--self-check`；Task 3 注册 `model_project`，Task 4 注册 `io`，
Task 5 注册 `physics`，Task 6 注册 `fit_compile`，Task 7 注册
`fit_search`，Task 8 注册 `analysis`，Task 10 注册 `services`，Task 11 最后一个
slice 注册 `gui`。每批必须同时提交对应的
`tools/reference_groups/{model_project,io,physics,fit_compile,fit_search,analysis,services,gui}.py`、registry
修改、registry/adapter unit test 和该 group GREEN 对比；Task 12 只全组重验，
不得首次补 adapter。`tools/reference_groups/__init__.py` 为 0 bytes。

approved-data manifest 精确包含三个 case ID：

```text
known_single_layer
workable_mo_si_multilayer
unstable_multilayer
```

`verification/approved-data/manifest.json` 与三个 record 是可独立重验的已提交证据，
不是对仓库外签核的松散 hash 声明。schema 固定为：

```text
ApprovedDataManifest
  schema: Literal["xrr-r23-approved-data-manifest-v1"]
  candidate_schema: Literal["xrr-r23-approved-data-candidate-v1"]
  r22_reference_sha256: SHA256Hex64
  workflow_contract_sha256: SHA256Hex64
  environment: CanonicalEnvironment
  candidate_report_sha256: SHA256Hex64
  domain_signoff_sha256: SHA256Hex64
  approved_source_tree_sha256: SHA256Hex64
  records_tree_sha256: SHA256Hex64
  cases: tuple[ApprovedCaseIndex, ApprovedCaseIndex, ApprovedCaseIndex]

ApprovedCaseIndex
  case_id: one of the three IDs above, sorted
  source: RelativeFileRecord
  record: RelativeFileRecord
  conclusion: exact signed product-level conclusion

ApprovedCaseRecord
  schema: Literal["xrr-r23-approved-case-record-v1"]
  case_id: same ID as manifest index
  source: RelativeFileRecord
  configuration_sha256: SHA256Hex64
  operations: non-empty ordered tuple of stable API/GUI action names
  runs: tuple[ApprovedRun, ApprovedRun, ApprovedRun, ApprovedRun]
  normalized_result: complete canonical candidate result object
  signoff: EmbeddedCaseSignoff

ApprovedRun
  ordinal: Literal[1, 2, 3, 4]
  seed: integer
  project: RelativeFileRecord
  exports: non-empty sorted tuple[RelativeFileRecord, ...]
  plots: non-empty sorted tuple[RelativeFileRecord, ...]
  normalized_result: complete canonical run result object

EmbeddedCaseSignoff
  reviewer: non-empty string
  role: non-empty string
  approved: Literal[true]
  conclusion: exact match to candidate and manifest conclusion

CanonicalEnvironment
  python_version: normalized 3.12 patch version
  platform: Literal["macos-arm64"]
  dependency_lock_sha256: SHA256Hex64
  production_tree_sha256: SHA256Hex64
  acceptance_test_tree_sha256: SHA256Hex64
  qt_runtime_identity: non-empty normalized string

RelativeFileRecord
  path: normalized relative POSIX path
  size: positive integer
  sha256: SHA256Hex64
```

四个 run 按 ordinal 排序；1-3 使用相同 seed/configuration，4 使用不同 seed 且相同
configuration。candidate report 和 signoff 禁止时间、绝对路径和临时目录，manifest 加三个
record 必须是这两份 canonical 外部 JSON 的**无损规范化投影**：builder 从已提交字段重建
两份 canonical bytes 后，必须分别重得 `candidate_report_sha256` 和
`domain_signoff_sha256`，不能丢弃 reviewer/role、任一 run、project/export/plot file record、
environment、workflow operation、warning、confidence、metric 或 conclusion。
`records_tree_sha256` 只对三个 record file 的
`(path,size,sha256)` 做 length-framed hash，避免把 manifest 自身纳入自引用。

manifest 记录相对于显式指定 data root 的 source path/hash；
`approved_source_tree_sha256` 对三个 source record 做 length-framed hash。普通
approved-data、identity 和 final freeze validation 都从已提交 manifest/records 重建并验证
candidate/signoff hash，再从显式 raw data root 重读三个 source，最后分别重算 committed
evidence tree 与 raw source tree；不要求最终发行时保留仓库外 candidate/signoff 文件，也不
信任 manifest 自报的 digest。release command 显式提供该 root：

```bash
/Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py approved-data --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final
```

该 option 对 approved-data job 是必需的。不得进行 environment search、home-directory search 或 synthetic substitution。

## 13. 架构测试

`tests/architecture/` 强制执行：

1. 第 6 节中的 dependency graph，包括 local import 和 type-checking import。
2. 不得存在包含多个 module 的 strongly connected component。
3. 精确的 `xrr_fitter.api.__all__` 和公共 signature。
4. 不得存在 legacy 顶层 `xrr/` package/import、legacy 顶层 `gui/` package/import、
   root `xrr_core.py`、root `xrr_app.py`、`tests_r21/`、`compat.py`、wildcard import、
   `sys.modules` alias 和 `__module__` assignment。测试按 qualified import root 判断，不对
   `xrr_fitter.gui` 做子串误报。
5. GUI domain import 只能通过 `xrr_fitter.api`。
6. `fit` 与 `analysis` 绝不相互导入。
   fit/analysis 共用的数值求值只归 `xrr_fitter.evaluation`；目标树不得出现第二套
   `fit/evaluation.py` 或 `fit/jacobian.py`。
7. `__main__` 在 GUI import 之前调用 `freeze_support()`。
8. Wheel 和 sdist 分别精确满足第 8 节的 product/member allowlist，包括受控的生成
   distribution metadata；不得用同一条宽泛规则检查二者。
9. Radon checker 接入 verification command，并扫描每个项目 Python 文件。
10. `test_naming_rules.py` 从 filesystem 扫描与 Radon 完全相同的
    `src/`、`tests/`、`tools/`、`examples/` Python 集合：普通 module 不得以前导
    `_` 命名，只精确豁免 `__init__.py`/`__main__.py`；文件/普通函数/变量为
    `snake_case`，class 为 `CapWords`，module constant 为 `UPPER_SNAKE_CASE`；只对能通过
    Qt base-class introspection 证明的真实 override 豁免 Qt camelCase method。文件名不得
    重复父目录前缀，永久 test filename 不得含 `task` 加阶段号，所有
    package `__init__.py` 必须为 0 bytes。

这些测试断言架构行为，而不是任意文件数量或 LOC 限制。

Architecture test 按其被测能力落地：任务 1/2 先启用 dependency、legacy-absence 和 quality wiring；任务 10 在完整 API/export 存在后启用 public API 与 distribution 内容测试；任务 11 在真实 `__main__`/GUI 存在后补入口和 GUI import gate。不得提前提交空测试、stub、skip 或 xfail 来制造 GREEN。

## 14. 实施顺序

### 任务 0：记录已完成 R22 的本地交接基线

任务 0 不属于 R23 开发，也不表示 R22 尚未完成。它只把负责人已经确认完成的 R22 状态
固化为后续对比所需的本地不可变输入；R23 的正式实施和 GitHub 控制从任务 1 开始。

**范围：** 不编辑 `xrr_fitter/xrr/**`、`xrr_fitter/gui/**`、`xrr_core.py`、
`xrr_app.py`、R7 runtime 或现有算法测试。只提交 GUI Task 10 验收记录、最终 runner
证据、R23 reference evidence tooling、release-identity builder/validator 及其 focused
test、重新构建的 release identity 和验收文档。若完成 GUI Task 10 需要修改 production
或 GUI 代码，先回到 R22 自身任务修复并重新跑全部 R22 gate；该修复不能混入本固化任务。

**创建：**

- `.integration/manifests/r22-final-transition.json`
- `.integration/manifests/r22-collection-requirements.lock`
- `.integration/evidence/r22-final-transition/task10-pending-release-identity.json`
- `.integration/evidence/r22-final-transition/task10-pending-product-manifest.tsv`
- `.integration/evidence/gui-task10/manifest.json`
- `.integration/evidence/gui-task10/{known-single-layer,workable-mo-si-multilayer,unstable-multilayer}.json`
- `.integration/evidence/gui-task10/release-gate.json`
- `.integration/evidence/r23-reference/**`
- `.integration/tools/build_r23_reference_bundle.py`
- `.integration/tools/build_r22_collection_lock.py`
- `.integration/tools/r22_final_transition_policy.py`
- `.integration/tools/check_r23_reference_tooling_quality.py`
- `.integration/tools/run_r22_preidentity_gate.py`
- `.integration/tools/rehash_r22_release.py`
- `.integration/tools/test_build_r23_reference_bundle.py`
- `.integration/tools/test_build_r22_collection_lock.py`
- `.integration/tools/test_r22_final_transition_policy.py`
- `.integration/tools/test_run_r22_preidentity_gate.py`
- `.integration/tools/test_rehash_r22_release.py`
- `.integration/tools/test_r23_reference_tooling_quality.py`
- `xrr_fitter/docs/acceptance/{known-single-layer,workable-mo-si-multilayer,unstable-multilayer}.md`

**修改：**

- `.integration/tools/build_release_identity.py`
- `.integration/tools/validate_r22_acceptance.py`
- `.integration/tools/test_build_release_identity.py`
- `.integration/tools/test_validate_r22_acceptance.py`
- `xrr_fitter/docs/acceptance/2026-07-23-real-data-status.md`
- `docs/XRR-R22-集成验收报告.md`
- `.integration/release/product-manifest.tsv`
- `.integration/release/release-identity.json`

#### 任务 0A：定义唯一允许的 pending -> final 迁移

- [ ] 等 R22 runner 自身的 canonical closeout 完全结束，并先把它已经产生的
  `sequential-release-identity-final.sha256`、
  `sequential-gui-task10-dataset-audit.json` 和 blocked 状态集成报告纳入一个经过审阅的
  pre-Task10 commit。它们必须在下述 baseline commit 中已跟踪且工作树干净；本 R23
  固化任务不得重写前两者，也不得把 v2 final identity hash 追加到它们。
- [ ] 选择 GUI Task 10 第一项真实数据验收编辑之前、canonical acceptance 已为
  `PASS`、`gui_task_10_status` 仍为 `blocked: missing approved dataset` 的 clean commit，
  将完整 40 位 commit 写入 `.integration/manifests/r22-final-transition.json` 的
  `baseline_commit`。这个值必须由执行者明确审阅，不得通过“最近一次提交”、commit
  message、mtime 或路径搜索猜测。
- [ ] builder 从该 commit 的 Git object 读取
  `.integration/release/release-identity.json` 和 `product-manifest.tsv`，验证前者为规范
  JSON、后者 SHA-256 与前者一致，然后原子写入两个 `task10-pending-*` 快照。不得从
  当前工作树复制，也不得接受调用方提供的未绑定 bytes。
- [ ] final release identity schema 升为 `xrr-r22-release-identity-v2`。旧 v1 pending
  identity 只作为只读输入；不回写、不“升级”旧文件，也不增加 v1/v2 runtime 兼容层。
  final identity 新增且只新增 `gui_task_10_acceptance`、`r23_reference_bundle` 和
  `pending_to_final_transition` 三个结构化组。
- [ ] `gui_task_10_status` 的唯一合法迁移精确为
  `blocked: missing approved dataset` -> `accepted: approved real-data complete`。
  `gui_task_10_acceptance.cases` 必须且只能包含
  `known_single_layer`、`workable_mo_si_multilayer`、`unstable_multilayer`，每项绑定
  approved input 的相对文件名、size、SHA-256、四次运行记录、项目/export/plot hash、
  结论和 domain-owner sign-off。空值、额外 case、synthetic substitution 或外部路径
  搜索均失败。builder/validator 必须接收唯一显式 `--approved-data-root`，拒绝 absolute
  case path、`..`、symlink 和 root 外解析结果，并从该 root 重新读取三个输入计算 hash。
- [ ] `.integration/manifests/r22-final-transition.json` 只允许包含 schema 和人工确认的
  `baseline_commit`，不得携带 path allowlist。唯一允许的 repository path/status/mode/reason
  集合写成 `.integration/tools/r22_final_transition_policy.py` 中的不可变常量，并由
  `test_r22_final_transition_policy.py` 固化。builder 和 validator 都只消费该常量；不能从
  CLI、JSON、目录扫描结果或环境变量追加 path。
- [ ] `pending_to_final_transition` 固定为以下字段形状；所有 tuple 都按 path 排序且 path
  唯一：

```text
PendingToFinalTransition
  schema: Literal["xrr-r22-pending-to-final-v1"]
  baseline_commit: FullGitCommit40
  transition_snapshots: tuple[FileRecord, FileRecord]
  status.before: Literal["blocked: missing approved dataset"]
  status.after: Literal["accepted: approved real-data complete"]
  repository_changes: tuple[GitFileChange, ...]
  self_generated_path: Literal[".integration/release/release-identity.json"]
  product_file_changes: tuple[ProductFileChange, ...]
  tooling_files: tuple[FileRecord, ...]

FileRecord
  path: normalized repository-relative POSIX path
  size: positive integer
  sha256: lowercase SHA256Hex64

ProductFileChange
  path: normalized repository-relative POSIX path
  change: Literal["added", "modified"]
  before_sha256: SHA256Hex64 | null
  after_sha256: SHA256Hex64
  reason: Literal["gui_task_10_acceptance"]

GitFileChange
  path: normalized repository-relative POSIX path
  status: Literal["A", "M"]
  before_blob: GitObjectId | null
  after_blob: GitObjectId
  before_mode: Literal["100644"] | null
  after_mode: Literal["100644"]
  reason: Literal["gui_task_10_acceptance", "r23_reference_bundle", "release_identity_tooling", "release_identity_output"]
```

- [ ] policy 的完整有限路径集合固定如下；花括号、glob、目录前缀和正则都不是合法
  policy 表达，实际 Python tuple 必须逐个列出这里的每一条具体 path：

```text
.integration/manifests/r22-final-transition.json
.integration/manifests/r22-collection-requirements.lock
.integration/evidence/r22-final-transition/task10-pending-release-identity.json
.integration/evidence/r22-final-transition/task10-pending-product-manifest.tsv
.integration/evidence/gui-task10/manifest.json
.integration/evidence/gui-task10/known-single-layer.json
.integration/evidence/gui-task10/workable-mo-si-multilayer.json
.integration/evidence/gui-task10/unstable-multilayer.json
.integration/evidence/gui-task10/release-gate.json
.integration/evidence/r23-reference/manifest.json
.integration/evidence/r23-reference/tooling-radon.json
.integration/evidence/r23-reference/json/model_project.json
.integration/evidence/r23-reference/json/io.json
.integration/evidence/r23-reference/json/physics.json
.integration/evidence/r23-reference/json/fit_compile.json
.integration/evidence/r23-reference/json/fit_search.json
.integration/evidence/r23-reference/json/analysis.json
.integration/evidence/r23-reference/json/services.json
.integration/evidence/r23-reference/json/gui.json
.integration/evidence/r23-reference/arrays/model_project.npz
.integration/evidence/r23-reference/arrays/io.npz
.integration/evidence/r23-reference/arrays/physics.npz
.integration/evidence/r23-reference/arrays/fit_compile.npz
.integration/evidence/r23-reference/arrays/fit_search.npz
.integration/evidence/r23-reference/arrays/analysis.npz
.integration/evidence/r23-reference/arrays/services.npz
.integration/evidence/r23-reference/arrays/gui.npz
.integration/tools/r22_final_transition_policy.py
.integration/tools/test_r22_final_transition_policy.py
.integration/tools/check_r23_reference_tooling_quality.py
.integration/tools/run_r22_preidentity_gate.py
.integration/tools/test_run_r22_preidentity_gate.py
.integration/tools/build_r23_reference_bundle.py
.integration/tools/test_build_r23_reference_bundle.py
.integration/tools/build_r22_collection_lock.py
.integration/tools/test_build_r22_collection_lock.py
.integration/tools/rehash_r22_release.py
.integration/tools/test_rehash_r22_release.py
.integration/tools/test_r23_reference_tooling_quality.py
.integration/tools/build_release_identity.py
.integration/tools/test_build_release_identity.py
.integration/tools/validate_r22_acceptance.py
.integration/tools/test_validate_r22_acceptance.py
.integration/release/product-manifest.tsv
.integration/release/release-identity.json
xrr_fitter/docs/acceptance/2026-07-23-real-data-status.md
xrr_fitter/docs/acceptance/known-single-layer.md
xrr_fitter/docs/acceptance/workable-mo-si-multilayer.md
xrr_fitter/docs/acceptance/unstable-multilayer.md
docs/XRR-R22-集成验收报告.md
```

  所有新增文件 mode 为 `100644`；既有文件必须保持 baseline mode。R22 固化不允许删除、
  rename、copy、submodule 或 executable-bit 变化。若实际实现需要这里没有的文件，先回到
  方案审查明确增加一条 path 和 reason，不能在执行时扩大 manifest。
- [ ] builder 在写 final identity 前比较 baseline Git tree 与当前 filesystem/index，拒绝
  policy 外的 tracked/untracked/mode change。提交后 validator 使用
  `git diff-tree --raw -r --no-renames --no-commit-id "$BASELINE_COMMIT" HEAD`，要求完整
  path/status/before-blob/after-blob/before-mode/after-mode 集合必须满足 raw diff path
  集合等于 `repository_changes` path 集合并上唯一的 `self_generated_path`；除 self path
  外逐项精确相等。唯一不进入内嵌 after hash 的 path 是
  `.integration/release/release-identity.json`；validator 仍要求它是 raw diff 中唯一的
  `self_generated_path`，并验证 Git blob bytes 等于当前 identity。其 SHA-256、HEAD 和
  archive 只由仓库外 receipt 绑定。
- [ ] `product_file_changes` 固定为状态文档和三份具名 acceptance record，不能包含其他
  path。validator 从 pending manifest 的完整 row 集合应用这四条差异，要求结果与 final
  product manifest 的 label/path/size/SHA-256 集合精确相等。这样不能把整个
  `product_identity` 设为例外。
- [ ] 以下字段必须在 pending snapshot 与 final identity 中逐值相等：
  `algorithm_identity`、`baseline_identities`、`canonical_acceptance`、既有
  `gate_evidence`、`release`，以及 pending product manifest 中未列入差异表的每一行。
  尤其 production source SHA-256 和 algorithm/test corpus SHA-256 必须不变。
- [ ] `gui_task_10_acceptance.files`、`r23_reference_bundle.files`、
  `pending_to_final_transition.transition_snapshots` 和 `tooling_files` 都使用穷尽的相对
  path/size/SHA-256 清单。每个仓库文件只能属于一个 FileRecord group；product manifest
  中的四份 acceptance 文档通过 case `record_path` 引用，不在 GUI group 重复。validator
  按 policy 重新枚举并要求集合精确相等；缺失、额外文件、symlink、跨组重复 path、空组
  或 hash 漂移均失败。
- [ ] GUI group 精确包含 final GUI manifest、三份 JSON run record、release gate、集成
  验收报告，以及 baseline 已存在且保持不变的
  `sequential-gui-task10-dataset-audit.json` 和
  `sequential-release-identity-final.sha256`。后两者只证明 pre-Task10 blocked baseline；
  不在 `repository_changes` 中，也不得写入 v2 final identity SHA-256。
- [ ] 禁止任何仓库内记录引用 final commit、tag、archive、final identity 自身 hash、
  final `release_binding_sha256`、rehash receipt 或 freeze receipt。集成验收报告必须把
  原有 v1 值明确标为 pre-Task10 baseline，并写明 final 值只存在仓库外 freeze receipt。
  `release_binding_sha256` 只对 final identity 中除 `schema` 和自身以外的字段做 canonical
  binding；最终身份文件 SHA-256 只写仓库外 receipt，从结构上消除自引用。

`run_r22_preidentity_gate.py` 不是另一个 release validator。它只执行下列固定
registry，不接受追加 gate、替换 test path 或关闭检查的 CLI option：

```text
active_non_slow
  cwd: /Users/dala/Desktop/xrr-rewrite-design-integration/xrr_fitter
  pytest: -o addopts= --strict-config --strict-markers -p no:cacheprovider
          -m "not slow" tests
  result: selected count > 0; failures/errors/skips/xfails/xpasses = 0

reference_tooling
  cwd: /Users/dala/Desktop/xrr-rewrite-design-integration
  pytest: -o addopts= --strict-config --strict-markers -p no:cacheprovider
          .integration/tools/test_build_r23_reference_bundle.py
          .integration/tools/test_build_r22_collection_lock.py
          .integration/tools/test_r22_final_transition_policy.py
          .integration/tools/test_build_release_identity.py
          .integration/tools/test_validate_r22_acceptance.py
          .integration/tools/test_rehash_r22_release.py
          .integration/tools/test_run_r22_preidentity_gate.py
          .integration/tools/test_r23_reference_tooling_quality.py
  result: selected count > 0; failures/errors/skips/xfails/xpasses/deselected = 0

canonical_baseline
  input: explicit run_id + artifact_root + pending transition snapshots
  check: canonical manifest/JSONL/result/exit, 180/20/20 plan, 220 cases,
         224 events, algorithm identity and pre-Task10 blocked identity

gui_task_10
  input: explicit GUI manifest + approved_data_root
  check: exactly three case IDs, each four runs, project/export/plot hashes,
         accepted conclusion and domain-owner sign-off

r23_reference
  input: explicit reference_root + tooling-radon.json
  check: exact eight-group schema, source/tool/config hashes, NPZ metadata,
         group non-emptiness and tooling Radon PASS
```

两个 pytest gate 都把 basetemp/JUnit 写到工具拥有的仓库外临时目录，并用
XML parser 读取计数，不解析终端文本。`canonical_baseline`、`gui_task_10` 和
`r23_reference` 调用 `validate_r22_acceptance.py` 和 reference builder 中唯一的
纯 validation function，不复制解析逻辑。输出 schema 固定为
`xrr-r22-preidentity-gate-v1`，只记录按 gate name 排序的状态、稳定计数和
输入 SHA-256；不记录 interpreter path、venv、临时目录、时间、final identity、
binding、commit 或 tag。它以同目录临时文件 + `fsync` + `os.replace` 原子写入
`.integration/evidence/gui-task10/release-gate.json`；任意 gate 失败都不留下 final/partial output。

#### 任务 0B：先写迁移与 reference tooling 的失败测试

- [ ] 先创建上述六个新 test 文件，并在 `.integration/tools/test_build_release_identity.py` 和
  `test_validate_r22_acceptance.py` 先增加 RED cases：非法状态迁移、production/algorithm
  漂移、canonical/gate drift、未声明 product row 变化、缺少三类 case、多余 evidence、
  before/after hash 不匹配、完整 Git diff 多/少 path、blob/mode drift、重复 path、symlink、
  非 ancestor baseline、伪造 pending snapshot、自引用路径和错误 release binding 必须分别
  失败。`test_r22_final_transition_policy.py` 还要断言 policy tuple 与第 0A 节固定集合精确
  相等。`test_run_r22_preidentity_gate.py` 覆盖 registry 多/少/改 path、pytest
  skip/xfail/xpass/error/empty collection、canonical drift、GUI case/sign-off 缺失、reference
  hash 漂移、原子成功和失败不留 output。
- [ ] `test_build_r23_reference_bundle.py` 覆盖八个 group 的精确集合、空/多/少 group、
  source/tool/config hash、canonical JSON、NPZ dtype/shape/key 顺序、approved path escape/symlink、
  双跑字节稳定和失败不留 partial tree；`test_build_r22_collection_lock.py` 覆盖唯一当前项目
  editable 的精确删除、零个/多个/外部 editable、非 pinned 输出、resolver failure、原子替换
  和双跑稳定；`test_r23_reference_tooling_quality.py` 覆盖六个新 production tool、两个修改
  production tool 与八个 test，精确 filesystem set 固定为以下 16 个 path，不能扫描一个
  宽泛目录后忽略缺失或额外文件：

```text
.integration/tools/build_r23_reference_bundle.py
.integration/tools/build_r22_collection_lock.py
.integration/tools/r22_final_transition_policy.py
.integration/tools/check_r23_reference_tooling_quality.py
.integration/tools/run_r22_preidentity_gate.py
.integration/tools/rehash_r22_release.py
.integration/tools/build_release_identity.py
.integration/tools/validate_r22_acceptance.py
.integration/tools/test_build_r23_reference_bundle.py
.integration/tools/test_build_r22_collection_lock.py
.integration/tools/test_r22_final_transition_policy.py
.integration/tools/test_r23_reference_tooling_quality.py
.integration/tools/test_run_r22_preidentity_gate.py
.integration/tools/test_rehash_r22_release.py
.integration/tools/test_build_release_identity.py
.integration/tools/test_validate_r22_acceptance.py
```

  quality test 同时覆盖 Radon 版本/CC/MI、缺失或额外 Python path 和确定性报告；
  `test_rehash_r22_release.py` 覆盖 repository/product/approved-data drift、两份 receipt 不一致、
  tag 类型/指向、archive hash 和所有原子输出失败路径。
- [ ] 运行 focused tests，确认新测试因 v2 schema/transition 尚未实现而失败，而不是因
  fixture 数据错误失败。对尚不存在的新 module，只接受明确指向计划实现 module/symbol 的
  missing-implementation RED；任何第三方 import、collection、fixture 或环境错误都必须先修正：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; BASETEMP=$(mktemp -d /tmp/xrr-r22-transition-red.XXXXXX); trap 'rm -rf "$BASETEMP"' EXIT; cd "$ROOT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -o addopts= --strict-config --strict-markers -p no:cacheprovider --basetemp "$BASETEMP" .integration/tools/test_build_r23_reference_bundle.py .integration/tools/test_build_r22_collection_lock.py .integration/tools/test_r22_final_transition_policy.py .integration/tools/test_build_release_identity.py .integration/tools/test_validate_r22_acceptance.py .integration/tools/test_rehash_r22_release.py .integration/tools/test_run_r22_preidentity_gate.py .integration/tools/test_r23_reference_tooling_quality.py -q
```

预期：新增 transition tests 为 `FAILED`；记录具体失败 nodeid。若当前完整 R22 gate 或
GUI Task 10 尚未结束，停在这里，不构建 final identity。

- [ ] 实现最小 v2 builder/validator 和 reference tooling，使上述八个已存在的测试 GREEN。新增
  `.integration/tools/build_r23_reference_bundle.py`、
  `build_r22_collection_lock.py`、`r22_final_transition_policy.py`、
  `check_r23_reference_tooling_quality.py`、`run_r22_preidentity_gate.py`、
  `rehash_r22_release.py`；此步不得再首次创建 test。lock builder 使用结构化 requirement parser，
  从当前 R22 lock 中移除且仅移除已验证指向当前项目的唯一 editable requirement；随后在
  工具拥有的临时 resolver venv 中用 `pip==26.1.2` 安装剩余精确 requirements 和
  `radon==6.0.1`，以 `pip freeze --exclude-editable` 生成包含 Radon 全部传递依赖的完整闭包，
  不能只向旧文本追加一行。输出中每个普通 requirement 必须为 `name==version`，VCS 必须
  固定 full commit；缺少/多个 editable、目标路径不符、非固定 requirement、缺失闭包或
  重复冲突均失败。对应测试证明 `mando`/platform 所需传递依赖已固定，并验证双跑字节
  稳定。不得编辑任何 production/GUI/algorithm module。
- [ ] quality test 使用 Radon API 扫描本任务新增或修改的全部 R22 Python tooling/test，
  执行第 9 节相同的 CC/MI 阈值；不能给 release tool、test helper 或 runner 豁免。

#### 任务 0C：生成锁和确定性 reference bundle

- [ ] 生成 collection lock，并在第二个临时路径重建后要求逐字节相同：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; LOCK="$ROOT/.integration/manifests/r22-collection-requirements.lock"; LOCK_CHECK_DIR=$(mktemp -d /tmp/xrr-r22-collection-lock.XXXXXX); LOCK_CHECK="$LOCK_CHECK_DIR/requirements.lock"; trap 'rm -rf "$LOCK_CHECK_DIR"' EXIT; test ! -e "$LOCK" && "$PYTHON" "$ROOT/.integration/tools/build_r22_collection_lock.py" --source "$ROOT/xrr_fitter/requirements-macos-arm64-py312.lock" --project-root "$ROOT/xrr_fitter" --output "$LOCK" && "$PYTHON" "$ROOT/.integration/tools/build_r22_collection_lock.py" --source "$ROOT/xrr_fitter/requirements-macos-arm64-py312.lock" --project-root "$ROOT/xrr_fitter" --output "$LOCK_CHECK" && cmp "$LOCK" "$LOCK_CHECK"
```
- [ ] 在工具拥有的临时外部 venv 中安装 dependency-only lock，运行
  reference/identity/transition/quality focused test；不得向活动 runner `.venv` 临时安装
  Radon：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; QUALITY_ENV=$(mktemp -d /tmp/xrr-r22-reference-quality.XXXXXX); QUALITY_REPORT=/Users/dala/Desktop/r22-reference-tooling-radon.json; trap 'rm -rf "$QUALITY_ENV"' EXIT; test ! -e "$QUALITY_REPORT" && python3.12 -m venv "$QUALITY_ENV" && "$QUALITY_ENV/bin/python" -m pip install pip==26.1.2 && "$QUALITY_ENV/bin/python" -m pip install -r "$ROOT/.integration/manifests/r22-collection-requirements.lock" && cd "$ROOT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$QUALITY_ENV/bin/python" -m pytest -o addopts= --strict-config --strict-markers -p no:cacheprovider --basetemp "$QUALITY_ENV/pytest-tmp" .integration/tools/test_build_r23_reference_bundle.py .integration/tools/test_build_r22_collection_lock.py .integration/tools/test_r22_final_transition_policy.py .integration/tools/test_build_release_identity.py .integration/tools/test_validate_r22_acceptance.py .integration/tools/test_rehash_r22_release.py .integration/tools/test_run_r22_preidentity_gate.py .integration/tools/test_r23_reference_tooling_quality.py -q && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$QUALITY_ENV/bin/python" .integration/tools/check_r23_reference_tooling_quality.py --repo-root "$ROOT" --output "$QUALITY_REPORT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$QUALITY_ENV/bin/python" .integration/tools/check_r23_reference_tooling_quality.py --repo-root "$ROOT" --output "$QUALITY_ENV/tooling-radon-second.json" && cmp "$QUALITY_REPORT" "$QUALITY_ENV/tooling-radon-second.json"
```
- [ ] 将三个经批准的原始数据文件复制到只读目录
  `/Users/dala/Desktop/xrr-approved-data-r22-final`，绝不能移动或改名覆盖；manifest 逐项
  只记录相对该只读 root 的复制后 path、size 和 SHA-256，不记录原始
  主机绝对位置。如必须保留复制来源，只写入仓库外的人工交接回执，不进入
  release identity/reference/approved-data manifest。先完成三类各四次运行、GUI 项目/
  export/plot 证据、三份具名 acceptance record 和 domain-owner sign-off，再将状态设为
  `accepted: approved real-data complete`。
- [ ] 使用固定输入和 seed 生成第 12 节完整 reference bundle，并在独立目录双跑验证
  逐字节一致：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; REFERENCE="$ROOT/.integration/evidence/r23-reference"; QUALITY_REPORT=/Users/dala/Desktop/r22-reference-tooling-radon.json; CHECK_ROOT=$(mktemp -d /tmp/xrr-r22-reference-check.XXXXXX); CHECK="$CHECK_ROOT/reference"; trap 'rm -rf "$CHECK_ROOT"' EXIT; test ! -e "$REFERENCE" && test -s "$QUALITY_REPORT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/.integration/tools/build_r23_reference_bundle.py" --repo-root "$ROOT/xrr_fitter" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --tooling-radon-report "$QUALITY_REPORT" --output "$REFERENCE" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/.integration/tools/build_r23_reference_bundle.py" --repo-root "$ROOT/xrr_fitter" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --tooling-radon-report "$QUALITY_REPORT" --output "$CHECK" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/.integration/tools/build_r23_reference_bundle.py" --compare "$REFERENCE" "$CHECK" && rm -f "$QUALITY_REPORT"
```

预期：八个 group 均非空；第二次输出与第一份逐文件、逐字节一致；manifest 记录 builder/input/config/environment hash。工具自身的 focused test 和完整 R22 release gate 都必须在其加入后重新达到 GREEN。

#### 任务 0D：先定稿仓库内容，再构建一次 final identity

- [ ] 使用 dependency-only collection lock 创建一次性外部环境，运行固定
  pre-identity registry，原子生成唯一的
  `.integration/evidence/gui-task10/release-gate.json`：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; GATE_ENV=$(mktemp -d /tmp/xrr-r22-preidentity.XXXXXX); PREFLIGHT="$ROOT/.integration/evidence/sequential-canonical-preflight.json"; OUTPUT="$ROOT/.integration/evidence/gui-task10/release-gate.json"; trap 'rm -rf "$GATE_ENV"' EXIT; test ! -e "$OUTPUT" && python3.12 -m venv "$GATE_ENV" && "$GATE_ENV/bin/python" -m pip install pip==26.1.2 && "$GATE_ENV/bin/python" -m pip install -r "$ROOT/.integration/manifests/r22-collection-requirements.lock" && RUN_ID=$("$GATE_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["identity"]["run_id"])' "$PREFLIGHT") && ARTIFACT=$("$GATE_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["artifact_root"])' "$PREFLIGHT") && cd "$ROOT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$GATE_ENV/bin/python" .integration/tools/run_r22_preidentity_gate.py --repo-root "$ROOT" --product-root "$ROOT/xrr_fitter" --run-id "$RUN_ID" --artifact-root "$ARTIFACT" --transition-manifest "$ROOT/.integration/manifests/r22-final-transition.json" --gui-task10-manifest "$ROOT/.integration/evidence/gui-task10/manifest.json" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --reference-root "$ROOT/.integration/evidence/r23-reference" --output "$OUTPUT"
```

  确认该 canonical JSON 中五个 gate 均为 `PASS`，并确认三类
  manifest/record、reference bundle、identity tooling、focused tests、Radon report 和
  验收文档已经定稿。此后仓库内内容只能因失败而回到本步重建，
  不能在 identity 之后“补一行记录”。
- [ ] 从 sequential preflight 解析 run ID/artifact，使用显式 transition baseline 和
  acceptance/reference manifest **只构建** final release identity；本步不调用 validator：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; PREFLIGHT="$ROOT/.integration/evidence/sequential-canonical-preflight.json"; RUN_ID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["identity"]["run_id"])' "$PREFLIGHT"); ARTIFACT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["artifact_root"])' "$PREFLIGHT"); env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/.integration/tools/build_release_identity.py" --root "$ROOT" --output-dir "$ROOT/.integration/release" --run-id "$RUN_ID" --canonical-root "$ARTIFACT" --transition-manifest "$ROOT/.integration/manifests/r22-final-transition.json" --gui-task10-manifest "$ROOT/.integration/evidence/gui-task10/manifest.json" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --r23-reference-root "$ROOT/.integration/evidence/r23-reference"
```

- [ ] 审阅 `git status`、`git diff --check`、完整 staged diff 和 transition 的逐路径差异；
  使用显式路径提交所有仓库内 final 文件，不使用 `git add -A`，不提交 venv/cache/临时
  report。提交后要求 tracked 和 untracked 状态都为空：

```bash
cd /Users/dala/Desktop/xrr-rewrite-design-integration && git diff --check && git add .integration/manifests/r22-final-transition.json .integration/manifests/r22-collection-requirements.lock .integration/evidence/r22-final-transition .integration/evidence/gui-task10 .integration/evidence/r23-reference .integration/tools/r22_final_transition_policy.py .integration/tools/test_r22_final_transition_policy.py .integration/tools/check_r23_reference_tooling_quality.py .integration/tools/run_r22_preidentity_gate.py .integration/tools/test_run_r22_preidentity_gate.py .integration/tools/build_r23_reference_bundle.py .integration/tools/build_r22_collection_lock.py .integration/tools/rehash_r22_release.py .integration/tools/test_build_r23_reference_bundle.py .integration/tools/test_build_r22_collection_lock.py .integration/tools/test_rehash_r22_release.py .integration/tools/test_build_release_identity.py .integration/tools/test_validate_r22_acceptance.py .integration/tools/test_r23_reference_tooling_quality.py .integration/tools/build_release_identity.py .integration/tools/validate_r22_acceptance.py .integration/release/product-manifest.tsv .integration/release/release-identity.json xrr_fitter/docs/acceptance/2026-07-23-real-data-status.md xrr_fitter/docs/acceptance/known-single-layer.md xrr_fitter/docs/acceptance/workable-mo-si-multilayer.md xrr_fitter/docs/acceptance/unstable-multilayer.md docs/XRR-R22-集成验收报告.md && git diff --cached --check && git diff --cached --name-status && git commit -m 'test: freeze R22 final approved baseline' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

若 `git status` 还列出 runner 已生成但上述命令未包含的文件，先判断其职责并将**精确路径**
加入相应结构化清单、测试和 staged path；不得用 `git add -A` 绕过审阅。

#### 任务 0E：从 clean HEAD 只读验证，再生成仓库外回执

- [ ] validator 与 `rehash_r22_release.py` 的 canonical JSON receipt 都必须包含：
  `status=PASS`、当前 `head_commit`、`head_tree`、release identity SHA-256、product manifest
  SHA-256、transition manifest SHA-256、approved-data tree SHA-256 和分组数量。approved-data
  tree hash 对三个 `(relative_path,size,sha256)` record 做 length-framed SHA-256。rehash
  receipt 还包含 validation receipt SHA-256 和所有互斥 FileRecord group 的 framed digest。
- [ ] 提交完成后从 clean HEAD 运行 validator。validator 自己使用同目录临时文件、
  `fsync` 和 `os.replace` 实现 `--output`；失败不留下 final/partial receipt。前后两次 status
  都必须为空。`--repo-root` 必须是包含 `.git`、`.integration/`、`docs/` 和
  `xrr_fitter/` 的顶层 repository；`--product-root` 单独指向 `xrr_fitter/`，不能用 product
  子目录冒充 repository root 而漏掉顶层 transition raw diff。命令中的
  `sequential-release-identity-pending.json` 只验证原 canonical runner pending -> PASS 链；
  两份 `task10-pending-*` snapshot 由 transition manifest 和 final identity 中的固定路径另行
  读取、重算并逐项验证，不能由该 sequential 参数替代：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; PREFLIGHT="$ROOT/.integration/evidence/sequential-canonical-preflight.json"; RUN_ID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["identity"]["run_id"])' "$PREFLIGHT"); ARTIFACT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["artifact_root"])' "$PREFLIGHT"); VALIDATION=/Users/dala/Desktop/r22-final-validation.json; test ! -e "$VALIDATION" && cd "$ROOT" && test -z "$(git status --porcelain=v1 --untracked-files=all)" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/.integration/tools/validate_r22_acceptance.py" --repo-root "$ROOT" --product-root "$ROOT/xrr_fitter" --runtime-root "$ROOT/.superpowers/sdd/r7" --artifact-root "$ARTIFACT" --run-id "$RUN_ID" --release-identity "$ROOT/.integration/release/release-identity.json" --pending-release-identity "$ROOT/.integration/evidence/sequential-release-identity-pending.json" --transition-manifest "$ROOT/.integration/manifests/r22-final-transition.json" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --output "$VALIDATION" && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

预期：外部 JSON 为 canonical `PASS`，同时证明原 canonical pending -> PASS 链、GUI
Task 10 blocked -> accepted 链、三类 approved data、reference bundle 和全部逐文件 hash。

- [ ] 只有 validator PASS 后，调用独立的正式 `rehash_r22_release.py`，重新计算 final
  identity 指向的每个 product、gate、GUI Task 10、reference、transition snapshot/tooling
  文件和三个外部 approved input。工具使用 strict canonical JSON/TSV parser，拒绝 duplicate
  key/path、path escape、symlink、mode/size/hash 漂移，并使用显式异常而非可被
  `PYTHONOPTIMIZE` 禁用的 `assert`。`--output` 同样原子写仓库外；仓库内文件不得引用该
  receipt 或其 hash：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; VALIDATION=/Users/dala/Desktop/r22-final-validation.json; REHASH=/Users/dala/Desktop/r22-final-current-files-rehash.json; test ! -e "$REHASH" && cd "$ROOT" && test -z "$(git status --porcelain=v1 --untracked-files=all)" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" .integration/tools/rehash_r22_release.py --repo-root "$ROOT" --release-identity .integration/release/release-identity.json --transition-manifest .integration/manifests/r22-final-transition.json --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --validation-receipt "$VALIDATION" --output "$REHASH" && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

- [ ] tag 前用同一 rehash tool 的 `--check-receipts` 模式重新计算当前 HEAD/identity/data，
  并逐字段核对两份 receipt；不能只检查文件存在、非空或字符串 `PASS`。随后创建 immutable
  annotated tag，将 archive 先写仓库外临时文件再原子 rename。最后由该工具的
  `--write-freeze-receipt` 原子写 canonical JSON，精确记录 final commit、tree、tag object、
  archive SHA-256、release identity SHA-256、validation receipt SHA-256、rehash receipt
  SHA-256 和 approved-data tree SHA-256；仓库内不再回写：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; PYTHON="$ROOT/xrr_fitter/.venv/bin/python"; ARCHIVE=/Users/dala/Desktop/xrr-r22-final.tar.gz; VALIDATION=/Users/dala/Desktop/r22-final-validation.json; REHASH=/Users/dala/Desktop/r22-final-current-files-rehash.json; FREEZE=/Users/dala/Desktop/r22-final-freeze.json; ARCHIVE_TMP=$(mktemp /Users/dala/Desktop/.xrr-r22-final.tar.gz.XXXXXX); trap 'rm -f "$ARCHIVE_TMP"' EXIT; cd "$ROOT" && test ! -e "$ARCHIVE" && test ! -e "$FREEZE" && test -z "$(git status --porcelain=v1 --untracked-files=all)" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" .integration/tools/rehash_r22_release.py --repo-root "$ROOT" --release-identity .integration/release/release-identity.json --transition-manifest .integration/manifests/r22-final-transition.json --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --validation-receipt "$VALIDATION" --rehash-receipt "$REHASH" --check-receipts && HEAD_COMMIT=$(git rev-parse HEAD) && TAG_COMMIT=$(git rev-parse -q --verify 'refs/tags/R22-final^{commit}' 2>/dev/null || true) && TAG_TYPE=$(git cat-file -t refs/tags/R22-final 2>/dev/null || true) && { test -z "$TAG_COMMIT" || { test "$TAG_COMMIT" = "$HEAD_COMMIT" && test "$TAG_TYPE" = tag; }; } && { test -n "$TAG_COMMIT" || git tag -a R22-final -m 'XRR R22 final accepted baseline' "$HEAD_COMMIT"; } && test "$(git rev-parse 'R22-final^{commit}')" = "$HEAD_COMMIT" && git archive --format=tar.gz --prefix=xrr-r22-final/ -o "$ARCHIVE_TMP" R22-final && mv "$ARCHIVE_TMP" "$ARCHIVE" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" .integration/tools/rehash_r22_release.py --repo-root "$ROOT" --release-identity .integration/release/release-identity.json --transition-manifest .integration/manifests/r22-final-transition.json --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --validation-receipt "$VALIDATION" --rehash-receipt "$REHASH" --archive "$ARCHIVE" --expected-tag R22-final --write-freeze-receipt "$FREEZE" && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

预期：`R22-final` 不存在时创建 annotated tag；已存在时只接受指向当前 clean HEAD 的
annotated tag，其他情况硬失败。archive 和全部 receipt 在仓库外，final identity 在提交
后未再重建，因而 commit、identity 和 archive 之间不存在循环引用。

- [ ] tagging 后，将 `/Users/dala/Desktop/xrr-rewrite-design-integration` 保持为只读 R22 worktree。

#### 任务 0F：确认 R22 只在本地冻结

- [ ] 任务 0E 的 local tag、archive、三份 receipt 和 clean-worktree 门禁全部通过后，确认
  R22 仓库仍没有任何 remote，且不存在 R23 directory。此任务不使用 `gh`、不连接 GitHub、
  不 push `integration-r22`/`R22-final`，也不创建 R22 GitHub Release：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-design-integration; cd "$ROOT"; test "$(git branch --show-current)" = integration-r22; test "$(git cat-file -t refs/tags/R22-final)" = tag; test "$(git rev-parse 'R22-final^{commit}')" = "$(git rev-parse HEAD)"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test -z "$(git remote)"; test ! -e /Users/dala/Desktop/xrr-rewrite-r23
```

R22 到此结束。后续 GitHub preflight、泄密扫描、remote 配置和 push 全部属于独立 R23
repository 的任务 1；任何 R22 ref 或 Release 都不得发布到该 remote。

### 任务 1：创建独立的 R23 Git 仓库

**创建：** `/Users/dala/Desktop/xrr-rewrite-r23`

- [ ] 先证明 R22 本地冻结已经完成、R22 仓库没有 remote、R23 路径不存在。R23 只读取
  `R22-final` 的 Git object 和仓库外 freeze 制品，不修改 R22 working tree：

```bash
set -euo pipefail; R22=/Users/dala/Desktop/xrr-rewrite-design-integration; R23=/Users/dala/Desktop/xrr-rewrite-r23; cd "$R22"; test "$(git branch --show-current)" = integration-r22; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test -z "$(git remote)"; test "$(git cat-file -t refs/tags/R22-final)" = tag; test "$(git rev-parse 'R22-final^{commit}')" = "$(git rev-parse HEAD)"; test -s /Users/dala/Desktop/xrr-r22-final.tar.gz; test -s /Users/dala/Desktop/r22-final-validation.json; test -s /Users/dala/Desktop/r22-final-current-files-rehash.json; test -s /Users/dala/Desktop/r22-final-freeze.json; test ! -e "$R23"
```

- [ ] 创建全新的普通 Git repository；不得使用 `git worktree`、`git clone`、复制 R22
  `.git`、把 R22 设为 remote，或让 R23 root commit 具有 parent。只从 local
  `R22-final` 精确提取三份稳定文档和图片，再复制本方案。R22 implementation、test、
  runner、integration evidence 和 Git history 都不进入 R23：

```bash
set -euo pipefail; R22=/Users/dala/Desktop/xrr-rewrite-design-integration; R23=/Users/dala/Desktop/xrr-rewrite-r23; PLAN=/Users/dala/Desktop/2026-07-26-xrr-r23-clean-break-plan.md; EXTRACT=$(mktemp -d /tmp/xrr-r23-docs.XXXXXX); trap 'rm -rf "$EXTRACT"' EXIT; test ! -e "$R23"; mkdir "$R23"; cd "$R23"; git init -b r23-clean-architecture; test -d .git; test ! -L .git; test "$(git rev-list --all --count)" -eq 0; test -z "$(git remote)"; mkdir -p docs/acceptance docs/architecture; git -C "$R22" show R22-final:xrr_fitter/docs/algorithm.md > docs/algorithm.md; git -C "$R22" show R22-final:xrr_fitter/docs/user-guide.md > docs/user-guide.md; git -C "$R22" show R22-final:xrr_fitter/docs/acceptance/real-data-template.md > docs/acceptance/real-data-template.md; git -C "$R22" archive R22-final xrr_fitter/docs/images | tar -x -C "$EXTRACT"; mv "$EXTRACT/xrr_fitter/docs/images" docs/images; cp "$PLAN" docs/architecture/r23-clean-break.md; cmp "$PLAN" docs/architecture/r23-clean-break.md; printf '%s\n' '__pycache__/' '*.py[cod]' '.pytest_cache/' '.coverage*' '*.egg-info/' 'build/' 'dist/' '.venv/' 'venv/' '.DS_Store' > .gitignore
```

- [ ] 运行 legacy-absence guard 并取得 GREEN。这里不制造“先复制整棵 R22 再删除”的 RED；
  独立 R23 从第一个文件开始就没有旧 layout。Task 2 用永久 architecture test 固化同一边界：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && test ! -e xrr_fitter && test ! -e .integration && test ! -e .superpowers && test ! -e docs/superpowers && test ! -e docs/XRR-R22-集成验收报告.md && test ! -e xrr_core.py && test ! -e xrr_app.py && test ! -e tests_r21
```

- [ ] 此任务只建立独立 repository、R23 `.gitignore` 和稳定文档；不创建 production
  implementation、compatibility module、Python stub、空 package/test 目录或临时占位文件。
  root `pyproject.toml`、lock、README、package/test init 和永久门禁由任务 2 与其测试同批创建。
- [ ] 创建唯一 root commit；commit 前后都证明 repository 没有 parent、没有 remote，且
  Desktop 方案与仓库内审计副本逐字节相等：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; test "$(git branch --show-current)" = r23-clean-architecture; test "$(git rev-list --all --count)" -eq 0; test -z "$(git remote)"; cmp /Users/dala/Desktop/2026-07-26-xrr-r23-clean-break-plan.md docs/architecture/r23-clean-break.md; git add .gitignore docs; git diff --cached --check; git diff --cached --name-status; git commit -m 'chore: establish clean R23 repository boundary'; test "$(git rev-list --all --count)" -eq 1; test "$(git rev-list --max-parents=0 --count HEAD)" -eq 1; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test -z "$(git remote)"
```

- [ ] 首次 push 前固定使用官方 `gitleaks 8.30.1` Darwin arm64 二进制，对这个**只有 R23
  root commit** 的全部可达历史扫描。资产固定为
  `gitleaks_8.30.1_darwin_arm64.tar.gz`，SHA-256 固定为
  `b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5`。不得使用 repository
  config/ignore、内联 `gitleaks:allow`、环境配置或浮动工具版本；临时 JSON 必须是空 list，
  命令结束自动清理：

```bash
set -euo pipefail; ROOT=/Users/dala/Desktop/xrr-rewrite-r23; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0; test "$(gh version | awk 'NR == 1 {print $3}')" = 2.89.0; gh auth status --active --hostname github.com; cd "$ROOT"; test "$(git rev-list --all --count)" -eq 1; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test -z "$(git remote)"; for CONTROL in .gitleaks.toml .gitleaksignore; do test ! -e "$CONTROL"; test ! -L "$CONTROL"; done; TOOL_DIR=$(mktemp -d /tmp/xrr-gitleaks-8.30.1.XXXXXX); ARCHIVE="$TOOL_DIR/gitleaks_8.30.1_darwin_arm64.tar.gz"; SCANNER="$TOOL_DIR/gitleaks"; IGNORE="$TOOL_DIR/empty.gitleaksignore"; REPORT="$TOOL_DIR/report.json"; trap 'rm -rf "$TOOL_DIR"' EXIT; gh release download v8.30.1 --repo gitleaks/gitleaks --pattern gitleaks_8.30.1_darwin_arm64.tar.gz --dir "$TOOL_DIR"; printf '%s  %s\n' b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5 "$ARCHIVE" | shasum -a 256 -c -; tar -xzf "$ARCHIVE" -C "$TOOL_DIR" gitleaks; test "$("$SCANNER" version)" = 8.30.1; : > "$IGNORE"; set +e; env -u GITLEAKS_CONFIG -u GITLEAKS_CONFIG_TOML "$SCANNER" --no-banner --no-color --redact=100 --ignore-gitleaks-allow --exit-code=17 --report-format=json --report-path="$REPORT" --gitleaks-ignore-path="$IGNORE" --timeout=3600 git --log-opts='--full-history --all --diff-filter=tuxdb' "$ROOT"; SCAN_STATUS=$?; set -e; test "$SCAN_STATUS" -eq 0; test "$(python3.12 -c 'import json,sys; rows=json.load(open(sys.argv[1], encoding="utf-8")); print(len(rows)) if isinstance(rows, list) else sys.exit(1)' "$REPORT")" -eq 0; test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

  同时人工审阅首个 commit 的文档、图片、证书/密钥扩展名和第三方 license。发现 credential、
  本机私有信息、未获准数据或许可证冲突时停止；修正待发布内容并重新创建尚未发布的独立
  R23 repository，不增加 ignore/allow，不把问题提交先 push 后删除。

- [ ] Gitleaks GREEN 后才创建固定目标 `zxc-1/xrr-fitter`。创建命令不添加 README、license、
  `.gitignore`、template 或初始 commit；repository 必须为 private，并关闭未使用的 Issues/Wiki。
  active account 必须精确为 `zxc-1` 且具有 `ADMIN` 权限。只添加 canonical HTTPS `origin`，
  只 push `r23-clean-architecture`。中断重跑只接受目标仍不存在、目标仍无 ref，或远端唯一
  ref 已精确等于本地 root commit；任何 R22 ref、额外 ref、错误 URL 或分叉都失败：

```bash
set -euo pipefail
test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"
test -z "${GH_HOST-}" || test "$GH_HOST" = github.com
export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0
ROOT=/Users/dala/Desktop/xrr-rewrite-r23
GITHUB_REPOSITORY=zxc-1/xrr-fitter
test "$(gh version | awk 'NR == 1 {print $3}')" = 2.89.0
gh auth status --active --hostname github.com
test "$(gh api user --jq .login)" = zxc-1
gh auth setup-git --hostname github.com
cd "$ROOT"
test "$(git branch --show-current)" = r23-clean-architecture
test "$(git rev-list --all --count)" -eq 1
test "$(git rev-list --max-parents=0 --count HEAD)" -eq 1
test -z "$(git status --porcelain=v1 --untracked-files=all)"
set +e
REPO_PROBE=$(gh api --include "repos/$GITHUB_REPOSITORY" 2>&1)
REPO_STATUS=$?
set -e
case "$REPO_STATUS" in
  0) : ;;
  1)
    test "$(printf '%s\n' "$REPO_PROBE" | sed -n '1p')" = 'HTTP/2.0 404 Not Found'
    gh repo create "$GITHUB_REPOSITORY" --private --disable-issues --disable-wiki --description 'X-ray reflectivity fitting application'
    ;;
  *)
    printf 'unexpected repository probe status: %s\n' "$REPO_STATUS" >&2
    exit 1
    ;;
esac
test "$(gh repo view "$GITHUB_REPOSITORY" --json nameWithOwner --jq .nameWithOwner)" = "$GITHUB_REPOSITORY"
test "$(gh repo view "$GITHUB_REPOSITORY" --json isPrivate --jq .isPrivate)" = true
test "$(gh repo view "$GITHUB_REPOSITORY" --json archivedAt --jq '.archivedAt == null')" = true
test "$(gh repo view "$GITHUB_REPOSITORY" --json isMirror --jq .isMirror)" = false
test "$(gh repo view "$GITHUB_REPOSITORY" --json viewerPermission --jq .viewerPermission)" = ADMIN
test "$(gh repo view "$GITHUB_REPOSITORY" --json hasIssuesEnabled --jq .hasIssuesEnabled)" = false
test "$(gh repo view "$GITHUB_REPOSITORY" --json hasWikiEnabled --jq .hasWikiEnabled)" = false
REMOTE_URL="https://github.com/$GITHUB_REPOSITORY"
test "$(gh repo view "$GITHUB_REPOSITORY" --json url --jq .url)" = "$REMOTE_URL"
REMOTE_NAMES=$(git remote | LC_ALL=C sort)
case "$REMOTE_NAMES" in
  "")
    test "$(gh repo view "$GITHUB_REPOSITORY" --json isEmpty --jq .isEmpty)" = true
    git remote add origin "$REMOTE_URL"
    ;;
  origin)
    test "$(git remote get-url origin)" = "$REMOTE_URL"
    ;;
  *)
    printf 'unexpected local remotes: %s\n' "$REMOTE_NAMES" >&2
    exit 1
    ;;
esac
HEAD_COMMIT=$(git rev-parse HEAD)
EXPECTED_REFS=refs/heads/r23-clean-architecture
REMOTE_HEAD=$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')
REMOTE_REFS=$(git ls-remote --refs origin | awk '{print $2}' | LC_ALL=C sort)
if test -z "$REMOTE_HEAD" && test -z "$REMOTE_REFS"; then
  test "$(gh repo view "$GITHUB_REPOSITORY" --json isEmpty --jq .isEmpty)" = true
  git push origin HEAD:refs/heads/r23-clean-architecture
elif test "$REMOTE_HEAD" = "$HEAD_COMMIT" && test "$REMOTE_REFS" = "$EXPECTED_REFS"; then
  :
else
  printf 'unexpected remote R23 state\n' >&2
  exit 1
fi
REMOTE_HEAD=$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')
REMOTE_REFS=$(git ls-remote --refs origin | awk '{print $2}' | LC_ALL=C sort)
test "$REMOTE_HEAD" = "$HEAD_COMMIT"
test "$REMOTE_REFS" = "$EXPECTED_REFS"
test "$(gh repo view "$GITHUB_REPOSITORY" --json isEmpty --jq .isEmpty)" = false
git fetch --no-tags origin refs/heads/r23-clean-architecture:refs/remotes/origin/r23-clean-architecture
git branch --set-upstream-to=origin/r23-clean-architecture r23-clean-architecture
gh repo edit "$GITHUB_REPOSITORY" --default-branch r23-clean-architecture
test "$(gh repo view "$GITHUB_REPOSITORY" --json defaultBranchRef --jq .defaultBranchRef.name)" = r23-clean-architecture
test "$(git rev-parse '@{upstream}')" = "$HEAD_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

这是 workflow 建立前唯一一次只做远端 object 校验的 push。远端从第一个 object 起就是 R23；
没有 `integration-r22` branch、`R22-final` tag、R22 Release 或 R22 commit ancestry。

### 任务 2：建立打包、质量门禁与 R22 参考

**创建：** `pyproject.toml`、`MANIFEST.in`、`README.md`、`AGENTS.md`、
`verification/release-spec.json`、`docs/architecture/r22-r23-test-ledger.csv`、空的
`src/xrr_fitter/__init__.py`、`tests/__init__.py`、`tests/conftest.py`、
`tests/outcome_gate.py`、`tools/check_radon.py`、`tools/check_hygiene.py`、
`tools/collect_test_manifest.py`、`tools/validate_test_ledger.py`、
`tools/build_r22_reference.py`、`tools/compare_r22_reference.py`、
`tools/lock_environment.py`、`tools/build_release_spec.py`、`tools/verify.py`、
空的 `tools/reference_groups/__init__.py`、`tools/reference_groups/registry.py`、
`tests/unit/tools/{test_check_radon,test_hygiene,test_collect_test_manifest,test_validate_test_ledger,test_build_r22_reference,test_compare_r22_reference,test_lock_environment,test_build_release_spec,test_verify}.py`、
`tests/architecture/{test_removed_legacy_modules,test_dependency_rules,test_naming_rules,test_quality_gate}.py`
和初始 `.github/workflows/verify.yml`。此任务不创建 `api.py`、
`__main__.py` 或行为 stub。

- [ ] 为 Radon checker、hygiene checker、lock generator、机器可读 collector、
  reference builder/comparator、ledger validator、pinned sdist metadata fixture 和 unified
  verifier 编写测试；由于这些工具尚不存在，运行精确 tool test path
  并获得 RED。
- [ ] `tests/unit/tools/test_lock_environment.py` 对 `--check` 固化合法 canonical lock、
  合法且与 pyproject 精确相等的 refnx full-commit direct reference、editable/local URL/
  绝对路径/未固定或未声明 VCS/错误 URL 或 commit/非 exact requirement/重复/乱序污染、缺失路径、
  directory、mocked `PermissionError`、非法 UTF-8 和 requirement parse error；所有 I/O 或
  parser error 都必须是明确非零，不能因检查器自身失败而误报通过。另覆盖 resolver
  failure 不覆盖原 lock、成功输出原子替换和同一输入双跑字节稳定。
- [ ] 严格按第 9 节实现 Radon checker。
- [ ] 将 `collect_test_manifest.py` 实现为 pytest collection plugin，写入包含 schema、显式
  `source_commit`、test-tree path/size/SHA-256、suite、按 node ID 排序的
  `{nodeid, markers}` record、count、Python version/platform/lock SHA-256 和 collection SHA-256 的
  canonical JSON；不记录生成时间、当前 HEAD 或临时路径，因此 metadata-only commit 后可
  重建出相同字节。R22 使用 `--expected-tag` 绑定 source commit，R23 使用
  `--source-commit` 并要求该 commit 到当前 HEAD 的 `tests/` diff 为空。它绝不解析
  terminal text，并始终向 pytest 传入 `-o addopts=`、`--strict-config`、
  `--strict-markers`、`-p no:cacheprovider` 和 `--collect-only`，使 R22 默认
  `-m 'not slow'` 无法漏收 slow node，同时保留严格配置/marker 检查。collector 覆盖而非
  继承调用者 `PYTHONPATH`：R22 精确为解析后的 `--repo-root`，R23 精确为
  解析后的 `--repo-root/src`；两者都不接受单独的 import-root CLI 覆盖。
  为污染的 caller PYTHONPATH、unknown marker、default marker 被清空、slow node 仍出现、
  marker metadata、source-commit guard、无 cache 写入和 manifest 字节稳定性编写单测。
  manifest 不记录 executable path、venv path、repo absolute path、cwd 或时间；同一
  fixture repo 从两个不同绝对路径的 venv 收集必须产生相同 bytes。
  `--output` 的父目录必须已存在且不是 symlink；缺失时硬失败，不自动创建
  方案未声明的目录，不留 partial file。
- [ ] collector 在读取 R22 前后都硬断言 source `HEAD == R22-final^{commit}` 且 `git status --porcelain=v1 --untracked-files=all` 为空。根据 R22 archive 中已固化的 dependency-only collection lock，在仓库外创建专用环境；不得复用活动 runner 的 `.venv`，也不得 editable-install R22：

```bash
test ! -e /Users/dala/Desktop/xrr-r22-collection-venv && python3.12 -m venv /Users/dala/Desktop/xrr-r22-collection-venv && /Users/dala/Desktop/xrr-r22-collection-venv/bin/python -m pip install pip==26.1.2 && /Users/dala/Desktop/xrr-r22-collection-venv/bin/python -m pip install -r /Users/dala/Desktop/xrr-rewrite-design-integration/.integration/manifests/r22-collection-requirements.lock
```

- [ ] 使用专用 R22 collection interpreter 运行 collector；collector 以 `--repo-root` 为 cwd 进行全量 collect，但绝不执行测试 body：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && test ! -e verification && mkdir -p verification/r22/collections && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-design-integration/xrr_fitter /Users/dala/Desktop/xrr-r22-collection-venv/bin/python tools/collect_test_manifest.py --repo-root /Users/dala/Desktop/xrr-rewrite-design-integration/xrr_fitter --expected-tag R22-final --lock-file /Users/dala/Desktop/xrr-rewrite-design-integration/.integration/manifests/r22-collection-requirements.lock --suite tests --output verification/r22/collections/tests-active.json && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-design-integration/xrr_fitter /Users/dala/Desktop/xrr-r22-collection-venv/bin/python tools/collect_test_manifest.py --repo-root /Users/dala/Desktop/xrr-rewrite-design-integration/xrr_fitter --expected-tag R22-final --lock-file /Users/dala/Desktop/xrr-rewrite-design-integration/.integration/manifests/r22-collection-requirements.lock --suite tests_r21 --output verification/r22/collections/tests-r21.json
```

- [ ] 根据这两个全量、带 hash 的 manifest 填充 `docs/architecture/r22-r23-test-ledger.csv`。R21 non-slow subset 从全量 manifest 按已固化 marker metadata 显式派生，再与其 428-node manifest 对账；不得再次调用带默认 `addopts` 的 pytest 来派生 subset。
- [ ] 实现 `tools/validate_test_ledger.py` 的两个且仅两个 phase。
  `source-draft` 校验精确 CSV header、UTF-8/LF、两份 source manifest schema/hash、
  `source_tree` 只能为 `tests`/`tests_r21`、source key 与 ledger 完全相等且每个恰好一行、
  `contract_id` 匹配 `[a-z][a-z0-9_.-]*`、`reason` 为去除首尾空白后的非空文本、
  action 为 `port`/`rewrite`/`merge`/`delete_layout_only`，以及 `target_nodeids` 为非空、去重、
  按字典序排列的 canonical JSON string array；此 phase 不声称尚未创建的
  target 已存在。`final` 在全部上述检查后，还要求每个 target 存在于
  R23 manifest，且 `delete_layout_only` 的每个 target 都位于
  `tests/architecture/`。两个 phase 都必须显式接收两份 source manifest 和 ledger；
  `final` 再强制显式 `--target-manifest`，不搜索默认路径。
- [ ] `tests/unit/tools/test_validate_test_ledger.py` 为每项 schema、覆盖、重复、
  canonical JSON、action、target existence 和 `delete_layout_only` 路径错误分别写失败
  case。初始 ledger 填完后，在下方 R23 lock environment 创建完成后运行
  `source-draft` 验证。
- [ ] 将 `build_r22_reference.py` 实现为纯数据 normalizer。它只读取 `xrr-r22-final.tar.gz`、外部 freeze receipt、archive 内的 release identity/product manifest/reference bundle，以及 approved-data record；先验证 archive SHA-256、bundle/tool/source hash 和 `R22-final` commit，再归一化。它绝不导入或启动 R22 Python，也不从 canonical summary 补猜不存在的 field。
  该 builder 原子拥有且只拥有 `verification/r22/reference/`；collector 原子
  拥有且只拥有 `verification/r22/collections/`。两者均拒绝已存在的非空
  output dir，不会改写兄弟子树。
- [ ] 要求归一化 manifest 包含非空的 `model_project`、`io`、`physics`、`fit_compile`、`fit_search`、`analysis`、`services` 和 `gui` artifact group。记录 builder file SHA-256 和每个 source artifact SHA-256。
- [ ] Task 2 的 comparator 只实现 strict manifest/NPZ parser、comparison-policy engine、
  `--self-check` 和空 group registry。unit test 证明 self-check 成功，任意
  `--group` 在未注册时硬失败，registry 不做 module discovery。
- [ ] 将 `tools/verify.py` 实现为唯一 subprocess-checking 入口。初始 mode 为 `quality` 和 `tools`；后续任务把 suite 添加到同一个显式 registry。
  `test_verify.py` 在两个不同绝对路径的 fixture repo 运行它，要求每次都从
  `Path(__file__)` 得到各自 root，不出现 Desktop 硬编码或继承的 `PYTHONPATH`。
- [ ] root `AGENTS.md` 只写稳定路标：Python 3.12/src layout、唯一公共 API/
  入口、`docs/architecture/r23-clean-break.md` 的依赖图入口、禁止 legacy/shim/
  fallback、每个领域生产与测试同批、`tools/verify.py` 和
  `tools/check_radon.py` 命令。不复制本方案全文、Task 0-14 流水账、临时路径
  或 R22 状态。
- [ ] 注册 `tests/outcome_gate.py`；所有由 `tools/verify.py` 启动的 pytest mode 默认在出现 skip、xfail、xpass 或意外 deselection 时失败。
- [ ] 将 `tests/conftest.py` 限定为 marker/option 注册和真正跨 suite 的
  fixture；Task 2 一次创建并测试，后续任务不再建第二个 `conftest.py`。
- [ ] 创建 `test_dependency_rules.py`、`test_removed_legacy_modules.py`、
  `test_naming_rules.py` 和 `test_quality_gate.py`；前三者分别固化依赖 allowlist、Task 1
  legacy-absence guard 和第 3 节命名/空-init 规则。`test_removed_legacy_modules.py` 还必须
  读取当前 repository 的唯一 root commit，断言它没有 parent、root tree 只含 Task 1 声明的
  `.gitignore`/稳定文档、当前 `.git` 是普通 directory，并拒绝 R22 layout 或 linked-worktree
  gitfile；由此把“GitHub 历史从 R23 开始”固化为永久架构合同。
  此时不得创建空的、跳过的或针对 stub 的 `test_public_api.py`/`test_distribution.py`；
  它们由任务 10 在真实实现完成后创建并立即纳入门禁。
- [ ] 初始 `.github/workflows/verify.yml` 只监听精确
  `refs/heads/r23-clean-architecture` push，顶层 `permissions: contents: read`，checkout
  `persist-credentials: false` 且所有 action `uses:` 固定完整 40 位 commit；standard runner
  上只运行当前已存在的 `quality`/`tools`，最后由 `if: always()` 的唯一 `checkpoint` 要求
  两者都为 `success`。设置 job timeout 与 `concurrency.cancel-in-progress: false`，不能取消旧
  SHA run 后只保留新 run。`test_quality_gate.py` 对 trigger、permission、action pin、runner
  label、job registry、checkpoint needs/result 逐项断言。
- [ ] 配置 src packaging 和显式 pytest marker。Task 2 不在 metadata 中发布
  `xrr-fitter`，因为真实 `__main__.py` 尚不存在且禁止 stub。Task 11 第一个
  GUI shell slice 将真实 `src/xrr_fitter/__main__.py`、
  `[project.gui-scripts] xrr-fitter = "xrr_fitter.__main__:main"` 和入口 test 同批提交；
  第十个 slice 在安装后入口存在时才启用完整 distribution smoke。
- [ ] 实现唯一 owner `tools/build_release_spec.py`。它用 TOML/JSON parser 读取
  显式 `--pyproject`、`--lock-file`、`--r22-root`，用 pinned build fixture 双跑固化
  第 8 节 sdist metadata 精确集合，并将
  build-system、runtime/test dependency、lock SHA-256、wheel/sdist content policy 写入
  `verification/release-spec.json`。在 collections 和 reference 两个子树完成后，该文件
  还存储对 `verification/r22/**` 所有 regular file 的
  `(relative_path,size,sha256)` 做 length-framed 得到的 `r22_oracle_tree_sha256`。
  release spec 必须由 unit test 重算，不接受手工增加 metadata glob或自报 hash。
  工具使用同目录临时文件 + `fsync` + `os.replace` 原子替换该文件；
  没有第二个 writer。`test_build_release_spec.py` 覆盖 dependency/lock/tree/metadata
  漂移、非 canonical output、双跑字节稳定和失败不留 partial file。
- [ ] 生成并验证 portable lock，然后在仓库外创建 R23 environment：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && test ! -e /Users/dala/Desktop/xrr-r23-venv && python3.12 tools/lock_environment.py --output requirements-macos-arm64-py312.lock && python3.12 tools/lock_environment.py --check requirements-macos-arm64-py312.lock && python3.12 -m venv /Users/dala/Desktop/xrr-r23-venv && /Users/dala/Desktop/xrr-r23-venv/bin/python -m pip install pip==26.1.2 && /Users/dala/Desktop/xrr-r23-venv/bin/python -m pip install -r requirements-macos-arm64-py312.lock
```

- [ ] 使用刚创建的 lock environment 验证初始 source ledger：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/validate_test_ledger.py --phase source-draft --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --ledger docs/architecture/r22-r23-test-ledger.csv
```

- [ ] 在迁移任何 production domain 前构建并验证 R22 reference：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/build_r22_reference.py --r22-archive /Users/dala/Desktop/xrr-r22-final.tar.gz --freeze-receipt /Users/dala/Desktop/r22-final-freeze.json --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --output verification/r22/reference && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --self-check verification/r22/reference/manifest.json && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/build_release_spec.py --pyproject pyproject.toml --lock-file requirements-macos-arm64-py312.lock --r22-root verification/r22 --output verification/release-spec.json
```

- [ ] 运行：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py quality && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py tools
```

预期：PASS，且没有 ignored Python file、legacy package 或尚未实现能力的 stub。`quality` 此时只运行 Radon、dependency、legacy-absence 和 quality-gate wiring；后续任务只向该显式集合增加已经存在且能通过的测试。

- [ ] 首次 Actions push 前通过 GitHub API 确认 standard runner online 且 label 完整；查询失败、
  runner 多于单页上限或没有匹配项都阻塞：

```bash
set -euo pipefail; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1; GITHUB_REPOSITORY=zxc-1/xrr-fitter; test "$(gh api "repos/$GITHUB_REPOSITORY/actions/permissions" --jq .enabled)" = true; RUNNER_TOTAL=$(gh api "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100" --jq .total_count); test "$RUNNER_TOTAL" -le 100; test "$(gh api "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100" --jq 'any(.runners[]; .status == "online" and ([.labels[].name] | (index("self-hosted") != null and index("macOS") != null and index("ARM64") != null and index("xrr-ci") != null)))')" = true
```

- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add pyproject.toml MANIFEST.in README.md requirements-macos-arm64-py312.lock AGENTS.md src tests tools verification/release-spec.json verification/r22 docs/architecture .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'build: enforce R23 quality and reference gates' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

本提交后第一次执行第 2.1 节完整 GitHub 分支发布门禁；只有 exact-SHA Actions run 为
`success`，才能开始任务 3。此后每个任务中的“提交”都隐含同一强制步骤，不再逐段复制命令。

从任务 3 到任务 13，任何提交只要修改
`docs/architecture/r22-r23-test-ledger.csv`，都必须在 stage 前运行下列完整
`source-draft` 命令；不得用局部 CSV lint 代替：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/validate_test_ledger.py --phase source-draft --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --ledger docs/architecture/r22-r23-test-ledger.csv
```

### 任务 3：迁移不可变模型

**创建：**
`src/xrr_fitter/model/{__init__,data,instrument,structure,parameters,fitting,analysis,project,operations,export}.py`、
`tests/unit/model/{test_prepared_data,test_instrument_values,test_resolution_mapping,test_fit_masks,test_materials,test_structures,test_parameters,test_fitting_values,test_analysis_values,test_project_state,test_operations,test_export_values}.py`、空的
`tests/support/__init__.py`、`tests/support/model_cases.py`、
`tools/reference_groups/model_project.py`。**修改：**
`tools/reference_groups/registry.py`、`tests/unit/tools/test_compare_r22_reference.py`、
`docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
`tests/architecture/test_quality_gate.py` 和 `.github/workflows/verify.yml`。

- [ ] 同批创建并取得 RED：`test_prepared_data.py`、`test_instrument_values.py`、
  `test_resolution_mapping.py`、`test_fit_masks.py`、`test_materials.py`、
  `test_structures.py`、`test_parameters.py`、`test_fitting_values.py`、
  `test_analysis_values.py`、`test_project_state.py`、`test_operations.py` 和
  `test_export_values.py`。它们覆盖 immutable validation、defensive copy、
  serialization、`ProjectFitResult`/`OperationEvent`/`OperationError` schema 以及
  export manifest/file-record value；不得把 fitting/analysis/operation/export value
  测试推迟到 I/O/service 任务。
- [ ] 因缺少 type，运行目标测试并获得 RED。
- [ ] 使用 ledger 中记录的 source path 从只读 R22 仓库提取并整合不可变 dataclass。例如，使用
  `git -C /Users/dala/Desktop/xrr-rewrite-design-integration show R22-final:xrr_fitter/xrr/_data_types.py`
  读取当前 prepared-data definition；不得在 R23 中假设存在 `R22-final` ref，更不能复制完整旧 package。
- [ ] 保留 defensive copy、只读 NumPy array、finite validation 和 serialization field value。
- [ ] 除已持久化的 R22 workspace field 外，不得将 GUI widget state 放入 domain state；将这些 field 存储在命名清晰的 project workspace value 中。
- [ ] 运行 model test、architecture 和全量 Radon，达到 GREEN。
- [ ] 实现并注册 `model_project` actual adapter；unit test 先确认未注册 RED，
  再确认该 adapter 只从 manifest input 构造本任务的 model value、拒绝多/少
  field，且不导入 `tests.support`。`tests/support/model_cases.py` 只服务测试。
- [ ] 运行 `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group model_project`，达到 GREEN。
- [ ] 为每个已迁移的 model behavior 更新 ledger。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/model tests/unit/model tests/support/__init__.py tests/support/model_cases.py tools/reference_groups/model_project.py tools/reference_groups/registry.py tests/unit/tools/test_compare_r22_reference.py docs/architecture/r22-r23-test-ledger.csv tools/verify.py tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: establish immutable R23 domain models' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 4：迁移数据与项目 I/O

**创建：** 空的 `src/xrr_fitter/io/__init__.py`、`xy.py`、`source.py`、
  `project_codec.py`、
  `tests/fixtures/source/header_and_duplicates.xy`、`tests/unit/io/test_xy_reader.py`、
  `test_source_validation.py` 和 `test_project_codec.py`。API-level
  project roundtrip integration test 延后到任务 10。同批创建
  `tools/reference_groups/io.py`。**修改：** `tools/reference_groups/registry.py`、
  `tests/unit/tools/test_compare_r22_reference.py`、
  `docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
  `tests/architecture/test_quality_gate.py` 和 `.github/workflows/verify.yml`。

- [ ] 用上述三个文件为 XY parsing、重复项合并、mask、相对路径、source
  hash、R22 project round trip 和原子保存编写 RED 测试。Project state invalidation
  不属于 I/O，由任务 10 的 service test 承担。
- [ ] 从只读 R22 仓库的
  `R22-final:xrr_fitter/tests/fixtures/header_and_duplicates.xy` 恢复唯一解析 fixture，
  要求 bytes/SHA-256 与 tag 中对象完全一致；不得手工重写或把 fixture 内嵌进 test。
- [ ] 实现唯一主要的 R22 兼容 codec；不得添加 legacy codec selector。
- [ ] 集中实现 `dataset_by_id`、`dataset_index` 和 `resolve_source_path`，不得在 GUI 中重复实现。
- [ ] 将序列化后的归一化输出与 R22 fixture 对比。
- [ ] 先用 registry test 取得 `io` 未注册 RED，再实现只调用当前 I/O/model
  的 actual adapter。运行 focused test、architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group io`，达到 GREEN。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/io tests/unit/io tests/fixtures/source/header_and_duplicates.xy tools/reference_groups/io.py tools/reference_groups/registry.py tests/unit/tools/test_compare_r22_reference.py docs/architecture/r22-r23-test-ledger.csv tools/verify.py tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate data and project persistence' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 5：迁移物理计算

**创建：**
`src/xrr_fitter/physics/{__init__,materials,stack,parratt,resolution,footprint,reflectivity,derivatives,sld_profile}.py`、
`tests/unit/physics/{test_material_sld,test_stack_expansion,test_resolution,test_instrument_model,test_reflectivity,test_periodic_reflectivity,test_derivatives,test_sld_profile}.py`、
`tests/regression/test_numerical_reference.py`、
`tools/reference_groups/physics.py`。**修改：** `tools/reference_groups/registry.py`、
`tests/unit/tools/test_compare_r22_reference.py`、
`docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
`tests/architecture/test_quality_gate.py` 和 `.github/workflows/verify.yml`。

- [ ] 为 SLD、stack expansion、Parratt、roughness、resolution escalation、mixed K-alpha、footprint/background order、derivative 和 SLD profile 编写 RED 测试。
- [ ] 整合 numerical leaf function，但不得改变 operation order。
- [ ] 运行固定版本的 refnx parity 和精确 R22 normalized reference 对比。
- [ ] 将任何 numerical drift 视为缺陷；结构迁移期间不得放宽 tolerance。
- [ ] 先取得 `physics` 未注册 RED，再实现 actual adapter。运行 physics test、
  numerical regression、architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group physics`，达到 GREEN。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/physics tests/unit/physics tests/regression/test_numerical_reference.py tools/reference_groups/physics.py tools/reference_groups/registry.py tests/unit/tools/test_compare_r22_reference.py docs/architecture/r22-r23-test-ledger.csv tools/verify.py tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: consolidate XRR physics engine' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 6：迁移拟合问题与求值

**创建：** `src/xrr_fitter/evaluation.py`、
`src/xrr_fitter/fit/{__init__,objective,parameters,problem,initialization,screening,candidates}.py`
（其中 `__init__.py` 为 0 bytes）、
`tests/unit/test_evaluation.py`、
`tests/unit/fit/{test_problem_compilation,test_objective,test_feature_detection,test_candidate_initialization,test_screening}.py`，以及
`tools/reference_groups/fit_compile.py`。**修改：** `tools/reference_groups/registry.py`、
`tests/unit/tools/test_compare_r22_reference.py`、
`docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
`tests/architecture/{test_dependency_rules,test_quality_gate}.py` 和
`.github/workflows/verify.yml`。

- [ ] 先创建并运行上述六个 behavior test 文件，因共享求值函数和 fit
  compiler/objective 尚未存在而取得 RED。
- [ ] 迁移 parameter coordinate encoding、compile rule、scale prior、objective、warning、invalid candidate handling 和解析 Jacobian。
- [ ] 将 fit/analysis 共用的 coordinate、residual/loss/likelihood 和 Jacobian 链式组合
  一次性落到 `evaluation.py`，用 `tests/unit/test_evaluation.py` 直接覆盖。fit 包只
  保留 problem compilation 和 `fit/objective.py` 中的优化目标策略；不得创建
  `fit/evaluation.py`、`fit/jacobian.py`，也不把这些函数放进 physics。
- [ ] 移除 facade-global monkeypatch seam 和包含 49 个 callable 的 operation object。
- [ ] 意外 evaluation error 必须向上传播；只有已有文档说明的物理约束失败才会生成 invalid candidate。
- [ ] 将 compiled array、parameter order、objective、Jacobian 和 warning 与 R22 对比。
- [ ] 先取得 `fit_compile` 未注册 RED，再实现 actual adapter。运行
  focused test、architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group fit_compile`，达到 GREEN。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/evaluation.py src/xrr_fitter/fit tests/unit/test_evaluation.py tests/unit/fit tools/reference_groups/fit_compile.py tools/reference_groups/registry.py tests/unit/tools/test_compare_r22_reference.py docs/architecture/r22-r23-test-ledger.csv tools/verify.py tests/architecture/test_dependency_rules.py tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate fit compilation and evaluation' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 7：迁移搜索、流水线、检查点与联合拟合

**创建：**
`src/xrr_fitter/fit/{local_search,global_search,stages,pipeline,checkpoint,resume,joint_problem,joint_sharing,joint_evaluation,joint_pipeline}.py`、
`tests/unit/fit/{test_local_solver,test_global_solver,test_stage_search,test_resume,test_checkpoint,test_candidate_ranking,test_joint_problem,test_joint_evaluation,test_joint_pipeline}.py`、
`tests/regression/test_recovery_metrics.py`、`tests/support/recovery_cases.py`、
`tools/reference_groups/fit_search.py`。**修改：** `tools/reference_groups/registry.py`、
`tests/unit/tools/test_compare_r22_reference.py`、
`src/xrr_fitter/fit/candidates.py`、
`docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
`tests/architecture/test_quality_gate.py` 和 `.github/workflows/verify.yml`。
不得在 `fit` 中创建 process/queue module。

- [ ] 为 deterministic seed、cancellation polling、纯 handler picklability、stage ordering、candidate ranking、checkpoint callback、resume mismatch rejection 和 joint sharing 编写 RED 测试。
  `tests/unit/fit/test_joint_evaluation.py` 必须直接覆盖 joint objective 和 analytic
  derivative；service 测试不代替这两项纯 fit 领域合同。
- [ ] 按职责整合当前 Stage-E 和 resume 微文件，不得复刻每个 helper 一个文件的结构。
- [ ] 为后续 service-owned worker boundary 返回 pickle-safe request/result value。
- [ ] 保持 independent 和 joint mode 显式；未知 mode 必须抛出异常，绝不 fallback 到 independent。
- [ ] 将 stage history、lineage、ranking 和 checkpoint 与 R22 对比。
- [ ] 先取得 `fit_search` 未注册 RED，再实现 actual adapter。运行 fit test、
  recovery regression、architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group fit_search`，达到 GREEN。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/fit/{candidates,local_search,global_search,stages,pipeline,checkpoint,resume,joint_problem,joint_sharing,joint_evaluation,joint_pipeline}.py tests/unit/fit/{test_local_solver,test_global_solver,test_stage_search,test_resume,test_checkpoint,test_candidate_ranking,test_joint_problem,test_joint_evaluation,test_joint_pipeline}.py tests/regression/test_recovery_metrics.py tests/support/recovery_cases.py tools/reference_groups/fit_search.py tools/reference_groups/registry.py tests/unit/tools/test_compare_r22_reference.py docs/architecture/r22-r23-test-ledger.csv tools/verify.py tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: consolidate deterministic fitting pipeline' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 8：迁移分析模块

**创建：**
`src/xrr_fitter/analysis/{__init__,classification,profiles,binary_profiles,derivatives,bootstrap,mcmc,diagnostics,report}.py`、
`src/xrr_fitter/model/provenance.py`、
`tests/unit/analysis/{test_classification,test_profiles,test_binary_profiles,test_derivatives,test_bootstrap,test_mcmc,test_diagnostics,test_report}.py`、
`tests/regression/test_profile_basin_regressions.py`、
`tests/unit/fit/test_frozen_stage_search.py`、
`tools/reference_groups/{analysis,fit_search}.py`。**修改：**
`src/xrr_fitter/evaluation.py`、
`src/xrr_fitter/model/{analysis,fitting,parameters,project}.py`、
`src/xrr_fitter/physics/stack.py`、
`src/xrr_fitter/fit/{candidates,checkpoint,joint_pipeline,local_search,pipeline,problem,stages}.py`、
`tests/unit/test_evaluation.py`、
`tests/unit/model/{test_analysis_values,test_fitting_values,test_project_state}.py`、
`tests/unit/fit/{test_joint_evaluation,test_joint_pipeline,test_objective,test_problem_compilation,test_stage_search}.py`、
`tools/reference_groups/{fit_compile,registry}.py`、
`tests/unit/tools/test_compare_r22_reference.py`、
`tests/unit/tools/test_verify.py`、
`docs/architecture/{r22-r23-test-ledger.csv,r23-clean-break.md}`、
`tools/verify.py` 和 `tests/architecture/test_dependency_rules.py`。

- [ ] 为 classification、profile、binary-derived profile、gradient、bootstrap、MCMC、diagnostic 和 report 编写 RED 测试。
- [ ] 迁移 analysis，且不得从 `fit` 进行任何 import。
- [ ] 直接消费任务 3 已实现的 model value 和任务 6 的
  `evaluation.py` 纯函数。前置合同复查已确认 Task 7 handoff 缺少 typed immutable
  evaluation context、完整 evidence provenance 和 fit-owned profile continuation；因此本任务
  只在上述扩展路径补齐这些共享/fit-owned 合同。`analysis` 仍不得导入或调用 `fit`，fit
  仍不得导入 `analysis`，不得将 likelihood/profile evaluation 下沉到 `physics`，也不得
  引入 callable bag。
- [ ] 按语义迁移 R21 独有的 reconvergence 和 profile-basin 行为。
- [ ] 保留 deterministic stochastic stream 和 cancellation，同时为 service-owned worker boundary 返回 pickle-safe request/result value。
- [ ] 先取得 `analysis` 未注册 RED，再实现 actual adapter。运行 analysis
  test、regression、architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group analysis`，达到 GREEN。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add docs/architecture/r22-r23-test-ledger.csv docs/architecture/r23-clean-break.md src/xrr_fitter/analysis src/xrr_fitter/evaluation.py src/xrr_fitter/fit/candidates.py src/xrr_fitter/fit/checkpoint.py src/xrr_fitter/fit/joint_pipeline.py src/xrr_fitter/fit/local_search.py src/xrr_fitter/fit/pipeline.py src/xrr_fitter/fit/problem.py src/xrr_fitter/fit/stages.py src/xrr_fitter/model/analysis.py src/xrr_fitter/model/fitting.py src/xrr_fitter/model/parameters.py src/xrr_fitter/model/project.py src/xrr_fitter/model/provenance.py src/xrr_fitter/physics/stack.py tests/architecture/test_dependency_rules.py tests/regression/test_profile_basin_regressions.py tests/unit/analysis tests/unit/fit/test_frozen_stage_search.py tests/unit/fit/test_joint_evaluation.py tests/unit/fit/test_joint_pipeline.py tests/unit/fit/test_objective.py tests/unit/fit/test_problem_compilation.py tests/unit/fit/test_stage_search.py tests/unit/model/test_analysis_values.py tests/unit/model/test_fitting_values.py tests/unit/model/test_project_state.py tests/unit/test_evaluation.py tests/unit/tools/test_compare_r22_reference.py tests/unit/tools/test_verify.py tools/reference_groups/analysis.py tools/reference_groups/fit_compile.py tools/reference_groups/fit_search.py tools/reference_groups/registry.py tools/verify.py && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: separate uncertainty analysis from fitting' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 9：迁移导出与示例的纯 I/O 能力

**创建：** `src/xrr_fitter/io/export_run.py`、`export_tables.py`、`export_plots.py`、
`examples.py`，`examples/{single-layer.xy,single-layer.xrrproj.json,mo-si-periodic.xy,mo-si-periodic.xrrproj.json}`，
以及 `tests/unit/io/{test_export_run,test_export_tables,test_export_plots,test_examples}.py`。
**修改：** `docs/architecture/r22-r23-test-ledger.csv`。
此任务不创建 `services/exports.py` 或公共 API stub。

- [ ] 为防冲突 publication、traversal safety、partial-directory rollback、deterministic table/plot、manifest completeness 和 deterministic example generation 编写 RED 测试。
- [ ] 使用唯一原子 publication path 实现 export I/O，且 cleanup failure 不得静默处理。
- [ ] `io.examples` 只提供 `build_single_layer_example()`、
  `build_mo_si_periodic_example()` 和唯一 publication 函数
  `write_examples(destination: Path) -> tuple[Path, ...]`。前两个 builder 构造 model value；
  writer 必须复用 `io.xy` 和 `io.project_codec`，按上述四个固定相对路径原子发布并返回排序
  path tuple，不另写一套 XY/JSON serializer。`test_examples.py` 在两个临时目录生成后比较
  四个文件逐字节相同，重新读取两个 project 并校验相对 data source、预测曲线和 manifest，
  同时证明目标目录有未知成员或任一写入失败时不留下半套示例。
- [ ] 用下面的唯一生成命令创建 root `examples/` 四个受审产物；提交前要求目录 regular-file
  集合精确等于这四项。不得把 R7 runtime copy、一次性调试 script、Python runner 或绝对
  路径带入 R23：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python -c 'from pathlib import Path; from xrr_fitter.io.examples import write_examples; write_examples(Path("examples"))'
```
- [ ] 运行 export-I/O/example unit test、当前已存在的 architecture test 和全量 Radon，达到 GREEN。
- [ ] 更新 `tests/test_export.py`、`tests/test_examples.py` 对应的 ledger row；integration export workflow 在任务 10 由完整 API 接管。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/io/export_run.py src/xrr_fitter/io/export_tables.py src/xrr_fitter/io/export_plots.py src/xrr_fitter/io/examples.py src/xrr_fitter/io/xy.py tests/unit/io/test_export_run.py tests/unit/io/test_export_tables.py tests/unit/io/test_export_plots.py tests/unit/io/test_examples.py tests/unit/io/test_xy_reader.py tests/architecture/test_dependency_rules.py examples/single-layer.xy examples/single-layer.xrrproj.json examples/mo-si-periodic.xy examples/mo-si-periodic.xrrproj.json docs/architecture/r22-r23-test-ledger.csv docs/architecture/r23-clean-break.md && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate deterministic export and example IO' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 10：迁移服务层并发布完整公共 API

**创建：**
`src/xrr_fitter/services/{__init__,datasets,structures,parameters,projects,fitting,batch,exports,workers}.py`、
`src/xrr_fitter/api.py`、
`tests/unit/services/{test_datasets,test_structures,test_parameters,test_projects,test_fitting,test_joint_registry,test_independent_batch,test_exports,test_workers}.py`、
`tests/integration/{test_project_roundtrip,test_single_fit_workflow,test_joint_fit_workflow,test_batch_resume,test_export_workflow,test_process_workers}.py`、
`tests/support/processes/{__init__.py,run_fit_worker.py,run_analysis_worker.py}`（其中
`__init__.py` 为 0 bytes）、`tests/architecture/{test_public_api,test_distribution}.py` 和
`tools/reference_groups/services.py`。**修改：**
`tools/reference_groups/registry.py`、`tests/unit/tools/test_compare_r22_reference.py`、
`tests/unit/tools/test_verify.py`、`tools/verify.py`、
`tests/architecture/{test_dependency_rules,test_quality_gate}.py`、
`.github/workflows/verify.yml`、`docs/algorithm.md` 和
`docs/architecture/r22-r23-test-ledger.csv`。完整的 installed-entrypoint verifier 等真实
`__main__` 在任务 11 出现后再创建。

- [ ] 为 source workflow、duplicate-stem ID 分配、structure change、oxide suggestion、invalidation、parameter reconciliation、纯 sharing validation、显式 fit preflight、fitting composition、batch dispatch 和 export dispatch 编写 RED 测试；oxide suggestion 归 `tests/unit/services/test_structures.py`，不再创建同义测试文件。
- [ ] 将 service operation 实现为 project state mutation 的唯一负责人；`services/exports.py` 只组合任务 9 已完成的 I/O capability。
- [ ] 在 `tests/unit/services/test_datasets.py` 固化 `stem`、`stem-2`、`stem-3` 顺序、
  persisted-ID namespace、display-name independence 和 remove 后最低可用 suffix 行为；
  `tests/integration/test_project_roundtrip.py` 证明保存/加载不重新编号。不得公开 allocator
  function，也不得要求 GUI 传 `dataset_id`。
- [ ] 只在 `services/fitting.py` 中组合 fit 与 analysis。
- [ ] 将 `services/workers.py` 实现为 spawn/queue/cancel/progress framing 的唯一负责人，并通过它路由纯 fit/analysis handler。
- [ ] 单独运行真实 spawn integration，并验证 nonblocking poll、恰好一个 terminal 后接一个 `stopped`、ordering、cooperative cancellation、spawn failure、malformed protocol、force-stop escalation、process reap、queue close 和 failure propagation。
- [ ] 一次性实现并启用第 7 节的完整 API list/signature，包括 `export_result`、`OperationEvent` schema 和 `OperationJob`；此前任何任务不得创建 public API stub。
- [ ] 创建 `tests/architecture/test_public_api.py` 和 `test_distribution.py`，并验证精确 `__all__`、signature、GUI/API import-rule fixture、package content policy 和无 legacy module；实际 GUI tree 的非空扫描由任务 11 启用，不能在 GUI 尚不存在时以空扫描冒充覆盖。
- [ ] 确保 acceptance/integration test 只导入 `xrr_fitter.api`，并运行 single/joint/export/process worker workflow。
- [ ] 重新执行 Task 2 的 lock 污染检查，并要求 `pyproject.toml`、
  `requirements-macos-arm64-py312.lock` 和 `verification/release-spec.json` 在本任务
  均无 diff。依赖和 lock 已由 Task 2 完成；Task 10 不做无意义的 metadata
  重写，也不触发第二个 release-spec writer。
- [ ] 先取得 `services` 未注册 RED，再实现 actual adapter。运行
  service/API/export integration、wheel/sdist member-list content test、
  architecture、Radon 和
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group services`，达到 GREEN。此时不得用缺失的 CLI smoke stub 冒充完整
  distribution mode。
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && /Users/dala/Desktop/xrr-r23-venv/bin/python tools/lock_environment.py --check requirements-macos-arm64-py312.lock && git diff --exit-code -- pyproject.toml requirements-macos-arm64-py312.lock verification/release-spec.json && git diff --check && git add src/xrr_fitter/services src/xrr_fitter/api.py tests/unit/services tests/unit/tools/test_compare_r22_reference.py tests/unit/tools/test_verify.py tests/support/processes tests/architecture/test_dependency_rules.py tests/architecture/test_public_api.py tests/architecture/test_distribution.py tests/architecture/test_quality_gate.py tests/integration/test_project_roundtrip.py tests/integration/test_single_fit_workflow.py tests/integration/test_joint_fit_workflow.py tests/integration/test_batch_resume.py tests/integration/test_export_workflow.py tests/integration/test_process_workers.py tools/reference_groups/services.py tools/reference_groups/registry.py tools/verify.py .github/workflows/verify.yml docs/algorithm.md docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: publish the complete R23 application API' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 11：按功能迁移 GUI

**创建：** `src/xrr_fitter/gui/**`、`src/xrr_fitter/__main__.py`、`tests/gui/**`、
`tests/integration/{test_entrypoints.py,test_gui_project_workflow.py}`、
`tools/verify_distribution.py`、`tests/unit/tools/test_verify_distribution.py`、
`tools/reference_groups/gui.py`。**修改：** `pyproject.toml`、`README.md`、
`verification/release-spec.json`、`tools/reference_groups/registry.py`、
`tools/compare_r22_reference.py`、`tests/unit/tools/test_compare_r22_reference.py`、
`tools/verify.py`、`tests/unit/tools/{test_verify,test_build_release_spec}.py`、
`tests/architecture/{test_distribution,test_public_api,test_dependency_rules,test_quality_gate}.py`、
`.github/workflows/verify.yml`、`docs/architecture/r22-r23-test-ledger.csv`、
`docs/user-guide.md` 和
`docs/images/{gui-dark-1280x760,gui-dark-1600x900-expert,gui-light-1280x760,gui-light-1600x900-expert}.png`。

按以下顺序执行 feature slice：

1. application shell、`ProjectDocument`、MainWindow layout；
2. project action 和 source recovery；
3. data import、instrument 和 mask；
4. structure 和 oxide workflow；
5. parameter table、expert mode 和 sharing；
6. fit controller、spawn worker、progress 和 cancellation；
7. result、candidate 和 MCMC；
8. plot 和 diagnostic interaction；
9. export dialog、accessibility、focus 和 workspace persistence；
10. 完整的一键 project workflow。

每个 slice 均须：

- [ ] 在迁移本 slice 的 test 或写 GUI production 前，将每个旧 action 对照第 7 节 API
  mapping；GUI 只能调用已经存在的 API operation。若旧 GUI 内含 domain mutation，后续
  RED test 必须断言对应 API operation，再由 widget/controller 调用它，不能把 mutation
  搬进 GUI helper。
- [ ] 如果 action 无法映射到现有 API，说明任务 10 的 service/API 合同不完整：立即停止
  当前 slice，此时不得创建该 slice 的 test/production。先把缺失合同、具名 service/API
  test、精确 production/test/stage path 写回任务 10；修正提交显式包含
  `docs/architecture/r23-clean-break.md`，并按 RED -> service/API GREEN -> services
  comparator -> architecture/Radon -> ledger 独立审查。然后从该 slice 的 mapping/RED
  重新开始。不得在 GUI slice 顺手修改 `src/xrr_fitter/services/**` 或扩张现有 stage 命令。
- [ ] 将 behavior test 迁移到使用稳定功能名称的文件，并运行得到 RED。
- [ ] 实现不使用 mixin 或隐式 MainWindow field 的具体 widget/controller。
- [ ] `gui/fitting/controller.py` 只用 `QTimer` 驱动 `OperationJob.poll()` 并把事件投射为 Qt signal；不得创建 `Process`、`Queue`、`Event`，也不得复制 worker protocol state machine。
- [ ] 将 public/dependency architecture test 切换到“GUI tree 必须非空”模式，扫描每个真实 GUI production module，证明其领域 import 只来自 `xrr_fitter.api`；零个扫描目标必须失败。
- [ ] `tests/gui/test_data_import.py` 必须迁移 R22
  `test_import_allocates_duplicate_stem_ids_and_preserves_active_dataset` 合同，断言 GUI 只调用
  `add_dataset` 并渲染 service 返回的 `stem`/`stem-2` ID；不得 monkeypatch 私有 allocator。
- [ ] 运行 focused GUI test，达到 GREEN。
- [ ] 运行全部既有 GUI test、architecture 和全量 Radon，达到 GREEN。
- [ ] 在将旧 R22 等价测试标记为已核销前，更新 ledger。
- [ ] 每个 feature slice 提交一次；不得将整个 GUI 积累到一个 commit 中。首个 GUI
  slice 一次将整个 `tests/gui` 目录注册到 `gui` mode/job，并将
  `tests/integration/test_entrypoints.py` 加入 `integration` 的精确列表。Slice 2-9
  新增的 GUI 测试由已注册目录自动收集，不重写 verifier/CI/quality-gate；
  slice 10 仅因新增 GUI integration path 和 `distribution` mode/job 再修改这三类
  wiring。

每个 slice 的 RED path 和生产所有权固定如下；同一行内的 test 先 RED，
再实现该行 production path，不跨 slice 预建空 module：

| Slice | RED test | Production ownership |
|---|---|---|
| 1 shell/document | `tests/gui/test_project_document.py`, `tests/integration/test_entrypoints.py` | `__main__.py`, `gui/{__init__,application,main_window,document}.py` |
| 2 project/source | `test_project_actions.py`, `test_source_recovery.py` | `gui/project/{__init__,actions,dialogs}.py`, `main_window.py`, `document.py` |
| 3 data/mask | `test_data_import.py`, `test_data_masks.py` | `gui/data/{__init__,panel,import_dialog,mask_editor}.py` |
| 4 structure/oxide | `test_structure_editor.py`, `test_oxide_workflow.py` | `gui/structure/{__init__,panel,editor,dialogs}.py` |
| 5 parameters/expert | `test_parameter_table.py`, `test_parameter_sharing.py`, `test_expert_views.py` | `gui/parameters/{__init__,panel,table,sharing}.py`, `main_window.py` |
| 6 fit lifecycle | `test_fit_controller.py`, `test_fit_progress.py` | `gui/fitting/{__init__,panel,controller,progress}.py` |
| 7 results/MCMC | `test_results.py` | `gui/results/{__init__,panel,candidates,uncertainty}.py` |
| 8 plots | `test_plots.py` | `gui/plots/{__init__,panel,reflectivity,diagnostics,sld,interactions}.py`, `gui/document.py`, `gui/data/{panel,mask_editor}.py`, `gui/main_window.py` |
| 9 export/workspace/a11y | `test_export_dialog.py`, `test_workspace.py`, `test_accessibility.py`, `test_focus_navigation.py` | `gui/export/{__init__,dialog}.py`, `gui/workspace.py`, `gui/accessibility.py` |
| 10 complete workflow | `tests/integration/test_gui_project_workflow.py`, distribution/GUI adapter tests | final orchestration, installed distribution verifier, `reference_groups/gui.py` |

表中省略的 `tests/gui/` 前缀只为提高表格可读性；完整路径就是第 10.3 节
列出的文件，不允许换成 task-number filename。每个子包的 `__init__.py` 由首次拥有该
子包的 slice 创建并保持 0 bytes；不得在更早 slice 预建。每个 slice GREEN 后、stage 前都运行：

Slice 8 经运行时审计后显式拥有绘图投影的同步事务边界：`ProjectDocument` 在发布新项目
前先执行已注册的具体 plot projection，失败时恢复旧投影且不得发成功 signal；dataset tree
和 mask editor 只在该同步投影成功后提交自身可见状态。否则 PySide6 queued/direct signal
slot 的异常不会回传给 `emit()`，无法满足活动数据集、mask 和候选重绘的原子回滚合同。
因此本 slice 可修改上表列出的 `document.py`、data owner 和 `main_window.py`，但不得借此
改动 service、扩张 `api.__all__`，或提前实现 Slice 9/10 的 export/workspace 编排。

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py gui && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py quality && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/validate_test_ledger.py --phase source-draft --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --ledger docs/architecture/r22-r23-test-ledger.csv
```

下列先给出前九个 slice 的精确 stage/commit 边界；第十个 slice 的提交
命令位于其 comparator/distribution gate 之后，以保证执行顺序无歧义。
第 1 个 slice 修改 entrypoint metadata 后和第 10 个 slice 完成全部 package member
后，都先用唯一 writer 原子重建 release spec：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/build_release_spec.py --pyproject pyproject.toml --lock-file requirements-macos-arm64-py312.lock --r22-root verification/r22 --output verification/release-spec.json
```

第 1 个 slice 在 stage 前额外运行 `tools/verify.py integration`，证明新增的
`test_entrypoints.py` 已进入既有 integration mode；不得只跑 GUI mode。

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add pyproject.toml README.md verification/release-spec.json src/xrr_fitter/__main__.py src/xrr_fitter/gui/__init__.py src/xrr_fitter/gui/application.py src/xrr_fitter/gui/main_window.py src/xrr_fitter/gui/document.py tests/gui/test_project_document.py tests/integration/test_entrypoints.py tests/architecture/test_dependency_rules.py tests/architecture/test_public_api.py tests/architecture/test_distribution.py tests/architecture/test_quality_gate.py tests/unit/tools/test_build_release_spec.py tests/unit/tools/test_verify.py tools/verify.py docs/architecture/r22-r23-test-ledger.csv .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: establish R23 GUI shell and entrypoint' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/main_window.py src/xrr_fitter/gui/document.py src/xrr_fitter/gui/project tests/gui/test_project_actions.py tests/gui/test_source_recovery.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI project and source workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/data tests/gui/test_data_import.py tests/gui/test_data_masks.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI data and mask workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/structure tests/gui/test_structure_editor.py tests/gui/test_oxide_workflow.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI structure workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/main_window.py src/xrr_fitter/gui/parameters tests/gui/test_parameter_table.py tests/gui/test_parameter_sharing.py tests/gui/test_expert_views.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI parameter workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/fitting tests/gui/test_fit_controller.py tests/gui/test_fit_progress.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI fit lifecycle' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/results tests/gui/test_results.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI result and MCMC workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/plots src/xrr_fitter/gui/document.py src/xrr_fitter/gui/data/panel.py src/xrr_fitter/gui/data/mask_editor.py src/xrr_fitter/gui/main_window.py tests/gui/test_plots.py docs/architecture/r23-clean-break.md docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI plots and diagnostics' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/export src/xrr_fitter/gui/workspace.py src/xrr_fitter/gui/accessibility.py tests/gui/test_export_dialog.py tests/gui/test_workspace.py tests/gui/test_accessibility.py tests/gui/test_focus_navigation.py docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: migrate GUI export and workspace workflows' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

第一个 shell slice 必须同时创建真实 `src/xrr_fitter/__main__.py`、在
`pyproject.toml` 加入 `[project.gui-scripts] xrr-fitter = "xrr_fitter.__main__:main"`，
先以 `tests/integration/test_entrypoints.py` subprocess test 取得 RED，再实现
`freeze_support()`、`--help` 和 GUI launch；`--help` 在导入 PySide6 前
成功退出。版本只由 `pyproject.toml` 和已安装 distribution metadata 表示，不在空
`__init__.py` 或 `__main__.py` 复制 `__version__`。第十个 slice 完成后，先以
`tests/unit/tools/test_verify_distribution.py` 取得 RED，再创建最终
`tools/verify_distribution.py`。该文件唯一拥有 `ArtifactManifest` strict parser、从实际
wheel/sdist 计算 canonical value 的 pure calculator 和同目录 staging + `fsync` + rename
atomic writer；distribution CLI 只能组合这组函数。测试逐项覆盖 duplicate/missing/extra
field、非 canonical JSON/path/order、artifact 多/少/换目录/内容漂移、head commit/tree
漂移、重复输出、原子成功以及失败不留 partial output；还要固定从 HEAD commit timestamp
派生 `SOURCE_DATE_EPOCH`，并证明同一 clean HEAD 的两次 wheel/sdist build 逐字节一致。
后续 identity 只能导入这些 pure
函数，不能复制 parser、artifact 枚举或 hash。此 slice 同批启用完整 `distribution`
mode/job；该工具不得在任务 10 以 stub 形式提前存在。第十个 slice 先用 registry test 取得 `gui` 未注册
RED，再实现 actual adapter；同一提交启用 `distribution` mode/job，但
`identity` 和 `release` mode 等任务 13 验收记录存在后再启用。

第十个 slice 完成后运行：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py gui
```

预期：GUI Tasks 1-10 的规范化行为和完整 synthetic workflow 均通过，
且没有 legacy import。

- [ ] 在最终 GUI comparator 之前，第十个 slice 再次执行上述
  `tools/build_release_spec.py` 精确命令，并先运行
  `tests/unit/tools/test_build_release_spec.py`；这一次输出必须反映最终
  `pyproject.toml`、真实 entrypoint、wheel/sdist content policy、lock/R22 digest 和 pinned
  build 生成 metadata 集合。它不快照此刻的完整 sdist member：任务 13/14 还会增加受管
  test/verification 输入，完整 tracked-input member 集合由 distribution verifier 在每次 clean
  HEAD build 时动态派生，并在任务 14 对最终 HEAD 精确核对。
- [ ] 在最终 GUI slice commit 前，运行
  `cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --manifest verification/r22/reference/manifest.json --group gui`，达到 GREEN。
- [ ] 上述 GUI comparator、完整 GUI mode、distribution tool unit test、release-spec
  rebuild、architecture 和 Radon 全部 GREEN 后，提交第十个 slice：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add src/xrr_fitter/gui/main_window.py src/xrr_fitter/gui/document.py tests/integration/test_gui_project_workflow.py tools/reference_groups/gui.py tools/reference_groups/registry.py tools/compare_r22_reference.py tests/unit/tools/test_compare_r22_reference.py tools/verify_distribution.py tests/unit/tools/test_verify_distribution.py tests/unit/tools/test_build_release_spec.py tests/architecture/test_distribution.py tests/architecture/test_public_api.py tests/architecture/test_dependency_rules.py tests/architecture/test_quality_gate.py tests/unit/tools/test_verify.py tools/verify.py verification/release-spec.json docs/user-guide.md docs/images/gui-dark-1280x760.png docs/images/gui-dark-1600x900-expert.png docs/images/gui-light-1280x760.png docs/images/gui-light-1600x900-expert.png docs/architecture/r22-r23-test-ledger.csv .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'refactor: complete R23 GUI and installed distribution' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

- [ ] 因 `distribution` 无条件要求 Git clean，完整 distribution mode 只在上述提交
  后从 clean HEAD 运行，不在 dirty pre-commit tree 上伪造通过：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; REPORT_ROOT=$(mktemp -d /tmp/xrr-r23-task11-distribution.XXXXXX); BUNDLE="$REPORT_ROOT/bundle"; trap 'rm -rf "$REPORT_ROOT"' EXIT; test -z "$(git status --porcelain=v1 --untracked-files=all)"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py distribution --report-dir "$BUNDLE" --artifact-dir "$BUNDLE/artifacts"; test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

  若 clean-HEAD distribution 失败，回到最小 RED test 修复并做一个新的显式提交，
  然后重跑；不 amend 或在工作树留下 build artifact。

### 任务 12：完成 R22 对比与测试台账草案

**只读输入：** `verification/r22/**`。
**创建：** `tests/acceptance/test_r22_reference_equivalence.py`。
**修改：** `tools/verify.py`、`tests/unit/tools/test_verify.py`、
`tests/architecture/test_quality_gate.py`、`.github/workflows/verify.yml` 和
`docs/architecture/r22-r23-test-ledger.csv`。builder、collector、comparator
engine 和八个 adapter 已在任务 2-11 中各自所属批次创建；本任务不修改
`tools/reference_groups/registry.py` 或首次新增 adapter。最终 R23
target manifest 必须等任务 13 的全部 acceptance test 提交后再生成。

- [ ] 重新运行 reference self-check，并从 `verification/release-spec.json`
  读取 Task 2 固化的 `r22_oracle_tree_sha256`，从 filesystem 重算
  `verification/r22/**` 的相对 path/size/hash framed digest 并精确匹配。该目录
  是冻结 oracle，Task 12 不得重写、格式化或重新生成。
- [ ] 用 comparator 的 `--all-groups` 封闭模式针对八个且仅八个 group
  运行 R23；少/多 registry entry 都失败，comparison report 只写入仓库外：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && REPORT_DIR=$(mktemp -d /tmp/xrr-r23-r22-reference.XXXXXX) && trap 'rm -rf "$REPORT_DIR"' EXIT && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --self-check verification/r22/reference/manifest.json --collections-root verification/r22/collections --release-spec verification/release-spec.json && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/compare_r22_reference.py --all-groups --manifest verification/r22/reference/manifest.json --report-dir "$REPORT_DIR"
```

- [ ] 复核初始已覆盖全部 source key 的每一行 `contract_id`/`action`/`reason`，
  并将任务 3-11 已落地的 target 更新为实际 node ID。任务 13 尚未提交的
  acceptance target 仍只是计划值；此时运行 `source-draft`，不宣称 target
  existence 已验证。
- [ ] 对 `delete_layout_only` 做人工及结构化分类复核。任务 10 前审计到的 34 个 architecture-only node 只作规模参考；最终裁决以 `R22-final` 全量 manifest 为准，不把 34 写成 release 数量门禁。
- [ ] 先创建 `tests/acceptance/test_r22_reference_equivalence.py`，并在
  `tests/unit/tools/test_verify.py` 与 `tests/architecture/test_quality_gate.py` 增加 exact registry、
  path、environment 和 CI job mapping assertion；此时不修改 `tools/verify.py` 或 workflow。
  用仓库外 basetemp 运行这三个 test path，必须因 `r22-reference` mode/job 尚未注册取得
  可归因 RED，不能因 frozen oracle 缺失、import、skip 或 fixture error 失败。

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; BASETEMP=$(mktemp -d /tmp/xrr-r23-task12-red.XXXXXX); trap 'rm -rf "$BASETEMP"' EXIT; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python -m pytest -o addopts= --strict-config --strict-markers -p no:cacheprovider --basetemp "$BASETEMP" tests/unit/tools/test_verify.py tests/architecture/test_quality_gate.py tests/acceptance/test_r22_reference_equivalence.py -q
```

- [ ] 实现唯一 `tools/verify.py` registry 与 workflow wiring 后，先运行
  `r22-reference`，再运行包含 verifier unit test 的 `tools`，最后运行包含 architecture 和
  全仓 Radon 的 `quality`；三者全部 GREEN 才能 stage：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src QT_QPA_PLATFORM=offscreen /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py r22-reference && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py tools && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py quality
```
- [ ] 提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --exit-code -- verification/r22 tests/regression src/xrr_fitter && git diff --check && git add tools/verify.py tests/unit/tools/test_verify.py tests/acceptance/test_r22_reference_equivalence.py tests/architecture/test_quality_gate.py docs/architecture/r22-r23-test-ledger.csv .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'test: bind R23 behavior to frozen R22 evidence' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

### 任务 13：完成统计、真实数据验收与发行身份工具

**创建：** `tests/acceptance/test_synthetic_recovery_corpus.py`、
`test_real_data_workflows.py`、`test_gui_real_data_workflows.py`、
`tools/freeze_approved_data.py`、`tests/unit/tools/test_freeze_approved_data.py`、
`tools/release_identity.py`、`tests/unit/tools/test_release_identity.py`、
`verification/approved-data/manifest.json`、
`verification/approved-data/records/{known_single_layer,workable_mo_si_multilayer,unstable_multilayer}.json`
和 `docs/acceptance/r23-release-acceptance.md`。
**修改：** `docs/architecture/r22-r23-test-ledger.csv`、`tools/verify.py`、
`tests/unit/tools/test_verify.py`、`tests/architecture/test_quality_gate.py` 和
`.github/workflows/verify.yml`；
本任务同批启用 `statistical`、`approved-data`、`identity` 和 `release` mode/job。

- [ ] 先只创建三份最终稳定 node ID 的 acceptance test 和
  `tests/unit/tools/test_freeze_approved_data.py`，取得可归因 RED；此时不得预建
  `test_release_identity.py` 或 `release_identity.py`。freeze test 覆盖 duplicate JSON key/case/path、非 canonical
  report、原始数据 hash 漂移、报告/签核 hash 不符、少/多 case、非批准结论、
  symlink/path escape、output 已存在、原子成功和失败无 partial output；还必须逐项篡改
  embedded reviewer/role、environment、workflow operation、四次 run、project/export/plot
  record、warning/confidence/metric 和 conclusion，证明无损投影重建 hash、records tree
  hash 或 source tree hash 会失败。
- [ ] `tools/freeze_approved_data.py` 仅读取显式 `--candidate-report`、
  `--domain-signoff`、`--approved-data-root` 和冻结的
  `verification/r22/reference/manifest.json`。它不运行拟合、不修改原始数据，
  也不搜索默认路径。`--check-candidate` 只严格验证报告并不写文件；
  `--output verification/approved-data` 要求签核存在，用同目录 staging +
  `fsync` + rename 一次性原子发布第 12 节 schema 的 manifest 和三份 record，不复制 raw
  data。发布前必须从拟写 records 重建 candidate/signoff canonical bytes 并匹配两个输入
  hash，再重算 raw source tree 和 records tree；任何 normalized field 都不能只存在于仓库外。
  同一文件公开唯一的 strict committed-manifest/record parser、无损 projection builder 和
  `calculate_approved_data_binding()` pure function，供普通 approved-data verifier 与发行身份
  共用；不得另建 hash/helper 转发模块。先单独运行
  `tests/unit/tools/test_freeze_approved_data.py` 和全仓 Radon 达到 GREEN。
- [ ] freeze owner GREEN 后再创建 `tests/unit/tools/test_release_identity.py`，先因
  `tools/release_identity.py` 不存在取得可归因 RED，再实现该单文件的 `build`/`validate`
  subcommand。它只组合任务 11 的 ArtifactManifest pure API、当前任务的
  `calculate_approved_data_binding()`、Git object、release spec、lock、R22 oracle 和 test
  manifest；不得复制 JSON parser、canonical writer、artifact enumeration、approved
  projection 或 tree-hash 逻辑。fixture-repo 测试覆盖完整 8.1 schema、每层
  missing/extra/tamper、duplicate key、实际 artifact/evidence/raw-data 漂移、test-tree drift、
  build 原子发布，以及 `R23-final` 非 annotated/错误指向和 freeze receipt 原子成功/失败。
  `validate --expected-tag` 与 `--write-freeze-receipt` 必须在本任务已经受测；任务 14 只调用。
- [ ] 使用已批准的固定随机种子和阈值运行完整 220-case 语料库。
  随后在可见桌面会话中，为三类已批准数据分别运行非 GUI 和 GUI
  workflow；每个 case 同 seed 重复三次、新 seed 再运行一次，验证物理合理性、
  warning、置信度诚实性、导出、保存/重开和结果持久化。
- [ ] 先直接运行普通 `approved-data`，确认它因当前 R23 签核记录缺失而
  RED，且不生成 candidate，缺失 evidence 不会自动切换模式。
  `tools/verify.py` 为这一个 preflight 原因专用返回码 `3`；其他配置、测试、
  GUI 或 I/O 失败不得返回 `3`。然后显式使用
  `--capture-candidate` 运行全部 workflow 并以 `0` 写出
  `approved-data-candidate.json`。该 flag 只能由人工候选验收命令使用；
  `release`、CI job 和普通 `approved-data` 路径显式禁止它，不存在自动 fallback：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; CANDIDATE_ROOT=/Users/dala/Desktop/xrr-r23-task13-candidate; test ! -e "$CANDIDATE_ROOT"; mkdir "$CANDIDATE_ROOT"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py statistical --report-dir "$CANDIDATE_ROOT/statistical"; set +e; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py approved-data --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --report-dir "$CANDIDATE_ROOT/missing-evidence"; STATUS=$?; set -e; test "$STATUS" -eq 3; test ! -e "$CANDIDATE_ROOT/missing-evidence/approved-data-candidate.json"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py approved-data --capture-candidate --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --report-dir "$CANDIDATE_ROOT/approved"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/freeze_approved_data.py --candidate-report "$CANDIDATE_ROOT/approved/approved-data-candidate.json" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --r22-reference verification/r22/reference/manifest.json --check-candidate
```

- [ ] 领域负责人审阅上述外部 candidate report 的三个 case、四次运行、项目/
  export/plot 哈希和结论，将 canonical JSON 签核写到
  `/Users/dala/Desktop/xrr-r23-task13-domain-signoff.json`。字段精确为
  `schema="xrr-r23-domain-signoff-v1"`、`reviewer`、`role`、`candidate_report_sha256`、
  以及按 case ID 排序的三个 `{case_id,approved,conclusion}`；`approved` 必须
  均为 `true`。工具不自动生成、猜测或替代人工签核。
- [ ] 签核完成后原子固化仓库内 records，再用新鲜外部 report 重跑两个
  mode 到 GREEN。普通 `approved-data` 必须只用已提交 evidence tree 和显式 raw data root
  重建 candidate/signoff、重跑 workflow、比较四次 run/artifact/result，不再读取外部
  candidate 或 signoff 文件：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; SIGNOFF=/Users/dala/Desktop/xrr-r23-task13-domain-signoff.json; CANDIDATE=/Users/dala/Desktop/xrr-r23-task13-candidate/approved/approved-data-candidate.json; test -s "$SIGNOFF"; test ! -e verification/approved-data; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/freeze_approved_data.py --candidate-report "$CANDIDATE" --domain-signoff "$SIGNOFF" --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --r22-reference verification/r22/reference/manifest.json --output verification/approved-data; CHECK_ROOT=$(mktemp -d /tmp/xrr-r23-task13-green.XXXXXX); trap 'rm -rf "$CHECK_ROOT"' EXIT; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py statistical --report-dir "$CHECK_ROOT/statistical"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/verify.py approved-data --approved-data-root /Users/dala/Desktop/xrr-approved-data-r22-final --report-dir "$CHECK_ROOT/approved"
```

- [ ] 记录精确 input/hash/environment/operation/output/conclusion。如果发现产品差异，
  先添加最小 RED regression test 并指明被违反的产品规则；不为匹配真实数据
  临时放宽 threshold。唯一人类摘要 `docs/acceptance/r23-release-acceptance.md` 只解释
  verdict 和指向上述 committed records，不复制另一套机器证据或创建每 case 重复文档。
- [ ] 更新 ledger 为实际 acceptance node ID，运行 `source-draft`；通过
  `tests/unit/tools/test_verify.py` 确认最终 mode registry、release 顺序和 CI job
  mapping 已包含 `statistical`、`approved-data`、`identity`、`release`，并确认
  只有缺失已签核 manifest 的 approved-data preflight 返回 `3`，其他失败原样
  传播非零状态。`tests/unit/tools/test_release_identity.py` 还要证明 artifact、committed manifest/record、重建的
  candidate/signoff、raw source 三棵 hash 中任意一棵漂移都会失败。全仓 Radon/
  architecture/tool test GREEN；`test_quality_gate.py` 还要用 fixture workflow 证明 Task 13
  当前 tree 的 readiness 为 false、加入有效 final test manifest 后为 true、tag trigger 精确、
  candidate jobs/checkpoint needs 完整，以及 Actions artifact 只能是
  `r23-release-${{ github.sha }}` 白名单 bundle。全部 GREEN 后提交：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && git diff --check && git add verification/approved-data tests/acceptance/{test_synthetic_recovery_corpus,test_real_data_workflows,test_gui_real_data_workflows}.py tests/unit/tools/test_freeze_approved_data.py tests/unit/tools/test_release_identity.py tests/unit/tools/test_verify.py tools/freeze_approved_data.py tools/release_identity.py tools/verify.py docs/acceptance/r23-release-acceptance.md docs/architecture/r22-r23-test-ledger.csv tests/architecture/test_quality_gate.py .github/workflows/verify.yml && git diff --cached --check && git diff --cached --name-status && git commit -m 'test: complete R23 acceptance and release identity' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

提交后立即执行第 2.1 节发布门禁。该 exact-SHA branch run 的
`candidate-readiness` 必须成功并明确输出 `false`；当时已注册的 standard jobs 必须成功，
候选 jobs 必须按受测条件全部为 `skipped`，唯一 `checkpoint` 必须验证这两个集合及其结果后
成功。任何候选 job 意外运行、任何 standard job 被跳过，或 readiness/checkpoint 不符合上述
状态都阻塞任务 14。

### 任务 14：最终发行包与干净工作树门禁

- [ ] 在创建会令 `candidate-readiness=true` 的 final manifest 前，通过 GitHub API 确认
  approved-visible runner online 且五个 label 同属该 runner；挂载只读性由该 run 的
  candidate preflight 再验证：

```bash
set -euo pipefail; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1; GITHUB_REPOSITORY=zxc-1/xrr-fitter; RUNNER_TOTAL=$(gh api "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100" --jq .total_count); test "$RUNNER_TOTAL" -le 100; test "$(gh api "repos/$GITHUB_REPOSITORY/actions/runners?per_page=100" --jq 'any(.runners[]; .status == "online" and ([.labels[].name] | (index("self-hosted") != null and index("macOS") != null and index("ARM64") != null and index("xrr-approved-data") != null and index("visible-gui") != null)))')" = true
```

- [ ] 从任务 13 的 clean HEAD 生成最终 R23 collection manifest。collector 将显式 `--source-commit`、test-tree 文件 hash、node/marker record 和 collection hash 写入 manifest，不记录生成时间或当前临时路径：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && test -z "$(git status --porcelain=v1 --untracked-files=all)" && test ! -e verification/r23 && mkdir verification/r23 && TEST_SOURCE_COMMIT=$(git rev-parse HEAD) && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/collect_test_manifest.py --repo-root /Users/dala/Desktop/xrr-rewrite-r23 --source-commit "$TEST_SOURCE_COMMIT" --lock-file /Users/dala/Desktop/xrr-rewrite-r23/requirements-macos-arm64-py312.lock --suite tests --output verification/r23/tests.json
```

- [ ] 运行 ledger validator 的 `final` phase，证明第 11 节 action 精确枚举、
  `contract_id`/`reason` 有效、R22 source key 集合与 ledger 完全相等且每个只出现
  一次、`target_nodeids` 是非空/去重/排序的 canonical JSON array、每个 target
  都存在于 R23 manifest，且 `delete_layout_only` 的全部 target 位于
  `tests/architecture/`。然后只提交 manifest 和必要的最终 ledger 修正：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/validate_test_ledger.py --phase final --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --target-manifest verification/r23/tests.json --ledger docs/architecture/r22-r23-test-ledger.csv && git diff --check && git add verification/r23/tests.json docs/architecture/r22-r23-test-ledger.csv && git diff --cached --check && git diff --cached --name-status && git commit -m 'test: freeze final R23 test migration manifest' && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

立即执行第 2.1 节发布门禁。此 commit 首次满足 `candidate-readiness=true`，所以它的 exact-SHA
branch run 必须实际完成 standard、statistical、approved-data、distribution、identity、release、
`candidate-readiness` 和 `checkpoint` 全部 job；任何 candidate job skipped 都是失败。
记录该 run ID，后续只从这个 run 下载 canonical release artifact。

- [ ] 从新的 clean HEAD 将同一 `TEST_SOURCE_COMMIT` 重新 collect 到外部 report，要求与已提交 manifest 逐字节一致，并证明 test tree 未变化。这允许 metadata-only manifest commit 位于 source commit 之后，但任何 test-tree 变化都会失败：

```bash
cd /Users/dala/Desktop/xrr-rewrite-r23 && TEST_SOURCE_COMMIT=$(/Users/dala/Desktop/xrr-r23-venv/bin/python -c 'import json; print(json.load(open("verification/r23/tests.json", encoding="utf-8"))["source_commit"])') && RECHECK_DIR=$(mktemp -d /tmp/xrr-r23-test-manifest.XXXXXX) && trap 'rm -rf "$RECHECK_DIR"' EXIT && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/collect_test_manifest.py --repo-root /Users/dala/Desktop/xrr-rewrite-r23 --source-commit "$TEST_SOURCE_COMMIT" --lock-file /Users/dala/Desktop/xrr-rewrite-r23/requirements-macos-arm64-py312.lock --suite tests --output "$RECHECK_DIR/tests.json" && cmp verification/r23/tests.json "$RECHECK_DIR/tests.json" && git diff --quiet "$TEST_SOURCE_COMMIT" HEAD -- tests && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src /Users/dala/Desktop/xrr-r23-venv/bin/python tools/validate_test_ledger.py --phase final --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --target-manifest verification/r23/tests.json --ledger docs/architecture/r22-r23-test-ledger.csv && test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

- [ ] 在具备可见桌面会话和已批准数据挂载的环境中，重新定位当前 HEAD 的唯一 successful
  branch `push` run，下载精确名为 `r23-release-$HEAD_COMMIT` 的 Actions artifact；同时通过
  本地唯一 release mode 运行完整质量、工具、测试、发行包和身份矩阵。CI bundle 和本地产物
  的 root member、wheel/sdist、artifact manifest 与 release identity 必须逐字节相同：

```bash
set -euo pipefail; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0; cd /Users/dala/Desktop/xrr-rewrite-r23; GITHUB_REPOSITORY=zxc-1/xrr-fitter; PYTHON=/Users/dala/Desktop/xrr-r23-venv/bin/python; REPORT_DIR=/Users/dala/Desktop/xrr-r23-release-final; ARTIFACT_DIR="$REPORT_DIR/artifacts"; HEAD_COMMIT=$(git rev-parse HEAD); RUN_REF=r23-clean-architecture; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test "$(git remote get-url origin)" = "https://github.com/$GITHUB_REPOSITORY"; test ! -e "$REPORT_DIR"; RUN_TOTAL=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq .total_count); test "$RUN_TOTAL" -le 100; RUN_ID=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq '[.workflow_runs[] | select(.path == ".github/workflows/verify.yml" and .status == "completed" and .conclusion == "success")] | if length == 1 then .[0].id else error("expected exactly one successful verify.yml run") end'); test "$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .path)" = .github/workflows/verify.yml; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json conclusion --jq .conclusion)" = success; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headSha --jq .headSha)" = "$HEAD_COMMIT"; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json event --jq .event)" = push; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headBranch --jq .headBranch)" = "$RUN_REF"; CHECKPOINT=$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json jobs --jq '[.jobs[] | select(.name == "checkpoint") | .conclusion] | if length == 1 then .[0] else "invalid" end'); test "$CHECKPOINT" = success; CI_BUNDLE=$(mktemp -d /tmp/xrr-r23-ci-release.XXXXXX); trap 'rm -rf "$CI_BUNDLE"' EXIT; gh run download "$RUN_ID" --repo "$GITHUB_REPOSITORY" --name "r23-release-$HEAD_COMMIT" --dir "$CI_BUNDLE"; env -u PYTHONPATH PYTHONDWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src "$PYTHON" tools/verify.py release --report-dir "$REPORT_DIR" --artifact-dir "$ARTIFACT_DIR"; cmp "$REPORT_DIR/artifact-manifest.json" "$CI_BUNDLE/artifact-manifest.json"; cmp "$REPORT_DIR/release-identity.json" "$CI_BUNDLE/release-identity.json"; env -u PYTHONPATH -u PYTHONHOME -u PYTHONOPTIMIZE "$PYTHON" -I -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); manifest=json.load(open(sys.argv[2], encoding="utf-8")); expected={"artifact-manifest.json", "release-identity.json"} | {item["path"] for item in manifest["artifacts"]}; members=list(root.rglob("*")); all(not path.is_symlink() for path in members) or sys.exit("bundle contains symlink"); all(path.is_file() or path.is_dir() for path in members) or sys.exit("bundle contains unsupported member type"); files={path.relative_to(root).as_posix() for path in members if path.is_file()}; directories={path.relative_to(root).as_posix() for path in members if path.is_dir()}; files == expected or sys.exit(f"unexpected bundle files: {files!r}"); directories == {"artifacts"} or sys.exit(f"unexpected bundle directories: {directories!r}")' "$CI_BUNDLE" "$REPORT_DIR/artifact-manifest.json"; set -- "$REPORT_DIR"/artifacts/*; test "$#" -eq 2; for SOURCE in "$REPORT_DIR"/artifacts/*; do cmp "$SOURCE" "$CI_BUNDLE/artifacts/$(basename "$SOURCE")"; done; test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

- [ ] `release` 要求 `artifact-dir` 起初不存在，在该路径创建唯一制品目录；distribution mode 从 clean tracked source 的外部 staging copy 使用已锁定 build dependency 构建，绝不在 repository root 运行 setuptools。等价的独立调用为：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; PYTHON=/Users/dala/Desktop/xrr-r23-venv/bin/python; REPORT_PARENT=$(mktemp -d /tmp/xrr-r23-distribution.XXXXXX); BUNDLE="$REPORT_PARENT/bundle"; trap 'rm -rf "$REPORT_PARENT"' EXIT; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src "$PYTHON" tools/verify.py distribution --report-dir "$BUNDLE" --artifact-dir "$BUNDLE/artifacts"
```

- [ ] `verify_distribution.py` 要求全新制品目录中恰好有一个 wheel 和一个 sdist，校验 wheel `Requires-Dist` 与 `pyproject.toml` 精确一致，使用 `tempfile.TemporaryDirectory` 创建冒烟测试虚拟环境，精确安装 `pip==26.1.2` 和同一 lock，再用 `pip install --no-deps` 安装唯一选定的 wheel；它还从 sdist 在已安装同一 lock 的环境中用 `--no-build-isolation` 重建 wheel。两个 fresh environment 都验证：

实现中将每个 fresh venv 转为 resolved absolute `Path`，然后只调用
`smoke_venv / "bin" / "python"` 和 `smoke_venv / "bin" / "xrr-fitter"`；三个 argv
分别为 `-c "import xrr_fitter.api"`、`-m xrr_fitter --help`、`--help`。子进程
删除 caller `PYTHONPATH`，`PATH` 设为空字符串，`HOME`、`XDG_CACHE_HOME` 和
`MPLCONFIGDIR` 均指向该 fresh environment 的临时目录；不得调用裸
`python`、`pip` 或 `xrr-fitter`。

`--help` 在导入 PySide6 前解析，且必须在不创建 `QApplication` 的情况下退出。专用入口
测试强制保证该行为；本方案不增加 source-tree `--version` 分支。

- [ ] distribution 成功后在 `report-dir` 写入 `artifact-manifest.json`，包含唯一 wheel/sdist 的 filename、size、SHA-256 和验证状态。release 将同一 `artifact-dir` 与 manifest 显式传给 `identity` mode；identity 重新读取两个制品并核对 hash，再将它们绑定到实际当前 source/test/reference、committed approved-evidence tree 和 raw approved-source tree identity。它从 committed records 重建 candidate/signoff canonical bytes 并重算 hash，不读取任务 13 的外部 candidate/signoff。结果只写入外部 `--report-dir`，不得回写工作树或只读取先前生成的 hash，从而避免 identity 包含自身 hash/commit 的循环。
- [ ] 在 release 前后运行 `tools/check_hygiene.py --require-git-clean`；不得用 Git clean
  代替对 ignored cache/egg-info/build artifact 的文件系统检查。开发/冒烟 venv、cache 和
  staging 均在仓库外，由 `TemporaryDirectory` 清理；
  `/Users/dala/Desktop/xrr-r23-release-final` 是保留的最终制品/报告，不视为临时目录。
  绝不得移除用户数据或已验收证据。
- [ ] 只有上述每个命令都以本轮新鲜结果达到 GREEN 后，才创建
  annotated `R23-final` tag。tag 不存在时创建；已存在时只接受“类型为
  annotated tag 且指向当前 clean HEAD”。不再创建另一份 source archive，因为
  已验证 sdist 就是本发行的 source archive。随后使用任务 13 已测试的
  `tools/release_identity.py validate` 重算当前 commit/tree、tag object、release identity、
  artifact manifest、wheel/sdist、committed approved-evidence tree、重建的
  candidate/signoff hash 和 raw approved-source tree，原子写外部 freeze receipt：

```bash
set -euo pipefail; cd /Users/dala/Desktop/xrr-rewrite-r23; PYTHON=/Users/dala/Desktop/xrr-r23-venv/bin/python; REPORT_DIR=/Users/dala/Desktop/xrr-r23-release-final; FREEZE="$REPORT_DIR/r23-final-freeze.json"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test -s "$REPORT_DIR/release-identity.json"; test -s "$REPORT_DIR/artifact-manifest.json"; test ! -e "$FREEZE"; HEAD_COMMIT=$(git rev-parse HEAD); TAG_COMMIT=$(git rev-parse -q --verify 'refs/tags/R23-final^{commit}' 2>/dev/null || true); TAG_TYPE=$(git cat-file -t refs/tags/R23-final 2>/dev/null || true); { test -z "$TAG_COMMIT" || { test "$TAG_COMMIT" = "$HEAD_COMMIT" && test "$TAG_TYPE" = tag; }; }; { test -n "$TAG_COMMIT" || git tag -a R23-final -m 'XRR R23 final accepted release' "$HEAD_COMMIT"; }; test "$(git rev-parse 'R23-final^{commit}')" = "$HEAD_COMMIT"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dala/Desktop/xrr-rewrite-r23/src "$PYTHON" tools/release_identity.py validate --repo-root /Users/dala/Desktop/xrr-rewrite-r23 --release-identity "$REPORT_DIR/release-identity.json" --artifact-dir "$REPORT_DIR/artifacts" --artifact-manifest "$REPORT_DIR/artifact-manifest.json" --expected-tag R23-final --write-freeze-receipt "$FREEZE"; test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

  receipt schema 固定为 `xrr-r23-final-freeze-v1`，包含 commit/tree/tag-object ID、
  release-identity SHA-256、artifact-manifest SHA-256、唯一 wheel/sdist 的文件名/
  size/SHA-256、`approved_evidence_tree_sha256`、`approved_source_tree_sha256`、
  `candidate_report_sha256`、`domain_signoff_sha256` 和 `status=PASS`；仓库内不回写该 hash。

- [ ] 只有 local freeze receipt 为 canonical `PASS` 后，才精确 push 单个 annotated
  `R23-final` tag。branch 已由前述 exact-SHA gate 推送，remote branch 必须仍等于 local HEAD；
  首次执行时 remote tag 必须尚不存在；网络中断后重跑同一门禁时，只接受远端 tag object/
  peeled commit 已分别精确等于本地 object/HEAD。push 后再次核对二者，再等待该 tag 的
  exact-SHA Actions run 与唯一 `checkpoint` GREEN，并下载 tag run 的 release artifact 与
  本地 final bundle 再次逐字节比较。禁止 `git push --tags`、删除/移动 tag 或 force：

```bash
set -euo pipefail; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0; ROOT=/Users/dala/Desktop/xrr-rewrite-r23; PYTHON=/Users/dala/Desktop/xrr-r23-venv/bin/python; REPORT_DIR=/Users/dala/Desktop/xrr-r23-release-final; GITHUB_REPOSITORY=zxc-1/xrr-fitter; cd "$ROOT"; HEAD_COMMIT=$(git rev-parse HEAD); RUN_REF=R23-final; test -z "$(git status --porcelain=v1 --untracked-files=all)"; test "$(git remote get-url origin)" = "https://github.com/$GITHUB_REPOSITORY"; test "$(git cat-file -t refs/tags/R23-final)" = tag; test "$(git rev-parse 'R23-final^{commit}')" = "$HEAD_COMMIT"; test "$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')" = "$HEAD_COMMIT"; LOCAL_TAG_OBJECT=$(git rev-parse refs/tags/R23-final); REMOTE_TAG_OBJECT=$(git ls-remote --tags origin refs/tags/R23-final | awk '{print $1}'); REMOTE_TAG_COMMIT=$(git ls-remote --tags origin 'refs/tags/R23-final^{}' | awk '{print $1}'); if test -z "$REMOTE_TAG_OBJECT" && test -z "$REMOTE_TAG_COMMIT"; then git push origin refs/tags/R23-final:refs/tags/R23-final; else test "$REMOTE_TAG_OBJECT" = "$LOCAL_TAG_OBJECT"; test "$REMOTE_TAG_COMMIT" = "$HEAD_COMMIT"; fi; REMOTE_TAG_OBJECT=$(git ls-remote --tags origin refs/tags/R23-final | awk '{print $1}'); REMOTE_TAG_COMMIT=$(git ls-remote --tags origin 'refs/tags/R23-final^{}' | awk '{print $1}'); test "$REMOTE_TAG_OBJECT" = "$LOCAL_TAG_OBJECT"; test "$REMOTE_TAG_COMMIT" = "$HEAD_COMMIT"; EXPECTED_REFS=$(printf '%s\n' refs/heads/r23-clean-architecture refs/tags/R23-final | LC_ALL=C sort); REMOTE_REFS=$(git ls-remote --refs origin | awk '{print $2}' | LC_ALL=C sort); test "$REMOTE_REFS" = "$EXPECTED_REFS"; RUN_ID=; for ATTEMPT in $(seq 1 60); do RUN_TOTAL=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq .total_count); test "$RUN_TOTAL" -le 100; RUN_ID=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/runs" -f head_sha="$HEAD_COMMIT" -f event=push -f branch="$RUN_REF" -F per_page=100 --jq '[.workflow_runs[] | select(.path == ".github/workflows/verify.yml")] | if length == 1 then .[0].id elif length == 0 then empty else error("ambiguous verify.yml tag run") end'); test -n "$RUN_ID" && break; sleep 5; done; test -n "$RUN_ID"; STATUS=; for ATTEMPT in $(seq 1 720); do STATUS=$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json status --jq .status); test "$STATUS" = completed && break; case "$STATUS" in queued|in_progress|requested|waiting|pending) sleep 60 ;; *) printf 'unexpected run status: %s\n' "$STATUS" >&2; exit 1 ;; esac; done; test "$STATUS" = completed; test "$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID" --jq .path)" = .github/workflows/verify.yml; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json conclusion --jq .conclusion)" = success; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headSha --jq .headSha)" = "$HEAD_COMMIT"; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json event --jq .event)" = push; test "$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json headBranch --jq .headBranch)" = "$RUN_REF"; CHECKPOINT=$(gh run view "$RUN_ID" --repo "$GITHUB_REPOSITORY" --json jobs --jq '[.jobs[] | select(.name == "checkpoint") | .conclusion] | if length == 1 then .[0] else "invalid" end'); test "$CHECKPOINT" = success; TAG_BUNDLE=$(mktemp -d /tmp/xrr-r23-tag-release.XXXXXX); trap 'rm -rf "$TAG_BUNDLE"' EXIT; gh run download "$RUN_ID" --repo "$GITHUB_REPOSITORY" --name "r23-release-$HEAD_COMMIT" --dir "$TAG_BUNDLE"; cmp "$REPORT_DIR/artifact-manifest.json" "$TAG_BUNDLE/artifact-manifest.json"; cmp "$REPORT_DIR/release-identity.json" "$TAG_BUNDLE/release-identity.json"; env -u PYTHONPATH -u PYTHONHOME -u PYTHONOPTIMIZE "$PYTHON" -I -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); manifest=json.load(open(sys.argv[2], encoding="utf-8")); expected={"artifact-manifest.json", "release-identity.json"} | {item["path"] for item in manifest["artifacts"]}; members=list(root.rglob("*")); all(not path.is_symlink() for path in members) or sys.exit("bundle contains symlink"); all(path.is_file() or path.is_dir() for path in members) or sys.exit("bundle contains unsupported member type"); files={path.relative_to(root).as_posix() for path in members if path.is_file()}; directories={path.relative_to(root).as_posix() for path in members if path.is_dir()}; files == expected or sys.exit(f"unexpected bundle files: {files!r}"); directories == {"artifacts"} or sys.exit(f"unexpected bundle directories: {directories!r}")' "$TAG_BUNDLE" "$REPORT_DIR/artifact-manifest.json"; set -- "$REPORT_DIR"/artifacts/*; test "$#" -eq 2; for SOURCE in "$REPORT_DIR"/artifacts/*; do cmp "$SOURCE" "$TAG_BUNDLE/artifacts/$(basename "$SOURCE")"; done
```

- [ ] 创建 draft GitHub `R23-final` Release。资产白名单恰好为本地 identity 绑定的唯一
  wheel、唯一 sdist、`artifact-manifest.json`、`release-identity.json` 和
  `r23-final-freeze.json`；不得上传整个 report dir、raw/candidate/signoff、测试报告或 Actions
  log。draft 下载后要求五个资产与本地逐字节相等，再发布并把 GitHub default branch 切到
  `r23-clean-architecture`。不得使用 `--clobber`：

```bash
set -euo pipefail; test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${GH_ENTERPRISE_TOKEN-}${GITHUB_ENTERPRISE_TOKEN-}"; test -z "${GH_HOST-}" || test "$GH_HOST" = github.com; export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0; ROOT=/Users/dala/Desktop/xrr-rewrite-r23; REPORT_DIR=/Users/dala/Desktop/xrr-r23-release-final; PYTHON=/Users/dala/Desktop/xrr-r23-venv/bin/python; GITHUB_REPOSITORY=zxc-1/xrr-fitter; cd "$ROOT"; test "$(git remote get-url origin)" = "https://github.com/$GITHUB_REPOSITORY"; WHEEL_REL=$("$PYTHON" -c 'import json,sys; rows=[item["path"] for item in json.load(open(sys.argv[1], encoding="utf-8"))["artifacts"] if item["kind"] == "wheel"]; print(rows[0]) if len(rows) == 1 else sys.exit(1)' "$REPORT_DIR/artifact-manifest.json"); SDIST_REL=$("$PYTHON" -c 'import json,sys; rows=[item["path"] for item in json.load(open(sys.argv[1], encoding="utf-8"))["artifacts"] if item["kind"] == "sdist"]; print(rows[0]) if len(rows) == 1 else sys.exit(1)' "$REPORT_DIR/artifact-manifest.json"); WHEEL="$REPORT_DIR/$WHEEL_REL"; SDIST="$REPORT_DIR/$SDIST_REL"; MANIFEST="$REPORT_DIR/artifact-manifest.json"; IDENTITY="$REPORT_DIR/release-identity.json"; FREEZE="$REPORT_DIR/r23-final-freeze.json"; for SOURCE in "$WHEEL" "$SDIST" "$MANIFEST" "$IDENTITY" "$FREEZE"; do test -s "$SOURCE"; done; gh release create R23-final "$WHEEL" "$SDIST" "$MANIFEST" "$IDENTITY" "$FREEZE" --repo "$GITHUB_REPOSITORY" --verify-tag --title 'XRR R23 final' --notes-file docs/acceptance/r23-release-acceptance.md --draft --latest=false; VERIFY_DIR=$(mktemp -d /tmp/xrr-r23-github-release.XXXXXX); trap 'rm -rf "$VERIFY_DIR"' EXIT; gh release download R23-final --repo "$GITHUB_REPOSITORY" --dir "$VERIFY_DIR"; set -- "$VERIFY_DIR"/*; test "$#" -eq 5; for SOURCE in "$WHEEL" "$SDIST" "$MANIFEST" "$IDENTITY" "$FREEZE"; do cmp "$SOURCE" "$VERIFY_DIR/$(basename "$SOURCE")"; done; test "$(gh release view R23-final --repo "$GITHUB_REPOSITORY" --json isDraft --jq .isDraft)" = true; test "$(gh release view R23-final --repo "$GITHUB_REPOSITORY" --json isPrerelease --jq .isPrerelease)" = false; gh repo edit "$GITHUB_REPOSITORY" --default-branch r23-clean-architecture; test "$(gh repo view "$GITHUB_REPOSITORY" --json defaultBranchRef --jq .defaultBranchRef.name)" = r23-clean-architecture; test "$(git ls-remote --heads origin refs/heads/r23-clean-architecture | awk '{print $1}')" = "$(git rev-parse HEAD)"
```

GitHub Release 的平铺资产只是传输层；后续下载后必须按
`artifact-manifest.json + artifacts/` 重建 bundle root 再调用 validator。GitHub 自动生成的
Source code zip/tar 不是 identity 绑定的 canonical source，唯一 canonical source archive
仍是本 Release 中上传并校验的 sdist。

## 15. 统一验证与 CI

`tools/verify.py` 是唯一聚合验证入口。它使用 `subprocess.run(args, check=True)`，不接受 shell 命令字符串，并精确提供以下模式：

```text
quality
tools
unit
integration
gui
spawn
regression
statistical
r22-reference
approved-data
distribution
identity
release
```

每个模式都有显式测试路径列表。不得实现路径发现兜底。`release` 严格按 `quality -> tools -> unit -> integration -> gui -> spawn -> regression -> statistical -> r22-reference -> distribution -> identity` 调用全部软件模式，并要求外部 `--report-dir` 和起初不存在的 `--artifact-dir`；任一子模式非零时立即失败。`approved-data` 是交付后 owner acceptance 的独立人工 mode，不属于 `release` 顺序。`distribution` 强制 `artifact-dir == report-dir/artifacts`，并在该 report dir 原子产出精确名为 `artifact-manifest.json` 的文件。独立 `identity` mode 必须显式接收 `--artifact-dir` 与 `--artifact-manifest`，要求前者 basename 为 `artifacts`、后者 basename 为 `artifact-manifest.json` 且二者 parent 相同；identity 的 `report-dir` 必须位于仓库外，并在其中原子写入精确文件名 `release-identity.json`。`release` 原样保留这些路径和文件名，不 flatten、rename 或另起一份 identity。

`approved-data --capture-candidate` 是任务 13 唯一显式的候选证据生成路径：它
要求起初不存在的外部 `--report-dir`，只能直接与 `approved-data`
一起调用。普通 `approved-data` 缺少已签核 manifest 时硬失败且不产生
candidate，并且只对这一个缺失原因返回 `3`；`release`、`identity` 和所有
CI job 都拒绝传入该 flag。
`tests/unit/tools/test_verify.py` 分别断言 report-dir 所有权、flag 调用范围、
candidate 不可隐式生成、专用返回码和 release/identity/CI 拒绝规则，防止
其退化成隐式 fallback。

pytest suite 所有权固定如下：

```text
tools:
  tests/unit/tools
unit:
  tests/unit/test_evaluation.py
  tests/unit/model
  tests/unit/io
  tests/unit/physics
  tests/unit/fit
  tests/unit/analysis
  tests/unit/services
integration:
  tests/integration/test_entrypoints.py
  tests/integration/test_project_roundtrip.py
  tests/integration/test_single_fit_workflow.py
  tests/integration/test_joint_fit_workflow.py
  tests/integration/test_batch_resume.py
  tests/integration/test_export_workflow.py
gui:
  tests/gui
  tests/integration/test_gui_project_workflow.py
spawn:
  tests/integration/test_process_workers.py
regression:
  tests/regression/test_numerical_reference.py
  tests/regression/test_profile_basin_regressions.py
  tests/regression/test_recovery_metrics.py
statistical:
  tests/acceptance/test_synthetic_recovery_corpus.py
r22-reference:
  tests/acceptance/test_r22_reference_equivalence.py
approved-data:
  tests/acceptance/test_real_data_workflows.py
  tests/acceptance/test_gui_real_data_workflows.py
```

`quality`、`distribution`、`identity` 是显式 checker/orchestrator mode，不通过扩大 pytest
path 重复收集上述 suite。

每个验证器 pytest 子进程都加载 `tests/outcome_gate.py`。所有 mode（不仅是 `release`）在终端统计包含 `skipped`、`xfailed`、`xpassed` 或 `deselected` 时都将 pytest 退出状态改为失败；空 collection 同样失败。有意的测试套件分区使用显式路径完成，不使用宽泛路径加 `-m` 取消选择。平台专属测试只在支持该平台的显式 job 中收集，不靠 skip 维持其他 job 为绿。

`tests/architecture/test_quality_gate.py` 验证精确模式注册表、release 顺序、非零状态传播、
每个 pytest mode 的严格结果插件加载、基础/严格 hygiene 分支、发行模式无法关闭 Git-clean
要求、外部报告目录处理和不存在 shell 执行。`.github/workflows/verify.yml` 在
`$RUNNER_TEMP` 下创建环境并只安装平台 lock；每个 job 先运行
`tools/check_hygiene.py --require-git-clean`，随后由 `tools/verify.py` 使用精确
`$GITHUB_WORKSPACE/src` import path 调用与本地相同的 mode，不进行 editable install。

当前唯一 lock 明确对应 macOS arm64 + Python 3.12，因此每个 CI job 在安装前先断言 `sys.platform == "darwin"`、`platform.machine() == "arm64"` 和 Python 3.12；runner label 必须与该平台匹配。不得在 x86_64/Linux 上复用这份 lock。未来增加平台必须新增固定版本、且其文件 SHA-256 由 release-spec/identity 绑定的独立平台 lock 和对应 job，不得在运行时重新解析宽范围依赖。

验证注册表和 workflow 随能力同批演进，不预注册未来空 suite：任务 2 只启用
`quality`/`tools`；任务 3-8 只随新增 suite path 扩展 `unit`/`regression`，任务 9
的测试已落在注册过的 `tests/unit/io` 下，不重写 wiring；任务 10
在完整 API/worker 存在后启用五个已存在的非 GUI workflow 为 `integration`、将
`test_process_workers.py` 单独注册为 `spawn`，并启用 distribution 内容检查；
任务 11 首 slice 加入 entrypoint 并启用 GUI，最后 slice 加入 GUI integration 和完整
`distribution`；任务 12 启用 `r22-reference`；任务 13 同批加入
`statistical`、`approved-data`、`identity` 和 `release` mode，但 CI 只注册
`statistical`、`identity` 和 `release` job。
`--capture-candidate` 仍只用于任务 13 的人工候选验收命令，绝不注册为 CI
job。任务 13 只验证 identity/release 的 wiring 和失败前置条件；二者在任务 14
提交最终 `verification/r23/tests.json` 前必须因该 manifest 缺失而硬失败，不能提前
宣称候选发行 GREEN。每个引入或扩展 mode 的提交都必须同时修改
`tools/verify.py`、`.github/workflows/verify.yml` 和 `test_quality_gate.py`。任务 13
提交前就必须验证最终 mode registry、CI job registry 和两者映射完全一致；
任务 14 只从该 clean HEAD 生成 manifest 并重验，不再修改测试或 wiring。

GitHub event contract 与能力同步演进：任务 2 的初始 workflow 只接受
`push` 到 `r23-clean-architecture`，运行当时已注册的 standard jobs；不使用
`pull_request_target`，也不把 fork PR 放到 self-hosted runner。任务 13 在最终四个 mode/job
存在时才加入精确 tag `R23-final` 的 `push` trigger，以及受测的 `candidate-readiness` job。
readiness 只检查 repository 内可由 standard runner 复算的静态条件：committed
`verification/r23/tests.json`、最终 ledger，以及候选 jobs 所需的已提交配置和
路径声明；它不得探测或读取 raw approved data。上述静态输入
同时存在且通过 strict preflight 时才输出 `true`：任务 13 branch push
必须明确输出 `false`，任务 14 final-manifest commit 的 branch push 和 `R23-final` tag push
必须输出 `true` 并运行完整候选矩阵。不得用 commit message、latest run、人工 skip、
workflow dispatch 或缺数据后成功退出决定 readiness。raw data 挂载、只读性和可见桌面会话
留给交付后 owner acceptance，不参与软件候选 readiness。

每版 workflow 都有唯一 `checkpoint` aggregator，并使用 `if: always()` 检查完整 `needs`
集合。任务 2-12 只要求本事件已注册的 standard jobs 全部为 `success`。任务 13 起再按
readiness 状态分支：`candidate-readiness` 自身和 standard jobs 必须为 `success`；当 readiness
输出 `false` 时，候选 jobs 必须全部且只能为 `skipped`；当 readiness 输出 `true` 时，候选 jobs
必须全部为 `success`，任一 `skipped/cancelled/neutral/failure` 都失败。job 缺失、出现未登记
job，或实际结果与相应阶段合同不符时 checkpoint 失败。
job-level event/capability condition、readiness output、checkpoint `needs` 和分支判定由
`test_quality_gate.py` 精确断言，不能靠 workflow overall status 掩盖未执行的必需门禁。

workflow 顶层 `permissions` 固定为 `contents: read`；所有 `uses:` 第三方 action 固定到完整
40 位 commit，不使用 mutable tag。GitHub Release 由任务 14 在本机通过已认证 `gh` 发布，
workflow 不获得 `contents: write`。跨 job 的 distribution bundle 只作为同一 run 的临时
artifact，固定短 retention，并保持 bundle root，不得把 approved raw data、candidate/
signoff 外部原件或 freeze receipt 之前的临时 report 上传为 Actions artifact/cache。

任务 2 首次 push 前必须确认至少一个 online standard self-hosted runner 具有
`self-hosted, macOS, ARM64, xrr-ci` 全部 label。真实数据不复制到 GitHub，也不作为
Task 14 软件发行的基础设施前置。

### 15.1 CI 矩阵

CI 使用显式作业，不在默认 pytest 选项中隐藏慢速测试。workflow 先设置
`PYTHON="$RUNNER_TEMP/venv/bin/python"`；runner class 固定为
`[self-hosted, macOS, ARM64, xrr-ci]`：

| Job | 精确 verifier 命令 | Runner / 输入 | 要求频率 |
|---|---|---|---|
| quality | `"$PYTHON" tools/verify.py quality` | standard | 每个 R23 branch/tag push |
| tools | `"$PYTHON" tools/verify.py tools` | standard | 每个 R23 branch/tag push |
| unit | `"$PYTHON" tools/verify.py unit` | standard | 能力注册后的每个 R23 branch/tag push |
| integration | `"$PYTHON" tools/verify.py integration` | standard | 能力注册后的每个 R23 branch/tag push |
| gui | `QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py gui` | standard | 能力注册后的每个 R23 branch/tag push |
| spawn | `"$PYTHON" tools/verify.py spawn` | standard | 能力注册后的每个 R23 branch/tag push |
| regression | `"$PYTHON" tools/verify.py regression` | standard | 能力注册后的每个 R23 branch/tag push |
| statistical | `"$PYTHON" tools/verify.py statistical --report-dir "$RUNNER_TEMP/statistical"` | standard | candidate-ready branch/tag push |
| r22-reference | `QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py r22-reference --report-dir "$RUNNER_TEMP/r22-reference"` | standard / committed R22 oracle | 能力注册后的每个 R23 branch/tag push |
| distribution | `"$PYTHON" tools/verify.py distribution --report-dir "$RUNNER_TEMP/distribution-bundle" --artifact-dir "$RUNNER_TEMP/distribution-bundle/artifacts"` | standard | 能力注册后的每个 R23 branch/tag push |
| identity | `"$PYTHON" tools/verify.py identity --report-dir "$RUNNER_TEMP/identity" --artifact-dir "$RUNNER_TEMP/downloaded-distribution/artifacts" --artifact-manifest "$RUNNER_TEMP/downloaded-distribution/artifact-manifest.json"` | standard / `needs: distribution` 下载物 | candidate-ready branch/tag push |
| release | `QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py release --report-dir "$RUNNER_TEMP/release" --artifact-dir "$RUNNER_TEMP/release/artifacts"` | standard | candidate-ready branch/tag push |
| candidate-readiness | strict preflight；只输出受测的 `ready=true/false` | standard / committed inputs | 任务 13 起每个 R23 branch/tag push |
| checkpoint | 无产品命令；验证本事件全部 required `needs` result | standard | 每个 R23 branch/tag push |

当必需测试被跳过、预期失败、意外取消选择，或缺少其外部已批准数据清单时，作业不得视为 GREEN。

CI 不注册 `approved-data` job，`identity` 和最终 `release` 只绑定软件制品与
`NOT_RUN: owner post-delivery acceptance` 状态。distribution 从生成时就只拥有
`$RUNNER_TEMP/distribution-bundle/`：其 root 含 `artifact-manifest.json`，唯一制品目录为
同 root 下的 `artifacts/`。workflow 原样上传整个 bundle root，不在上传/下载阶段 flatten、
rename 或重排；identity job 通过 `needs: distribution` 精确下载到
`$RUNNER_TEMP/downloaded-distribution`，用显式
`--artifact-dir`/`--artifact-manifest` 验证，不重建或重新选择制品。这些 job
按已提交 manifest 逐文件验证 path/size/SHA-256；hash 不一致或 capture flag
直接失败。standard CI 不伪装执行 owner real-data acceptance；`release` job 自己完整重跑 distribution 和
identity，不消费其他 job 的历史成功状态。

candidate-ready 的 `release` job 只上传名为 `r23-release-${{ github.sha }}` 的单一 Actions
artifact，内容 root 精确为 `artifact-manifest.json`、`release-identity.json` 和
`artifacts/` 下唯一 wheel/sdist；不得包含其他 report、log 或 raw evidence。任务 14 使用
exact-SHA run ID 下载该 artifact，并与本地 release mode 产物逐字节比较。为使该比较成立，
distribution builder 在本地与 CI 都从当前 HEAD commit timestamp 派生同一
`SOURCE_DATE_EPOCH`，测试必须证明相同 clean HEAD 的两次 wheel/sdist build 逐字节一致。

## 16. 每个提交的审查门禁

每个实施提交都必须展示：

1. 精确的新建或已变更行为测试及其初始 RED 原因；
2. 聚焦的 GREEN 命令；
3. `tests/architecture` GREEN；
4. 全仓库 Radon GREEN；
5. 比较器建立后，相关 R22 golden 对比 GREEN；
6. 已移除或重写旧测试对应的台账更新；
7. 不存在无关格式化或生成制品；
8. `origin/r23-clean-architecture` 精确等于该本地 commit；
9. 该 exact SHA 的 GitHub `push` run 与唯一 `checkpoint` job 均为 `success`。

任务 1 只有第 8 项、没有 workflow 时不得伪造第 9 项；任务 2 起两项都是硬门禁。
不得合并移动生产代码却推迟其测试迁移的提交，也不得在 GitHub run 未完成时开始下一批。

## 17. 最终完成定义

只有以下陈述全部成立时，R23 才算完成：

- 只安装 `src/xrr_fitter`。
- 只有 `xrr_fitter.api` 是受支持的 Python API。
- 只有 `python -m xrr_fitter` 和 `xrr-fitter` 启动应用。
- 不存在旧包、模块、垫片、别名或双实现。
- 所有导入都满足依赖图，且不存在延迟导入环。
- 每个项目 Python 文件都通过 Radon 策略。
- 带哈希的 `R22-final` 活动/R21 collection manifest 中，每个节点都在台账中恰好核销一次。
- 所有保留的产品行为都在新测试树中通过。
- R22 归一化数值和工作流对比通过。
- 完整 220 个案例语料库在未改变的已批准阈值下通过。
- GUI Tasks 1-10 在新架构下通过。
- 全部三类已批准真实数据均通过；committed manifest/records 能无损重建 candidate/signoff，
  并分别绑定 committed evidence tree 与 raw source tree hash。
- R22 项目通过 R23 主编解码器正确加载/保存。
- Wheel 和 sdist 能在干净环境中构建、安装和启动。
- 发行身份重新计算实际当前文件、fixture 和证据的哈希。
- `R23-final` 是指向最终 clean HEAD 的 annotated tag，且外部
  `r23-final-freeze.json` 为 canonical `PASS`，绑定 tag object、release identity、
  artifact manifest、wheel/sdist、candidate/signoff、approved evidence/source tree hash。
- 本地 R22 仓库的 `R22-final`、archive、release identity、validation、rehash 和 freeze
  receipt 保持只读并通过重算；R22 仓库仍无 remote。
- GitHub `origin/r23-clean-architecture` 等于最终 HEAD；远端 ref 集合只有该 branch 和
  `R23-final` annotated tag，tag object/peeled commit 等于本地，final branch/tag exact-SHA
  Actions run 均 GREEN。远端不存在 `integration-r22`、`R22-final` 或 R22 Release。
- GitHub `R23-final` Release 的五个白名单资产已下载回读并与 canonical wheel、sdist、
  artifact manifest、release identity、freeze receipt 逐字节相等；未上传 raw approved data、
  candidate/signoff、凭据或临时报告。
- R23 repository working tree 干净，不包含缓存、virtualenv、egg-info、部分导出或未跟踪生成证据。

## 18. 回滚与失败处理

- R22 在本地 `R22-final` 保持原样；回滚意味着停止推进 R23，并保留已 push 的远端分支和审计
  历史。经记录后可以移除本地独立 R23 repository directory，但不得删除或改写远端 branch；
  不得添加兼容代码。
- 如果行为对比失败，返回到产生差异制品的最早已迁移领域。
- 每次只改变一个变量，并在改变生产行为前添加聚焦的 RED 回归测试。
- 如果目标模块不进行无意义碎片化就无法满足 Radon，应重新考虑职责归属或简化控制流；不得添加 Radon 例外。
- 如果测试无法迁移，先在台账中分类，并证明它是产品行为还是过时布局断言，之后才能继续。
- 不得放宽数值阈值、跳过已批准数据、用合成数据替代，或接受存在阻塞门禁的发行版。
- 已 push 的失败 commit 不 amend/rebase/force；添加聚焦修复提交并等待新 exact-SHA run。
  远端 tag 或已发布 Release 不删除、不移动、不覆盖。tag push 后若发现代码/制品问题，停止
  本次发行并使用新的发行版本处理；只有已证明未改变 commit、输入、runner 或 artifact 的
  GitHub 基础设施故障才允许重跑同一 workflow run。
