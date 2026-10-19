# GUI 验证与 PR 交付执行计划

> **For agentic workers:** 使用 `executing-plans` 按任务推进；新增行为修补先取得 RED，再作最小修改并验证 GREEN。

**Goal:** 完成 GUI 控件操作、1201 点拟合矩阵和当前代码验证，提交、推送 PR 并确认其准确 head SHA 的全部既有检查通过。

**Architecture:** 延续现有 Qt / public API / worker 架构。复用操作驱动器和真实对话框，操作证据与数值拟合、原生验收证据分开记录；不以源码存在、方法直调或测试总数代替操作验收。

**Tech Stack:** Python 3.12、PySide6、pytest-qt、Matplotlib、pyqtgraph。

## Global Constraints

- 继续使用 `feat/gui-redesign-checklist` 现有工作区，保留既有 staged / unstaged 内容，不复制到新工作区或回滚。
- 不改变科学预算、最终求解点数、资格阈值、模型、依赖、项目 schema 或 CI 门禁。
- 证据和临时文件放外层工作区 `.codex-artifacts/` 下的独立运行目录；原始目录 `goal-pr-green-20261015-9kfxuajb` 与旧失败日志保留，最新目录见日期检查点。
- 使用 `/Users/dala/Desktop/XRR-Fitter/venvs/repo-qt6112/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1`；Qt 自动测试使用 `QT_QPA_PLATFORM=offscreen`。
- 原生 GUI 与 offscreen GUI 测试分时；完整数值基准与重拟合测试分时。
- PR 创建后附加到当前聊天，检查准确提交的全部实际 checks。当前配置预计 12 项：7 个 standard job、checkpoint、3 个 Audit job 和 windows-headless；用户未授权合并。

## Task 1: 核对控件身份与操作证据

**Files:** `tests/gui/control_operation_driver.py`、`tests/gui/test_control_operations_*.py`；仓库外 `control-reconciliation.json`。

- [x] 对照旧 321 条清单逐项检查当前控件：存在、改名、同一控件别名、固定输出（非选项）分别记录源码和运行证据，不为凑数量新增控件。
- [x] 复核已有真实 QTest 输入及断言；未满足的拟合、绘图、源恢复、布局和弹窗路径补充可重放的操作用例。
- [x] 将鼠标/键盘、Qt context/wheel event、方法直调、原生验收分别记账；参数化状态单独列明。
- [x] 运行完整控件用例并保存逐操作记录，只将通过用例中的有效证据纳入最终清单。

对账完成不等于全状态验收：319 项有通过用例证据；范围拖动 QTest 已通过；1 项结构树拖放仅受控生命周期通过、原生待验；另 1 项为无生产调用的 helper 按钮，予以排除。最新仓库外 `gui-resume-20261017-k7snq9lo/control-reconciliation-final.json` 绑定完整 GUI 1947 项通过结果，保留来源、历史失败和证据类型，不将夹具或几何检查计作原生输入。

## Task 2: 拟合基准矩阵

**Files:** 仓库外 `benchmark/benchmark_1201.py`、输入项目、逐阶段进度、结果与哈希。

- [x] 核对前向生成真值、独立噪声、1201 点、默认 fast 预算和 4 workers。
- [x] 串行运行单层 robust_log / gaussian / poisson 各手动与自动、双层手动与自动、双集独立及联合各手动与自动。
- [x] 分别报告运行耗时、逐阶段耗时、发布状态、质量标记和真值恢复；失败或未合格不得记为成功联合拟合。
- [x] 对联合自动拒绝路径从成员资格开始定位；保留失败输入，若需物理正常对照则另建输入并明确区别，不降低门槛。

仓库外 `benchmark/README.md`、`truth-recovery.json` 和 `phase-summary.csv` 记录矩阵结果及 42 次文件哈希核对。默认自动联合因成员边界命中拒绝发布；固定已知仪器参数的成功对照单列，不用于默认速度比较。

## Task 3: 原生补验与最小修补

**Files:** 仅已复现问题对应的 GUI 模块及回归测试；仓库外原生时间线。

- [x] 补验原生先验菜单和 splitter 拖动，区分坐标路由/控件门禁与应用错误。
- [x] 复用已完成默认 MCMC 的保存项目确认全参数纵向滚动；不重复采样，不掩盖未收敛诊断。
- [x] 每个已批准的行为缺陷先记录复现测试 RED，最小改动后运行同一测试 GREEN 和相关回归。
- [x] 原生范围手柄拖动、松开后提交已从同一干净输入重放；核对真实采样点范围、点掩码及旧结果失效，不以 AX 勾选外观代替范围模式启用。
- [ ] 结构树拖放完成态按用户选择延后，不把受控 Qt 生命周期记为原生通过。

## Task 4: 最终验证与差异审查

- [x] 以 `python tools/verify.py MODE --report-dir=<仓库外独立目录>` 完成七模式验证；后续 GUI 最小修补重跑 quality、gui、typing，其余不变输入证据明确按哈希复用。
- [x] 补验既有 typing、advisories 和 coverage 审计；审计工具按现有 hash-pinned lock 装入仓库外独立环境，不改变 GUI 环境或依赖锁。coverage 与 GUI/数值任务分时执行。
- [ ] PR 上核对 Audit advisories 的 distribution / installed SBOM / byte binding，以及 Windows CLI / 导出 / executable / byte binding；本机锁定包漏洞查询不能替代这些检查。
- [x] 确认完整 `python tools/check_radon.py` 通过，不删除断言或降低复杂度阈值。
- [x] 对完整 staged / unstaged / untracked 差异按原任务独立审查，保留预存改动；检查失败发布、取消、运行门禁与显示修补不触及科学模型。
- [x] 核对验证前后源码哈希与原始 index；当前无自建工作进程遗留，仓库卫生检查通过，外部证据及历史失败产物有意保留。

## Task 5: 提交、推送、PR 全绿

- [x] 将经过审查的相关文件显式暂存并提交；GitHub 本机认证不可用时由用户重新登录，不读取或传递凭据。
- [x] 正常推送 `feat/gui-redesign-checklist`，创建目标为 `main` 的 PR，并附加到当前聊天。
- [ ] 检查 PR head SHA 与本地提交一致；跟踪全部既有 checks，失败则保留日志、修复、验证、提交后重新等待。
- [ ] 所有需求证据充分且 PR 全绿后完成目标；汇报改动、验证、清理项和残余科学/平台限制。

## 历史本地检查点（2026-10-15）

- 用户选择先完成本地工作；暂不重试 GitHub 登录或推送。
- `local-verification-round2` 在不变源码上完成 7 个模式：quality 224、tools 1417、unit 3170、integration 41、spawn 4、regression 70 个用例通过；GUI 为 1889 通过、1 失败，唯一失败是范围拖动，未标记全绿。
- 后验图固定 4 位有效数字造成窄分布刻度重复。新增厚度、scale、background × 两种窗口宽度的 6 个 RED，用标准自动精度、独立倍率间距和每格 180 逻辑像素修补后，16 个布局用例及 138 个相关用例 GREEN；原滚动、字号、直方图与分位线断言保留。
- 后验精度修补后的正式 GUI 复验完成：1895 通过、1 失败，仍为范围拖动（目标 0.2 ± 0.03，实际 0.28）；验证前后 1002 个工作区文件哈希未变，未跳过该失败。7 条 warnings 来自现有 pyqtgraph 的 NumPy shape 弃用提示。
- 正式 typing 和 advisories 审计已通过，源码哈希未变；macOS / Windows 锁中的 44 / 38 个包完整查询且未报已知漏洞。VCS refnx 和原生库不在漏洞服务扫描范围，Windows 实机与字节绑定审计仍未验证。报告位于仓库外 `pr-audits-20261015`。
- 正式 coverage 审计在 GUI 结束后串行完成：3285 个测试通过，合并 25 个进程数据文件并生成 JSON / XML / HTML，输入哈希与 typing / advisories 一致。该审计排除 GUI，语句覆盖率 93.22%、分支覆盖率 80.48%，不能替代原生验收。
- coverage 结束后的原生工具复查仍返回 Mac locked；未启动新 GUI，也未修改系统锁屏设置，最新后验图的原生显示继续待验收。
- 只重绘既有默认 MCMC 保存结果，没有重跑采样或修改科学参数。样本为 128000×8，最大 split Rhat 为 41.13029142662623，仍未收敛，显示验收不能当作科学收敛验收。
- 范围手柄松开跳动、原生拖放后树丢行的最小修补均已单独提出，等待用户范围确认。保留失败截图、事件时间线和输入项目；不靠重载、放宽容差或跳过测试掩盖。
- 独立审查代理不可用；当前审查为主代理直接审查完整 `origin/main` 到工作树的差异，不能声称完成独立 review。最终 GUI/quality 复验与交付仍以最新仓库外报告和准确提交 checks 为准。

## 终态修补后本地检查点（2026-10-17）

- 用户批准的检查点提示、非终态进度及真实结果反馈已按 RED → GREEN 最小修补。checkpoint 提示读取已发布项目；阶段帧最高 99%；全部本次成员有效发布后才到 100%。失败、部分成功和需复核的原始原因显示于稳定主画布区域，不改 worker、科学预算、模型或资格阈值。
- 修补后串行正式门禁：quality 224 passed（含完整 Radon）、GUI 1947 passed、typing passed。GUI 的 7 条 warning 来自既有 pyqtgraph / NumPy shape 弃用提示，零失败、零跳过。1010 文件验证前后哈希一致；本条及任务勾选是随后同步的文档记录，不改变任何生产或测试文件。
- 修补前同日完整门禁：tools 1417、unit 3170、integration 41、spawn 4、regression 70 通过；advisories 和 coverage 通过。其覆盖的非 GUI 生产、测试与锁文件未改，复用时保留原运行日期、命令及输入哈希，不冒称重跑。coverage 排除 GUI：语句 93.2153%、分支 80.4848%。
- 独立全差异审查提出的三项 P2，以及增量复审发现的提示隐藏问题均已闭环；最终补审 P0=0、P1=0、P2=0。真实 synthetic GUI 工作流保留原输入、缩减测试预算和数值断言，仅将误导的纯成功文案期望改为可见的复核原因。
- 控件对账绑定 112 个记录文件、435 条操作记录和最新完整 GUI 门禁。319 个身份有通过用例证据，1 个原生拖放待验，1 个未实例化 helper 排除；不宣称全状态／全平台验收。
- 科学基准只作哈希复核：152 个科学源码、42 次产物核对全部匹配，没有重拟合。12 项默认矩阵中 11 项发布全点 E 结果，自动 joint 因成员边界资格拒绝发布；固定已知仪器参数对照单列，不作为等维速度比较。
- 原生后验 8 参数纵向滚动、先验菜单／非法值／取消已通过；canvas splitter 观察到布局变化，固定外层缝不移动。原生范围提交、结构树拖放仍未验收。128000×8 的既有 MCMC 最大 split-Rhat 41.13029142662623，仍未收敛，未重新采样。
- 四张终态图片是正式主题下的 offscreen typed fixture 渲染，只证明所示尺寸与文本的可见性，不替代原生输入或真实数值结果。
- 最新证据位于仓库外 `.codex-artifacts/gui-resume-20261017-k7snq9lo`：`feedback-final-results.json`、`feedback-final-input-manifest.json`、`feedback-review-final.md`、`control-reconciliation-final.json`、`benchmark-revalidation-final.json`。未覆盖历史失败日志或原始项目。
- 此检查点未提交、推送或创建 PR；本地核验完成后再检查 GitHub 认证。准确 PR head 的既有 checks、Windows 实际检查和原生待验项仍未完成，不能将本机通过写成整个交付目标完成。

## 提交钩子兼容整理（2026-10-17）

- 既有 Ruff 0.16.1 提交钩子的只读检查报出 117 项；根因主要为 pytest fixture 的导入注册被识别成未使用。18 处 fixture 改为显式同名导出，保留原注册、参数名和用例；不删除 fixture，不添加 ignore/noqa，不绕过钩子。另显式保留 `zip(strict=False)` 的原截断语义，绑定测试菜单回调的当轮参数，以具名函数替换两处赋值 lambda。
- 仅对原待提交的文件执行既有钩子要求的整理；41 个文件变化，其中 7 个生产文件的完整 Python AST 与整理前逐项一致，科学表达式、worker、预算、资格门槛、依赖和 CI 均未改。Ruff lint 与 format 的只读复查已通过。
- 上一检查点的 152 个科学文件哈希全部匹配是整理前证据；当前 150 个仍逐字节匹配，`services/batch.py` 和 `services/batch_publication.py` 仅格式改变且 AST 相同。保留其原字节、前后 SHA-256 和 AST 比较，不将语义相同冒称字节相同；科学基准没有重跑。
- 整理后重新执行 quality、GUI、integration、spawn、regression、typing，使用原门禁与独立外部目录；以 `precommit-verification-results.json` 和 `precommit-input-manifest.json` 为该阶段的准确证据。旧 tools/unit/coverage/advisories 的运行日期与输入保持原记录；远端 PR 的全部实际检查仍须绑定最终提交核验。

## 续接交付检查点（2026-10-18）

- 已读取原会话并确认提交钩子整理后的六项门禁实际完成：quality 224、GUI 1947、integration 41、spawn 4、regression 70 通过，typing 通过。续接时 1010 个文件与该轮输入逐字节一致，原始 index 未变；本次仓库改动仅同步本交付记录，未改生产、测试、依赖或 CI。
- 本轮前置 quality 224 passed（含完整 Radon）；使用项目验证环境与仓库外 `--basetemp` 的 11 文件 GUI 抽样 95 passed。最初未指定 `--basetemp` 的抽样在 macOS 临时路径文件框中挂起；仅更换外置临时目录后，同一用例及完整抽样通过，未修改或跳过测试。失败日志与诊断记录保留。
- 原生范围验证使用保存项目副本和真实 CUA 指针输入。首次观察确认 AX 勾选不等于模式启用；真实点击后范围区域出现、平移关闭，拖动左界并松开仅发布一次。去除指针事件过滤器、重新加载同一干净输入后，第二次重放得到相同请求与提交。
- 两次请求均为 `[1.4302408477259447, 8.132438583912341]`；源数据保留 991/1201 点，实际提交范围均为 `[1.431, 7.8]`。后者由有效采样点的最小/最大值确定，不应等于视图边界。重放的完整点掩码哈希与 `api.import_data` 计算一致，旧结果及 checkpoint 已失效，界面返回参数待复核。原项目副本与源数据哈希不变，未保存、重拟合、重采样或操作树拖放。
- 控件身份对账仍为 319 项有通过用例证据、1 项原生树拖放延后、1 项未实例化 helper 排除；范围另补原生提交证据。科学源码仍为 150 文件字节相同、2 个 batch 文件仅格式改变且 AST 相同；默认自动 joint 拒绝发布与既有 MCMC 未收敛限制保持原结论。
- 本轮独立审查服务不可用，没有新的独立审查结论；既有全差异及最后增量审查绑定的生产、测试源码未变。证据位于仓库外 `.codex-artifacts/gui-delivery-20261018`，包括 `focused-canonical-state.json`、`scope-recheck.json`、`native-range-observed/result.json`、`native-range-replay/result.json`；本文档同步后的 quality 以 `delivery-quality-state.json` 为准。
- 自建原生验证窗口和测试子进程已退出，原始 index、输入项目、历史失败日志及仓库外证据保留。提交、推送和创建 PR 等待本轮确认；不合并、不强推，远端准确 head 的全部实际 checks 与 Windows / SBOM / 字节绑定检查仍待完成。

## 默认临时目录的 CI 补验（2026-10-18）

- 用户确认继续后，已提交并正常推送 `7682d5f`，创建并附加 PR #32，目标为 `main`，没有合并或强推。初次远端 Windows、advisories 等检查通过；下载后的发布产物、SBOM、EXE 字节与准确提交的合并快照绑定一致。审计 `aggregate=incomplete` 的原生/OS/loader 边界仍保留，不等于完整供应链闭包。
- 远端 GUI 长时间未结束时，在本地按 `tools/verify.py gui` 默认的系统临时目录布局重放已知导入用例，60 秒挂起；先前指定外置报告目录只能避开该条件，不构成对默认路径的修复。运行日志保留于 `gui-delivery-20261018/ci-temp-probe.json`，不能将本地复现直接冒称远端日志的结论。
- 测试驱动器有两处可复现问题：文件名输入经补全后可能同时保留父目录和目标文件，点击 Open 仅导航目录；已处理但未关闭的 modal 到达等待上限后只停止计时器，没有退出 `exec()`。分别取得 RED 后，仅在 `tests/gui/control_operation_driver.py` 为单一路径加引号，并在超时记录失败原因后拒绝仍打开的 dialog；不直接接受文件框、不放宽断言、不跳过用例或增加 CI 超时。
- `test_control_operation_driver.py` 新增默认系统临时路径、已有目录选择及未关闭 modal 的回归。修补后 4 个驱动器用例通过，原意外第二 modal 的失败断言保留；新增系统路径用例走真实文件框、基底选择与项目提交。初次试验的取消路径产生额外提示，修正为原有接受流程后重新取得 RED → GREEN，旧失败记录保留。
- 原先 60 秒挂起的既有导入用例，使用同样的默认系统临时目录布局重放后通过。首次正式 quality 报出驱动器 `visit` 的 CC 11 超过既有上限 10；仅合并超时分支的重复类型判断，保留失败原因、计时器停止和 dialog 拒绝行为，未改复杂度政策。该失败及后续完整复验分别保留，不用局部测试替代完整门禁。
- 本阶段仅修改两个 GUI 测试文件及本文档，生产源码、依赖、CI、科学参数和阈值均不变。使用默认报告路径重放完整正式 GUI 门禁，quality 和 GUI 的准确输入及结果以仓库外 `.codex-artifacts/gui-ci-driver-20261018` 为准；后续提交需重新核对新 PR head 的全部既有 checks，不能复用旧 head 的全绿结论。

## CI 修补续接复核（2026-10-19）

- 续接时确认上一轮默认报告路径验证已实际结束：quality 224 passed，GUI 1950 passed、7 条既有 pyqtgraph / NumPy 弃用 warning，零失败、零跳过。`gui-ci-driver-20261018/round2/verification-complete.json`、前后输入清单和 1010 个当前文件逐字节一致，原 index 未变；不将这轮历史完整门禁冒称本日重跑。
- 本轮从新的系统临时目录复验驱动器与真实导入对话框，12 passed；包含单文件替换已有目录、默认系统路径、未关闭 modal、有身份的意外第二 modal、真实基底选择及项目提交。未放宽超时、断言或复杂度政策，测试前后源码和 index 未变。
- 旧 PR head `7682d5f` 的 GUI 已取消、checkpoint 失败，其余 10 项 checks 通过。远端旧日志只有最后进度而无准确用例栈，仍不把本地文件框复现冒称确定的远端根因。补丁提交后必须重新检查准确新 head 的全部实际 checks，并重新绑定 Windows / SBOM / 发布产物字节证据。
- 前两次独立审查请求因 503 未能开始工作，主代理先行直接复核。按用户随后要求，以完整继承主线程模型与推理程度的新子代理完成只读增量审查：未发现 P0/P1/P2/P3 问题；只核查日志与输入哈希，没有重跑门禁。修补仅涉及两个测试文件及本文档，不改变生产源码、科学模型、预算、阈值、依赖或 CI。结构树原生拖放继续按用户选择延后，既有 MCMC 未收敛限制不变。
- 本轮正式 quality 224 passed，包含完整 Radon。首次运行正确拒绝了工具本轮自动创建的 `.serena` 顶层目录；核对创建时间后，仅将该工具配置移至仓库外保留再复验，没有改变卫生政策。审查结果的本文档补记随后再次执行正式 quality，以最终输入清单匹配待提交字节。
- 本轮证据位于仓库外 `.codex-artifacts/gui-ci-resume-20261019-w5njwjq0`：`baseline.json`、`focused-state.json` 和逐文件清单。本文档同步后的正式 quality、提交钩子、准确提交与最终远端结果分别以该目录后续记录为准；尚未执行的检查不提前标为通过。系统临时测试目录移入该证据目录保留，旧失败日志和原项目不覆盖，不合并、不强推。
