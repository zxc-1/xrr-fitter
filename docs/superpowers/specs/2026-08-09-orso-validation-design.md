# ORSO 验证套件接入设计

## 目标

把 `github.com/reflectivity/analysis` 的标准测例接入回归测试，让本项目的反射率计算
获得一个社区认可的外部对标基准。产出是一组冻结在仓库内的测例数据与一个回归测试
模块，不改变任何生产代码的数值行为。

## 套件的真实结构与语义映射

以下各条均已对照套件 master 分支核实（`validation/` 目录树、`scripts/test_discovery.py`、
`scripts/test_refnx.py`、`test/unpolarised/README` 与 `test1.layers` 实际数值），不是推断。

**`.txt` 不是数据文件，是 manifest。** `validation/test/unpolarised/` 下有 `test0.txt`
至 `test7.txt` 八个 manifest，外加 `layers/` 与 `data/` 两个子目录。manifest 里 `#` 开头
的行是出处说明，非注释行恰好两行：层表文件名、数据文件名，均相对同目录解析。例如
`test5.txt` 的两行是 `layers/test1.layers` 与 `data/test5.dat`——**多个 manifest 共用同一
层表**，所以 `layers/` 只有六个文件（没有 `test4`/`test5`）。读取器必须先解 manifest 再取
两个被引用文件；直接 `loadtxt` 那个 `.txt` 会失败。

**层表恒为 4 列，数据文件 2 至 4 列，两者列数含义完全不同。** 套件自身的断言是
`slabs.shape[1] == 4` 与 `data.shape[1] in [2, 3, 4]`。层表列为
`thickness / SLD_real / SLD_imag / roughness`，首末行是入射与衬底半无限介质，SLD 以
`1e-6 Å^-2` 为单位。数据文件列为 `Q / R / [dR (1σ)] / [dQ (1σ)]`，**由数据文件的列数选
模式**：小于 4 列走裸核对标，等于 4 列走分辨率卷积对标。读取器按同样的断言分派，列数未
登记时直接失败，不猜测布局。

**层表布局与本项目既有的 pinned-refnx 测例完全一致。**
`tests/regression/test_numerical_reference.py:14` 已在做
`SlabStack(layers[:, 0], (layers[:, 1] + 1j * layers[:, 2]) * 1e-6, layers[1:, 3])`，而套件
的裸核测例把同一个 4 列数组直接喂给 refnx 的 `abeles`。因此层表到 `SlabStack` 的转换已经
存在，本项要新增的是 manifest 解析与测例遍历，不是一套新的语义映射。

**roughness 归属由套件的参考实现定死，不是开放问题。** 套件用
`structure |= SLD(complex(slab[1], slab[2]))(slab[0], slab[-1])` 逐行累加，即第 N 行的
roughness 是该行顶界面的粗糙度、属于界面 `[N-1, N]`，入射介质那一行的 roughness 被 refnx
忽略。这与 `physics/sld_profile.py:12` 的 `interfaces` 及 `:31` 的
`stack.roughness_a[index]` 的界面索引方式一致，也正是既有 `layers[1:, 3]` 切片的含义。
ORSO issue #41 的措辞分歧不影响本套件——参考实现的行为就是判据。`layers/test1.layers` 的
roughness 列取值含 `0/3/1/5`，非零，所以归属接反会真的失败，测例有判别力。

**`dQ` 是 1-σ，不是 FWHM。** `test/unpolarised/README` 写明列义为 `dQ (Å**-1) (1sd)`，
manifest 头部进一步写 `dQ/Q = 0.05 FWHM or 0.0212 1-sigma`，且套件把该列乘
`2*sqrt(2*ln2)` 换成 FWHM 后才交给 refnx。本项目 Gauss-Hermite 卷积内部按 1-σ，因此直接
取用、不做换算。**不设 FWHM 的 `xfail` 占位测例**：语义已定，占位没有价值；而且
`tests/outcome_gate.py` 的 `FORBIDDEN_OUTCOMES` 把 `xfailed`/`xpassed`/`skipped`/
`deselected` 一律判为整轮失败，本仓库根本不存在可用的 `xfail` 机制。同理，冻结数据缺失
时也不能 `pytest.skip`，只能失败——这与下文的 sha256 约定一致。

## 代码边界

- 新增 `tests/regression/test_orso_validation.py`。
- 新增 `tools/sync_orso_suite.py`，按 `tools/freeze_approved_data.py` 的既有姿势把套件
  文件连同 sha256 冻结进仓库。测试期不联网：CI 不得依赖 GitHub 可达性，套件更新必须
  是一次可审计的显式提交。冻结对象是**三层**：manifest（`test*.txt`）、层表
  （`layers/*.layers`）、数据（`data/*.dat`）。三层各自算 sha256，缺任一层都算同步失败——
  只冻 manifest 会得到一组指向不存在文件的指针。冻结列表由 manifest 的引用推导，**不整
  目录扫描**：上游 `data/` 里躺着一个 `Untitled.ipynb`，扫目录会把它一起冻进仓库。同步脚本
  pin 到套件 commit `6a01b4a4febfc52cd3881d2147c732dd1701bc8e`，与 `refnx` 的 commit pin 同
  一姿势。
- `pyproject.toml` 不增加依赖。三层都是纯文本，manifest 用标准库逐行解析，层表与数据用
  已有的 `numpy.loadtxt`。
- 不改动 `src/` 任何文件。

## 失败与状态

- 套件的容差**有两档**，不是一个值。`validation/scripts/test_refnx.py` 里未卷积路径用
  `rtol=8e-5`，卷积（分辨率）路径用 `rtol=0.03`，全程不设 `atol`。分派依据是数据列数：
  `>= 4` 列走卷积路径取宽松档，`< 4` 列走未卷积路径取严格档。本项目沿用这两档，且必须
  按同一条件分派——用 `8e-5` 卡卷积路径会因两边卷积核实现不同而必然失败。
- 即使严格档的 `8e-5` 也比现有 refnx 对标的 `5e-7` 松两个量级，因此套件通过是**必要不
  充分**条件，两套测试都保留。
- 单个测例失败时，测试输出必须给出该测例的 max relative deviation 与所属子目录，便于
  区分"我们错了"与"语义映射选错了"。
- 冻结文件的 sha256 不匹配时测试失败并提示重跑同步脚本，不静默接受新内容。

## 验证

- `pytest tests/regression/test_orso_validation.py -v`，逐测例列出最差相对偏差。
- 首次全绿后把最差偏差记入断言旁的注释，作为下次回归的基线。
- 三处读取约定各配一个显式测例：manifest 指向的文件缺失时报错而非跳过、层表列数不为 4
  时报错、roughness 归属反接（改用 `layers[:-1, 3]`）时至少一个测例转红。第三条需要一份
  roughness 非零的层表，`layers/test1.layers` 的第 4 列为 `0/3/1/5`，满足条件。
- 更新 `docs/algorithm.md`，在 `## Pinned Refnx Benchmark` 之后增加一节说明本套件的
  覆盖范围与容差。

## 非目标

- 不把套件容差当作项目的验收标准，现有更严的 refnx 对标仍是主门禁。
- 不宣称"通过 ORSO 验证"等价于 XRR 路径被验证。套件覆盖的是中子反射的典型 SLD 量级
  （约 `1e-6 Å^-2`），X 射线约 `1e-5 Å^-2` 且吸收项占比完全不同。这一边界必须写进
  README 的能力声明。
- 不实现套件中与本项目能力无关的模式（磁性、极化）。
