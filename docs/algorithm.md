# XRR Fitter Algorithm

## Coordinates And Units

Input files store instrument `2θ` in degrees. The incident angle and momentum
transfer are

```text
θ = 2θ / 2 + Δθ
qz = (4π / λ) sin(θ)
```

Angles are converted to radians only for trigonometric functions. Length uses
Å, momentum transfer uses `Å⁻¹`, and complex X-ray SLD uses `Å⁻²`.
`read_xy()` derives `qz_a_inv` directly from the stored `two_theta_deg`, the
explicit import offset, and `BeamSpec.effective_wavelength_a`.

## Complex SLD And Wavevector

Passive absorption is represented by a nonnegative imaginary SLD:

```text
ρ = ρ′ + iρ″,  ρ″ ≥ 0
kz,j = sqrt((qz / 2)² - 4π(ρj - ρfronting))
```

`_layer_kz()` preserves the complete complex difference from the fronting
medium. It selects `Im(kz) < 0`; on the real axis it selects `Re(kz) ≥ 0`.
Only an exactly zero relative imaginary SLD receives the versioned
`1e-36 Å⁻²` branch selector.

## Fresnel, Roughness, And Parratt

The ideal interface amplitude and Nevot–Croce correction are

```text
rj,j+1 = (kz,j - kz,j+1) / (kz,j + kz,j+1)
rj,j+1 ← rj,j+1 exp(-2 kz,j kz,j+1 σj²)
```

The roughness factor is not clipped. Its magnitude can exceed one in an
evanescent region; candidates with ideal reflectivity above the configured
physical threshold are diagnosed by the fitting layer.

`parratt_reflectivity()` starts at the backing interface and recurses upward:

```text
βj+1 = exp(-2 i kz,j+1 dj+1)
Xj = (rj,j+1 + Xj+1 βj+1) / (1 + rj,j+1 Xj+1 βj+1)
R = |X0|²
```

`abeles_reflectivity()` is an independent characteristic-matrix
implementation used for internal cross-checks; it does not call Parratt.

## Resolution

`gaussian_smear()` convolves in q space. Relative, absolute, and per-point
standard deviations combine in quadrature:

```text
σq,total² = (η qz)² + σq,absolute² + σq,point²
```

`theta_domain_smear()` is the mutually exclusive expert angular-divergence
mode. Both functions use normalized Gauss–Hermite quadrature with per-point
orders `17 → 33 → 65`. The first finer value is accepted when

```text
|Rfine - Rcoarse| ≤ max(1e-12, 1e-4 |Rfine|)
```

Samples below the physical zero boundary are omitted and retained weights are
renormalized. If 33→65 still fails, the 65-point value is returned and
`GaussHermiteConvergenceWarning` plus a structured `PhysicsDiagnostic` records
the affected point indices.

## Beam And Instrument Model

For monochromatic Cu Kα, one wavelength-specific stack is evaluated. For a
mixed Kα beam, material SLDs and stacks are expanded independently at `λ₁` and
`λ₂`, then each receives its own q grid, Parratt calculation, and resolution
convolution:

```text
Rmix = (Rλ1 + r21 Rλ2) / (1 + r21)
```

The footprint factor is

```text
F(θ) = 1                              when θfp = 0
F(θ) = min(1, sin(θ) / sin(θfp))      otherwise
```

Background families are

```text
Bconstant(qz) = B0
Blinear(qz) = B0 + B1 qz
Bpower(qz) = B0 + B2 qz^-p,  B2 ≥ 0,  1 ≤ p ≤ 4
```

`instrument_reflectivity()` composes them in this order:

```text
Rmodel = S · F(θ) · Rsmear + B(qz)
```

Thus neither scale nor footprint multiplies the background.

## SLD Depth Profile

`sld_depth_profile()` returns read-only depth and complex SLD arrays. Sharp
interfaces use exact piecewise media values. Rough interfaces use ordered
error-function transitions whose nonnegative media weights sum to one, so a
stack with passive input media cannot acquire negative absorption in the
displayed profile.

## Interface Transition Kernels

An `InterfaceTransition` replaces the Névot-Croce roughness of one incident
interface with an explicitly discretized composition profile. The normalized
coordinate is `t = 0` at the incident side and `t = 1` at the layer material
side; each kernel `f(t)` returns the fraction of the layer material, satisfies
`f(0) = 0` and `f(1) = 1` exactly, and is monotone non-decreasing.

Shape constants are fixed, not fitted: `ERF_HALF_WIDTH_SIGMAS = 2.0`,
`TANH_HALF_WIDTH = 2.0`, `EXPONENTIAL_RATE = 4.0`. Without them "an erf
transition" carries no numerical meaning, because the mapping from a declared
width to a slope is exactly what these constants pin down. Each kernel is
renormalized by its own value at the endpoints so the contract holds:

| kind | `f(t)` |
| --- | --- |
| `erf` | `0.5 * (1 + erf(2 * (2t - 1) / sqrt(2)) / erf(2 / sqrt(2)))` |
| `linear` | `t` |
| `tanh` | `0.5 * (1 + tanh(2 * (2t - 1)) / tanh(2))` |
| `sine` | `0.5 * (1 - cos(pi * t))` |
| `exponential` | `(1 - exp(-4t)) / (1 - exp(-4))` |
| `step` | `0` for `t < 0.5`, else `1` |

### Composition Of Several Branches

A transition holds one or more weighted branches, each with its own kind and
declared width. Weights are normalized at construction time, so what is stored
and written to disk always sums to one. The discretized region spans the widest
branch, `W = max(b.thickness_a)`. At depth `z` each branch is evaluated at its
own normalized coordinate `t_b = clip(z / b.thickness_a, 0, 1)` and the
fractions are combined by weight:

```
f(z) = sum_b w_b * f_b(clip(z / b.thickness_a, 0, 1))
```

Narrower branches saturate at `1` before the region ends, which is how branches
of different widths compose one profile. Because every kernel is exactly `0` at
`t = 0` and exactly `1` at `t = 1`, and the weights sum to one, the composed
profile also hits both endpoints exactly.

The region is split into `N = max(1, ceil(W / microslab_max_a))` microslabs, and
fractions are sampled at slab *centers* rather than edges, which keeps the first
and last fraction strictly inside `(0, 1)`. Expansion emits exactly `N + 1` rows
per graded layer: `N` microslabs plus one body slab of thickness
`thickness_a - W`. The body slab is emitted unconditionally, even at zero
thickness, so the row count stays predictable. `MAX_TRANSITION_SLABS = 512`
bounds `N`: at 512 rows per interface the Parratt recursion and its analytic
Jacobian stay well inside the cost of an ordinary multilayer, while the profile
resolution is finer than any width a laboratory measurement can constrain.

### Roughness Is Replaced, Not Added

A transition and Névot-Croce describe the same physical broadening of the same
interface. Applying both would widen it twice, so a layer that declares a
transition must declare `roughness_a = 0` (rejected at construction), and
`compile_fit_problem` emits that axis as locked at zero and refuses parameter
settings that reopen it. The layer dialog disables the roughness input while the
transition toggle is on, so the exclusion is visible before commit time rather
than surfacing as a rejection afterwards. Microslab boundaries are numerical
subdivisions rather than physical interfaces, so they carry zero roughness and
stay out of the dynamic roughness limits.

### Displayed Profiles Show Steps, By Design

`sld_depth_profile()` takes its sharp branch when every roughness in the stack
is zero (`physics/sld_profile.py`), which is exactly the case for a structure
whose only interface widths come from transitions. The transition region is then
drawn as `N` discrete steps rather than a smooth curve. That is the true
discretization the reflectivity calculation uses, not a plotting defect — do not
"fix" it by smoothing the display, which would show a profile the model never
evaluated.

## SLD 剖面不确定度带

带由 `McmcReport.samples_physical` 的每一行经 `rebuild_structure` →
`expand_structure` → `sld_depth_profile` 重放得到，按选定界面平移对齐后插值到
所有抽样对齐深度范围的**交集**，实部与虚部分别取 2.5/16/50/84/97.5 分位。

取交集而非并集：并集边缘只有少数抽样支撑，会产生看起来收窄实则无统计意义的伪带宽。
实虚分别取分位而非对复数取模：虚部承载吸收信息，而吸收是 XRR 密度对比的主要来源。

抽样超过 500 条时按 `np.linspace` 均匀抽稀，抽稀比例写进图注。单条抽样重放失败被计数
并跳过，失败率超过 5% 时整体失败而非给出基于少数抽样的带。

默认对齐基底界面：XRR 的基底通常是已知单晶，把不确定度累积推向表面侧符合实际认知。
界面提供切换到表面对齐，因为“带宽在哪里为零”完全取决于这个选择。切换是视图态，不写回项目。

图注写成 `16–84%` 与 `2.5–97.5%` 分位区间而非 1σ/2σ：后验通常不是高斯的，σ 需要额外
假定高斯性才有意义，分位数无歧义。

## Pinned Refnx Benchmark

The development reference is refnx commit
`3d3808f66a14a8200eba020f8dff53f4d1e059bc`. Run the deterministic 500-stack
air-fronting parity benchmark with:

```bash
python tools/verify.py regression --report-dir /tmp/xrr-r23-regression
```

The benchmark covers `0–20` finite layers, `2–5000 Å` thickness, real SLD
`0–150e-6 Å⁻²`, imaginary SLD `0–20e-6 Å⁻²`, and roughness within the
Nevot–Croce parity domain `min(50 Å, 0.30 d_eff)`. It evaluates every point on
the fixed mixed `qz=1e-4–1 Å⁻¹` grid.

Pinned refnx characteristic-matrix Abeles is the primary oracle. Thick,
absorbing stacks can make that matrix backend nonfinite or finite but
numerically inconsistent at low q. Each point is therefore cross-checked
against the same pinned commit's stable public Python Parratt backend; the
Abeles result is used only where both references agree within the approved
tolerance, and the stable result is used elsewhere. No q point is skipped and
this test-only oracle selection never changes production physics. Separate
analytic tests preserve complete complex non-air fronting behavior and the
independence of the local Abeles implementation.

## ORSO Community Validation Suite

The suite is vendored at commit
`6a01b4a4febfc52cd3881d2147c732dd1701bc8e` under `tests/fixtures/orso`, hash
bound by `index.json` and refreshed only through
`tools/sync_orso_suite.py --fetch`. Tests never reach the network.

Eight unpolarised cases are compared. Layer tables are four columns
(thickness, real SLD, imaginary SLD, top-interface roughness). Data files with
fewer than four columns compare the bare Parratt kernel at `rtol=8e-5`; a
fourth column is a pointwise 1-sigma `dQ` and selects the smeared comparison at
`rtol=0.03`. Both values are the suite's own published tolerances.

The smeared cases carry a convention boundary worth stating plainly. The suite
generated its reference data with a resolution kernel truncated at `±3.5σ`, and
all four reference implementations pin that same limit: refnx's `_INTLIMIT`,
refl1d's `linspace(q - 3.5 dQ, q + 3.5 dQ)`, BornAgain's
`DistributionGaussian(0, 1, 21, 3.5)`, and GenX's `resintrange = 3.5`. Our
production smearing in `xrr_fitter.physics.resolution` is deliberately
untruncated, which is the more faithful convolution but disagrees with the suite
by up to `9.0e-02` at deep interference minima, where the tails beyond `3.5σ`
sample regions one to two orders of magnitude brighter than the minimum itself.
The regression module therefore compares the smeared tier through a
transcription of the suite's own truncated quadrature, and separately holds the
untruncated production path to the same `rtol=0.03` at every non-minimum point.
Neither path relaxes a tolerance and no production numerics change.

The suite covers neutron SLD magnitudes near `1e-6 Å⁻²`. X-ray work runs near
`1e-5 Å⁻²` with a different absorption balance, so passing the suite does not
by itself validate the XRR path. The pinned refnx benchmark above, two orders
of magnitude tighter, remains the primary gate.

## Synthetic Recovery Slow Corpus

The reproducible automatic-fitter stage gate lives in
`tests/acceptance/test_synthetic_recovery_corpus.py`. Its optimizer cases are
marked `@pytest.mark.slow`; focused metric and profile regressions remain in
`tests/regression/` and run through `tools/verify.py regression`.

The corpus is fixed by deterministic integer seeds and generates every curve
with the real `instrument_reflectivity()` path before fitting with the real
`fit_dataset()` API. It does not mock optimizer output and does not hard-code
fitted answers. Each case records two different seeds: `seed` drives curve and
noise generation, while `fit_seed` is passed to
`FitConfig.fast(master_seed=fit_seed)` and drives the optimizer seed tree.

### Corpus Definition

Recovery samples:

- Single layer: generation seeds `11000–11019`; fit seeds `21000–21019`.
- Double layer: generation seeds `12000–12019`; fit seeds `22000–22019`.
- Mo/Si periodic multilayers: generation seeds `13000–13019`, spanning
  `10–100` repeats; fit seeds `23000–23019`.
- Oxide/cap stacks: generation seeds `14000–14019`; fit seeds
  `24000–24019`.
- Angle offset, scale, background, and resolution effects: 20 seeds
  `15000–15019`; fit seeds `25000–25019`.
- Footprint truncation with multiple `θ_fp` values:
  - both geometry-locked and released-fit modes use the same 20 generated
    curves and generation seeds `17000–17019`;
  - locked fit seeds are `27000–27019`, and released fit seeds are
    `27100–27119`.
- Mixed Kα:
  - dual-wavelength and monochromatic modes use the same 20 generated curves
    and generation seeds `18000–18019`;
  - dual fit seeds are `28000–28019`, and monochromatic fit seeds are
    `28100–28119`.

Ambiguity and inverse-crime guard samples:

- Deliberately ambiguous low-q curves: generation seeds `16000–16019`; fit
  seeds `26000–26019`.
- Model-error injection: 20 seeds total, 4 per class:
  - microslab interdiffusion gradient interfaces: generation
    `19000–19003`, fit `29000–29003`;
  - non-Gaussian roughness mixture: generation `19010–19013`, fit
    `29010–29013`;
  - undisclosed extra thin oxide layer: generation `19020–19023`, fit
    `29020–29023`;
  - residual Kα satellite line: generation `19030–19033`, fit
    `29030–29033`;
  - mild nonlinear detector response: generation `19040–19043`, fit
    `29040–29043`.

Noise coverage is rotated through the recovery groups so each condition appears
in at least 20 fixed-seed samples:

- no added noise;
- `1%` lognormal multiplicative noise;
- `5%` lognormal multiplicative noise;
- high-angle additive background noise plus `1%` lognormal noise.

### Acceptance Metrics

The slow recovery gate computes medians and 95th percentiles from the generated
truth and fitted best candidate. Point-estimate errors are included only for
parameters whose profile is closed on both sides and whose generating value is
inside the physical fit bounds by at least `5%` of that parameter span. Open
profile intervals are not forced to hit the point estimate, but the profile
range must cover the generating value and the confidence must downgrade from
`可信`. A primary thickness or period profile that is open on both sides is
treated as a continuous family of solutions and classified `多解`.

For each binary periodic block, uncertainty additionally profiles the joint
derived coordinates `component.i.period_a = d0 + d1` and
`component.i.layer.0.fraction = d0/(d0+d1)`. Holding either derived coordinate
fixed reparameterizes the two bounded layer thicknesses into the fixed value
plus its feasible complementary nuisance coordinate, then reoptimizes every
other parameter with the same residual and a chain-rule Jacobian (the physical
model Jacobian remains analytic; only the two-dimensional bounded coordinate
map is differentiated locally). The
slow gate consumes these joint profiles directly; it does not substitute two
independent layer-thickness profiles for period or fraction coverage.

Thresholds enumerated in design §14.3:

- Thickness and period absolute relative error: median `≤2%`, p95 `≤5%`.
- Layer thickness fraction absolute error: median `≤0.02`, p95 `≤0.05`.
- Relative density absolute relative error: median `≤3%`, p95 `≤8%`.
- Roughness for `σ≥2 Å`: median absolute error `≤1 Å`, p95 `≤3 Å`.
- Angle offset absolute error: median `≤0.002°`, p95 `≤0.005°`.
- Scale and background absolute relative error: median `≤5%`.
- Relative resolution for `η≥0.002`: median absolute error `≤0.001`.

The released-footprint spill-angle threshold is median absolute error
`≤0.02°`, p95 `≤0.05°`.

Additional slow-gate checks:

- Ambiguous samples: no sample may be `可信`, and at least `90%` must be
  `多解` or `不可信`.
- Footprint samples: locked `θ_fp` must remain locked, and released `θ_fp`
  must recover the generating spill angle within the corpus gate.
- Mixed Kα samples: the mixed doublet model must satisfy the same recovery
  metrics; monochromatic fits must expose mismatch by confidence downgrade,
  systematic residual/ACF evidence, or both, and must never be a `可信`
  recovery when thickness/period is biased by more than `5%`.
- Model-error samples: no sample may be `可信` while thickness/period differs
  from the generating value by more than `5%`; the residual ACF downgrade must
  fire for at least `70%` of the model-error corpus. This percentage counts
  the production field `result.uncertainty.residual_autocorrelation`, not the
  broader `systematic_residual` flag or a test-only ACF helper. The production
  check sorts fitted unweighted log residuals by q, examines absolute ACF over
  the first `min(20, N//5)` nonzero lags, and requires at least two lags above
  `3/sqrt(N)`.

### Commands

The slow cases use `FitConfig.fast(master_seed=fit_seed)` and retain its
default `local_workers=4`. Local refinement batches and uncertainty profiles
use a spawn `ProcessPoolExecutor` only from an importable real `__main__`
file, in the main non-daemon process, and when no cancellation callback forces
the threaded path. REPL/`<stdin>` and nested workers therefore fall back from
the process pool. Stage-B/Stage-E differential evolution currently uses its
default single SciPy worker; `local_workers` does not silently change DE
parallelism. Executable/script entry points must use an
`if __name__ == "__main__":` guard, and a frozen GUI entry point must also call
`multiprocessing.freeze_support()` before creating the application.

Strict collection for the default non-slow selection:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider --strict-config --strict-markers -m "not slow" tests/acceptance/test_synthetic_recovery_corpus.py --collect-only -q
```

Strict collection for the slow corpus selection:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider --strict-config --strict-markers -m slow tests/acceptance/test_synthetic_recovery_corpus.py --collect-only -q
```

Fast generation/metric helper checks that do not run the full slow optimizer
corpus:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider --strict-config --strict-markers tests/acceptance/test_synthetic_recovery_corpus.py::test_slow_corpus_definitions_cover_design_contract tests/acceptance/test_synthetic_recovery_corpus.py::test_recovery_metric_helpers_filter_profiles_and_measure_p95 -q
```

Full slow stage gate, expected to be run separately because it may take a long
time:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider --strict-config --strict-markers -m slow tests/acceptance/test_synthetic_recovery_corpus.py -v
```

Pass/fail evidence belongs to the canonical acceptance receipt generated by the
release gate. This algorithm document defines the thresholds but does not copy
a potentially stale run result.

## Stable Public API

Application and GUI code cross the domain boundary only through
`xrr_fitter.api`. The operation surface is:

```text
new_project()
load_project(path)
save_project(project, path)
inspect_sources(project)
select_active_dataset(project, dataset_id)
select_candidate(project, dataset_id, candidate_id)
set_batch_mode(project, mode)
set_expert_mode(project, enabled)
set_workspace_state(project, state)
clear_fit_results(project, dataset_ids)

import_data(path, beam, import_angle_offset_deg, column_mapping)
add_dataset(project, source_path, instrument, display_name, column_mapping,
            import_angle_offset_deg, beam)
remove_dataset(project, dataset_id)
set_fit_mask(project, dataset_id, mask)
set_instrument(project, dataset_id, instrument)
preview_source_update(project, dataset_id, new_path)
accept_source_update(project, preview)

set_structure(project, dataset_id, structure)
describe_parameters(project, dataset_id)
analyze_structure(project, dataset_id)
validate_parameter_settings(definitions, settings)
set_parameter_settings(project, dataset_id, settings)
validate_structure(structure, beam)
validate_sharing_rules(project, rules)
set_sharing_rules(project, rules)
suggest_oxide_layers(structure)
accept_oxide_suggestion(project, dataset_id, suggestion)
record_oxide_decision(project, dataset_id, decision)

preflight_fit(project)
fit_project(project, progress_callback, checkpoint_callback)
run_mcmc(project, dataset_id, candidate_id, config, progress_callback)
start_fit_job(project, checkpoint_path)
start_mcmc_job(project, dataset_id, candidate_id, config)
export_result(result, output_dir)
```

`inspect_sources()` returns an ordered `ProjectValidation` without mutation;
schema/object-graph validation remains an internal construction/save/load
invariant. `load_project()` invalidates stale source-dependent state, while an
explicit source replacement uses `preview_source_update()` followed by
`accept_source_update()`. All other state transitions use their owning
`set_*` operation so invalidation cannot be bypassed by application code.

The persisted `XrrProject` is the declaration and provenance source of truth.
For export, a `ProjectFitResult` is first normalized to its `updated_project`,
then each persisted `dataset.last_valid_result` supplies candidates,
confidence, uncertainty, and reporting arrays. An external result object never
overrides a different persisted graph.

## Exact Service Objective

For dataset `d`, fitted rows use unweighted physical log residuals:

```text
Delta_di = log10(Rmodel_di + Rfloor_d) - log10(Robs_di + Rfloor_d)
L_C(Delta) = 2 C^2 (sqrt(1 + (Delta/C)^2) - 1)
z_d = (log10(S_d) - log10(S_hat_d)) / tau_S,d       # only when active
J_d = (sum_i w_di^2 L_C(Delta_di) + z_d^2) / N_d
```

When the scale prior is inactive, the `z_d^2` term is omitted completely.
Region weights remain outside the robust argument, so every data point enters
saturation at the same physical threshold `C`.

The single-dataset local solver passes rows as `Delta_di`. Its custom SciPy
loss uses the following data-row values for `z = Delta_di^2` and
`m_di = w_di^2`:

```text
rho0 = 4 m_di C^2 (sqrt(1 + z/C^2) - 1)
rho1 = 2 m_di / sqrt(1 + z/C^2)
rho2 = -(m_di/C^2) (1 + z/C^2)^(-3/2)
```

An active scale-prior row is Gaussian: `rho0=2 z_d^2`, `rho1=2`,
`rho2=0`. Therefore SciPy's `0.5 * sum(rho0)` equals `N_d * J_d`.

For `K` datasets and `N_total = sum_d N_d`:

```text
J_joint = (1/K) sum_d J_d
alpha_d = N_total / (K N_d)
```

The joint local residual vector still contains raw `Delta_di`; its external
row weight is:

```text
sqrt(alpha_d) * w_di
```

For an active prior row:

```text
rho0 = 2 alpha_d z_d^2
rho1 = 2 alpha_d
rho2 = 0
```

Thus `0.5 * sum(rho0) = N_total * J_joint`. Pre-scaling `Delta`, placing
`alpha_d` inside the soft-L1 argument, or using one global `f_scale` changes
the saturation threshold and is not equivalent.

## Joint Global-To-Local Mapping

Joint compilation creates one stable global unit vector. Unshared variables
retain dataset-qualified identities; validated sharing rules map multiple
local variables to one global coordinate. Scattering copies that coordinate
to every member without averaging or clipping. Density/SLD semantics must
match; instrument resolution/footprint sharing additionally requires the same
nonempty `instrument_id` and compatible active modes. Thickness/period,
roughness, angle offset, scale, and all background families remain local.

Every scalar candidate objective is recomputed from the same `J_d` or
`J_joint` formula used for ranking. Joint projected candidates may retain a
global `ranking_objective`; that value does not replace the dataset-local
reporting objective.

## Service Seed Tree

`SERVICE_SEED_TREE_VERSION = 1`. The service root is:

```text
SeedSequence(project.master_seed,
             spawn_key=(SERVICE_SEED_TREE_VERSION,)).spawn(3)
```

The fixed branches are independent, joint, then MCMC. Independent and MCMC
derive dataset children in sorted dataset-ID order; MCMC derives candidate
children in sorted candidate-ID order. Optimizer generations then use
`SeedSequence(runtime_master_seed).spawn(count)`, so scheduling and persisted
project order do not determine the seed identity.

Bootstrap is a current runtime exception rather than an undocumented fourth
service branch. It derives a deterministic uncertainty seed with:

```text
SeedSequence([config.master_seed, 0x554E434552544149]).generate_state(...)
```

It is not the MCMC branch, is not itself produced by `.spawn()`, and is not
reported as an optimizer generation in `FitResult.child_seeds`. Example-data
generation has its own `EXAMPLE_SEED_TREE_VERSION = 1` domain and separate
generation/project seeds.

## Source Freshness And TOCTOU

Relative source declarations are resolved against runtime `base_directory`
only for I/O. Revalidation hashes raw bytes and returns `ok`, `missing`,
`unreadable`, or `hash_mismatch` without rewriting the declared path or hash.

Fitting and export perform two checks: first validate the current source
identity/hash; then read with the persisted beam, offset, and column mapping
and compare the parsed file's raw-byte hash again. A source changed between
those steps is rejected before result publication or export allocation.

In independent mode, non-current input invalidates only the affected dataset
and its candidate selection. In joint mode, one non-current member invalidates
all derived dataset state and all selections because the result graph is one
transaction.

## Checkpoint Publication And Resume

Checkpoint callbacks receive immutable whole-project snapshots. The service
attaches the new checkpoint before invoking user code; a callback exception
therefore keeps the newest resumable evidence and returns an explicit
`checkpoint_failed` result instead of losing state or claiming success.

Resume validates exact data, structure, instrument, parameter-setting and
config fingerprints, candidate IDs/order, stage summaries, consumed seed
ledger, and—when joint—the global layout plus cross-dataset checkpoint
coherence. A mismatch preserves the previous valid result/checkpoint and
returns an untrusted failure. It never silently drops evidence and starts a
fresh fit; joint failure never falls back to independent fitting.

## Export Publication Model

Export preflight validates the persisted project graph, selected candidates,
source freshness, reporting-array shapes, and legal finite/null state before
allocating a run. All dataset and root artifacts are written into one private
`.partial-*` sibling directory. The manifest is checked for containment and
nonempty files, then files and direct-child directories are fsynced.

Only after the complete tree is durable is it published by an exclusive,
same-filesystem, no-replace rename to `<UTC timestamp>-<8hex>`. Any exception
removes only the partial directory owned by that call and propagates the
original error. Existing completed runs are never overwritten.
