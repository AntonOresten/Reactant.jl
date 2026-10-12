# StructuredReactant compatibility work: results and upstream extraction

Report date: 2026-09-14. Audited branch: `structured`, HEAD
`8d7b9292d6c79c8aa53dfed4afc6b8d5af384ff1`, based on Reactant main
`64e349ffe`. This report covers the compatibility/differentiation investigation,
including its later isolated experiments. It does not count the previously
merged specialization/latency PR as a new result.

The useful output is a collection of independently reproducible compiler fixes,
a broader regression corpus, and a substantially more capable experimental
tracing implementation. The collection is not yet a release-ready combined
stack. Several valuable Reactant fixes can be submitted without
StructuredReactant. The largest shared reference/object-state changes need
design and performance work before integration.

This report is organized by ownership and extraction boundary, rather than by
experiment chronology. Item IDs identify work packages, not necessarily separate
PRs. Some packages should be split; others span two repositories.

## Recommended first submissions

These have narrow implementation boundaries and useful existing reproductions.
“First submission” means prepare a clean branch and run its independent tests;
it does not mean submit the experimental snapshot wholesale.

| Priority | Package | Why it is a good first extraction |
|---|---|---|
| 1 | **R01: Reactant constant-cache ownership** | A small wrong-result fix. Deduplicate immutable MLIR values while returning independent mutable tracer handles. Separate tests already use ordinary Reactant and the standard native artifact. |
| 2 | **R02: Reactant cached-call alias signatures** | Different argument alias patterns could reuse an incompatible callee. Independent of the reference prototype and CompilerCaching. Tested on three Julia versions with the standard artifact. |
| 3 | **R08: Reactant traced-range element type** | One-line implementation correction with a 66-assertion regression and version/frontend controls. |
| 4 | **R09 and R05: pass printing and fill rank** | Two separate small fixes: the MLIR wrapper's argument mismatch, and preservation of tuple-known array rank. |
| 5 | **I01: IRStructurizer expression operand tracking** | A computed callee was omitted from use/replacement analysis. Standalone execution regression and full dependency suites exist. |
| 6 | **C01: CompilerCaching 1.11 method-table edges** | Standalone stale-cache reproduction; the complete patched library suite passes on all three Julia versions. |
| 7 | **J01–J04, J06–J07: native optimizer fixes** | Small Julia-independent MLIR reductions, including wrong results, invalid IR, and nonterminating rewriting. Existing standalone patches apply to the recorded upstream base. |
| 8 | **R13–R15: generated-function worlds and compiler metadata** | Small Reactant source changes with failure controls. Extract their regressions from the larger invoke fixtures before submission. |

The broadcast and view fixes are also useful early work, but should be split
carefully because the same files contain later reference-prototype changes.
The explicit-invoke implementation has strong correctness evidence but is a
larger compiler integration change; it belongs after the small fixes.

## What exists, and where

| Material | Authoritative location | Status |
|---|---|---|
| Earlier accepted implementation changes | Current worktree under `src/`, `ext/`, and `lib/StructuredReactant/src/` | Uncommitted. Includes several independent Reactant fixes and frontend corrections. |
| Latest finalized combined Julia source | [preserved: structured-compatibility-41/trace-label-world-v3](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-41/README.md) | Frozen Reactant and StructuredReactant copies, per-file hashes, incremental patches, and versioned environments. Preferred reference snapshot for consolidation. |
| Newer selected-wrapper cache experiment | [preserved: structured-compatibility-42/dependency-cache-v2](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-42/dependency-cache-v2/reactant.patch) | Public-invoke and combined checks pass on the recorded configurations; additional cache/world/concurrency questions remain. Round 42 has no finalized overall handoff. |
| Final tested IRStructurizer candidate | [preserved: structured-compatibility-38/identity-use-index-v1/IRStructurizer](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-38/README.md) | Upstream main `496c3d3` plus the two local use-tracking changes, I01 and I02. |
| Final native source and tools | [preserved: structured-compatibility-39/native-zero-like-v1](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-39/README.md) | Separate Enzyme and Enzyme-JAX trees, incremental patches, frozen CPU library and command-line tools. Not the standard package artifact. |
| New CompilerCaching candidate | [preserved: structured-compatibility-42/CompilerCaching-mt-edges-v3](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-42/compiler-caching-mt-edges-v3.patch) | Library-only change, full normal-loading tests completed successfully. It is not wired into the Reactant cache experiment. |
| Regression requirements | [test/](/home/anton/dev/jool/Reactant/lib/StructuredReactant/test) and [compatibility ledger](/home/anton/dev/jool/Reactant/lib/StructuredReactant/test/compatibility.md) | Many new acceptance files are deliberately standalone and are not included in the regular frontend runner. Some retain ordinary failures. |
| Extraction checks made for this report | [preserved: structured-upstream-report-2026-09-14/audit.json](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/audit.json) | 29 patch application checks succeeded against specified bases. Includes hashes and source-preservation results; not new numerical testing. |

Frontend source size, counting physical `.jl` lines including comments/blanks:
2,186 at HEAD, 2,533 in the worktree, and 3,616 in the latest finalized prototype.
The prototype also adds shared Reactant implementation: `BranchReferences.jl`,
`ArrayIdentity.jl`, `ReferenceProjections.jl`, and `ADShadows.jl` total 961 lines,
in addition to changes in existing Reactant files. Frontend line count alone
therefore understates the implementation and review burden.

For this report, all recorded files in the selected final manifests were
rehashed: 548 Reactant, 30 frontend, 32 IRStructurizer, 1,862 Enzyme, and 3,093
Enzyme-JAX files. There were no mismatches. These are manifest file counts,
not numbers of changed files. No implementation was promoted, staged, committed,
or published while preparing this report.

## Reactant fixes independent of StructuredReactant

These address Reactant behavior and can be justified without introducing the
automatic frontend. That does not mean every existing patch is independent of
the experimental *Reactant* source on which it was developed. The extraction
column identifies that distinction.

| ID | Fix and ownership | Existing evidence | Extraction status and source |
|---|---|---|---|
| **R01** | **Constant-cache ownership**, `src/Ops.jl`. Cache the MLIR constant value, not a mutable `TracedRArray` handle. Mutating one returned array must not mutate another logically independent constant or poison a later lookup. | Isolated `constant-cache-only` controls used the standard native artifact on Julia 1.11–1.13. Tests check values, independent arrays, reload order, and derivatives. | In worktree. [Focused source-and-test patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R01-constant-cache.patch) applies to HEAD excerpts. [Original evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-11/README.md). Good first PR. |
| **R02** | **Cached-call alias signatures**, `src/Tracing.jl`. `(x,x)` and `(x,y)` may have different flattened MLIR argument counts even when tensor types match. Encode traversal-local alias identities. | Both call orders fail on the old source. 36 numerical/mutation/reverse checks per frontend per Julia version: 216 assertions. Standard native artifact; upstream AD 115 per frontend. | In worktree. [Standalone patch with test](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-10/cache-key-only.patch), [evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-10/README.md). Applies to HEAD excerpts. Keep separate from conditional-reference reconstruction and CompilerCaching. |
| **R03** | **Fused broadcast result typing**, `src/TracedRArray.jl`. Preserve primitive/traced alternatives through nested broadcast inference instead of joining too early to an unusable abstract type. | 52 added broadcast regressions; combined broadcast, activation, and NNlib runs through vanilla and automatic routes. | In worktree. [Original patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-3/reactant-broadcast.patch), [evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-3/README.md). Extract the inference/copy changes separately from later allocation, literal, and reference changes. |
| **R04** | **Complex scalar literals in broadcasts**, `TracedRArray.jl`/`TracedUtils.jl`. Let the Julia broadcast function promote unsupported complex literal types before attempting tensor construction; unwrap scalar `Ref`s and apply assignment conversion first. | Round 28 literal, broadcast, and AD checks; targeted version coverage on 1.11/1.12, broader 1.13 checks. | Scratch only. Use the [final v8 patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-28/patches-v8/reactant-complex-scalar-broadcast.patch), not intermediate literal experiments. Applies to its round-25 base. Semantically independent; needs an isolated ordinary-Reactant rerun on the extracted patch. |
| **R05** | **Preserve tuple-known rank in `Ops.fill`**, `src/Ops.jl`. The tuple entry point lost rank information when delegating through a vector. | 34 targeted checks; type-number and array-operation checks in both modes. Resolved the traced three-matrix example without an iterator special case. | In worktree. [Patch with tests](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-5/reactant-fill-rank.patch) applies to HEAD excerpts. [Evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-5/README.md). |
| **R06** | **Traced scalar view indices, view copying, and strided dynamic reads**, `Indexing.jl`, `TracedUtils.jl`, and `stdlibs/LinearAlgebra.jl`. Represent a traced scalar view index with rank-zero array semantics; preserve dimensions and parent transformations; choose gather when dynamic slicing cannot represent strides. | 312 view assertions per frontend on Julia 1.11–1.13. Fixes the remaining nested-AD view case; upstream AD reaches 115. Some broad view execution requires native J04. | In worktree. Extract from the worktree against HEAD together with [test/core/traced_views.jl](/home/anton/dev/jool/Reactant/test/core/traced_views.jl). Separate scalar view support, strided-read selection, and recursive writeback if they can retain independent tests. Sparse-copy ambiguity resolution belongs with this change. [Evidence](/home/anton/dev/jool/Reactant/lib/StructuredReactant/test/compatibility.md:1204). |
| **R07** | **OffsetArray and nested-wrapper coordinates/writeback**. Translate between original and materialized parent axes; recurse through transpose/adjoint/offset parents; allocate broadcast outputs through `similar`. | Complete OffsetArrays file: 78 checks in each of three modes on each of three Julia versions. Both values and axes are observed. | In worktree. [Original delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-7/reactant-offset-views.patch) predates the final host-axis refinement; extract final worktree behavior, not that delta alone. [Final evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-7/README.md). Shares some writeback infrastructure with R06; do not include experimental reference projection hooks. |
| **R08** | **TracedUnitRange materialization element type**, `src/Indexing.jl`. Use `unwrapped_eltype(v)` rather than a possibly already traced endpoint type. | 66 assertions per frontend on three versions; 396 total. Standard artifact, with full ranges/indexing/wrapped-array checks on 1.13. | In worktree. [Focused source-and-test patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R08-range-element-type.patch), [evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-12/README.md). Applies to HEAD excerpts. |
| **R09** | **MLIR pass-manager printing wrapper**, `src/mlir/IR/Pass.jl`. Supply the current API's printing flags/output path and dispose flags correctly. | Four assertions include actual before/after IR dumps. The [upstream wrapper](https://github.com/EnzymeAD/Reactant.jl/blob/main/src/mlir/IR/Pass.jl) fetched on the report date still has the older call shape. | In worktree. [Focused source-and-test patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R09-pass-printing.patch) applies to HEAD excerpts. Does not fix expired MLIR-handle printing. |
| **R10** | **`kron`/`kron!` module ownership across Julia versions**, `stdlibs/LinearAlgebra.jl`. Extend the owning module without stale/inactive import issues. | QA and complete linear algebra file, 222 assertions per frontend per version in the recorded matrix. | In worktree. [Focused source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R10-kron-owner.patch) applies to HEAD excerpts. Needs a minimal version/ownership regression in the PR; do not bundle unrelated view writeback hunks. |
| **R11** | **SubArray metadata and broadcast unaliasing**. Keep cached host geometry out of tensor tracking, invalidate fast-linear metadata when index types change, use conservative aliasing for traced indices, and preserve wrapper types in `unaliascopy`. | Round 28 explicit-region/view checks and wrapper snapshots. Later frontend bounds work fixes a separate prepared-call problem. Plain checked traced index-array bounds remain a distinct limitation. | Scratch only. [Final metadata patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-28/patches-v8/reactant-view-metadata.patch) and [final alias/copy patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-28/patches-v8/reactant-view-alias-and-copy.patch) apply to their experimental base. Rebase onto R06/R07 and test vanilla `@trace`; these do not inherently require conditional-reference machinery. The top-level older alias-only patch is incomplete. |
| **R12** | **Schedule existing Enzyme batch legalization**, `src/compiler/Compiler.jl`. Run `enzyme-batch-to-stablehlo` before XLA export. | Batched forward-over-reverse products previously retained unsupported `enzyme.extract`/`enzyme.concat`. Native reproducer and Julia product/HVP checks pass with the pass. | Scratch only. [One-site source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-25/lower-enzyme-batches.patch) applies to HEAD excerpts. No conversion machinery is copied into Reactant. Use a reproducer supported by the standard artifact or declare the needed native derivative updates. |
| **R13** | **Generated return-type definition order**, `Interpreter.jl`/`utils.jl`. Define the generated method after bindings used by its inference callback exist. | The same no-invoke higher-derivative source fails on two source-loading controls and passes after reordering. Normal precompilation had hidden the issue. | Scratch only. [Focused source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R13-generator-order.patch) applies to HEAD excerpts. Extract the standalone source-loading regression from [round 41](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-41/README.md); no invoke implementation is required by the fix. |
| **R14** | **Render compiler labels in the trace world**, `src/utils.jl`. Late-defined enum display methods must be visible when producing labels. | Source-loaded AD failures plus late-enum checks across 1.11–1.13. Final source-loaded AD passes 115 per frontend. The original normal-loading vanilla enum control already passed and must not be presented as a failure. | Scratch only. [Focused two-site patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches/R14-label-world.patch) applies to HEAD excerpts. Uses the supplied world, not an unrestricted latest-world lookup. Package the two call sites together; the earlier `Base.invoke_in_world` attempt did not fix the problem. |
| **R15** | **Opaque compiler metadata and cache identity**, `src/Tracing.jl`. Preserve `Method`, `MethodInstance`, and `CodeInstance` as host metadata and include identity in trace-cache keys. | Pre-invoke ordinary Reactant control fails four focused MethodInstance cache/key/numerical checks. Final metadata fixture passes 107 assertions per frontend on 1.13, including analytic derivatives. | Scratch only. [Focused metadata patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-41/metadata-v1/metadata.patch) applies to HEAD excerpts. Its early candidate name is historical: the same metadata changes survive in the final stack. Extract the no-invoke cache regression separately from tests that depend on R16. Preserving CodeInstances is not support for calling them. |
| **R16** | **Preserve explicit `invoke` targets**, `src/utils.jl`. Retain the selected method through inference/rewriting and compile that target through Reactant's native wrapper. | Original LU reconstruction errors are fixed. Initial acceptance: 608 assertions across versions/frontends; final combined invoke/metadata/bounds matrix: 1,301. These are overlapping suites, not additive unique tests. | Scratch only, larger review. Start with [round-40 v6 Reactant patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-40/selected-invoke-v6/reactant.patch) and retain the final round-41 routing/source-inference corrections. Remove unrelated dictionary/reference prerequisites during extraction. Keep the frontend's prepared-invoke changes separate. Direct internal bridge calls still crash in the round-42 baseline and candidate. |
| **R17** | **Dependency-aware cache for selected invocation wrappers**. Store reusable inferred wrapper IR in CompilerCaching; recreate OpaqueClosures in the calling world. | Round-42 v2 public-invoke tests pass 30 each on 1.12/1.13; combined 1.13 checks pass 225 vanilla/230 structured. Old closures can retain old dynamic dispatch even when their CI remains valid. | Experimental extension of R16. [v2 patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-42/dependency-cache-v2/reactant.patch). Adds CompilerCaching to Reactant, unlike the smaller fixes. Enabled only on 1.12+; 1.11 retains exact-world fallback. Concurrency, old-world calls, cache growth, and the internal bridge crash still need work. No persistence or cold-latency claim. |
| **R18** | **Inactive fields in differentiated composite results**, `src/Enzyme.jl`. Preserve inactive return fields and avoid nonexistent tangent outputs; classify flattened leaf activity correctly. | Inactive-result and explicit-shadow tests, including Boolean fields that change between executable calls and comparison with native Enzyme. | Shared Reactant semantics, potentially separable from the reference redesign, but the existing [AD patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-24/explicit-ad-shadows.patch) also contains S03. Extract activity/result bookkeeping independently, then validate without reference types. Do not submit that whole patch as a small inactive-field fix. |
| **R19** | **Avoid path bookkeeping specialization**, `src/Ops.jl`. Use an explicitly typed path set and a comparator that does not specialize on every full tuple shape. | Serial prototype comparisons identify and remove repeated warm Julia compilation in path sorting. Existing `case` checks and versioned prototype gates pass. | Scratch only. [Focused patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-13/path-order.patch) applies to HEAD excerpts. It is ordinary Reactant code, but all performance evidence is from the reference prototype. An isolated ordinary-Reactant benchmark is required before making a general latency claim. |

For the small patches generated during this report, only the source/test hunks
were isolated and application checked. Their numerical evidence comes from the
recorded runs, not a fresh execution of the newly assembled PR-sized patches.
R03/R04/R06/R07/R11 especially need careful hunk selection: copying whole final
source files would pull in unrelated prototype work.

## Shared Reactant changes that are not ready as small fixes

These belong in Reactant if the design is accepted, because vanilla `@trace`
exposes the same semantic needs. They should not be hidden inside the
StructuredReactant package. They also should not be submitted as a bundle of
routine bug fixes without a design review.

| ID | Scope and surviving implementation | Evidence and remaining work |
|---|---|---|
| **S01: conditional references and logical identity** | `Types.jl`, `BranchReferences.jl`, `ReferenceProjections.jl`, `ArrayIdentity.jl`, `Ops.jl`, `Tracing.jl`, and array access/writeback integration. Distinguish rebinding an array variable from mutating its selected storage. Preserve aliases through branches, loops, views, wrappers, and cached calls. The frontend carries/joins the state supplied by these shared mechanisms. | The original ordinary-Reactant example `y = c ? copy(x) : x; x .*= 10; sum(y)` gives the wrong result on the baseline for the aliasing branch. Direct vanilla `@trace` controls and substantial mutation/gradient acceptance improve with the prototype. Remaining representation, dispatch, typed-state, and cost concerns prevent promotion. [Initial requirement and control](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-8/README.md), [later identity work](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-15/README.md). |
| **S02: cached object graphs and container state** | Mutable argument isolation for cached `Ops.call`, returned aliases/freshness, cycles, undefined fields, dictionary/host-array state, and executable result reconstruction. Later dictionary cache keys include table history; value-type conversion preserves key layout. | Many genuine fixes, but the later dictionary patches are deltas on our graph-cloning prototype. Identity-hashed mutable-key cloning still changes observable iteration and can give wrong primal/gradient/HVP results. Distinguish pre-existing Reactant defects from regressions introduced by this prototype. [Object/result work](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-22/README.md), [dictionary correction and retained failures](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-37/README.md). |
| **S03: derivative storage aligned with reference graphs** | `ADShadows.jl` and `Enzyme.jl`: preserve backing storage in zero derivatives, map explicit forward/reverse buffers to selected views, deduplicate shared storage, preserve independent batch lanes, and write back once. Includes the later shadow-layout correction. | Explicit/shared/overlapping-buffer numerical tests pass on recorded candidates, including higher derivatives. This implementation depends on S01 and changes the AD boundary. Extract R18 if possible; keep storage-graph support with the reference design. [AD integration](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-24/README.md), [final layout delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-28/patches-v8/reactant-shadow-layout.patch). |

The dictionary implementation deserves particular scrutiny under our separation
of concerns goal. It copies `Dict` slot/probe/tombstone metadata and uses Base
layout details across tested Julia versions. CompilerCaching does not solve
that object-graph problem. The two dictionary increments are
[cache-state.patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-36/dictionary-cache-state-v1/cache-state.patch)
and [value-layout.patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-37/value-layout-v1/value-layout.patch).
They pass 47 and 41 focused assertions per frontend per version respectively,
but are not evidence that arbitrary mutable-key cloning is sound or that the
entire patch is appropriate for upstream Reactant.

### Known cost of the reference design

The earlier serial CPU/PJRT comparison used the same native library for both
variants and timed tracing only, with two fresh processes per workload/variant
and five warm retraces per process. The baseline was StructuredReactant without
the reference prototype, not vanilla Reactant.

| 256-element Float64 workload | Baseline warm trace | Prototype after path-sort correction | Raw StableHLO operations: baseline → prototype |
|---|---:|---:|---:|
| Two independent loop-carried arrays | 2.54 ms | 20.4 ms | 94 → 5,612 |
| Four independent loop-carried arrays | 3.52 ms | 108 ms | 136 → 38,821 |

An attempted compact representation in round 14 did not improve the measured
warm times. Later correctness changes have not received an equivalent cost
comparison. These numbers are a demonstrated design concern, not a measurement
of the latest snapshot and not a nanoGPT benchmark. Before accepting S01–S03,
make ordinary independent-array paths cheap and rerun an uncontended benchmark.
See [round 13 evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-13/README.md) and
[round 14 evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-14/README.md).

## IRStructurizer and CompilerCaching

| ID | Upstream package and change | Evidence | Extraction boundary |
|---|---|---|---|
| **I01** | **IRStructurizer: visit expression operand one when it is a value.** Only `:invoke` has the non-value MI/CI prefix. Computed `:call` callees and `:new`/`:splatnew` type operands must participate in uses, users, and replacement. | API baseline 11 pass/15 fail; separate execution baseline 1 pass/3 fail. Full candidate suites including 30 added assertions: 1,710/1,711/1,711 on Julia 1.11/1.12/1.13. | [Standalone source/test patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-34/use-index-v1/uses-with-tests.patch) applies to pinned upstream main. No Reactant dependency. |
| **I02** | **IRStructurizer: operand identity in use indexing/replacement.** After SSA normalization, use identity rather than user `hash`/`==`; do not merge equal-but-distinct mutable literals or differently typed numeric constants. | 24 focused assertions on each version; full integrated suites 1,734/1,735/1,735. Fixes the automatic Static integration failure without a Static-specific frontend rule. | [Standalone delta with tests](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-38/identity-use-index-v1/identity-uses-with-tests.patch), based on I01. Requires API review because literal equality semantics change. Can be reviewed after I01 or as a clearly separated second commit. |
| **C01** | **CompilerCaching: register Julia 1.11 method-table and invoke-signature dependencies.** Convert CI dependencies to MI form and let Julia's standard backedge decoder handle the version's edge encoding. | Correctly encoded baseline: 3 pass/4 fail, retaining value 2 after dispatch should change to 7. Candidate: 7/7. Complete normal-loading library suites: **355/422/436**, all passing. | [Source, documentation, and test patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-42/compiler-caching-mt-edges-v3.patch) applies to the frozen 0.5.1 source. Independent of Reactant, R16/R17, IRStructurizer, and native builds. Julia 1.11 still lacks per-CI dependency precision and the newer binding-edge support. |

The public files fetched on 2026-09-14 still contain the affected implementations:
[IRStructurizer use tracking](https://github.com/maleadt/IRStructurizer.jl/blob/main/src/ir/uses.jl)
and [CompilerCaching source](https://github.com/maleadt/CompilerCaching.jl/blob/main/src/CompilerCaching.jl).
Fetched blob IDs were `6b629642d567eb150457743fe6393688a645932f` and
`8c5f530c9880cb4aa7e1e264fd96285bc984d25a` respectively. The latter exactly
matches the frozen CompilerCaching baseline source. This is a source spot-check,
not a search of all open PRs.

Earlier main-branch IRStructurizer loop-promotion/hoisting fixes were already
upstream work. Do not repackage them as ours. The expanded frontend corpus
requires those fixes; the official 0.6.4 and cached 0.6.3 sources had retained
structuring/execution failures in the recorded release controls. Detecting a
capability on main is not proof that a released compatibility bound suffices.

The source-mode CompilerCaching full-suite attempts initially failed package-image
reuse checks. The final normal-loading runs passed all assertions, including
those checks. Preserve both records; report the successful command's loading
mode accurately. See [final status records](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-42/library-full-normal-results.json).

## Native changes: Enzyme-JAX

These are native MLIR/HLO changes with Julia-independent reductions. They belong
in [Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX), not in frontend tensor
special cases. Some extend capability and need design review; others are narrow
correctness fixes. The original native base is
`3c890ff39e3f2b57206d923bab948adf2da5589b`.

| ID | Change | Evidence and extraction |
|---|---|---|
| **J01** | **DUS/slice intersection and indexing**: disjoint writes, clamped starts, and strided slice rebasing. | Baseline aborts with an invalid negative slice limit. Standalone structural/numerical cases and Polybench follow-up. [Patch with lit tests](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-2/native/dus-slice-simplify.patch) applies independently to the recorded upstream base. |
| **J02** | **Concat/reshape/reduce semantics**: preserve valid axes, reduction initializers, and captured values during batching. | Three independent pre-fix failures include 17 instead of 27 and 7 instead of 4. Existing concat/reduction pipelines pass after correction. [Patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-2/native/concat-reshape-reduce.patch), [reproduction notes](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-2/native/concat-reshape-reduce.md). Applies to the recorded upstream base. |
| **J03** | **Concat/reshape/slice element order and stride alignment**. | Six numerical/layout regressions; old rewrite gives wrong numbers or invalid IR. Exact captured two-matrix optimization pipeline agrees with an independent oracle after the fix. [Patch with tests](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-2/native/concat-reshape-slice.patch) applies to the recorded upstream base. |
| **J04** | **Do not create IR on a failed reshape/slice rewrite match**. | Failed matches inserted constants and repeatedly restarted greedy rewriting. Bounded native regressions fail promptly before the fix; Gram-Schmidt and traced-view follow-ups complete afterward. [Patch with tests](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-2/native/reshape-failed-match.patch) applies to the recorded upstream base. |
| **J05** | **Normalize eligible general loops for checkpointed reverse AD**. Count a replayable loop in an inactive helper and expose counted replay to the existing checkpoint machinery; retain ordinary primal behavior and reject effects that cannot be replayed. | Native structure/effect checks and Julia convergence/mutation/nesting checks exist. Later general-loop acceptance passes 180 primal/derivative assertions with automatic policy. This is a new pass/capability, not a small local rewrite fix. Source survives in the final native tree; originally included in the [combined native patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-23/native-integration/enzyme-jax.patch). Extract pass source, registration/build wiring, and tests without the other optimizer patches. Reactant pass scheduling and frontend policy are separate integration changes. |
| **J06** | **Correct insertion point in `slice_if`**. | Three standalone pre-fix verification failures; structural/numerical regression and Julia projected-view follow-up. [Standalone patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-11/slice-if-upstream.patch) applies to the recorded upstream base. |
| **J07** | **Keep zero-extent results in exportable StableHLO form**. Replace illegal `tensor.empty` outputs with empty StableHLO constants, avoiding canonicalization loops. | Four structural cases, nine native numerical checks; ordinary Reactant empty-array primal failures disappear when changing only the native library. [Standalone patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-12/empty-tensor-upstream.patch) applies to the recorded upstream base. |
| **J08** | **Clear consumed branch-result adjoints**. | Independent native derivative is 180 instead of 100 before the fix. The native matrix passes 32 checks afterward; changing only the library changes the Julia policy/gradient control from 14/36 to 36/36. [Source and regression patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-15/enzyme-nested-if-adjoint.patch). Generic HLO reverse semantics; no frontend dependency. |
| **J09** | **Batched select, transpose, broadcast, sum, and dot derivatives**. Preserve derivative lanes, contraction batch dimensions, and permutations. | 33 numerical expectations across 15 native modules, including five contraction layouts. Two-lane controls fail on the predecessor. [Rule delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-24/batched-native-derivatives.patch) and [round-24 evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-24/README.md). Consolidate with the final select-predicate changes; do not assume this one incremental delta contains all earlier select work. |
| **J10** | **Product/sum/padding AD, active initial values, and batched zeros**. Zero-safe product derivatives avoid division; preserve initial/padding derivatives and lane shapes. | 239 real numerical expectations across 126 modules, plus later corrected complex-product controls. Julia products and checkpointed higher derivatives tested across versions. [Rule patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-25/product-reductions-and-padding.patch) is against J09's source. Split by coherent derivative families with native tests rather than one broad unexplained rule change. |
| **J11** | **Complex component reverse zeros use cotangent shape**. | Four batched reverse conjugation cases fail before correction. Corrected matrix: 56 numerical expectations across 52 modules, including product controls. [Final v2 patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-26/complex-component-batches-v2.patch) applies to the recorded round-25 base; earlier v1 is a failed build attempt. |
| **J12** | **Stop pad/concat rewrite cycling while preserving useful factoring**. | Minimal native regression hangs before the fix. Corrected captured pipeline passes 20 numerical expectations; both frontends' batched complex HVPs and broader AD complete under default optimization. [Final patch with regression](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-27/pad-concat-convergence-with-regression.patch). Applies to its round-26 base. The earlier one-rule and analysis-attribute attempts did not solve it. |
| **J13** | **Conditional dynamic tape placeholders and reverse tape-count dominance**. | Original conditional-loop AD reductions fail; subsequent fixes remove invalid dynamic constants and forward-region SSA references. Intermediate v2 reaches 8/18 numerical executions; J14 brings the same matrix to 18/18. [Empty-cache delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-34/empty-conditional-cache-v1.patch), [reverse-count delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-34/reverse-cache-count-v2.patch). Use final source and retain J14 as a tested prerequisite; do not present v1/v2 alone as full support. |
| **J14** | **Fold constant `if` operations with dynamic result shapes** while preserving consumer types. | Six reverse cases verify, 18 analytic executions pass. Separate FileCheck regression covers true/false branches, call interfaces, and the optimization-disable attribute. [Source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-38/native-constant-if-v1/constant-if.patch), [test](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-38/native-if-inline-dynamic.mlir). Applies to J13's final source. |
| **J15** | **Raise shape-aware zero splats into StableHLO**, paired with E05. | Dynamic inactive-zero and higher-derivative checks pass with the paired implementation. [ArithRaising patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-39/native-zero-like-v1/jax.patch), [evidence](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-39/README.md). Requires the Enzyme interface change and coordinated review; it does not solve runtime dynamic-tape allocation in XLA. |

J01–J04/J06/J07 were application-checked against original pinned upstream file
excerpts during this report. Other checked native deltas were checked against
their stated experimental predecessor, not current upstream main. All need
current-base reconciliation before submission. The native test files should
travel with each extraction; a shared `.so` alone is not a useful upstream
deliverable.

## Native changes: generic Enzyme

These belong in [Enzyme](https://github.com/EnzymeAD/Enzyme), even when their
smallest convenient integration test uses Enzyme-JAX dialects. The recorded
Enzyme base is `3e7cc67727eadcdb6cc19bc24292a6fe7aed100c`.

| ID | Change | Evidence and extraction |
|---|---|---|
| **E01** | **Erase consumed gradient reads/writes during branch removal**, `enzyme/Enzyme/MLIR/Passes/RemovalUtils.cpp`. | Early-return reverse AD exposed gradient operations left under an `if`. Native regression and existing branch/cache/checkpoint tests pass after cleanup; expanded Julia primal/forward/reverse checks follow. Extract the first file from [full-repository patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-23/native-integration/enzyme-repository-with-zero-seed.patch) with its regression. |
| **E02** | **Classify integer/Boolean tensors as inactive by element type**, `enzyme/Enzyme/MLIR/Interfaces/EnzymeLogic.h`. | Two `if`s and one select reproduce a wrong gradient without Julia, arrays, or loops. Four focused numerical checks and the earlier adjoint matrix pass after correction. [Small source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-17/enzyme-tensor-activity.patch), [native regression](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-17/if_predicate_activity.mlir). Do not replace this fix with `dataflow=true`; that control produced wrong zero gradients. |
| **E03** | **Keep return activity consistent when removing a zero seed**, `enzyme/Enzyme/MLIR/Dialect/Ops.cpp`. | Old compiler fails the standalone zero/nonzero-seed case; corrected compiler passes four numerical checks. Extract the `Ops.cpp` hunk from [repository patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-23/native-integration/enzyme-repository-with-zero-seed.patch). [Reproduction/build notes](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-23/native-integration/README-zero-seed.md). |
| **E04** | **Publish and restore checkpoint replay seed mappings**, `LoopCheckpointing.h`. | A tensor from a forward branch was incorrectly referenced from a reverse region. Native baseline/candidate controls, static-periodic and dynamic-binomial tests, and nested-capture primal/reverse/HVP checks exist. [Source patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-30/scoped-checkpoint-seeds-v1.patch) plus [Enzyme-JAX regression patch](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-30/checkpoint-scoped-seeds-regression.patch). Applies to the recorded Enzyme predecessor; coordinate the cross-repository test. |
| **E05** | **Provide the primal shape when constructing inactive zero derivatives**. Add a shape-aware interface path and update uses/insertion points; static types keep a fallback. | Seven dynamic direct cases fail on the predecessor, static control passes both; 25 direct numerical executions pass on the candidate. Six reverse and six HVP cases verify and pass 18 analytic executions each after concrete input specialization. [Seven-file interface delta](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-39/native-zero-like-v1/enzyme.patch), paired with J15. This is an API proposal requiring upstream review, not a small frontend fix. Unranked/broader batched support remains outside the evidence. |

Path convention matters: the original `enzyme-repository*.patch` files use the
full repository's `enzyme/` prefix. Later build-tree patches start at
`Enzyme/...`, relative to that subtree. Normalize the paths when preparing PRs;
do not apply both representations of the same change.

## Changes that stay in StructuredReactant

These implement the automatic frontend's integration policy or lowering. They
should use generic upstream facilities, while retaining responsibility for
symbolic execution and runtime tracing decisions.

| Package | Frontend responsibility | Evidence / dependency boundary |
|---|---|---|
| **F01: inference and native-call boundaries** | Overlay-aware result inference, keyword/callback routing, updatable-stdlib recognition, explicit native escapes, and avoiding host exception-path inlining that corrupts traced loop structure. | FFT, RNG, linear algebra, callback and host-error controls. Do not copy Base/stdlib algorithms into the frontend. Later host-only prepared invokes retain selected targets and call-site flags. |
| **F02: branch/loop state semantics** | Run condition effects exactly once; update loop carries simultaneously; recognize mutation-only traced loops; preserve caller-observable state and exits. | Condition, recurrence, mutation-loop and alias acceptance. The generic alias/storage representation is S01; its region scheduling and carries belong here. |
| **F03: iterator and bounds semantics** | Preserve native range iterator values/state/availability, final iterator projections, lexical bounds context and call-site `@inbounds`. Prove counted-loop eligibility instead of adding an ad hoc loop recognizer. | Rounds 29–34 have iterator, overflow, observable-result, side-effect, primal/reverse/HVP, and bounds controls. Keep IRStructurizer promotion enabled; earlier disable-promotion experiments regressed counted lowering. |
| **F04: frontend IR preparation and use analysis** | Preserve undefined phi edges/dominance during our own cleanup, prepare operand/continuation information, and recognize unused loop results through upstream use APIs. | Phi verification controls and counted/observable-result gates. The phi bug was ours, not IRStructurizer's. I01/I02 are the generic upstream use-analysis fixes. |
| **F05: checkpoint policy and AD callbacks** | Select configured/default checkpoint strategy and propagate it through nested structured calls and function-valued `Enzyme.Const` callbacks. | Native replay support is J05/E04, not frontend derivative machinery. Enabling automatic general-loop policy requires the native pass/artifact; configuration alone cannot manufacture backend support. |
| **F06: diagnostics, recovery, and dependency capability checks** | Reject unsupported symbolic states with method context, clean up failed tracing scopes, and recognize required structurizer capabilities. | Recovery gate has 24 checks, including successful subsequent traces. Unsafe formatting of expired MLIR handles remains a shared native-lifetime problem; do not hide it with a frontend-only printer. |
| **F07: selected invocation integration** | Preserve selected methods in prepared `:invoke`, resolve them separately from ordinary dispatch, and interpret Julia's keyword adapter correctly. | Complements shared R16; does not replace it. Runtime shadow-representation redispatch still needs an explicit semantic contract. |

The final source for these is the round-41 frontend snapshot, not every earlier
incremental patch concatenated in date order. The snapshot incorporates fixes
that superseded temporary workarounds. Its broad compatibility dependencies
must remain visible in the extraction plan.

## What the test evidence establishes

- The original published Polybench corpus has **29 valid Julia references**;
  all 29 pass four default-optimized executions on recorded combined stacks.
  Original Durbin remains invalid. A **separate corrected Durbin** has a pinned
  C-reference comparison (76 host checks) and 75 primal/gradient/HVP checks per
  frontend. Do not relabel that as the unmodified 30th kernel passing.
- The 50-file, two-frontend survey completed 100 processes: 91 exited zero and
  nine did not. It used an earlier stack, not the final round-41 source. Later
  follow-ups repaired a missing LogExpFunctions test dependency, fixed Static's
  use-index failure, and fixed LU invocation. This is useful coverage, not a
  full final-stack Reactant-suite result.
- The regular frontend suite reached 400/402/402 on Julia 1.11/1.12/1.13 at
  recorded milestones. Later final invoke/metadata/bounds checks pass 1,301
  across both frontends and three versions. The final unchanged upstream AD
  file passes 115 per frontend on 1.13.
- Native checkpoint corpus results retain **29 passing and one failing RUN
  directive**, matching the baseline's nested-full-binomial golden mismatch.
  This is not an all-green native suite.
- Automatic and binomial conditional-loop primal/reverse/HVP gates pass 96
  each on the combined native candidate. Disabled checkpointing still records
  48 passes and eight errors at XLA `dynamic_pad` import.
- All of this is CPU evidence. Earlier selected MPI, sharding, Python/export,
  and integration runs belong to their recorded snapshots. GPU/TPU behavior,
  complete supported-version coverage, latest-stack precompile reliability,
  and latest-stack latency remain unestablished.

Some historical wrapper probes labelled “vanilla” actually hardcoded structured
programs. Round 28 explicitly withdraws those comparisons. The independent
vanilla `@trace`, upstream AD/broadcast, selected-invoke, and native controls
remain useful; do not cite the mislabeled wrapper runs as independent vanilla
proof. The early complex tensor-interface diagnosis was also withdrawn: the
minimal harness had omitted required existing lowering passes.

## Outstanding issues and work to hold back

| Issue | Owner / disposition |
|---|---|
| Identity-hashed mutable-key cloning changes iteration and derivatives | S02 design bug. The final prototype is not generally semantics-preserving here. No fix to submit yet. |
| Symbolic membership, shape, definedness, typed host fields, fresh-parent wrapper dispatch, typed sharded results | Shared representation/capability boundaries, with retained counterexamples. Broader tests and a precise support contract are required. |
| General unrestricted dynamic AD tapes rejected by XLA | Native/backend design issue. Specializing bounds after differentiation for interpreter tests does not establish runtime support. E05/J15 solve an earlier zero-construction failure only. |
| Direct internal `invoke_with_reactant` bridge crashes | Reproduces on round-42 old-cache control and new cache. Affects the experimental R16 bridge; do not call it a proven upstream-main crash. `@noinline` alone did not fix it. |
| Cache concurrency, older-world execution, growth, runtime shadow redispatch, direct CodeInstance invocation | Outstanding R16/R17 contracts. The passing public-invoke subset is insufficient for promotion of the complete bridge/cache design. |
| Expired MLIR location/value printing | Shared lifetime/diagnostics bug, separate from R09. Reduced reproducers exist; no verified fix. |
| Remaining `hlo_call` count, alias-sensitive ABI/function-count assertions, macro rejection expectations, and timing assertions | Keep the original failures visible and classify individually. Some are contract/expectation differences rather than numerical errors; the entire suite is still not green. |
| Reference trace cost and IR growth | S01–S03 integration blocker until remeasured and improved. R19 removed repeated bookkeeping JIT, not the expanded reference program. |
| Persistent compilation artifacts / ObjCache | Not implemented by this run. C01 and R17 concern invalidation/in-process reuse. They do not persist prepared frontend/native code or demonstrate a cold-start win. |
| Multithreaded frontend or improved precompile UX | Not delivered by these correctness experiments. Parallel test/build execution is not frontend multithreading. |

Do not promote failed or superseded controls: global disabling of native-leaf
constant propagation, disabled structurizer promotion, getfield-deduplication
workarounds removed later, `dataflow=true`, one-rule pad/concat fixes, the first
complex-zero helper, or the noinline-only invocation bridge. Their saved files
exist to explain failures, not as pending patches to merge.

## Concrete extraction and integration plan

1. **Prepare independent Reactant PRs against current upstream main.** Start
   with R01, R02, R08, R09, and R05. Carry only their source hunks and independent
   regression files. Add a baseline-fails/candidate-passes check in an environment
   that does not load StructuredReactant. Use the standard native artifact where
   the original evidence already did. Run the affected upstream files and QA.
   Reactant currently declares Julia 1.10 support; this investigation tested
   1.11–1.13, so independent PRs also need a 1.10 compatibility check or an
   explicit version guard. [Current package compatibility](https://github.com/EnzymeAD/Reactant.jl/blob/main/Project.toml).
2. **Prepare the dependency-only PRs.** I01 first, I02 as a separate semantic
   change on top, and C01 independently. Include the library-native tests and
   the Julia version matrix. Check open upstream work before publication.
3. **Submit native optimizer/AD corrections in coherent units.** The early
   independent optimizer patches and small generic Enzyme fixes are the easiest
   starting points. Preserve standalone lit/numerical tests and genuine failing
   controls. Coordinate Enzyme changes with Enzyme-JAX regression coverage when
   necessary. Rebase every patch; do not submit the cumulative native patch.
4. **Extract the larger ordinary-Reactant fixes.** R03/R04/R06/R07/R11 and
   R13–R16 need clean reproducer packaging and targeted ordinary-Reactant runs.
   Split R18 out of AD storage work if its independent implementation remains
   sound. Benchmark R19 separately before advertising it as a latency patch.
5. **Review S01–S03 as a design proposal before integrating them.** Explain
   object identity, alias graphs, loop carries, mutation effects, cache keys,
   and explicit derivative buffers together. Fix retained semantic failures and
   establish a cheap common case. Keep the narrow PRs free of this architecture.
6. **Integrate native artifacts and then establish one combined baseline.**
   Once upstream revisions are available, update the appropriate native pins/
   artifact packaging and the Reactant pass pipeline. Re-resolve a dedicated
   environment; run frontend/acceptance/upstream suites on that exact stack.
   Then measure cold trace, warm trace, MLIR, XLA, package precompile, and program
   execution separately, with GPU testing when available.

The native runtime currently used by the latest prototype has SHA-256
`da4b141c5d3ada07ab8294f6f738168fd9397dcf422f8664ee299f57f2a8cfa7`.
Its [artifact manifest](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-compatibility-39/native-zero-like-v1/artifacts/artifacts.json)
also names the frozen optimizer, interpreter, and FileCheck binaries. Source
changes must be upstreamed and distributed through the normal build/artifact
route; pointing users at this temporary local `.so` is not integration.

## Recovering material from `/tmp`

Preserve the following deliberately, rather than adding every scratch directory
to git:

| Material to retain | Purpose |
|---|---|
| Final source snapshots and their hash manifests | Recover any change not yet represented by a clean patch. Round 41 is the finalized Julia source; round 42 v2 is a separate optional cache experiment. |
| Selected standalone patches, normalized to repository roots | Reviewable implementation units. Prefer final `patches-v8` view/literal files and final v2 native corrections where earlier files are known incomplete. |
| Minimal reproducers, independent numerical references, and passing/failing status records | Establish why each PR is necessary and what it actually fixes. Do not carry whole benchmark/model environments into unrelated bug-fix PRs. |
| Exact build commands, dependency pins, source manifests, and artifact hashes | Reproduce native acceptance and establish which library was actually loaded. Frozen binaries are useful local controls, not source patches. |
| Acceptance requirements that still fail | Prevent a narrower passing suite from redefining the original correctness requirement. Keep separate from the default suite until the support policy is explicit. |

The report audit generated six additional focused patches for R01, R08, R09,
R10, R13, and R14 under
[preserved: structured-upstream-report-2026-09-14/focused-patches](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/focused-patches).
The [audit script](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/audit.py) records how
they were selected and the [audit results](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/structured-upstream-report-2026-09-14/audit.json)
identify all 29 application checks. Application checks are against original file
excerpts or recorded predecessor trees; they are not full package rebuilds.

Patch dependency order is not the same as semantic dependency. For example,
R13 was discovered on an invoke prototype but its source fix does not require
the invoke feature. Conversely, a small-looking `ADShadows.jl` delta genuinely
depends on the new reference machinery. Resolve those boundaries before making
PR branches. Do not blindly apply every `consolidated-*.patch`: those are
snapshots of overlapping, evolving work and may duplicate later changes.

The experiment trees, patches, environments, sources, logs, and frozen native
artifacts were moved out of `/tmp` on 2026-09-14 into
`/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/tmp/`.
All local experiment links in this report point directly to that durable location.
Original `/tmp` names are compatibility symlinks; historical scripts and manifests
retain those names unchanged. After a reboot that clears `/tmp`, run:

```sh
python3 /home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/restore-tmp-links.py
```

See the [archive README](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/README.md)
and its manifest for preservation details. The archive is outside the git checkout.
This protects against temporary-directory cleanup, not disk failure. No upstream
submission or integrated release is implied.
