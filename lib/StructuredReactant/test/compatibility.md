> Preservation update (2026-09-14): the experiment directories now live in the
> [durable archive](/home/anton/dev/jool/structured-reactant-artifacts/2026-09-14/README.md).
> Historical `/tmp` paths below are retained as symlinks. After reboot, the archive
> restoration script recreates them without changing the recorded experiments.

## Round 34: conditional loop results and upstream use tracking

The isolated frontend candidate passes 990 checks across complete Julia 1.13
selections and 309 targeted checks each on Julia 1.11 and 1.12. The selections
include the 402-check frontend suite, 115 unchanged upstream AD assertions in
automatic mode, conditional primal/reverse/HVP under automatic and binomial
checkpointing, observable previous/current values, and iterator/exit guards.
This is selected correctness coverage, not a full Reactant-suite or latency result.

A counted loop may differ between continue and break only in unused post-loop
results. The candidate now proves those results unused with IRStructurizer's use
index, retaining conservative guards for live and unknown uses. A historical
swapped-output classification assertion was corrected after IR inspection:
separate final projections preserve the observable swap. All its numerical checks
passed before the assertion change; the corrected selection passes 72/72.

That analysis exposed an upstream use-index bug: dynamic `:call` callees and
`:new`/`:splatnew` type operands were skipped. An isolated IRStructurizer patch
fixes use queries and replacements. An executable callee-replacement control
fails on current upstream main and passes with the patch. Full dependency suites
pass 1710/1711/1711 assertions on Julia 1.11/1.12/1.13. Main was rechecked at
`496c3d35620d699a3b74937ca6b394bbdc31cee8`, and the patch applies cleanly.
No dependency declaration or main implementation was changed.

Disabled-checkpointing dynamic derivatives still fail. The improved frontend
moves the diagnostic from unknown loop count to invalid dynamic cache constants.
Two native Enzyme-JAX fixes are prepared; the first is still building and its
regression matrix is queued after artifact verification. No native fix is claimed
passing yet. A separate vanilla import proves that unbounded runtime tape shapes
also encounter CPU/XLA's unsupported `stablehlo.dynamic_pad`; static specialization
after differentiation is only a numerical control, not runtime support.

`/tmp/structured-compatibility-34/README.md`, patches, logs, `verify.py`, and
`verification-summary.json` record the exact scope and pending native work.
Twenty protected main implementation paths remain unchanged. No commit, promotion,
publication, precompile workload, or GPU execution was added.

## Round 33: phi cleanup and upstream promotion

The isolated `promoted-v3` candidate passes 933 checks across complete Julia 1.13
selections and 186 targeted checks each on Julia 1.11 and 1.12. These include the
402-check frontend suite, upstream AD, iterator/control-flow regressions, nested
checkpoint derivatives, and 35 new exception-loop primal/reverse/HVP checks.
Broadcast coverage remains 16 upstream assertions plus 52 earlier additions.
This is not a full Reactant-suite pass or a latency result.

Our phi simplification incorrectly removed phis with omitted predecessor edges,
which represent undefined incoming values. The candidate retains those phis and
checks dominance before replacing a phi with an SSA value. An ordinary Julia IR
fixture verifies before cleanup, fails with the old cleanup, and verifies after
the fix. Constructor and loop-shape errors also retain frontend context.

Keep loop promotion in IRStructurizer. Disabling it globally fixed focused
old-release cases but regressed the existing optimized-loop assertion. The final
candidate restores promotion and removes the older frontend getfield-dedup
workaround. Its source is verified against upstream main at `496c3d35620d699a3b74937ca6b394bbdc31cee8`.
Official 0.6.4 still reproduces undefined SSA and a primal iterator timeout; it
lacks the main-branch guards, just like cached 0.6.3. Released dependency
compatibility is not established, and the dependency declaration is unchanged.

The indexed checkpoint selection still has 56 passes and two errors with
checkpointing disabled; none of that partial selection contributes to 933.
A new small native reproducer isolates the failure: cache removal creates an
invalid `tensor<?x2xf64>` constant for a branch that bypasses a loop. Enzyme alone
verifies; `remove-unnecessary-enzyme-ops` fails. This belongs in native cache
materialization, and no native fix is claimed yet.

Sources, logs, bounded dependency controls, native input/pipeline reductions,
verification script, and incremental patch are in `/tmp/structured-compatibility-33`.
The preexisting worktree implementation remains unchanged. No commit, promotion,
publication, dependency change, or precompile workload was added. All owned runs
are terminal; the redundant legacy broad rerun was explicitly stopped and its
incomplete selections remain excluded.

# Broader compatibility and differentiation work

This work targets automatic control-flow tracing across Reactant's supported
programs, including differentiation and checkpointing. Passing the frontend's
own suite is necessary but does not establish compatibility with all of Reactant.

## Current status (2026-09-13)

The latest isolated frontend candidate is round 34's `unused-loop-results-v1`,
based on round 33's `promoted-v3`, round 29 Reactant `bounds-v4`, and round 30's
`native-scoped-seeds-v1`, with the separately tested upstream operand-use fix.
The new native candidates remain under validation. Implementation promotion,
released dependency support, native artifact integration, broader Reactant and
device validation remain open. Later round summaries supersede historical results;
none establishes a full-suite or latency claim.

## Counted-loop eligibility and released dependency checks (round 32)

The isolated change reuses IRStructurizer's DefIndex and accepts Pi-only counted
continuations only after proving same-range, unmodified-state progress. It
preserves distinct final iterator projections while retaining guards for other
unequal exit carries. No dependency or precompile workload was added.

Complete selections pass 893 checks on Julia 1.13 with pinned structurizer main:
frontend 402, iterator protocol 63, historical exit 15, AD 115, broadcast 68,
complex checkpoints 42, nested capture primal/reverse/HVP 48, new counted-path
68 and guard 72. Julia 1.11 and 1.12 each pass 146 targeted checks. Broadcast's
68 comprise 16 upstream checks and 52 earlier regressions; the AD file matches
HEAD. These are selected correctness checks, not full Reactant/device coverage.

The counted-path checks observe the current iterator after exit and exercise
empty, shifted and near-overflow ranges with reverse/HVP derivatives. The
exhausted-exit example folds its output tag to zero, so it does not independently
observe the exhausted private carry. Altered steps, changing ranges, supplied
states, swapped exit carries and early exits retain general lowering. A terminal
side-effect control legitimately counts its loop because structurization places
the effect afterward. Its initial lowering assertion was corrected after an IR
dump; all values/derivatives already passed and no candidate code changed.

Cached IRStructurizer 0.6.3 on Julia 1.12 passes 246 complete checks (protocol,
AD and broadcast), but the full frontend fails structuring `while_stamp!` with
undefined SSA `%3`. Its explicit iterator loop is a WhileOp with mutation in the
condition region, unlike pinned main's LoopOp. The release acceptance stalled
in primal PJRT CPU execution and was interrupted after 555 seconds; it is not
counted as completed. The precise input and responsibility boundary need a
labelled reduced reproducer, including a same-version control for `while_stamp!`.

The two disabled-checkpoint derivative errors on dynamic tape shapes remain;
that partial selection is excluded. Source/library pins and preserved main
implementation paths are verified. See `/tmp/structured-compatibility-32/README.md`,
`counted-v1.patch` and `NEXT.md` for exact scope, corrections and remaining work.

## Range iterator values, state and availability (round 31)

The private iterator carrier now preserves Julia's numeric state and exposes
its logical tuple/nothing alternatives at ordinary call boundaries. Branch and
PiNode facts establish nonemptiness; continue-edge facts apply only to later
loop entries, and static carry facts are checked on updates. Tuple projection
retains the carrier until needed, and tuple identity uses the existing logical
scalar/array comparisons. No Base methods, dependencies, or precompile workloads
were added; four frontend source files change in the isolated candidate.

Availability must be represented separately from the next integer. Julia's
OrdinalRange protocol tests equality with the final state before advancing and
converts the result to the range element type. A supplied state beyond the end
can still produce a pair, while an exhausted iterator's speculative addition
can wrap. The new tests cover both cases, including typemax(Int).

Complete Julia 1.13 selections pass 753 checks: frontend 402, protocol/boundaries
63, explicit iterator exits and gradients 15, upstream automatic AD 115 and
broadcast 68, complex checkpoints 42, and nested capture primal/reverse/HVP 48.
Julia 1.11 and 1.12 each pass 78 targeted protocol/exit checks. An initial 300+15
second frontend timeout during concurrent package rebuilds is excluded; the
unchanged-source isolated rerun passed all 402 checks. These are correctness
results, not latency measurements or a full-suite/released-dependency claim.

Indexed automatic/default and Binomial(3) paths each retain 24 passing checks.
The two disabled-checkpoint derivative errors still report dynamic tape tensor
shapes, so that partial selection is excluded from the complete total.

The explicit iterator-exit test selects the general loop, confirmed by the
`v5-lowering` diagnostic (counted slot 0). It therefore does not prove the
separate counted-exit shortcut. The temporary native-optimizer suspicion was
withdrawn: labelled inputs identified a frontend availability error at n=2;
the standalone native control had checked the already-correct n=4 case.

`/tmp/structured-compatibility-31/README.md`, `verification-summary.json`,
`NEXT.md`, and `iteration-v5-experimental.patch` record the scope and remaining
work. The verifier checks source/library pins, explicit migration modes/reports,
25 candidate sources, 442 predecessor sources and 20 protected implementation
paths. Branch-fact preparation, patch review, cumulative integration and the
broader compatibility backlog remain open.

## Scoped checkpoint replay values (round 30)

A reduced native reproducer isolates the indexed derivative failure to Enzyme
itself, before its cleanup passes. Cloning a checkpoint body used the replay's
popped outside tensor, while its reverse rules still referenced the tensor in
the forward region. Temporarily publishing those mappings during the reverse
visitor, then restoring them, fixes both binomial and statically bounded periodic
replay. The change is in generic Enzyme, with a separate Enzyme-JAX regression;
no Julia source or dataflow setting changes in this round.

Complete Julia 1.13 selections pass 858 checks: frontend/upstream AD/broadcast
768, existing complex checkpointing 42, and nested capture primal/reverse/HVP
48. The new native regression fails on the preceding library and passes native
verification and FileCheck on the candidate. Existing native checkpoint tests
pass 29 of 30 RUN directives; the remaining full-IR golden fails identically on
the baseline, with byte-identical compiler output. It is not counted as passing.

Indexed default and explicit binomial structured paths now each pass 24 checks,
including derivatives, across changing counts and indices. Disabled checkpointing
still fails two derivative stages, and vanilla's default policy also fails two.
These partial runs are excluded from the complete-selection total. This
supersedes round 29's unresolved native-dominance diagnosis below; it does not
prove the explicit final-iterator projection or the full counted-loop extension.

The verifier preserves 442 Julia sources and 20 main-worktree implementation
paths and checks both 266-file native source manifests. Patches apply to the
recorded experimental bases, which include earlier fixes; integration with
current upstream and released artifacts remains open. See round 30 `README.md`,
`verification-summary.json`, and the two standalone patch files.

## Bounds annotations and indexed checkpointing (round 29)

The isolated bounds patch retains prepared statement flags, respects resolved
`:boundscheck` expressions, and places annotated native calls inside Reactant's
tracing wrapper. Julia controls inlining and propagation; an extra bounds context
is not passed between interpreted methods. This fixes explicit `@inbounds`
indexed views while preserving `@noinline`, call-site overrides, and global
bounds settings. No dependency or precompile workload is added.

Complete Julia 1.13 selections pass 971 checks: 768 frontend/upstream AD/broadcast,
174 bounds-context checks across auto/yes/no settings, and 29 indexed-view checks.
Julia 1.11 and 1.12 each pass 129 targeted checks, including existing complex
checkpointing. This selection is different from round 28's; it does not rerun
all prior targeted cases. The verifier checks source/library pins and preserves
442 predecessor sources and 20 main-worktree implementation files. These runs use the archived
IRStructurizer-main checkout; released-version testing remains open.

Plain traced index-array checking still fails once in each frontend, with 20
passing checks and one host scalar-indexing error. Those partial controls are
excluded from the passing-gate count. The explicit inbounds control now passes
all 29 checks with structured tracing, including mutation and reverse gradients.

The new indexed recurrence separates another issue: a loop can continue with
its next iterator but exit with the current one. Experimental v4 preserves that
distinction while recovering counted execution. Its 24 primal checks pass, but
six derivative stages still fail. Automatic/binomial checkpointing reaches an
`enzyme.push` operand-dominance error; disabled checkpointing reaches dynamic
cache-shape limitations. Vanilla passes explicit binomial derivatives but fails
its default-policy derivative stages. These runs are not complete passing gates.

Verified pre-AD inputs and native pipeline dumps are in round 29's
`v4-ad-input-v3`. Both inputs verify; structured then reproduces the invalid
cached-index reference while the vanilla binomial control compiles. The precise
responsible pass remains to be isolated. An explicit iterator/destructuring
probe also exposes a private `Iteration` protocol gap in both v3 and v4, before
AD. The experimental patch remains separate. See round 29 `README.md`,
`NEXT.md`, and `structured-bounds-v3.patch` for evidence and next actions.

## View state, nested AD, and broadcast literals (round 28)

Five isolated patches change six Julia source files. Reactant now keeps SubArray
geometry and cached integer indexing metadata out of tensor tracking, recognizes
already-aligned wrapper shadows, preserves Julia scalar promotion for complex
broadcast literals, and supports conservative view alias checks with
type-preserving copies of reshaped/permuted wrappers. StructuredReactant routes
function-valued Enzyme.Const callbacks through its existing callback boundary,
preserving automatic control flow through nested AD.

The selected-view gradient failure was not introduced by the metadata fix: an
unchanged-source control bypassing the constructor failure also returned zero
gradients. Matching wrapper layouts avoid redundant nested deferred writebacks.
The fix retains overlapping, shared, batched and stacked derivative-storage
checks. Literal tests preserve Complex{Int} dispatch and check ComplexF32/F64
promotion, assignment and derivatives without changing the user's literal.

The combined v8 candidate passes 1,082 checks across complete Julia 1.13 gates:
frontend 402, upstream AD 115 per frontend, upstream broadcast 68 per frontend,
and 314 targeted checks. Julia 1.11 and 1.12 each pass a 110-check automatic
subset for snapshots, literals, view AD and checkpointing. The first full wrapper
run timed out after 300 seconds with 55 passing checks; an unchanged-source
900-second retry passed all 60. Neither run establishes latency or precompile
performance. No new precompile workload is added.

**Historical attribution correction:** the round 19
`wrapper-matrix-values-axes.jl` probe hardcodes structured compilation and ignores
FRONTEND. Earlier vanilla/automatic labels for that exact probe denote duplicate
structured runs, not a vanilla comparison. Those comparison claims are
withdrawn. The final wrapper gate runs once, as structured; the separate upstream
AD and broadcast comparisons use verified mode-aware harnesses.

The runtime-index geometry probe still errors once in each frontend when plain
traced index-array bounds checking attempts host iteration. Its other 20 checks
pass per frontend, including changing row indices on one executable and host-view
input conversion. Those partial runs are excluded from the complete-gate count.
An explicit inbounds control passes with vanilla but still reaches checking with
automatic tracing. A captured call has the inbounds flag in IRStructurizer's IR;
the frontend's prepared representation drops it, and the emitter always returns
true for :boundscheck. Round 29 addresses this frontend gap; the plain checked
index-array path remains open.

The failed v6 conservative-alias candidate exposed Base.unaliascopy's requirement
to preserve wrapper types. V7 fixed the copy protocol, but its optional invoke
fallback for host indices recursively redispatched on a strided-view case. V8
retains conservative checks and the copy fix without that fallback. All failed
candidates and controls remain available for diagnosis. Function-valued Const
routing does not yet establish callable-struct callback coverage.

The reviewable patches and README are in `/tmp/structured-compatibility-28`.
The verifier confirms unchanged predecessor/worktree fingerprints and the pinned
native library. Implementation remains in scratch. Existing broader
counterexamples, integration, released-structurizer and full device coverage stay
open. IRStructurizer remains archived main revision
`496c3d35620d699a3b74937ca6b394bbdc31cee8`.

## Native pad/concat rewrite convergence (round 27)

Round 27 first loaded round 26's complex-component zero fix into the unchanged
round 25 Julia candidate: 74 default-pipeline numerical checks passed. The
batched complex HVP timeout remained. Capturing its exact MLIR and serialized
pipeline reproduced the timeout in the native compiler without Julia.

Rewrite diagnostics found a cycle: `BinopPadToConcat` expands a binary operation
over a pad; concat factoring, slice/concat simplification and `ConcatToPad`
reconstruct the original operation. Both `ConcatPushBinop` and the more general
`ConcatElementwise` can perform the inverse factoring. Guarding only the first
does not terminate the captured pipeline. An earlier unchanged-analysis-attribute
guard also did not fix it; neither failed candidate is treated as validated.

The isolated `native-convergence-v2` candidate makes both factoring rules respect
the existing expansion rule's consumer context: add/multiply chains ending at
slices retain concatenated pieces. Whole-result factoring remains enabled. Only
Enzyme-JAX's `EnzymeHLOOpt.cpp` changes relative to round 26's native source;
the Julia/StructuredReactant candidate is unchanged. A small standalone native
regression hangs before the patch and passes afterward, including a positive
factoring check. It covers real and complex arithmetic and needs neither frontend.

The captured default pipeline now terminates, and its four independent input
cases pass all 20 numerical expectations. Julia 1.13 CPU/PJRT integration with
the new library passes 106 targeted checks, including both frontends' batched
complex HVPs and 42 checkpointed checks. Broader gates pass the 402-check frontend
suite and all 115 upstream differentiation checks in each frontend. No transpose
policy override, disabled optimizer, new frontend special case or new precompile
workload is used.

The first import preflight crashed inside Julia's LLVM code generation during
StructuredReactant's existing precompile workload, before integration tests
started. One unchanged retry succeeded and all checks above ran afterward.
The crash log is retained; its cause is not established and it is not evidence
of stable precompilation. These concurrent correctness runs make no latency claim.

The preference-only environment is
`/tmp/structured-compatibility-27/env-native27-1.13`; it selects the frozen library
in `native-convergence-v2/artifacts`. `verification-summary.json` records source
and artifact fingerprints. `pad-concat-convergence-with-regression.patch` applies
independently to the recorded round 26 source. The patch remains in scratch;
implementation promotion, explicit-region view metadata, prior broader
counterexamples, released-structurizer coverage and full Reactant/device
validation remain open. IRStructurizer is the archived main revision
`496c3d35620d699a3b74937ca6b394bbdc31cee8`, not a newly fetched main.

## Complex differentiation and native component zeros (round 26)

The Julia candidate remains round 25's `batch-lowering-v1` with its native-v5
library. Round 26 adds an independently verified Enzyme-JAX fix in
`/tmp/structured-compatibility-26/native-v2`; the Julia matrix below does not yet
validate that new native library. Its patch, source/artifact fingerprints,
commands and outcomes are recorded in the round 26 directory.

The four complex product failures previously attributed to Enzyme's builtin
tensor interface were caused by an incomplete native test pipeline. Reactant's
per-function Enzyme postpasses raise the temporary complex operations correctly.
All four cases pass on the unchanged round 25 binary with those postpasses:
eight numerical expectations. The same corrected pipeline still fails all four
on the round 24 binary, separately supporting the product-rule fixes. The
generic tensor-interface diagnosis is withdrawn; the original logs are retained.

A new primitive matrix found a narrower bug: `real` and `imag` reverse rules
construct zero components with primal shape, losing the batch dimension.
Enzyme-JAX now derives those zeros from the cotangent. The isolated v2 binary
passes 48 primitive modules and four product modules, totaling 56 numerical
expectations. The patch applies to both recorded round 24 and round 25 native
sources. No complex arithmetic or derivative machinery is added to the frontend.

On Julia 1.13 with the unchanged round 25 library, both frontends pass 32 new
default-pipeline checks for complex losses, product gradients, ordinary second
derivatives, batched forward derivatives and mutation. Automatic tracing also
passes four selected-view checks and 42 checkpointed complex gradient/HVP checks
with runtime counts zero through six and three policies. The corrected vanilla
`@trace` view variant fails while tracing SubArray's typed integer metadata.
This is a comparison with vanilla `@trace`, not glou-nes's current frontend.

Batched complex HVP compilation times out after 900 seconds in both frontends
under the default optimizer, in the post transpose/reshape propagation phase.
Keeping AD enabled and setting both propagation directions to `:down` passes
eight numerical checks per frontend. This diagnostic does not fix the default
pipeline. Disabling optimization entirely leaves Enzyme operations at XLA export
and is not counted as a successful AD control. Two initial fixture mistakes,
scalar indexing to construct zero and expecting an `@trace if` expression to
return its branch value, were corrected and their original failures preserved.

The native optimization timeout, explicit-region view metadata, native/Julia
integration, prior broad compatibility counterexamples, released-structurizer
coverage and full Reactant/device validation remain open. These CPU checks make
no latency or precompile claim; existing worktree implementation is preserved.

## Product derivatives and batch legalization (round 25)

The combined implementation remains isolated in
`/tmp/structured-compatibility-25/batch-lowering-v1`. StructuredReactant and the
archived IRStructurizer main dependency are unchanged from round 24. Two new
standalone acceptance files, `product_derivatives.jl` and
`product_checkpointing.jl`, retain the numerical requirements in the worktree;
they require the combined candidate and are not included in the regular runner.

Enzyme-JAX now handles batched strong-zero/extrema zeros with the correct shape,
preserves derivative lanes in reductions, and sums contributions to an active
initial value. Static product gradients use products of the other elements,
without division; their derivatives remain correct at multiple zero factors.
Forward sum/product differentiation and padding derivatives now support active
initial/padding values and independent lanes. Numerical padding checks include
interior padding, cropping, empty input, and empty output.

Batched forward-over-reverse products also exposed a missing Reactant lowering
step: Enzyme's temporary extract/concatenate operations survived into XLA export.
Reactant now schedules the existing `enzyme-batch-to-stablehlo` pass in its
shared final lowering phase. The conversion remains in Enzyme-JAX. No derivative
or batch-conversion machinery is copied into StructuredReactant.

The round 25 native runner passed 239 numerical expectations across 126 modules,
including the prior dot/broadcast/select regressions, and reported four complex
product failures. Round 26 corrects that runner's missing per-function Enzyme
postpasses and withdraws the generic tensor-interface diagnosis: the four cases
pass on the same binary with Reactant's actual pipeline. The original verifier
failures and historical counts remain recorded. Neither result establishes
complete complex differentiation support.

The final combined Julia 1.13 candidate passes upstream autodiff 115 through each
frontend, the full frontend suite 402, prior checkpoint policies 36, and wrappers
60 through each frontend. Product derivatives pass 26 through each frontend;
checkpointed gradients and Hessian-vector products pass 42 checks. The exact new
worktree files also pass those checks against the combined candidate.

Julia 1.11 and 1.12 each pass the targeted 68 product/checkpoint checks. These
include executable reuse at nonzero, single-zero, multiple-zero and all-zero
inputs, and runtime loop counts zero through six under three checkpoint policies.
The references multiply the remaining factors directly and do not reuse the AD
implementation. This is a versioned AD subset, not full Reactant coverage on
every Julia release.

Separate Reactant and Enzyme-JAX source patches pass apply/whitespace checks
against the recorded round 24 predecessors. Source and binary hashes, historical
failures, corrected test pipelines, commands and terminal statuses are retained
in the round artifact directory. Existing worktree implementation files were
preserved. Complex tensor operations, dynamic-shape zero-safe products, previous
symbolic/dictionary/sharding counterexamples, implementation/artifact integration,
released-structurizer compatibility and full Reactant/device validation remain
open. No latency or precompile-performance claim is made from these CPU runs.

## Explicit derivative storage and native batching (round 24)

The implementation remains isolated in
`/tmp/structured-compatibility-24/explicit-shadows-v7`. The existing worktree
implementation is preserved. This round changes Reactant's AD integration and
Enzyme-JAX's HLO derivative rules; StructuredReactant's source is unchanged from
round 23 and the same archived IRStructurizer main dependency is selected.

Supplied derivative buffers now preserve aliases through selected arrays and
views, including distinct overlapping views of one backing array. All logical
views initialize their shadows before flattening deduplicates their shared
parent. Separate batch lanes retain independent temporary storage even when
stacked directions share one user matrix. Nonzero reverse seeds are applied once
to physical storage, and writeback preserves untouched portions of user buffers.

Forward differentiation of composite results now retains inactive result fields
without declaring nonexistent derivative outputs. Ordinary and batched tangents
match native Enzyme, including a Boolean field changing between executable calls.

Native fixes cover scalar-predicate selection, reverse transposes, broadcasts,
sum reductions and dot_general with multiple derivative lanes. The dot rule was
the remaining failure in the selected-array mutation test: optimization rewrites
sum of squares into a contraction. The fix preserves the lane in contraction
batch dimensions, result types and output permutations. All 33 native numerical
expectations pass, including derivatives of both operands for five contraction
layouts at widths one and two.

Final Julia 1.13 gates pass: upstream autodiff 115 through each frontend; regular
frontend 402; checkpoint policies 36; wrappers 60 through each frontend. Four new
standalone acceptance files pass 108 checks together: shared shadows (14),
distinct overlapping views (10), forward/reverse shadow modes (28), and inactive
composite results (56). The preceding reference AD/annotation files pass 11 and
8 checks. These files are not added to the regular runner and require the
combined candidate environment for the new behavior.

The final combined candidate also passes 84 targeted checks on each of Julia
1.11 and 1.12: distinct overlapping views (10), inactive composite returns (56),
stacked forward directions (4), stacked reverse mutation (6), and tuple-batched
reverse mutation (8). The same cases pass on Julia 1.13. This is a cross-version
AD subset, not the full Reactant or frontend suite on every version.

The exported Julia and native patches pass apply/whitespace checks. Earlier
failures, invalid fixtures, stale-binary attempts and intermediate regressions
remain recorded in the round README. The final tests are numerical checks; no
failed case was replaced with a skip. Implementation/native artifact integration,
full Reactant coverage and device validation remain open. This CPU-only work
makes no latency or precompile-performance claim.

## Integration boundaries and nested differentiation (round 23)

The combined candidate is `/tmp/structured-compatibility-23/shadow-bindings-v5`.
Use `/tmp/structured-compatibility-23/native-seed-env-<version>` for the rebuilt
CPU library; the candidate's original environments retain the preceding library
for comparison. Implementation remains isolated. The two new acceptance files
are standalone and are not included in `runtests.jl`.

A twelve-file upstream CPU matrix caught shared regressions in the prior candidate.
Compared with the frozen working-tree baseline, cloning every subtrace's mutable
arguments added loop results and lost writes to Const mutable AD state. The v1
correction confines isolation and detached-state export to cached `Ops.call`.

The remaining nested forward-over-reverse failure exposed three AD interfaces:
zero derivative buffers must preserve backing storage; integer/Boolean reference
metadata is inactive; and type inspection must use existing bindings rather than
emit reference materialization in the caller's region. The final Julia correction
is in Reactant's AD integration. It does not copy AD infrastructure into the frontend.

The resulting IR then exposed a separate Enzyme canonicalizer bug. Removing a zero
return seed retained the old activity flag requiring that seed. A two-line fix was
built and verified independently: the preserved baseline fails a minimal reproducer,
while the new compiler and StableHLO interpreter pass four numerical checks.

| Final CPU gate | Result |
|---|---:|
| Upstream `core/autodiff`, each frontend, Julia 1.13 | 115 pass |
| Frontend suite, Julia 1.13 | 402 pass |
| Reference-backed nested/selected/mutating AD, Julia 1.11/1.12/1.13 | 11 pass each |
| Cached-object acceptance, Julia 1.13 | 118 pass |
| Wrapper value/gradient reports, Julia 1.13 | 60/60 |
| Checkpoint-policy reports, Julia 1.13 | 36/36 |

Full upstream control-flow testing has no errors: vanilla records 146 passes and
one mismatch; automatic records 142 passes and five mismatches. The shared mismatch
is an exact function-count assertion affected by alias-aware cached-call variants.
The four automatic-only mismatches expect macro rejection of return/break/continue.
The assertions remain unchanged. A separate exact-zero timing assertion in the earlier
tracing matrix also fails on the frozen working-tree baseline. These files do not
establish full Reactant-suite coverage.

`reference_autodiff.jl` retains the eleven passing numerical checks. The separate
`reference_ad_annotations.jl` currently errors in both testsets before its eight
intended assertions: a selected reference-backed primal still cannot use an explicit
dense forward direction or reverse seed. Mapping user-supplied derivative storage
is the next AD task. This is not complete reference/alias compatibility.

All earlier container, sharding, wrapper and dynamic-state boundaries remain in
scope, as do released-structurizer compatibility, integration and device validation.
No GPU, latency, or precompile-performance claim is made. The initial v3 precompile
timeout and checkpoint-probe old-library assertion are retained separately from
actual test failures; the corrected policy probe passes all 36 reports.

Commands, logs, source hashes, native build evidence, combined/incremental patches,
and remaining work are recorded in `/tmp/structured-compatibility-23/README.md`,
`verification-summary.json`, `native-integration/README-zero-seed.md`, and `NEXT.md`.

## Result graphs and native host iteration (round 22)

The latest isolated candidate is `/tmp/structured-compatibility-22/host-dictionary-v6`.
The core patch preserves unchanged dictionary table layout, constructs returned
mutable keys through the same memo as returned values, maps output paths to
concretized keys, and allocates mutable result objects before initializing their
indexed fields. This fixes returned-key aliases/freshness, cycles in either root
order, and leading undefined mutable fields. Low-level field stores retain
runtime field-type checks.

The broad integration check also exposed a frontend failure: an `IdDict` whose
`Any` storage contains tracers sent iteration into Julia's C-backed table accessor.
The final candidate keeps Base's public `IdDict` iterator native. Traced loop
bodies and user-defined iterators still enter structured tracing; no dictionary
implementation or foreign-call machinery is copied into the frontend.

| CPU gate | Julia 1.11 | Julia 1.12 | Julia 1.13 |
|---|---:|---:|---:|
| Frontend suite | 400 (v5) | 402 (v5) | 402 (v6) |
| Cached-object acceptance | 118 (v5) | 118 (v5) | 118 (v5) |
| Executable result graphs, each frontend | 49 (v5) | 49 (v5) | 49 (v5) |
| Host iterator values/forward/reverse derivatives | 12 (v6) | 12 (v6) | 12 (v6) |
| Unchanged Optimisers tests, automatic frontend | 5 (v6) | 5 (v6) | 5 (v6) |

The v5 and v6 core source hashes are identical; v6 changes only the frontend's
native iterator policy. The original iterator regression records six passes and
one compile error on v5. Its v6 runs check both signs through the same compiled
functions and compare derivatives with finite differences. The Julia 1.11/1.12
Optimisers runs initially lacked the dependency; only the corrected offline
environments establish their five passes.

On Julia 1.13, unchanged custom-struct tests pass 9 with the existing 1 broken,
and StructArrays passes 14, through both frontends. Automatic tracing routes
six and seven compilation requests respectively. The final v6 run also passes
all five dictionary branch reports and 36 checkpoint-policy reports. The v5
core candidate passes 148 host-array assertions, 36 core-call assertions and
60 wrapper primal/gradient reports on 1.13; its core-call and 16 dictionary
cache/gradient reports also pass on 1.11 and 1.12.

The stronger `dictionary_boundaries.jl` gate intends 282 assertions and still
fails for mutable-key iteration. All assertions execute, with no expected-broken
annotations or cache disabling. Its v5 results are:

| Julia / frontend | Pass | Fail | Error |
|---|---:|---:|---:|
| 1.11 / structured | 242 | 40 | 0 |
| 1.12 / structured | 228 | 54 | 0 |
| 1.13 / structured | 232 | 50 | 0 |
| 1.13 / vanilla | 228 | 54 | 0 |

Those failure counts vary with object/hash layout; a lower count is not evidence
of an additional fix. The passing portions cover unchanged integer-key layout,
canonical equal-key replacement, actual cache reuse, and executable result aliases.

The two-logical-device CPU/PJRT probe on v4 passes plain tensor and loosely typed
container returns, but typed key/value containers fail on the one-buffer versus
two-buffer wrapper type. Both frontends reproduce it. This is a retained sharded
result-type limitation, not a full sharding-suite result.

Implementation promotion, native-artifact integration, released-IRStructurizer
compatibility and full Reactant/device validation remain open, alongside the
previously recorded symbolic container-state and wrapper boundaries. The native
library and archived IRStructurizer main dependency were not changed this round.
The three new acceptance files remain standalone, outside `runtests.jl`.
These concurrent CPU checks make no latency or precompile-performance claim.

Exact commands, logs, source/test/native hashes and terminal statuses are in
`/tmp/structured-compatibility-22/verification-summary.json`. The core and frontend
patches are separate: `dictionary-result-graphs.patch` and
`native-host-dictionary-iteration.patch`; `round22.patch` combines them relative
to round 21's `container-state-v8`.

## Dictionary and host-container state (round 21)

The isolated `/tmp/structured-compatibility-21/container-state-v8` candidate fixes
round 20's six cached-object acceptance failures: numeric keys now distinguish
concrete types, and dictionary tracing/restoration preserves fresh identities,
key/value aliases, cycles, and entry mutation. These are Reactant core changes;
CompilerCaching's code cache does not own the object graph passed to a traced call.

Branch captures now include host mutable state that can acquire traced values.
Dictionary entries and host-array elements use logical state recipes rather than
Julia's dictionary or array storage internals. Supported cases include rebinding,
shared elements, cyclic containers, selected fresh containers, and array
initialization/resizing when both branches agree on shape and definedness.
Unsupported symbolic membership, shape, definedness or typed-field state is not
claimed as implemented.

StructuredReactant keeps Base scalar reads of ordinary host Arrays native.
A new custom-array counterexample confirms that user indexing still enters the
frontend, but exposed a shared core path bug afterward: Reactant interpreted a
custom array's field path as an element index. The final candidate makes its
getter/setter agree with the tracer's struct traversal. No Julia indexing
algorithm, structurizer or cache implementation is copied into the frontend.

| CPU gate | Julia 1.11 | Julia 1.12 | Julia 1.13 |
|---|---:|---:|---:|
| Frontend suite | 400 | 402 | 402 |
| Existing cached-object gate | 118 | 118 | 118 |
| Host-container acceptance | 148 | 148 | 148 |
| Core cached-call gate | 36 | 36 | 36 |
| Dictionary cache/gradient reports | 16/16 | 16/16 | 16/16 |
| Dictionary branch reports | 5/5 | 5/5 | — |
| Wrapper primal/gradient reports | — | — | 60/60 |
| Checkpoint-policy reports | — | — | 36/36 |

A dash denotes no recorded run for that gate/version on this candidate. The
1.13 dictionary cache/gradient column uses vanilla; the 1.11/1.12 columns use
StructuredReactant. The preceding v7 candidate passed those 16 reports on both
frontends and all five dictionary branch reports on 1.11/1.12.

The host-container gate has 148 assertions covering two input sets, repeated
execution of both branches, mutation and identity, matrix/Cartesian/unsigned
indexing, custom indexing containing traced control flow, and nonlinear
finite-difference gradients. `host_array_state.jl` and `dictionary_state.jl` remain
standalone gates outside `runtests.jl`; the implementation remains isolated.

The final standalone dictionary gate includes the preceding 118-test object gate, then records 124 passes, 14 failures and 1 error in its own tests. It intends 146 dictionary assertions; conditional membership prevents eight assertions from executing. The key-iteration failure count varies with object/hash layout, so a single passing iteration probe is insufficient.

Mutable-key dictionary iteration remains a wrong-result bug: an observational
cached call can choose a different first key than its caller immediately before
the call. Conditional dictionary membership is explicitly rejected. These remain
ordinary failures, with no expected-broken annotations or cache disabling.
Typed host fields, fresh-parent wrapper dispatch, runtime Periodic loop counts,
immutable/mutable cycle boundaries and broader integration also remain open.

The source uses unmodified IRStructurizer main at
`496c3d35620d699a3b74937ca6b394bbdc31cee8` and the unchanged round-17 tensor-activity
native library. Commands, source/test/native hashes, the patch relative to round
20, terminal statuses and exact failing checks are in
`/tmp/structured-compatibility-21/verification-summary.json` and `README.md`.
Released-structurizer compatibility, implementation/native-artifact integration,
full Reactant/device coverage and controlled latency measurements remain open.
These concurrent CPU checks establish no speed or precompile-performance claim.

## Cached object identity and effects (round 20)

The isolated `/tmp/structured-compatibility-20/object-cache-v4` candidate fixes
cached mutable-object and host-vector identity, result freshness, field/element
rebinding, and updates to input storage or objects that a callee subsequently
detaches. Complete caller/callee object correspondence is retained separately
from the existing tensor argument map. Tensor results, field state and returned
objects share one restoration memo. Tracking-only traversal preserves cycles
without allocating replacement object identities; host-vector cache keys now
encode aliasing, shape and initialized positions.

This is Reactant core call-cache/tracing work. It does not change CompilerCaching,
IRStructurizer, StructuredReactant's interpreter, or the native library. The
implementation remains isolated; `cached_objects.jl` is a standalone acceptance
gate outside `runtests.jl`, with 118 intended assertions per frontend.

On Julia 1.13, that gate records 112 passes and six ordinary failures through
each frontend. The failures reduce to two shared cache bugs: numeric values of
different concrete types inside `Any` fields can collide in the key, and fresh
dictionaries still reuse cached identity. All assertions execute; none are
expected-broken skips. Passing cases check actual cache reuse, caller aliases,
fresh allocations, cycles, undefined fields, vector resizing, mutation without
returned values, disconnected state, nonlinear finite-difference gradients, and
12 gradient comparisons across three checkpoint policies and four runtime counts.

The same candidate passes the 402-check frontend suite, 246 aliasing checks,
36 core call-cache checks, all 60 wrapper primal/gradient checks, and the existing
36-case checkpoint-policy matrix on Julia 1.13. The earlier mutable-object gate
improves from 60 to 64 passes; its conditional typed-host-field compile error
still prevents four assertions. The wrapper run initially selected no cases due
to a harness filter; only the corrected unfiltered run establishes the 60 passes.

Final cross-version validation passes 400/402/402 frontend checks on Julia
1.11/1.12/1.13. Each version passes 14 object reports, 10 vector reports, eight
extended mutation/checkpoint reports, and 36 core call-cache assertions. All
runners have terminated; the six new acceptance failures remain open.

Exact cross-version outcomes, commands, logs, source hashes, and the patch relative
to round 19 are in `/tmp/structured-compatibility-20/verification-summary.json`.
`NEXT.md` records the remaining numeric-key and dictionary work. These concurrent
CPU correctness checks make no latency or precompile-performance claim. Native
integration, released-structurizer compatibility, broader reference semantics and
full Reactant/device validation remain open.

## Mutable fields and branch effects (rounds 18–19)

The standalone `mutable_objects.jl` acceptance gate checks field rebinding,
retained aliases, shared and cyclic objects, cached object identity, and nonlinear
gradients. It has 68 intended assertions. The isolated round-19-v6 implementation
passes 60; four cached-static-object identity assertions fail, and one typed-host-
field compilation error prevents four assertions from executing (65 reported
entries). The gate remains separate from `runtests.jl` and retains ordinary
failures rather than expected-broken skips.

A branch previously copied array changes back through post-branch field paths.
After a field was rebound to a fresh array, that could overwrite the old array
in the caller and corrupt gradients. The prototype now records captured storage
and field bindings separately. Each changed storage is joined once at the if;
field recipes preserve the selected reference. It also terminates cyclic mutable
object traversal while preserving repeated paths through shared objects.

The nonlinear rebinding control improves from four passes and four wrong results
to eight passes. A separate six-check shared-cycle primal/gradient case passes.
The direct-Reactant and mutable-struct comparison improves from 10/20 to 18/20;
the remaining two failures are identity queries in a vanilla caller that observe
the physical tracer carrier. Typed `Ref{Int}` field updates now report the need
for symbolic field state instead of silently applying both branches' effects.

The final isolated candidate passes 400/402/402 frontend checks on Julia
1.11/1.12/1.13. All three versions pass 60 wrapper, 36 checkpoint-policy,
eight nonlinear-rebinding and six shared-cycle checks. Julia 1.13 additionally
passes 246 aliasing, 50 loop-composition, 44 selected-object and 24 semantic
checks. The expanded identity gate retains 168 passes and three
compilation errors: fresh-parent wrapper dispatch and runtime Periodic loop
counts still prevent 18 of its 186 intended assertions from executing. The
cross-version matrix is recorded in `/tmp/structured-compatibility-19/README.md`
and `verification-summary.json`; only terminal successful processes count as
validated there.

An intermediate effect representation retained unused branch outputs and two
snapshots for each changed storage. Its final offset-array gradient hit
900-second deadlines on Julia 1.11 and 1.13. A live stack was in native MLIR
greedy pattern rewriting. Removing redundant outputs restores completion of the
Julia 1.13 wrapper gate, and joining storage once further reduces the simple
mutation probe from 20 to 11 raw StableHLO operations. These concurrent CPU runs
are correctness and IR evidence, not controlled latency benchmarks. The captured
slow native input is retained for diagnosis; no optimizer defect is inferred
from that single stack.

The source still uses unmodified IRStructurizer main and round 17's frozen
native library. No implementation is promoted. Cached-call object mappings,
symbolic typed fields, mutable loop state, reference cost, released structurizer
support, native integration, full Reactant integration and device coverage remain
open. `NEXT.md` in the round-19 directory records the next cache counterexample
and the required distinction between captured and newly allocated objects.

## Loop wrapper state and tensor activity (rounds 16–17)

The original 90-check array identity gate now passes in the isolated frontend.
An expanded gate in `array_identity.jl` contains 186 intended assertions: 144 pass
and seven compilation errors prevent 42 assertions from executing. The remaining
cases require structural state for wrappers around fresh parents, and mutable
helper exports for selected broadcasts such as `a.^2`. The gate remains outside
`runtests.jl`; those failures are ordinary assertions, not expected-broken skips.

Flattening every wrapper into a dense loop carrier regressed parent and axis
semantics. The current experiment consults joined loop argument types and preserves
a common wrapper structure. Incoming-value and cached identity queries then pass,
while unsupported fresh-parent dispatch is diagnosed explicitly. The representation
and its cost remain under investigation; no implementation is promoted.

This exposed a separate native gradient error, reduced to two `stablehlo.if`
operations and a `stablehlo.select`, without Julia, arrays, or loops. Enzyme's
default activity analysis recognized scalar integers but missed integer tensors,
including `tensor<i1>` predicates. A small isolated fix classifies tensor element
types. The native reproducer's four numerical assertions pass afterward, as do the
previous 32-case adjoint/checkpoint matrix and existing native if tests.

With the fixed native library, the frontend suites pass 400/402/402 checks on
Julia 1.11/1.12/1.13. All three versions pass the 60 wrapper and 36 policy checks.
Julia 1.13 additionally passes 50 loop-composition checks, 24 semantic checks,
246 expanded aliasing checks, and 44 selected-object checks. The multiplication
form of the identity-dependent loop loss also agrees with finite differences;
the power form retains its independent mutable-helper export failure. Cross-version
and additional acceptance results are recorded in `/tmp/structured-compatibility-17`.

An alternative `dataflow=true` control compiles but returns incorrect zero gradients,
including on the small native reproducer. It is not enabled. Neither the native
artifact nor the representation prototype has been integrated upstream. Released
structurizer support, reference cost, general mutable objects, full integration,
and device coverage remain open. These CPU runs are correctness evidence, not
latency measurements. See round 17's `README.md`, `verification-summary.json`,
`enzyme-tensor-activity.patch`, and `if_predicate_activity.mlir` for the evidence.

## Selected object semantics and native adjoints (round 15)

The isolated frontend now preserves branch-selected views, types, callable
values, and immutable containers through call and constructor boundaries. The
new `selected_objects.jl` is a standalone acceptance gate, outside `runtests.jl`:
44 assertions check dispatch, metadata, mutation, and finite-difference gradients.
The earlier identity helper incorrectly converted every concrete scalar to Bool;
it now preserves the scalar element type.

The round-14 prototype passes 12/24 combined semantic checks; the current v7
experiment passes 24/24 and the 402-check frontend suite. This initially exposed
incorrect gradients through longer loops despite correct primal results.
An independent native MLIR reproducer returns derivative 180 instead of 100,
without Julia or StructuredReactant. The stablehlo.if reverse rule failed to
clear consumed result adjoints, allowing nested conditional adjoints to survive
another reverse loop iteration. The isolated native fix clears each active
result before propagating it; both the small reproducer and a 32-assertion
native matrix pass afterward.

Changing only the native library, with frontend/core sources fixed, takes the
expanded policy/loop-gradient control from 14/36 to 36/36. The fixed build also
passes 50 loop-composition checks and 60 wrapper checks on Julia 1.13. The 36
policy checks, all 24 semantic cases, and all 60 wrapper checks pass on Julia
1.11 and 1.12 as well. Their full frontend suites pass 400 and 402 assertions,
respectively; Julia 1.13 passes 402. The additional Julia 1.13 complex-projection
probe passes 10 checks. All runners for this round are terminal.

This is correctness evidence under concurrent CPU load, not a latency result.
Neither the representation prototype nor the native artifact is promoted.
The corrected identity gate still fails to compile loop-selected immutable
wrapper identity (84 assertions pass; six are not reached). Root-result/cached
representation boundaries, mutable object semantics, reference cost, released
structurizer support, native integration, and full integration/device coverage
remain open. A separate checked_srem_int limitation, runtime Periodic case, and
minimal-pipeline select derivative failure are retained in the diagnostic record.

Source snapshots, original failures, controlled reports, and the clean native
patch are in `/tmp/structured-compatibility-15`; see its `README.md` and
`verification-summary.json`. Check all live job handles before rerunning.

## Logical identity and representation limits (round 14)

The next isolated prototype reduces loop storage state to one selected backing
ID, coordinate map, and conjugation flag per binding. On Julia 1.13 it preserves
the recorded mutation, wrapper, and differentiation cases: frontend 402,
alias acceptance 238, loop composition 50, wrappers 60, complex gradients 10,
view metadata 20, and empty views 10.

A serial comparison against the previous path-order prototype shows no speed
win. Each workload/variant uses two fresh CPU/PJRT processes in alternating
order, with five warm retraces per process and the same timed tracing boundary
as round 13. Two-array warm tracing changes from 20.47 to 22.38 ms; four arrays
from 110.86 to 119.76 ms. Allocations decrease only from 9.73/60.86 MB to
9.62/59.98 MB, while raw StableHLO operations increase from 5,612/38,821 to
5,808/39,429. Compact carried fields alone do not remove expansion of possible
reference paths into gathers and conditional updates. These comparisons are
between experimental reference implementations, not against vanilla Reactant.

A separate logical identity implementation gives ordinary allocations fresh
identities, preserves them across region argument cloning, and carries selected
identities through loops. Cached calls map returned arguments to caller
identities while freshening newly allocated results. Aligning reference and
identity traversal order fixes a cached-loop call whose operands otherwise
mismatched its inferred signature. That revision passes the old expanded alias
gate (238+8), 50 loop-composition, 60 wrapper, and 10 complex-gradient checks on
Julia 1.11, 1.12, and 1.13. The first extended identity gate records 66 passes
and one explicit error on each version. It includes retained copies, swaps,
nested loops, cached calls, and 18 identity-dependent gradient checks under auto,
periodic, and binomial checkpointing. The rejected loop-view function prevents
six intended assertions from executing.

Further helper tests exposed another wrong result: a branch-selected view's
identity compares correctly directly, but incorrectly through ordinary and
cached helpers. Saved inferred IR shows the helper becoming `return false` for
the physical dense-tracer/SubArray signature. The third isolated revision keeps
array identity comparisons symbolic through the custom interpreter's builtin
inference hooks. The helper retains its comparison and both reproductions pass
on Julia 1.11, 1.12, and 1.13. The regular frontend suites pass 400/402/402 checks;
the 246-check expanded alias gate and the 50/60/10 loop/wrapper/complex probes
also pass per version. The final 90-check identity/dispatch acceptance source
records 82 passes, two failed assertions, and one compilation error on each
version. The compilation error prevents six intended assertions from executing,
so the reported test total is 85, not 90.

The representation remains incomplete in two distinct ways:

- Identity of a loop-selected immutable wrapper needs its structural fields,
  beyond the opaque mutable-object representative. The prototype explicitly
  rejects the current loop-view identity query while retaining wrapper mutation.
- A selected view can dispatch to the general array method when an inference
  barrier hides its type. The program should select the `SubArray` method. This
  produces a wrong result even with the identity-inference correction; preserving
  `===` alone does not preserve arbitrary method dispatch.

The new standalone [array_identity.jl](array_identity.jl) gate retains these
requirements as regular assertions, outside `runtests.jl`. Its 90 intended
numerical checks include the two helper boundaries, dynamic dispatch, loop
identity, and checkpointed gradients. It must pass together with the mutation
and performance gates before promoting the reference representation. No
reference implementation or path-ordering correction was promoted this round.

Sources, commands, raw IR, source fingerprints, benchmarks, and child exit codes
are in `/tmp/structured-compatibility-14`; `README.md` and
`verification-summary.json` distinguish revision results and remaining failures.
All execution uses the same frozen experimental native library and unmodified
IRStructurizer main at `496c3d3`. This does not validate released artifacts or
non-CPU devices, nor close the other requirements recorded below.

## Version coverage and reference cost (round 13)

The frozen round-12 reference prototype now passes the complete recorded corpus
on Julia 1.11, 1.12 and 1.13, with the same native library. Frontend counts are
400/402/402; each version also passes the 238-check mutation/alias corpus,
50 loop-composition checks, 10 complex-gradient checks, 20 view-metadata
controls, 60 wrapper checks, and 10 empty-view checks. This establishes the
recorded cases across versions, not complete Julia reference semantics.

A serial CPU/PJRT cost comparison exposes a significant prototype regression.
The baseline is the current StructuredReactant working tree without the
reference architecture. Both variants use the same fixed native library and
Julia 1.13. Each workload/variant runs in two fresh processes, with alternating
order and five retraces per process. Packages are precompiled beforehand.
The timed boundary is `TracedUtils.make_mlir_fn`; package loading, context/cache
setup, MLIR printing, optimization and XLA compilation are excluded. Executables
are separately checked against Julia at changed input values, including that
rebinding leaves caller arrays unchanged.

| Probe (256-element Float64 arrays) | Baseline warm trace | Reference prototype | With path-ordering correction |
|---|---:|---:|---:|
| Elementwise expression | 0.87 ms | 0.94 ms | not remeasured |
| Two independent loop-carried arrays | 2.54 ms | 261 ms | 20.4 ms |
| Four independent loop-carried arrays | 3.52 ms | 636 ms | 108 ms |

Scoped compilation logs identify repeated `unique!` set widening and tuple-sort
specializations for paths of varying lengths. An isolated shared Reactant
correction uses `Set{Tuple}` explicitly and compares path elements without
specializing on complete tuple shapes. Both `if_condition` and `case` use it.
Median warm Julia compilation drops from 239/522 ms to about 0.09/0.14 ms in
the two/four-array probes. The correction passes the full version corpus above
and the existing four `case` checks in both frontends on each version. Older
case-test environments initially lacked SpecialFunctions; repaired, separately
resolved offline environments preserve those initial load failures in the audit.

This correction leaves the raw program size unchanged: the two/four-array
probes emit 5,612/38,821 StableHLO operations versus 94/136 in the baseline.
After the correction, warm allocations remain about 9.7/60.9 MB versus
0.65/1.09 MB. One first precompilation attempt crashed in Julia's LLVM backend
while compiling broadcast code; an identical-source retry succeeded. That
intermittent failure has not been reduced or claimed fixed.

A further semantic check demonstrates why the older 238-check gate is
insufficient. `y = c ? x : copy(x); y === x` returns false even when the selected
array is `x`; the same happens after loop rebindings. Three of six predicate
checks fail. These functions and assertions are now part of the explicit
`aliasing_acceptance.jl` gate, outside the default frontend suite. Mutation and
array identity are separate observable requirements. The full expanded run retains all 238 prior passes and records five passing
identity assertions (including compilation) and three failures. The optimized
prototype has not passed this expanded gate.

No reference or path-ordering implementation was promoted this round. The
remaining work includes compact reference state, observable identity, the
precompilation failure, native artifact integration, and the other open
compatibility requirements. All commands, raw IR, measurements, frozen package
sources, statuses and the isolated `path-order.patch` are in
`/tmp/structured-compatibility-13`; `verification-summary.json` records the
verified results and failures.

## Range indexing, wrapper metadata, and empty tensors (round 12)

A shared range-indexing fix is now in the working tree. Materializing a
`TracedUnitRange` for array/range selectors used its possibly already traced
endpoint type as the tensor element type. It now uses `unwrapped_eltype(v)`.
The new `test/core/range_indexing.jl` checks selectors, output shapes, reuse with
changed endpoints, and reverse derivatives. All 66 assertions pass with vanilla
and automatic frontends on Julia 1.11, 1.12, and 1.13 (396 total).
Full ranges (9), indexing (181), and wrapped arrays (33) pass per frontend on
1.13 with the standard native artifact. The production fix and test match the
validated isolated files byte for byte.

Full traced-view tests time out in both frontends with the standard native
artifact. An unchanged-source control reproduces the hang; a reduced native
stack identifies the previously diagnosed `ReshapeDynamicSlice` failed-match
rewrite loop. With the previously fixed native library, all 312 view checks
pass per frontend. These are separate native-artifact results, not a claim that
the standard artifact passes those tests.

The reference prototype remains isolated. OffsetArrays opt into reference
projection in their extension; SubArray projections translate offset parent
indices into the coordinate tensor's one-based indexing. Branch normalization
now removes residual reference state after materialization, and an extension
hook restores static array axes after selecting the reference. This prevents
correct values from hiding lost metadata. Six wrapper cases pass 60 checks of
values, axes, caller-visible mutations, and reverse finite differences at zero,
one, and multiple iterations. They include views of offset, permuted, and
strided parents, whole offset vectors and matrices, and scalar offset indexing.
A fixed-length `unsafe_view` control now passes; ordinary checked dynamic-range
construction remains unresolved. Different branch axes and arbitrary wrapper
metadata are not established as supported.

Empty views exposed a separate native optimizer bug: `ZeroExtentTensorCanon`
replaced empty StableHLO results and operands with `tensor.empty`, which XLA
cannot import. The isolated native fix uses empty StableHLO constants and skips
already canonical constants. Four reduced tensor cases pass structural checks;
a nine-check interpreter test also covers nonempty results of mixed-result
branches and empty operands. The established native optimizer/checkpointing
regression suite passes. With identical Julia sources and only the native
library changed, vanilla's two empty-array primal compilations change from XLA
export errors to passes; all ten value/mutation/gradient assertions then pass.

The final combined CPU/PJRT prototype passes 402 frontend checks, 238 alias
acceptance checks, 50 loop-composition checks, 10 complex-gradient checks,
20 view-metadata controls, the 60 wrapper checks above, and 10 empty-view
value/gradient checks. Each process asserts the actual loaded native-library
path. Source snapshots, hashes, environments, commands, baseline failures,
terminal statuses, and standalone upstream patches are preserved in
`/tmp/structured-compatibility-12`; `verification-summary.json` records the
counts. These concurrent correctness runs make no latency claim.

Only the small range fix and its regression have been promoted this round.
The reference architecture, its cross-version/performance validation, native
artifact integration, and the other open compatibility requirements remain
work in progress.

## Constant ownership and projected references (round 11)

A second shared Reactant cache bug is fixed in the working tree. `Ops.constant`
previously cached and returned a mutable `TracedRArray`, so mutating one request
also changed another request with identical values, including later requests.
The cache now stores the immutable MLIR SSA value and returns a fresh array
handle. This preserves constant deduplication without sharing mutable bindings.
The standalone vanilla reproducer failed four of eight numerical assertions
before the fix. `test/core/constant_aliases.jl` covers requests before and after
mutation, three array shapes, executable reuse, and reverse derivatives.

The new 36-check file passes under vanilla and automatic frontends on Julia
1.11, 1.12, and 1.13 (216 checks total), using the standard native artifact.
Full autodiff passes 115 checks per frontend. Full ops retains exactly its
round-10 results: 370 passes and two LU errors for vanilla; 369 passes, the same
two errors, and the HLO-count mismatch for automatic. Existing constant-dedup
checks pass. The production Julia sources match the tested isolated checkout;
this fix adds no dependency or precompile workload.

The larger reference prototype remains isolated. Projection maps preserve views
of loop-carried storage, and conjugation state distinguishes complex adjoints
from transposes. Its frozen v3 source passes the original loop-composition audit
(50/50), complex primal and real/imaginary gradient checks (10/10), alias
acceptance (238), and the regular frontend suite (402). With the final native fix below, all of those suites pass again; five additional
view cases pass 50 value/gradient checks, and rebinding a variable that initially
holds a view passes ten more. Dynamic-range view construction remains failing. The prototype currently opts
in specific storage wrappers; it has not established arbitrary-wrapper support,
cross-version correctness, or acceptable compilation/runtime cost.

A case carrying views of fresh parent allocations exposed two distinct issues.
Generic carry reconstruction promoted host metadata inside array wrappers,
causing a native Julia trap on both the prior prototype and baseline source.
The isolated frontend now uses Adapt's storage protocol to carry wrapper parents
while retaining their metadata. That removes the trap, but the default compile
then fails MLIR verification. Raw-module verification and a native-only replay
establish that the emitted IR is valid: the `slice_if` optimization introduces
the invalid dependency. It creates a replacement if at the later slice, after
users of other results (or inside a nested region). Three reduced MLIR modules
reproduce this with the standalone optimizer. The local native fix retains the
original if insertion point. The rebuilt optimizer passes all three structural
regressions, their 16 numerical branch checks, the previously failing saved
module, and the existing native regression suite. The original fresh-parent
case now passes all ten value/gradient checks for 0, 1, 2, 3 and 6 iterations.
The final CPU environment's actual loaded native-library path is recorded.

A diagnostic control supplying a fixed length and using `unsafe_view` still
fails: indexing `TracedUnitRange{TracedRNumber{Int64}}` constructs a tensor with
an already wrapped element type. This is a separate shared indexing bug, not
an alias result mismatch. The original checked-view bounds failure and unknown
length handling remain open; the control does not justify removing checks.

Private ownership of loop selector/projection handles was also added in isolation;
it did not fix that native failure. Neither the reference prototype nor its
patched native library has been promoted. Artifacts, preserved revisions,
commands, source fingerprints, and native reductions are in
`/tmp/structured-compatibility-11`. These CPU correctness checks make no latency
or precompile-performance claim.

## Call-cache alias keys and reference-state experiments (round 10)

The working tree now includes a shared Reactant call-cache correction in
`src/Tracing.jl`, with `test/core/call_aliases.jl`. Callee argument flattening
emits one MLIR argument per distinct tracer, but the old key recorded only tensor
types. A call using `(x, y)` could be reused for `(x, x)` and produce an invalid
argument count. Traversal-local identities now encode array and scalar alias
patterns. This concerns the per-trace MLIR callee cache, not CompilerCaching's
CodeInstance cache or persistence across processes.

The new regression passes 36 checks on each of Julia 1.11, 1.12 and 1.13 under
both vanilla and automatic frontends (216 total). It checks both call orders,
several inputs per executable, scalar arguments, caller-visible mutations, and
reverse gradients against finite differences. The unchanged source fails both
call orders. Final validation uses the standard native artifact `57eb4f169d`,
with actual loaded-library paths recorded. Full autodiff passes 115 checks in
each frontend. The tested production source matches the working tree exactly.

The full operation file is not green: vanilla has 370 passes and two LU
reconstruction stack overflows; automatic has 369 passes, the same two errors,
and an HLO-call-count assertion failure. Targeted unchanged-source controls
reproduce all three failures with the same artifact. They remain required work;
they are not classified as regressions introduced by the cache-key change.

Separately, the isolated reference implementation now checks that counted-loop
continue and break projections agree and keeps caller-observable backing arrays
in explicit loop state. Swapped variables and retained prior allocations match
Julia, including input mutations and reverse gradients. The extended loop audit
improves from round 9's 34/50 to 48/50. Its remaining two wrong results concern a
view selected in a loop and read after its parent is mutated.

The combined isolated implementation also preserves aliases across cached calls,
including returned arguments. Its 12 call-boundary checks, 238 alias-acceptance
assertions, and 402 frontend assertions pass. The loop composition audit still
fails those two view cases. These larger reference changes remain outside the
branch and use the patched native library plus isolated general-loop checkpoint
integration. Cross-version, broader composition, and latency validation remain
required before promotion. The branch's separate alias-acceptance gate therefore
remains failing despite the improved isolated results.

Commands, failed revisions, final patches, source fingerprints, reports and
unchanged-source controls are in `/tmp/structured-compatibility-10`; round 9's
records are preserved in `/tmp/structured-compatibility-9`. Concurrent correctness
runs do not establish compilation latency or precompile-performance results.

## Earlier validated source and remaining integration work

The branch recognizes updatable stdlibs by package metadata, preserves explicit
native execution during inference and emission, and keeps pure host helpers
with unreachable error returns out of line. The last policy prevents host
`sqrt(size(x, 1))` error paths from becoming traced loop-exit discriminators.
The native-call regressions pass 27 checks, including the expected DomainError
behavior. No numerical-function-specific frontend handling was added.

Automatic upstream integration files pass: random (50), linear algebra (222),
FFT (179), and scatter/gather (1,275). Those runs predate the latest host-policy
change. A Reactant `Ops.fill` fix now preserves tuple-known array rank across
its vector-based implementation, resolving the three-matrix kernel without
an iterator adapter. Its targeted 34 checks, type-number checks (60 per mode),
and array-operation checks (402 per mode) pass in vanilla and automatic mode.

The isolated CPU native library includes fixes for disjoint update/slice
simplification, reduction batching, and concatenating reshaped slices. The
last fix also checks alignment of adjacent strided slices. Reduced numerical
regressions fail on the relevant pre-fix libraries and pass after each fix.
The final combined default-optimized Polybench matrix passes all 29 cases
with valid host references, across both shapes and both input sets (116
numerical executions). This includes `2mm`, `doitgen`, `correlation`, `3mm`,
and Gram-Schmidt. The published `durbin` case remains an invalid reference.

Gram-Schmidt's remaining timeout was isolated to `reshape_dynamic_slice`.
A failed match inserted a constant, repeatedly restarting greedy rewriting.
Both that rewrite and `reshape_dus` now delay IR creation until all checks
succeed. Six small cases verify declined and successful rewrites numerically,
with a rewrite limit making the regression fail promptly on the old binary.
All prior native regressions and existing dynamic-slice rewrite tests pass.
The full Gram-Schmidt rerun passes all four numerical executions. `durbin`
accesses index zero in ordinary Julia and is not counted as a pass.

The general-loop normalization prototype passes native structure, numerical,
and effect-rejection tests. In an isolated Reactant pipeline, reverse binomial
checkpointing passes general convergence loops, mutation, and nested counted
loops; a vanilla annotated convergence loop passes too. Early-return reverse
differentiation now passes with a separate shared-Enzyme fix: the branch
helper erases gradient reads and writes after consuming them into SSA values.
The reduced nested-branch numerical regression and eight existing Enzyme
branch/cache/checkpoint tests pass, as do the original five Julia acceptance
cases (17 checks). A further 135 primal, forward-, and reverse-mode checks
pass for break,
mutation in conditions, nested general loops, branching bodies, and nested
returns, under three strategies and three inputs per executable. Their
gradient oracle uses central differences in Float64. These native
patches and the normalization pipeline remain isolated; they are not yet
integrated into the default package artifact or automatic policy. The full
frontend suite also passes all 378 checks with the later shared-Enzyme fix
and normalization pipeline on Julia 1.13.

An isolated automatic-policy candidate enables binomial checkpointing for
both runtime-counted loops and general conditions. It passes 180 expanded
primal/derivative checks, including `:auto`, and all 378 checks in the full
frontend suite on Julia 1.13.
The two-line integration patch is recorded separately in
`/tmp/structured-compatibility-5/automatic-general/automatic-general-pipeline.patch`.
The branch's default policy has not been changed before native-artifact
integration.

Commands, immutable sources, machine-readable results, and logs are in
`/tmp/structured-compatibility-5` and earlier numbered artifact directories.
Native sources, frozen libraries, and standalone upstream patches are in
`/tmp/structured-compatibility-2/native`. No project-wide native preference,
production dependency, or main-repository commit was added. These concurrent
correctness checks do not establish latency results.

## Broader integration round (2026-09-12)

A subsequent CPU round uses the isolated general-loop normalization pipeline,
automatic general-loop checkpoint policy, and shared Enzyme/native fixes
reported above. Both frontends pass all six convolution sections (585 checks
and six existing broken checks per mode), activations (86), NNlib (56), Lux
(4), Flux (1), random (50), FFT (179), linear algebra (222), and scatter (1,275).

Additional full-file checks with assertions on the loaded package paths pass
in both modes: KernelAbstractions (15), PythonCall/TensorFlow export (6),
Enzyme-JAX integration (17), NumPyro (2), and Zygote (3, on Julia 1.11).
Sharding passes 93 checks on PJRT and 109 on IFRT, with communication
optimization passing 54 on each runtime. Two-rank MPI passes 590 checks per
frontend. These are CPU checks; architecture-gated GPU and callback cases
are not thereby verified.

The full automatic differentiation file initially had 114 passes and one
error on both PJRT and IFRT: nested forward-over-reverse AD creates a view
with a traced loop index, which Base's view constructor did not accept.
Reactant now represents such an index by a rank-zero traced array, preserves
rank when copying the view, and writes through the parent's transformation
protocol. This also fixes mutation through transposed views. One-dimensional
view reindexing follows parent indices directly, and strided dynamic reads
select gather independently of index order. These changes are in Reactant;
no view-specific behavior was added to the frontend.

The new view file passes **312 checks per frontend on Julia 1.11, 1.12, and
1.13**, including real/complex arrays, nested and strided views, reshape,
transpose, adjoint, overlap, copy independence, and executable reuse. Sparse
views retain Reactant's existing unsupported-type diagnostic through the
SparseArrays extension. The full differentiation file now passes all 115
checks in both modes on PJRT and IFRT; PJRT also passes on Julia 1.11/1.12.
Existing indexing (181) and broadcast (68) checks pass in both modes on 1.13.
The full frontend suite passes 378 checks and the separate native checkpoint
acceptance passes 167 with the array implementation.

QA passes 15 checks with one existing broken check. It required qualifying
`Base.TupleOrBottom`, resolving the sparse-copy intersection, and selecting
the owning module for `kron`/`kron!` across Julia versions. A constant module
alias avoids static import checks interpreting an inactive import branch.
The full linear-algebra file passes 222 checks per frontend on all three
Julia versions. The automatic control-flow file retains 143 passes and four
annotation-contract assertions that differ when annotations are removed;
vanilla passes 147.

The array changes and QA corrections are promoted to the worktree. Exact
source hashes and final frozen-source reports are under `view-index/` in the
artifact directory below. The final direct worktree run passes all six jobs:
312 view checks and 115 AD checks per frontend, four printing checks, and
378 frontend checks. This run uses the ordinary frontend policy and compiler
pipeline, with the fixed CPU library selected only in its isolated environment.

The unmodified native-library control passes the complete AD file in both
modes, but both new view suites reach the 20-minute limit in greedy rewriting
on a column update. A direct update without views reproduces it:
`x[:, i] = x[:, i] .* 2 .+ 1`. Bounded native replay isolates the existing
`reshape_dynamic_slice` failed-match bug: the old library fails at 100
rewrites, while the already patched optimizer succeeds. Thus the full new
view matrix currently requires the isolated native fix described above.
No optimization is disabled in the regression tests or frontend.

The investigation also found an outdated Julia wrapper for MLIR's IR-printing
API. The wrapper now supplies and disposes default printing flags and an
empty output-directory argument. A focused test verifies four properties,
including actual before/after dumps. Separately, formatting an exception that
contains an expired MLIR location can still crash; that lifetime/diagnostic
problem remains open.

The first extended environments accidentally resolved older local source paths
from stale Project `[sources]` entries. Those results are retained but excluded
from the claims above. Corrected environments pin both Project and Manifest
paths, and `migration.jl` now checks expected loaded paths and records actual
package paths. The unaffected convolution/integration matrix did not mutate
its environment. Python export jobs also use separate temporary directories,
since the upstream tests use fixed export filenames. Raw reports, provenance,
terminal statuses, and the verified aggregate are in
`/tmp/structured-compatibility-6`.

## Further compatibility and failure recovery (2026-09-12)

A further frozen-source CPU matrix runs 20 upstream files across vanilla,
structured with annotations retained, and automatic mode (58 jobs, because AD's
two other modes were already covered). 52 jobs pass. Retaining annotations passes
all 147 control-flow checks and all 218 tracing checks; automatic mode differs
on five assertions expecting macro rejection. Vanilla's tracing file also trips
an exact-zero compilation-time assertion at 0.96 microseconds. The remaining
three failures are the same OffsetArray view-write mismatch, one in each mode.
Full compile (109), batching (13), aliasing (7), and wrapped-array (33) files pass
in all three modes, as do the selected callback, optimizer, struct-array,
factorization, and BLAS files. These are correctness results under concurrent
load, not latency measurements.

The view changes are extended in Reactant to preserve coordinates across
materialization. Traced parents lose their offset axes; host parents can retain
them, so indexing translates between the actual source and destination axes.
Broadcast allocation delegates to Base.similar, which lets OffsetArrays retain
its axes through its existing allocation protocol. Offset materialization and
writeback recurse through the parent, as do transpose and adjoint writeback.
The corrected full OffsetArrays file passes **78 checks in each of the three
modes on Julia 1.11, 1.12, and 1.13**, covering real/complex values, nested views,
transpose/adjoint, executable reuse, and host-only view materialization.

The ordinary full QA run passes 15 checks and its existing broken documentation
check. Loading OffsetArrays explicitly exposes stale imports and a non-owner
ReactantCore qualification; these are corrected, and all five import checks pass
with the extension loaded. No checks were disabled. The first older-Julia test
environments lacked OffsetArrays and NNlib, so those load failures are excluded.
Corrected environments resolve both dependencies offline and verify Project and
Manifest source paths. A host-view comparison initially used Array(...), which
rejects offset axes; the final test compares the arrays directly, including axes.

A capability check now recognizes unmodified IRStructurizer main at `496c3d3`
even though its Project.toml still reports 0.6.4. The artificial 0.6.5-DEV version
change is no longer needed. This does not fix the limitations of the tagged
0.6.4 source. The frontend suite adds 24 checks for errors inside branches,
loops, nested regions, and scalar-indexing scopes, verifying state restoration
and successful subsequent compilation with both positive and negative inputs.
The final cross-version gate passes **400/402/402 frontend checks on Julia
1.11/1.12/1.13**, including the host-axis refinement. Broadcast and traced-view
regressions also pass. `verification-summary.json` in the artifact directory
below ties the completed reports to source hashes and records the superseded
test-harness failures separately.

### Open: conditional array aliases and their derivatives

A separate numerical audit finds a remaining wrong-result bug, including in
vanilla Reactant with an explicit @trace. This program returns 3 instead of 30
for x=[1,2] and a traced false condition:

```julia
function conditional_alias(x, c)
    y = c ? copy(x) : x
    x .*= 10
    return sum(y)
end
```

The merge retains tensor values but loses the conditional sharing between y
and x. A zero-iteration runtime loop exposes the same problem after a later
mutation. Checking only the loop's immediate outputs missed it. The strict
20-check audit has eight matches and 12 mismatches across primal and
finite-difference-checked reverse results, including tuples, non-inlined calls,
views, and loops. This is an open requirement of the goal, not a supported
case or a passing compatibility result.

An isolated continuation experiment fixes the simple conditional and its
derivative, but fails loop mutation propagation, misses indirect aliases, and
fails the full frontend suite. It has not been promoted. Fixing the shared
region/tracer alias semantics remains necessary.

### Open: displaying expired MLIR handles

A failed compile can leave an MLIR location in a MethodError. Normal showerror
succeeds, but the raw display used by Julia Test dereferences the destroyed
context and segfaults in mlirLocationPrint. A separate experiment also reaches
mlirValuePrint on an expired value. Small CPU reproducers now isolate these
failures; no frontend-specific printing workaround was added.

The frozen sources, commands, statuses, complete reports, rejected experiment,
and independently reviewable patches for this round are under
`/tmp/structured-compatibility-7`. Its README explains which later reports
supersede environment/test-harness failures. The native library and general-loop
checkpoint integration gaps described above remain unchanged.

## Conditional alias representation and acceptance (2026-09-12)

A shared Reactant prototype gives selected array results conditional references
to their backing storage, so later reads and writes can follow the selected
identity. It fixes all 20 checks in the original audit. The broader 48-check
variant initially passes 46; memoizing shared fresh allocations resolves the
remaining two. Direct vanilla `@trace` tests, without loading StructuredReactant,
improve from six of 12 to all 12 primal/reverse checks. This supports fixing the
shared tracing layer, but does not prove that this implementation is sufficient.

The retained standalone `aliasing_acceptance.jl` covers 17 patterns, each on
positive and negative inputs and both traced Boolean values. It compares return
values and mutated input arrays to Julia, and reverse gradients to central
finite differences. Each compiled executable is reused across those values.
It remains a required failing acceptance gate, separate from `runtests.jl`.

| Source | Passing assertions | Failing assertions | Not reached after compilation failure |
|---|---:|---:|---:|
| Unchanged round-7 baseline | 135 | 87 | 16 |
| Isolated reference prototype | 199 | 15 | 24 |

The complete gate would execute 238 assertions. Twelve of the 17 patterns pass
all checks in the prototype. The remaining cases expose missing explicit alias
state in nested regions, alias changes during runtime-loop iterations, and
sharing between overlapping views of a fresh parent. A nested reference can
retain a child-region predicate; MLIR rejects that use outside the region.
The existing frontend suite also regresses on `for_break` (130 passes and one
error in its initial testset; later files are not reached). No reference-field,
property-interception, or merge implementation from this experiment is promoted.

The design requirement is now clearer: distinguish variable rebinding from
storage mutation, clone reference relationships consistently for a subtrace,
and pass both backing storage and alias selectors through explicit region
inputs/results. A loop needs a stable representation of the possible reference
relationships in its carried state, including zero iterations. Independent
materialized view results cannot substitute for their shared parent. Any final
implementation also needs latency/precompile validation; the prototype has no
performance claim.

A control case additionally exposes the existing default-pipeline gap for reverse
AD of a counted loop with an early break: cache removal cannot determine its trip
count. A separate current-source snapshot with the existing general-loop
normalization and automatic binomial-checkpoint policy passes all 14 checks for
that case, including mutated inputs and finite-difference reverse gradients.
It contains no alias prototype. The native integration work remains necessary.

Commands, source hashes, all terminal statuses, rejected prototype revisions,
the standalone acceptance snapshot, and the reviewable prototype patch are under
`/tmp/structured-compatibility-8`. Its `verification-summary.json` includes the
checks not reached after compilation failures, rather than treating them as
passing or reducing the acceptance requirements.

## Native checkpointing acceptance

`native_checkpointing.jl` is a standalone acceptance suite for the unreleased
native work. All 167 checks pass. It runs the ordinary differentiation cases,
the original reverse
checkpointing cases, and the expanded finite-difference cases. It is separate
from `runtests.jl` because the standard Reactant artifact does not yet provide
the normalization pass or shared Enzyme fix. Run it in the isolated environment:

```sh
JULIA_DEPOT_PATH="$(cat /tmp/structured-baseline-1.13/depot-path.txt)" \
  julia +1.13 --project=/tmp/structured-compatibility-5/gradient-cleanup/env \
  lib/StructuredReactant/test/native_checkpointing.jl
```

That environment selects a frozen CPU library and prepends
`normalize-enzyme-hlo-loops` to the Enzyme pipeline. The default frontend policy
still only enables automatic binomial checkpointing for runtime-counted loops;
automatic policy/integration for general conditions remains required work.

## Requirements and evidence

- Run Reactant's upstream tests unchanged, comparing vanilla, structured with
  annotations retained, and structured with bare control-flow annotations
  removed. Classify numerical failures, unsupported cases, native failures,
  missing dependencies, and tests that require unavailable hardware separately.
- Check forward and reverse differentiation of branches, counted and
  data-dependent loops, early exits, mutation, nested loops, and callbacks.
  Reuse executables at zero, one, and multiple iterations, checking derivatives
  against independent analytic or numerical references.
- Support Reactant's checkpoint strategies at the frontend entry point and
  propagate them through nested calls without global mutable policy. Establish
  an automatic policy for runtime-counted loops. General data-dependent loops
  remain part of the objective even where the current native implementation
  cannot yet checkpoint them.
- Preserve observable array identity (`===`) as well as shared mutations,
  including identity predicates used in control flow.
- Preserve explicit annotations and options. Keep structurization and compiler
  caching in their upstream packages; native autodiff and optimizer fixes belong
  in the corresponding native implementation.
- Verify diagnostics and state restoration after failed traces, invalidation,
  world-age behavior, and Julia 1.11–1.13 compatibility after implementation
  changes. Measure latency separately after correctness is established.

## Initial results

CPU/PJRT, Julia 1.13.0, IRStructurizer main at `496c3d3` (local version label
`0.6.5-DEV`), with the condition-mutation and simultaneous-carry fixes applied.

| Upstream file | Vanilla | Structured, annotations kept | Automatic |
|---|---:|---:|---:|
| `core/control_flow.jl` | 147 pass | 147 pass | native optimizer failure |
| `core/autodiff.jl` | 115 pass | 115 pass | 115 pass |

The automatic control-flow process was sampled inside `ConcatReshapeSlice`
before terminating with a native segmentation fault. A subsequent run without
sampling reached `loop_batched` and timed out after unrolling more than 256
emitting iterations. These are incomplete runs, not passes; the original native
abort has not been independently reduced.

A separate fixed-source matrix covered 26 additional core files in all three
modes. After correcting the harness to preserve compiled thunks and supply the
upstream worker imports, the totals were:

| Mode | Passing checks | Remaining assertions/errors |
|---|---:|---|
| Vanilla | 2393 | two LU-factorization stack overflows |
| Structured, annotations kept | 2392 | same two errors; one HLO text assertion |
| Automatic | 2389 | same issues; two loop-presence assertions; one alias rejection assertion |

The HLO text assertion counts integer additions introduced by loop counters as
if they were duplicate copies of the imported floating-point addition. A reduced
IR probe confirms that the imported function appears once. The loop-presence
assertions expose mutation-only loops being unrolled. The alias rejection
assertion required a numerical probe: the previous frontend
returned the wrong second array after mutation through one of two initial
aliases. That bug is fixed below. The shared LU errors also reproduce in vanilla.
This matrix excludes `core/control_flow.jl` and `core/autodiff.jl` (reported
above), GPU callbacks, QA, sharding, and the integration/NN/Python suites;
those remaining surfaces are not claimed as verified.

The initial checkpoint probe also exposed an upstream limitation:
`AutoDiffWhileRev::needsBinomialCheckpointing` requires a canonical counted
loop, with the induction variable in the first position. A general condition
is not sufficient. A standalone vanilla `@trace checkpointing=Binomial(4)`
loop controlled by `sum(x)` also fails during differentiation. Consequently,
the previous claim that general data-dependent loops merely need checkpoint
hints was too broad. Supporting these loops remains required work.

Logs, isolated source snapshots, test environments, subprocess statuses, and
native reproducers are in `/tmp/structured-compatibility`. This is an ongoing
correctness audit, not a latency benchmark or a completion claim.

## Changes and follow-up

`structured(f; checkpointing=...)` now carries a task-scoped strategy through
callbacks and keyword entry points. Nested explicit wrappers preserve their
own strategy rather than being inlined into the enclosing scope. The default
uses `Binomial(4)` for runtime-counted loops; static loops retain ordinary
caching. Tests cover analytic derivatives of a nonlinear recurrence with zero,
one, and multiple iterations, mutation, static strategies, nested overrides,
and restoration after failed compilation.

Loop entry aliases are checked before array carries are copied. Host-controlled
iterations run until a carry separates from an independently captured alias;
rolling an unresolved captured alias is rejected instead of silently changing
its identity. The upstream mutate-then-reassign example now matches Julia at
zero, one, and eight iterations, across two input sets. More general alias
relationships and data-dependent transitions still require coverage and work.

The complete frontend suite passes 279 checks with IRStructurizer main on
Julia 1.13. It also passes 277 checks on Julia 1.12 and 1.13 with released
structurizers, and 275 on Julia 1.11, each with the two existing structurizer
skips. Commands, logs, and source fingerprints are in the artifact directory.
No precompile workload was added. Latency has not been benchmarked for this
change.

Priority follow-up remains:

1. Stop unnecessary unrolling of mutation-only loops without moving host table
   construction or host collections onto the device.
2. Implement and verify general-condition checkpointing in the native autodiff
   layer; retain the vanilla and structured reproducers as acceptance tests.
3. Extend the matrix through integration libraries, NN, exceptions/failure
   recovery, and available accelerator configurations. Add differentiation
   cases for nested loops, early exits, condition mutation, and higher order.
4. Resolve or characterize the shared LU errors, remaining alias patterns, and
   the earlier native slice optimizer failure; then measure latency again.

## Mutation loops and expanded checks

The frontend now notices mutation of captured traced arrays when deciding
whether to roll a loop. It retains the first host iteration and rolls the
remaining counted, while, or break-based iterations. Captured host mutable
containers keep this mutation-only path native. Shape-derived host tables
continue to run on the host; a separately reduced mixed host Ref/device update
also matches Julia. A false host while condition returns immediately rather
than creating an empty region.

On Julia 1.13 with IRStructurizer main, the complete frontend suite passes 317
checks, including 37 new mutation-loop assertions. A subsequently added
15-check differentiation file passes independently: forward mode for general
conditions, mutation, nested counted loops, and early returns, and reverse mode
for nested counted loops. Each executable is reused across three inputs with
independent chain-rule references. General-condition reverse mode, including
mutation and early return, still fails native cache removal despite an explicit
Binomial strategy; those acceptance failures remain open.

Upstream automatic batching now passes all 13 checks. The complete automatic
control-flow file finishes with 143 passes and four differing assertions that
expect the macro to reject return, break, or continue. The migration harness
preserves the bare macro's statement result, including a terminal if returning
nothing. Its earlier identity replacement changed that contract and triggered a
failure-printer crash when an MLIR-backed diagnostic was displayed. The five
terminal-if checks now pass. A separate value-returning conditional remains a
safe-diagnostics probe; correcting the harness is not evidence that arbitrary
mixed return types are supported.

The broader fixed-frontend CPU matrix covers 19 integration and six NN files in
three modes. It is still running. Completed results identify structured-only
failures in FFT structurization, RNG seeding, constant-operand linear algebra,
and activation broadcasting. The convolution file exceeded the per-file time
limit in all modes; it remains unverified. No full integration pass is claimed.
The frontend snapshot predates the mutation-loop change; final affected-file
rechecks must use the final source and harness.

Released IRStructurizer 0.6.4 rejects the new while-mutation probe with an
undefined-SSA validation error before emission; the same probe passes on main.
The Julia 1.12 run timed out, and the Julia 1.11 run is still in progress. Thus
cross-version validation of this round is incomplete. The new failing tests
have not been changed into skips. This needs an upstream-version comparison and
validation against main on the supported Julia versions.

The native disjoint-update/slice fix is drafted in an isolated checkout of the
pinned Enzyme-JAX source, with eight additional lit cases covering disjointness,
clamped starts, and strided reads. It has not yet been compiled or validated.
An isolated CPU build of the unmodified native baseline is running first; no
project preferences, native artifacts, or precompile workload were changed.

Commands, snapshots, status files, logs, and the native patch are in
`/tmp/structured-compatibility-2`. The native build and integration matrix must
be polled through their actual process/tool handles before restarting anything.
These results are correctness evidence under concurrent load, not latency data.

The conditional-return probe reduced to `set_mlir_data!(::MemoryRef{Nothing},
::MLIR.Value)` in the region builder. Branch results mixing nothing and a traced
value are now rejected by the frontend before reaching that call. Four focused
checks verify the printable diagnostic and successful compilation/execution of
both branches afterward. The host Ref case and these diagnostics have been
added to the regular suite.

The released-structurizer version round is not a passing gate: Julia 1.13 and
1.11 reject the new while probe during structurization, and Julia 1.12 times out
in checkpointing. The 1.11 runner also correctly refused to attest source
fingerprints after the diagnostic edit changed the worktree. A fresh,
sequential Julia 1.11–1.13 run now uses an immutable frontend snapshot and
IRStructurizer main in separate environments. Inspect its final statuses before
making any cross-version claim. This does not resolve compatibility with the
released structurizer.

## Native call boundaries and broadcast inference

The next round recognizes Reactant/Enzyme package extensions as native methods
using Julia's extension ownership metadata. Native keyword calls now retain the
callee while wrapping actual callbacks, including callbacks passed by keyword.
Previously a keyword callee could be repeatedly wrapped and cause a stack
overflow. The complete upstream automatic FFT file now passes 179 checks.
A generic overlaid keyword-call regression passes four primal checks, and an
additional reverse-mode probe passes both branch derivatives.

The host-Ref test now distinguishes a compiler-eliminated local Ref from a Ref
observed by an out-of-line mutating call. Julia 1.12 can scalar-replace the first
one, so requiring an unrolled loop there was too strict. Both versions retain
numerical checks, and the observable-Ref case requires host execution. The
expanded mutation file passes all 52 checks on Julia 1.13. The frozen frontend
suite with these changes passes 355 checks on 1.13; its 1.12/1.11 checks are
still running. This snapshot predates the overlay-inference change below.

A separately reduced broadcast failure also occurs with vanilla Reactant:
inference returns `Union{Bool,TracedRNumber{Bool}}`, which Base joins to `Number`
before tensor allocation. Reactant now preserves the union through fused
broadcasts and recognizes alternatives with the same primitive tensor element
type. No frontend-specific Broadcasted override is used. The standalone native
regression passes 52 checks spanning host/traced Bool and Float32 results,
fused and zero-dimensional broadcasts, executable reuse, and empty-shape IR.
The upstream broadcast file with these tests passes 68 checks in both vanilla
and automatic modes. Activations and NNlib also pass in both modes (NNlib: 56
checks), with the combined native and frontend fixes.

The captured-constant matrix/vector failures reduced to a root-lookup mismatch:
`CC._return_type` uses the ordinary method table even when given a custom
interpreter. Reactant's overlay returns a traced tensor, but the ordinary
multiplication method infers an array of traced scalars. Frontend return-type
queries now select through the interpreter's method table before inference.
All 12 focused native-boundary checks pass, including both branch derivatives
and ordinary host execution. The reduced captured matrix/vector cases now infer
the same result types and produce the same values as vanilla. A new frozen
full-suite/version/integration run validates this latest change separately;
its completion is still required.

The earlier frozen integration recheck still has RNG seeding errors, RNG stack
overflows, and constant-operand type assertions. Scatter-mean failures from the
previous round also remain open. These
are not claimed as fixed by the passing FFT/broadcast checks. General-condition
reverse-mode checkpointing and the default native Polybench optimizer failures
remain required work. The unmodified native baseline build remains active;
the drafted optimizer patch has not yet been compiled or run.

Frozen source trees, environments, full-file reports, reduced probes, and
launchers for this round are in `/tmp/structured-compatibility-3`. None of these
concurrent CPU runs establishes latency or precompile-performance results.


## Additional CPU coverage and dictionary cache validity (rounds 35–36)

These results use experimental source copies. No implementation was promoted,
committed, or staged; no GPU or latency claim is involved.

Round35 rechecked the latest frontend/IRStructurizer stack: eight whole migration
files pass 149 assertions per frontend. All 29 valid original Polybench kernels
pass four executions each; original Durbin retains its invalid host reference.
A separately corrected Durbin changes three loop bounds and passes 75 compiled
primal/reverse/HVP checks per frontend, backed by a pinned C reference and an
independent dense-system derivative oracle. Commands, hashes, and verification:
`/tmp/structured-compatibility-35/README.md` and `verify-results.py`.

Round36 found another shared Reactant call-cache correctness bug. Equal ordered
Dict entries can have different capacities or tombstone histories, producing
different iteration after an identical insertion. TracedToTypes omitted that
state. An isolated patch records copied table metadata at the shared cache
boundary. Both frontends' baseline fails 24 of 46 numerical/reuse assertions;
the candidate passes all 46 on Julia1.13. The tests include primal, reverse and
Hessian-vector results on three inputs, and require cache counts [1,2,2] so an
equivalent third dictionary still reuses code. Julia1.11/1.12 runs are pending.

The unchanged 282-assertion dictionary gate still fails only mutable_key_many
and fresh_key_many: structured238pass44fail, vanilla236pass46fail. Counts vary
with object allocation; this is not a reduction in the known cloning bug.
The independent deterministic mutable-key reduction remains wrong in both
frontends because cloning changes identity-dependent hashes and iteration.
That requires further shared graph-semantics work; the table-state patch does
not solve it.

A broader migration survey is running 50 additional whole files in both modes.
Activation and convolution are completed pairs: 671 passes and six existing
broken checks per frontend. Consult terminal per-mode reports before counting
other files. Dedicated GPU, multiprocess, Python-setup and QA suites remain
separate requirements. The native checkpoint v1 build/monitor is still active;
v2 is prepared but not built. No full-suite or checkpoint completion claim.

Current patch, acceptance sources, exact positive/negative controls, source
fingerprints, partial survey status and live-job continuation instructions:
`/tmp/structured-compatibility-36/README.md`, `verification-summary.json`, and
`verify.py`. The survey uses the prior source stack; only the targeted cache
checks use the new table-state candidate.


## Dictionary conversion, invoke dispatch, and native checkpoint checks (round 37)

Experimental copies only; main implementation remains unpromoted. The round36
cache-state fix now passes47/47 per frontend on Julia1.11,1.12,1.13, against baseline
23pass24fail/47 each. The old1.13 tombstone fixture was invalid on1.11/1.12; those
failure logs are retained. Native witness searches now pin a distinguishing fixture
per version. Use round37 verify.py rather than round36's superseded fixture assumptions.

A separate shared Reactant bug rebuilds Dict tables when changing only their value
type, changing which tensor first(values(d)) selects. The isolated value-layout-v1
candidate preserves slots/keys and initializes new value storage before rehashing
changed keys. On the distinguishing1.13 fixture, both frontends improve from
22pass18fail/40 to40/40 across primal/reverse/HVP. The portable1.11 candidate also
passes41 value-layout plus47 cache-reuse checks per frontend. Further version jobs
are recorded individually. Independent native layout matrices determine each
baseline's expected failures; hash-table behavior differs across Julia versions.
The unchanged282 dictionary gate still fails only the known mutable_key_many and
fresh_key_many groups (structured236pass46fail; vanilla230pass52fail). This is not
a fix for cloning mutable identity-hashed keys.

Full control_flow with annotations preserved gives146pass1fail through both
frontends. Automatic mode adds four failures from tests expecting a removed macro
to reject break/continue/return. The remaining shared function-count assertion
expects2 but sees3: an independent audit proves the third outlined call aliases
its two arguments, changing the unique tensor-argument ABI; cache counts are[1,1,2].
The original assertion remains unchanged. Additional survey failures are recorded,
including LU reconstruction reduced to generic explicit invoke: both frontend
paths discard the selected method and redispatch, causing recursion. No invoke
fix or triangular-specific workaround is implemented.

Native checkpoint v1 built and exposed a subsequent SSA dominance error. The
isolated v2 correction passes native AD IR verification for6/6 cases and the same
29/30 checkpoint directives as baseline. Only8/18 attempted numerical executions
succeed; ten wrappers still fail shape refinement or zero-sized tape/update
validation. Full dynamic-loop differentiation and the CPU/XLA dynamic_pad limit
remain unresolved. No checkpoint policy was changed to hide a failure.

The broader100-job CPU survey remains owned by round36's runner and uses the prior
source stack, not the new dictionary candidates. Terminal-only results, exact
commands, source/binary hashes, negative controls, and remaining work are in
`/tmp/structured-compatibility-37/README.md` and `verification-summary.json`.
No full-suite, timing, precompile, GPU, or implementation-readiness claim.


## Generic use identity and dynamic constant-branch folding (round 38)

Round37's12 dictionary version jobs and100 selected whole-file CPU survey jobs
are all terminal. The combined dictionary candidate passes41+47 checks per frontend
on1.11/1.12/1.13. LogExpFunctions' survey load failure was a missing direct test
dependency already present in the pinned manifest; the isolated environment repair
makes its unchanged full integration file pass110 assertions in both frontends.

A generic IRStructurizer candidate uses operand identity after SSA normalization
in its index and linear use/replacement APIs. It fixes unhashable Static literals
and prevents equal-but-distinct mutable operands or differently typed numeric
constants from being merged. The24 new focused assertions pass on all three Julia
versions; integrated full suites pass1734/1735/1735. Frontend402, counted guards72,
observable results70, and the complete Static integration12 per frontend all pass.
No Static-specific frontend workaround is involved. Literal equality semantics
change to identity and need upstream API review.

A native Enzyme-JAX candidate lets constant-if elimination handle dynamic result
shapes while preserving consumer types with shape-compatible conversions. All six
native reverse-AD cases now verify and all18 analytic numerical checks pass, compared
with8/18 before this fix. AD precedes constant trip-count specialization; this does
not prove unbounded runtime XLA tapes. The unchanged checkpoint corpus retains its
baseline29pass1fail vector. A separate FileCheck regression verifies constant true/
false folding, call result types, and the existing optimization-disable attribute.

The combined isolated stack passes96 conditional primal/reverse/HVP checks with
automatic checkpointing and96 with binomial checkpointing. The unchanged upstream
AD file passes115 checks in each frontend. Disabled checkpointing retains48pass8errors:
dynamic reverse reaches unsupported XLA dynamic_pad import; dynamic HVP fails earlier
on a dynamically shaped zero constant. The HVP failure is now independently reduced
to native MLIR: both dynamic cases fail and the static-loop control verifies.
Enzyme's inactive-shadow creation currently receives a type without a primal shape.
A general shape-aware construction remains to be implemented and validated.

No implementation is promoted, staged, committed, or published. No GPU, timing,
precompile, full-suite, or overall-completion claim. Exact candidates, patches,
commands, hashes, invalid-fixture records, verification and remaining work:
`/tmp/structured-compatibility-38/README.md` and `verification-summary.json`.
Explicit invoke dispatch, mutable identity-hashed dictionary cloning, and the
remaining native/backend problems are still open.


## Shape-aware inactive shadows in native AD (round 39)

An isolated Enzyme interface candidate accepts the primal value when constructing
an inactive zero shadow. Ranked dynamic tensors use standard tensor.dim and
tensor.splat; Enzyme-JAX raises these supported constant splats to StableHLO
shape operations and broadcasts. Static/scalar types retain a default type-only
implementation. No new StructuredReactant special case or checkpoint policy
change is involved. This is an experimental upstream API proposal.

The original dynamic HVP reductions now verify. Six native reverse cases retain
18/18 analytic results, and forward differentiation of those six reverse programs
verifies and passes18/18 analytic HVP checks. AD happens with symbolic bounds;
only afterward are concrete test inputs supplied. The HVP numerical harness uses
the existing constant-pad simplification rule and shape refinement, because the
StableHLO interpreter cannot execute dynamic result types. This does not establish
runtime dynamic-tape support in XLA.

Eight direct cases cover inactive arguments, intermediates, returns, CFG merges,
multiple dynamic dimensions, complex tensors, and a static control. Seven dynamic
cases fail on the previous library and pass on the candidate; the static control
passes on both. All25 numerical executions pass, including empty dimensions and
both branch choices. Unranked tensors, independent validation of batched shadows,
and remaining type-only zero call sites are outside this evidence.

The unchanged native checkpoint corpus retains29pass1fail. Combined Julia checks
pass96 automatic and96 binomial conditional primal/reverse/HVP assertions, plus
115 upstream autodiff assertions in each frontend. Disabled checkpointing retains
48pass8errors, but HVP now advances to the same unsupported XLA dynamic_pad import
as reverse AD. Its earlier invalid zero-constant error is gone.

No implementation is promoted, staged, committed, or published. Protected main
files retain their hashes. Candidate sources, patches, frozen artifacts, exact
commands, failed numerical harness attempts, and verification are recorded in
`/tmp/structured-compatibility-39/README.md` and `verification-summary.json`.
Explicit invoke, identity-hashed dictionary cloning, runtime dynamic tapes, and
the previously documented full-file survey differences remain open.


## Exact invoke preservation across frontends (round 40)

An isolated shared Reactant candidate preserves explicit invoke targets through
optimization and compiles the selected MethodInstance through the existing native
wrapper. The StructuredReactant candidate retains prepared invoke targets and
resolves selected methods separately from ordinary dispatch. Keyword invoke uses
Julia's existing adapter through structured interpretation, avoiding a callback
wrapper that changed dispatch. These are integration changes; no new CFG algorithm,
IRStructurizer patch, tensor special case, or native rebuild is involved this round.

Analytic primal/reverse/HVP checks, static parameters, keyword calls, callable
objects, nospecialize, and method-redefinition cases pass in both frontends:
88 assertions each on Julia 1.11, 108 each on 1.12 and 1.13 (608 total). The lower
1.11 count excludes the newer direct Method invocation API. The unchanged upstream
operations file now passes372 vanilla checks and371 structured checks with the
same retained hlo_call-repeat mismatch. Both previous LU errors are fixed.

Additional boundary checks retain4pass1error vanilla and9pass1error structured.
Selector rejection, type selectors supplied as arguments, and keyword-invoked
traced branching/gradients pass. A Method supplied as a kernel argument fails in
shared input conversion before invocation. Four bounds/inlining policy checks
per frontend agree with ordinary Julia. A candidate source-loading binding-world
error is also retained; the successful acceptance matrix uses normal package loading.

The temporary selected-wrapper cache uses exact world equality, not dependency
reuse or persistence. Runtime shadow-type redispatch, broader native-leaf callback
and flag handling, CodeInstance selectors, and concurrency still need review.
No implementation is promoted, staged, committed, or published. No GPU, timing,
precompile, full-suite or completion claim. Exact sources, patches, failed attempts,
commands, hashes, numerical evidence and remaining work are recorded in
`/tmp/structured-compatibility-40/README.md` and `verification-summary.json`.


## Compiler metadata, invocation routing, and source-loading worlds (round 41)

An isolated shared Reactant candidate preserves Method, MethodInstance, and
CodeInstance objects as opaque host metadata and includes their identity in trace
cache keys. The old MethodInstance behavior can produce wrong numerical results:
two distinct metadata arguments reused one cached body. An ordinary-Reactant
control from before the invoke work fails all four focused cache/key/numerical
assertions; the candidate passes them. Method kernel arguments now pass conversion,
and cached selected-method primal, reverse, and HVP results are checked analytically.
This establishes metadata preservation, not direct CodeInstance invocation support.

The larger fixture exposed prepared host invokes entering the interpreter through
ENV, dynamic invoke selectors escaping Reactant overlays, and a typed invoke being
re-inferred in a native wrapper. The frontend now keeps host-only selected calls
native while retaining target and call-site bounds behavior. Shared Reactant routes
dynamic/splatted invoke syntax through its adapter and emits inferable native invoke
syntax. No new CFG algorithm or native rebuild is involved.

Two shared Reactant source-loading problems were also isolated. A generated
return-type declaration preceded bindings used by its inference callback; moving
the declaration after those bindings fixes the same no-invoke HVP that fails on
both the round 40 source and an older control. Separately, method-label formatting
dispatched to enum display methods in the generator's older world. Both label
rendering sites now use the trace's supplied world through the compiler primitive
already used in this driver. Base.invoke_in_world was an unsuccessful intermediate
attempt, retained with its failures. No latest-world lookup or precompile workload
is added.

The final isolated stack passes 1,301 combined invocation/metadata/bounds assertions
across both frontends on Julia 1.11.9, 1.12.7, and 1.13.0. The late-defined enum
regression passes four assertions per version and four with normal 1.13 package
loading. The unchanged upstream autodiff file passes 115 assertions per frontend;
before the label fix, the same source-loaded file retained 104 passes and four
finite-difference errors in each frontend. The pre-label candidate additionally
passes 230 normal-loading combined assertions and 96 automatic plus 96 binomial
checkpoint assertions. Those results are identified by their actual candidate.

No implementation is promoted, staged, committed, or published. All owned jobs are
terminal and the 20 protected main paths retain their hashes. No GPU, latency,
precompile-time, full-suite, or completion claim. Exact sources, incremental patches,
retained failure controls, commands, hashes, and verification are in
`/tmp/structured-compatibility-41/README.md` and `verification-summary.json`.
The selected-wrapper cache still requires exact world equality; dependency-aware
reuse, runtime shadow-type redispatch, direct CodeInstance selectors, runtime dynamic
tapes, identity-hashed dictionary cloning, and the earlier survey differences remain
open. Private dependency/native fixes still require review and upstreaming.
