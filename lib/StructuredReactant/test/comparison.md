# Automatic control-flow comparison

This audit compares StructuredReactant with ordinary Julia results and vanilla
Reactant, using the public test corpus linked by
[Reactant.jl#3265](https://github.com/EnzymeAD/Reactant.jl/issues/3265).
It does **not** measure glou-nes's current Julia 1.12 implementation: the issue
links the older [PR #1828](https://github.com/EnzymeAD/Reactant.jl/pull/1828),
and no current source revision was available for a direct comparison.

## Corpus and protocol

`polybench.jl` loads the 30 kernel bodies from
[`90eb97192a740f9a81ced7553f3943aec293cbfd`](https://github.com/glou-nes/Reactant.jl/blob/90eb97192a740f9a81ced7553f3943aec293cbfd/test/auto_cf/polybench.jl).
The downloaded file's SHA-256 must be
`25b3999e9e9ab8d593f34204e4f597d04c53f23d7936f138a356579ea14d1485`.
The kernels are MIT-licensed by the Enzyme Automatic Differentiation Compiler.
No downloaded source is vendored here or fetched by the test runner.

The original test harness is replaced with these checks:

- Run ordinary Julia on independent input copies, checking for finite results.
- Compile the original kernel bodies; only the host copy drops the three
  `TracedRArray` assertions in `kernel_correlation`.
- Use proper vectors, distinct mutable arguments, and nondegenerate matrices.
  Cholesky receives a positive-definite matrix.
- Test sizes 4 and 5, with two sets of values per executable. Both returned
  values and the state of input arguments must match Julia.
- Keep scalar parameters on the host, as in the original corpus. Track arrays.
- Use `rtol=1e-7`, `atol=1e-9`. This checks the published Julia algorithms,
  not conformance to a separate mathematical PolyBench specification.

The public `kernel_durbin` accesses index zero with a proper vector. Its original
row-matrix fixture has `size(y, 1) == 1`, so the problematic loop does not run.
The harness reports an invalid reference instead of changing the algorithm.

## Running

Use a Julia environment containing Reactant and, for the structured mode,
StructuredReactant. The runner selects the CPU backend. For a CPU-only package
setup, set `gpu = "none"` and `gpu_version = "none"` under `[Reactant_jll]` in
that environment's `LocalPreferences.toml` before loading Reactant.

Download the pinned source and its license once:

```sh
curl --fail --location https://raw.githubusercontent.com/glou-nes/Reactant.jl/90eb97192a740f9a81ced7553f3943aec293cbfd/test/auto_cf/polybench.jl -o /tmp/polybench-upstream.jl
curl --fail --location https://raw.githubusercontent.com/glou-nes/Reactant.jl/90eb97192a740f9a81ced7553f3943aec293cbfd/LICENSE -o /tmp/polybench-LICENSE
```

Run one case per process when testing the default optimization pipeline, since
some cases abort inside native code. For example, from the repository root:

```sh
CASE=kernel_2mm FRONTEND=structured julia --project=lib/StructuredReactant/test lib/StructuredReactant/test/polybench.jl /tmp/polybench-upstream.jl /tmp/structured-2mm.toml
CASE=kernel_2mm FRONTEND=vanilla julia --project=lib/StructuredReactant/test lib/StructuredReactant/test/polybench.jl /tmp/polybench-upstream.jl /tmp/vanilla-2mm.toml
```

Omit `CASE` to run the whole corpus. `SKIP` accepts comma-separated kernel names.
`OPTIMIZE_FALSE=1` disables optional MLIR optimization, retains per-shape MLIR
beside the result file, and records `stablehlo.while` and `stablehlo.if` counts.
It is a diagnostic control, not a replacement for the default-pipeline results.
The runner exits nonzero for numerical mismatches, errors, or invalid references.
Vanilla mode does not import StructuredReactant.

The default frontend suite also includes `while_conditions.jl`, the reduced
Deriche recurrence in `recurrence.jl`, and the existing loop-exit, invalidation,
and world-age tests in `runtests.jl` and `cache.jl`.

## Original audit, 2026-09-11

Julia 1.13.0, CPU/PJRT, StructuredReactant source at `8d7b9292d`, Reactant based on
`64e349ffe`, IRStructurizer 0.6.4. These are correctness checks, not timings.

| Pipeline | Frontend | Correct kernels | Numerical mismatches | Julia errors | Native aborts | Invalid host reference |
|---|---|---:|---:|---:|---:|---:|
| Default | Structured | 16 | 0 | 0 | 13 | 1 |
| Default | Vanilla | 13 | 0 | 3 | 13 | 1 |
| Optional MLIR optimization disabled | Structured | 28 | 1 | 0 | 0 | 1 |
| Optional MLIR optimization disabled | Vanilla | 26 | 0 | 3 | 0 | 1 |

The three vanilla Julia errors are traced Boolean conditions in correlation,
Floyd–Warshall, and Nussinov. Both frontends hit native optimization aborts (11 kernels in common);
the stack traces include `DUSSliceSimplify`/`stablehlo.slice`. The unoptimized
control separates those failures from the frontend's emitted semantics.

Deriche disagrees with Julia under StructuredReactant, while vanilla agrees.
`recurrence.jl` reduces this to a forward recurrence that is wrong from its third
output onward. The backward recurrence agrees. The subsequent fix is described below.

Of the 29 kernels with valid references, only four produce `stablehlo.while`
under StructuredReactant in this harness: Deriche, Gram–Schmidt, LU decomposition
(`ludcmp`), and `symm`. The other kernels' loops unroll. A numerical pass does not
establish automatic loop preservation.

Repeating the default structured matrix with IRStructurizer main at
`496c3d35620d699a3b74937ca6b394bbdc31cee8` gives the same counts. Deriche still
mismatches in the optimization-disabled control. In that isolated checkout,
only the package version was changed to `0.6.5-DEV` to activate the frontend's
existing post-0.6.4 compatibility path; production dependencies were unchanged.

The condition corpus has 43 successful checks and six explicit known-broken
checks. Short-circuiting, guarded mutation, mutation in a helper, and condition
rebinding pass. Plain `while` condition mutation can run an extra time, including
when the body executes zero times. The initial condition emission used to decide
whether to roll leaves mutations visible before the loop starts. The recurrence
corpus adds four passing backward checks and four known-broken forward checks.
At the time of the audit these wrong results were recorded as `@test_broken`.
The original suite had 200 passing and 12 broken checks: two pre-existing skips
and ten checks reproducing the two bugs.

Raw logs, source snapshots, environments, subprocess statuses, MLIR, and the
aggregate report for this run are in `/tmp/structured-control-flow-comparison`.

## Frontend fixes

Both wrong-result bugs are fixed in `src/regions.jl`, without changing
IRStructurizer, Reactant, dependencies, or the precompile workload:

- A source `while` is rotated. Its initial condition runs once before the
  StableHLO loop; the body computes the next condition after its own work. The
  StableHLO condition reads the carried Boolean, and the final condition's
  forwarded values are returned without evaluating it again.
- Loop-carried updates collect all next MLIR values before changing any tracer.
  This preserves simultaneous assignment, including a next value that refers to
  another old carry. The former update order lost that old value in Deriche.

All ten former broken checks are now regular assertions. Added checks exercise
aliased condition captures, mutation inside a non-inlined helper, nested carried
state, and the at-most-one-iteration return case from issue #3265.

The numerical corpus now passes **29 of 29 kernels with valid Julia references**
with optional MLIR optimization disabled. Durbin's invalid reference is retained.
The remaining native optimization crash also reproduces in vanilla Reactant:

```julia
function disjoint_update_slice(x, y)
    z = copy(x)
    z[4, 4] = y[1, 1]
    return z[1:1, 1:1]
end
```

On a 4×4 input and 1×1 update, `DUSSliceSimplify` in the pinned Enzyme-JAX
optimizer tries to slice the 1×1 update with a negative limit. The update is
outside the region being read. The frontend does not disable this rewrite or
vendor a workaround. This native issue remains unresolved by the frontend fix.

An additional exploratory probe that starts a nested named tuple with host
`0.0` fields, then changes its field types inside the loop, raises an
IRStructurizer “carried value but no entry value” error even on main. That is
separate from simultaneous carry updates; the nested-update regression uses
traced initial fields. The probe and logs are preserved for follow-up.

Final CPU suite validation on the fixed source:

| Julia | IRStructurizer | Passing | Existing skips |
|---|---|---:|---:|
| 1.11.9 | 0.6.4 | 219 | 2 |
| 1.12.7 | 0.6.3 | 221 | 2 |
| 1.13.0 | 0.6.4 | 221 | 2 |
| 1.13.0 | main at `496c3d3` | 223 | 0 |

The main checkout uses the version-label adjustment described above. Each run
includes 56 condition checks and 12 recurrence checks. The 29 valid Polybench
kernels account for 116 numerical executions. No latency claim is made from
these concurrent correctness runs.

Fix validation, commands, source fingerprints, and the native reproducer are in
`/tmp/structured-control-flow-fixes` (`summary.json` and `final-*.log`).
