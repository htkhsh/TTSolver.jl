# AMEn/TAMEn Workspace Optimization Design

## Goal

Integrate the allocation-reducing AMEn/TAMEn implementation currently kept in
`src_mod` into the maintained `src` implementation without regressing numerical
accuracy or removing the conserved-observable API.

The optimized implementation must:

- reuse local matvec, environment-contraction, and BiCGSTAB scratch storage;
- preserve existing public call patterns and return values;
- preserve `conserved_observables` and `correct_2_norm` behavior;
- keep optimized and pre-migration results within the requested solver
  tolerance; and
- reduce warmed AMEn/TAMEn allocations on representative problems.

The pre-migration reference is the maintained implementation at commit
`9c0b86c` (`Modify tt_round`). The ignored `src_mod` directory is an
implementation reference, not a replacement source tree.

## Integration Strategy

Port the optimized internals as one coherent workspace architecture while
manually merging the maintained public API. Do not copy `src_mod` over `src`.
This avoids restoring older code that predates the conserved-observable
keywords and also avoids importing its CRLF line endings.

The change is divided by responsibility:

- `amen_utils.jl` owns reusable storage and allocation-aware contraction
  kernels;
- `amen_solve.jl` owns cache lifetime within AMEn solves and the cached
  BiCGSTAB iteration;
- `tamen.jl` owns cache lifetime across TAMEn sweeps and, optionally, across
  separate TAMEn calls; and
- `TTSolver.jl` exposes the public workspace type.

No unrelated refactoring is included.

## Public API

The maintained TAMEn keywords remain supported and a workspace keyword is
added:

```julia
tamen(
    X,
    A,
    tol,
    opts=Dict{Symbol,Any}(),
    z=nothing;
    conserved_observables=nothing,
    correct_2_norm=nothing,
    workspace::Union{Nothing,TAMEnWorkspace}=nothing,
)
```

`conserved_observables` retains its current type, dimension, and conflict
validation. `correct_2_norm` retains its current Boolean and conflict
validation. Supplying `workspace` must not change numerical options, warm-start
the solution, or bypass those validations.

`TAMEnWorkspace` is exported so callers performing parameter sweeps or
segmented time evolution can reuse capacity across calls:

```julia
work = TAMEnWorkspace(Float64)
U1 = tamen(X1, A1, tol, opts1; workspace=work)
U2 = tamen(X2, A2, tol, opts2; workspace=work)
```

Omitting `workspace` creates one workspace per `tamen` call. Scratch storage is
still reused across that call's time steps and sweeps.

One workspace belongs to one sequential computation. Sharing it across
concurrent or nested solves is unsupported. Returned solutions and environment
outputs own their storage and remain valid after the workspace is reused.
Workspace capacity may grow and remains retained until the workspace is
released.

The public `amen_solve`, `bicgstab`, and `bicgstab_inplace` signatures and
return tuple shapes remain unchanged.

## Workspace Architecture

### Local matvec storage

`LocalMatvecBuffer{T}` owns a one-dimensional grow-only vector. A request for a
larger active view replaces the vector with capacity at least the requested
size and approximately 1.5 times the previous capacity. Existing live views
therefore retain their original owner.

`LocalMatvecTermBuffers{T}` stores the six temporary matrices required by one
operator term. `LocalMatvecCache{T}` stores the packed input, term buffers, and
borrowed output for one sequential local-matvec stream. Separate caches are
used for the local solve and residual/enrichment paths so one path cannot
invalidate the other's borrowed output.

`prepare_local_matvec!` validates dimensions, refreshes operator-dependent
data, and binds active matrix views to cached storage. The existing
`LocalMatvecWorkspace(...)` constructor and allocating `local_matvec(...)`
wrapper remain available for compatibility.

The optimized kernel:

- uses `copyto!` for the common `b == 1` input layout;
- replaces element-wise regrouping loops with `permutedims!` into disjoint
  buffers;
- performs contractions through `mul!`; and
- skips the operator GEMM for an exact rank-one identity with a matching
  standard floating-point element type.

The identity shortcut is exact, not tolerance-based. Non-finite intermediates
use the ordinary GEMM path so `0*Inf` and `NaN` propagation is preserved.

### Environment-contraction storage

`EnvironmentUpdateWorkspace{T}` owns independent buffers for contraction
stages, conjugation, and packing noncontiguous inputs. Type-specific banks
preserve mixed-precision arithmetic instead of converting all inputs to the
workspace's primary type.

New bang forms update caller-provided outer containers:

```julia
leftreduce_matrix!(dest, src, w, A, x, work)
rightreduce_matrix!(dest, src, w, A, x, work)
leftreduce_vector!(dest, src, w, xs, work)
rightreduce_vector!(dest, src, w, xs, work)
```

Each destination entry is replaced by a newly owned result. Source arrays are
never overwritten, even if `dest` began as a shallow copy of `src`. The
existing non-bang functions delegate through a temporary workspace and retain
their owning-output behavior.

Per-term function barriers specialize contractions when AMEn stores operators
or cores in `Vector{Any}` containers.

### BiCGSTAB storage

`BiCGSTABCache` retains the six recurrence vectors used by local iterative
solves. Solution vectors remain independently owned. Separate type banks
support cases where the solution and right-hand side use different element
types.

The local solver reuses the already computed initial residual when element
types match. Before a failed iteration is restarted with a preconditioner, it
recomputes the true residual and restarts only if the requested tolerance is
still unmet.

Convergence and reported relative residual use one consistent denominator:

```text
relative residual = norm(b - A*x) / norm(b)
```

For a zero right-hand side, an exactly zero residual reports `0`; a nonzero
residual reports `Inf`. Non-finite residuals never count as converged.

### TAMEn workspace

`TAMEnWorkspace{T}` groups:

- solve and residual `LocalMatvecCache` instances;
- one `EnvironmentUpdateWorkspace`; and
- one `BiCGSTABCache`.

`amen_solve` creates these caches once per call and reuses them across sweeps.
`tamen` obtains them from its supplied or internally created
`TAMEnWorkspace` and reuses them across all time steps and sweeps.

Workspace reuse does not reuse operator values, environments, residuals,
Krylov state, or a previous solution. All such active data is refreshed for
each solve.

## Numerical Compatibility

All accuracy comparisons use deterministic random seeds and identical input
cores, options, and tolerances.

For a reference vector `u_ref` and optimized vector `u_opt`, define

```text
relative_l2(u_opt, u_ref) =
    norm(u_opt - u_ref) / max(norm(u_ref), eps(real(eltype(u_ref))))
```

For each controlled AMEn solve, the expanded optimized solution must satisfy:

```text
relative_l2(x_opt, x_ref) <= tol
```

and its directly recomputed residual must satisfy:

```text
norm(A*x_opt - b) / norm(b) <= tol
```

The residual requirement uses a nonzero right-hand side and a well-conditioned
test operator chosen so the requested tolerance is achievable.

For each controlled TAMEn problem, compare every requested snapshot from the
optimized implementation with the corresponding pre-migration snapshot:

```text
maximum(relative_l2(u_opt(t), u_ref(t))) <= tol
```

This optimized-versus-reference condition checks that workspace reuse does not
change the maintained algorithm beyond its requested accuracy. It does not
replace independent correctness checks. Existing analytic-solution and
conserved-observable tests retain their current absolute or relative error
limits, including the decay ODE and Chebyshev/CN conserved-functional tests.

The following numerical cases are required:

1. dense real local operators;
2. sparse local operators;
3. complex-valued cores and operators;
4. exact local identity terms;
5. zero and nonzero right-hand sides in BiCGSTAB;
6. AMEn direct and iterative local-solve branches;
7. TAMEn Chebyshev and Crank-Nicolson schemes;
8. TAMEn with and without conserved observables; and
9. repeated TAMEn calls using one workspace.

## Allocation and Performance Validation

Wall-clock timing is recorded for information but is not a pass/fail condition,
because shared CI machines make small timing thresholds unreliable.

Allocation checks run after compilation and warmup. They verify:

- repeated `local_matvec!` calls with a prepared workspace remain within the
  existing low-allocation bound;
- cached environment updates allocate less than the pre-migration allocating
  path on the same cores and operators;
- the second compatible `tamen(...; workspace=work)` call allocates less than
  a warmed call that constructs a fresh workspace; and
- the representative larger TAMEn benchmark allocates fewer bytes than the
  pre-migration implementation.

Tests assert allocation reduction, not an exact speedup ratio. Benchmark output
records median wall time, allocated bytes, and allocation count for before/after
comparison.

## Error Handling and Invariants

Preparation functions reject negative dimensions, mismatched vector lengths,
operator-term count mismatches, and inconsistent environment/operator shapes
with `DimensionMismatch`.

A prepared local-matvec workspace is invalid after the same cache is prepared
for another operator. Borrowed outputs are valid only until the next use of
that output cache. These internal lifetime rules are documented in the type and
function docstrings.

Public `tamen` results must never alias scratch storage. Reusing a workspace
after a solve cannot modify a previously returned solution.

Existing mutation behavior is retained: `opts` may receive defaults as it does
today, while input TT cores, operators, observables, and returned results are
not mutated by scratch reuse.

## Test Strategy

Implementation follows test-driven development:

1. Add behavioral equality tests for cached local matvec and environment
   updates against their existing allocating forms.
2. Add BiCGSTAB convergence, zero-RHS, restart, and cache-reuse tests.
3. Add workspace ownership and repeated-call tests.
4. Record deterministic pre-migration AMEn/TAMEn reference outputs.
5. Add end-to-end relative-L2, true-residual, analytic, and conservation tests.
6. Confirm those tests fail when pointed at missing workspace APIs or an
   intentionally perturbed result.
7. Port the smallest coherent implementation layer needed to make each group
   pass.
8. Run the complete test suite and allocation benchmarks.

Reference fixtures store only compact deterministic outputs needed for the
comparison. They do not serialize implementation-specific workspaces or
intermediate environments.

## Files Changed

- `src/amen_utils.jl`: workspace types and optimized contraction kernels.
- `src/amen_solve.jl`: cache propagation and cached BiCGSTAB implementation.
- `src/tamen.jl`: API-preserving workspace integration.
- `src/TTSolver.jl`: `TAMEnWorkspace` export.
- `test/runtests.jl` and focused test files if splitting the suite improves
  targeted execution.
- `benchmark/bench_tamen.jl`: reproducible before/after allocation reporting.

`src_mod` remains ignored and unchanged.

## Non-goals

- Further changes to `tt_round`; its scale correction is already integrated.
- Changes to KSL, TT constructors, or unrelated TT arithmetic.
- A shared thread-safe workspace.
- Warm-starting TAMEn or BiCGSTAB from a previous solve.
- New time-integration schemes.
- Changing conserved-observable mathematics or validation.
- Requiring a fixed wall-clock speedup on every machine.
- Reformatting unchanged files or normalizing `src_mod` line endings.
