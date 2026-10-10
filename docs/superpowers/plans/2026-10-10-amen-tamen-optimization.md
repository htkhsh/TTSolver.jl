# AMEn/TAMEn Workspace Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the allocation-reducing AMEn/TAMEn workspace implementation while preserving the maintained conservation API and keeping optimized results within the requested solver tolerance.

**Architecture:** Introduce grow-only scratch banks in `amen_utils.jl`, propagate them through `amen_solve.jl`, and group them in an optionally caller-owned `TAMEnWorkspace` used by `tamen.jl`. Preserve all existing allocating wrappers and public call patterns, merge the workspace keyword into the current conserved-observable signature, and validate numerical equivalence against deterministic pre-migration fixtures before measuring allocations.

**Tech Stack:** Julia 1.11, LinearAlgebra, SparseArrays, TensorOperations, BenchmarkTools, Julia `Test`, TTSolver TT types.

**Spec:** `docs/superpowers/specs/2026-10-10-amen-tamen-optimization-design.md`

## Global Constraints

- Use commit `9c0b86c` (`Modify tt_round`) as the pre-migration numerical reference.
- Keep `conserved_observables` and `correct_2_norm`, including their current validation and conflict behavior.
- Add `workspace::Union{Nothing,TAMEnWorkspace}=nothing` without changing positional arguments or return tuple shapes.
- Export `TAMEnWorkspace` for sequential reuse across separate `tamen` calls.
- Never share one workspace across concurrent or nested solves.
- Returned solutions and environment outputs own their storage and must not alias reusable scratch.
- Preserve mixed-precision arithmetic with type-specific scratch banks.
- Require optimized-versus-reference maximum relative L2 error `<= tol` for controlled AMEn solutions and all controlled TAMEn snapshots.
- Retain every existing analytic-solution and conserved-observable error threshold.
- Use warmed allocation counts as the performance gate; wall-clock timing is informational only.
- Leave `src_mod` ignored and unchanged; do not copy it wholesale or import CRLF formatting.
- Do not change KSL, `tt_round`, TT constructors, conservation mathematics, or time-integration schemes.

## Review Focus

- Zero right-hand side: BiCGSTAB must report relative residual `0` only for an exact zero residual and `Inf` for a nonzero residual; Task 4 pins both cases.
- Mixed precision: cached matvec, environment updates, and BiCGSTAB must retain product/solution element types without narrowing; Tasks 2, 3, and 4 include mixed-type assertions.
- Noncontiguous and sparse inputs: packing must occur only when BLAS cannot consume the input directly, with numerical results unchanged; Tasks 2 and 3 exercise both forms.
- Workspace growth and reuse: growing a buffer must not invalidate an already returned owning result, and a later solve must not mutate an earlier result; Tasks 2 and 6 test those lifetimes.
- Keyword composition: `workspace`, `conserved_observables`, and `correct_2_norm` must work together while legacy conflict errors remain unchanged; Task 6 exercises the combined and conflicting calls.

## File Structure

- Modify `src/amen_utils.jl`: scratch-buffer types, cached local matvec, allocation-aware environment contractions, and cache preparation helpers.
- Modify `src/amen_solve.jl`: cache propagation through sweeps and cached BiCGSTAB recurrence.
- Modify `src/tamen.jl`: API-preserving `TAMEnWorkspace` use across time steps and sweeps.
- Modify `src/TTSolver.jl`: export `TAMEnWorkspace`.
- Modify `test/runtests.jl`: include focused optimization test files.
- Create `test/amen_tamen_test_utils.jl`: deterministic problem builders and `relative_l2` helper.
- Create `test/fixtures/amen_tamen_reference.jl`: compact literal outputs captured from commit `9c0b86c`.
- Create `test/amen_optimization_tests.jl`: local matvec, environment, BiCGSTAB, and AMEn tests.
- Create `test/tamen_optimization_tests.jl`: public workspace, snapshot, conservation, ownership, and allocation tests.
- Modify `benchmark/bench_tamen.jl`: warmed fresh-workspace and reused-workspace allocation/time reporting.

---

### Task 1: Freeze the pre-migration numerical reference

**Files:**
- Create: `test/amen_tamen_test_utils.jl`
- Create: `test/fixtures/amen_tamen_reference.jl`
- Create: `test/amen_optimization_tests.jl`
- Create: `test/tamen_optimization_tests.jl`
- Modify: `test/runtests.jl:1-7`

**Interfaces:**
- Consumes: current `amen_solve`, `tamen`, `extract_snapshot`, `tt_full`, and commit `9c0b86c` behavior.
- Produces: `relative_l2(actual, reference)`, deterministic `amen_reference_problem(branch)`, deterministic `tamen_reference_problem(scheme; conserved=false)`, and literal reference arrays `AMEN_DIRECT_REFERENCE`, `AMEN_ITERATIVE_REFERENCE`, `TAMEN_CHEB_REFERENCE`, and `TAMEN_CN_REFERENCE`.

- [ ] **Step 1: Add deterministic test helpers**

Create `test/amen_tamen_test_utils.jl` with these interfaces:

```julia
relative_l2(actual, reference) =
    norm(actual - reference) /
    max(norm(reference), eps(real(eltype(reference))))

amen_reference_problem(branch::Symbol)
tamen_reference_problem(scheme::String; conserved::Bool=false)
```

Use fixed literal cores and operators; do not generate fixtures from random values. Configure `amen_reference_problem(:direct)` so every local problem is below `max_full_size`, and `amen_reference_problem(:iterative)` so at least one local problem uses BiCGSTAB. Configure TAMEn with a nonzero initial state, snapshot times `[0.0, 0.5, 1.0]`, both `"cheb"` and `"CN"`, and `tol = 1e-6`.

- [ ] **Step 2: Capture compact reference outputs before production edits**

At commit `9c0b86c`, run the helper problems with `Random.seed!(20261010)` and print expanded AMEn solutions plus each expanded TAMEn snapshot using `repr`. Paste those literal arrays, the exact options, and `REFERENCE_COMMIT = "9c0b86c"` into `test/fixtures/amen_tamen_reference.jl`.

Expected: repeated runs print identical arrays. Inspect each array for finite values before recording it.

- [ ] **Step 3: Add characterization tests for the fixtures**

In the focused files, add:

```julia
@testset "pre-migration AMEn reference" begin
    @test relative_l2(run_amen(:direct), AMEN_DIRECT_REFERENCE) <= 10eps(Float64)
    @test relative_l2(run_amen(:iterative), AMEN_ITERATIVE_REFERENCE) <= 10eps(Float64)
end

@testset "pre-migration TAMEn reference" begin
    @test maximum(relative_l2.(run_tamen("cheb"), TAMEN_CHEB_REFERENCE)) <= 10eps(Float64)
    @test maximum(relative_l2.(run_tamen("CN"), TAMEN_CN_REFERENCE)) <= 10eps(Float64)
end
```

`run_amen` and `run_tamen` are test-only wrappers in `test/amen_tamen_test_utils.jl`; `run_tamen` returns expanded vectors at the three fixed snapshot times.

- [ ] **Step 4: Include the focused files from the suite**

After the package imports in `test/runtests.jl`, add:

```julia
include("amen_tamen_test_utils.jl")
include("fixtures/amen_tamen_reference.jl")
include("amen_optimization_tests.jl")
include("tamen_optimization_tests.jl")
```

- [ ] **Step 5: Run the characterization tests**

Run:

```bash
julia --project=. -e 'using Test, LinearAlgebra, SparseArrays, Random, TTSolver; include("test/amen_tamen_test_utils.jl"); include("test/fixtures/amen_tamen_reference.jl"); include("test/amen_optimization_tests.jl"); include("test/tamen_optimization_tests.jl")'
```

Expected: the reference test sets pass against unmodified production code. No workspace tests exist yet.

- [ ] **Step 6: Commit the reference harness**

```bash
git add test/runtests.jl test/amen_tamen_test_utils.jl test/fixtures/amen_tamen_reference.jl test/amen_optimization_tests.jl test/tamen_optimization_tests.jl
git commit -m "test: freeze AMEn TAMEn numerical reference"
```

---

### Task 2: Add reusable local-matvec storage

**Files:**
- Modify: `test/amen_optimization_tests.jl`
- Modify: `src/amen_utils.jl:1-233`

**Interfaces:**
- Consumes: existing `LocalMatvecWorkspace`, `local_matvec!`, and `local_matvec` behavior.
- Produces: `LocalMatvecBuffer{T}`, `LocalMatvecTermBuffers{T}`, `LocalMatvecCache{T}`, `AmenMatvecCaches{T}`, `_matvec_storage!`, `_matvec_matrix!`, `_matvec_cache_for`, `prepare_local_matvec!`, and `_cached_local_matvec!`.

- [ ] **Step 1: Add failing cached-matvec tests**

Add `@testset "cached local matvec"` covering dense real, dense complex, sparse, exact rank-one identity, and `b == 1`/`b > 1` cases. For every case assert:

```julia
expected = TTSolver.local_matvec(args...)
cache = TTSolver.LocalMatvecCache(eltype(x))
actual = copy(TTSolver._cached_local_matvec!(cache, cached_args...))
@test actual ≈ expected rtol=1e-13 atol=1e-13
@test eltype(actual) === eltype(expected)
```

Add `@test_throws DimensionMismatch` cases for a wrong input length, mismatched operator-term counts, and inconsistent environment shape. Add a buffer-growth assertion that a previously copied owning output remains unchanged after preparing a larger problem. Add a noncontiguous input view and compare it with its contiguous copy.

- [ ] **Step 2: Run the focused test and verify RED**

Run the focused AMEn test command from Task 1.

Expected: FAIL because `LocalMatvecCache` and `_cached_local_matvec!` are undefined; characterization tests remain green.

- [ ] **Step 3: Implement buffer and cache types**

In `src/amen_utils.jl`, add these exact constructors and helpers:

```julia
LocalMatvecBuffer(::Type{T}) where T
LocalMatvecTermBuffers(::Type{T}) where T
LocalMatvecCache(::Type{T}) where T
AmenMatvecCaches(::Type{T}) where T
_matvec_storage!(buffer::LocalMatvecBuffer{T}, required::Int) where T
_matvec_matrix!(buffer, rows, columns)
_matvec_cache_for(cache::LocalMatvecCache, ::Type{T}) where T
```

Use replacement-on-growth with capacity `max(required, capacity + cld(capacity, 2))` and type-specific cache banks.

- [ ] **Step 4: Implement prepared and borrowed-output matvec paths**

Change `LocalMatvecWorkspace` to accept cached matrix views, add exact identity flags, and implement:

```julia
prepare_local_matvec!(cache::LocalMatvecCache{T}, x,
    rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, ra1, ra2) where T
_cached_local_matvec!(cache, x, rx1, m, rx2, b, rw1, n, rw2,
    WAX1, A, WAX2, ra1, ra2)
```

Keep the existing allocating constructor and `local_matvec` wrapper. Use `copyto!` for `b == 1`, `permutedims!` for regrouping, `mul!` for contractions, and skip only exact finite standard-float identity terms. Materialize the reverse-QR transpose in `orthogonalise_block` so later reshapes are BLAS-compatible.

- [ ] **Step 5: Verify numerical equality and warmed allocations**

Run the focused test. Then warm `prepare_local_matvec!` and ten `local_matvec!` calls before measuring:

```julia
allocated = @allocated begin
    for _ in 1:10
        TTSolver.local_matvec!(result, x, workspace)
    end
end
@test allocated <= 4096
```

Expected: all Task 1-2 tests pass, including dense/sparse/complex/identity equality and the existing allocation bound.

- [ ] **Step 6: Commit local-matvec caching**

```bash
git add src/amen_utils.jl test/amen_optimization_tests.jl
git commit -m "perf: reuse AMEn local matvec storage"
```

---

### Task 3: Add allocation-aware environment updates

**Files:**
- Modify: `test/amen_optimization_tests.jl`
- Modify: `src/amen_utils.jl:36-127`

**Interfaces:**
- Consumes: Task 2 `LocalMatvecBuffer` and `_matvec_storage!`.
- Produces: `EnvironmentUpdateBuffers{T}`, `EnvironmentUpdateWorkspace{T}`, `leftreduce_matrix!`, `rightreduce_matrix!`, `leftreduce_vector!`, and `rightreduce_vector!`; existing non-bang wrappers remain unchanged to callers.

- [ ] **Step 1: Add failing environment-equivalence tests**

For all four reducers, build fixed real and complex cores and assert:

```julia
expected = allocating_reducer(src, w, terms...)
dest = copy(src)
work = TTSolver.EnvironmentUpdateWorkspace(eltype(w))
returned = bang_reducer(dest, src, w, terms..., work)
@test returned === dest
@test all(isapprox.(dest, expected; rtol=1e-13, atol=1e-13))
```

Add sparse operator input, noncontiguous `PermutedDimsArray` input, and mixed `Float32`/`Float64` input assertions. Save deep copies of `src` and the first output, reuse the workspace with a larger problem, then assert the source and first owning output are unchanged. Add mismatched term-count, core-shape, mode-size, and environment-rank `DimensionMismatch` assertions.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: FAIL because `EnvironmentUpdateWorkspace` and bang reducers are undefined.

- [ ] **Step 3: Implement environment storage and input packing**

Add:

```julia
EnvironmentUpdateBuffers(::Type{T}) where T
EnvironmentUpdateWorkspace(::Type{T}) where T
_environment_buffers(work::EnvironmentUpdateWorkspace, ::Type{T}) where T
_environment_array!(work, ::Type{T}, slot, dims...) where T
_environment_input!(work, slot, matrix)
_environment_product_type(::Type{T}, ::Type{S}) where {T,S}
```

Pass sparse matrices and column-major `StridedMatrix` inputs through directly; pack other layouts into typed scratch. Use `Base.promote_op` for product element types.

- [ ] **Step 4: Implement the four bang reducers and compatibility wrappers**

Implement each public internal bang signature from the spec and a per-term function barrier. Use `mul!` and `permutedims!`, allocate only each returned environment entry, and never write into a source entry. Rewrite the existing non-bang functions as wrappers that create a temporary workspace and delegate.

- [ ] **Step 5: Verify equality, ownership, mixed precision, and allocation reduction**

Run the focused test. After warmup, compare `@allocated` for ten cached bang updates with ten existing allocating-wrapper calls on identical inputs.

Expected: all equality and invariant tests pass; cached bang updates allocate fewer bytes than wrappers.

- [ ] **Step 6: Commit environment caching**

```bash
git add src/amen_utils.jl test/amen_optimization_tests.jl
git commit -m "perf: reuse AMEn environment scratch"
```

---

### Task 4: Add cached BiCGSTAB with RHS-relative convergence

**Files:**
- Modify: `test/amen_optimization_tests.jl`
- Modify: `src/amen_utils.jl`
- Modify: `src/amen_solve.jl:405-494`

**Interfaces:**
- Consumes: Task 2 buffers and the existing public `bicgstab`/`bicgstab_inplace` callbacks.
- Produces: `BiCGSTABWorkspace`, `BiCGSTABCache`, `prepare_bicgstab!`, `_bicgstab!`, `_bicgstab_residual!`, `_bicgstab_converged`, `_bicgstab_relres`, and `_needs_local_restart!`.

- [ ] **Step 1: Add failing residual-semantics and cache tests**

Use a fixed nonsymmetric real matrix and a fixed complex matrix. Assert for allocating, public in-place, and cached internal solvers:

```julia
@test flag == 0
@test solution ≈ A \ rhs rtol=1e-12
@test reported_relres ≈ norm(rhs - A * solution) / norm(rhs) atol=1e-14
```

For `A = I`, `rhs = zeros(3)`, assert a zero initial guess returns relative residual `0`. With a nonzero `x0` and `maxiter=0`, assert the relative residual is `Inf`. Add mixed solution/RHS element types and assert no narrowing. Reuse one cache at a larger vector length and assert the previously returned solution is unchanged. Exercise `_needs_local_restart!` once with a converged solution and once with a perturbed solution.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: missing cache APIs fail; the public residual assertion also exposes the current initial-residual denominator when `x0` is nonzero.

- [ ] **Step 3: Implement BiCGSTAB scratch preparation**

Add the exact types and preparation methods specified by Task 4 Interfaces. Keep solution vectors outside the cache. Store separate typed banks keyed by `(eltype(x), eltype(b))`.

- [ ] **Step 4: Implement one RHS-relative recurrence**

Move the in-place recurrence into:

```julia
_bicgstab!(x, mvfun!, b, work::BiCGSTABWorkspace;
    initial_residual=nothing, tol=1e-6, maxiter=100, M=nothing)
```

Use `norm(b)` for entry, iteration, restart, and returned relative residual. On the first iteration initialize `p` directly from `r` because cached `p` and `v` may contain prior data. Make `bicgstab_inplace` allocate an ordinary workspace and delegate so its public callback contract remains unchanged. Update allocating `bicgstab` to the same convergence semantics.

- [ ] **Step 5: Run focused tests and verify GREEN**

Expected: real, complex, zero-RHS, mixed-precision, restart, and ownership tests pass; existing in-place destination-reuse tests remain green.

- [ ] **Step 6: Commit cached BiCGSTAB**

```bash
git add src/amen_utils.jl src/amen_solve.jl test/amen_optimization_tests.jl
git commit -m "perf: reuse AMEn BiCGSTAB storage"
```

---

### Task 5: Propagate caches through AMEn sweeps

**Files:**
- Modify: `test/amen_optimization_tests.jl`
- Modify: `src/amen_solve.jl:21-404`

**Interfaces:**
- Consumes: `AmenMatvecCaches`, `EnvironmentUpdateWorkspace`, `BiCGSTABCache`, and all cached kernels from Tasks 2-4.
- Produces: backward-compatible `amen_solve(...)` and internal `amenany_sweep(...; matvec_caches=nothing, environment_workspace=nothing, bicgstab_cache=nothing)`.

- [ ] **Step 1: Strengthen AMEn accuracy tests before integration**

For both direct and iterative reference problems assert:

```julia
actual = run_amen(branch)
reference = branch === :direct ? AMEN_DIRECT_REFERENCE : AMEN_ITERATIVE_REFERENCE
@test relative_l2(actual, reference) <= AMEN_TOL
@test norm(A_dense * actual - b_dense) / norm(b_dense) <= AMEN_TOL
```

Add one complex iterative problem and compare with its dense solve at `rtol=AMEN_TOL`. Add a test invoking `amenany_sweep` with explicit caches; it is the RED assertion for the new internal keyword interface.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: explicit cache keywords are rejected; existing reference and dense accuracy checks pass.

- [ ] **Step 3: Allocate caches once in `amen_solve`**

Construct one `AmenMatvecCaches`, `EnvironmentUpdateWorkspace`, and
`BiCGSTABCache` after the working element type is known. Pass the same objects
to every sweep through the new `amenany_sweep` keywords. When called directly
without keywords, `amenany_sweep` constructs compatible defaults.

- [ ] **Step 4: Replace hot-path allocations in `amenany_sweep`**

Use cached residual matvecs during right initialization and enrichment, bang
environment reducers in both sweep directions, a prepared solve matvec per
site, and cached BiCGSTAB work for iterative local solves. Reuse the initial
residual when types match, copy each owned local solution before using cache
views, and verify the true residual before the optional preconditioned restart.

- [ ] **Step 5: Verify AMEn accuracy and full existing AMEn tests**

Run the focused AMEn test and then:

```bash
julia --project=. test/runtests.jl
```

Expected: direct, iterative, complex, reference, and true-residual assertions pass; all existing tests through the AMEn-dependent TAMEn tests remain green.

- [ ] **Step 6: Commit AMEn integration**

```bash
git add src/amen_solve.jl test/amen_optimization_tests.jl
git commit -m "perf: reuse scratch across AMEn sweeps"
```

---

### Task 6: Integrate and export the TAMEn workspace without API regression

**Files:**
- Modify: `test/tamen_optimization_tests.jl`
- Modify: `src/amen_utils.jl`
- Modify: `src/tamen.jl:165-548`
- Modify: `src/TTSolver.jl:20-35`

**Interfaces:**
- Consumes: all Task 2-5 caches and the current conservation keyword normalization.
- Produces: exported `TAMEnWorkspace(::Type{T})` and `tamen(...; conserved_observables=nothing, correct_2_norm=nothing, workspace::Union{Nothing,TAMEnWorkspace}=nothing)`.

- [ ] **Step 1: Add failing public workspace and composition tests**

Add assertions that `TAMEnWorkspace` is exported, default calls remain valid, and these calls succeed with identical numerical results:

```julia
work = TAMEnWorkspace(Float64)
plain = tamen(X, A, tol, copy(opts))
cached = tamen(X, A, tol, copy(opts); workspace=work)
conserved = tamen(X, A, tol, copy(opts);
    conserved_observables=observables,
    correct_2_norm=false,
    workspace=work)
```

Retain and rerun every existing invalid-observable, dimension, and keyword-conflict assertion. Add `@test_throws TypeError` for a non-workspace value passed as `workspace`.

- [ ] **Step 2: Add failing snapshot, repeated-call, and ownership tests**

For Chebyshev and CN fixtures assert:

```julia
snapshots = run_tamen(scheme; workspace=work)
reference = scheme == "cheb" ? TAMEN_CHEB_REFERENCE : TAMEN_CN_REFERENCE
@test maximum(relative_l2.(snapshots, reference)) <= TAMEN_TOL
```

Run the same problem twice with one workspace and assert the snapshots agree within `TAMEN_TOL`. Preserve deep copies of the first returned TT cores, run a larger second problem, and assert the first result is unchanged. For both schemes, reuse existing conserved-functional assertions and their `2e-5` absolute threshold. Retain the existing decay analytic threshold `5e-3`.

- [ ] **Step 3: Run focused TAMEn tests and verify RED**

Expected: `TAMEnWorkspace` is undefined or unexported and `workspace` is rejected; all pre-workspace reference tests remain green.

- [ ] **Step 4: Define the grouped workspace and merge the keyword API**

Add:

```julia
struct TAMEnWorkspace{T}
    matvec::AmenMatvecCaches{T}
    environment::EnvironmentUpdateWorkspace{T}
    bicgstab::BiCGSTABCache{T,T}
end

TAMEnWorkspace(::Type{T}) where T
```

Export it from `src/TTSolver.jl`. Merge `workspace` into the current keyword
signature after `correct_2_norm`; do not remove `_normalize_conserved_observables`
or any conflict check.

- [ ] **Step 5: Reuse the grouped workspace across TAMEn sweeps**

After TAMEn working cores and element type are known, use the supplied workspace
or construct one. Pass its three components to every `amenany_sweep` call.
Do not store solution, environment, operator, or Krylov state beyond the active
call except as overwriteable scratch capacity.

- [ ] **Step 6: Verify accuracy, conservation, ownership, and allocation**

Run the focused TAMEn tests. Warm both paths, then assert:

```julia
fresh_bytes = @allocated tamen(X, A, tol, copy(opts))
reused_bytes = @allocated tamen(X, A, tol, copy(opts); workspace=work)
@test reused_bytes < fresh_bytes
```

Expected: both time schemes meet fixture `TAMEN_TOL`, existing analytic and
conservation thresholds pass, earlier results remain unchanged, and compatible
workspace reuse allocates fewer bytes.

- [ ] **Step 7: Commit TAMEn integration**

```bash
git add src/amen_utils.jl src/tamen.jl src/TTSolver.jl test/tamen_optimization_tests.jl
git commit -m "perf: add reusable TAMEn workspace"
```

---

### Task 7: Benchmark and complete package verification

**Files:**
- Modify: `benchmark/bench_tamen.jl`
- Modify: `test/amen_optimization_tests.jl` only if verification exposes a missing regression assertion.
- Modify: `test/tamen_optimization_tests.jl` only if verification exposes a missing regression assertion.

**Interfaces:**
- Consumes: exported `TAMEnWorkspace` and all completed numerical/allocation tests.
- Produces: reproducible fresh-versus-reused benchmark output and a fully verified branch.

- [ ] **Step 1: Extend the benchmark with fresh and reused workspace cases**

For both existing benchmark problems, warm and measure:

```julia
tamen(X, A, tol, copy(opts))
work = TAMEnWorkspace(Float64)
tamen(X, A, tol, copy(opts); workspace=work)

fresh = @benchmark tamen($X, $A, $tol, copy($opts)) samples=10 evals=1
reused = @benchmark tamen($X, $A, $tol, copy($opts); workspace=$work) samples=10 evals=1
```

Print median time, median memory, and allocation count for both. Do not add a
wall-time assertion.

- [ ] **Step 2: Run focused numerical and allocation tests**

Run:

```bash
julia --project=. -e 'using Test, LinearAlgebra, SparseArrays, Random, TTSolver; include("test/amen_tamen_test_utils.jl"); include("test/fixtures/amen_tamen_reference.jl"); include("test/amen_optimization_tests.jl"); include("test/tamen_optimization_tests.jl")'
```

Expected: every test set passes; AMEn and TAMEn reference differences are
`<= tol`; true residuals, analytic solutions, and conserved observables meet
their specified limits; allocation comparisons favor cached paths.

- [ ] **Step 3: Run the complete package suite**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test(; allow_reresolve=false)'
```

Expected: both commands exit successfully with zero failures and no unexpected
warnings. Record the slowest test set and total elapsed time.

- [ ] **Step 4: Run the benchmark and record before/after metrics**

Run:

```bash
julia --project=. benchmark/bench_tamen.jl
```

Expected: the script completes, prints finite numerical results, and reports
fewer allocated bytes for the representative larger TAMEn case than the
pre-migration measurement. Timing is reported without a pass/fail threshold.

- [ ] **Step 5: Check formatting, API remnants, and ignored reference integrity**

Run:

```bash
git diff --check
git status --short
rg -n "function tamen|TAMEnWorkspace|conserved_observables|correct_2_norm|amenany_sweep" src test
sha256sum src_mod/amen_utils.jl src_mod/amen_solve.jl src_mod/tamen.jl src_mod/TTSolver.jl | sort
```

Expected: no whitespace errors; only intended files are modified; the current
TAMEn signature contains all three keywords; the ignored reference hashes remain:

```text
3e0553b362c21114c773efc280b81baf26ab09d7deedde1b35fa3c38fb0930d9  src_mod/amen_solve.jl
c0ce06aa9b958d5b2b0e3a99e01d4df90dd46e9cc15616bebe1ef7e2bac826e8  src_mod/amen_utils.jl
d25919ab1128f88f0ef23d68d49c4667438baf747e727dab80a7424d7305f724  src_mod/tamen.jl
d828f9ead1cbe3c9074df3c383e1fb892464a1760c3d6b2ce2fc57ef0c03ed4f  src_mod/TTSolver.jl
```

- [ ] **Step 6: Commit benchmark and final test adjustments**

```bash
git add benchmark/bench_tamen.jl test/amen_optimization_tests.jl test/tamen_optimization_tests.jl
git commit -m "bench: compare fresh and reused TAMEn workspaces"
```

- [ ] **Step 7: Review the complete branch diff**

Run:

```bash
git diff 4a1121e..HEAD --stat
git log --oneline 4a1121e..HEAD
```

Expected: commits follow Tasks 1-7, no unrelated production files changed, and
the implementation matches the approved spec.
