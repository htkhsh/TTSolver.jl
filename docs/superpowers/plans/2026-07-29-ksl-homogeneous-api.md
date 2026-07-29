# Homogeneous KSL API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make homogeneous KSL the only supported `tt_ksl` API and replace `passes` with `symmetric::Bool=true`.

**Architecture:** Keep the existing homogeneous projector-splitting core, but make it the primary method instead of routing through a zero source tensor. Select the step size and reverse sweep from the Boolean `symmetric` option, and delete every source-term environment and contraction. Lock the new API and the existing homogeneous numerical behavior with dense-reference tests.

**Tech Stack:** Julia 1.x, TTSolver TT types, LinearAlgebra matrix exponentials, ExponentialUtilities, Julia `Test`.

## Global Constraints

- Support only `dx/dt + A*x = 0`.
- Expose `tt_ksl(x0::TTTensor, A::TTMatrix, tau::Real; symmetric::Bool=true, ...)`.
- Remove the nonhomogeneous overload rather than deprecating it.
- Remove the `passes` keyword rather than deprecating it.
- `symmetric=true` means a left-to-right `tau/2` sweep followed by a right-to-left `tau/2` sweep.
- `symmetric=false` means one left-to-right sweep using `tau`.
- Do not change rank adaptation, exponential algorithms, or sweep direction.

---

### Task 1: Homogeneous KSL API and numerical contract

**Files:**
- Modify: `test/runtests.jl`
- Modify: `src/tt_ksl.jl`

**Interfaces:**
- Consumes: `TTTensor`, `TTMatrix`, `tt_full`, and `tt_mkron` from the existing TTSolver API.
- Produces: `tt_ksl(x0::TTTensor, A::TTMatrix, tau::Real; tol=1e-12, rmax=typemax(Int), symmetric=true, expmethod=:krylov, krylov_m=30)::TTTensor`.

- [ ] **Step 1: Add failing tests for the new Boolean API and removed APIs**

Append this test set to `test/runtests.jl`:

```julia
@testset "homogeneous KSL API" begin
    A_dense = [1.0 0.2; -0.1 0.7]
    x_dense = [0.3, -0.8]
    tau = 0.2

    A = TTMatrix([reshape(A_dense, 1, 2, 2, 1)])
    x0 = TTTensor([reshape(x_dense, 1, 2, 1)])

    for symmetric in (false, true)
        x = tt_ksl(x0, A, tau; symmetric, expmethod=:taylor)
        @test vec(tt_full(x)) ≈ exp(-tau * A_dense) * x_dense rtol=1e-12
    end

    y = TTTensor([reshape([0.4, 0.1], 1, 2, 1)])
    @test_throws MethodError tt_ksl(x0, A, y, tau)
    @test_throws MethodError tt_ksl(x0, A, tau; passes=2)
end
```

The new keyword is the production change that makes the first loop pass. Removing the old method and keyword makes the final two assertions pass.

- [ ] **Step 2: Run the focused test and verify the expected failures**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: the new KSL test set fails because `symmetric` is forwarded to the old nonhomogeneous method, while the nonhomogeneous signature and `passes` are still accepted. Existing earlier test sets should pass.

- [ ] **Step 3: Convert `tt_ksl` into the primary homogeneous method**

Change the docstring signature and equation to:

```julia
"""
    tt_ksl(x0, A, tau; kwargs...)

Homogeneous linear KSL (projector-splitting) time integrator in TT format.

Integrates `dx/dt + A*x = 0` over one time step of length `tau`.

With `symmetric=true`, performs a left-to-right half-sweep followed by a
right-to-left half-sweep. With `symmetric=false`, performs one full
left-to-right sweep.
"""
```

Replace the method header with:

```julia
function tt_ksl(x0::TTTensor, A::TTMatrix, tau::Real;
                tol::Real=1e-12, rmax::Int=typemax(Int),
                symmetric::Bool=true, expmethod::Symbol=:krylov,
                krylov_m::Int=30)
```

Remove `ry`, `cry`, and all `phiy` initialization. Define the sweep step as:

```julia
tau_sweep = symmetric ? tau / 2 : tau
```

Use `tau_sweep` in every local core and bond exponential. Replace:

```julia
if passes > 1
```

with:

```julia
if symmetric
```

- [ ] **Step 4: Remove all source-term updates from both sweeps**

Delete the calls to `compute_rhs_ksl`, the `rhs` and `rhss` variables, and these source updates:

```julia
sol = sol .+ tau_pass .* rhs
S = S .- tau_pass .* rhss
```

The homogeneous updates must have these forms:

```julia
sol = expv_local(x -> -tau_sweep * bfun3(Phi1, A1, Phi2, x), sol_prev)
S = expv_local(x -> tau_sweep * bfun2(phia[i + 1], Phi2, x), S_prev_vec)
```

and, in the reverse sweep:

```julia
sol = expv_local(x -> -tau_sweep * bfun3(Phi1, A1, Phi2, x), sol_prev)
S = expv_local(x -> tau_sweep * bfun2(Phi1, phia[i], x), S_prev_vec)
```

Delete every `compute_next_Phi_y` update.

- [ ] **Step 5: Delete obsolete nonhomogeneous helpers and wrapper**

Delete:

```julia
compute_next_Phi_y(...)
compute_rhs_ksl(...)
tt_ksl(x0::TTTensor, A::TTMatrix, tau::Real; kwargs...)
```

The last item is the old zero-source convenience wrapper; the method created in Step 3 replaces it. Keep the deprecated `tt_ksl_ml` alias because it points to the homogeneous function name and is outside the requested API removal.

- [ ] **Step 6: Run the focused test and verify it passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all test sets pass, including both symmetric modes and both removed-API assertions.

- [ ] **Step 7: Add complex multi-site regression coverage**

Add this helper and test set to `test/runtests.jl`:

```julia
function dense_two_site_tt(v::AbstractVector, n1::Int, n2::Int)
    M = permutedims(reshape(v, n2, n1))
    F = svd(M)
    r = length(F.S)
    return TTTensor([
        reshape(F.U * Diagonal(F.S), 1, n1, r),
        reshape(F.Vt, r, n2, 1),
    ])
end

@testset "complex homogeneous KSL" begin
    H1 = ComplexF64[0.3 0.2im; -0.2im -0.1]
    H2 = ComplexF64[0.7 0.1; 0.1 -0.4]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    A = im * (tt_mkron(H1, I2) + tt_mkron(I2, H2))
    A_dense = im * (kron(H1, I2) + kron(I2, H2))
    x_dense = ComplexF64[0.2 + 0.1im, -0.3im, 0.7, -0.4 + 0.2im]
    x0 = dense_two_site_tt(x_dense, 2, 2)
    tau = 0.15

    x = tt_ksl(x0, A, tau; symmetric=true, expmethod=:taylor)
    result = vec(tt_full(x))

    @test result ≈ exp(-tau * A_dense) * x_dense rtol=1e-11
    @test norm(result) ≈ norm(x_dense) rtol=1e-11
end
```

These assertions catch incorrect complex contractions, incorrect half-step scaling, or omission of the reverse sweep.

- [ ] **Step 8: Run KSL tests and the full package test suite**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test(; allow_reresolve=false)'
```

Expected: both commands exit successfully and report all tests passing.

- [ ] **Step 9: Check formatting, removed symbols, and repository callers**

Run:

```bash
git diff --check
rg -n "passes|compute_next_Phi_y|compute_rhs_ksl|tt_ksl\\([^,]+,[^,]+,[^,]+,[^,]+"
```

Expected: `git diff --check` prints nothing. The search finds no removed implementation or caller; occurrences in the committed design and plan documents are expected and should be inspected rather than edited.

- [ ] **Step 10: Commit the implementation**

```bash
git add src/tt_ksl.jl test/runtests.jl
git commit -m "refactor: restrict KSL to homogeneous evolution"
```

