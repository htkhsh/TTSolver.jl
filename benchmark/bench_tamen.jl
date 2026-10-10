using TTSolver
using LinearAlgebra
using BenchmarkTools
using Random

Random.seed!(1234)

println("=== Benchmark: fresh and reused TAMEn workspaces ===\n")

# Simple decay ODE benchmark
function bench_simple_decay()
    alpha = 0.5
    n = 4  # dimension
    x0 = tt_unit([n], 2)
    X = Matrix{Any}(undef, 2, 1)
    core = x0.cores[1]
    X[1, 1] = reshape(core, size(core, 1), size(core, 2), 1, size(core, 3))
    X[2, 1] = reshape(ones(5), 1, 5, 1, 1)
    A = TTMatrix([reshape(-alpha .* Matrix{Float64}(I, n, n), 1, n, n, 1)])
    opts = Dict(:verb => 0, :nswp => 1, :local_iters => 1, :time_scheme => "cheb", :time_error_damp => 0.0)
    return X, A, opts
end

# Larger ODE benchmark
function bench_larger_ode()
    d = 3  # number of dimensions
    n_each = 4
    n = fill(n_each, d)
    
    # Create initial condition
    x0_cores = [randn(1, n_each, 2), randn(2, n_each, 2), randn(2, n_each, 1)]
    x0 = TTTensor(x0_cores)
    
    # Setup for tamen
    X = Matrix{Any}(undef, d + 1, 1)
    for i in 1:d
        core = x0.cores[i]
        r1, ni, r2 = size(core)
        X[i, 1] = reshape(core, r1, ni, 1, r2)
    end
    X[d + 1, 1] = reshape(ones(5), 1, 5, 1, 1)
    
    # Create diagonal A matrix (decay)
    A_cores = Vector{Array{Float64, 4}}(undef, d)
    for i in 1:d
        A_cores[i] = reshape(-0.1 .* Matrix{Float64}(I, n_each, n_each), 1, n_each, n_each, 1)
    end
    A = TTMatrix(A_cores)
    
    opts = Dict(:verb => 0, :nswp => 1, :local_iters => 1, :time_scheme => "cheb", :time_error_damp => 0.0)
    return X, A, opts
end

function benchmark_workspace_case(label, builder; tol=1e-4)
    X, A, opts = builder()
    work = TAMEnWorkspace(Float64)

    fresh_result = tamen(deepcopy(X), A, tol, copy(opts))
    reused_result = tamen(
        deepcopy(X), A, tol, copy(opts); workspace=work,
    )
    fresh_norm = norm(tt_full(extract_snapshot(
        fresh_result[1], fresh_result[2], 1.0, opts[:time_scheme],
    )))
    reused_norm = norm(tt_full(extract_snapshot(
        reused_result[1], reused_result[2], 1.0, opts[:time_scheme],
    )))
    @assert isfinite(fresh_norm) && isfinite(reused_norm)

    fresh = @benchmark tamen(
        deepcopy($X), $A, $tol, copy($opts),
    ) samples=3 evals=1
    reused = @benchmark tamen(
        deepcopy($X), $A, $tol, copy($opts); workspace=$work,
    ) samples=3 evals=1
    fresh_median = median(fresh)
    reused_median = median(reused)

    println("--- $label ---")
    println("final norm: fresh=$fresh_norm, reused=$reused_norm")
    println(
        "fresh: time=$(fresh_median.time / 1e6) ms, " *
        "memory=$(fresh_median.memory) bytes, " *
        "allocations=$(fresh_median.allocs)",
    )
    println(
        "reused: time=$(reused_median.time / 1e6) ms, " *
        "memory=$(reused_median.memory) bytes, " *
        "allocations=$(reused_median.allocs)",
    )
    println()
    return (; fresh, reused, fresh_norm, reused_norm)
end

simple = benchmark_workspace_case(
    "Simple Decay ODE (1D, n=4, nt=5)", bench_simple_decay,
)
larger = benchmark_workspace_case(
    "Larger ODE (3D, n=4 each, nt=5)", bench_larger_ode,
)
