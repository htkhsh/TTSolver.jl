using TTSolver
using LinearAlgebra
using BenchmarkTools
using Random

Random.seed!(1234)

println("=== Benchmark: tamen function (型推論改善前) ===\n")

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
    opts = Dict(:verb => 0, :nswp => 2, :local_iters => 3, :time_scheme => "cheb", :time_error_damp => 1e3)
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
    
    opts = Dict(:verb => 0, :nswp => 3, :local_iters => 5, :time_scheme => "cheb", :time_error_damp => 1e3)
    return X, A, opts
end

# Warmup
println("Warming up...")
X, A, opts = bench_simple_decay()
tamen(X, A, 1e-4, copy(opts))
X, A, opts = bench_larger_ode()
tamen(X, A, 1e-4, copy(opts))
println("Warmup complete.\n")

# Benchmark simple decay
println("--- Simple Decay ODE (1D, n=4, nt=5) ---")
X, A, opts = bench_simple_decay()
b1 = @benchmark tamen($X, $A, 1e-4, $(copy(opts))) samples=10 evals=1
display(b1)
println("\n")

# Benchmark larger ODE
println("--- Larger ODE (3D, n=4 each, nt=5) ---")
X, A, opts = bench_larger_ode()
b2 = @benchmark tamen($X, $A, 1e-4, $(copy(opts))) samples=10 evals=1
display(b2)
println("\n")

# Memory allocation check
println("--- Memory Allocation Check ---")
X, A, opts = bench_simple_decay()
@time tamen(X, A, 1e-4, copy(opts))
X, A, opts = bench_larger_ode()
@time tamen(X, A, 1e-4, copy(opts))

# Save median times for comparison
println("\n=== Summary ===")
println("Simple Decay - Median time: $(median(b1).time / 1e6) ms")
println("Larger ODE - Median time: $(median(b2).time / 1e6) ms")
