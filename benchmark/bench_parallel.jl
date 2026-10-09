using TTSolver
using LinearAlgebra
using Random

Random.seed!(1234)

println("=== Quick Parallel Benchmark ===")
println("Threads: ", Threads.nthreads(), "\n")

# Small fast test (Ra=1)
function setup_small()
    d = 2
    n_each = 4
    x0 = TTTensor([randn(1, n_each, 2), randn(2, n_each, 1)])
    X = Matrix{Any}(undef, d + 1, 1)
    for i in 1:d
        core = x0.cores[i]
        r1, ni, r2 = size(core)
        X[i, 1] = reshape(core, r1, ni, 1, r2)
    end
    X[d + 1, 1] = reshape(ones(3), 1, 3, 1, 1)
    
    A_cores = [reshape(-0.1 .* Matrix{Float64}(I, n_each, n_each), 1, n_each, n_each, 1) for _ in 1:d]
    A = TTMatrix(A_cores)
    opts = Dict(:verb => 0, :nswp => 2, :local_iters => 3, :time_scheme => "cheb", :time_error_damp => 1e3)
    return X, A, opts
end

# Larger with high Ra
function setup_large_ra()
    d = 3
    n_each = 6
    x0 = TTTensor([randn(1, n_each, 3), randn(3, n_each, 3), randn(3, n_each, 1)])
    X = Matrix{Any}(undef, d + 1, 1)
    for i in 1:d
        core = x0.cores[i]
        r1, ni, r2 = size(core)
        X[i, 1] = reshape(core, r1, ni, 1, r2)
    end
    X[d + 1, 1] = reshape(ones(5), 1, 5, 1, 1)
    
    Ra = 10
    A = Matrix{Any}(undef, d, Ra)
    for k in 1:Ra, i in 1:d
        A[i, k] = reshape(randn(n_each, n_each) * 0.1, 1, n_each, n_each, 1)
    end
    opts = Dict(:verb => 0, :nswp => 2, :local_iters => 5, :time_scheme => "cheb", :time_error_damp => 1e3)
    return X, A, opts
end

println("Warmup...")
X, A, opts = setup_small()
tamen(X, A, 1e-3, copy(opts))
X, A, opts = setup_large_ra()
tamen(X, A, 1e-3, copy(opts))
println("Done.\n")

println("--- Small (Ra=1) ---")
for i in 1:3
    local X, A, opts = setup_small()
    @time tamen(X, A, 1e-3, copy(opts))
end

println("\n--- Large Ra (Ra=10) ---")
for i in 1:3
    local X, A, opts = setup_large_ra()
    @time tamen(X, A, 1e-3, copy(opts))
end
