using TTSolver
using LinearAlgebra
using BenchmarkTools

println("=== Quick Type Stability Check ===\n")

# 小さいテストケース
function setup_test()
    alpha = 0.5
    n = 4
    x0 = tt_unit([n], 2)
    X = Matrix{Any}(undef, 2, 1)
    core = x0.cores[1]
    X[1, 1] = reshape(core, size(core, 1), size(core, 2), 1, size(core, 3))
    X[2, 1] = reshape(ones(5), 1, 5, 1, 1)
    A = TTMatrix([reshape(-alpha .* Matrix{Float64}(I, n, n), 1, n, n, 1)])
    opts = Dict(:verb => 0, :nswp => 1, :local_iters => 2, :time_scheme => "cheb", :time_error_damp => 1e3)
    return X, A, opts
end

# Warmup
println("Warming up...")
X, A, opts = setup_test()
tamen(X, A, 1e-4, copy(opts))
println("Done.\n")

# 時間計測
println("--- Timing (1 sweep) ---")
X, A, opts = setup_test()
@time tamen(X, A, 1e-4, copy(opts))
@time tamen(X, A, 1e-4, copy(opts))
@time tamen(X, A, 1e-4, copy(opts))

println("\n--- Benchmark ---")
b = @benchmark tamen($X, $A, 1e-4, $(copy(opts))) samples=50 evals=1
display(b)
println("\nMedian: $(median(b).time / 1e6) ms")
