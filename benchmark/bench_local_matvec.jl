using TTSolver: local_matvec
using LinearAlgebra
using Random
using TensorOperations

Random.seed!(42)

# local_matvecの直接テスト
function setup_local_matvec(; rx1=10, m=8, rx2=10, rw1=10, n=8, rw2=10, Ra=8, ra_val=2)
    T = Float64
    ra1 = fill(ra_val, Ra)
    ra2 = fill(ra_val, Ra)
    
    x = randn(T, rx1 * m * rx2, 1)
    WAX1 = [randn(T, rw1, rx1, ra1[k]) for k in 1:Ra]
    WAX2 = [randn(T, rw2, rx2, ra2[k]) for k in 1:Ra]
    A = [randn(T, ra1[k], n, m, ra2[k]) for k in 1:Ra]
    
    return x, rx1, m, rx2, 1, rw1, n, rw2, WAX1, A, WAX2, Ra, ra1, ra2
end

# 並列版（シンプル）
function local_matvec_parallel(x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, Ra, ra1, ra2)
    T = eltype(x)
    x4 = reshape(x, rx1, m, rx2, b)
    
    # 各スレッドの結果を格納
    results = Vector{Array{T,4}}(undef, Ra)
    
    Threads.@threads for k in 1:Ra
        WAX1k = WAX1[k]
        WAX2k = WAX2[k]
        A4 = ndims(A[k]) == 4 ? A[k] : reshape(A[k], ra1[k], n, m, ra2[k])
        wk = zeros(T, rw1, n, rw2, b)
        @tensor wk[rw1i, n_i, rw2i, bi] = WAX1k[rw1i, rx1i, ra1i] * A4[ra1i, n_i, m_i, ra2i] * x4[rx1i, m_i, rx2i, bi] * WAX2k[rw2i, rx2i, ra2i]
        results[k] = wk
    end
    
    w = results[1]
    for k in 2:Ra
        w .+= results[k]
    end
    return reshape(w, rw1 * n * rw2, b)
end

println("Threads: ", Threads.nthreads())

# Medium case (Ra=8)  
println("\n--- Medium: Ra=8, rx=10, n=8 ---")
args = setup_local_matvec(rx1=10, m=8, rx2=10, rw1=10, n=8, rw2=10, Ra=8, ra_val=2)

# warmup
local_matvec(args...)
local_matvec_parallel(args...)

println("Original:")
@time for _ in 1:500; local_matvec(args...); end

println("Parallel:")
@time for _ in 1:500; local_matvec_parallel(args...); end

# Large case (Ra=32, smaller matrices)
println("\n--- Large Ra: Ra=32, rx=8, n=8 ---")
args = setup_local_matvec(rx1=8, m=8, rx2=8, rw1=8, n=8, rw2=8, Ra=32, ra_val=1)

local_matvec(args...)
local_matvec_parallel(args...)

println("Original:")
@time for _ in 1:200; local_matvec(args...); end

println("Parallel:")
@time for _ in 1:200; local_matvec_parallel(args...); end
