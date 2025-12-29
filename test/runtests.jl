using Test
using LinearAlgebra
using TTSolver

@testset "basic TT constructors" begin
    x = tt_ones([2, 3])
    @test tt_dims(x) == [2, 3]
    @test norm(x) ≈ sqrt(6)

    u = tt_unit([2, 3], 4)
    full_u = reshape(expand_dR(u), :, 1)
    @test count(!iszero, full_u) == 1
    @test full_u[4] == 1

    v = tt_x([2, 3])
    full_v = reshape(expand_dR(v), :, 1)
    @test vec(full_v) == collect(0.0:5.0)
end

@testset "TT ops" begin
    a = tt_unit([3], 2)
    b = tt_unit([3], 3)
    c = a + b
    full_c = reshape(expand_dR(c), :, 1)
    @test vec(full_c) == [0.0, 1.0, 1.0]

    M = [1.0 2.0 3.0; 0.0 1.0 4.0; 5.0 6.0 0.0]
    A = TTMatrix([reshape(M, 1, 3, 3, 1)])
    x = TTTensor([reshape([1.0, 2.0, 3.0], 1, 3, 1)])
    y = A * x
    full_y = reshape(expand_dR(y), :, 1)
    @test vec(full_y) == M * [1.0, 2.0, 3.0]

    N = M + Matrix{Float64}(I, 3, 3)
    B = TTMatrix([reshape(N, 1, 3, 3, 1)])
    C = A * B
    y1 = C * x
    y2 = A * (B * x)
    full_y1 = reshape(expand_dR(y1), :, 1)
    full_y2 = reshape(expand_dR(y2), :, 1)
    @test full_y1 == full_y2
end

@testset "time utils" begin
    t, S = chebdiff(5)
    @test length(t) == 5
    @test size(S) == (5, 5)
    @test minimum(t) >= 0 && maximum(t) <= 1

    p = cheb2_interpolant(t, collect(range(0.0, 1.0; length=7)))
    @test size(p) == (7, 5)
end

@testset "extract snapshot" begin
    X = Matrix{Any}(undef, 2, 1)
    X[1, 1] = reshape([1.0, 1.0], 1, 2, 1, 1)
    X[2, 1] = reshape([1.0, 2.0, 3.0], 1, 3)
    t = [collect(0.0:0.5:1.0)]
    x_snap = extract_snapshot(X, t, 0.5, "cn")
    full = reshape(expand_dR(x_snap), :, 1)
    @test vec(full) == [2.0, 2.0]
end

@testset "tamen zero ode" begin
    x0 = tt_unit([2], 1)
    X = tkron(x0, tt_ones(3))
    A = TTMatrix([reshape(zeros(2, 2), 1, 2, 2, 1)])
    opts = Dict(:verb => 0, :nswp => 2, :local_iters => 2, :time_scheme => "cheb", :time_error_damp => 1e3)
    U, t, opts, x = tamen(X, A, 1e-6, opts)
    x_end = extract_snapshot(U, t, 1.0, "cheb")
    full_end = reshape(expand_dR(x_end), :, 1)
    full0 = reshape(expand_dR(x0), :, 1)
    @test vec(full_end) ≈ vec(full0) atol=1e-4
end

@testset "tamen decay ode" begin
    alpha = 0.5
    x0 = tt_unit([2], 2)
    X = Matrix{Any}(undef, 2, 1)
    core = x0.cores[1]
    X[1, 1] = reshape(core, size(core, 1), size(core, 2), 1, size(core, 3))
    X[2, 1] = reshape(ones(5), 1, 5, 1, 1)
    A = TTMatrix([reshape(-alpha .* Matrix{Float64}(I, 2, 2), 1, 2, 2, 1)])
    opts = Dict(:verb => 0, :nswp => 1, :local_iters => 3, :time_scheme => "cheb", :time_error_damp => 1e3)
    U, t, opts, x = tamen(X, A, 1e-6, opts)
    x_end = extract_snapshot(U, t, 1.0, "cheb")
    full_end = reshape(expand_dR(x_end), :, 1)
    full0 = reshape(expand_dR(x0), :, 1)
    expected = exp(-alpha) .* full0
    @test vec(full_end) ≈ vec(expected) atol=5e-3
end
