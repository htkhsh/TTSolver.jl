using Test
using LinearAlgebra
using TTSolver

@testset "basic TT constructors" begin
    x = tt_ones([2, 3])
    @test tt_dims(x) == [2, 3]
    @test norm(x) ≈ sqrt(6)

    u = tt_unit([2, 3], 4)
    full_u = vec(tt_full(u))
    @test count(!iszero, full_u) == 1
    @test full_u[4] == 1

    v = tt_x([2, 3])
    full_v = vec(tt_full(v))
    @test full_v == collect(0.0:5.0)
end

@testset "TT ops" begin
    a = tt_unit([3], 2)
    b = tt_unit([3], 3)
    c = a + b
    full_c = vec(tt_full(c))
    @test full_c == [0.0, 1.0, 1.0]

    M = [1.0 2.0 3.0; 0.0 1.0 4.0; 5.0 6.0 0.0]
    A = TTMatrix([reshape(M, 1, 3, 3, 1)])
    x = TTTensor([reshape([1.0, 2.0, 3.0], 1, 3, 1)])
    y = A * x
    full_y = vec(tt_full(y))
    @test full_y == M * [1.0, 2.0, 3.0]

    N = M + Matrix{Float64}(I, 3, 3)
    B = TTMatrix([reshape(N, 1, 3, 3, 1)])
    C = A * B
    y1 = C * x
    y2 = A * (B * x)
    full_y1 = vec(tt_full(y1))
    full_y2 = vec(tt_full(y2))
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
    full = vec(tt_full(x_snap))
    @test full == [2.0, 2.0]
end

@testset "tamen zero ode" begin
    x0 = tt_unit([2], 1)
    X = tkron(x0, tt_ones(3))
    A = TTMatrix([reshape(zeros(2, 2), 1, 2, 2, 1)])
    opts = Dict(:verb => 0, :nswp => 2, :local_iters => 2, :time_scheme => "cheb", :time_error_damp => 1e3)
    U, t, opts, x = tamen(X, A, 1e-6, opts)
    x_end = extract_snapshot(U, t, 1.0, "cheb")
    full_end = vec(tt_full(x_end))
    full0 = vec(tt_full(x0))
    @test full_end ≈ full0 atol=1e-4
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
    full_end = vec(tt_full(x_end))
    full0 = vec(tt_full(x0))
    expected = exp(-alpha) .* full0
    @test full_end ≈ expected atol=5e-3
end

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
