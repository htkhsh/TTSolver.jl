using Test
using LinearAlgebra
using SparseArrays
using Random
using TTSolver

include("amen_tamen_test_utils.jl")
include("fixtures/amen_tamen_reference.jl")
include("amen_optimization_tests.jl")
include("tamen_optimization_tests.jl")

@testset "tt_round preserves tensor scale" begin
    tolerance = 0.1

    @testset "rank-one real tensor" begin
        x = TTTensor([
            reshape([1.0, 1.0], 1, 2, 1),
            reshape([1.0, 1.0, 1.0], 1, 3, 1),
        ])

        @test tt_full(tt_round(x, tolerance)) ≈ tt_full(x) rtol=1e-14 atol=1e-14
    end

    @testset "zero tensor" begin
        x = TTTensor([
            reshape([0.0, 0.0], 1, 2, 1),
            reshape([1.0, 1.0, 1.0], 1, 3, 1),
        ])

        @test all(iszero, tt_full(tt_round(x, tolerance)))
    end

    @testset "rank-one TT matrix" begin
        A = TTMatrix([
            reshape([1.0, 2.0, 3.0, 4.0], 1, 2, 2, 1),
            reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2, 1),
        ])

        @test tt_full(tt_round(A, tolerance)) ≈ tt_full(A) rtol=1e-14 atol=1e-14
    end

    @testset "rank-one complex tensor" begin
        x = TTTensor([
            reshape(ComplexF64[1 + 2im, -0.5im], 1, 2, 1),
            reshape(ComplexF64[2 - im, -1 + 0.25im], 1, 2, 1),
        ])

        @test tt_full(tt_round(x, tolerance)) ≈ tt_full(x) rtol=1e-14 atol=1e-14
    end
end

@testset "in-place local matvec" begin
    rx1, m, rx2, b = 2, 3, 2, 2
    rw1, n, rw2 = 2, 2, 2
    ra1 = [1, 2]
    ra2 = [2, 1]
    Ra = length(ra1)

    x = reshape(ComplexF64.(1:(rx1 * m * rx2 * b)), :, b) ./ 7
    WAX1 = [
        reshape(ComplexF64.(1:(rw1 * rx1 * ra1[k])), rw1, rx1, ra1[k]) ./ (3 + k)
        for k in 1:Ra
    ]
    A = [
        reshape(ComplexF64.(1:(ra1[k] * n * m * ra2[k])), ra1[k] * n, m * ra2[k]) ./ (5 + k)
        for k in 1:Ra
    ]
    WAX2 = [
        reshape(ComplexF64.(1:(rw2 * rx2 * ra2[k])), rw2, rx2, ra2[k]) ./ (7 + k)
        for k in 1:Ra
    ]

    expected = zeros(ComplexF64, rw1, n, rw2, b)
    x4 = reshape(x, rx1, m, rx2, b)
    for k in 1:Ra, a in 1:rw1, ni in 1:n, c in 1:rw2, bi in 1:b,
        i in 1:rx1, mi in 1:m, j in 1:rx2, p in 1:ra1[k], q in 1:ra2[k]
        A4 = reshape(A[k], ra1[k], n, m, ra2[k])
        expected[a, ni, c, bi] +=
            WAX1[k][a, i, p] * A4[p, ni, mi, q] * x4[i, mi, j, bi] * WAX2[k][c, j, q]
    end

    workspace_defined = isdefined(TTSolver, :LocalMatvecWorkspace)
    matvec_defined = isdefined(TTSolver, :local_matvec!)
    @test workspace_defined
    @test matvec_defined
    if workspace_defined && matvec_defined
        workspace = TTSolver.LocalMatvecWorkspace(
            x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, ra1, ra2,
        )
        result = similar(x, rw1 * n * rw2, b)
        returned = TTSolver.local_matvec!(result, x, workspace)

        @test returned === result
        @test reshape(result, rw1, n, rw2, b) ≈ expected rtol=1e-13
        @test TTSolver.local_matvec(
            x, rx1, m, rx2, b, rw1, n, rw2,
            WAX1, A, WAX2, Ra, ra1, ra2,
        ) ≈ result rtol=1e-13

        sparse_workspace = TTSolver.LocalMatvecWorkspace(
            x, rx1, m, rx2, b, rw1, n, rw2,
            WAX1, sparse.(A), WAX2, ra1, ra2,
        )
        sparse_result = similar(result)
        TTSolver.local_matvec!(sparse_result, x, sparse_workspace)
        @test sparse_result ≈ result rtol=1e-13

        TTSolver.local_matvec!(result, x, workspace)
        allocated = @allocated begin
            for _ in 1:10
                TTSolver.local_matvec!(result, x, workspace)
            end
        end
        @test allocated <= 4096
    end
end

@testset "in-place BiCGSTAB" begin
    A = ComplexF64[4 1-1im; 2+1im 3]
    rhs = ComplexF64[1, 2-1im]
    mvfun! = (dest, src) -> mul!(dest, A, src)

    solver_defined = isdefined(TTSolver, :bicgstab_inplace)
    @test solver_defined
    if solver_defined
        solution, flag, residual, _ = TTSolver.bicgstab_inplace(
            mvfun!, rhs; tol=1e-12, maxiter=20,
        )
        @test flag == 0
        @test residual <= 1e-12
        @test solution ≈ A \ rhs rtol=1e-12
    end
end

@testset "in-place BiCGSTAB reuses matvec outputs" begin
    A = ComplexF64[
        4 1 0 0 0 1;
        2 5 1 0 0 0;
        0 1 6 1 0 0;
        0 0 2 7 1 0;
        0 0 0 1 8 1;
        1 0 0 0 2 9
    ]
    rhs = ComplexF64[1, 2, 3, 4, 5, 6]
    destinations = Any[]
    tracked_mvfun! = function (dest, src)
        push!(destinations, dest)
        return mul!(dest, A, src)
    end

    TTSolver.bicgstab_inplace(tracked_mvfun!, rhs; tol=0.0, maxiter=3)

    @test length(destinations) > 2
    @test length(unique(objectid.(destinations))) <= 2
end

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

function keyword_tamen_problem()
    state = TTTensor([reshape([1.0, 0.0], 1, 2, 1)])
    space_time = tkron(state, tt_ones(3))
    generator = TTMatrix([reshape(zeros(2, 2), 1, 2, 2, 1)])
    observable = tt_ones(2)
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 1,
        :local_iters => 2,
        :time_scheme => "CN",
    )
    return space_time, generator, observable, options
end

@testset "tamen conserved-observable keyword validation" begin
    X, A, observable, options = keyword_tamen_problem()

    result = tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=TTTensor[],
        correct_2_norm=false,
    )
    @test length(result) == 5
    legacy_empty = merge(copy(options), Dict(:obs => [observable]))
    @test length(tamen(
        X, A, 1e-5, legacy_empty;
        conserved_observables=TTTensor[],
    )) == 5

    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=observable,
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=[ones(2)],
    )
    @test_throws DimensionMismatch tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=[tt_ones(3)],
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, merge(copy(options), Dict(:obs => [observable]));
        conserved_observables=[observable],
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5,
        merge(copy(options), Dict(:correct_2_norm => false));
        correct_2_norm=false,
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        correct_2_norm=1,
    )
end

function conservative_tamen_problem()
    generator_dense = [
        -1.0  0.5  0.0  0.0;
         1.0 -0.5  0.0  0.0;
         0.0  0.0 -0.2  0.4;
         0.0  0.0  0.2 -0.4
    ]
    initial_dense = [0.8, 0.2, 0.3, 0.7]
    initial = TTTensor([reshape(initial_dense, 1, 4, 1)])
    generator = TTMatrix([reshape(generator_dense, 1, 4, 4, 1)])
    first_mass = TTTensor([reshape([1.0, 1.0, 0.0, 0.0], 1, 4, 1)])
    second_mass = TTTensor([reshape([0.0, 0.0, 1.0, 1.0], 1, 4, 1)])
    return initial, generator, [first_mass, second_mass]
end

@testset "tamen conserved linear functionals" begin
    initial, generator, observables = conservative_tamen_problem()
    initial_values = [tt_dot(observable, initial) for observable in observables]

    for scheme in ("cheb", "CN")
        Random.seed!(20260922)
        X = tkron(initial, tt_ones(5))
        options = Dict{Symbol,Any}(
            :verb => 0,
            :nswp => 4,
            :local_iters => 8,
            :time_scheme => scheme,
            :time_error_damp => 1e3,
        )
        U, time_grid, _, _, _ = tamen(
            X, generator, 1e-6, options;
            conserved_observables=observables,
            correct_2_norm=false,
        )
        snapshots = extract_snapshot(U, time_grid, [0.0, 0.5, 1.0], scheme)
        for snapshot in snapshots, (observable, initial_value) in zip(observables, initial_values)
            @test tt_dot(observable, snapshot) ≈ initial_value atol=2e-5
        end
        @test abs(norm(last(snapshots)) - norm(initial)) > 1e-3
    end
end

@testset "tamen legacy observable option" begin
    initial, generator, observables = conservative_tamen_problem()
    X = tkron(initial, tt_ones(5))
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 4,
        :local_iters => 8,
        :time_scheme => "CN",
        :time_error_damp => 1e3,
        :obs => [first(observables)],
        :correct_2_norm => false,
    )
    U, time_grid, _, _, _ = tamen(X, generator, 1e-6, options)
    final_state = extract_snapshot(U, time_grid, 1.0, "CN")
    @test tt_dot(first(observables), final_state) ≈
          tt_dot(first(observables), initial) atol=2e-5
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
