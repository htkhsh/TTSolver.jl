@testset "pre-migration AMEn reference" begin
    Random.seed!(20261010)
    @test relative_l2(run_amen(:direct), AMEN_DIRECT_REFERENCE) <= 10eps(Float64)

    Random.seed!(20261010)
    @test relative_l2(run_amen(:iterative), AMEN_ITERATIVE_REFERENCE) <= 10eps(Float64)
end

function local_matvec_case(::Type{T}; b::Int=1, sparse_operator::Bool=false,
    identity_operator::Bool=false, mode_size::Int=2) where {T}
    rx1, rx2 = 2, 2
    rw1, rw2 = 2, 2
    m = n = mode_size
    ra1 = [1]
    ra2 = [1]

    x = reshape(T.(1:(rx1 * m * rx2 * b)), rx1 * m * rx2, b) ./ T(7)
    if T <: Complex
        x .+= im .* reverse(x; dims=1) ./ T(11)
    end
    WAX1 = [reshape(T.(1:(rw1 * rx1)), rw1, rx1, 1) ./ T(5)]
    WAX2 = [reshape(T.(1:(rw2 * rx2)), rw2, rx2, 1) ./ T(6)]
    dense_operator = identity_operator ?
        Matrix{T}(I, n, m) :
        reshape(T.(1:(n * m)), n, m) ./ T(9)
    if T <: Complex && !identity_operator
        dense_operator .+= im .* reverse(dense_operator; dims=1) ./ T(13)
    end
    operators = [sparse_operator ? sparse(dense_operator) : dense_operator]

    return (;
        x, rx1, m, rx2, b, rw1, n, rw2,
        WAX1, operators, WAX2, ra1, ra2,
    )
end

function allocating_local_matvec(case)
    return TTSolver.local_matvec(
        case.x, case.rx1, case.m, case.rx2, case.b,
        case.rw1, case.n, case.rw2,
        case.WAX1, case.operators, case.WAX2,
        length(case.operators), case.ra1, case.ra2,
    )
end

function cached_local_matvec!(cache, case; x=case.x)
    return TTSolver._cached_local_matvec!(
        cache, x, case.rx1, case.m, case.rx2, case.b,
        case.rw1, case.n, case.rw2,
        case.WAX1, case.operators, case.WAX2,
        case.ra1, case.ra2,
    )
end

@testset "cached local matvec" begin
    cases = [
        local_matvec_case(Float64; b=1),
        local_matvec_case(ComplexF64; b=2),
        local_matvec_case(Float64; b=2, sparse_operator=true),
        local_matvec_case(Float64; b=1, identity_operator=true),
    ]

    for case in cases
        expected = allocating_local_matvec(case)
        cache = TTSolver.LocalMatvecCache(eltype(case.x))
        actual = copy(cached_local_matvec!(cache, case))
        @test actual ≈ expected rtol=1e-13 atol=1e-13
        @test eltype(actual) === eltype(expected)
    end

    contiguous = local_matvec_case(Float64; b=2)
    storage = zeros(Float64, 2size(contiguous.x, 1), contiguous.b)
    noncontiguous = @view storage[1:2:end, :]
    copyto!(noncontiguous, contiguous.x)
    cache = TTSolver.LocalMatvecCache(Float64)
    @test copy(cached_local_matvec!(cache, contiguous; x=noncontiguous)) ≈
          copy(cached_local_matvec!(cache, contiguous; x=copy(noncontiguous)))

    small = local_matvec_case(Float64; b=1)
    owning_small = copy(cached_local_matvec!(cache, small))
    saved_small = copy(owning_small)
    larger = local_matvec_case(Float64; b=2, mode_size=4)
    cached_local_matvec!(cache, larger)
    @test owning_small == saved_small

    @test_throws DimensionMismatch cached_local_matvec!(
        cache, small; x=view(small.x, 1:(size(small.x, 1) - 1), :),
    )

    mismatched_terms = merge(small, (; WAX2=Any[]))
    @test_throws DimensionMismatch cached_local_matvec!(cache, mismatched_terms)

    bad_environment = merge(small, (; WAX1=[zeros(3, small.rx1, 1)]))
    @test_throws DimensionMismatch cached_local_matvec!(cache, bad_environment)

    workspace = TTSolver.prepare_local_matvec!(
        cache, small.x, small.rx1, small.m, small.rx2, small.b,
        small.rw1, small.n, small.rw2,
        small.WAX1, small.operators, small.WAX2, small.ra1, small.ra2,
    )
    result = similar(allocating_local_matvec(small))
    TTSolver.local_matvec!(result, small.x, workspace)
    allocated = @allocated begin
        for _ in 1:10
            TTSolver.local_matvec!(result, small.x, workspace)
        end
    end
    @test allocated <= 4096
end
