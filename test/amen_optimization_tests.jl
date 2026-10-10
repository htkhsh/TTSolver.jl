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

function environment_case(::Type{T}; mode_size::Int=2,
    sparse_operator::Bool=false, noncontiguous::Bool=false,
    x_type::Type=T) where {T}
    wl = wr = xl = xr = 2
    n = m = mode_size
    w = reshape(T.(1:(wl * n * wr)), wl, n, 1, wr) ./ T(7)
    x = reshape(x_type.(1:(xl * m * xr)), xl, m, 1, xr) ./ x_type(9)
    if T <: Complex
        w .+= im .* reverse(w; dims=2) ./ T(13)
    end
    if x_type <: Complex
        x .+= im .* reverse(x; dims=2) ./ x_type(11)
    end
    if noncontiguous
        w = PermutedDimsArray(permutedims(w, (2, 1, 3, 4)), (2, 1, 3, 4))
        x = PermutedDimsArray(permutedims(x, (2, 1, 3, 4)), (2, 1, 3, 4))
    end
    dense_operator = reshape(x_type.(1:(n * m)), n, m) ./ x_type(17)
    operator = sparse_operator ? sparse(dense_operator) : dense_operator
    left_matrix = [reshape(T.(1:(wl * xl)), wl, xl, 1) ./ T(5)]
    right_matrix = [reshape(T.(1:(wr * xr)), wr, xr, 1) ./ T(6)]
    left_vector = [reshape(T.(1:(wl * xl)), wl, xl) ./ T(5)]
    right_vector = [reshape(T.(1:(wr * xr)), wr, xr) ./ T(6)]
    return (;
        w, x, operators=[operator], vectors=[x],
        left_matrix, right_matrix, left_vector, right_vector,
    )
end

function reference_left_matrix(src, w, operators, x)
    wl, n, _, wr = size(w)
    xl, m, _, xr = size(x)
    outputs = Vector{Any}(undef, length(operators))
    for k in eachindex(operators)
        p = size(src[k], 3)
        q = size(operators[k], 2) ÷ m
        A4 = reshape(Matrix(operators[k]), p, n, m, q)
        T = promote_type(eltype(w), eltype(src[k]), eltype(A4), eltype(x))
        output = zeros(T, wr, xr, q)
        for c in 1:wr, j in 1:xr, qi in 1:q, a in 1:wl,
            i in 1:xl, pi in 1:p, ni in 1:n, mi in 1:m
            output[c, j, qi] += conj(w[a, ni, 1, c]) * src[k][a, i, pi] *
                                A4[pi, ni, mi, qi] * x[i, mi, 1, j]
        end
        outputs[k] = output
    end
    return outputs
end

function reference_right_matrix(src, w, operators, x)
    wl, n, _, wr = size(w)
    xl, m, _, xr = size(x)
    outputs = Vector{Any}(undef, length(operators))
    for k in eachindex(operators)
        q = size(src[k], 3)
        p = size(operators[k], 1) ÷ n
        A4 = reshape(Matrix(operators[k]), p, n, m, q)
        T = promote_type(eltype(w), eltype(src[k]), eltype(A4), eltype(x))
        output = zeros(T, wl, xl, p)
        for a in 1:wl, i in 1:xl, pi in 1:p, c in 1:wr,
            j in 1:xr, qi in 1:q, ni in 1:n, mi in 1:m
            output[a, i, pi] += conj(w[a, ni, 1, c]) * A4[pi, ni, mi, qi] *
                                x[i, mi, 1, j] * src[k][c, j, qi]
        end
        outputs[k] = output
    end
    return outputs
end

function reference_left_vector(src, w, vectors)
    wl, n, _, wr = size(w)
    outputs = Vector{Any}(undef, length(vectors))
    for k in eachindex(vectors)
        xl, _, _, xr = size(vectors[k])
        T = promote_type(eltype(w), eltype(src[k]), eltype(vectors[k]))
        output = zeros(T, wr, xr)
        for c in 1:wr, j in 1:xr, a in 1:wl, i in 1:xl, ni in 1:n
            output[c, j] += conj(w[a, ni, 1, c]) * src[k][a, i] *
                            vectors[k][i, ni, 1, j]
        end
        outputs[k] = output
    end
    return outputs
end

function reference_right_vector(src, w, vectors)
    wl, n, _, wr = size(w)
    outputs = Vector{Any}(undef, length(vectors))
    for k in eachindex(vectors)
        xl, _, _, xr = size(vectors[k])
        T = promote_type(eltype(w), eltype(src[k]), eltype(vectors[k]))
        output = zeros(T, wl, xl)
        for a in 1:wl, i in 1:xl, c in 1:wr, j in 1:xr, ni in 1:n
            output[a, i] += conj(w[a, ni, 1, c]) *
                            vectors[k][i, ni, 1, j] * src[k][c, j]
        end
        outputs[k] = output
    end
    return outputs
end

function check_environment_reducers(case)
    work = TTSolver.EnvironmentUpdateWorkspace(eltype(case.w))
    tolerance = eltype(case.w) === Float32 ? 5e-6 : 1e-13

    expected_left_matrix = reference_left_matrix(
        case.left_matrix, case.w, case.operators, case.x,
    )
    left_matrix_dest = Any[case.left_matrix...]
    returned = TTSolver.leftreduce_matrix!(
        left_matrix_dest, case.left_matrix, case.w,
        case.operators, case.x, work,
    )
    @test returned === left_matrix_dest
    @test all(isapprox.(
        left_matrix_dest, expected_left_matrix; rtol=tolerance, atol=tolerance,
    ))

    expected_right_matrix = reference_right_matrix(
        case.right_matrix, case.w, case.operators, case.x,
    )
    right_matrix_dest = Any[case.right_matrix...]
    returned = TTSolver.rightreduce_matrix!(
        right_matrix_dest, case.right_matrix, case.w,
        case.operators, case.x, work,
    )
    @test returned === right_matrix_dest
    @test all(isapprox.(
        right_matrix_dest, expected_right_matrix; rtol=tolerance, atol=tolerance,
    ))

    expected_left_vector = reference_left_vector(
        case.left_vector, case.w, case.vectors,
    )
    left_vector_dest = Any[case.left_vector...]
    returned = TTSolver.leftreduce_vector!(
        left_vector_dest, case.left_vector, case.w, case.vectors, work,
    )
    @test returned === left_vector_dest
    @test all(isapprox.(
        left_vector_dest, expected_left_vector; rtol=tolerance, atol=tolerance,
    ))

    expected_right_vector = reference_right_vector(
        case.right_vector, case.w, case.vectors,
    )
    right_vector_dest = Any[case.right_vector...]
    returned = TTSolver.rightreduce_vector!(
        right_vector_dest, case.right_vector, case.w, case.vectors, work,
    )
    @test returned === right_vector_dest
    @test all(isapprox.(
        right_vector_dest, expected_right_vector; rtol=tolerance, atol=tolerance,
    ))

    return (;
        work, left_matrix_dest, right_matrix_dest,
        left_vector_dest, right_vector_dest,
        expected_left_matrix, expected_right_matrix,
        expected_left_vector, expected_right_vector,
    )
end

function environment_allocation_counts(case)
    work = TTSolver.EnvironmentUpdateWorkspace(eltype(case.w))
    dest = copy(case.left_matrix)
    TTSolver.leftreduce_matrix!(
        dest, case.left_matrix, case.w, case.operators, case.x, work,
    )
    TTSolver.leftreduce_matrix(case.left_matrix, case.w, case.operators, case.x)
    cached = @allocated begin
        for _ in 1:10
            TTSolver.leftreduce_matrix!(
                dest, case.left_matrix, case.w, case.operators, case.x, work,
            )
        end
    end
    allocating = @allocated begin
        for _ in 1:10
            TTSolver.leftreduce_matrix(
                case.left_matrix, case.w, case.operators, case.x,
            )
        end
    end
    return cached, allocating
end

@testset "cached environment updates" begin
    for T in (Float64, ComplexF64)
        check_environment_reducers(environment_case(T))
    end

    check_environment_reducers(environment_case(Float64; sparse_operator=true))
    check_environment_reducers(environment_case(Float64; noncontiguous=true))

    mixed = environment_case(Float32; x_type=Float64)
    mixed_results = check_environment_reducers(mixed)
    @test eltype(first(mixed_results.left_matrix_dest)) ===
          eltype(first(mixed_results.expected_left_matrix))
    @test eltype(first(mixed_results.right_matrix_dest)) ===
          eltype(first(mixed_results.expected_right_matrix))
    @test eltype(first(mixed_results.left_vector_dest)) ===
          eltype(first(mixed_results.expected_left_vector))
    @test eltype(first(mixed_results.right_vector_dest)) ===
          eltype(first(mixed_results.expected_right_vector))

    base = environment_case(Float64)
    work = TTSolver.EnvironmentUpdateWorkspace(Float64)
    source_before = deepcopy(base.left_matrix)
    first_output = copy(base.left_matrix)
    TTSolver.leftreduce_matrix!(
        first_output, base.left_matrix, base.w, base.operators, base.x, work,
    )
    first_output_before = deepcopy(first_output)
    larger = environment_case(Float64; mode_size=4)
    larger_output = copy(larger.left_matrix)
    TTSolver.leftreduce_matrix!(
        larger_output, larger.left_matrix, larger.w,
        larger.operators, larger.x, work,
    )
    @test base.left_matrix == source_before
    @test first_output == first_output_before

    @test_throws DimensionMismatch TTSolver.leftreduce_matrix!(
        Any[], base.left_matrix, base.w, base.operators, base.x, work,
    )
    bad_core = reshape(collect(1.0:16.0), 2, 2, 2, 2)
    @test_throws DimensionMismatch TTSolver.leftreduce_matrix!(
        copy(base.left_matrix), base.left_matrix, bad_core,
        base.operators, base.x, work,
    )
    bad_mode_operator = [ones(3, 2)]
    @test_throws DimensionMismatch TTSolver.leftreduce_matrix!(
        copy(base.left_matrix), base.left_matrix, base.w,
        bad_mode_operator, base.x, work,
    )
    bad_rank_environment = [ones(2, 2, 2)]
    @test_throws DimensionMismatch TTSolver.leftreduce_matrix!(
        copy(bad_rank_environment), bad_rank_environment, base.w,
        base.operators, base.x, work,
    )

    cached, allocating = environment_allocation_counts(base)
    @test cached < allocating
end
