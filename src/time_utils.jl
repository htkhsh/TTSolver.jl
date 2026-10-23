function chebdiff(N::Integer)
    if N <= 1
        t = ones(Float64, 1)
        S = zeros(Float64, 1, 1)
        return t, S
    end
    x = collect((N - 1):-1:0)
    t = 0.5 .* (cos.(pi .* x ./ N) .+ 1.0)
    c = vcat(ones(Float64, N - 1), 2.0) .* ((-1.0) .^ x)
    T = repeat(t, 1, N)
    T = T .- T'
    S = (c * (1.0 ./ c)') ./ (T .+ Matrix{Float64}(I, N, N))
    diag_term = sum(S, dims=2)[:]
    diag_term .+= 0.5 * (-1.0)^N .* (c ./ t)
    S = S .- Diagonal(diag_term)
    return t, S
end

function _cheb2_interpolant_dense(x::AbstractVector, y::AbstractVector, nloc::Int)
    n = length(x)
    M = length(y)
    if n == 1
        return ones(Float64, M, 1), ones(Int, M, 1)
    end
    n_sets = div(n, nloc)

    w = (-1.0) .^ collect((nloc - 1):-1:0)
    w[end] *= 0.5
    w = repeat(w, n_sets)

    j0 = fld(nloc, 2)
    idx = (nloc - j0) .+ (0:(n_sets - 1)) .* nloc
    idx_end = nloc .+ (0:(n_sets - 1)) .* nloc
    cfac = (1.0 + cos(pi * j0 / nloc))
    x0 = (2.0 .* x[idx] .- x[idx_end] .* cfac) ./ (1.0 - cos(pi * j0 / nloc))
    x0 = repeat(reshape(x0, 1, n_sets), nloc, 1)
    x0 = reshape(x0, 1, n)

    x_test = reshape(x, 1, n)
    w = reshape(w, 1, n) .* (x0 .- x_test)

    x_edges = x[nloc:nloc:n]
    ind = vec(sum(y .> x_edges', dims=2))
    ind[ind .>= n_sets] .= n_sets - 1
    ind = ind .* nloc
    ind = repeat(reshape(ind, M, 1), 1, nloc)
    ind .+= repeat(reshape(1:nloc, 1, nloc), M, 1)

    x_loc = x[ind]
    w_loc = w[ind]
    y_loc = repeat(reshape(y, M, 1), 1, nloc)

    p = w_loc ./ (y_loc .- x_loc)
    Z = sum(p, dims=2)
    p = p ./ Z
    p[isnan.(p)] .= 1.0
    return p, ind
end

function cheb2_interpolant(x::AbstractVector, y::AbstractVector)
    n = length(x)
    M = length(y)
    if n == 1
        return ones(Float64, M, 1)
    end
    p, ind = _cheb2_interpolant_dense(x, y, n)
    return p
end

function cheb2_interpolant(x::AbstractVector, y::AbstractVector, nloc::Integer)
    return _cheb2_interpolant_dense(x, y, nloc)
end
