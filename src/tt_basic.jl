"""
    tt_ind2sub(n, idx)

Convert linear index `idx` to subscripts for tensor of size `n`.
Uses kron-compatible (big-endian) ordering where first dimension varies slowest.
"""
function tt_ind2sub(n::AbstractVector{<:Integer}, idx::Integer)
    dims = Tuple(reverse(n))  # Reverse for kron-compatible ordering
    total = prod(dims)
    (idx >= 1 && idx <= total) || error("index out of range")
    subs = collect(Tuple(CartesianIndices(dims)[idx]))
    return reverse(subs)  # Reverse back so index 1 corresponds to first core
end

function tt_ones(n::Integer, d::Integer=1; T=Float64)
    return tt_ones(fill(n, d); T=T)
end

function tt_ones(n::AbstractVector{<:Integer}; T=Float64)
    d = length(n)
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        cores[i] = reshape(fill(one(T), n[i]), 1, n[i], 1)
    end
    return TTTensor(cores)
end

function tt_eye(n::Integer, d::Integer=1; T=Float64)
    return tt_eye(fill(n, d); T=T)
end

function tt_eye(n::AbstractVector{<:Integer}; T=Float64)
    d = length(n)
    cores = Vector{Array{T,4}}(undef, d)
    for i in 1:d
        core = Matrix{T}(I, n[i], n[i])
        cores[i] = reshape(core, 1, n[i], n[i], 1)
    end
    return TTMatrix(cores)
end

function tt_unit(n::Integer, d::Integer=1, j::Real=1; T=Float64)
    return tt_unit(fill(n, d), j; T=T)
end

function tt_unit(n::AbstractVector{<:Integer}, j::Union{Real,AbstractVector{<:Integer}}=1; T=Float64)
    d = length(n)
    jval = j isa Real ? Int(round(j)) : j
    jvec = jval isa Integer ? tt_ind2sub(n, jval) : collect(jval)
    length(jvec) == d || error("index vector has wrong length")
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        v = zeros(T, n[i])
        v[jvec[i]] = one(T)
        cores[i] = reshape(v, 1, n[i], 1)
    end
    return TTTensor(cores)
end

function tt_x(n::Integer, d::Integer=1, xmin::Real=0; T=Float64)
    return tt_x(fill(n, d), xmin; T=T)
end

"""
    tt_x(n, xmin=0; T=Float64)

Create a TT-tensor whose `tt_full` gives [xmin, xmin+1, ..., xmin+prod(n)-1]
with kron-compatible (big-endian) ordering.

The linear index is computed as: idx = (i_1 - 1)*prod(n[2:end]) + (i_2 - 1)*prod(n[3:end]) + ... + (i_d - 1) + xmin
where first dimension varies slowest and last dimension varies fastest.
"""
function tt_x(n::AbstractVector{<:Integer}, xmin::Real=0; T=Float64)
    d = length(n)
    cores = Vector{Array{T,3}}(undef, d)
    if d == 1
        vals = collect(0:(n[1] - 1)) .+ xmin
        cores[1] = reshape(T.(vals), 1, n[1], 1)
        return TTTensor(cores)
    end

    # tt_full contracts cores 1→d, producing result with dimension order [n[1], n[2], ..., n[d]].
    # reshape(result, n...) makes n[1] vary fastest (column-major).
    # permutedims(arr, d:-1:1) then makes n[d] vary fastest.
    #
    # For vec(tt_full(tt_x)) = [0,1,...,prod(n)-1] with kron ordering:
    # idx = (i_1 - 1)*prod(n[2:end]) + (i_2 - 1)*prod(n[3:end]) + ... + (i_d - 1)
    #
    # Before permutedims, the internal order has n[1] varying fastest.
    # So internal idx = (i_d - 1)*prod(n[1:d-1]) + ... + (i_1 - 1)
    #
    # TT contraction: sum over ranks
    # We need: cores[1] contributes values based on i_1 (fastest internal)
    #          cores[d] contributes values based on i_d (slowest internal)

    # For kron-compatible output, we want:
    # external idx = (i_1 - 1) * stride_1 + ... + (i_d - 1) * stride_d
    # where stride_k = prod(n[k+1:d])
    
    # After permutedims, external indexing is kron-compatible.
    # Internal indexing has reversed strides:
    # internal idx = (i_1 - 1) * 1 + (i_2 - 1) * n[1] + ... + (i_d - 1) * prod(n[1:d-1])

    # First core (i_1 varies fastest internally, stride = 1)
    # Contributes: (i_1 - 1) * prod(n[2:d]) to external index
    stride_external = prod(n[2:d])
    core1 = Array{T,3}(undef, 1, n[1], 2)
    core1[1, :, 1] = T.(collect(0:(n[1] - 1)) .* stride_external) .+ xmin
    core1[1, :, 2] = ones(T, n[1])
    cores[1] = core1

    # Middle cores
    for k in 2:(d - 1)
        stride_external = prod(n[k+1:d])
        core = zeros(T, 2, n[k], 2)
        for j in 1:n[k]
            core[1, j, 1] = one(T)        # Pass through previous sum
            core[2, j, 2] = one(T)        # Pass through constant 1
            core[2, j, 1] = T(stride_external * (j - 1))  # Add contribution
        end
        cores[k] = core
    end

    # Last core (i_d varies slowest internally, stride = prod(n[1:d-1]))
    # Contributes: (i_d - 1) * 1 to external index (stride_d = 1)
    last = Array{T,3}(undef, 2, n[d], 1)
    last[1, :, 1] = ones(T, n[d])
    last[2, :, 1] = T.(collect(0:(n[d] - 1)))
    cores[d] = last

    return TTTensor(cores)
end

function diag_offset(n::Integer, offset::Integer, len::Integer; T=Float64)
    if len <= 0
        return zeros(T, n, n)
    end
    return diagm(offset => ones(T, len))
end

function tt_shift(n::Integer, d::Integer, i::Integer; T=Float64)
    return tt_shift(fill(n, d), i; T=T)
end

function tt_shift(n::AbstractVector{<:Integer}, i::Union{Integer,AbstractVector{<:Integer}}; T=Float64)
    d = length(n)
    if prod(n) == 1
        if i == 0
            return tt_eye(1; T=T)
        end
        core = reshape(zeros(T, 1), 1, 1, 1, 1)
        return TTMatrix([core])
    end

    trans = false
    ishift = i isa Integer ? i : minimum(i)
    if ishift < 0
        trans = true
        i = -i
    end

    if i isa Integer
        idx = i + 1
        ivec = tt_ind2sub(n, idx) .- 1
    else
        ivec = collect(i)
    end
    length(ivec) == d || error("shift vector has wrong length")

    cores = Vector{Array{T,4}}(undef, d)

    if d == 1
        main = diag_offset(n[1], -ivec[1], n[1] - ivec[1]; T=T)
        core = reshape(main, 1, n[1], n[1], 1)
        return TTMatrix([core])
    end

    i_d = ivec[d]
    core_d = zeros(T, 2, n[d], n[d], 1)
    core_d[1, :, :, 1] = diag_offset(n[d], -i_d, n[d] - i_d; T=T)
    core_d[2, :, :, 1] = diag_offset(n[d], -(i_d + 1), n[d] - i_d - 1; T=T)
    if trans
        core_d = permutedims(core_d, (1, 3, 2, 4))
    end
    cores[d] = core_d

    for j in 2:(d - 1)
        ij = ivec[j]
        core = zeros(T, 2, n[j], n[j], 2)
        core[1, :, :, 1] = diag_offset(n[j], -ij, n[j] - ij; T=T)
        core[2, :, :, 1] = diag_offset(n[j], -(ij + 1), n[j] - ij - 1; T=T)
        core[1, :, :, 2] = diag_offset(n[j], n[j] - ij, ij; T=T)
        core[2, :, :, 2] = diag_offset(n[j], n[j] - ij - 1, ij + 1; T=T)
        if trans
            core = permutedims(core, (1, 3, 2, 4))
        end
        cores[j] = core
    end

    i1 = ivec[1]
    core1 = zeros(T, 1, n[1], n[1], 2)
    core1[1, :, :, 1] = diag_offset(n[1], -i1, n[1] - i1; T=T)
    core1[1, :, :, 2] = diag_offset(n[1], n[1] - i1, i1; T=T)
    if trans
        core1 = permutedims(core1, (1, 3, 2, 4))
    end
    cores[1] = core1

    return TTMatrix(cores)
end

"""
    tt_rand(n; r=2, T=Float64)
    tt_rand(n, d; r=2, T=Float64)

Create a random TT-tensor with specified ranks.

# Arguments
- `n`: Mode sizes (integer or vector)
- `d`: Number of dimensions (optional if n is a vector)
- `r`: TT-ranks (integer for uniform ranks, or vector of length d+1)
- `T`: Element type (default: Float64)

# Examples
```julia
x = tt_rand([4, 5, 6]; r=3)  # Random TT with uniform rank 3
x = tt_rand(4, 3; r=2)       # 3D tensor with mode size 4, rank 2
```
"""
function tt_rand(n::Integer, d::Integer=1; r::Union{Integer,AbstractVector{<:Integer}}=2, T=Float64)
    return tt_rand(fill(n, d); r=r, T=T)
end

function tt_rand(n::AbstractVector{<:Integer}; r::Union{Integer,AbstractVector{<:Integer}}=2, T=Float64)
    d = length(n)
    
    # Handle ranks
    if r isa Integer
        ranks = fill(r, d + 1)
        ranks[1] = 1
        ranks[d + 1] = 1
    else
        length(r) == d + 1 || error("Rank vector must have length d+1")
        ranks = collect(r)
    end
    
    # Ensure boundary ranks are 1
    ranks[1] = 1
    ranks[d + 1] = 1
    
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        cores[i] = randn(T, ranks[i], n[i], ranks[i + 1])
    end
    
    return TTTensor(cores)
end

"""
    tt_matrix(A::AbstractMatrix; T=nothing)

Convert a dense matrix to a rank-1 TT-matrix (single core).
If T is not specified, uses `promote_type(eltype(A), Float64)` to ensure
numeric type compatibility.

# Examples
```julia
A = [1 2; 3 4]
A_tt = tt_matrix(A)
tt_full(A_tt) == A  # true
```
"""
function tt_matrix(A::AbstractMatrix; T=nothing)
    n, m = size(A)
    if T === nothing
        T = promote_type(eltype(A), Float64)
    end
    core = reshape(T.(A), 1, n, m, 1)
    return TTMatrix([core])
end

"""
    tt_mkron(A, B, ...)
    tt_mkron(matrices::AbstractVector)

Create a TT-matrix from the Kronecker product of dense matrices.
The result satisfies: `tt_full(tt_mkron(A, B, ...)) == kron(A, B, ...)`.

# Arguments
- `A, B, ...`: Dense matrices to combine
- `matrices`: Vector of matrices

# Examples
```julia
A = [1 0; 0 -1]
B = Matrix(I, 3, 3)
H = tt_mkron(A, B)
tt_full(H) == kron(A, B)  # true

# Equivalent to:
H = tt_kron(tt_matrix(A), tt_matrix(B))
```
"""
function tt_mkron(A::AbstractMatrix, B::AbstractMatrix)
    return tt_kron(tt_matrix(A), tt_matrix(B))
end

function tt_mkron(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, rest::AbstractMatrix...)
    return tt_mkron(tt_mkron(A, B), C, rest...)
end

function tt_mkron(A::TTMatrix, B::AbstractMatrix, rest::AbstractMatrix...)
    return tt_mkron(tt_kron(A, tt_matrix(B)), rest...)
end

function tt_mkron(A::TTMatrix)
    return A
end

function tt_mkron(matrices::AbstractVector{<:AbstractMatrix})
    isempty(matrices) && error("Need at least one matrix")
    result = tt_matrix(matrices[1])
    for i in 2:length(matrices)
        result = tt_kron(result, tt_matrix(matrices[i]))
    end
    return result
end

function tt_kron(a::TTTensor, b::TTTensor)
    return TTTensor(vcat(a.cores, b.cores))
end

function tt_kron(a::TTMatrix, b::TTMatrix)
    return TTMatrix(vcat(a.cores, b.cores))
end
