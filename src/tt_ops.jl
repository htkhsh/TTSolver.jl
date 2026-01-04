import Base: +, -, *
import LinearAlgebra: dot, norm

function +(a::TTTensor, b::TTTensor)
    na = tt_dims(a)
    nb = tt_dims(b)
    na == nb || error("mode sizes are inconsistent")

    da = length(a.cores)
    db = length(b.cores)
    da == db || error("number of cores is inconsistent")
    if da == 1
        return TTTensor([a.cores[1] .+ b.cores[1]])
    end

    ra = tt_ranks(a)
    rb = tt_ranks(b)
    ra[1] == rb[1] || error("left TT rank is inconsistent")
    ra[end] == rb[end] || error("right TT rank is inconsistent")

    cores = Vector{Array{promote_type(eltype(a.cores[1]), eltype(b.cores[1])),3}}(undef, da)
    for i in 1:da
        r1a, n, r2a = size(a.cores[i])
        r1b, _, r2b = size(b.cores[i])
        r1 = i == 1 ? r1a : r1a + r1b
        r2 = i == da ? r2a : r2a + r2b
        core = zeros(promote_type(eltype(a.cores[i]), eltype(b.cores[i])), r1, n, r2)
        if i == 1
            core[:, :, 1:r2a] .= a.cores[i]
            core[:, :, (r2 - r2b + 1):r2] .= b.cores[i]
        elseif i == da
            core[1:r1a, :, :] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :] .= b.cores[i]
        else
            core[1:r1a, :, 1:r2a] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, (r2 - r2b + 1):r2] .= b.cores[i]
        end
        cores[i] = core
    end
    return TTTensor(cores)
end

function +(a::TTMatrix, b::TTMatrix)
    na, ma = tt_dims(a)
    nb, mb = tt_dims(b)
    na == nb || error("row mode sizes are inconsistent")
    ma == mb || error("col mode sizes are inconsistent")

    da = length(a.cores)
    db = length(b.cores)
    da == db || error("number of cores is inconsistent")

    ra = tt_ranks(a)
    rb = tt_ranks(b)
    ra[1] == rb[1] || error("left TT rank is inconsistent")
    ra[end] == rb[end] || error("right TT rank is inconsistent")

    cores = Vector{Array{promote_type(eltype(a.cores[1]), eltype(b.cores[1])),4}}(undef, da)
    for i in 1:da
        r1a, n, m, r2a = size(a.cores[i])
        r1b, _, _, r2b = size(b.cores[i])
        r1 = i == 1 ? r1a : r1a + r1b
        r2 = i == da ? r2a : r2a + r2b
        core = zeros(promote_type(eltype(a.cores[i]), eltype(b.cores[i])), r1, n, m, r2)
        if i == 1
            core[:, :, :, 1:r2a] .= a.cores[i]
            core[:, :, :, (r2 - r2b + 1):r2] .= b.cores[i]
        elseif i == da
            core[1:r1a, :, :, :] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :, :] .= b.cores[i]
        else
            core[1:r1a, :, :, 1:r2a] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :, (r2 - r2b + 1):r2] .= b.cores[i]
        end
        cores[i] = core
    end
    return TTMatrix(cores)
end

-(a::TTTensor, b::TTTensor) = a + (-one(eltype(a.cores[1])) * b)
-(a::TTMatrix, b::TTMatrix) = a + (-one(eltype(a.cores[1])) * b)

function *(a::TTTensor, α::Number)
    cores = [copy(core) for core in a.cores]
    cores[1] = cores[1] .* α
    return TTTensor(cores)
end
*(α::Number, a::TTTensor) = a * α

function *(a::TTMatrix, α::Number)
    cores = [copy(core) for core in a.cores]
    cores[1] = cores[1] .* α
    return TTMatrix(cores)
end
*(α::Number, a::TTMatrix) = a * α

function +(a::TTTensor, α::Number)
    T = promote_type(eltype(a.cores[1]), typeof(α))
    return a + (α * tt_ones(tt_dims(a); T=T))
end

+(α::Number, a::TTTensor) = a + α

function -(a::TTTensor, α::Number)
    return a + (-α)
end

-(α::Number, a::TTTensor) = α + (-one(eltype(a.cores[1])) * a)

function *(A::TTMatrix, x::TTTensor)
    nA, mA = tt_dims(A)
    nX = tt_dims(x)
    mA == nX || error("matrix column modes must match input sizes")

    d = length(nA)
    T = promote_type(eltype(A.cores[1]), eltype(x.cores[1]))
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        coreA = A.cores[i]
        coreX = x.cores[i]
        @tensor coreY[ra1, rx1, n, ra2, rx2] := coreA[ra1, n, m, ra2] * coreX[rx1, m, rx2]
        r1 = size(coreA, 1) * size(coreX, 1)
        r2 = size(coreA, 4) * size(coreX, 3)
        cores[i] = reshape(coreY, r1, nA[i], r2)
    end
    return TTTensor(cores)
end

function *(A::TTMatrix, B::TTMatrix)
    nA, mA = tt_dims(A)
    nB, mB = tt_dims(B)
    mA == nB || error("matrix inner sizes are inconsistent")

    d = length(nA)
    T = promote_type(eltype(A.cores[1]), eltype(B.cores[1]))
    cores = Vector{Array{T,4}}(undef, d)
    for i in 1:d
        coreA = A.cores[i]
        coreB = B.cores[i]
        @tensor coreC[ra1, rb1, n, m, ra2, rb2] := coreA[ra1, n, p, ra2] * coreB[rb1, p, m, rb2]
        r1 = size(coreA, 1) * size(coreB, 1)
        r2 = size(coreA, 4) * size(coreB, 4)
        cores[i] = reshape(coreC, r1, nA[i], mB[i], r2)
    end
    return TTMatrix(cores)
end

function dot(a::TTTensor, b::TTTensor)
    return tt_dot(a, b)
end

function norm(a::TTTensor)
    return sqrt(real(dot(a, a)))
end

"""
    tt_norm(x::TTTensor)

Compute the Frobenius norm of a TT-tensor.
Equivalent to `sqrt(tt_dot(x, x))`.
"""
function tt_norm(x::TTTensor)
    return sqrt(real(tt_dot(x, x)))
end

"""
    tt_round(x::TTTensor, eps::Real; rmax::Int=typemax(Int))

Round (compress) a TT-tensor to relative accuracy `eps` with maximum rank `rmax`.

Uses the standard TT-rounding algorithm:
1. Left-to-right QR orthogonalization
2. Right-to-left SVD with truncation

# Arguments
- `x::TTTensor`: Input TT-tensor
- `eps::Real`: Relative accuracy for truncation
- `rmax::Int=typemax(Int)`: Maximum allowed rank

# Returns
- `TTTensor`: Rounded TT-tensor

# Example
```julia
a = tt_ones([2, 3, 4])
b = a + a  # ranks increase
b_rounded = tt_round(b, 1e-14)  # ranks back to 1
```
"""
function tt_round(x::TTTensor{T}, eps::Real; rmax::Int=typemax(Int)) where T
    d = length(x.cores)
    d == 1 && return TTTensor([copy(x.cores[1])])
    
    # Copy cores
    cores = [copy(core) for core in x.cores]
    n = tt_dims(x)
    r = collect(tt_ranks(x))
    
    # Left-to-right QR orthogonalization
    nrms = ones(real(T), d)
    for i in 1:d-1
        r1, ni, r2 = size(cores[i])
        core_mat = reshape(cores[i], r1 * ni, r2)
        Q, R = qr(core_mat)
        Q = Matrix(Q)
        nrms[i+1] = norm(R)
        if nrms[i+1] != 0
            R = R ./ nrms[i+1]
        end
        
        rnew = size(Q, 2)
        r[i+1] = rnew
        cores[i] = reshape(Q, r1, ni, rnew)
        
        # Multiply R into next core
        r2_next = size(cores[i+1], 3)
        core_next = reshape(cores[i+1], r2, n[i+1] * r2_next)
        core_next = R * core_next
        cores[i+1] = reshape(core_next, rnew, n[i+1], r2_next)
    end
    
    # Right-to-left SVD with truncation
    ep = eps / sqrt(d - 1)
    for i in d:-1:2
        r1, ni, r2 = size(cores[i])
        core_mat = reshape(cores[i], r1, ni * r2)
        
        F = svd(core_mat)
        s = F.S
        
        # Truncation
        rnew = _tt_chop(s, norm(s) * ep)
        rnew = min(rnew, rmax, length(s))
        
        U = F.U[:, 1:rnew]
        S = Diagonal(s[1:rnew])
        Vt = F.Vt[1:rnew, :]
        
        # cores[i] = Vt reshaped
        cores[i] = reshape(Vt, rnew, ni, r2)
        
        # Multiply U*S into previous core
        r1_prev = size(cores[i-1], 1)
        core_prev = reshape(cores[i-1], r1_prev * n[i-1], r1)
        core_prev = core_prev * (U * S)
        r[i] = rnew
        cores[i-1] = reshape(core_prev, r1_prev, n[i-1], rnew)
    end
    
    # Normalize first core and balance norms
    r1, n1, r2 = size(cores[1])
    nrms[1] = norm(cores[1])
    if nrms[1] != 0
        cores[1] = cores[1] ./ nrms[1]
    end
    
    # Balance norms across all cores
    log_nrm = sum(log.(abs.(nrms) .+ eps))
    nrm_balanced = exp(log_nrm / d)
    for i in 1:d
        cores[i] = cores[i] .* nrm_balanced
    end
    
    return TTTensor(cores)
end

"""
    tt_round(A::TTMatrix, eps::Real; rmax::Int=typemax(Int))

Round (compress) a TT-matrix to relative accuracy `eps`.
"""
function tt_round(A::TTMatrix{T}, eps::Real; rmax::Int=typemax(Int)) where T
    d = length(A.cores)
    d == 1 && return TTMatrix([copy(A.cores[1])])
    
    # Copy cores
    cores = [copy(core) for core in A.cores]
    n, m = tt_dims(A)
    r = collect(tt_ranks(A))
    
    # Left-to-right QR orthogonalization
    nrms = ones(real(T), d)
    for i in 1:d-1
        r1, ni, mi, r2 = size(cores[i])
        core_mat = reshape(cores[i], r1 * ni * mi, r2)
        Q, R = qr(core_mat)
        Q = Matrix(Q)
        nrms[i+1] = norm(R)
        if nrms[i+1] != 0
            R = R ./ nrms[i+1]
        end
        
        rnew = size(Q, 2)
        r[i+1] = rnew
        cores[i] = reshape(Q, r1, ni, mi, rnew)
        
        # Multiply R into next core
        r2_next = size(cores[i+1], 4)
        core_next = reshape(cores[i+1], r2, n[i+1] * m[i+1] * r2_next)
        core_next = R * core_next
        cores[i+1] = reshape(core_next, rnew, n[i+1], m[i+1], r2_next)
    end
    
    # Right-to-left SVD with truncation
    ep = eps / sqrt(d - 1)
    for i in d:-1:2
        r1, ni, mi, r2 = size(cores[i])
        core_mat = reshape(cores[i], r1, ni * mi * r2)
        
        F = svd(core_mat)
        s = F.S
        
        # Truncation
        rnew = _tt_chop(s, norm(s) * ep)
        rnew = min(rnew, rmax, length(s))
        
        U = F.U[:, 1:rnew]
        S = Diagonal(s[1:rnew])
        Vt = F.Vt[1:rnew, :]
        
        # cores[i] = Vt reshaped
        cores[i] = reshape(Vt, rnew, ni, mi, r2)
        
        # Multiply U*S into previous core
        r1_prev = size(cores[i-1], 1)
        core_prev = reshape(cores[i-1], r1_prev * n[i-1] * m[i-1], r1)
        core_prev = core_prev * (U * S)
        r[i] = rnew
        cores[i-1] = reshape(core_prev, r1_prev, n[i-1], m[i-1], rnew)
    end
    
    # Normalize first core and balance norms
    nrms[1] = norm(cores[1])
    if nrms[1] != 0
        cores[1] = cores[1] ./ nrms[1]
    end
    
    # Balance norms across all cores
    log_nrm = sum(log.(abs.(nrms) .+ eps))
    nrm_balanced = exp(log_nrm / d)
    for i in 1:d
        cores[i] = cores[i] .* nrm_balanced
    end
    
    return TTMatrix(cores)
end

"""
Helper function for truncation: find the smallest rank such that
the truncation error is below `eps`.
"""
function _tt_chop(s::AbstractVector, eps::Real)
    n = length(s)
    if eps <= 0
        return n
    end
    
    # Cumulative sum of squared singular values from the end
    cumsum_sq = cumsum(abs2.(reverse(s)))
    
    # Find first index where cumsum exceeds eps^2
    for i in 1:n
        if cumsum_sq[i] >= eps^2
            return n - i + 1
        end
    end
    return 1
end

"""
    tt_hadamard(a::TTTensor, b::TTTensor)

Element-wise (Hadamard) product of two TT-tensors.

The result has TT-ranks equal to the product of input ranks.

# Example
```julia
a = tt_ones([2, 3, 4])
b = tt_ones([2, 3, 4])
c = tt_hadamard(a, b)  # c ≈ a (since 1*1 = 1)
```
"""
function tt_hadamard(a::TTTensor{Ta}, b::TTTensor{Tb}) where {Ta, Tb}
    na = tt_dims(a)
    nb = tt_dims(b)
    na == nb || error("mode sizes must match")
    
    d = length(a.cores)
    T = promote_type(Ta, Tb)
    cores = Vector{Array{T,3}}(undef, d)
    
    for i in 1:d
        core_a = a.cores[i]  # ra1 × n × ra2
        core_b = b.cores[i]  # rb1 × n × rb2
        
        ra1, n, ra2 = size(core_a)
        rb1, _, rb2 = size(core_b)
        
        # Hadamard product via element-wise multiplication
        # Result: (ra1*rb1) × n × (ra2*rb2)
        core_c = zeros(T, ra1 * rb1, n, ra2 * rb2)
        for j in 1:n
            # kron of slices: (ra1 × ra2) ⊗ (rb1 × rb2) → (ra1*rb1 × ra2*rb2)
            core_c[:, j, :] = kron(core_a[:, j, :], core_b[:, j, :])
        end
        cores[i] = core_c
    end
    
    return TTTensor(cores)
end

"""
    tt_hadamard(A::TTMatrix, B::TTMatrix)

Element-wise (Hadamard) product of two TT-matrices.
"""
function tt_hadamard(A::TTMatrix{Ta}, B::TTMatrix{Tb}) where {Ta, Tb}
    nA, mA = tt_dims(A)
    nB, mB = tt_dims(B)
    nA == nB && mA == mB || error("matrix sizes must match")
    
    d = length(A.cores)
    T = promote_type(Ta, Tb)
    cores = Vector{Array{T,4}}(undef, d)
    
    for i in 1:d
        core_A = A.cores[i]  # ra1 × n × m × ra2
        core_B = B.cores[i]  # rb1 × n × m × rb2
        
        ra1, n, m, ra2 = size(core_A)
        rb1, _, _, rb2 = size(core_B)
        
        # Result: (ra1*rb1) × n × m × (ra2*rb2)
        core_C = zeros(T, ra1 * rb1, n, m, ra2 * rb2)
        for j in 1:n
            for k in 1:m
                core_C[:, j, k, :] = kron(core_A[:, j, k, :], core_B[:, j, k, :])
            end
        end
        cores[i] = core_C
    end
    
    return TTMatrix(cores)
end