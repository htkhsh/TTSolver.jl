function orthogonalise_block(x_next, x, dir::Integer)
    if dir > 0
        rx1, n1, _, rx = size(x)
        xmat = reshape(x, rx1 * n1, rx)
        F = qr(xmat)
        Q = Matrix(F.Q)
        R = Matrix(F.R)
        x_new = reshape(Q, rx1, n1, 1, size(Q, 2))
        if x_next !== nothing && !isempty(x_next)
            rx, n2, _, rx2 = size(x_next)
            xnext = reshape(x_next, rx, n2 * rx2)
            xnext = R * xnext
            xnext = reshape(xnext, size(R, 1), n2, 1, rx2)
            x_next = xnext
        end
        return x_next, x_new, size(Q, 2)
    else
        rx, n1, _, rx2 = size(x)
        xmat = transpose(reshape(x, rx, n1 * rx2))
        F = qr(xmat)
        Q = Matrix(F.Q)
        R = Matrix(F.R)
        if x_next !== nothing && !isempty(x_next)
            rx0, n0, _, _ = size(x_next)
            xnext = reshape(x_next, rx0 * n0, rx)
            xnext = xnext * transpose(R)
            xnext = reshape(xnext, rx0, n0, 1, size(R, 1))
            x_next = xnext
        end
        x_new = transpose(Q)
        x_new = reshape(x_new, size(Q, 2), n1, 1, rx2)
        return x_next, x_new, size(Q, 2)
    end
end

function leftreduce_vector(WX1::AbstractVector, w::AbstractArray, x::AbstractVector)
    WX2 = copy(WX1)
    rw1, n, _, rw2 = size(w)
    wc = reshape(w, rw1, n * rw2)
    for k in eachindex(x)
        rx1, _, _, rx2 = size(x[k])
        tmp = wc' * WX2[k]
        tmp = reshape(tmp, n, rw2 * rx1)
        tmp = permutedims(tmp, (2, 1))
        tmp = reshape(tmp, rw2, rx1 * n)
        xk = reshape(x[k], rx1 * n, rx2)
        WX2[k] = tmp * xk
    end
    return WX2
end

function leftreduce_matrix(WAX1::AbstractVector, w::AbstractArray, A::AbstractVector, x::AbstractArray)
    WAX2 = copy(WAX1)
    rx1, m, _, rx2 = size(x)
    rw1, n, _, rw2 = size(w)
    wc = reshape(w, rw1, n * rw2)
    xc = reshape(x, rx1 * m, rx2)
    for k in eachindex(A)
        ra1 = size(WAX1[k], 3)
        tmp = reshape(WAX2[k], rw1, rx1 * ra1)
        tmp = wc' * tmp
        tmp = reshape(tmp, n, rw2 * rx1 * ra1)
        tmp = permutedims(tmp, (2, 1))
        tmp = reshape(tmp, rw2 * rx1, ra1 * n)
        Ak = reshape(A[k], ra1 * n, :)
        tmp = tmp * Ak
        tmp = reshape(tmp, rw2, :)
        tmp = permutedims(tmp, (2, 1))
        tmp = reshape(tmp, rx1 * m, :)
        tmp = transpose(xc) * tmp
        tmp = reshape(tmp, :, rw2)
        tmp = permutedims(tmp, (2, 1))
        WAX2[k] = reshape(tmp, rw2, rx2, :)
    end
    return WAX2
end

function rightreduce_vector(WX2::AbstractVector, w::AbstractArray, x::AbstractVector)
    WX1 = copy(WX2)
    rw1, n, _, rw2 = size(w)
    wc = reshape(w, rw1, n * rw2)
    for k in eachindex(x)
        rx1, _, _, rx2 = size(x[k])
        tmp = reshape(x[k], rx1 * n, rx2)
        tmp = tmp * transpose(WX1[k])
        tmp = reshape(tmp, rx1, n * rw2)
        WX1[k] = permutedims(tmp * wc', (2, 1))
    end
    return WX1
end

function rightreduce_matrix(WAX2::AbstractVector, w::AbstractArray, A::AbstractVector, x::AbstractArray)
    WAX1 = copy(WAX2)
    rw1, n, _, rw2 = size(w)
    wc = reshape(conj.(w), rw1, n * rw2)
    rx1, m, _, rx2 = size(x)
    xc = reshape(x, rx1 * m, rx2)
    for k in eachindex(A)
        rw2_k, rx2_k, ra2 = size(WAX2[k])
        if ndims(A[k]) == 2
            ra1 = size(A[k], 1) ÷ n
            m_k = size(A[k], 2) ÷ ra2
            m_k == m || error("mode size mismatch in rightreduce_matrix")
        else
            ra1 = size(A[k], 1)
            n = size(A[k], 2)
            m_k = size(A[k], 3)
            ra2 = size(A[k], 4)
        end
        wx = reshape(permutedims(WAX2[k], (2, 1, 3)), rx2_k, rw2_k * ra2)
        tmp = xc * wx
        tmp = reshape(tmp, rx1, m_k, rw2_k, ra2)
        tmp = permutedims(tmp, (2, 4, 3, 1))
        tmp = reshape(tmp, m_k * ra2, rw2_k * rx1)
        Ak = reshape(A[k], :, m_k * ra2)
        tmp = Ak * tmp
        tmp = reshape(tmp, ra1, n, rw2_k, rx1)
        tmp = permutedims(tmp, (2, 3, 1, 4))
        tmp = reshape(tmp, n * rw2_k, ra1 * rx1)
        tmp = wc * tmp
        tmp = reshape(tmp, rw1, ra1, rx1)
        tmp = permutedims(tmp, (1, 3, 2))
        WAX1[k] = tmp
    end
    return WAX1
end

struct LocalMatvecWorkspace{T,TA}
    rx1::Int
    m::Int
    rx2::Int
    b::Int
    rw1::Int
    n::Int
    rw2::Int
    A::TA
    ra1::Vector{Int}
    ra2::Vector{Int}
    xc::Matrix{T}
    left::Vector{Matrix{T}}
    right::Vector{Matrix{T}}
    x_contract::Vector{Matrix{T}}
    a_input::Vector{Matrix{T}}
    a_contract::Vector{Matrix{T}}
    left_input::Vector{Matrix{T}}
end

function LocalMatvecWorkspace(x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, ra1, ra2)
    T = eltype(x)
    ranks1 = Int.(ra1)
    ranks2 = Int.(ra2)
    Ra = length(A)
    left = Vector{Matrix{T}}(undef, Ra)
    right = Vector{Matrix{T}}(undef, Ra)
    x_contract = Vector{Matrix{T}}(undef, Ra)
    a_input = Vector{Matrix{T}}(undef, Ra)
    a_contract = Vector{Matrix{T}}(undef, Ra)
    left_input = Vector{Matrix{T}}(undef, Ra)
    for k in 1:Ra
        left[k] = copy(reshape(WAX1[k], rw1, rx1 * ranks1[k]))
        right[k] = Matrix{T}(undef, ranks2[k] * rw2, rx2)
        @inbounds for q in 1:ranks2[k], c in 1:rw2, j in 1:rx2
            right[k][q + (c - 1) * ranks2[k], j] = WAX2[k][c, j, q]
        end
        x_contract[k] = Matrix{T}(undef, b * rx1 * m, ranks2[k] * rw2)
        a_input[k] = Matrix{T}(undef, m * ranks2[k], rw2 * b * rx1)
        a_contract[k] = Matrix{T}(undef, ranks1[k] * n, rw2 * b * rx1)
        left_input[k] = Matrix{T}(undef, rx1 * ranks1[k], n * rw2 * b)
    end
    xc = Matrix{T}(undef, b * rx1 * m, rx2)
    return LocalMatvecWorkspace(
        rx1, m, rx2, b, rw1, n, rw2, A, ranks1, ranks2, xc,
        left, right, x_contract, a_input, a_contract, left_input,
    )
end

function local_matvec!(w, x, workspace::LocalMatvecWorkspace{T}) where T
    rx1 = workspace.rx1
    m = workspace.m
    rx2 = workspace.rx2
    b = workspace.b
    rw1 = workspace.rw1
    n = workspace.n
    rw2 = workspace.rw2

    @inbounds for bi in 1:b, i in 1:rx1, mi in 1:m, j in 1:rx2
        workspace.xc[bi + (i - 1) * b + (mi - 1) * b * rx1, j] =
            x[
                i + (mi - 1) * rx1 + (j - 1) * rx1 * m +
                (bi - 1) * rx1 * m * rx2
            ]
    end

    fill!(w, zero(T))
    wmat = reshape(w, rw1, n * rw2 * b)
    for k in eachindex(workspace.A)
        r1 = workspace.ra1[k]
        r2 = workspace.ra2[k]
        x_contract = workspace.x_contract[k]
        a_input = workspace.a_input[k]
        a_contract = workspace.a_contract[k]
        left_input = workspace.left_input[k]

        mul!(x_contract, workspace.xc, transpose(workspace.right[k]))
        @inbounds for bi in 1:b, i in 1:rx1, c in 1:rw2, q in 1:r2, mi in 1:m
            a_input[mi + (q - 1) * m, c + (bi - 1) * rw2 + (i - 1) * rw2 * b] =
                x_contract[
                    bi + (i - 1) * b + (mi - 1) * b * rx1,
                    q + (c - 1) * r2,
                ]
        end

        Ak = workspace.A[k]
        A2 = ndims(Ak) == 2 ? Ak : reshape(Ak, r1 * n, m * r2)
        mul!(a_contract, A2, a_input)
        @inbounds for bi in 1:b, c in 1:rw2, ni in 1:n, p in 1:r1, i in 1:rx1
            left_input[i + (p - 1) * rx1, ni + (c - 1) * n + (bi - 1) * n * rw2] =
                a_contract[p + (ni - 1) * r1, c + (bi - 1) * rw2 + (i - 1) * rw2 * b]
        end
        mul!(wmat, workspace.left[k], left_input, one(T), one(T))
    end
    return w
end

function local_matvec(x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, Ra, ra1, ra2)
    T = eltype(x)
    w = zeros(T, rw1 * n * rw2, b)
    workspace = LocalMatvecWorkspace(
        x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, ra1, ra2,
    )
    return local_matvec!(w, x, workspace)
end

function assemble_local_matrix(WAX1::AbstractVector, A::AbstractVector, WAX2::AbstractVector)
    Ra = length(A)
    sparseflag = all(k -> issparse(A[k]), 1:Ra)
    if sparseflag
        B = nothing
        for k in 1:Ra
            rw1, rx1, ra1 = size(WAX1[k])
            # WAX2 has shape (rw2, rx2, ra2), not (ra2, rw2, rx2) like MATLAB
            rw2, rx2, ra2 = size(WAX2[k])
            Bk = spzeros(eltype(A[k]), 0, 0)
            n = size(A[k], 1) ÷ ra1
            m = size(A[k], 2) ÷ ra2
            Ak = reshape(A[k], ra1, :)
            for j in 1:ra1
                tmp = sparse(WAX1[k][:, :, j])
                Akj = reshape(sparse(Ak[j, :]), n, m * ra2)
                Bk = Bk + kron(Akj, tmp)
            end
            if ra2 > 1
                Bk = reshape(Bk, rw1 * n * rx1 * m, ra2)
                for j in 1:ra2
                    # WAX2 has shape (rw2, rx2, ra2), need to extract WAX2[:, :, j]
                    tmp = sparse(WAX2[k][:, :, j])
                    Akj = reshape(Bk[:, j], rw1 * n, rx1 * m)
                    B = B === nothing ? kron(Akj, tmp) : (B + kron(Akj, tmp))
                end
            else
                tmp = sparse(WAX2[k][:, :, 1])
                B = B === nothing ? kron(Bk, tmp) : (B + kron(Bk, tmp))
            end
        end
    else
        B = nothing
        for k in 1:Ra
            rw1, rx1, ra1 = size(WAX1[k])
            # WAX2 has shape (rw2, rx2, ra2), not (ra2, rw2, rx2) like MATLAB
            rw2, rx2, ra2 = size(WAX2[k])
            Bk = reshape(WAX1[k], rw1 * rx1, ra1)
            tmp = reshape(A[k], ra1, :)
            if ndims(A[k]) == 2
                n = size(A[k], 1) ÷ ra1
                m = size(A[k], 2) ÷ ra2
            else
                n = size(A[k], 2)
                m = size(A[k], 3)
            end
            Bk = Bk * tmp
            Bk = reshape(Bk, rw1, rx1, n, m * ra2)
            Bk = reshape(PermutedDimsArray(Bk, (1, 3, 2, 4)), rw1 * n * rx1 * m, ra2)
            # WAX2 has shape (rw2, rx2, ra2), need to reshape to (ra2, rw2 * rx2)
            tmp = permutedims(WAX2[k], (3, 1, 2))  # now (ra2, rw2, rx2)
            tmp = reshape(tmp, ra2, rw2 * rx2)
            Bk = Bk * tmp
            Bk = reshape(Bk, rw1 * n, rx1 * m, rw2, rx2)
            Bk = reshape(PermutedDimsArray(Bk, (1, 3, 2, 4)), rw1 * n * rw2, rx1 * m * rx2)
            B = B === nothing ? Bk : (B .+ Bk)
        end
    end
    return B === nothing ? spzeros(eltype(A[1]), 0, 0) : B, sparseflag
end

function assemble_local_vector(WX1::AbstractVector, x::AbstractVector, WX2::AbstractVector)
    w = zeros(eltype(x[1]), 0, 0)
    for k in eachindex(x)
        rx1, n, _, rx2 = size(x[k])
        xcore = reshape(x[k], rx1, n, rx2)
        WX1k = WX1[k]
        WX2k = WX2[k]
        @tensor wk[rw1i, n_i, rw2i] := WX1k[rw1i, rx1i] * xcore[rx1i, n_i, rx2i] * WX2k[rw2i, rx2i]
        wk = reshape(wk, :, 1)
        if size(w, 1) == 0
            w = wk
        else
            w .+= wk
        end
    end
    return w
end

function assemble_local_rjacobi(WAX1::AbstractVector, A::AbstractVector, WAX2::AbstractVector)
    rx1 = size(WAX1[1], 2)
    rx2 = size(WAX2[1], 2)  # Julia shape (rw2, rx2, ra2)
    P = [zeros(eltype(A[1]), 0, 0) for _ in 1:rx1]
    ind1 = (0:(rx1 - 1)) .* (rx1 + 1) .+ 1
    for k in eachindex(A)
        ra1 = size(WAX1[k], 3)
        # Julia shape for WAX2: (rw2, rx2, ra2)
        rw2, rx2_k, ra2 = size(WAX2[k])
        B1 = reshape(WAX1[k], rx1 * rx1, ra1)
        B1 = B1[ind1, :]
        B1 = transpose(B1)
        tmp = reshape(A[k], :, ra2)
        n = ndims(A[k]) == 2 ? (size(A[k], 1) ÷ ra1) : size(A[k], 2)
        # Permute WAX2 from (rw2, rx2, ra2) to (ra2, rw2, rx2) then reshape to (ra2, rx2 * rx2)
        WAX2_perm = permutedims(WAX2[k], (3, 1, 2))  # (ra2, rw2, rx2)
        Bk = tmp * reshape(WAX2_perm, ra2, rw2 * rx2_k)
        Bk = reshape(Bk, ra1, n, rw2, rx2_k)
        Bk = permutedims(Bk, (1, 3, 2, 4))
        Bk = reshape(Bk, ra1, n * rw2 * n * rx2_k)
        for i in 1:rx1
            P[i] = P[i] + transpose(B1[:, i]) * Bk
        end
    end
    for i in 1:rx1
        n = size(A[1], 2)
        rx2_val = size(WAX2[1], 2)  # Get rx2 from first WAX2
        P[i] = reshape(P[i], n * rx2_val, n * rx2_val)
        P[i] = inv(P[i])
    end
    return P
end

function local_precvec(x, rx1, m, rx2, b, P)
    if size(P, 2) == rx1
        w = zeros(eltype(x), m * rx2, b * rx1)
        xc = transpose(reshape(x, rx1, m * rx2 * b))
        xc = reshape(xc, m * rx2, b * rx1)
        for i in 1:rx1
            for j in 1:b
                w[:, j + (i - 1) * b] = P[i] * xc[:, j + (i - 1) * b]
            end
        end
        w = reshape(w, m * rx2 * b, rx1)
        w = transpose(w)
        w = reshape(w, rx1 * m * rx2, b)
        return w
    end
    return x
end
