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

function leftreduce_vector(WX1::Vector, w::AbstractArray, x::Vector)
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

function leftreduce_matrix(WAX1::Vector, w::AbstractArray, A::Vector, x::AbstractArray)
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

function rightreduce_vector(WX2::Vector, w::AbstractArray, x::Vector)
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

function rightreduce_matrix(WAX2::Vector, w::AbstractArray, A::Vector, x::AbstractArray)
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

function local_matvec(x, rx1, m, rx2, b, rw1, n, rw2, WAX1, A, WAX2, Ra, ra1, ra2)
    T = eltype(x)
    w = zeros(T, rw1 * n * rw2, b)
    w4 = reshape(w, rw1, n, rw2, b)
    x4 = reshape(x, rx1, m, rx2, b)
    xmat = reshape(x, rx1 * m * rx2, b)
    xc = reshape(PermutedDimsArray(xmat, (2, 1)), b * rx1 * m, rx2)
    
    # Pre-allocate buffers (will be resized as needed)
    wk_buf = Matrix{T}(undef, 0, 0)
    wk2_buf = Matrix{T}(undef, 0, 0)
    wk3_buf = Matrix{T}(undef, 0, 0)
    rows1 = b * rx1 * m
    rows3 = rw1
    
    @inbounds for k in 1:Ra
        WAX1k = WAX1[k]
        WAX2k = WAX2[k]
        # Julia shape: WAX1k is (rw1, rx1, ra1), WAX2k is (rw2, rx2, ra2)
        if issparse(A[k]) ||
           size(WAX1k, 1) != rw1 || size(WAX1k, 2) != rx1 || size(WAX1k, 3) != ra1[k] ||
           size(WAX2k, 1) != rw2 || size(WAX2k, 2) != rx2 || size(WAX2k, 3) != ra2[k]
            # WAX2 has shape (rw2, rx2, ra2), need to reshape to (ra2*rw2, rx2)
            tmp = permutedims(WAX2[k], (3, 1, 2))  # (ra2, rw2, rx2)
            tmp = reshape(tmp, :, rx2)  # (ra2*rw2, rx2)
            cols1 = size(tmp, 1)
            if size(wk_buf, 1) != rows1 || size(wk_buf, 2) != cols1
                wk_buf = Matrix{T}(undef, rows1, cols1)
            end
            mul!(wk_buf, xc, transpose(tmp))
            wk = wk_buf
            wk = reshape(wk, b * rx1, m * ra2[k] * rw2)
            wk = reshape(PermutedDimsArray(wk, (2, 1)), m * ra2[k], rw2 * b * rx1)
            rows2 = size(A[k], 1)
            cols2 = size(wk, 2)
            if size(wk2_buf, 1) != rows2 || size(wk2_buf, 2) != cols2
                wk2_buf = Matrix{T}(undef, rows2, cols2)
            end
            mul!(wk2_buf, A[k], wk)
            wk = wk2_buf
            wk = reshape(wk, :, rx1)
            wk = reshape(PermutedDimsArray(wk, (2, 1)), rx1 * ra1[k], :)
            tmp = reshape(WAX1[k], rw1, :)
            cols3 = size(wk, 2)
            if size(wk3_buf, 1) != rows3 || size(wk3_buf, 2) != cols3
                wk3_buf = Matrix{T}(undef, rows3, cols3)
            end
            mul!(wk3_buf, tmp, wk)
            wk = wk3_buf
            wk = reshape(wk, :, b)
            w .+= wk
        else
            A4 = ndims(A[k]) == 4 ? A[k] : reshape(A[k], ra1[k], n, m, ra2[k])
            # WAX1k shape: (rw1, rx1, ra1)
            # WAX2k shape: (rw2, rx2, ra2) - use directly without permutedims
            # A4 shape: (ra1, n, m, ra2)
            # x4 shape: (rx1, m, rx2, b)
            # w4 shape: (rw1, n, rw2, b)
            @tensor w4[rw1i, n_i, rw2i, bi] += WAX1k[rw1i, rx1i, ra1i] * A4[ra1i, n_i, m_i, ra2i] * x4[rx1i, m_i, rx2i, bi] * WAX2k[rw2i, rx2i, ra2i]
        end
    end
    return w
end

function assemble_local_matrix(WAX1::Vector, A::Vector, WAX2::Vector)
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

function assemble_local_vector(WX1::Vector, x::Vector, WX2::Vector)
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

function assemble_local_rjacobi(WAX1::Vector, A::Vector, WAX2::Vector)
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
