function check_vector(x, xname::AbstractString="x"; d=nothing, n=nothing, Rx=nothing, rx=nothing)
    if x === nothing
        return 0, Int[], 0, Array{Int}(undef, 0, 0), :none
    end

    if x isa TTTensor
        d_in = length(x.cores)
        n_in = tt_dims(x)
        rx_in = tt_ranks(x)
        if d !== nothing && d_in != d
            error("dim of $xname is inconsistent")
        end
        if n !== nothing && (length(n_in) != length(n) || any(n_in .!= n))
            error("mode sizes of $xname are inconsistent")
        end
        if Rx !== nothing && Rx != 1
            error("canonical rank of $xname is inconsistent")
        end
        if rx !== nothing && (length(rx_in) != length(rx) || any(rx_in .!= rx))
            error("TT ranks of $xname are inconsistent")
        end
        return d_in, n_in, 1, rx_in, :tt_tensor
    end

    x isa AbstractMatrix || error("$xname must be a TTTensor or d x R cell array")
    Rx_in = size(x, 2)
    if Rx !== nothing && Rx_in != Rx
        error("canonical rank of $xname is inconsistent")
    end

    if (xname == "x" || xname == "z") && Rx_in > 1
        return check_vector(x[:, end:end], xname; d=d, n=nothing, Rx=nothing, rx=(rx === nothing ? nothing : rx[:, end:end]))
    end

    d_in = size(x, 1)
    if d !== nothing && d_in != d
        error("dim of $xname is inconsistent")
    end

    n_in = n === nothing ? ones(Int, d_in) : copy(n)
    if n !== nothing && length(n) != d_in
        error("mode sizes of $xname are inconsistent")
    end

    rx_in = rx === nothing ? ones(Int, d_in + 1, Rx_in) : copy(rx)
    if rx !== nothing && (size(rx, 1) != d_in + 1 || size(rx, 2) != Rx_in)
        error("TT ranks of $xname are inconsistent")
    end

    for k in 1:Rx_in
        for i in 1:d_in
            core = x[i, k]
            core isa AbstractArray || error("$xname{$i,$k} must be an array")
            ndims(core) == 4 || error("$xname{$i,$k} must be a 4D core")
            n_in[i] = size(core, 2) * size(core, 3)
            if n !== nothing && n_in[i] != n[i]
                error("mode size ($i,$k) of $xname is inconsistent")
            end
            rx_in[i + 1, k] = size(core, 4)
            if rx_in[i, k] != size(core, 1)
                error("TT rank ($i,$k) of $xname is inconsistent")
            end
            if rx !== nothing && rx_in[i + 1, k] != rx[i + 1, k]
                error("TT rank ($i,$k) of $xname is inconsistent")
            end
        end
    end

    if minimum(rx_in) == 0
        return d_in, n_in, 0, rx_in, :dR
    end
    return d_in, n_in, Rx_in, rx_in, :dR
end

function check_matrix(a, aname::AbstractString="a"; d=nothing, n=nothing, Ra=nothing, ra=nothing)
    if a === nothing
        return 0, Int[], 0, Array{Int}(undef, 0, 0)
    end

    if a isa TTMatrix
        d_in = length(a.cores)
        n_in, m_in = tt_dims(a)
        ra_in = tt_ranks(a)
        if d !== nothing && d_in != d
            error("dim of $aname is inconsistent")
        end
        if n !== nothing && (length(n_in) != length(n) || any(n_in .!= n) || any(m_in .!= n))
            error("mode sizes of $aname are inconsistent")
        end
        if Ra !== nothing && Ra != 1
            error("canonical rank of $aname is inconsistent")
        end
        if ra !== nothing && (length(ra_in) != length(ra) || any(ra_in .!= ra))
            error("TT ranks of $aname are inconsistent")
        end
        return d_in, n_in, 1, ra_in
    end

    a isa AbstractMatrix || error("$aname must be a TTMatrix or d x R cell array")
    d_in = size(a, 1)
    if d !== nothing && d_in != d
        error("dim of $aname is inconsistent")
    end

    Ra_in = size(a, 2)
    if Ra !== nothing && Ra_in != Ra
        error("canonical rank of $aname is inconsistent")
    end

    n_in = n === nothing ? ones(Int, d_in) : copy(n)
    if n !== nothing && length(n) != d_in
        error("mode sizes of $aname are inconsistent")
    end

    ra_in = ra === nothing ? ones(Int, d_in + 1, Ra_in) : copy(ra)
    if ra !== nothing && (size(ra, 1) != d_in + 1 || size(ra, 2) != Ra_in)
        error("TT ranks of $aname are inconsistent")
    end

    for k in 1:Ra_in
        for i in 1:d_in
            core = a[i, k]
            core isa AbstractArray || error("$aname{$i,$k} must be an array")
            if core isa SparseMatrixCSC
                n1, n2 = size(core)
                if n !== nothing
                    n_i = n[i]
                    n1 % n_i == 0 || error("$aname{$i,$k} is sparse, sizes not divisible by n")
                    n2 % n_i == 0 || error("$aname{$i,$k} is sparse, sizes not divisible by n")
                    r1 = n1 ÷ n_i
                    r2 = n2 ÷ n_i
                    if ra !== nothing && (ra[i + 1, k] != r2 || ra[i, k] != r1)
                        error("TT rank ($i,$k) of $aname is inconsistent")
                    end
                    ra_in[i, k] = r1
                    ra_in[i + 1, k] = r2
                    n_in[i] = n_i
                else
                    n_in[i] = n1
                    n2 == n_in[i] || error("block ($i,$k) of $aname is not square")
                    ra_in[i, k] = 1
                    ra_in[i + 1, k] = 1
                    if ra !== nothing && (ra[i + 1, k] != ra_in[i, k] || ra[i, k] != ra_in[i + 1, k])
                        error("TT rank ($i,$k) of $aname is inconsistent")
                    end
                end
            else
                ndims(core) == 4 || error("$aname{$i,$k} must be a 4D core")
                n_in[i] = size(core, 2)
                size(core, 3) == n_in[i] || error("mode size ($i,$k) of $aname is inconsistent")
                if n !== nothing && n_in[i] != n[i]
                    error("mode size ($i,$k) of $aname is inconsistent")
                end
                ra_in[i + 1, k] = size(core, 4)
                if ra_in[i, k] != size(core, 1)
                    error("TT rank ($i,$k) of $aname is inconsistent")
                end
                if ra !== nothing && ra_in[i + 1, k] != ra[i + 1, k]
                    error("TT rank ($i,$k) of $aname is inconsistent")
                end
            end
        end
    end

    return d_in, n_in, Ra_in, ra_in
end

function dr_from_tt_tensor(x::TTTensor)
    n = tt_dims(x)
    d = length(n)
    cores = Matrix{Any}(undef, d, 1)
    for i in 1:d
        core = x.cores[i]
        r1 = size(core, 1)
        r2 = size(core, 3)
        cores[i, 1] = reshape(core, r1, n[i], 1, r2)
    end
    return cores
end

function dot_dR(x, y)
    d, n, Rx, rx, xtype = check_vector(x, "x")
    dy, ny, Ry, ry, ytype = check_vector(y, "y")
    d == dy || error("Sizes of x and y should be the same")
    all(n .== ny) || error("Sizes of x and y should be the same")

    x_cells = xtype == :tt_tensor ? dr_from_tt_tensor(x) : x
    y_cells = ytype == :tt_tensor ? dr_from_tt_tensor(y) : y

    S = zero(eltype(y_cells[1, 1]))
    for m in 1:Ry
        for k in 1:Rx
            Skm = ones(eltype(y_cells[1, 1]), 1, 1)
            for i in 1:d
                xcore = reshape(x_cells[i, k], rx[i, k], n[i], rx[i + 1, k])
                ycore = reshape(y_cells[i, m], ry[i, m], n[i], ry[i + 1, m])
                @tensor Skm2[rxp, ryp] := Skm[rx, ry] * xcore[rx, n_i, rxp] * ycore[ry, n_i, ryp]
                Skm = Skm2
            end
            S += Skm[1, 1]
        end
    end
    return S
end

function expand_dR(y)
    d, n, Ry, ry, ytype = check_vector(y, "y")
    m = ones(Int, d)

    y_cells = ytype == :tt_tensor ? dr_from_tt_tensor(y) : y
    if ytype != :tt_tensor
        for i in 1:d
            core = y_cells[i, 1]
            core isa SparseMatrixCSC && error("Expansion of sparse matrices is not allowed yet")
            m[i] = size(core, 3)
        end
    end

    Y = nothing
    for k in 1:Ry
        Yk = ones(eltype(y_cells[1, 1]), 1, 1)
        ncum = 1
        mcum = 1
        for i in 1:d
            core = y_cells[i, k]
            yi = reshape(core, ry[i, k], n[i] * m[i] * ry[i + 1, k])
            Yk = Yk * yi
            Yk = reshape(Yk, ncum, mcum, n[i], m[i] * ry[i + 1, k])
            Yk = permutedims(Yk, (1, 3, 2, 4))
            ncum *= n[i]
            mcum *= m[i]
            Yk = reshape(Yk, ncum * mcum, ry[i + 1, k])
        end
        Yk = reshape(Yk, ncum, mcum, size(Yk, 2))
        Y = Y === nothing ? Yk : Y + Yk
    end
    return Y
end
