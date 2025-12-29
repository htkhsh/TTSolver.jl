function extract_snapshot(X, t, tx, scheme::AbstractString="cheb")
    tx_vec = tx isa Number ? [tx] : collect(tx)
    nt = map(length, t)
    ind = ones(Int, length(tx_vec))
    for k in length(t):-1:2
        left = t[k - 1][nt[k - 1]]
        right = t[k][nt[k]]
        for i in 1:length(tx_vec)
            if tx_vec[i] > left && tx_vec[i] <= right
                ind[i] = k
            end
        end
    end

    out_cells = Vector{Any}(undef, length(tx_vec))
    for i in 1:length(tx_vec)
        if X isa TTTensor
            d = length(X.cores) - 1
            Xcells = dr_from_tt_tensor(X)
            x_cells = Xcells[1:d, 1]
            Xt = Xcells[d + 1, 1]
        elseif X isa AbstractVector && !isempty(X) && X[1] isa TTTensor
            d = length(X[ind[i]].cores) - 1
            Xcells = dr_from_tt_tensor(X[ind[i]])
            x_cells = Xcells[1:d, 1]
            Xt = Xcells[d + 1, 1]
        else
            d = size(X, 1) - 1
            x_cells = X[1:d, ind[i]]
            Xt = X[d + 1, ind[i]]
        end

        tloc = t[ind[i]]
        if occursin("cheb", lowercase(scheme))
            p = vec(cheb2_interpolant(tloc, [tx_vec[i]]))
            if ndims(Xt) == 2
                Xt = Xt * p
            else
                r1, n_t, _, r2 = size(Xt)
                Xt = reshape(Xt, r1, n_t, r2)
                @tensor Xt2[r1i, r2i] := Xt[r1i, n_i, r2i] * p[n_i]
                Xt = Xt2
            end
        elseif occursin("cn", lowercase(scheme))
            if ndims(Xt) > 2
                r1, n_t, _, r2 = size(Xt)
                Xt = reshape(Xt, r1, n_t * r2)
                r2 == 1 || error("CN interpolation expects rank-1 time core")
            end
            k = findfirst(x -> tx_vec[i] <= x, tloc)
            k = k === nothing ? length(tloc) : (k - 1)
            k = max(k, 1)
            denom = tloc[k + 1] - tloc[k]
            p1 = (tx_vec[i] - tloc[k]) / denom
            p2 = (tloc[k + 1] - tx_vec[i]) / denom
            Xt = Xt[:, k] * p2 + Xt[:, k + 1] * p1
        else
            error("Only Chebyshev and CN schemes are implemented")
        end

        r1, n_d, m_d, r2 = size(x_cells[d])
        xmat = reshape(x_cells[d], r1 * n_d * m_d, r2)
        xmat = xmat * Xt
        x_cells[d] = reshape(xmat, r1, n_d, m_d, 1)

        out_cells[i] = x_cells
    end

    if length(tx_vec) == 1
        if X isa TTTensor || (X isa AbstractVector && !isempty(X) && X[1] isa TTTensor)
            n = [size(out_cells[1][i], 2) for i in 1:length(out_cells[1])]
            return _dr_to_tttensor(reshape(out_cells[1], :, 1), n)
        end
        return reshape(out_cells[1], :, 1)
    end

    if X isa TTTensor || (X isa AbstractVector && !isempty(X) && X[1] isa TTTensor)
        return [begin
                    n = [size(out_cells[i][j], 2) for j in 1:length(out_cells[i])]
                    _dr_to_tttensor(reshape(out_cells[i], :, 1), n)
                end for i in 1:length(out_cells)]
    end

    d = size(out_cells[1], 1)
    out = Matrix{Any}(undef, d, length(tx_vec))
    for i in 1:length(tx_vec)
        out[:, i] = out_cells[i]
    end
    return out
end
