function _dr_to_tttensor(X::Matrix{Any}, n::Vector{Int})
    d = size(X, 1)
    cores = Vector{Array{eltype(X[1, 1]), 3}}(undef, d)
    for i in 1:d
        core = X[i, 1]
        r1 = size(core, 1)
        r2 = size(core, 4)
        cores[i] = reshape(core, r1, n[i], r2)
    end
    return TTTensor(cores)
end

function parse_matrix(A, n::Vector{Int}, X::Matrix{Any}, t, vectype)
    d = length(n) - 1
    if !(A isa AbstractArray) && !(A isa TTMatrix)
        if vectype == :tt_tensor
            Xtt = _dr_to_tttensor(X, n)
            A = A(Xtt, t)
        else
            A = A(X, t)
        end
    end

    if A isa TTMatrix
        D = length(A.cores)
    else
        D = size(A, 1)
    end

    if D == d
        _, _, Ra, ras = check_matrix(A, "A"; d=d, n=n[1:d])
        ras = hcat(ras, ones(Int, d + 1, 1))
        ras = vcat(ras, ones(Int, 1, Ra + 1))
    elseif D == d + 1
        _, _, Ra, ras = check_matrix(A, "A"; d=d + 1, n=n)
        ras = hcat(ras, ones(Int, d + 2, 1))
    else
        error("dim(A) must be either d or d+1")
    end

    As = Matrix{Any}(undef, d + 1, Ra + 1)
    for i in 1:(d + 1), k in 1:(Ra + 1)
        As[i, k] = nothing
    end
    if A isa TTMatrix
        cores = core2cell(A)
        for i in 1:D
            As[i, 1] = cores[i]
        end
    else
        for i in 1:D
            for k in 1:Ra
                As[i, k] = A[i, k]
            end
        end
    end

    for i in 1:d
        sparseflag = true
        for k in 1:Ra
            if As[i, k] !== nothing && !issparse(As[i, k])
                sparseflag = false
                break
            end
        end
        T = As[i, 1] === nothing ? Float64 : eltype(As[i, 1])
        if sparseflag
            As[i, Ra + 1] = spdiagm(0 => ones(T, n[i]))
        else
            As[i, Ra + 1] = reshape(Matrix{T}(I, n[i], n[i]), 1, n[i], n[i], 1)
        end
    end

    for k in 1:Ra
        if As[d + 1, k] === nothing
            T = As[1, 1] === nothing ? Float64 : eltype(As[1, 1])
            As[d + 1, k] = Matrix{T}(I, n[d + 1], n[d + 1])
        end
        if n[d + 1] > 1
            As[d + 1, k] = reshape(As[d + 1, k], ras[d + 1, k] * n[d + 1], n[d + 1])
            As[d + 1, k] = sparse(As[d + 1, k])
        else
            As[d + 1, k] = reshape(As[d + 1, k], ras[d + 1, k], 1, 1)
        end
        As[d + 1, k] = -As[d + 1, k]
    end

    # Set up time identity block for the identity matrix column
    T = As[1, 1] === nothing ? Float64 : eltype(As[1, 1])
    As[d + 1, Ra + 1] = Matrix{T}(I, n[d + 1], n[d + 1])
    if n[d + 1] > 1
        As[d + 1, Ra + 1] = reshape(As[d + 1, Ra + 1], ras[d + 1, Ra + 1] * n[d + 1], n[d + 1])
        As[d + 1, Ra + 1] = sparse(As[d + 1, Ra + 1])
    else
        As[d + 1, Ra + 1] = reshape(As[d + 1, Ra + 1], ras[d + 1, Ra + 1], 1, 1)
    end
    As[d + 1, Ra + 1] = -As[d + 1, Ra + 1]

    return As, Ra, ras
end

function parse_rhs(Rhs, n::Vector{Int}, X::Matrix{Any}, t, vectype)
    if Rhs === nothing || (Rhs isa AbstractArray && isempty(Rhs))
        return nothing, 0, 0
    end

    d = length(n) - 1

    if !(Rhs isa AbstractArray) && !(Rhs isa TTTensor)
        if vectype == :tt_tensor
            Xtt = _dr_to_tttensor(X, n)
            Rhs = Rhs(Xtt, t)
        else
            Rhs = Rhs(X, t)
        end
    end

    if Rhs isa TTTensor
        D = length(Rhs.cores)
    else
        D = size(Rhs, 1)
    end

    if D == d
        _, _, Rrhs, rrhs = check_vector(Rhs, "y"; d=d, n=n[1:d])
        rrhs = vcat(rrhs, ones(Int, 1, Rrhs))
    elseif D == d + 1
        _, _, Rrhs, rrhs = check_vector(Rhs, "y"; d=d + 1, n=n)
    else
        error("dim(RHS) must be either d or d+1")
    end

    Rhsc = Matrix{Any}(undef, d + 1, Rrhs)
    for i in 1:(d + 1), k in 1:Rrhs
        Rhsc[i, k] = nothing
    end
    if Rhs isa TTTensor
        cores = core2cell(Rhs)
        for k in 1:D
            core = cores[k]
            r1 = size(core, 1)
            r2 = size(core, 3)
            Rhsc[k, 1] = reshape(core, r1, n[k], 1, r2)
        end
    else
        for i in 1:D
            for k in 1:Rrhs
                Rhsc[i, k] = Rhs[i, k]
            end
        end
    end

    for k in 1:Rrhs
        if Rhsc[d + 1, k] === nothing
            T = Rhsc[1, 1] === nothing ? Float64 : eltype(Rhsc[1, 1])
            Rhsc[d + 1, k] = ones(T, 1, n[d + 1], 1, 1)
        elseif ndims(Rhsc[d + 1, k]) == 2
            Rhsc[d + 1, k] = reshape(Rhsc[d + 1, k], 1, n[d + 1], 1, 1)
        end
    end

    return Rhsc, Rrhs, rrhs
end

function tamen(X, A, tol, opts=Dict{Symbol,Any}(), z=nothing)
    d, n, _, rx, vectype = check_vector(X, "x")
    d -= 1

    if X isa AbstractMatrix && size(X, 2) > 1
        X = X[:, end:end]
    end
    if X isa AbstractMatrix && X[1] isa TTTensor
        X = X[1]
    end
    if X isa TTTensor
        cores = core2cell(X)
        Xcells = Matrix{Any}(undef, d + 1, 1)
        r = tt_ranks(X)
        for i in 1:(d + 1)
            Xcells[i, 1] = reshape(cores[i], r[i], n[i], 1, r[i + 1])
        end
        X = Xcells
        rx = r
    end

    get!(opts, :nswp, 20)
    get!(opts, :kickrank, 4)
    get!(opts, :verb, 1)
    get!(opts, :local_iters, 100)
    get!(opts, :trunc_norm, "fro")
    get!(opts, :time_error_damp, 100)
    get!(opts, :time_scheme, "cheb")
    get!(opts, :correct_2_norm, true)
    get!(opts, :obs, nothing)
    get!(opts, :rhs, nothing)

    if !(opts[:local_iters] isa AbstractVector)
        opts[:local_iters] = vcat(fill(opts[:local_iters], d), 0)
    end

    for i in 1:(d - 1)
        X[i + 1, 1], X[i, 1], rx[i + 1] = orthogonalise_block(X[i + 1, 1], X[i, 1], 1)
    end

    Aux = nothing
    raux = nothing
    Raux = 0
    if opts[:obs] !== nothing && !(opts[:obs] isa AbstractArray && isempty(opts[:obs]))
        obs = opts[:obs]
        obs isa AbstractArray || error("Aux vectors must be given in an array")
        Raux = size(obs, 2)
        Aux = Matrix{Any}(undef, d, Raux)
        if obs[1] isa TTTensor
            raux = ones(Int, d + 1, Raux)
            for i in 1:Raux
                obs_tt = obs[i]
                _, _, _, raux[:, i] = check_vector(obs_tt, "aux"; d=d, n=n[1:d])
                cores = core2cell(tt_matrix(obs_tt, n[1:d], ones(Int, d)))
                for j in 1:d
                    Aux[j, i] = cores[j]
                end
            end
        else
            _, _, _, raux = check_vector(obs, "aux"; d=d, n=n[1:d])
            for i in 1:d, k in 1:Raux
                Aux[i, k] = obs[i, k]
            end
        end
    end

    if z === nothing || (z isa AbstractArray && isempty(z))
        z = Matrix{Any}(undef, d + 1, 1)
        if opts[:kickrank] isa Number
            rz = vcat(1, fill(Int(opts[:kickrank]), d), 1)
        elseif length(opts[:kickrank]) == d
            rz = vcat(1, collect(opts[:kickrank]), 1)
        else
            error("number of rz components should be $d")
        end
        T = eltype(X[1, 1])
        for i in (d + 1):-1:2
            z[i, 1] = randn(T, rz[i], n[i], 1, rz[i + 1])
            _, z[i, 1], rz[i] = orthogonalise_block(nothing, z[i, 1], -1)
        end
        z[1, 1] = randn(T, 1, n[1], 1, rz[2])
    end

    X_global = Matrix{Any}(undef, d + 1, 10)
    t_global = Vector{Any}(undef, 10)

    time_scheme = lowercase(String(opts[:time_scheme]))
    if occursin("cheb", time_scheme)
        t_coarse, St = chebdiff(n[d + 1])
        rhst = reshape(sum(St, dims=2), 1, n[d + 1], 1, 1)
        iSt = inv(St)
        t_fine, St2 = chebdiff(n[d + 1] * 2)
        P = cheb2_interpolant(t_coarse, t_fine)
        iSt2 = St2 \ P
    elseif occursin("cn", time_scheme)
        if iseven(n[d + 1])
            error("Crank-Nicolson scheme requires odd number of time points")
        end
        nt = n[d + 1]
        w0 = fill(0.5 / (nt - 1), nt)
        w1 = fill(0.5 / (nt - 1), nt - 1)
        Mt = spdiagm(nt, nt, -1 => w1, 0 => w0)
        Mt[1, 1] = 0
        Gt = spdiagm(nt, nt, -1 => fill(-1.0, nt - 1), 0 => fill(1.0, nt))
        rhst = reshape([1.0; zeros(nt - 1)], 1, nt, 1, 1)
        t_coarse = collect(0:(nt - 1)) / (nt - 1)
        w20 = fill(0.25 / (nt - 1), 2 * nt - 1)
        w21 = fill(0.25 / (nt - 1), 2 * nt - 2)
        Mt2 = spdiagm(2 * nt - 1, 2 * nt - 1, -1 => w21, 0 => w20)
        Mt2[1, 1] = 0
        Gt2 = spdiagm(2 * nt - 1, 2 * nt - 1, -1 => fill(-1.0, 2 * nt - 2), 0 => fill(1.0, 2 * nt - 1))
    else
        error("Only Chebyshev and CN schemes are implemented")
    end

    time_step = 1.0
    time_global = 0.0
    t_cnt = 0
    time_resid = 0.0

    while time_global < 1.0
        if time_global + time_step > 1.0
            time_step = 1.0 - time_global
        end

        rx = vcat([size(X[i, 1], 1) for i in 1:(d + 1)], 1)
        X_initial = deepcopy(X)

        x0 = deepcopy(X)
        x0[d, 1] = reshape(x0[d, 1], rx[d] * n[d], rx[d + 1])
        Xt_mat = reshape(X[d + 1, 1], rx[d + 1], n[d + 1])
        x0[d, 1] = x0[d, 1] * Xt_mat[:, end]
        x0[d, 1] = reshape(x0[d, 1], rx[d], n[d], 1, 1)
        x0norm = opts[:correct_2_norm] ? norm(x0[d, 1]) : 0.0
        x0[d + 1, 1] = rhst

        ZAX = nothing
        ZY = nothing
        reject = false
        dtime_step = 1.0

        for swp in 1:opts[:nswp]
            As, Ra, ras = parse_matrix(A, n, X, time_global .+ t_coarse .* time_step, vectype)
            for k in 1:Ra
                As[1, k] = As[1, k] * time_step
            end
            Rhs, Rrhs, rfs = parse_rhs(opts[:rhs], n, X, time_global .+ t_coarse .* time_step, vectype)
            for k in 1:Rrhs
                Rhs[1, k] = Rhs[1, k] * time_step
            end

            if occursin("cheb", time_scheme)
                As[d + 1, Ra + 1] = reshape(St, 1, n[d + 1], n[d + 1], 1)
            else
                As_true = As[d + 1, 1:Ra]
                for k in 1:Ra
                    As[d + 1, k] = reshape(As[d + 1, k], ras[d + 1, k] * n[d + 1], n[d + 1])
                    As[d + 1, k] = reshape(As[d + 1, k]', n[d + 1], n[d + 1] * ras[d + 1, k])
                    As[d + 1, k] = Mt * As[d + 1, k]
                    As[d + 1, k] = reshape(As[d + 1, k], n[d + 1] * n[d + 1], ras[d + 1, k])
                    As[d + 1, k] = reshape(As[d + 1, k]', ras[d + 1, k] * n[d + 1], n[d + 1])
                    As[d + 1, k] = sparse(As[d + 1, k])
                end
                As[d + 1, Ra + 1] = Gt
                if Rrhs > 0
                    error("Crank-Nicolson RHS is not implemented")
                end
            end

            if Rrhs > 0
                x0rhs = hcat(x0, Rhs)
                rfs = hcat(vcat([size(x0[i, 1], 1) for i in 1:(d + 1)], 1), rfs)
            else
                x0rhs = x0
                rfs = vcat([size(x0[i, 1], 1) for i in 1:(d + 1)], 1)
            end

            # Note: ZAX and ZY are reused across sweeps for efficiency.
            # The amenany_sweep function handles dimension changes internally.

            X, rx, z, ZAX, ZY, opts, errs, resids, XAX, XY, XAUX = amenany_sweep(X, As, x0rhs, z, tol, opts, ZAX, ZY, Aux)
            max_err = maximum(errs)
            max_res = maximum(resids)

            u0 = XY[1, d + 1]
            if Raux > 0 && XAUX !== nothing
                Caux = zeros(eltype(u0), rx[d + 1], sum(raux[d + 1, :]))
                pos = 1
                for k in 1:Raux
                    Caux[:, pos:(pos + raux[d + 1, k] - 1)] = XAUX[k, d + 1]
                    pos += raux[d + 1, k]
                end
                Q, _ = qr(Caux)
                Caux = Matrix(Q)
                u0_save = Caux' * u0
                u0_change = u0 - Caux * u0_save
                if x0norm > 0
                    u0_change = u0_change * sqrt(abs(x0norm^2 - norm(u0_save)^2)) / norm(u0_change)
                end
                u0 = Caux * u0_save + u0_change
            elseif x0norm > 0
                u0 = u0 * (x0norm / norm(u0))
            end

            if occursin("cheb", time_scheme)
                At = zeros(eltype(u0), rx[d + 1] * n[d + 1], rx[d + 1] * n[d + 1])
                for k in 1:Ra
                    Atk = reshape(XAX[k, d + 1], rx[d + 1] * rx[d + 1], ras[d + 1, k])
                    Atk2 = reshape(As[d + 1, k], ras[d + 1, k], n[d + 1] * n[d + 1])
                    Atk2 = Atk * Atk2
                    Atk2 = reshape(Atk2, rx[d + 1], rx[d + 1], n[d + 1], n[d + 1])
                    Atk2 = permutedims(Atk2, (1, 3, 2, 4))
                    Atk2 = reshape(Atk2, rx[d + 1] * n[d + 1], rx[d + 1] * n[d + 1])
                    At .+= Atk2
                end
                AT = Matrix(I, rx[d + 1] * n[d + 1], rx[d + 1] * n[d + 1]) + kron(iSt, Matrix(I, rx[d + 1], rx[d + 1])) * At
                yt = repeat(u0, 1, n[d + 1])
                for k in 1:Rrhs
                    rhs_t = Rhs[d + 1, k]
                    if ndims(rhs_t) == 4
                        rhs_t = reshape(rhs_t, size(rhs_t, 1), n[d + 1])
                    end
                    yt .+= XY[k + 1, d + 1] * rhs_t * iSt'
                end
                yt = reshape(yt, rx[d + 1] * n[d + 1], 1)
                xt = AT \ yt
                X[d + 1, 1] = reshape(xt, rx[d + 1], n[d + 1], 1, 1)

                AT2 = kron(P, Matrix(I, rx[d + 1], rx[d + 1])) + kron(iSt2, Matrix(I, rx[d + 1], rx[d + 1])) * At
                yt = repeat(u0, 1, 2 * n[d + 1])
                for k in 1:Rrhs
                    rhs_t = Rhs[d + 1, k]
                    if ndims(rhs_t) == 4
                        rhs_t = reshape(rhs_t, size(rhs_t, 1), n[d + 1])
                    end
                    yt .+= XY[k + 1, d + 1] * rhs_t * iSt2'
                end
                yt = reshape(yt, rx[d + 1] * n[d + 1] * 2, 1)
                time_resid = norm(AT2 * xt - yt) / norm(yt)
            else
                At = spzeros(ComplexF64, rx[d + 1] * n[d + 1], rx[d + 1] * n[d + 1])
                for k in 1:Ra
                    At2 = As_true[k]
                    if ras[d + 1, k] > 1
                        At2 = reshape(At2, ras[d + 1, k], n[d + 1] * n[d + 1])
                        for j in 1:ras[d + 1, k]
                            At .+= kron(reshape(At2[j, :], n[d + 1], n[d + 1]), sparse(XAX[k, d + 1][:, :, j]))
                        end
                    else
                        XAXk = XAX[k, d + 1]
                        if ndims(XAXk) == 3
                            XAXk = XAXk[:, :, 1]
                        end
                        At .+= kron(At2, sparse(XAXk))
                    end
                end
                AT = kron(Gt, spdiagm(0 => ones(rx[d + 1]))) + kron(Mt, spdiagm(0 => ones(rx[d + 1]))) * At
                yt = vcat(u0, zeros(eltype(u0), rx[d + 1] * (n[d + 1] - 1)))
                xt = AT \ yt
                X[d + 1, 1] = reshape(xt, rx[d + 1], n[d + 1], 1, 1)

                xt2 = At * xt
                xt2 = reshape(xt2, rx[d + 1], n[d + 1])
                yt2 = zeros(eltype(u0), rx[d + 1], 2 * n[d + 1] - 1)
                yt2[:, 1:2:(2 * n[d + 1] - 1)] = xt2
                yt2[:, 2:2:(2 * n[d + 1] - 2)] = (xt2[:, 1:(n[d + 1] - 1)] + xt2[:, 2:n[d + 1]]) * 0.5
                yt2 = yt2 * Mt2'
                yt2 = yt2 / Gt2'
                xt2 = zeros(eltype(u0), rx[d + 1], 2 * n[d + 1] - 1)
                xt = reshape(xt, rx[d + 1], n[d + 1])
                xt2[:, 1:2:(2 * n[d + 1] - 1)] = xt
                xt2[:, 2:2:(2 * n[d + 1] - 2)] = (xt[:, 1:(n[d + 1] - 1)] + xt[:, 2:n[d + 1]]) * 0.5
                time_resid = norm(xt2 .+ yt2 .- u0 * ones(1, 2 * n[d + 1] - 1)) / norm(u0) / sqrt(2 * n[d + 1] - 1)
            end

            if opts[:verb] > 0
                println("tamen: t=$(time_global + time_step), swp=$swp, err=$(max_err), res=$(max_res), time_res=$(time_resid), rank=$(maximum(rx))")
            end

            dtime_step = tol / (opts[:time_error_damp] * time_resid)
            if dtime_step < 1 && swp > 1
                X = X_initial
                reject = true
                break
            end

            if swp >= opts[:nswp] || (opts[:trunc_norm] == "fro" && max_err < tol) || (opts[:trunc_norm] != "fro" && max_res < tol)
                break
            end
        end

        if opts[:verb] > 0
            println("\t t=$(time_global + time_step), time_res=$(time_resid), rank=$(maximum(rx))")
        end
        if !reject
            t_cnt += 1
            if t_cnt > size(X_global, 2)
                X_global = hcat(X_global, Matrix{Any}(undef, d + 1, 10))
                t_global = vcat(t_global, Vector{Any}(undef, 10))
            end
            X_global[:, t_cnt] = X
            t_global[t_cnt] = time_global .+ t_coarse .* time_step
            time_global += time_step
        end

        if occursin("cheb", time_scheme)
            time_step = time_step * min(0.5 * dtime_step^(1 / n[d + 1]), 2.0)
        else
            time_step = time_step * min(0.5 * sqrt(dtime_step), 2.0)
        end
        if opts[:verb] > 0
            if reject
                println("new time step=$(time_step) !REJECT!")
            else
                println("new time step=$(time_step)")
            end
        end
    end

    X_global = X_global[:, 1:t_cnt]
    t_global = t_global[1:t_cnt]

    if vectype == :tt_tensor
        for k in 1:t_cnt
            X_global[1, k] = _dr_to_tttensor(X_global[:, k:k], n)
        end
        X_global = X_global[1, :]
        if t_cnt == 1
            X_global = X_global[1]
        end
    end

    x_out = nothing
    if true
        Xt_mat = reshape(X[d + 1, 1], rx[d + 1], n[d + 1])
        Xt = Xt_mat[:, end]
        x_out = X[1:d, 1:1]
        x_out[d, 1] = reshape(x_out[d, 1], rx[d] * n[d], rx[d + 1])
        x_out[d, 1] = x_out[d, 1] * Xt
        x_out[d, 1] = reshape(x_out[d, 1], rx[d], n[d])
        rx[d + 1] = 1
        for i in 1:d
            x_out[i, 1] = reshape(x_out[i, 1], rx[i], n[i], 1, rx[i + 1])
        end
        if vectype == :tt_tensor
            x_out = _dr_to_tttensor(x_out, n[1:d])
        end
    end

    opts[:local_iters] = opts[:local_iters][1]
    return X_global, t_global, opts, x_out, z
end
