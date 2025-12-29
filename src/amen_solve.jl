function _dr_from_tt_matrix(A::TTMatrix)
    d = length(A.cores)
    cells = Matrix{Any}(undef, d, 1)
    for i in 1:d
        cells[i, 1] = A.cores[i]
    end
    return cells
end

function _dr_to_tt_tensor(x::Matrix{Any})
    d = size(x, 1)
    cores = Vector{Array{eltype(x[1, 1]), 3}}(undef, d)
    for i in 1:d
        core = x[i, 1]
        r1, n, _, r2 = size(core)
        cores[i] = reshape(core, r1, n, r2)
    end
    return TTTensor(cores)
end

function amen_solve(A, y, tol, opts=Dict{Symbol,Any}(), x0=nothing, aux=nothing, z=nothing)
    d, n, _, _, vectype = check_vector(y, "y")
    if y isa TTTensor
        y = dr_from_tt_tensor(y)
    end

    check_matrix(A, "A"; d=d, n=n)
    if A isa TTMatrix
        A = _dr_from_tt_matrix(A)
    end

    if x0 === nothing || (x0 isa AbstractArray && isempty(x0))
        x = Matrix{Any}(undef, d, 1)
        rx = vcat(1, fill(2, d - 1), 1)
        T = eltype(y[1, 1])
        for i in d:-1:2
            x[i, 1] = randn(T, rx[i], n[i], 1, rx[i + 1])
            x[i - 1, 1], x[i, 1], rx[i] = orthogonalise_block(nothing, x[i, 1], -1)
        end
        x[1, 1] = randn(T, 1, n[1], 1, rx[2])
    else
        x = x0
        check_vector(x, "x"; d=d, n=n)
        if x isa TTTensor
            x = dr_from_tt_tensor(x)
        end
    end

    if aux !== nothing && !(aux isa AbstractArray && isempty(aux))
        aux isa AbstractArray || error("Aux vectors must be given in an array")
        if aux isa AbstractVector && aux[1] isa TTTensor
            Raux = length(aux)
            aux_in = Matrix{Any}(undef, d, Raux)
            for i in 1:Raux
                aux_tt = aux[i]
                check_vector(aux_tt, "aux"; d=d, n=n)
                aux_in[:, i] = dr_from_tt_tensor(aux_tt)
            end
            aux = aux_in
        else
            check_vector(aux, "aux"; d=d, n=n)
        end
    else
        aux = nothing
    end

    get!(opts, :nswp, 20)
    get!(opts, :kickrank, 4)
    get!(opts, :verb, 1)
    get!(opts, :trunc_norm, "fro")

    if z === nothing || (z isa AbstractArray && isempty(z))
        z = Matrix{Any}(undef, d, 1)
        rz = vcat(1, fill(opts[:kickrank], d - 1), 1)
        if opts[:kickrank] == 0
            rz .= 0
        end
        T = eltype(y[1, 1])
        for i in d:-1:2
            z[i, 1] = randn(T, rz[i], n[i], 1, rz[i + 1])
            z[i - 1, 1], z[i, 1], rz[i] = orthogonalise_block(nothing, z[i, 1], -1)
        end
        z[1, 1] = randn(T, 1, n[1], 1, rz[2])
    end

    ZAX = nothing
    ZY = nothing

    swp = 0
    for s in 1:opts[:nswp]
        x, rx, z, ZAX, ZY, opts, errs, resids, _, _, _ = amenany_sweep(x, A, y, z, tol, opts, ZAX, ZY, aux)
        swp = s
        max_err = maximum(errs)
        max_res = maximum(resids)
        if opts[:verb] > 0
            println("amen_solve: swp=$s, err=$(max_err), res=$(max_res), rank=$(maximum(rx))")
        end
        if (opts[:trunc_norm] == "fro" && max_err < tol) || (opts[:trunc_norm] != "fro" && max_res < tol)
            break
        end
    end

    if vectype == :tt_tensor
        x = _dr_to_tt_tensor(x)
    end

    return x, opts, swp, z
end

function amenany_sweep(x, A, y, z, tol, opts, ZAX=nothing, ZY=nothing, aux=nothing)
    d, n, _, rx, _ = check_vector(x, "x")
    _, _, Ra, ra = check_matrix(A, "a"; d=d, n=n)
    _, _, Ry, _ = check_vector(y, "y"; d=d, n=n)
    _, _, Rz, rz = check_vector(z, "z"; d=d, n=n)
    _, _, Raux, raux = check_vector(aux, "o")

    loctol = tol / sqrt(d)

    get!(opts, :resid_damp, 4)
    get!(opts, :rmax, Inf)
    get!(opts, :max_full_size, 50)
    get!(opts, :local_iters, 100)
    get!(opts, :trunc_norm, "fro")
    get!(opts, :kickrank2, 0)
    get!(opts, :verb, 1)
    get!(opts, :local_prec, "")

    T = eltype(x[1, 1])

    XAX = Array{Any}(undef, Ra, d + 1)
    XY = Array{Any}(undef, Ry, d + 1)
    XAUX = Raux > 0 ? Array{Any}(undef, Raux, d + 1) : nothing

    for k in 1:Ra
        XAX[k, 1] = ones(T, 1, 1, 1)
        XAX[k, d + 1] = ones(T, 1, 1, 1)
        for i in 2:d
            XAX[k, i] = nothing
        end
    end
    for k in 1:Ry
        XY[k, 1] = ones(T, 1, 1)
        XY[k, d + 1] = ones(T, 1, 1)
        for i in 2:d
            XY[k, i] = nothing
        end
    end
    if Raux > 0
        for k in 1:Raux
            XAUX[k, 1] = ones(T, 1, 1)
            XAUX[k, d + 1] = ones(T, 1, 1)
            for i in 2:d
                XAUX[k, i] = nothing
            end
        end
    end

    if ZAX === nothing
        ZAX = similar(XAX)
        for k in 1:Ra
            ZAX[k, 1] = ones(T, 1, 1, 1)
            ZAX[k, d + 1] = ones(T, 1, 1, 1)
            for i in 2:d
                ZAX[k, i] = nothing
            end
        end
    end
    if ZY === nothing
        ZY = similar(XY)
        for k in 1:Ry
            ZY[k, 1] = ones(T, 1, 1)
            ZY[k, d + 1] = ones(T, 1, 1)
            for i in 2:d
                ZY[k, i] = nothing
            end
        end
    end

    for i in d:-1:2
        if Rz > 0
            if ZY[1, i] === nothing
                crzy = reshape(z[i, 1], rz[i] * n[i] * rz[i + 1], 1)
            else
                crzy = assemble_local_vector(vec(ZY[:, i]), collect(y[i, :]), vec(ZY[:, i + 1]))
            end
            if ZAX[1, i] === nothing
                crzAx = zeros(T, rz[i] * n[i] * rz[i + 1], 1)
            else
                Ai = collect(A[i, :])
                for k in 1:Ra
                    if ndims(Ai[k]) == 4
                        Ai[k] = reshape(Ai[k], ra[i, k] * n[i], n[i] * ra[i + 1, k])
                    end
                end
                crzAx = local_matvec(x[i, 1], rx[i], n[i], rx[i + 1], 1, rz[i], n[i], rz[i + 1], vec(ZAX[:, i]), Ai, vec(ZAX[:, i + 1]), Ra, ra[i, :], ra[i + 1, :])
            end
            crz = crzy - crzAx
            crz = reshape(crz, rz[i], n[i] * rz[i + 1])
            if opts[:kickrank2] > 0
                U, S, V = svd(crz; full=false)
                crz = V[:, 1:max(rz[i] - opts[:kickrank2], 1)]'
                crz = vcat(crz, randn(T, opts[:kickrank2], n[i] * rz[i + 1]))
            end
            crz = reshape(crz, rz[i], n[i], 1, rz[i + 1])
            z[i - 1, 1], z[i, 1], rz[i] = orthogonalise_block(z[i - 1, 1], crz, -1)
        end

        x[i - 1, 1], x[i, 1], rx[i] = orthogonalise_block(x[i - 1, 1], x[i, 1], -1)

        XAX[:, i] = rightreduce_matrix(vec(XAX[:, i + 1]), x[i, 1], collect(A[i, :]), x[i, 1])
        XY[:, i] = rightreduce_vector(vec(XY[:, i + 1]), x[i, 1], collect(y[i, :]))
        if Rz > 0
            ZAX[:, i] = rightreduce_matrix(vec(ZAX[:, i + 1]), z[i, 1], collect(A[i, :]), x[i, 1])
            ZY[:, i] = rightreduce_vector(vec(ZY[:, i + 1]), z[i, 1], collect(y[i, :]))
        end
    end

    errs = zeros(Float64, d)
    resids = zeros(Float64, d)

    for i in 1:d
        sol_prev = reshape(x[i, 1], rx[i] * n[i] * rx[i + 1], 1)
        lociters = opts[:local_iters]
        if lociters isa AbstractVector
            lociters = lociters[i]
        end
        rhs = assemble_local_vector(vec(XY[:, i]), collect(y[i, :]), vec(XY[:, i + 1]))
        norm_rhs = norm(rhs)

        XAX1 = vec(XAX[:, i])
        Ai = collect(A[i, :])
        XAX2 = vec(XAX[:, i + 1])
        ra1 = ra[i, :]
        ra2 = ra[i + 1, :]
        for k in 1:Ra
            if ndims(Ai[k]) == 4
                Ai[k] = reshape(Ai[k], ra1[k] * n[i], n[i] * ra2[k])
            end
        end
        mvfun = xvec -> local_matvec(xvec, rx[i], n[i], rx[i + 1], 1, rx[i], n[i], rx[i + 1], XAX1, Ai, XAX2, Ra, ra1, ra2)
        resids[i] = norm(rhs - mvfun(sol_prev)) / norm_rhs

        if lociters > 0
            if rx[i] * n[i] * rx[i + 1] < opts[:max_full_size]
                B, sparseflag = assemble_local_matrix(XAX1, Ai, XAX2)
                rhs_local = rhs
                if sparseflag
                    rhs_local = reshape(rhs, rx[i] * n[i], rx[i + 1])
                    rhs_local = permutedims(rhs_local, (2, 1))
                    rhs_local = reshape(rhs_local, rx[i + 1] * rx[i] * n[i], 1)
                end
                sol = B \ rhs_local
                if sparseflag
                    sol = reshape(sol, rx[i + 1], rx[i] * n[i])
                    sol = permutedims(sol, (2, 1))
                    sol = reshape(sol, rx[i] * n[i] * rx[i + 1], 1)
                end
            else
                tol_loc = max(loctol / opts[:resid_damp], eps(Float64) * 1e3)
                sol, flg, _, _ = bicgstab(mvfun, rhs; x0=sol_prev, tol=tol_loc, maxiter=lociters)
                if flg > 0
                    precfun = nothing
                    if opts[:local_prec] == "r"
                        P = assemble_local_rjacobi(XAX1, Ai, XAX2)
                        precfun = xvec -> local_precvec(xvec, rx[i], n[i], rx[i + 1], 1, P)
                    end
                    sol, _, _, _ = bicgstab(mvfun, rhs; x0=sol, tol=tol_loc, maxiter=lociters, M=precfun)
                end
            end
        else
            sol = sol_prev
        end

        errs[i] = norm(sol - sol_prev) / norm(sol)

        sol = reshape(sol, rx[i] * n[i], rx[i + 1])
        if loctol > 0 && i < d && (Rz + Raux) > 0
            U, S, V = svd(sol; full=false)
            s = S
            if opts[:trunc_norm] == "fro" || lociters == 0
                sum_s = cumsum(reverse(s .^ 2))
                target = (loctol * norm(s))^2
                r = findfirst(x -> x >= target, sum_s)
                r = r === nothing ? length(s) : (length(s) - r + 1)
            else
                res_new = norm(rhs - mvfun(sol)) / norm_rhs
                r = min(rx[i] * n[i], rx[i + 1])
                r = max(r - rz[i + 1, 1], 1)
                cursol = U[:, 1:r] * Diagonal(s[1:r]) * V[:, 1:r]'
                res = norm(rhs - mvfun(cursol)) / norm_rhs
                drank = res < max(loctol, res_new * opts[:resid_damp]) ? -1 : 1
                while r > 0 && r <= length(s)
                    cursol = U[:, 1:r] * Diagonal(s[1:r]) * V[:, 1:r]'
                    res = norm(rhs - mvfun(cursol)) / norm_rhs
                    if drank > 0
                        if res < max(loctol, res_new * opts[:resid_damp])
                            break
                        end
                    else
                        if res >= max(loctol, res_new * opts[:resid_damp])
                            break
                        end
                    end
                    r += drank
                end
                if drank < 0
                    r += 1
                end
                r = min(r, length(s))
            end
            r = min(r, length(s))
            if isfinite(opts[:rmax])
                r = min(r, Int(opts[:rmax]))
            end
            u = U[:, 1:r]
            v = Diagonal(s[1:r]) * V[:, 1:r]'
            sol = u * v
        else
            Q, R = qr(sol)
            u = Matrix(Q)
            v = Matrix(R)
            r = size(u, 2)
        end

        if opts[:verb] > 1
            println("Rank: $r")
        end

        if Rz > 0
            crzy = assemble_local_vector(vec(ZY[:, i]), collect(y[i, :]), vec(ZY[:, i + 1]))
            crzAx = local_matvec(reshape(sol, rx[i] * n[i] * rx[i + 1], 1), rx[i], n[i], rx[i + 1], 1, rz[i], n[i], rz[i + 1], vec(ZAX[:, i]), Ai, vec(ZAX[:, i + 1]), Ra, ra1, ra2)
            crz = crzy - crzAx
            crz = reshape(crz, rz[i] * n[i], rz[i + 1])
            if opts[:kickrank2] > 0 && i < d
                U2, _, _ = svd(crz; full=false)
                crz = U2[:, 1:max(rz[i + 1, 1] - opts[:kickrank2], 1)]
                crz = hcat(crz, randn(T, rz[i] * n[i], opts[:kickrank2]))
            end
            rzold = rz[i + 1]
            crz = reshape(crz, rz[i], n[i], 1, rz[i + 1])
            _, z[i, 1], rz[i + 1] = orthogonalise_block(nothing, crz, 1)
        end

        if i < d
            if Rz > 0
                crzy = assemble_local_vector(vec(XY[:, i]), collect(y[i, :]), vec(ZY[:, i + 1]))
                crzAx = local_matvec(reshape(sol, rx[i] * n[i] * rx[i + 1], 1), rx[i], n[i], rx[i + 1], 1, rx[i], n[i], rzold, XAX1, Ai, vec(ZAX[:, i + 1]), Ra, ra1, ra2)
                crz = crzy - crzAx
                crz = reshape(crz, rx[i] * n[i], rzold)
                u = hcat(u, crz)
            end
            for k in 1:Raux
                craux = reshape(aux[i, k], raux[i, k], n[i] * raux[i + 1, k])
                craux = XAUX[k, i] * craux
                craux = reshape(craux, rx[i] * n[i], raux[i + 1, k])
                u = hcat(u, craux)
            end
            if size(u, 2) > r
                Q, R = qr(u)
                u = Matrix(Q)
                rv = Matrix(R)
                rvx = rv[:, 1:r]
                if Rz > 0
                    rv = rv[:, (r + rzold + 1):end]
                else
                    rv = rv[:, (r + 1):end]
                end
                for k in 1:Raux
                    XAUX[k, i + 1] = rv[:, 1:raux[i + 1, k]]
                    rv = rv[:, (raux[i + 1, k] + 1):end]
                end
                rv = rvx
                v = rv * v
                r = size(u, 2)
            end
            cr2 = reshape(x[i + 1, 1], rx[i + 1], n[i + 1] * rx[i + 2])
            cr2 = v * cr2
            rx[i + 1] = r
            x[i + 1, 1] = reshape(cr2, rx[i + 1], n[i + 1], 1, rx[i + 2])
            x[i, 1] = reshape(u, rx[i], n[i], 1, rx[i + 1])

            XAX[:, i + 1] = leftreduce_matrix(vec(XAX[:, i]), x[i, 1], Ai, x[i, 1])
            XY[:, i + 1] = leftreduce_vector(vec(XY[:, i]), x[i, 1], collect(y[i, :]))
            if Rz > 0
                ZAX[:, i + 1] = leftreduce_matrix(vec(ZAX[:, i]), z[i, 1], Ai, x[i, 1])
                ZY[:, i + 1] = leftreduce_vector(vec(ZY[:, i]), z[i, 1], collect(y[i, :]))
            end
        else
            x[i, 1] = reshape(sol, rx[i], n[i], 1, rx[i + 1])
        end
    end

    return x, rx, z, ZAX, ZY, opts, errs, resids, XAX, XY, XAUX
end

function bicgstab(mvfun, b; x0=nothing, tol=1e-6, maxiter=100, M=nothing)
    n = length(b)
    x = x0 === nothing ? zeros(eltype(b), n) : copy(vec(x0))
    r = b - mvfun(x)
    r_hat = copy(r)
    rho = one(eltype(b))
    alpha = one(eltype(b))
    omega = one(eltype(b))
    v = zeros(eltype(b), n)
    p = zeros(eltype(b), n)
    res0 = norm(r)
    if res0 == 0
        return x, 0, 0.0, 0
    end
    for iter in 1:maxiter
        rho_new = dot(r_hat, r)
        if rho_new == 0
            return x, 1, norm(r) / res0, iter
        end
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        p_hat = M === nothing ? p : M(p)
        v = mvfun(p_hat)
        alpha = rho_new / dot(r_hat, v)
        s = r - alpha * v
        if norm(s) < tol * res0
            x = x + alpha * p_hat
            return x, 0, norm(s) / res0, iter
        end
        s_hat = M === nothing ? s : M(s)
        t = mvfun(s_hat)
        omega = dot(t, s) / dot(t, t)
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        if norm(r) < tol * res0
            return x, 0, norm(r) / res0, iter
        end
        rho = rho_new
    end
    return x, 2, norm(r) / res0, maxiter
end
