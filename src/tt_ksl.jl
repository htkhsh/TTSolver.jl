"""
    tt_ksl(x0, A, y, tau; kwargs...)

Linear KSL (Projector-Splitting) time scheme in the TT format.

Integrates the system `dx/dt + Ax = y` on `t ∈ [0, τ]`.

Uses matrix exponential as the local solver. The matrix `A` is assumed to be 
time-independent, and R.H.S. `y` is given at the middle time point.

# Arguments
- `x0::TTTensor`: Initial condition
- `A::TTMatrix`: Time-independent operator
- `y::TTTensor`: Right-hand side (source term)
- `tau::Real`: Time step size

# Keyword Arguments
- `tol::Real=1e-12`: Tolerance for exponential computation
- `rmax::Int=typemax(Int)`: Maximum TT-rank (for rank truncation after SVD)
- `passes::Int=2`: Number of half-sweeps (1 = forward only, 2 = forward + backward)
- `expmethod::Symbol=:krylov`: Method for matrix exponential (`:krylov` or `:taylor`)
- `krylov_m::Int=30`: Krylov subspace dimension (for `:krylov` method)

# Returns
- `x::TTTensor`: Solution at time `τ`

# References
- Lubich, C., Oseledets, I. (2014). A projector-splitting integrator for dynamical 
  low-rank approximation. BIT Numerical Mathematics.
"""
function tt_ksl(x0::TTTensor, A::TTMatrix, y::TTTensor, tau::Real;
                tol::Real=1e-12, rmax::Int=typemax(Int), passes::Int=2,
                expmethod::Symbol=:krylov, krylov_m::Int=30)
    
    n = tt_dims(x0)
    d = length(n)
    rx = collect(tt_ranks(x0))
    ra = tt_ranks(A)
    ry = tt_ranks(y)
    
    T = promote_type(eltype(x0.cores[1]), eltype(A.cores[1]), eltype(y.cores[1]))
    
    # Convert to cell arrays (vectors of 3D/4D arrays)
    crx = [convert(Array{T}, copy(core)) for core in x0.cores]
    cry = [convert(Array{T}, copy(core)) for core in y.cores]
    cra = [convert(Array{T}, copy(core)) for core in A.cores]
    
    # Initialize phi arrays
    # phia[i]: rx[i]' × rx[i] × ra[i]  (for Galerkin projection of A)
    # phiy[i]: rx[i]' × ry[i]          (for RHS projection)
    phia = Vector{Array{T,3}}(undef, d + 1)
    phiy = Vector{Array{T,2}}(undef, d + 1)
    phia[1] = ones(T, 1, 1, 1)
    phia[d + 1] = ones(T, 1, 1, 1)
    phiy[1] = ones(T, 1, 1)
    phiy[d + 1] = ones(T, 1, 1)
    
    tau_pass = tau / passes
    
    # Select exponential method
    # Note: Krylov method from ExponentialUtilities.jl doesn't support ComplexF64 SymTridiagonal
    # Fall back to Taylor for complex data
    use_krylov = expmethod == :krylov && T <: Real
    
    function expv_local(A_func, u0)
        if use_krylov
            # Use Krylov subspace method from ExponentialUtilities.jl (real only)
            len = length(u0)
            A_op = LinearOperator(eltype(u0), len, A_func)
            return expv(1.0, A_op, u0; m=krylov_m, tol=tol)
        else
            # Use Taylor series expansion (works for both real and complex)
            return expv_taylor(A_func, u0, tol)
        end
    end
    
    # Right-to-left orthogonalization and phi computation
    for i in d:-1:2
        r1, ni, r2 = size(crx[i])
        cr = reshape(crx[i], r1, ni * r2)
        
        # QR on the transpose for right orthogonalization
        cr_t = collect(transpose(cr))  # (ni*r2) × r1
        Q, R = qr(cr_t)
        Q = Matrix(Q)
        rnew = size(Q, 2)
        
        # crx[i] = Q^T reshaped, crx[i-1] gets R^T
        cr2 = reshape(crx[i - 1], rx[i - 1] * n[i - 1], rx[i])
        cr2 = cr2 * collect(transpose(R))
        rx[i] = rnew
        crx[i] = reshape(collect(transpose(Q)), rnew, ni, r2)
        crx[i - 1] = reshape(cr2, rx[i - 1], n[i - 1], rnew)
        
        # Compute phis (right to left)
        phia[i] = compute_next_Phi(phia[i + 1], crx[i], cra[i], crx[i], "rl")
        phiy[i] = compute_next_Phi_y(phiy[i + 1], crx[i], cry[i], "rl")
    end
    
    # Forward sweep
    for i in 1:d
        Phi1 = phia[i]
        Phi2 = phia[i + 1]
        A1 = cra[i]
        y1 = cry[i]
        
        # Compute RHS: phiy[i] * y1 * phiy[i+1]
        rhs = compute_rhs_ksl(phiy[i], y1, phiy[i + 1], rx[i], n[i], rx[i + 1])
        
        # Previous solution
        sol_prev = vec(crx[i])
        
        # K-step: exp(-τA) * x + τ * rhs  (for dx/dt + Ax = y)
        sol = expv_local(x -> -tau_pass * bfun3(Phi1, A1, Phi2, x), sol_prev)
        sol = sol .+ tau_pass .* rhs
        
        if i < d
            # SVD for S-step with rank truncation
            sol_mat = reshape(sol, rx[i] * n[i], rx[i + 1])
            F = svd(sol_mat)
            rnew = min(length(F.S), rmax)  # Apply rank limit
            
            U = F.U[:, 1:rnew]
            S_prev = Diagonal(F.S[1:rnew]) * F.Vt[1:rnew, :]  # rnew × rx[i+1]
            
            # Update core and phi
            crx[i] = reshape(U, rx[i], n[i], rnew)
            phia[i + 1] = compute_next_Phi(phia[i], crx[i], cra[i], crx[i], "lr")
            
            # Compute projected RHS for S
            rhss = reshape(rhs, rx[i] * n[i], rx[i + 1])
            rhss = U' * rhss
            rhss = vec(rhss)
            
            S_prev_vec = vec(S_prev)
            
            # S-step: exp(τA) * S - τ * rhs  (backward correction)
            S = expv_local(x -> tau_pass * bfun2(phia[i + 1], Phi2, x), S_prev_vec)
            S = S .- tau_pass .* rhss
            
            # Stuff back
            S = reshape(S, rnew, rx[i + 1])
            cr2 = reshape(crx[i + 1], rx[i + 1], n[i + 1] * rx[i + 2])
            cr2 = S * cr2
            rx[i + 1] = rnew
            crx[i + 1] = reshape(cr2, rnew, n[i + 1], rx[i + 2])
            
            # Update phiy
            phiy[i + 1] = compute_next_Phi_y(phiy[i], crx[i], cry[i], "lr")
        else
            crx[i] = reshape(sol, rx[i], n[i], rx[i + 1])
        end
    end
    
    # Backward sweep (if passes > 1)
    if passes > 1
        for i in d:-1:1
            Phi1 = phia[i]
            Phi2 = phia[i + 1]
            A1 = cra[i]
            y1 = cry[i]
            
            # Compute RHS
            rhs = compute_rhs_ksl(phiy[i], y1, phiy[i + 1], rx[i], n[i], rx[i + 1])
            
            # Previous solution
            sol_prev = vec(crx[i])
            
            # V-step: exp(-τA) * x + τ * rhs
            sol = expv_local(x -> -tau_pass * bfun3(Phi1, A1, Phi2, x), sol_prev)
            sol = sol .+ tau_pass .* rhs
            
            if i > 1
                # SVD for S-step (different direction) with rank truncation
                sol_mat = reshape(sol, rx[i], n[i] * rx[i + 1])
                F = svd(collect(transpose(sol_mat)))  # (n*rx2) × rx1
                rnew = min(length(F.S), rmax)  # Apply rank limit
                
                V = F.U[:, 1:rnew]  # (n*rx2) × rnew
                S_prev = collect(transpose(Diagonal(F.S[1:rnew]) * F.Vt[1:rnew, :]))  # rx1 × rnew
                
                # Update core and phi: new core is V^T reshaped
                crx[i] = reshape(collect(transpose(V)), rnew, n[i], rx[i + 1])
                phia[i] = compute_next_Phi(phia[i + 1], crx[i], cra[i], crx[i], "rl")
                
                # Compute projected RHS for S
                rhss = reshape(rhs, rx[i], n[i] * rx[i + 1])
                rhss = rhss * conj(V)
                rhss = vec(rhss)
                
                S_prev_vec = vec(S_prev)
                
                # S-step: exp(τA) * S - τ * rhs
                S = expv_local(x -> tau_pass * bfun2(Phi1, phia[i], x), S_prev_vec)
                S = S .- tau_pass .* rhss
                
                # Stuff back
                S = reshape(S, rx[i], rnew)
                cr2 = reshape(crx[i - 1], rx[i - 1] * n[i - 1], rx[i])
                cr2 = cr2 * S
                rx[i] = rnew
                crx[i - 1] = reshape(cr2, rx[i - 1], n[i - 1], rnew)
                
                # Update phiy
                phiy[i] = compute_next_Phi_y(phiy[i + 1], crx[i], cry[i], "rl")
            else
                crx[i] = reshape(sol, rx[i], n[i], rx[i + 1])
            end
        end
    end
    
    return TTTensor(crx)
end

"""
Compute the next Phi matrix for operator projection.
Direction "lr" computes from left to right, "rl" from right to left.

Phi represents the Galerkin projection: <x|A|y> contracted up to site i.
"""
function compute_next_Phi(Phi_prev::Array{T,3}, x::Array, A::Array, y::Array, direction::String) where T
    # x: rx1 × n × rx2
    # A: ra1 × n × m × ra2  
    # y: ry1 × m × ry2
    
    rx1, nx, rx2 = size(x)
    ry1, ny, ry2 = size(y)
    ra1, na, ma, ra2 = size(A)
    
    if direction == "lr"
        # Phi_prev: rx1' × ry1 × ra1
        # Output: rx2' × ry2 × ra2
        
        # Contract x with Phi_prev: x'[n*rx2, rx1] * Phi[rx1', ry1*ra1]
        x_mat = reshape(x, rx1, nx * rx2)
        Phi = reshape(Phi_prev, size(Phi_prev, 1), size(Phi_prev, 2) * ra1)
        Phi = x_mat' * Phi  # (n*rx2) × (ry1*ra1)
        
        # Reshape and contract with A
        Phi = reshape(Phi, nx * rx2 * ry1, ra1)
        Phi = collect(transpose(Phi))  # ra1 × (n*rx2*ry1)
        Phi = reshape(Phi, ra1 * na, rx2 * ry1)
        A_mat = reshape(A, ra1 * na, ma * ra2)
        Phi = collect(transpose(A_mat)) * Phi  # (m*ra2) × (rx2*ry1)
        Phi = reshape(Phi, ma, ra2 * rx2 * ry1)
        Phi = collect(transpose(Phi))  # (ra2*rx2*ry1) × m
        Phi = reshape(Phi, ra2 * rx2, ry1 * ma)
        
        # Contract with y
        y_mat = reshape(y, ry1 * ny, ry2)
        Phi = Phi * y_mat  # (ra2*rx2) × ry2
        Phi = reshape(Phi, ra2, rx2 * ry2)
        Phi = collect(transpose(Phi))
        Phi = reshape(Phi, rx2, ry2, ra2)
        
    else  # "rl"
        # Phi_prev: ry2 × ra2 × rx2'
        # Output: ry1 × ra1 × rx1'
        
        # Contract y with Phi_prev  
        y_mat = reshape(y, ry1 * ny, ry2)
        Phi = reshape(Phi_prev, ry2, ra2 * size(Phi_prev, 3))
        Phi = y_mat * Phi  # (ry1*m) × (ra2*rx2')
        
        # Contract with A
        Phi = reshape(Phi, ry1, ma * ra2 * size(Phi_prev, 3))
        Phi = collect(transpose(Phi))  # (m*ra2*rx2') × ry1
        Phi = reshape(Phi, ma * ra2, size(Phi_prev, 3) * ry1)
        A_mat = reshape(A, ra1 * na, ma * ra2)
        Phi = A_mat * Phi  # (ra1*n) × (rx2'*ry1)
        Phi = reshape(Phi, ra1 * na * size(Phi_prev, 3), ry1)
        Phi = collect(transpose(Phi))  # ry1 × (ra1*n*rx2')
        
        # Contract with x
        Phi = reshape(Phi, ry1 * ra1, na * size(Phi_prev, 3))
        x_mat = reshape(x, rx1, nx * rx2)
        Phi = Phi * x_mat'  # (ry1*ra1) × rx1
        Phi = reshape(Phi, ry1, ra1, rx1)
    end
    
    return Phi
end

"""
Compute the next Phi matrix for RHS projection (A = empty case).

MATLAB convention for phiy storage:
- phiy[i]: rx[i] × ry[i]  (stored consistently as rx × ry)

Direction "lr": computes phiy[i+1] from phiy[i]
  Input Phi_prev: rx[i] × ry[i]
  Output: rx[i+1] × ry[i+1]
  
Direction "rl": computes phiy[i] from phiy[i+1]  
  Input Phi_prev: rx[i+1] × ry[i+1]
  Output: rx[i] × ry[i]
"""
function compute_next_Phi_y(Phi_prev::Array{T,2}, x::Array, y::Array, direction::String) where T
    rx1, nx, rx2 = size(x)
    ry1, ny, ry2 = size(y)
    
    if direction == "lr"
        # Phi_prev: rx1 × ry1
        # Output: rx2 × ry2
        
        # x'[n*rx2, rx1] * Phi_prev[rx1, ry1] = [n*rx2, ry1]
        x_mat = reshape(x, rx1, nx * rx2)
        Phi = collect(transpose(x_mat)) * Phi_prev  # (n*rx2) × ry1
        
        # Reshape to [n, rx2*ry1] then transpose to [rx2*ry1, n]
        Phi = reshape(Phi, nx, rx2 * ry1)
        Phi = collect(transpose(Phi))  # (rx2*ry1) × n
        Phi = reshape(Phi, rx2, ry1 * nx)  # rx2 × (ry1*n)
        
        # Contract with y: y[ry1*n, ry2]
        y_mat = reshape(y, ry1 * ny, ry2)
        Phi = Phi * y_mat  # rx2 × ry2
        
    else  # "rl"
        # Phi_prev: rx2 × ry2
        # Output: rx1 × ry1
        
        # y[ry1*n, ry2] * Phi_prev'[ry2, rx2] = [ry1*n, rx2]
        y_mat = reshape(y, ry1 * ny, ry2)
        Phi = y_mat * collect(transpose(Phi_prev))  # (ry1*n) × rx2
        
        # Reshape to [ry1, n*rx2]
        Phi = reshape(Phi, ry1, ny * rx2)
        
        # Contract with x': x[rx1, n*rx2]' = [n*rx2, rx1]
        # Phi[ry1, n*rx2] * x'[n*rx2, rx1] would give [ry1, rx1]
        # But we want [rx1, ry1], so we compute x * Phi'
        x_mat = reshape(x, rx1, nx * rx2)
        Phi = x_mat * collect(transpose(Phi))  # rx1 × ry1
    end
    
    return Phi
end

"""
Compute RHS projection for KSL: phiy[i] * y * phiy[i+1]

phiy[i]: rx[i] × ry[i]
phiy[i+1]: rx[i+1] × ry[i+1]
y: ry[i] × n × ry[i+1]
Output: rx[i] × n × rx[i+1] (flattened)
"""
function compute_rhs_ksl(phiy1::Array{T,2}, y::Array, phiy2::Array{T,2}, rx1, n, rx2) where T
    ry1, ny, ry2 = size(y)
    
    # phiy1: rx1 × ry1
    # phiy2: rx2 × ry2
    
    # phiy1[rx1, ry1] * y[ry1, n*ry2] = [rx1, n*ry2]
    y_mat = reshape(y, ry1, ny * ry2)
    rhs = phiy1 * y_mat  # rx1 × (n*ry2)
    
    # Reshape to [rx1*n, ry2]
    rhs = reshape(rhs, size(phiy1, 1) * ny, ry2)
    
    # rhs[rx1*n, ry2] * phiy2'[ry2, rx2] = [rx1*n, rx2]
    rhs = rhs * collect(transpose(phiy2))  # (rx1*n) × rx2
    
    return vec(rhs)
end

"""
3D block matvec for K/V step: y = (Phi1 ⊗ A ⊗ Phi2) * x

Phi1: ry1 × rx1 × ra1
A: ra1 × n × m × ra2  
Phi2: rx2 × ra2 × ry2
x: rx1 × m × rx2 (flattened)
"""
function bfun3(Phi1::Array{T,3}, A::Array, Phi2::Array{T,3}, x::AbstractVector) where T
    ry1 = size(Phi1, 1)
    rx1 = size(Phi1, 2)
    ra1 = size(Phi1, 3)
    
    rx2 = size(Phi2, 1)
    ra2 = size(Phi2, 2)
    ry2 = size(Phi2, 3)
    
    na = size(A, 2)
    ma = size(A, 3)
    
    # x is rx1 × m × rx2 flattened
    xc = collect(x)  # Convert to regular Vector
    y = reshape(xc, rx1 * ma, rx2)
    
    # Contract with Phi2
    Phi2_mat = reshape(Phi2, rx2, ra2 * ry2)
    y = y * Phi2_mat  # (rx1*m) × (ra2*ry2)
    y = reshape(y, rx1, ma * ra2 * ry2)
    y = collect(transpose(y))
    y = reshape(y, ma * ra2, ry2 * rx1)
    
    # Contract with A
    A_mat = reshape(A, ra1 * na, ma * ra2)
    y = A_mat * y  # (ra1*n) × (ry2*rx1)
    y = reshape(y, ra1 * na * ry2, rx1)
    y = collect(transpose(y))
    y = reshape(y, rx1 * ra1, na * ry2)
    
    # Contract with Phi1
    Phi1_mat = reshape(Phi1, ry1, rx1 * ra1)
    y = Phi1_mat * y  # ry1 × (n*ry2)
    
    return vec(y)
end

"""
2D block matvec for S step: y = (Phi1 ⊗ Phi2) * x

Phi1: ry1 × rx1 × ra
Phi2: rx2 × ra × ry2
x: rx1 × rx2 (flattened)
"""
function bfun2(Phi1::Array{T,3}, Phi2::Array{T,3}, x::AbstractVector) where T
    ry1 = size(Phi1, 1)
    rx1 = size(Phi1, 2)
    ra = size(Phi1, 3)
    
    rx2 = size(Phi2, 1)
    ry2 = size(Phi2, 3)
    
    # x is rx1 × rx2 flattened
    xc = collect(x)  # Convert to regular Vector
    y = reshape(xc, rx1, rx2)
    
    # Contract with Phi2
    Phi2_mat = reshape(Phi2, rx2, ra * ry2)
    y = y * Phi2_mat  # rx1 × (ra*ry2)
    y = reshape(y, rx1 * ra, ry2)
    
    # Contract with Phi1
    Phi1_mat = reshape(Phi1, ry1, rx1 * ra)
    y = Phi1_mat * y  # ry1 × ry2
    
    return vec(y)
end

"""
    LinearOperator(n, matvec)

Simple wrapper for matrix-free linear operator for use with ExponentialUtilities.jl.
"""
struct LinearOperator{T,F}
    size::Tuple{Int,Int}
    matvec::F
end

LinearOperator(n::Int, matvec::F) where F = LinearOperator{ComplexF64,F}((n, n), matvec)
LinearOperator(::Type{T}, n::Int, matvec::F) where {T,F} = LinearOperator{T,F}((n, n), matvec)

Base.size(A::LinearOperator) = A.size
Base.size(A::LinearOperator, i::Int) = A.size[i]
Base.eltype(::LinearOperator{T}) where T = T
LinearAlgebra.mul!(y::AbstractVector, A::LinearOperator, x::AbstractVector) = (y .= A.matvec(x); y)
LinearAlgebra.ishermitian(::LinearOperator) = false
LinearAlgebra.issymmetric(::LinearOperator) = false
LinearAlgebra.opnorm(A::LinearOperator, p::Real=2) = 1.0  # Conservative estimate

# Make it work with ExponentialUtilities
Base.:(*)(A::LinearOperator, x::AbstractVector) = A.matvec(x)

"""
    expv_taylor(A, u0, tol; m=20, rescale=true)

Compute exp(A)*u0 using Taylor series expansion with automatic rescaling.

# Arguments
- `A`: Function that computes A*x (or a matrix)
- `u0`: Initial vector
- `tol`: Tolerance for convergence

# Keyword Arguments
- `m::Int=20`: Maximum number of Taylor terms
- `rescale::Bool=true`: Whether to rescale by estimated norm of A
"""
function expv_taylor(A, u0::Vector{T}, tol::Real; m::Int=20, rescale::Bool=true) where T
    n = length(u0)
    
    # Estimate norm of A if rescaling
    if rescale && n > 0
        v = randn(T, n)
        nv = norm(v)
        if nv > 0
            v = v / nv
        end
        nrmA = 1.0
        for _ in 1:5
            v_new = A(v)
            nrmA = norm(v_new)
            if nrmA < eps(real(T)) * 100
                nrmA = one(real(T))
                break
            end
            v = v_new / nrmA
        end
        nrmA = max(ceil(Int, real(nrmA)), 1)
    else
        nrmA = 1
    end
    
    u = copy(u0)
    for _ in 1:nrmA
        nrm0 = norm(u)
        if nrm0 < eps(real(T)) * 100
            break
        end
        du = copy(u)
        u_new = copy(u)
        for i in 2:m
            du = A(du) / ((i - 1) * nrmA)
            u_new = u_new + du
            if norm(du) / nrm0 < tol
                break
            end
        end
        u = u_new
    end
    
    return u
end

# Backward compatible alias
const expv_full = expv_taylor

# Convenience method for homogeneous case (y = 0)
function tt_ksl(x0::TTTensor, A::TTMatrix, tau::Real; kwargs...)
    n = tt_dims(x0)
    d = length(n)
    T = eltype(x0.cores[1])
    cores = [zeros(T, 1, ni, 1) for ni in n]
    y = TTTensor(cores)
    return tt_ksl(x0, A, y, tau; kwargs...)
end

# Deprecated alias
const tt_ksl_ml = tt_ksl
