# Spin-Boson model simulation using tt_ksl
#
# This example demonstrates the KSL (Projector-Splitting) integrator for
# time evolution of quantum systems in TT format.
#
# Note: KSL is a rank-preserving scheme. For problems requiring rank adaptation,
# consider using tAMEn or other rank-adaptive methods.

using LinearAlgebra
using TTSolver

include("../../common/plot_utils.jl")

function boson_operators(nb::Integer)
    a = zeros(Float64, nb, nb)
    for n in 2:nb
        a[n - 1, n] = sqrt(n - 1)
    end
    adag = transpose(a)
    x = a + adag
    n_op = Diagonal(0:(nb - 1))
    return a, adag, x, n_op
end

# Parameters
nb = 6          # Number of boson states (small for this 2-site problem)
Tfin = 10.0     # Final time (reduced for demonstration)
dt = 0.05       # Time step for KSL
Nsteps = Int(Tfin / dt)
rmax = 2        # Maximum TT rank (limited to 2 for 2-site system)

sigma_z = ComplexF64[1 0; 0 -1]
sigma_x = ComplexF64[0 1; 1 0]
id2 = Matrix{ComplexF64}(I, 2, 2)

_, _, x_op, n_op = boson_operators(nb)
idb = Matrix{ComplexF64}(I, nb, nb)

# Weak coupling parameters for better KSL performance
Δ0 = 1.0
Δx = 0.5    # Reduced tunneling
ω = 1.0
g = 0.1     # Weak coupling

# Hamiltonian in TT format
H_tt =
    0.5 * Δ0 * tt_mkron(sigma_z, idb) +
    0.5 * Δx * tt_mkron(sigma_x, idb) +
    ω * tt_mkron(id2, Matrix{ComplexF64}(n_op)) +
    g * tt_mkron(sigma_z, Matrix{ComplexF64}(x_op))

# Full Hamiltonian for exact solution
H_full =
    0.5 * Δ0 * kron(sigma_z, idb) +
    0.5 * Δx * kron(sigma_x, idb) +
    ω * kron(id2, Matrix{ComplexF64}(n_op)) +
    g * kron(sigma_z, Matrix{ComplexF64}(x_op))

# Operator A = iH for Schrödinger equation: dψ/dt = -iHψ => dψ/dt + (iH)ψ = 0
A_tt = (1im) * H_tt

# Initial state |↑, 0⟩ as rank-2 TT (maximum for 2-site system)
# With kron-compatible TT: TTTensor([spin_core, boson_core]).full() == kron(spin, boson)
psi0 = kron(ComplexF64[1.0, 0.0], ComplexF64[1.0; zeros(ComplexF64, nb - 1)])
# reshape to (2, nb) so rows=spin, cols=boson
psi0_mat = reshape(psi0, nb, 2)'  # shape (2, nb)

# Pad to rank-2 (KSL is rank-preserving)
F = svd(psi0_mat)  # U[:,k] = spin, V[:,k] = boson
r = min(rmax, min(2, nb))  # Ensure rank 2 for KSL capacity
U_padded = zeros(ComplexF64, 2, r)
S_padded = zeros(ComplexF64, r)
Vt_padded = zeros(ComplexF64, r, nb)

nSV = min(length(F.S), r)
U_padded[:, 1:nSV] = F.U[:, 1:nSV]
S_padded[1:nSV] = F.S[1:nSV]
Vt_padded[1:nSV, :] = F.Vt[1:nSV, :]

# TT([spin_core, boson_core]).full() == kron(spin, boson)
spin_core = reshape(U_padded * Diagonal(S_padded), 1, 2, r)
boson_core = reshape(Vt_padded, r, nb, 1)
x0 = TTTensor([spin_core, boson_core])

nrm = tt_norm(x0)
x0 = (1.0 / nrm) * x0

# Use the TT initial state for exact comparison (ensures consistency)
psi0 = vec(tt_full(x0))

println("Initial TT ranks: ", tt_ranks(x0))
println("Initial norm: ", tt_norm(x0))

# Time array
ts = collect(range(0.0, Tfin; length=Nsteps + 1))

# Storage for results
psi_exact = Vector{Vector{ComplexF64}}(undef, length(ts))
psi_tt = Vector{Vector{ComplexF64}}(undef, length(ts))
norm_exact = zeros(Float64, length(ts))
norm_tt = zeros(Float64, length(ts))

# Initial conditions
psi_exact[1] = psi0
psi_tt[1] = vec(tt_full(x0))
norm_exact[1] = sum(abs2, psi_exact[1])
norm_tt[1] = sum(abs2, psi_tt[1])

# Time evolution using tt_ksl with Krylov method (default)
println("\nRunning KSL time evolution with Krylov method...")
println("Parameters: nb=$nb, rmax=$rmax, dt=$dt, Tfin=$Tfin")
println("Note: KSL is rank-preserving; max rank for 2-site system is 2")

let u = x0
    for i in 2:length(ts)
        # Use tt_ksl with rmax and default Krylov method
        u = tt_ksl(u, A_tt, dt; rmax=rmax, tol=1e-12, krylov_m=30)

        # Compute norm and wavefunction
        norm_tt[i] = real(tt_dot(u, u))
        psi_tt[i] = vec(tt_full(u))

        # Exact solution
        psi_exact[i] = exp(-1im * H_full * ts[i]) * psi0
        norm_exact[i] = sum(abs2, psi_exact[i])

        if i % 40 == 0
            err = norm(psi_tt[i] - psi_exact[i])
            println("  Step $i/$(length(ts)-1), t=$(round(ts[i], digits=2)), ranks=$(tt_ranks(u)), error=$err")
        end
    end
end

# Compute populations
p_up_exact, p_dn_exact = compute_spin_populations(psi_exact, nb)
p_up_tt, p_dn_tt = compute_spin_populations(psi_tt, nb)

# Compute errors
max_error_up = maximum(abs.(p_up_exact - p_up_tt))
max_error_dn = maximum(abs.(p_dn_exact - p_dn_tt))
println("\nResults:")
println("  Max error P(↑): $max_error_up")
println("  Max error P(↓): $max_error_dn")
println("  Max norm deviation: $(maximum(abs.(norm_tt .- 1.0)))")

# Plot populations
fig_pop = plot_spin_boson_populations(
    ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt;
    method="KSL",
    title_suffix="KSL, Krylov, g=$g",
    filename="sb_ksl.png"
)
display(fig_pop)

# Plot norm
fig_norm = plot_norm_conservation(
    ts, norm_exact, norm_tt;
    method="KSL",
    filename="sb_ksl_norm.png"
)
display(fig_norm)

# Save data
save_spin_boson_data("sb_ksl_data.csv", ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt)
