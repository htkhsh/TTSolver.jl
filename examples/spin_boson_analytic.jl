using LinearAlgebra
using TTSolver
using Plots
using DelimitedFiles

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

function tt_kron_matrix(A::AbstractMatrix, B::AbstractMatrix)
    return TTMatrix([
        reshape(A, 1, size(A, 1), size(A, 2), 1),
        reshape(B, 1, size(B, 1), size(B, 2), 1),
    ])
end

nb = 6
nt = 3
tol = 1e-4
Tfin = 40.0
Nsteps = 150
dt = Tfin / Nsteps

sigma_z = ComplexF64[1 0; 0 -1]
sigma_x = ComplexF64[0 1; 1 0]
id2 = Matrix{ComplexF64}(I, 2, 2)

_, _, x_op, n_op = boson_operators(nb)
idb = Matrix{ComplexF64}(I, nb, nb)

Δ0 = 1.0
Δx = 1.5
ω = 1.2
g = 0.3

H_tt =
    0.5 * Δ0 * tt_kron_matrix(sigma_z, idb) +
    0.5 * Δx * tt_kron_matrix(sigma_x, idb) +
    ω * tt_kron_matrix(id2, Matrix{ComplexF64}(n_op)) +
    g * tt_kron_matrix(sigma_z, Matrix{ComplexF64}(x_op))

opts = Dict(
    :verb => 1,
    :nswp => 2,
    :local_iters => 5,
    :time_scheme => "cn",
    :time_error_damp => 10.0,
    :kickrank => 4,
    :correct_2_norm => true,
)

x0 = tt_unit([2, nb], [1, 1]; T=ComplexF64)
U = tkron(x0, tt_ones(nt; T=ComplexF64))

A = (-1im) * H_tt

ts = collect(range(0.0, Tfin; length=Nsteps + 1))

H_full =
    0.5 * Δ0 * kron(sigma_z, idb) +
    0.5 * Δx * kron(sigma_x, idb) +
    ω * kron(id2, Matrix{ComplexF64}(n_op)) +
    g * kron(sigma_z, Matrix{ComplexF64}(x_op))

psi0 = kron(ComplexF64[1.0, 0.0], ComplexF64[1.0; zeros(ComplexF64, nb - 1)])

psi_exact = Vector{Vector{ComplexF64}}(undef, length(ts))
psi_tt = Vector{Vector{ComplexF64}}(undef, length(ts))
norm_exact = zeros(Float64, length(ts))
norm_tt = zeros(Float64, length(ts))
psi_exact[1] = psi0
psi_tt[1] = vec(reshape(expand_dR(x0), :, 1))
norm_exact[1] = sum(abs2, psi_exact[1])
norm_tt[1] = sum(abs2, psi_tt[1])

scheme = String(opts[:time_scheme])
let u = x0
    for i in 2:length(ts)
        A_step = dt * A
        U = tkron(u, tt_ones(nt; T=ComplexF64))
        U, tgrid, _, _ = tamen(U, A_step, tol, opts)
        u = extract_snapshot(U, tgrid, 1.0, scheme)
        norm_tt[i] = real(dot(u, u))
        psi_tt[i] = vec(reshape(expand_dR(u), :, 1))
        psi_exact[i] = exp(-1im * H_full * ts[i]) * psi0
        norm_exact[i] = sum(abs2, psi_exact[i])
    end
end

# Convert TT (column-major) ordering to kron (row-major) ordering for comparison
function tt_to_kron_order(psi_tt, n1, n2)
    psi_kron = zeros(eltype(psi_tt), n1 * n2)
    for i in 1:n1
        for j in 1:n2
            tt_idx = (j - 1) * n1 + i
            kron_idx = (i - 1) * n2 + j
            psi_kron[kron_idx] = psi_tt[tt_idx]
        end
    end
    return psi_kron
end

psi_tt_kron = [tt_to_kron_order(v, 2, nb) for v in psi_tt]

p_up_exact = [sum(abs2, reshape(v, 2, nb)[1, :]) for v in psi_exact]
p_dn_exact = [sum(abs2, reshape(v, 2, nb)[2, :]) for v in psi_exact]
p_up_tt = [sum(abs2, reshape(v, 2, nb)[1, :]) for v in psi_tt_kron]
p_dn_tt = [sum(abs2, reshape(v, 2, nb)[2, :]) for v in psi_tt_kron]

plt = plot(ts, p_up_exact; label="exact P(↑)", linewidth=2)
plot!(plt, ts, p_up_tt; label="tAMEn P(↑)", linestyle=:dash)
plot!(plt, ts, p_dn_exact; label="exact P(↓)", linewidth=2)
plot!(plt, ts, p_dn_tt; label="tAMEn P(↓)", linestyle=:dash)
plot!(plt; xlabel="t", ylabel="population", title="Spin-Boson (analytic)")

savefig(plt, "spin_boson_analytic.png")
display(plt)

plt_norm = plot(ts, norm_exact; label="exact norm", linewidth=2)
plot!(plt_norm, ts, norm_tt; label="tAMEn norm", linestyle=:dash)
plot!(plt_norm; xlabel="t", ylabel="norm", title="Norm check")

savefig(plt_norm, "spin_boson_analytic_norm.png")
display(plt_norm)

data = hcat(ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt)
writedlm("spin_boson_analytic_data.csv", data, ',')
