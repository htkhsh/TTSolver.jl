using LinearAlgebra
using TTSolver
using Plots

J = 1.0
H = [0.0 J; J 0.0]

x0 = tt_unit([2], 1; T=ComplexF64)
core = x0.cores[1]
full0 = reshape(expand_dR(x0), :, 1)

nt = 5
X = Matrix{Any}(undef, 2, 1)
X[1, 1] = reshape(copy(core), size(core, 1), size(core, 2), 1, size(core, 3))
timecore = Array{ComplexF64,4}(undef, 1, nt, 1, 1)
timecore .= one(ComplexF64)
X[2, 1] = timecore
@assert ndims(X[2, 1]) == 4

A = TTMatrix([reshape(-1im .* H, 1, 2, 2, 1)])
opts = Dict(:verb => 1, :nswp => 2, :local_iters => 3, :time_scheme => "cheb", :time_error_damp => 1)

U, t, _, _ = tamen(X, A, 1e-4, opts)

ts = collect(range(0.0, 1.0; length=41))
psi_exact = [exp(-1im .* H * ti) * vec(full0) for ti in ts]
psi_tt = [vec(reshape(expand_dR(extract_snapshot(U, t, ti, "cheb")), :, 1)) for ti in ts]

p1_exact = [abs2(v[1]) for v in psi_exact]
p2_exact = [abs2(v[2]) for v in psi_exact]
p1_tt = [abs2(v[1]) for v in psi_tt]
p2_tt = [abs2(v[2]) for v in psi_tt]

plt = plot(ts, p1_exact; label="exact |ψ1|^2", linestyle=:solid, linewidth=3)
plot!(plt, ts, p1_tt; label="tAMEn |ψ1|^2", linestyle=:dash, linewidth=3)
plot!(plt, ts, p2_exact; label="exact |ψ2|^2", linestyle=:solid, linewidth=3)
plot!(plt, ts, p2_tt; label="tAMEn |ψ2|^2", linestyle=:dash, linewidth=3)
plot!(plt; xlabel="t", ylabel="probability", title="Coupled Oscillator (2-site) Schrödinger")
savefig(plt, "schrodinger_chain.png")
