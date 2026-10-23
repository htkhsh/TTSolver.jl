# Driven spin-boson model: KSL and tAMEn compared with a dense RK4 solution.
# Run from the repository root with `julia --project examples/time_dependent/sb_driven.jl`.

using DelimitedFiles
using LinearAlgebra
using CairoMakie
using TTSolver

const NB = 6
const TFIN = 4.0
const DT = 0.05
const NT = 9
const DENSE_SUBSTEPS = 8

const SIGMA_Z = ComplexF64[1 0; 0 -1]
const SIGMA_X = ComplexF64[0 1; 1 0]
const ID2 = Matrix{ComplexF64}(I, 2, 2)
const IDB = Matrix{ComplexF64}(I, NB, NB)
const X_BOSON = zeros(ComplexF64, NB, NB)
const N_BOSON = Matrix(Diagonal(0:(NB - 1)))
for n in 2:NB
    X_BOSON[n - 1, n] = sqrt(n - 1)
    X_BOSON[n, n - 1] = sqrt(n - 1)
end

# H(t) = H0 + cos(Ωt) Hdrive.
const DELTA_Z = 1.0
const DELTA_X = 0.6
const DRIVE_AMPLITUDE = 0.8
const DRIVE_FREQUENCY = 1.3
const OMEGA = 1.0
const COUPLING = 0.12

const H0_DENSE =
    0.5 * DELTA_Z * kron(SIGMA_Z, IDB) +
    0.5 * DELTA_X * kron(SIGMA_X, IDB) +
    OMEGA * kron(ID2, N_BOSON) +
    COUPLING * kron(SIGMA_Z, X_BOSON)
const HDRIVE_DENSE = 0.5 * DRIVE_AMPLITUDE * kron(SIGMA_X, IDB)

const H0_TT =
    0.5 * DELTA_Z * tt_mkron(SIGMA_Z, IDB) +
    0.5 * DELTA_X * tt_mkron(SIGMA_X, IDB) +
    OMEGA * tt_mkron(ID2, N_BOSON) +
    COUPLING * tt_mkron(SIGMA_Z, X_BOSON)
const HDRIVE_TT = 0.5 * DRIVE_AMPLITUDE * tt_mkron(SIGMA_X, IDB)

drive(t) = cos(DRIVE_FREQUENCY * t)
dense_hamiltonian(t) = H0_DENSE + drive(t) * HDRIVE_DENSE

function rk4_step(psi, time, dt)
    k1 = -1im * dense_hamiltonian(time) * psi
    k2 = -1im * dense_hamiltonian(time + dt / 2) * (psi + (dt / 2) * k1)
    k3 = -1im * dense_hamiltonian(time + dt / 2) * (psi + (dt / 2) * k2)
    k4 = -1im * dense_hamiltonian(time + dt) * (psi + dt * k3)
    psi + (dt / 6) * (k1 + 2k2 + 2k3 + k4)
end

function dense_trajectory(psi0, times)
    states = Vector{Vector{ComplexF64}}(undef, length(times))
    states[1] = copy(psi0)
    psi = copy(psi0)
    for index in 2:length(times)
        left = times[index - 1]
        step = (times[index] - left) / DENSE_SUBSTEPS
        for substep in 1:DENSE_SUBSTEPS
            psi = rk4_step(psi, left + (substep - 1) * step, step)
        end
        states[index] = copy(psi)
    end
    states
end

function populations(states)
    p_up = [sum(abs2, state[1:NB]) for state in states]
    p_down = [sum(abs2, state[(NB + 1):(2NB)]) for state in states]
    p_up, p_down
end

function tamen_time_operator(_, normalized_times)
    physical_times = TFIN .* normalized_times
    cells = Matrix{Any}(undef, 3, 2)
    operators = (H0_TT, HDRIVE_TT)
    coefficients = (ones(length(physical_times)), drive.(physical_times))
    for column in 1:2
        operator = operators[column]
        cells[1, column] = (-1im * TFIN) .* ComplexF64.(operator.cores[1])
        cells[2, column] = ComplexF64.(operator.cores[2])
        time_core = Matrix(Diagonal(ComplexF64.(coefficients[column])))
        cells[3, column] = reshape(time_core, 1, length(physical_times), length(physical_times), 1)
    end
    cells
end

function main()
    isapprox(TFIN / DT, round(TFIN / DT); atol=eps(Float64), rtol=0) ||
        error("TFIN must be an integer multiple of DT")
    times = collect(0:DT:TFIN)
    # KSL preserves the input ranks, so reserve the full two-site rank up front.
    spin_core = zeros(ComplexF64, 1, 2, 2)
    spin_core[1, 1, 1] = 1
    spin_core[1, 2, 2] = 1
    boson_core = zeros(ComplexF64, 2, NB, 1)
    boson_core[1, 1, 1] = 1
    initial_tt = TTTensor([spin_core, boson_core])
    initial_dense = vec(tt_full(initial_tt))

    println("Driven spin-boson: H(t) = H0 + cos(Ωt) Hdrive")
    println("Running dense RK4 reference...")
    dense_states = dense_trajectory(initial_dense, times)

    println("Running KSL (midpoint Hamiltonian per step)...")
    ksl_states = Vector{Vector{ComplexF64}}(undef, length(times))
    ksl_states[1] = copy(initial_dense)
    state = initial_tt
    for index in 2:length(times)
        left = times[index - 1]
        dt = times[index] - left
        hamiltonian = H0_TT + drive(left + dt / 2) * HDRIVE_TT
        state = tt_ksl(state, 1im * hamiltonian, dt; rmax=2, tol=1e-12, krylov_m=30)
        ksl_states[index] = vec(tt_full(state))
    end

    println("Running tAMEn (space-time Chebyshev solve)...")
    initial_space_time = tt_kron(initial_tt, tt_ones(NT; T=ComplexF64))
    space_time, time_grid, _, _, _ = tamen(
        initial_space_time,
        tamen_time_operator,
        1e-7;
        verb=0,
        nswp=4,
        local_iters=8,
        kickrank=4,
        time_error_damp=10.0,
        time_scheme="cheb",
        correct_2_norm=true,
    )
    tamen_tt_states = extract_snapshot(space_time, time_grid, times ./ TFIN, "cheb")
    tamen_states = [vec(tt_full(snapshot)) for snapshot in tamen_tt_states]

    up_dense, down_dense = populations(dense_states)
    up_ksl, down_ksl = populations(ksl_states)
    up_tamen, down_tamen = populations(tamen_states)
    norm_dense = [sum(abs2, state) for state in dense_states]
    norm_ksl = [sum(abs2, state) for state in ksl_states]
    norm_tamen = [sum(abs2, state) for state in tamen_states]
    error_ksl = [norm(ksl_states[i] - dense_states[i]) for i in eachindex(times)]
    error_tamen = [norm(tamen_states[i] - dense_states[i]) for i in eachindex(times)]

    println("\nMaximum errors against dense RK4:")
    println("  KSL  state error: $(maximum(error_ksl))")
    println("  tAMEn state error: $(maximum(error_tamen))")
    println("  KSL  norm error:  $(maximum(abs.(norm_ksl .- norm_dense)))")
    println("  tAMEn norm error: $(maximum(abs.(norm_tamen .- norm_dense)))")
    println("  tAMEn internal intervals: $(length(time_grid))")

    fig = Figure(size=(900, 900))
    population_axis = Axis(fig[1, 1]; xlabel="Time", ylabel="Population",
                           title="Driven spin-boson populations")
    lines!(population_axis, times, up_dense; label="Dense P(↑)", color=:navy)
    lines!(population_axis, times, down_dense; label="Dense P(↓)", color=:firebrick)
    lines!(population_axis, times, up_ksl; label="KSL P(↑)", color=:dodgerblue, linestyle=:dash)
    lines!(population_axis, times, down_ksl; label="KSL P(↓)", color=:darkorange, linestyle=:dash)
    lines!(population_axis, times, up_tamen; label="tAMEn P(↑)", color=:royalblue, linestyle=:dot)
    lines!(population_axis, times, down_tamen; label="tAMEn P(↓)", color=:orangered, linestyle=:dot)
    axislegend(population_axis; position=:rt, nbanks=2)

    floor = eps(Float64)
    norm_axis = Axis(fig[2, 1]; xlabel="Time", ylabel="|‖ψ‖² - 1|",
                     title="Norm conservation error", yscale=log10)
    lines!(norm_axis, times, max.(abs.(norm_dense .- 1), floor); label="Dense", color=:black)
    lines!(norm_axis, times, max.(abs.(norm_ksl .- 1), floor); label="KSL", color=:dodgerblue)
    lines!(norm_axis, times, max.(abs.(norm_tamen .- 1), floor); label="tAMEn", color=:orangered)
    axislegend(norm_axis; position=:lt)

    state_error_axis = Axis(fig[3, 1]; xlabel="Time", ylabel="‖ψ - ψ_dense‖₂",
                            title="State error against dense RK4", yscale=log10)
    lines!(state_error_axis, times, max.(error_ksl, floor); label="KSL", color=:dodgerblue)
    lines!(state_error_axis, times, max.(error_tamen, floor); label="tAMEn", color=:orangered)
    axislegend(state_error_axis; position=:lt)

    data = hcat(
        times,
        up_dense, down_dense, up_ksl, down_ksl, up_tamen, down_tamen,
        norm_dense, norm_ksl, norm_tamen, error_ksl, error_tamen,
    )
    output = joinpath(@__DIR__, "sb_driven_data.csv")
    open(output, "w") do io
        println(io, "time,p_up_dense,p_down_dense,p_up_ksl,p_down_ksl,p_up_tamen,p_down_tamen,norm_dense,norm_ksl,norm_tamen,state_error_ksl,state_error_tamen")
        writedlm(io, data, ',')
    end
    println("Saved comparison data: $output")
    figure_path = joinpath(@__DIR__, "sb_driven_comparison.png")
    save(figure_path, fig)
    println("Saved comparison figure: $figure_path")
end

main()
