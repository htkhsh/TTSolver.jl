@testset "pre-migration TAMEn reference" begin
    Random.seed!(20261010)
    @test maximum(relative_l2.(run_tamen("cheb"), TAMEN_CHEB_REFERENCE)) <=
          10eps(Float64)

    Random.seed!(20261010)
    @test maximum(relative_l2.(run_tamen("CN"), TAMEN_CN_REFERENCE)) <=
          10eps(Float64)
end

function run_tamen_result(problem; workspace=nothing, conserved=false)
    kwargs = conserved ?
        (;
            conserved_observables=problem.observables,
            correct_2_norm=false,
        ) :
        (; correct_2_norm=false)
    workspace === nothing || (kwargs = merge(kwargs, (; workspace)))
    return tamen(
        deepcopy(problem.space_time),
        problem.generator,
        problem.tol,
        copy(problem.options);
        kwargs...,
    )
end

function expanded_tamen_snapshots(result, scheme)
    solution, time_grid = result[1], result[2]
    snapshots = extract_snapshot(solution, time_grid, REFERENCE_TIMES, scheme)
    return [vec(tt_full(snapshot)) for snapshot in snapshots]
end

@testset "public TAMEn workspace API" begin
    @test :TAMEnWorkspace in names(TTSolver)
    work = TAMEnWorkspace(Float64)
    problem = tamen_reference_problem("CN"; conserved=true)

    Random.seed!(20261010)
    plain = run_tamen_result(problem)
    Random.seed!(20261010)
    cached = run_tamen_result(problem; workspace=work)
    @test maximum(relative_l2.(
        expanded_tamen_snapshots(cached, "CN"),
        expanded_tamen_snapshots(plain, "CN"),
    )) <= TAMEN_TOL

    Random.seed!(20261010)
    conserved = run_tamen_result(problem; workspace=work, conserved=true)
    conserved_snapshots = expanded_tamen_snapshots(conserved, "CN")
    initial_value = tt_dot(problem.observable, problem.initial)
    @test all(abs(sum(snapshot) - initial_value) <= 2e-5 for snapshot in conserved_snapshots)

    @test_throws TypeError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        copy(problem.options); workspace=1,
    )
    @test_throws ArgumentError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        copy(problem.options); conserved_observables=problem.observable,
        workspace=work,
    )
    @test_throws ArgumentError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        copy(problem.options); conserved_observables=[ones(2)], workspace=work,
    )
    @test_throws DimensionMismatch tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        copy(problem.options); conserved_observables=[tt_ones(3)], workspace=work,
    )
    @test_throws ArgumentError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        merge(copy(problem.options), Dict(:obs => [problem.observable]));
        conserved_observables=[problem.observable], workspace=work,
    )
    @test_throws ArgumentError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        merge(copy(problem.options), Dict(:correct_2_norm => false));
        correct_2_norm=false, workspace=work,
    )
    @test_throws ArgumentError tamen(
        deepcopy(problem.space_time), problem.generator, problem.tol,
        copy(problem.options); correct_2_norm=1, workspace=work,
    )
end

@testset "TAMEn workspace accuracy and ownership" begin
    work = TAMEnWorkspace(Float64)
    for scheme in ("cheb", "CN")
        reference = scheme == "cheb" ? TAMEN_CHEB_REFERENCE : TAMEN_CN_REFERENCE

        Random.seed!(20261010)
        first_result = run_tamen_result(
            tamen_reference_problem(scheme); workspace=work,
        )
        first_snapshots = expanded_tamen_snapshots(first_result, scheme)
        @test maximum(relative_l2.(first_snapshots, reference)) <= TAMEN_TOL

        Random.seed!(20261010)
        second_result = run_tamen_result(
            tamen_reference_problem(scheme); workspace=work,
        )
        second_snapshots = expanded_tamen_snapshots(second_result, scheme)
        @test maximum(relative_l2.(second_snapshots, first_snapshots)) <= TAMEN_TOL

        conserved_problem = tamen_reference_problem(scheme; conserved=true)
        Random.seed!(20261010)
        conserved_result = run_tamen_result(
            conserved_problem; workspace=work, conserved=true,
        )
        conserved_snapshots = expanded_tamen_snapshots(conserved_result, scheme)
        initial_value = tt_dot(
            conserved_problem.observable, conserved_problem.initial,
        )
        @test all(
            abs(sum(snapshot) - initial_value) <= 2e-5
            for snapshot in conserved_snapshots
        )
    end

    owner_problem = tamen_reference_problem("cheb")
    Random.seed!(20261010)
    first_result = run_tamen_result(owner_problem; workspace=work)
    first_solution = first_result[1]
    saved_cores = deepcopy(core2cell(first_solution))

    larger_problem = merge(owner_problem, (;
        space_time=tkron(owner_problem.initial, tt_ones(5)),
    ))
    Random.seed!(20261011)
    run_tamen_result(larger_problem; workspace=work)
    @test core2cell(first_solution) == saved_cores
end

@testset "TAMEn workspace allocation" begin
    problem = tamen_reference_problem("CN")
    work = TAMEnWorkspace(Float64)

    Random.seed!(20261010)
    run_tamen_result(problem)
    Random.seed!(20261010)
    run_tamen_result(problem; workspace=work)

    Random.seed!(20261010)
    fresh_bytes = @allocated run_tamen_result(problem)
    Random.seed!(20261010)
    reused_bytes = @allocated run_tamen_result(problem; workspace=work)
    @test reused_bytes < fresh_bytes
end

@testset "TAMEn decay accuracy with workspace" begin
    alpha = 0.5
    initial = tt_unit([2], 2)
    space_time = tkron(initial, tt_ones(5))
    generator = TTMatrix([
        reshape(-alpha .* Matrix{Float64}(I, 2, 2), 1, 2, 2, 1),
    ])
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 1,
        :local_iters => 3,
        :time_scheme => "cheb",
        :time_error_damp => 1e3,
    )
    solution, time_grid, _, _, _ = tamen(
        space_time, generator, TAMEN_TOL, options;
        workspace=TAMEnWorkspace(Float64),
    )
    final_state = extract_snapshot(solution, time_grid, 1.0, "cheb")
    @test vec(tt_full(final_state)) ≈
          exp(-alpha) .* vec(tt_full(initial)) atol=5e-3
end
