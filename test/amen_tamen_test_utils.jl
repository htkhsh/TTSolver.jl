const AMEN_TOL = 1e-10
const TAMEN_TOL = 1e-6
const REFERENCE_TIMES = [0.0, 0.5, 1.0]

relative_l2(actual, reference) =
    norm(actual - reference) /
    max(norm(reference), eps(real(eltype(reference))))

function amen_reference_problem(branch::Symbol)
    branch in (:direct, :iterative) ||
        throw(ArgumentError("branch must be :direct or :iterative"))

    dense_operator = [
        4.0 -0.5  0.0  0.2
        0.3  3.5 -0.4  0.0
        0.0  0.25 2.8  0.6
       -0.1  0.0  0.2  3.2
    ]
    dense_rhs = [1.0, -2.0, 0.5, 1.5]
    operator = TTMatrix([reshape(copy(dense_operator), 1, 4, 4, 1)])
    rhs = TTTensor([reshape(copy(dense_rhs), 1, 4, 1)])
    initial = TTTensor([reshape([0.25, -0.5, 0.75, 0.1], 1, 4, 1)])
    residual = Matrix{Any}(undef, 1, 1)
    residual[1, 1] = reshape([0.4, -0.2, 0.1, 0.3], 1, 4, 1, 1)
    options = Dict{Symbol,Any}(
        :nswp => 2,
        :kickrank => 1,
        :verb => 0,
        :local_iters => 80,
        :max_full_size => branch === :direct ? 100 : 1,
        :resid_damp => 1,
        :trunc_norm => "fro",
    )
    return (;
        operator,
        rhs,
        initial,
        residual,
        options,
        dense_operator,
        dense_rhs,
        tol=AMEN_TOL,
    )
end

function run_amen(branch::Symbol)
    problem = amen_reference_problem(branch)
    solution, _, _, _ = amen_solve(
        problem.operator,
        problem.rhs,
        problem.tol,
        copy(problem.options),
        deepcopy(problem.initial),
        nothing,
        deepcopy(problem.residual),
    )
    return vec(tt_full(solution))
end

function tamen_reference_problem(scheme::String; conserved::Bool=false)
    lowercase(scheme) in ("cheb", "cn") ||
        throw(ArgumentError("scheme must be \"cheb\" or \"CN\""))

    initial = TTTensor([reshape([0.8, 0.2], 1, 2, 1)])
    space_time = tkron(initial, tt_ones(3))
    dense_generator = [-0.2 0.1; 0.2 -0.1]
    generator = TTMatrix([reshape(copy(dense_generator), 1, 2, 2, 1)])
    observable = tt_ones(2)
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 1,
        :kickrank => 1,
        :local_iters => 2,
        :max_full_size => 100,
        :time_scheme => scheme,
        :time_error_damp => 0.0,
    )
    observables = conserved ? [observable] : nothing
    return (;
        initial,
        space_time,
        generator,
        observable,
        observables,
        options,
        dense_generator,
        tol=TAMEN_TOL,
        times=copy(REFERENCE_TIMES),
    )
end

function run_tamen(scheme::String; conserved::Bool=false, workspace=nothing)
    problem = tamen_reference_problem(scheme; conserved)
    kwargs = conserved ?
        (; conserved_observables=problem.observables, correct_2_norm=false) :
        (; correct_2_norm=false)
    if workspace !== nothing
        kwargs = merge(kwargs, (; workspace))
    end
    solution, time_grid, _, _, _ = tamen(
        deepcopy(problem.space_time),
        problem.generator,
        problem.tol,
        copy(problem.options);
        kwargs...,
    )
    snapshots = extract_snapshot(solution, time_grid, problem.times, scheme)
    return [vec(tt_full(snapshot)) for snapshot in snapshots]
end
