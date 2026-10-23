const REFERENCE_COMMIT = "9c0b86c"

const AMEN_REFERENCE_OPTIONS = Dict{Symbol,Any}(
    :nswp => 2,
    :kickrank => 1,
    :verb => 0,
    :local_iters => 80,
    :resid_damp => 1,
    :trunc_norm => "fro",
    :direct_max_full_size => 100,
    :iterative_max_full_size => 1,
    :tol => AMEN_TOL,
)

const TAMEN_REFERENCE_OPTIONS = Dict{Symbol,Any}(
    :nswp => 1,
    :kickrank => 1,
    :verb => 0,
    :local_iters => 2,
    :max_full_size => 100,
    :time_error_damp => 0.0,
    :correct_2_norm => false,
    :tol => TAMEN_TOL,
    :snapshot_times => copy(REFERENCE_TIMES),
)

const AMEN_DIRECT_REFERENCE = [
    0.15548339693299518,
   -0.5699319760138418,
    0.1297077575786305,
    0.4655021213054917,
]

const AMEN_ITERATIVE_REFERENCE = [
    0.15548339693299523,
   -0.5699319760138418,
    0.12970775757863054,
    0.4655021213054916,
]

const TAMEN_CHEB_REFERENCE = [
    [0.7779100416631242, 0.22208995833687634],
    [0.7442880221555506, 0.2557119778444496],
    [0.6790522347872915, 0.3209477652127087],
]

# The historical Chebyshev snapshots above include incorrect interpolation
# weights. This independent reference uses exact rational collocation for
# p' = 1/10 - 3p/10, p(0) = 4/5 at nodes [1/4, 3/4, 1], followed by
# degree-two interpolation at [0, 1/2, 1]. It is not from REFERENCE_COMMIT.
const TAMEN_CHEB_COLLOCATION_REFERENCE = [
    Float64.([495//619, 124//619]),
    Float64.([455//619, 164//619]),
    Float64.([1261//1857, 596//1857]),
]

const TAMEN_CN_REFERENCE = [
    [0.8000000000000002, 0.20000000000000007],
    [0.7348837209302327, 0.26511627906976754],
    [0.678853434288805, 0.3211465657111954],
]
