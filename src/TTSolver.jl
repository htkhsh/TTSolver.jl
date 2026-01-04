module TTSolver

using LinearAlgebra
using Random
using SparseArrays
using TensorOperations
using ExponentialUtilities: expv

include("tt_types.jl")
include("tt_basic.jl")
include("dr_utils.jl")
include("tt_ops.jl")
include("amen_utils.jl")
include("amen_solve.jl")
include("time_utils.jl")
include("tamen.jl")
include("extract_snapshot.jl")
include("tt_ksl.jl")

export TTTensor, TTMatrix
export tt_dims, tt_ranks, core2cell, cell2core, tt_matrix
export tt_ind2sub, tt_ones, tt_eye, tt_unit, tt_x, tt_shift, tt_rand
export tt_kron, tt_mkron, tt_full, tt_dot, tt_norm
export tt_round, tt_hadamard
export check_vector, check_matrix
export amen_solve
export chebdiff, cheb2_interpolant, tamen
export extract_snapshot
export tt_ksl, expv_full

# Deprecated aliases (for backward compatibility)
const tkron = tt_kron
const expand_dR = tt_full
const dot_dR = tt_dot
export tkron, expand_dR, dot_dR

end
