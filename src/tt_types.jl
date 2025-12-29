struct TTTensor{T}
    cores::Vector{Array{T,3}}
end

struct TTMatrix{T}
    cores::Vector{Array{T,4}}
end

function TTTensor(cores::Vector{<:Array})
    isempty(cores) && error("TTTensor cores must be non-empty")
    first_core = cores[1]
    ndims(first_core) == 3 || error("TTTensor cores must be 3D arrays")
    T = eltype(first_core)
    typed = Vector{Array{T,3}}(undef, length(cores))
    r_prev = size(first_core, 1)
    for (i, core_any) in pairs(cores)
        ndims(core_any) == 3 || error("TTTensor core $i must be 3D")
        core = convert(Array{T,3}, core_any)
        size(core, 1) == r_prev || error("TTTensor rank mismatch at core $i")
        r_prev = size(core, 3)
        typed[i] = core
    end
    return TTTensor{T}(typed)
end

function TTMatrix(cores::Vector{<:Array})
    isempty(cores) && error("TTMatrix cores must be non-empty")
    first_core = cores[1]
    ndims(first_core) == 4 || error("TTMatrix cores must be 4D arrays")
    T = eltype(first_core)
    typed = Vector{Array{T,4}}(undef, length(cores))
    r_prev = size(first_core, 1)
    for (i, core_any) in pairs(cores)
        ndims(core_any) == 4 || error("TTMatrix core $i must be 4D")
        core = convert(Array{T,4}, core_any)
        size(core, 1) == r_prev || error("TTMatrix rank mismatch at core $i")
        r_prev = size(core, 4)
        typed[i] = core
    end
    return TTMatrix{T}(typed)
end

function tt_dims(tt::TTTensor)
    return [size(core, 2) for core in tt.cores]
end

function tt_dims(tt::TTMatrix)
    n = [size(core, 2) for core in tt.cores]
    m = [size(core, 3) for core in tt.cores]
    return n, m
end

function tt_ranks(tt::TTTensor)
    d = length(tt.cores)
    r = Vector{Int}(undef, d + 1)
    r[1] = size(tt.cores[1], 1)
    for i in 1:d
        r[i + 1] = size(tt.cores[i], 3)
    end
    return r
end

function tt_ranks(tt::TTMatrix)
    d = length(tt.cores)
    r = Vector{Int}(undef, d + 1)
    r[1] = size(tt.cores[1], 1)
    for i in 1:d
        r[i + 1] = size(tt.cores[i], 4)
    end
    return r
end

function core2cell(tt::TTTensor)
    return copy(tt.cores)
end

function core2cell(tt::TTMatrix)
    return copy(tt.cores)
end

function cell2core(::Type{TTTensor}, cores::Vector{<:Array})
    return TTTensor(cores)
end

function cell2core(::Type{TTMatrix}, cores::Vector{<:Array})
    return TTMatrix(cores)
end

function tt_matrix(tt::TTTensor, n::Vector{Int}, m::Vector{Int})
    d = length(tt.cores)
    length(n) == d || error("n has wrong length for tt_matrix")
    length(m) == d || error("m has wrong length for tt_matrix")
    cores = Vector{Array{eltype(tt.cores[1]),4}}(undef, d)
    for i in 1:d
        core = tt.cores[i]
        r1 = size(core, 1)
        r2 = size(core, 3)
        size(core, 2) == n[i] * m[i] || error("size mismatch at core $i")
        cores[i] = reshape(core, r1, n[i], m[i], r2)
    end
    return TTMatrix(cores)
end

function tt_matrix(tt::TTTensor)
    n = tt_dims(tt)
    m = ones(Int, length(n))
    return tt_matrix(tt, n, m)
end
