function tt_ind2sub(n::AbstractVector{<:Integer}, idx::Integer)
    dims = Tuple(n)
    total = prod(dims)
    (idx >= 1 && idx <= total) || error("index out of range")
    return collect(Tuple(CartesianIndices(dims)[idx]))
end

function tt_ones(n::Integer, d::Integer=1; T=Float64)
    return tt_ones(fill(n, d); T=T)
end

function tt_ones(n::AbstractVector{<:Integer}; T=Float64)
    d = length(n)
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        cores[i] = reshape(fill(one(T), n[i]), 1, n[i], 1)
    end
    return TTTensor(cores)
end

function tt_eye(n::Integer, d::Integer=1; T=Float64)
    return tt_eye(fill(n, d); T=T)
end

function tt_eye(n::AbstractVector{<:Integer}; T=Float64)
    d = length(n)
    cores = Vector{Array{T,4}}(undef, d)
    for i in 1:d
        core = Matrix{T}(I, n[i], n[i])
        cores[i] = reshape(core, 1, n[i], n[i], 1)
    end
    return TTMatrix(cores)
end

function tt_unit(n::Integer, d::Integer=1, j::Real=1; T=Float64)
    return tt_unit(fill(n, d), j; T=T)
end

function tt_unit(n::AbstractVector{<:Integer}, j::Union{Real,AbstractVector{<:Integer}}=1; T=Float64)
    d = length(n)
    jval = j isa Real ? Int(round(j)) : j
    jvec = jval isa Integer ? tt_ind2sub(n, jval) : collect(jval)
    length(jvec) == d || error("index vector has wrong length")
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        v = zeros(T, n[i])
        v[jvec[i]] = one(T)
        cores[i] = reshape(v, 1, n[i], 1)
    end
    return TTTensor(cores)
end

function tt_x(n::Integer, d::Integer=1, xmin::Real=0; T=Float64)
    return tt_x(fill(n, d), xmin; T=T)
end

function tt_x(n::AbstractVector{<:Integer}, xmin::Real=0; T=Float64)
    d = length(n)
    cores = Vector{Array{T,3}}(undef, d)
    if d == 1
        vals = collect(0:(n[1] - 1)) .+ xmin
        cores[1] = reshape(T.(vals), 1, n[1], 1)
        return TTTensor(cores)
    end

    core1 = Array{T,3}(undef, 1, n[1], 2)
    core1[1, :, 1] = T.(collect(0:(n[1] - 1)) .+ xmin)
    core1[1, :, 2] = ones(T, n[1])
    cores[1] = core1

    ni = n[1]
    for i in 2:(d - 1)
        core = zeros(T, 2, n[i], 2)
        for j in 1:n[i]
            core[1, j, 1] = one(T)
            core[2, j, 2] = one(T)
            core[2, j, 1] = T(ni * (j - 1))
        end
        cores[i] = core
        ni *= n[i]
    end

    last = Array{T,3}(undef, 2, n[d], 1)
    last[1, :, 1] = ones(T, n[d])
    last[2, :, 1] = T.(collect(0:(n[d] - 1)) .* ni)
    cores[d] = last

    return TTTensor(cores)
end

function diag_offset(n::Integer, offset::Integer, len::Integer; T=Float64)
    if len <= 0
        return zeros(T, n, n)
    end
    return diagm(offset => ones(T, len))
end

function tt_shift(n::Integer, d::Integer, i::Integer; T=Float64)
    return tt_shift(fill(n, d), i; T=T)
end

function tt_shift(n::AbstractVector{<:Integer}, i::Union{Integer,AbstractVector{<:Integer}}; T=Float64)
    d = length(n)
    if prod(n) == 1
        if i == 0
            return tt_eye(1; T=T)
        end
        core = reshape(zeros(T, 1), 1, 1, 1, 1)
        return TTMatrix([core])
    end

    trans = false
    ishift = i isa Integer ? i : minimum(i)
    if ishift < 0
        trans = true
        i = -i
    end

    if i isa Integer
        idx = i + 1
        ivec = tt_ind2sub(n, idx) .- 1
    else
        ivec = collect(i)
    end
    length(ivec) == d || error("shift vector has wrong length")

    cores = Vector{Array{T,4}}(undef, d)

    if d == 1
        main = diag_offset(n[1], -ivec[1], n[1] - ivec[1]; T=T)
        core = reshape(main, 1, n[1], n[1], 1)
        return TTMatrix([core])
    end

    i_d = ivec[d]
    core_d = zeros(T, 2, n[d], n[d], 1)
    core_d[1, :, :, 1] = diag_offset(n[d], -i_d, n[d] - i_d; T=T)
    core_d[2, :, :, 1] = diag_offset(n[d], -(i_d + 1), n[d] - i_d - 1; T=T)
    if trans
        core_d = permutedims(core_d, (1, 3, 2, 4))
    end
    cores[d] = core_d

    for j in 2:(d - 1)
        ij = ivec[j]
        core = zeros(T, 2, n[j], n[j], 2)
        core[1, :, :, 1] = diag_offset(n[j], -ij, n[j] - ij; T=T)
        core[2, :, :, 1] = diag_offset(n[j], -(ij + 1), n[j] - ij - 1; T=T)
        core[1, :, :, 2] = diag_offset(n[j], n[j] - ij, ij; T=T)
        core[2, :, :, 2] = diag_offset(n[j], n[j] - ij - 1, ij + 1; T=T)
        if trans
            core = permutedims(core, (1, 3, 2, 4))
        end
        cores[j] = core
    end

    i1 = ivec[1]
    core1 = zeros(T, 1, n[1], n[1], 2)
    core1[1, :, :, 1] = diag_offset(n[1], -i1, n[1] - i1; T=T)
    core1[1, :, :, 2] = diag_offset(n[1], n[1] - i1, i1; T=T)
    if trans
        core1 = permutedims(core1, (1, 3, 2, 4))
    end
    cores[1] = core1

    return TTMatrix(cores)
end

function tkron(a::TTTensor, b::TTTensor)
    return TTTensor(vcat(a.cores, b.cores))
end

function tkron(a::TTMatrix, b::TTMatrix)
    return TTMatrix(vcat(a.cores, b.cores))
end
