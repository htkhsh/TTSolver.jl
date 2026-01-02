import Base: +, -, *
import LinearAlgebra: dot, norm

function +(a::TTTensor, b::TTTensor)
    na = tt_dims(a)
    nb = tt_dims(b)
    na == nb || error("mode sizes are inconsistent")

    da = length(a.cores)
    db = length(b.cores)
    da == db || error("number of cores is inconsistent")
    if da == 1
        return TTTensor([a.cores[1] .+ b.cores[1]])
    end

    ra = tt_ranks(a)
    rb = tt_ranks(b)
    ra[1] == rb[1] || error("left TT rank is inconsistent")
    ra[end] == rb[end] || error("right TT rank is inconsistent")

    cores = Vector{Array{promote_type(eltype(a.cores[1]), eltype(b.cores[1])),3}}(undef, da)
    for i in 1:da
        r1a, n, r2a = size(a.cores[i])
        r1b, _, r2b = size(b.cores[i])
        r1 = i == 1 ? r1a : r1a + r1b
        r2 = i == da ? r2a : r2a + r2b
        core = zeros(promote_type(eltype(a.cores[i]), eltype(b.cores[i])), r1, n, r2)
        if i == 1
            core[:, :, 1:r2a] .= a.cores[i]
            core[:, :, (r2 - r2b + 1):r2] .= b.cores[i]
        elseif i == da
            core[1:r1a, :, :] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :] .= b.cores[i]
        else
            core[1:r1a, :, 1:r2a] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, (r2 - r2b + 1):r2] .= b.cores[i]
        end
        cores[i] = core
    end
    return TTTensor(cores)
end

function +(a::TTMatrix, b::TTMatrix)
    na, ma = tt_dims(a)
    nb, mb = tt_dims(b)
    na == nb || error("row mode sizes are inconsistent")
    ma == mb || error("col mode sizes are inconsistent")

    da = length(a.cores)
    db = length(b.cores)
    da == db || error("number of cores is inconsistent")

    ra = tt_ranks(a)
    rb = tt_ranks(b)
    ra[1] == rb[1] || error("left TT rank is inconsistent")
    ra[end] == rb[end] || error("right TT rank is inconsistent")

    cores = Vector{Array{promote_type(eltype(a.cores[1]), eltype(b.cores[1])),4}}(undef, da)
    for i in 1:da
        r1a, n, m, r2a = size(a.cores[i])
        r1b, _, _, r2b = size(b.cores[i])
        r1 = i == 1 ? r1a : r1a + r1b
        r2 = i == da ? r2a : r2a + r2b
        core = zeros(promote_type(eltype(a.cores[i]), eltype(b.cores[i])), r1, n, m, r2)
        if i == 1
            core[:, :, :, 1:r2a] .= a.cores[i]
            core[:, :, :, (r2 - r2b + 1):r2] .= b.cores[i]
        elseif i == da
            core[1:r1a, :, :, :] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :, :] .= b.cores[i]
        else
            core[1:r1a, :, :, 1:r2a] .= a.cores[i]
            core[(r1 - r1b + 1):r1, :, :, (r2 - r2b + 1):r2] .= b.cores[i]
        end
        cores[i] = core
    end
    return TTMatrix(cores)
end

-(a::TTTensor, b::TTTensor) = a + (-one(eltype(a.cores[1])) * b)
-(a::TTMatrix, b::TTMatrix) = a + (-one(eltype(a.cores[1])) * b)

function *(a::TTTensor, α::Number)
    cores = [copy(core) for core in a.cores]
    cores[1] = cores[1] .* α
    return TTTensor(cores)
end
*(α::Number, a::TTTensor) = a * α

function *(a::TTMatrix, α::Number)
    cores = [copy(core) for core in a.cores]
    cores[1] = cores[1] .* α
    return TTMatrix(cores)
end
*(α::Number, a::TTMatrix) = a * α

function +(a::TTTensor, α::Number)
    T = promote_type(eltype(a.cores[1]), typeof(α))
    return a + (α * tt_ones(tt_dims(a); T=T))
end

+(α::Number, a::TTTensor) = a + α

function -(a::TTTensor, α::Number)
    return a + (-α)
end

-(α::Number, a::TTTensor) = α + (-one(eltype(a.cores[1])) * a)

function *(A::TTMatrix, x::TTTensor)
    nA, mA = tt_dims(A)
    nX = tt_dims(x)
    mA == nX || error("matrix column modes must match input sizes")

    d = length(nA)
    T = promote_type(eltype(A.cores[1]), eltype(x.cores[1]))
    cores = Vector{Array{T,3}}(undef, d)
    for i in 1:d
        coreA = A.cores[i]
        coreX = x.cores[i]
        @tensor coreY[ra1, rx1, n, ra2, rx2] := coreA[ra1, n, m, ra2] * coreX[rx1, m, rx2]
        r1 = size(coreA, 1) * size(coreX, 1)
        r2 = size(coreA, 4) * size(coreX, 3)
        cores[i] = reshape(coreY, r1, nA[i], r2)
    end
    return TTTensor(cores)
end

function *(A::TTMatrix, B::TTMatrix)
    nA, mA = tt_dims(A)
    nB, mB = tt_dims(B)
    mA == nB || error("matrix inner sizes are inconsistent")

    d = length(nA)
    T = promote_type(eltype(A.cores[1]), eltype(B.cores[1]))
    cores = Vector{Array{T,4}}(undef, d)
    for i in 1:d
        coreA = A.cores[i]
        coreB = B.cores[i]
        @tensor coreC[ra1, rb1, n, m, ra2, rb2] := coreA[ra1, n, p, ra2] * coreB[rb1, p, m, rb2]
        r1 = size(coreA, 1) * size(coreB, 1)
        r2 = size(coreA, 4) * size(coreB, 4)
        cores[i] = reshape(coreC, r1, nA[i], mB[i], r2)
    end
    return TTMatrix(cores)
end

function dot(a::TTTensor, b::TTTensor)
    return dot_dR(a, b)
end

function norm(a::TTTensor)
    return sqrt(real(dot(a, a)))
end
