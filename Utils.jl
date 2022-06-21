using TensorOperations, JuMP, ComplexOptInterface

"""Dense identity matrix."""
eye(d) = Matrix{ComplexF64}(I(d))

"""GHZ state (normalized)."""
function ghz(d::Integer=2; parties::Integer=2, ket::Bool=false)
    ghz = zeros(ComplexF64, d^parties)
    offset = 0
    for p in 0:parties-1
        offset += d^p
    end
    for p in 0:d-1
        ghz[p * offset + 1] = 1 / sqrt(d)
    end
    if ket
        return ghz
    end
    ghz * ghz'
end

"""
Enforce PSD constraint on the input matrix.
Uses dummy vars. to enforce PPT (not possible directly in ComplexOptInterface?).
"""
function ispsd(problem::Model, A::AbstractMatrix)
    @assert size(A,1) == size(A,2) "Matrix must be square."
    @assert length(size(A)) == 2 "Matrix must be bidimensional."

    dim = size(A, 1)
    PSD = @variable(problem, [1:dim, 1:dim] in ComplexOptInterface.HermitianPSDCone())
    @constraint(problem, A .== PSD)
end

"""
Partial trace and partial transpose operations were
taken from https://github.com/iitis/QuantumInformation.jl
"""

"""
- `ρ`: quantum state.
- `idims`: dimensins of subsystems.
- `isystems`: traced subsystems.
Return [partial trace](https://en.wikipedia.org/wiki/Partial_trace) of matrix `ρ` over the subsystems determined by `isystems`.
"""
function ptrace(ρ::AbstractMatrix, idims::Vector{Int}, isystems::Vector{Int})
    dims = reverse(idims)
    systems = length(idims) .- isystems .+ 1

    if size(ρ,1) != size(ρ,2)
        throw(ArgumentError("Non square matrix passed to ptrace"))
    end
    if prod(dims)!=size(ρ,1)
        throw(ArgumentError("Product of dimensions do not match shape of matrix."))
    end
    if maximum(systems) > length(dims) || minimum(systems) < 1
        throw(ArgumentError("System index out of range"))
    end
    offset = length(dims)
    keep = setdiff(1:offset, systems)

    traceidx = [1:offset; 1:offset]
    traceidx[keep] .+= offset

    tensor = reshape(ρ, [dims; dims]...)
    keepdim = prod([size(tensor, x) for x in keep])
    return reshape(tensortrace(tensor, Tuple(traceidx)), keepdim, keepdim)
end

"""
- `ρ`: quantum state.
- `idims`: dimensins of subsystems.
- `sys`: traced subsystem.
"""
ptrace(ρ::AbstractMatrix, idims::Vector{Int}, sys::Int) = ptrace(ρ, idims, [sys])

"""
- `ρ`: quantum state.
- `idims`: dimensins of subsystems.
- `isystems`: transposed subsystems.
"""
function ptranspose(ρ::AbstractMatrix, idims::Vector{Int}, isystems::Vector{Int})
    dims = reverse(idims)
    systems = length(idims) .- isystems .+ 1

    if size(ρ,1)!=size(ρ,2)
        throw(ArgumentError("Non square matrix passed to ptrace"))
    end
    if prod(dims)!=size(ρ,1)
        throw(ArgumentError("Product of dimensions do not match shape of matrix."))
    end
    if maximum(systems) > length(dims) ||  minimum(systems) < 1
        throw(ArgumentError("System index out of range"))
    end

    offset = length(dims)
    tensor = reshape(ρ, [dims; dims]...)
    perm = collect(1:(2offset))
    for s in systems
        idx1 = findfirst(x->x==s, perm)
        idx2 = findfirst(x->x==(s + offset), perm)
        perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
    end
    tensor = permutedims(tensor, invperm(perm))
    reshape(tensor, size(ρ))
end

"""
- `ρ`: quantum state.
- `idims`: dimensins of subsystems.
- `sys`: transposed subsystem.
"""
ptranspose(ρ::AbstractMatrix, idims::Vector{Int}, sys::Int) = ptranspose(ρ, idims, [sys])