# We don't yet support sparse arrays. Throwing errors on this dispatches to avoid
# ambiguities.
function Base.getindex(::AbstractSparseArray{<:TracedRNumber,<:Any,1}, ::Int64)
    return error("Sparse arrays are not supported by reactant yet.")
end

function Base.getindex(::AbstractSparseMatrixCSC{<:TracedRNumber}, ::Int64, ::Int64)
    return error("Sparse arrays are not supported by reactant yet.")
end

function Base.getindex(::CHOLMOD.Sparse{<:TracedRNumber}, ::Int64, ::Int64)
    return error("Sparse arrays are not supported by reactant yet.")
end

# Resolve copy's overlap with SparseArrays while retaining the unsupported-type diagnostic.
function Base.copy(
    ::SubArray{TracedRNumber{T},N,P}
) where {T,N,P<:Union{SparseArrays.SparseVector,AbstractSparseMatrixCSC}}
    return error("Sparse arrays are not supported by reactant yet.")
end
