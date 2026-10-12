# Standalone CPU acceptance gate for host containers crossing traced branches.
module HostArrayStateAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x; track_numbers=Number)
concrete(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
concrete(x::AbstractArray) = Array(x)
concrete(x::Tuple) = map(concrete, x)
concrete(x) = x
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same(x, y) for (x, y) in zip(a, b))
same(a::AbstractArray, b::AbstractArray) = axes(a) == axes(b) && isapprox(a, b; rtol=1e-7, atol=1e-9)
same(a, b) = isequal(a, b)

function host_array_rebind(x, c)
    a = Any[x]
    if c
        a[1] = copy(x)
    end
    a[1] .*= 2
    return a[1], x, a[1] === x
end
function host_ref_value(x, c)
    a = Ref{Any}(1)
    if c
        a[] = 3
    end
    return a[] + sum(x)
end
function shared_vector_elements(x, c)
    a = Any[x, x]
    if c
        a[1] = copy(x)
    end
    a[1] .*= 2
    a[2] .+= 1
    return a[1], a[2], x, a[1] === a[2]
end
function vector_cycle_branch(x, c)
    a = Any[x, nothing]; a[2] = a
    b = Any[copy(x), nothing]; b[2] = b
    r = c ? a : b
    r[1] .*= 2
    return r === r[2], r === a, r[1], x
end
function new_vector_branch(x, c)
    a = Any[x]
    r = c ? a : Any[copy(x)]
    r[1] .*= 2
    return r === a, r[1] === x, x, r[1]
end
function initialized_vector_branch(x, c)
    a = Vector{Any}(undef, 2); a[1] = x
    if c
        a[2] = x
    else
        a[2] = copy(x)
    end
    a[2] .*= 2
    return isassigned(a, 2), a[2] === x, x, a[2]
end
function resized_vector_branch(x, c)
    a = Any[x]
    if c
        push!(a, x)
    else
        push!(a, copy(x))
    end
    a[2] .*= 2
    return length(a), a[2] === x, x, a[2]
end
function selected_matrix(x, c)
    a = Matrix{Any}(undef, 1, 2); a[1, 1] = x; a[1, 2] = x
    b = Matrix{Any}(undef, 1, 2); b[1, 1] = copy(x); b[1, 2] = x
    r = c ? a : b
    r[CartesianIndex(1, 1)] .*= 2
    return r[UInt(1)], r[1, 2], x, r === a
end
# User-defined indexing can contain traced control flow, even when the index
# itself is a host integer. Such calls must still enter the frontend.
struct BranchingIndex{T} <: AbstractVector{Any}
    value::T
end
Base.size(a::BranchingIndex) = (1,)
function Base.getindex(a::BranchingIndex, i::Int)
    if sum(a.value) > 0
        return 2 .* a.value
    end
    return 3 .* a.value
end
function custom_array_read(x, c)
    a = BranchingIndex(x)
    return a[1], x
end
function vector_branch_loss(x, c)
    a, b, z, _ = shared_vector_elements(x, c)
    return sum(a .^ 2) + 2sum(b .^ 2) + 3sum(z .^ 2)
end
gradient_condition(f, x, c) = Enzyme.gradient(Reverse, f, x, Enzyme.Const(c))[1]

@testset "Host container branch state" begin
    for f in (host_array_rebind, host_ref_value, shared_vector_elements,
              vector_cycle_branch, new_vector_branch, initialized_vector_branch,
              resized_vector_branch, selected_matrix, custom_array_read)
        @testset "$f" begin
            x = [0.1, 0.3, 0.7, 1.1]
            executable = Reactant.compile(structured(f), (R(copy(x)), R(true)))
            for values in (x, -2x), c in (false, true, false, true)
                reference = copy(values); xr = R(copy(values))
                expected = f(reference, c)
                @test same(concrete(executable(xr, R(c))), expected)
                @test same(Array(xr), reference)
            end
        end
    end
    @testset "Host vector nonlinear gradient" begin
        x = [0.1, 0.3, 0.7, 1.1]; p = structured(vector_branch_loss)
        executable = Reactant.compile(gradient_condition, (p, R(copy(x)), R(true)))
        for values in (x, -2x), c in (false, true)
            epsilon = 1e-5
            expected = [
                (vector_branch_loss(values .+ (eachindex(values) .== i) .* epsilon, c) -
                 vector_branch_loss(values .- (eachindex(values) .== i) .* epsilon, c)) / (2epsilon)
                for i in eachindex(values)
            ]
            @test isapprox(Array(executable(p, R(copy(values)), R(c))), expected; rtol=1e-6, atol=1e-7)
        end
    end
end
end # module
