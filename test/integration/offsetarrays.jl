using Reactant, Test, OffsetArrays

function scalar_index(x)
    @allowscalar getindex(x, -1, 0)
end

@testset "OffsetArrays" begin
    A = Float64.(reshape(1:15, 3, 5))
    OA = OffsetArray(A, -1:1, 0:4)
    rOA = Reactant.to_rarray(OA)

    oval = scalar_index(OA)
    cval = scalar_index(rOA)
    @test cval ≈ oval

    tval = @jit scalar_index(rOA)
    @test tval ≈ oval
end

@testset "OffsetArray View" begin
    U = zeros(Float64, 128, 128, 1)
    vU = OffsetArray(U, -7:120, -7:120, 1:1)
    rU = Reactant.to_rarray(vU)

    @jit fill!(@view(rU[1:112, 1:112, 1]), 1.0)
    fill!(@view(vU[1:112, 1:112, 1]), 1.0)

    @test parent(rU) ≈ parent(vU)
end

module OffsetViewMutationTests
using Reactant, Test, OffsetArrays, LinearAlgebra

function column!(x, i)
    v = view(x, :, i)
    v .= v .* 2 .+ 1
    return x
end
function nested!(x, i)
    v = view(view(x, -1:1, :), :, i)
    v .= v .* 2 .+ 1
    return x
end
function transposed!(x, i)
    v = view(transpose(x), i, :)
    v .= v .* 2 .+ 1
    return x
end
function adjointed!(x, i)
    v = view(adjoint(x), i, :)
    v .= v .* (2.0 + 1.0im) .+ 1
    return x
end
function offset_adjoint!(x, i)
    v = view(OffsetArray(adjoint(parent(x)), -1:2, 2:5), :, i)
    v .= v .* (2.0 + 1.0im) .+ 1
    return x
end

@testset "Offset view writeback" begin
    for T in (Float64, ComplexF64)
        values = reshape(T.(1:16), 4, 4)
        T <: Complex && (values .+= im .* reverse(values))
        functions = T <: Complex ?
            (column!, nested!, transposed!, adjointed!, offset_adjoint!) :
            (column!, nested!, transposed!)
        for f in functions
            input = OffsetArray(copy(values), -1:2, 2:5)
            compiled = @compile f(Reactant.to_rarray(input), Reactant.to_rarray(2; track_numbers=true))
            for i in (2, 3, 5)
                reference = OffsetArray(copy(values), -1:2, 2:5)
                expected = f(reference, i)
                actual_input = Reactant.to_rarray(OffsetArray(copy(values), -1:2, 2:5))
                actual = compiled(actual_input, Reactant.to_rarray(i; track_numbers=true))
                @test axes(actual) == axes(expected)
                @test Array(parent(actual)) ≈ parent(expected)
                @test Array(parent(actual_input)) ≈ parent(reference)
            end
        end
    end
end
end

@testset "Host offset view materialization" begin
    values = reshape(collect(1.0:16.0), 4, 4)
    offset = OffsetArray(values, -1:2, 2:5)
    for v in (view(offset, -1:1, 2:4), view(offset, :, 3), view(offset, -1:1, :))
        materialized = Reactant.ReactantCore.materialize_traced_array(v)
        @test materialized == v
    end
end
