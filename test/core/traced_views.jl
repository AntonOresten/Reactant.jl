using Reactant, Test

module TracedScalarViewTests
using Reactant, Test, SparseArrays

column_read(x, i) = copy(view(x, :, i))
row_read(x, i) = copy(view(x, i, :))
scalar_read(x, i) = copy(view(x, i, i))
linear_read(x, i) = copy(view(x, i))
nested_read(x, i) = copy(view(view(x, 2:4, :), :, i))
nested_dynamic_read(x, i) = copy(view(view(x, :, i), 2:3))
function column_write!(x, i)
    v = view(x, :, i)
    v .= v .* 2 .+ 1
    return x
end
function nested_write!(x, i)
    v = view(view(x, 2:4, :), :, i)
    v .= v .* 2 .+ 1
    return x
end
function alias_write!(x, i)
    v = view(x, :, i)
    w = view(v, 2:4)
    w .= view(v, 1:3)
    return x
end
strided_read(x,i) = copy(view(x, 1:2:4, i))
nested_strided_read(x,i) = copy(view(view(x, :, 1:2:4), i, :))
function scalar_write!(x,i)
    view(x,i,i) .= 7
    return x
end
function row_write!(x,i)
    v=view(x,i,:); v .= v .* 2 .+ 1
    return x
end
function reshaped_write!(x,i)
    v=view(reshape(x,2,8),:,i); v .= v .* 2 .+ 1
    return x
end
function transposed_write!(x,i)
    v=view(transpose(x),:,i); v .= v .* 2 .+ 1
    return x
end
function independent_copy!(x,i)
    v=view(x,:,i); y=copy(v); v .= 0
    return y
end
function adjoint_write!(x, i)
    v = view(adjoint(x), :, i)
    v .= v .* 2 .+ 1
    return x
end

@testset "Views with traced scalar indices" begin
    for T in (Float64, ComplexF64)
        x = reshape(T.(1:16), 4, 4)
        if T <: Complex
            x .+= im .* reshape(collect(16:-1:1), 4, 4)
        end
        for f in (column_read, row_read, scalar_read, linear_read, nested_read,
                  nested_dynamic_read, column_write!, nested_write!, alias_write!,
                  strided_read, nested_strided_read, scalar_write!, row_write!,
                  reshaped_write!, transposed_write!, independent_copy!, adjoint_write!)
            @testset "$T $f" begin
                xr = Reactant.to_rarray(x)
                ir = Reactant.to_rarray(2; track_numbers=true)
                compiled = @compile f(xr, ir)
                for i in (1, 2, 4)
                    xr = Reactant.to_rarray(x)
                    ir = Reactant.to_rarray(i; track_numbers=true)
                    reference = copy(x)
                    expected = f(reference, i)
                    actual = compiled(xr, ir)
                    @test size(actual) == size(expected)
                    @test Array(actual) ≈ expected
                    @test Array(xr) ≈ reference
                end
            end
        end
    end
end
@testset "Sparse view dispatch" begin
    T = Reactant.TracedRNumber{Float64}
    parents = (
        SparseVector(3, Int[], T[]),
        SparseMatrixCSC(3, 3, [1, 1, 1, 1], Int[], T[]),
    )
    for parent in parents
        v = view(parent, axes(parent)...)
        @test_throws ErrorException("Sparse arrays are not supported by reactant yet.") copy(v)
    end
    for parent in (sparse([1.0, 0.0, 2.0]), sparse([1.0 0.0; 0.0 2.0]))
        v = copy(view(parent, axes(parent)...))
        @test v isa AbstractSparseArray
        @test v == parent
    end
end
end # module
