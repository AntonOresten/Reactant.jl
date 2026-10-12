module ReferenceADAnnotations
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x; track_numbers=Number)
cubic(x)=sum(x .^ 3)
function mutation_loss(x)
    x .*= 2
    return sum(x .^ 2)
end
function selected_forward(x,y,flag,direction)
    selected=flag ? x : y
    return Enzyme.autodiff(Forward,Const(cubic),Duplicated(selected,direction))[1]
end
function selected_reverse(x,y,flag,seed)
    selected=flag ? x : y
    Enzyme.autodiff(Reverse,Const(mutation_loss),Active,Duplicated(selected,seed))
    return seed,x,y
end
@testset "explicit derivative buffers across references" begin
    x=Float64[0.7,1.2,-0.4]
    y=Float64[-1.1,0.5,1.8]
    direction=Float64[-0.2,0.3,0.6]
    @testset "forward dense direction" begin
        f=Reactant.compile(structured(selected_forward),(R(x),R(y),R(false),R(direction)))
        for flag in (false,true)
            @test Float64(f(R(x),R(y),R(flag),R(direction))) ≈ sum(3 .* (flag ? x : y).^2 .* direction)
        end
    end
    @testset "reverse dense seed and mutation" begin
        f=Reactant.compile(structured(selected_reverse),(R(x),R(y),R(false),R(direction)))
        for flag in (false,true)
            derivative,ax,ay=f(R(x),R(y),R(flag),R(direction))
            @test Array(derivative) ≈ 8 .* (flag ? x : y) .+ 2 .* direction
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
        end
    end
end
end
