module ReferenceShadowAliases
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x;track_numbers=Number)
function alias_loss(pair)
    a,b=pair
    a .*= 2
    return sum(b .^ 2)
end
function alias_forward(x,y,flag,d)
    a=flag ? x : y
    b=flag ? x : y
    derivative=Enzyme.autodiff(Forward,Const(alias_loss),Duplicated((a,b),(d,d)))[1]
    return derivative,x,y,d
end
function alias_reverse(x,y,flag,d)
    a=flag ? x : y
    b=flag ? x : y
    Enzyme.autodiff(Reverse,Const(alias_loss),Active,Duplicated((a,b),(d,d)))
    return d,x,y
end
@testset "shared explicit derivative storage" begin
    x=Float64[0.7,1.2,-0.4];y=Float64[-1.1,0.5,1.8];d=Float64[-0.2,0.3,0.6]
    @testset "forward aliases and writeback" begin
        f=Reactant.compile(structured(alias_forward),(R(x),R(y),R(false),R(d)))
        for flag in (false,true)
            selected=flag ? x : y
            derivative,ax,ay,ad=f(R(x),R(y),R(flag),R(d))
            @test Float64(derivative) ≈ sum(8 .* selected .* d)
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
            @test Array(ad) ≈ 2 .* d
        end
    end
    @testset "reverse aliases and seed accumulation" begin
        f=Reactant.compile(structured(alias_reverse),(R(x),R(y),R(false),R(d)))
        for flag in (false,true)
            selected=flag ? x : y
            derivative,ax,ay=f(R(x),R(y),R(flag),R(d))
            @test Array(derivative) ≈ 8 .* selected .+ 2 .* d
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
        end
    end
end
end
