module DistinctShadowViews
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x;track_numbers=Number)
function loss(v)
    a,b=v
    a .*= 2
    return sum(b.^2)
end
function forward_views(x,flag,d)
    a=flag ? view(x,1:2) : view(x,2:3)
    b=flag ? view(x,2:3) : view(x,1:2)
    da=flag ? view(d,1:2) : view(d,2:3)
    db=flag ? view(d,2:3) : view(d,1:2)
    tangent=Enzyme.autodiff(Forward,Const(loss),Duplicated((a,b),(da,db)))[1]
    return tangent,x,d
end
function reverse_views(x,flag,d)
    a=flag ? view(x,1:2) : view(x,2:3)
    b=flag ? view(x,2:3) : view(x,1:2)
    da=flag ? view(d,1:2) : view(d,2:3)
    db=flag ? view(d,2:3) : view(d,1:2)
    Enzyme.autodiff(Reverse,Const(loss),Active,Duplicated((a,b),(da,db)))
    return x,d
end
@testset "distinct overlapping views share shadow storage" begin
    x=Float64[0.7,1.2,-0.4];d=Float64[-0.2,0.3,0.6]
    for (fn,reverse) in ((forward_views,false),(reverse_views,true))
        @testset "$fn" begin
            compiled=Reactant.compile(structured(fn),(R(x),R(false),R(d)))
            for flag in (false,true)
                a=flag ? (1:2) : (2:3);b=flag ? (2:3) : (1:2)
                z=copy(x);z[a].*=2
                actual=compiled(R(x),R(flag),R(d))
                if reverse
                    expected=copy(d);expected[b].+=2 .* z[b];expected[a].*=2
                    @test Array(actual[1]) ≈ z
                    @test Array(actual[2]) ≈ expected
                else
                    dz=copy(d);dz[a].*=2
                    @test Float64(actual[1]) ≈ 2sum(z[b].*dz[b])
                    @test Array(actual[2]) ≈ z
                    @test Array(actual[3]) ≈ dz
                end
            end
        end
    end
end
end
