module InactiveResultAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x;track_numbers=Number)
mixed_result(x)=(value=sum(x.^2), tag=7, positive=sum(x)>0)
forward_result(f,x,d)=Enzyme.autodiff(Forward,Const(f),Duplicated(x,d))
forward_with_primal(f,x,d)=Enzyme.autodiff(ForwardWithPrimal,Const(f),Duplicated(x,d))
forward_batch(f,x,d1,d2)=Enzyme.autodiff(Forward,Const(f),BatchDuplicated(x,(d1,d2)))
forward_batch_with_primal(f,x,d1,d2)=Enzyme.autodiff(ForwardWithPrimal,Const(f),BatchDuplicated(x,(d1,d2)))
@testset "inactive fields in differentiated results" begin
    x=Float64[0.7,1.2,-0.4];d=Float64[-0.2,0.3,0.6];d2=Float64[0.7,-0.5,0.2]
    f=get(ENV,"FRONTEND","structured")=="vanilla" ? mixed_result : structured(mixed_result)
    for (fn,batch,primal) in ((forward_result,false,false),(forward_with_primal,false,true),(forward_batch,true,false),(forward_batch_with_primal,true,true))
        directions=batch ? (d,d2) : (d,)
        compiled=Reactant.compile(fn,(f,R(x),map(R,directions)...))
        @testset "$fn" for input in (x,-x)
            expected=fn(mixed_result,copy(input),map(copy,directions)...)
            actual=compiled(f,R(input),map(R,directions)...)
            @test length(actual)==length(expected)
            tangents=batch ? actual[1] : (actual[1],)
            reference=batch ? expected[1] : (expected[1],)
            for (t,dt) in zip(tangents,reference)
                @test Float64(t.value) ≈ dt.value
                @test Int(t.tag)==dt.tag
                @test Bool(t.positive)==dt.positive
            end
            if primal
                @test Float64(actual[2].value) ≈ expected[2].value
                @test Int(actual[2].tag)==expected[2].tag
                @test Bool(actual[2].positive)==expected[2].positive
            end
        end
    end
end
end
