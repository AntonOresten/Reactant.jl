using Reactant, Enzyme, Test
pair_sum!(x,y) = (x .*=2; sum(x)+3sum(y))
function alias_sequence(call,x,y,shared_first)
    first=shared_first ? call(pair_sum!,x,x) : call(pair_sum!,x,y)
    second=shared_first ? call(pair_sum!,y,x) : call(pair_sum!,y,y)
    third=call(pair_sum!,x,y)
    fourth=call(pair_sum!,x,x)
    return first+second+third+fourth
end
native_call(f,args...)=f(args...)
cached_call(f,args...)=Reactant.Ops.call(f,args...)
array_alias_calls(x,y,shared_first)=alias_sequence(cached_call,x,y,shared_first)
array_alias_reference(x,y,shared_first)=alias_sequence(native_call,x,y,shared_first)
scalar_pair(x,y)=x+3y
function scalar_alias_calls(x,y,shared_first)
    first=shared_first ? Reactant.Ops.call(scalar_pair,x,x) : Reactant.Ops.call(scalar_pair,x,y)
    second=shared_first ? Reactant.Ops.call(scalar_pair,y,x) : Reactant.Ops.call(scalar_pair,y,y)
    return first+second+Reactant.Ops.call(scalar_pair,x,y)
end
scalar_alias_reference(x,y,s)=s ? scalar_pair(x,x)+scalar_pair(y,x)+scalar_pair(x,y) : scalar_pair(x,y)+scalar_pair(y,y)+scalar_pair(x,y)
alias_gradient(x,y,s)=Enzyme.gradient(Reverse,array_alias_calls,x,y,Enzyme.Const(s))
R(x)=Reactant.to_rarray(x;track_numbers=Number)
@testset "Cached call alias signatures" begin
    for shared in (false,true)
        @testset "shared first = $shared" begin
            primal=Reactant.compile(array_alias_calls,(R([1.0,2.0]),R([3.0,4.0]),shared))
            reverse=Reactant.compile(alias_gradient,(R([1.0,2.0]),R([3.0,4.0]),shared))
            scalar=Reactant.compile(scalar_alias_calls,(R(1.0),R(2.0),shared))
            for scale in (1.0,-0.5,2.0)
                x=[0.1,0.3].*scale;y=[0.7,1.1].*scale
                hx=copy(x);hy=copy(y);expected=array_alias_reference(hx,hy,shared)
                xr=R(x);yr=R(y)
                @test Float64(primal(xr,yr,shared)) ≈ expected
                @test Array(xr) ≈ hx
                @test Array(yr) ≈ hy
                epsilon=1e-5
                dx=[(array_alias_reference(x.+(eachindex(x).==i).*epsilon,copy(y),shared)-array_alias_reference(x.-(eachindex(x).==i).*epsilon,copy(y),shared))/(2epsilon) for i in eachindex(x)]
                dy=[(array_alias_reference(copy(x),y.+(eachindex(y).==i).*epsilon,shared)-array_alias_reference(copy(x),y.-(eachindex(y).==i).*epsilon,shared))/(2epsilon) for i in eachindex(y)]
                grad=reverse(R(x),R(y),shared)
                @test Array(grad[1]) ≈ dx rtol=1e-6 atol=1e-7
                @test Array(grad[2]) ≈ dy rtol=1e-6 atol=1e-7
                @test Float64(scalar(R(scale),R(2scale),shared)) ≈ scalar_alias_reference(scale,2scale,shared)
            end
        end
    end
end
