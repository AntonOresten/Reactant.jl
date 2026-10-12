# Standalone CPU acceptance gate; use FRONTEND=structured or FRONTEND=vanilla.
# Includes the preceding object gate. Remaining failures stay ordinary failures.
module DictionaryStateAcceptance
using Test, Reactant, StructuredReactant, Enzyme
include("cached_objects.jl")
using .CachedObjectAcceptance: R, concrete, same, cached, COUNTS, key, frontend, gradient, Node
new_dict(x)=Dict(:function=>^)
function fresh_dict(x)
    a=cached(new_dict,x);b=cached(new_dict,x)
    return a===b,a[:function]===b[:function],sum(x)
end
dict_aliases(d)=(d,d,Dict(:value=>d[:value]))
function captured_dict(x)
    a=Dict(:value=>x);b=Dict(:value=>x)
    a1,a2,af=cached(dict_aliases,a);b1,b2,bf=cached(dict_aliases,b)
    return a1===a,b1===b,a1===a2,b1===b2,a1===b1,af===bf,af[:value]===x,bf[:value]===x,sum(x)
end
function update_dict(d,x)
    d[:value] .*=2
    delete!(d,:value)
    d[:fresh]=copy(x);d[:fresh] .+=1
    d[:other]=d[:fresh]
    return nothing
end
function dict_mutation(x)
    d=Dict{Symbol,Any}(:value=>x,:retained=>Ref(^));old=d[:retained]
    cached(update_dict,d,x)
    return haskey(d,:value),d[:fresh],d[:other],x,d[:fresh]===d[:other],d[:retained]===old
end
function dict_loss(x)
    _,a,b,z,_...=dict_mutation(x)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end
function make_dict_cycle(x)
    d=Dict{Symbol,Any}(:value=>x);d[:self]=d;d
end
function fresh_dict_cycle(x)
    a=cached(make_dict_cycle,x);b=cached(make_dict_cycle,x)
    return a===a[:self],b===b[:self],a===b,a[:value]===x,b[:value]===x,sum(x)
end
function captured_dict_cycle(x)
    a=make_dict_cycle(x);b=make_dict_cycle(x)
    ar,_,af=cached(dict_aliases,a);br,_,bf=cached(dict_aliases,b)
    return ar===a,br===b,ar===ar[:self],br===br[:self],af===bf,sum(x)
end
function detach_dict(holder,x)
    holder[].value[:value] .*=2
    holder[]=Node(Dict(:fresh=>copy(x)),nothing)
    holder[].value[:fresh] .+=1
    return nothing
end
function detached_dict(x)
    d=Dict(:value=>x);node=Node(d,nothing);holder=Ref(node)
    cached(detach_dict,holder,x)
    return d[:value],x,holder[].value[:fresh],holder[]===node,holder[].value===d
end
classify(box)=box[] isa Int ? 10 : 20
function numeric_types(x)
    a=Ref{Any}(1);b=Ref{Any}(1.0)
    return cached(classify,a),cached(classify,b),sum(x)
end
function mutable_keys(d,k)
    return d,k,first(keys(d)),Ref(k)
end
function dict_key_alias(x)
    k=Ref(^);j=Ref(^);a=Dict(k=>x);b=Dict(j=>x)
    ar,kr,ka,boxa=cached(mutable_keys,a,k);br,jr,kb,boxb=cached(mutable_keys,b,j)
    return ar===a,br===b,kr===k,jr===j,ka===k,kb===j,boxa[]===k,boxb[]===j,sum(x)
end
function fresh_key(x)
    k=Ref(^);return Dict(k=>x),k
end
function fresh_dict_key(x)
    a,k=cached(fresh_key,x);b,j=cached(fresh_key,x)
    return a===b,k===j,first(keys(a))===k,first(keys(b))===j,a[k]===x,b[j]===x,sum(x)
end
function clear_dict(d)
    empty!(d);return d
end
function dict_empty(x)
    a=Dict(:value=>x);b=Dict(:value=>x)
    ar=cached(clear_dict,a);br=cached(clear_dict,b)
    return isempty(a),isempty(b),ar===a,br===b,sum(x)
end
first_dict_key(d)=first(keys(d))
function key_iteration(x)
    a=Ref(^);b=Ref(^);d=Dict(a=>x,b=>x)
    before=first(keys(d));returned=cached(first_dict_key,d)
    return before===returned,first(keys(d))===before,sum(x)
end
function key_field_update(d)
    k=first(keys(d));k[]=sin;return k
end
function key_mutation(x)
    k=Ref{Any}(^);d=Dict(k=>x);returned=cached(key_field_update,d)
    return returned===k,k[]===sin,first(keys(d))===k,sum(x)
end
function traced_key(d,k)
    return d,k,first(keys(d))
end
function key_with_tensor(x)
    k=Ref(x);d=Dict(k=>x)
    dr,kr,dk=cached(traced_key,d,k)
    return dr===d,kr===k,dk===k,kr[]===x,sum(x)
end
function conditional_entries(x,c)
    d=Dict{Symbol,Any}(:value=>x)
    if c
        d[:value]=copy(x);d[:value] .*=2
    end
    return d[:value],x,d[:value]===x
end
function conditional_keys(x,c)
    d=Dict{Symbol,Any}(:value=>x)
    if c
        delete!(d,:value);d[:new]=copy(x)
    end
    return haskey(d,:value),haskey(d,:new),sum(x)
end
function dictionary_step(x)
    d=Dict{Symbol,Any}(:value=>x);cached(update_dict,d,x);d[:fresh]
end
function dictionary_loop_loss(x,n)
    y=copy(x)
    for i in 1:n
        y=dictionary_step(y)
    end
    return sum(y.^2)
end
loop_gradient(f,x,n)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(n))[1]
function conditional_alias_values(x,c)
    d=Dict{Symbol,Any}(:a=>x,:b=>x)
    if c
        d[:a]=copy(x)
    end
    d[:a] .*=2
    d[:b] .+=1
    return d[:a],d[:b],x,d[:a]===d[:b]
end
function conditional_rebind_loss(x,c)
    a,b,z,_=conditional_alias_values(x,c)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end
function conditional_host_values(x,c)
    d=Dict{Symbol,Any}(:n=>1)
    if c
        d[:n]=3
    end
    return d[:n]+sum(x)
end
function branch_new_dictionary(x,c)
    a=Dict(:value=>x)
    d=c ? a : Dict(:value=>copy(x))
    d[:value] .*=2
    return x,d[:value],d===a,d[:value]===x
end
function branch_new_cycle(x,c)
    a=make_dict_cycle(x);b=make_dict_cycle(copy(x));d=c ? a : b
    d[:value] .*=2
    return x,d[:value],d===d[:self],d===a
end
gradient_condition(f,x,c)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(c))[1]
function key_iteration_many(x)
 a1=Ref(1);b1=Ref(101);d1=Dict(a1=>x,b1=>x)
 before1=first(keys(d1));returned1=cached(first_dict_key,d1)
 a2=Ref(2);b2=Ref(102);d2=Dict(a2=>x,b2=>x)
 before2=first(keys(d2));returned2=cached(first_dict_key,d2)
 a3=Ref(3);b3=Ref(103);d3=Dict(a3=>x,b3=>x)
 before3=first(keys(d3));returned3=cached(first_dict_key,d3)
 a4=Ref(4);b4=Ref(104);d4=Dict(a4=>x,b4=>x)
 before4=first(keys(d4));returned4=cached(first_dict_key,d4)
 a5=Ref(5);b5=Ref(105);d5=Dict(a5=>x,b5=>x)
 before5=first(keys(d5));returned5=cached(first_dict_key,d5)
 a6=Ref(6);b6=Ref(106);d6=Dict(a6=>x,b6=>x)
 before6=first(keys(d6));returned6=cached(first_dict_key,d6)
 a7=Ref(7);b7=Ref(107);d7=Dict(a7=>x,b7=>x)
 before7=first(keys(d7));returned7=cached(first_dict_key,d7)
 a8=Ref(8);b8=Ref(108);d8=Dict(a8=>x,b8=>x)
 before8=first(keys(d8));returned8=cached(first_dict_key,d8)
 a9=Ref(9);b9=Ref(109);d9=Dict(a9=>x,b9=>x)
 before9=first(keys(d9));returned9=cached(first_dict_key,d9)
 a10=Ref(10);b10=Ref(110);d10=Dict(a10=>x,b10=>x)
 before10=first(keys(d10));returned10=cached(first_dict_key,d10)
 a11=Ref(11);b11=Ref(111);d11=Dict(a11=>x,b11=>x)
 before11=first(keys(d11));returned11=cached(first_dict_key,d11)
 a12=Ref(12);b12=Ref(112);d12=Dict(a12=>x,b12=>x)
 before12=first(keys(d12));returned12=cached(first_dict_key,d12)
 a13=Ref(13);b13=Ref(113);d13=Dict(a13=>x,b13=>x)
 before13=first(keys(d13));returned13=cached(first_dict_key,d13)
 a14=Ref(14);b14=Ref(114);d14=Dict(a14=>x,b14=>x)
 before14=first(keys(d14));returned14=cached(first_dict_key,d14)
 a15=Ref(15);b15=Ref(115);d15=Dict(a15=>x,b15=>x)
 before15=first(keys(d15));returned15=cached(first_dict_key,d15)
 a16=Ref(16);b16=Ref(116);d16=Dict(a16=>x,b16=>x)
 before16=first(keys(d16));returned16=cached(first_dict_key,d16)
 a17=Ref(17);b17=Ref(117);d17=Dict(a17=>x,b17=>x)
 before17=first(keys(d17));returned17=cached(first_dict_key,d17)
 a18=Ref(18);b18=Ref(118);d18=Dict(a18=>x,b18=>x)
 before18=first(keys(d18));returned18=cached(first_dict_key,d18)
 a19=Ref(19);b19=Ref(119);d19=Dict(a19=>x,b19=>x)
 before19=first(keys(d19));returned19=cached(first_dict_key,d19)
 a20=Ref(20);b20=Ref(120);d20=Dict(a20=>x,b20=>x)
 before20=first(keys(d20));returned20=cached(first_dict_key,d20)
 a21=Ref(21);b21=Ref(121);d21=Dict(a21=>x,b21=>x)
 before21=first(keys(d21));returned21=cached(first_dict_key,d21)
 a22=Ref(22);b22=Ref(122);d22=Dict(a22=>x,b22=>x)
 before22=first(keys(d22));returned22=cached(first_dict_key,d22)
 a23=Ref(23);b23=Ref(123);d23=Dict(a23=>x,b23=>x)
 before23=first(keys(d23));returned23=cached(first_dict_key,d23)
 a24=Ref(24);b24=Ref(124);d24=Dict(a24=>x,b24=>x)
 before24=first(keys(d24));returned24=cached(first_dict_key,d24)
 return (before1===returned1,before2===returned2,before3===returned3,before4===returned4,before5===returned5,before6===returned6,before7===returned7,before8===returned8,before9===returned9,before10===returned10,before11===returned11,before12===returned12,before13===returned13,before14===returned14,before15===returned15,before16===returned16,before17===returned17,before18===returned18,before19===returned19,before20===returned20,before21===returned21,before22===returned22,before23===returned23,before24===returned24,sum(x))
end

@testset "Dictionary state: $(get(ENV,"FRONTEND","structured"))" begin
    @testset "Dictionary graph keys" begin
        a=Dict(:x=>^);b=Dict(:x=>^)
        @test key((a,a)) != key((a,b))
        @test key(Dict{Any,Any}(1=>^)) != key(Dict{Any,Any}(1.0=>^))
        @test key(Dict{Symbol,Any}(:x=>1)) != key(Dict{Symbol,Any}(:x=>1.0))
        @test key(make_dict_cycle(1)) == key(make_dict_cycle(1))
    end
    for f in (fresh_dict,captured_dict,dict_mutation,fresh_dict_cycle,
              captured_dict_cycle,detached_dict,numeric_types,dict_key_alias,
              fresh_dict_key,dict_empty)
        @testset "$f" begin
            empty!(COUNTS)
            x=[.1,.3,.7,1.1];executable=Reactant.compile(frontend(f),(R(copy(x)),))
            expected_counts=f in (dict_mutation,detached_dict) ? [1] : f===numeric_types ? [1,2] : [1,1]
            @test COUNTS == expected_counts
            for values in (x,-2x)
                reference=copy(values);xr=R(copy(values));expected=f(reference)
                @test same(concrete(executable(xr)),expected)
                @test same(Array(xr),reference)
            end
        end
    end
    @testset "Key identity and iteration" begin
        x=[.1,.3,.7,1.1]
        for f in (key_mutation,key_with_tensor)
            executable=Reactant.compile(frontend(f),(R(copy(x)),))
            xr=R(copy(x));reference=copy(x);expected=f(reference)
            @test same(concrete(executable(xr)),expected)
            @test same(Array(xr),reference)
        end
        executable=Reactant.compile(frontend(key_iteration_many),(R(copy(x)),))
        actual=concrete(executable(R(copy(x))))
        # Every comparison asks whether an observational cached call chose the
        # same first key as its caller immediately before that call.
        for i in 1:24
            @test actual[i] === true
        end
    end
    for f in (conditional_entries,conditional_keys,conditional_alias_values,
              conditional_host_values,branch_new_dictionary,branch_new_cycle)
        @testset "$f" begin
            x=[.1,.3,.7,1.1];executable=Reactant.compile(structured(f),(R(copy(x)),R(true)))
            for c in (false,true,false,true)
                xr=R(copy(x));reference=copy(x);expected=f(reference,c)
                @test same(concrete(executable(xr,R(c))),expected)
                @test same(Array(xr),reference)
            end
        end
    end
    @testset "Dictionary nonlinear gradient" begin
        x=[.1,.3,.7,1.1];p=frontend(dict_loss)
        executable=Reactant.compile(gradient,(p,R(copy(x))))
        for values in (x,-2x)
            eps=1e-5
            expected=[(dict_loss(values.+(eachindex(values).==i).*eps)-dict_loss(values.-(eachindex(values).==i).*eps))/(2eps) for i in eachindex(values)]
            @test isapprox(Array(executable(p,R(copy(values)))),expected;rtol=1e-6,atol=1e-7)
        end
    end
    @testset "Conditional dictionary gradient" begin
        x=[.1,.3,.7,1.1];p=structured(conditional_rebind_loss)
        executable=Reactant.compile(gradient_condition,(p,R(copy(x)),R(true)))
        for c in (false,true)
            eps=1e-5
            expected=[(conditional_rebind_loss(x.+(eachindex(x).==i).*eps,c)-conditional_rebind_loss(x.-(eachindex(x).==i).*eps,c))/(2eps) for i in eachindex(x)]
            @test isapprox(Array(executable(p,R(copy(x)),R(c))),expected;rtol=1e-6,atol=1e-7)
        end
    end
    for policy in (:auto,Reactant.Binomial(3),Reactant.Binomial(8))
        @testset "Dictionary checkpointing: $policy" begin
            x=[.1,.3,.7,1.1];p=structured(dictionary_loop_loss;checkpointing=policy)
            executable=Reactant.compile(loop_gradient,(p,R(copy(x)),R(3)))
            for n in (0,1,2,4)
                eps=1e-5
                expected=[(dictionary_loop_loss(x.+(eachindex(x).==i).*eps,n)-dictionary_loop_loss(x.-(eachindex(x).==i).*eps,n))/(2eps) for i in eachindex(x)]
                @test isapprox(Array(executable(p,R(copy(x)),R(n))),expected;rtol=1e-6,atol=1e-7)
            end
        end
    end
end
end # module
