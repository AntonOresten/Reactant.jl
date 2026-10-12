# Standalone CPU acceptance gate for dictionary call and executable boundaries.
# FRONTEND selects the cached-call frontend. Remaining failures are ordinary tests.
module DictionaryBoundaryAcceptance
using Test, Reactant, StructuredReactant
Reactant.set_default_backend("cpu")
frontend(f)=get(ENV,"FRONTEND","structured")=="vanilla" ? f : structured(f)
R(x)=Reactant.to_rarray(x;track_numbers=Number)
concrete(x::Reactant.ConcreteRNumber{T}) where T=T(x)
concrete(x::Tuple)=map(concrete,x)
concrete(x)=x
const COUNTS=Int[]
record_cache()=(push!(COUNTS,length(Reactant.Compiler.callcache()));nothing)
function cached(f,args...)
    if Reactant.ReactantCore.within_compile()
        result=Reactant.Ops.call(frontend(f),args...)
        Reactant.call_with_native(record_cache)
        return result
    end
    return f(args...)
end
first_key(d)=first(keys(d))
noop(d)=nothing
function change_values(d,x)
    for k in keys(d)
        d[k]=copy(x)
    end
    return first(keys(d))
end
function integer_history(x,::Val{N},f) where N
    d=Dict{Int,Any}()
    sizehint!(d,2^(3+N%5))
    for k in 1:(12+N)
        d[97k+N]=x
    end
    for k in 1:(N%7)
        delete!(d,97k+N)
    end
    before=Tuple(keys(d))
    result=cached(f,d)
    return result===nothing || result===first(before), Tuple(keys(d))==before
end
function integer_update(x,::Val{N}) where N
    d=Dict{Int,Any}();sizehint!(d,2^(3+N%5))
    for k in 1:(12+N)
        d[97k+N]=x
    end
    for k in 1:(N%7)
        delete!(d,97k+N)
    end
    before=Tuple(keys(d));result=cached(change_values,d,x)
    return result===first(before),Tuple(keys(d))==before,all(v->v!==x,values(d))
end
function new_key_dictionary(x)
    a=Ref(^);b=Ref(^);d=Dict(a=>x,b=>x)
    return d,first(keys(d))
end
function fresh_key_iteration(x)
    d,a=cached(new_key_dictionary,x);e,b=cached(new_key_dictionary,x)
    return first(keys(d))===a,first(keys(e))===b,d!==e,a!==b,haskey(d,a),haskey(e,b),sum(x)
end
function mutable_key_update(d)
    before=first(keys(d));before[]=sin
    return before,first(keys(d))
end
function mutable_key_iteration(x)
    a=Ref{Any}(^);b=Ref{Any}(^);d=Dict(a=>x,b=>x)
    before=first(keys(d));u,v=cached(mutable_key_update,d)
    return before===u,u===v,first(keys(d))===u,haskey(d,u),u[]===sin,sum(x)
end
function integer_reads(x)
    return (integer_history(x,Val(1),first_key),integer_history(x,Val(2),first_key),integer_history(x,Val(3),first_key),integer_history(x,Val(4),first_key),integer_history(x,Val(5),first_key),integer_history(x,Val(6),first_key),integer_history(x,Val(7),first_key),integer_history(x,Val(8),first_key),integer_history(x,Val(9),first_key),integer_history(x,Val(10),first_key),integer_history(x,Val(11),first_key),integer_history(x,Val(12),first_key),integer_history(x,Val(13),first_key),integer_history(x,Val(14),first_key),integer_history(x,Val(15),first_key),integer_history(x,Val(16),first_key),sum(x))
end
function integer_noops(x)
    return (integer_history(x,Val(1),noop),integer_history(x,Val(2),noop),integer_history(x,Val(3),noop),integer_history(x,Val(4),noop),integer_history(x,Val(5),noop),integer_history(x,Val(6),noop),integer_history(x,Val(7),noop),integer_history(x,Val(8),noop),integer_history(x,Val(9),noop),integer_history(x,Val(10),noop),integer_history(x,Val(11),noop),integer_history(x,Val(12),noop),integer_history(x,Val(13),noop),integer_history(x,Val(14),noop),integer_history(x,Val(15),noop),integer_history(x,Val(16),noop),sum(x))
end
function integer_updates(x)
    return (integer_update(x,Val(1)),integer_update(x,Val(2)),integer_update(x,Val(3)),integer_update(x,Val(4)),integer_update(x,Val(5)),integer_update(x,Val(6)),integer_update(x,Val(7)),integer_update(x,Val(8)),integer_update(x,Val(9)),integer_update(x,Val(10)),integer_update(x,Val(11)),integer_update(x,Val(12)),integer_update(x,Val(13)),integer_update(x,Val(14)),integer_update(x,Val(15)),integer_update(x,Val(16)),sum(x))
end

mutable struct EqualKey
    value::Int
end
Base.isequal(a::EqualKey,b::EqualKey)=a.value==b.value
Base.hash(k::EqualKey,h::UInt)=hash(k.value,h)
function replace_equal(d,k,x)
    delete!(d,k)
    fresh=EqualKey(k.value)
    d[fresh]=x
    return fresh
end
function canonical_key_replacement(x)
    ka=EqualKey(1);kb=EqualKey(1);a=Dict(ka=>x);b=Dict(kb=>x)
    ar=cached(replace_equal,a,ka,x);br=cached(replace_equal,b,kb,x)
    return ar!==ka,br!==kb,ar!==br,Base.getkey(a,ka,nothing)===ar,
        Base.getkey(b,kb,nothing)===br,first(keys(a))===ar,first(keys(b))===br,sum(x)
end
function new_mutable_key_dictionary(x)
    a=Ref{Any}(^);b=Ref{Any}(^);d=Dict(a=>x,b=>x)
    return d,first(keys(d))
end
function mutable_key_case(x,::Val{N}) where N
    a=Ref{Any}(N);b=Ref{Any}(N+100);d=Dict(a=>x,b=>x)
    before=first(keys(d));u,v=cached(mutable_key_update,d)
    return before===u,u===v,first(keys(d))===u,haskey(d,u),u[]===sin
end
function fresh_key_many(x)
    d1,k1=cached(new_mutable_key_dictionary,x)
    d2,k2=cached(new_mutable_key_dictionary,x)
    d3,k3=cached(new_mutable_key_dictionary,x)
    d4,k4=cached(new_mutable_key_dictionary,x)
    d5,k5=cached(new_mutable_key_dictionary,x)
    d6,k6=cached(new_mutable_key_dictionary,x)
    d7,k7=cached(new_mutable_key_dictionary,x)
    d8,k8=cached(new_mutable_key_dictionary,x)
    d9,k9=cached(new_mutable_key_dictionary,x)
    d10,k10=cached(new_mutable_key_dictionary,x)
    d11,k11=cached(new_mutable_key_dictionary,x)
    d12,k12=cached(new_mutable_key_dictionary,x)
    d13,k13=cached(new_mutable_key_dictionary,x)
    d14,k14=cached(new_mutable_key_dictionary,x)
    d15,k15=cached(new_mutable_key_dictionary,x)
    d16,k16=cached(new_mutable_key_dictionary,x)
    d17,k17=cached(new_mutable_key_dictionary,x)
    d18,k18=cached(new_mutable_key_dictionary,x)
    d19,k19=cached(new_mutable_key_dictionary,x)
    d20,k20=cached(new_mutable_key_dictionary,x)
    d21,k21=cached(new_mutable_key_dictionary,x)
    d22,k22=cached(new_mutable_key_dictionary,x)
    d23,k23=cached(new_mutable_key_dictionary,x)
    d24,k24=cached(new_mutable_key_dictionary,x)
    d25,k25=cached(new_mutable_key_dictionary,x)
    d26,k26=cached(new_mutable_key_dictionary,x)
    d27,k27=cached(new_mutable_key_dictionary,x)
    d28,k28=cached(new_mutable_key_dictionary,x)
    d29,k29=cached(new_mutable_key_dictionary,x)
    d30,k30=cached(new_mutable_key_dictionary,x)
    d31,k31=cached(new_mutable_key_dictionary,x)
    d32,k32=cached(new_mutable_key_dictionary,x)
    k1[]=sin
    return ((first(keys(d1))===k1,haskey(d1,k1),k1[]===sin),(first(keys(d2))===k2,haskey(d2,k2),d2!==d1,k2!==k1,k2[]===^),(first(keys(d3))===k3,haskey(d3,k3),d3!==d2,k3!==k2,k3[]===^),(first(keys(d4))===k4,haskey(d4,k4),d4!==d3,k4!==k3,k4[]===^),(first(keys(d5))===k5,haskey(d5,k5),d5!==d4,k5!==k4,k5[]===^),(first(keys(d6))===k6,haskey(d6,k6),d6!==d5,k6!==k5,k6[]===^),(first(keys(d7))===k7,haskey(d7,k7),d7!==d6,k7!==k6,k7[]===^),(first(keys(d8))===k8,haskey(d8,k8),d8!==d7,k8!==k7,k8[]===^),(first(keys(d9))===k9,haskey(d9,k9),d9!==d8,k9!==k8,k9[]===^),(first(keys(d10))===k10,haskey(d10,k10),d10!==d9,k10!==k9,k10[]===^),(first(keys(d11))===k11,haskey(d11,k11),d11!==d10,k11!==k10,k11[]===^),(first(keys(d12))===k12,haskey(d12,k12),d12!==d11,k12!==k11,k12[]===^),(first(keys(d13))===k13,haskey(d13,k13),d13!==d12,k13!==k12,k13[]===^),(first(keys(d14))===k14,haskey(d14,k14),d14!==d13,k14!==k13,k14[]===^),(first(keys(d15))===k15,haskey(d15,k15),d15!==d14,k15!==k14,k15[]===^),(first(keys(d16))===k16,haskey(d16,k16),d16!==d15,k16!==k15,k16[]===^),(first(keys(d17))===k17,haskey(d17,k17),d17!==d16,k17!==k16,k17[]===^),(first(keys(d18))===k18,haskey(d18,k18),d18!==d17,k18!==k17,k18[]===^),(first(keys(d19))===k19,haskey(d19,k19),d19!==d18,k19!==k18,k19[]===^),(first(keys(d20))===k20,haskey(d20,k20),d20!==d19,k20!==k19,k20[]===^),(first(keys(d21))===k21,haskey(d21,k21),d21!==d20,k21!==k20,k21[]===^),(first(keys(d22))===k22,haskey(d22,k22),d22!==d21,k22!==k21,k22[]===^),(first(keys(d23))===k23,haskey(d23,k23),d23!==d22,k23!==k22,k23[]===^),(first(keys(d24))===k24,haskey(d24,k24),d24!==d23,k24!==k23,k24[]===^),(first(keys(d25))===k25,haskey(d25,k25),d25!==d24,k25!==k24,k25[]===^),(first(keys(d26))===k26,haskey(d26,k26),d26!==d25,k26!==k25,k26[]===^),(first(keys(d27))===k27,haskey(d27,k27),d27!==d26,k27!==k26,k27[]===^),(first(keys(d28))===k28,haskey(d28,k28),d28!==d27,k28!==k27,k28[]===^),(first(keys(d29))===k29,haskey(d29,k29),d29!==d28,k29!==k28,k29[]===^),(first(keys(d30))===k30,haskey(d30,k30),d30!==d29,k30!==k29,k30[]===^),(first(keys(d31))===k31,haskey(d31,k31),d31!==d30,k31!==k30,k31[]===^),(first(keys(d32))===k32,haskey(d32,k32),d32!==d31,k32!==k31,k32[]===^),sum(x))
end
function mutable_key_many(x)
    return (mutable_key_case(x,Val(1)),mutable_key_case(x,Val(2)),mutable_key_case(x,Val(3)),mutable_key_case(x,Val(4)),mutable_key_case(x,Val(5)),mutable_key_case(x,Val(6)),mutable_key_case(x,Val(7)),mutable_key_case(x,Val(8)),mutable_key_case(x,Val(9)),mutable_key_case(x,Val(10)),mutable_key_case(x,Val(11)),mutable_key_case(x,Val(12)),mutable_key_case(x,Val(13)),mutable_key_case(x,Val(14)),mutable_key_case(x,Val(15)),mutable_key_case(x,Val(16)),sum(x))
end
function static_key(x)
    k=Ref(^);return Dict(k=>x),k
end
function tensor_key(x)
    k=Ref(x);return Dict(k=>x),k
end
function key_value_alias(x)
    k=Ref(x);return Dict(k=>k),k
end
cached_static(x)=cached(static_key,x)
cached_tensor(x)=cached(tensor_key,x)
cached_alias(x)=cached(key_value_alias,x)

@testset "Dictionary boundaries: $(get(ENV,"FRONTEND","structured"))" begin
    for f in (integer_reads,integer_noops,integer_updates,mutable_key_many,fresh_key_many)
        @testset "$f" begin
            empty!(COUNTS)
            x=[.1,.3,.7,1.1];executable=Reactant.compile(frontend(f),(R(copy(x)),))
            n=f===fresh_key_many ? 32 : 16
            @test COUNTS == (f===fresh_key_many ? fill(1,n) : collect(1:n))
            for values in (x,-2x)
                expected=f(copy(values));actual=concrete(executable(R(copy(values))))
                for i in 1:n
                    @test actual[i] == expected[i]
                end
                @test actual[end] ≈ expected[end]
            end
        end
    end
    @testset "Canonical equal-key replacement" begin
        empty!(COUNTS)
        x=[.1,.3,.7,1.1];executable=Reactant.compile(frontend(canonical_key_replacement),(R(copy(x)),))
        @test COUNTS == [1,1]
        for values in (x,-2x)
            @test concrete(executable(R(copy(values)))) == canonical_key_replacement(copy(values))
        end
    end
    for f in (static_key,tensor_key,key_value_alias,cached_static,cached_tensor,cached_alias)
        @testset "Executable result: $f" begin
            empty!(COUNTS)
            x=[.1,.3,.7,1.1];a=R(copy(x));b=R(-2x)
            executable=Reactant.compile(frontend(f),(a,))
            @test COUNTS == (f in (cached_static,cached_tensor,cached_alias) ? [1] : Int[])
            d,k=executable(a);e,j=executable(b)
            @test d!==e
            @test k!==j
            @test first(keys(d))===k
            @test first(keys(e))===j
            @test first(keys(d))!==first(keys(e))
            @test haskey(d,k)
            @test haskey(e,j)
            if f in (key_value_alias,cached_alias)
                @test first(values(d))===k
                @test first(values(e))===j
            else
                @test Array(first(values(d)))==x
                @test Array(first(values(e)))==-2x
            end
            if f in (tensor_key,cached_tensor)
                @test k[]===first(values(d))
                @test j[]===first(values(e))
            end
            if f in (tensor_key,key_value_alias,cached_tensor,cached_alias)
                @test k[] isa Reactant.AbstractConcreteArray && Array(k[])==x
                @test j[] isa Reactant.AbstractConcreteArray && Array(j[])==-2x
            end
        end
    end
end
end # module
