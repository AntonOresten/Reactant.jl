# Standalone acceptance gate for cached object identity and caller-visible effects.
# Run on CPU with FRONTEND=structured (default) or FRONTEND=vanilla.
# Remaining failures are ordinary assertions, not expected-broken skips.
module CachedObjectAcceptance

using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
frontend(f)=get(ENV,"FRONTEND","structured")=="vanilla" ? f : structured(f)
R(x)=Reactant.to_rarray(x;track_numbers=Number)
concrete(x::Reactant.ConcreteRNumber{T}) where T=T(x)
concrete(x::AbstractArray)=Array(x)
concrete(x::Tuple)=map(concrete,x)
concrete(x)=x
same(a::Tuple,b::Tuple)=length(a)==length(b)&&all(same(x,y) for (x,y) in zip(a,b))
same(a::AbstractArray,b::AbstractArray)=axes(a)==axes(b)&&isapprox(a,b;rtol=1e-7,atol=1e-9)
same(a::AbstractFloat,b::AbstractFloat)=isapprox(a,b;rtol=1e-7,atol=1e-9)
same(a,b)=isequal(a,b)
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
static_box(x)=Ref(^)
function fresh_static(x)
    a=cached(static_box,x);b=cached(static_box,x)
    return a===b,a[]===b[],sum(x)
end
aliases(box)=(box,box,Ref(box[]))
function captured_array(x)
    a=Ref(x);b=Ref(x)
    a1,a2,af=cached(aliases,a);b1,b2,bf=cached(aliases,b)
    return a1===a,a1===a2,b1===b,b1===b2,a1===b1,af===a,bf===b,af===bf,af[]===x,bf[]===x,sum(x)
end
function captured_static(x)
    a=Ref(^);b=Ref(^)
    a1,a2,af=cached(aliases,a);b1,b2,bf=cached(aliases,b)
    return a1===a,a1===a2,b1===b,b1===b2,a1===b1,af===a,bf===b,af===bf,af[]===^,bf[]===^,sum(x)
end
mutable struct Node
    value::Any
    link::Any
end
function new_cycle(x)
    n=Node(x,nothing);n.link=n;return n
end
function fresh_cycle(x)
    a=cached(new_cycle,x);b=cached(new_cycle,x)
    return a===a.link,b===b.link,a===b,a.value===x,b.value===x,sum(x)
end
cycle_aliases(n)=(n,n,new_cycle(n.value))
function captured_cycle(x)
    a=new_cycle(x);b=new_cycle(x)
    a1,a2,af=cached(cycle_aliases,a);b1,b2,bf=cached(cycle_aliases,b)
    return a1===a,b1===b,a1===a2,b1===b2,a1===b1,af===bf,af===af.link,bf===bf.link,a1===a1.link,b1===b1.link,sum(x)
end
function rebind(box,x)
    box[]=copy(x);box[] .+=1
    return box
end
function field_rebind(x)
    a=Ref(x);b=Ref(x)
    ar=cached(rebind,a,x);br=cached(rebind,b,x)
    return a[],b[],x,ar===a,br===b,a[]===x,b[]===x,a[]===b[]
end
function detach(box)
    box[] .*=2
    box[]=copy(box[])
    box[] .+=1
    return box
end
function storage_detach(x)
    a=Ref(x);ar=cached(detach,a)
    return a[],x,ar===a,a[]===x
end
function rebind_loss(x)
    a,b,z,_...=field_rebind(x)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end
function detach_loss(x)
    a,z,_...=storage_detach(x)
    return sum(a.^2)+2sum(z.^2)
end
gradient(f,x)=Enzyme.gradient(Reverse,f,x)[1]
key(x)=begin k=Any[];Reactant.make_tracer(Reactant.OrderedIdDict(),x,k,Reactant.TracedToTypes);k end
mutable struct Partial
    a::Any
    b::Any
    Partial()=new()
end
function rebind_noreturn(box,x)
    box[]=copy(x);box[] .*=3
    return nothing
end
function no_return(x)
    a=Ref(x);b=Ref(x)
    cached(rebind_noreturn,a,x);cached(rebind_noreturn,b,x)
    return a[],b[],x,a[]===x,b[]===x,a[]===b[]
end
function replace_nested(parent,x)
    parent.value.value .*=2
    parent.value=new_cycle(copy(x))
    parent.value.value .+=1
    return parent
end
function detached_object(x)
    child=new_cycle(x);parent=Node(child,nothing)
    result=cached(replace_nested,parent,x)
    return x,child.value,parent.value.value,child===child.link,parent.value===child,result===parent
end
increment(box)=(box[]+=1;box)
function static_updates(x)
    a=Ref(1);b=Ref(1)
    ar=cached(increment,a);br=cached(increment,b);again=cached(increment,a)
    return a[],b[],ar===a,br===b,again===a,sum(x)
end
initialize(box,x)=(box.a=x;box)
function field_definedness(x)
    a=Partial();b=Partial()
    ar=cached(initialize,a,x);br=cached(initialize,b,x)
    return isdefined(a,:a),isdefined(a,:b),isdefined(b,:a),isdefined(b,:b),a.a===x,b.a===x,ar===a,br===b,sum(x)
end
function cached_step(x)
    box=Ref(x);cached(rebind_noreturn,box,x)
    return box[]
end
function loop_loss(x,n)
    y=copy(x)
    for i in 1:n
        y=cached_step(y)
    end
    return sum(y.^2)
end
loop_gradient(f,x,n)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(n))[1]
vector_aliases(a)=(a,[a[1]])
function captured_vector(x)
    a=Any[x];b=Any[x]
    ar,af=cached(vector_aliases,a);br,bf=cached(vector_aliases,b)
    return ar===a,br===b,ar===br,af===bf,af[1]===x,bf[1]===x,sum(x)
end
function vector_cycle(x)
    a=Any[x,nothing];a[2]=a;return a
end
function cycle_vector(x)
    a=cached(vector_cycle,x);b=cached(vector_cycle,x)
    return a===a[2],b===b[2],a===b,a[1]===x,b[1]===x,sum(x)
end
function vector_update(a)
    a[1] .*=2
    a[1]=copy(a[1]);a[1] .+=1
    return nothing
end
function vector_mutation(x)
    a=Any[x];cached(vector_update,a)
    return a[1],x,a[1]===x
end
function vector_push(a,x)
    push!(a,copy(x));a[end] .+=1;return a
end
function vector_resize(x)
    a=Any[x];ar=cached(vector_push,a,x)
    return length(a),a[1],a[2],ar===a,a[1]===x,a[2]===x
end
new_host_vector(x)=[^]
function fresh_host_vector(x)
    a=cached(new_host_vector,x);b=cached(new_host_vector,x)
    return a===b,a[1]===b[1],sum(x)
end
function vector_loss(x)
    a,z,_=vector_mutation(x)
    return sum(a.^2)+2sum(z.^2)
end
classify(box)=box[] isa Int ? 10 : 20
function typed_host_field(x)
    a=Ref{Any}(1);b=Ref{Any}(1.0)
    return cached(classify,a),cached(classify,b),sum(x)
end
new_dictionary(x)=Dict(:function=>^)
function fresh_dictionary(x)
    a=cached(new_dictionary,x);b=cached(new_dictionary,x)
    return a===b,a[:function]===b[:function],sum(x)
end

@testset "Cached object graphs: $(get(ENV, "FRONTEND", "structured"))" begin
    @testset "Cache keys" begin
        @test key(Ref(1)) == key(Ref(1))
        @test key(Ref(1)) != key(Ref(2))
        a=Ref(^);b=Ref(^)
        @test key((a,a)) != key((a,b))
        @test key(new_cycle(1)) == key(new_cycle(1))
        a=Partial();a.a=1;b=Partial();b.b=1
        @test key(a) != key(b)
        @test key(vector_cycle(1)) == key(vector_cycle(1))
        @test key(reshape(Any[^,^],1,2)) != key(reshape(Any[^,^],2,1))
        a=Any[^];b=Any[^]
        @test key((a,a)) != key((a,b))
        a=Vector{Any}(undef,2);a[1]=^;b=Vector{Any}(undef,2);b[2]=^
        @test key(a) != key(b)
        # Dynamic host fields can select different Julia methods.
        @test key(Ref{Any}(1)) != key(Ref{Any}(1.0))
    end
    cases=(fresh_static,captured_array,captured_static,fresh_cycle,captured_cycle,
        field_rebind,storage_detach,no_return,detached_object,static_updates,
        field_definedness,captured_vector,cycle_vector,vector_mutation,
        vector_resize,fresh_host_vector,typed_host_field,fresh_dictionary)
    for f in cases
        @testset "$f" begin
            empty!(COUNTS)
            x=[.1,.3,.7,1.1]
            executable=Reactant.compile(frontend(f),(R(copy(x)),))
            expected_counts=f in (storage_detach,detached_object,vector_mutation,vector_resize) ? [1] :
                f===static_updates ? [1,1,2] : f===typed_host_field ? [1,2] : [1,1]
            @test COUNTS == expected_counts
            for values in (x,-2x)
                reference=copy(values);xr=R(copy(values))
                expected=f(reference);actual=concrete(executable(xr))
                @test same(actual,expected)
                @test same(Array(xr),reference)
            end
        end
    end
    for f in (rebind_loss,detach_loss,vector_loss)
        @testset "Nonlinear gradient: $f" begin
            x=[.1,.3,.7,1.1];p=frontend(f)
            executable=Reactant.compile(gradient,(p,R(copy(x))))
            for values in (x,-2x)
                eps=1e-5
                expected=[(f(values.+(eachindex(values).==i).*eps)-f(values.-(eachindex(values).==i).*eps))/(2eps) for i in eachindex(values)]
                @test isapprox(Array(executable(p,R(copy(values)))),expected;rtol=1e-6,atol=1e-7)
            end
        end
    end
    # The explicit structured wrapper makes the counted loop and checkpoint
    # policy observable even when the cached callee uses ordinary Reactant.
    for policy in (:auto,Reactant.Binomial(3),Reactant.Binomial(8))
        @testset "Cached mutation checkpointing: $policy" begin
            x=[.1,.3,.7,1.1];p=structured(loop_loss;checkpointing=policy)
            executable=Reactant.compile(loop_gradient,(p,R(copy(x)),R(3)))
            for n in (0,1,2,4)
                eps=1e-5
                expected=[(loop_loss(x.+(eachindex(x).==i).*eps,n)-loop_loss(x.-(eachindex(x).==i).*eps,n))/(2eps) for i in eachindex(x)]
                @test isapprox(Array(executable(p,R(copy(x)),R(n))),expected;rtol=1e-6,atol=1e-7)
            end
        end
    end
end
end # module
