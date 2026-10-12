# Standalone acceptance gate for the experimental reference implementation.
# Run separately from runtests.jl; failures remain ordinary assertions/errors.
# Each primal checks observable results, identity and mutation of the input.
module MutableObjectAcceptance

using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x;track_numbers=Number)
concrete(x::Reactant.ConcreteRNumber{T}) where {T}=T(x)
concrete(x::AbstractArray)=Array(x)
concrete(x::Tuple)=map(concrete,x)
concrete(x)=x
same(a::Tuple,b::Tuple)=length(a)==length(b) && all(same(x,y) for (x,y) in zip(a,b))
same(a::AbstractArray,b::AbstractArray)=axes(a)==axes(b) && isapprox(a,b;rtol=1e-7,atol=1e-9)
same(a::AbstractFloat,b::AbstractFloat)=isapprox(a,b;rtol=1e-7,atol=1e-9)
same(a,b)=isequal(a,b)
function singleton_choice(x,c)
    a=Ref(^);r=c ? a : Ref(^)
    return r===a,r[]===^,r===r,sum(x)
end
function new_box_identity(x,c)
    a=c ? Ref(^) : Ref(^);b=Ref(^)
    return a===b,a[]===b[],sum(x)
end
function boxed_arrays(x,c)
    r=c ? Ref(x) : Ref(copy(x))
    return r[]===x,sum(r[]),r===r
end
function retained_box_alias(x,c)
    a=Ref(x);b=Ref(copy(x));r=c ? a : b
    r[] .+=1
    return a[],b[],x,r===a,r===b
end
function rebind_box_array(x,c)
    a=Ref(x);b=Ref(copy(x));r=c ? a : b
    r[]=copy(x)
    r[] .+=1
    return a[],b[],x,r[]===x,r===a,r===b
end
function rebind_box_scalar(x,c)
    a=Ref(sum(x));b=Ref(2sum(x));r=c ? a : b
    r[]=3sum(x)
    return a[],b[],sum(x)
end
function host_box_alias(x,c)
    a=Ref(1);b=Ref(2);r=c ? a : b
    r[]=3
    return a[],b[],sum(x)
end
mutable struct GraphNode
    value::Any
    link::Any
end
function graph_cycle(x,c)
    a=GraphNode(x,nothing);a.link=a
    b=GraphNode(copy(x),nothing);b.link=b
    r=c ? a : b
    return r===r.link,r.value===x,sum(r.value)
end
make_box(x,c)=c ? Ref(x) : Ref(copy(x))
function cached_box(x,c)
    Reactant.ReactantCore.within_compile() && return Reactant.Ops.call(structured(make_box),x,c)
    return make_box(x,c)
end
function cached_box_identity(x,c)
    a=cached_box(x,c);b=cached_box(x,c)
    return a===b,a[]===x,b[]===x
end
make_static_box(x,c)=c ? Ref(^) : Ref(^)
function cached_static_box(x,c)
    Reactant.ReactantCore.within_compile() && return Reactant.Ops.call(structured(make_static_box),x,c)
    return make_static_box(x,c)
end
function cached_static_box_identity(x,c)
    a=cached_static_box(x,c);b=cached_static_box(x,c)
    return a===b,a[]===b[],sum(x)
end
function box_loss(x,c)
    r=c ? Ref(x) : Ref(copy(x))
    r[] .*=2
    return sum(r[].^2)
end
gradient(f,x,c)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(c))[1]

mutable struct Holder
    value::Any
    other::Any
end
function generic_fields(x,c)
    a=Holder(x,x); b=Holder(copy(x),x); r=c ? a : b
    r.value=copy(x)
    r.value .*=2
    r.other .+=1
    return a.value,b.value,x,r.value===x,r.other===x
end
function disconnected_storage(x,c)
    a=Ref(x)
    if c
        a[] .*=2
        a[]=copy(a[])
    end
    return a[],x,a[]===x
end
function sibling_fields(x,c)
    a=Holder(x,x)
    if c
        a.value=copy(x)
    end
    a.value .*=2
    a.other .+=1
    return a.value,a.other,x,a.value===a.other
end
function generic_loss(x,c)
    a,b,z,_...=generic_fields(x,c)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end

function cyclic_shared_fields(x,c)
    node=GraphNode(x,nothing); node.link=node
    other=GraphNode(copy(x),nothing); other.link=other
    a=Holder(node,node); b=Holder(other,node); r=c ? a : b
    r.value.value .*=2
    return x,r.value.value,r.other.value,r.value===r.value.link,r.value===r.other
end
function cyclic_shared_loss(x,c)
    a,b,z,_...=cyclic_shared_fields(x,c)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end

function nonlinear_rebind_loss(x,c)
    a,b,z,_...=rebind_box_array(x,c)
    return sum(a.^2)+2sum(b.^2)+3sum(z.^2)
end

@testset "Mutable objects across regions and calls" begin
    functions=(singleton_choice,new_box_identity,boxed_arrays,retained_box_alias,
        rebind_box_array,rebind_box_scalar,host_box_alias,graph_cycle,
        cached_box_identity,cached_static_box_identity,generic_fields,
        disconnected_storage,sibling_fields,cyclic_shared_fields,nonlinear_rebind_loss)
    for f in functions
        @testset "$f" begin
            x=[.1,.3,.7,1.1]
            compiled=Reactant.compile(structured(f),(R(copy(x)),R(true)))
            for c in (false,true,false,true)
                xr=R(copy(x)); reference=copy(x)
                actual=concrete(compiled(xr,R(c))); expected=f(reference,c)
                @test same(actual,expected) && same(Array(xr),reference)
            end
        end
    end
    for f in (box_loss,nonlinear_rebind_loss,generic_loss,cyclic_shared_loss)
        @testset "Gradient: $f" begin
            x=[.1,.3,.7,1.1]; program=structured(f)
            compiled=Reactant.compile(gradient,(program,R(copy(x)),R(true)))
            for c in (false,true)
                epsilon=1e-5
                expected=[(f(x .+ (eachindex(x).==i).*epsilon,c)-
                           f(x .- (eachindex(x).==i).*epsilon,c))/(2epsilon)
                          for i in eachindex(x)]
                actual=Array(compiled(program,R(copy(x)),R(c)))
                @test isapprox(actual,expected;rtol=1e-6,atol=1e-7)
            end
        end
    end
end

end # module
