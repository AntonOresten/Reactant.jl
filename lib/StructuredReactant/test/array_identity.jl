# Explicit acceptance gate for observable array identity. This is kept outside
# runtests.jl until branch/loop references preserve these semantics completely.
module ArrayIdentityAcceptance
using Reactant, StructuredReactant, Enzyme, Test, LinearAlgebra
Reactant.set_default_backend("cpu")
if haskey(ENV,"EXPECTED_REACTANT_ROOT")
    @assert realpath(pkgdir(Reactant)) == realpath(ENV["EXPECTED_REACTANT_ROOT"])
end
R(x)=Reactant.to_rarray(x;track_numbers=Number)
function retain_previous(x,n)
    a=copy(x);b=copy(x)
    for i in 1:n
        b=a
        a=copy(x)
    end
    return a===b,b===x,a===x
end
function retain_original(x,n)
    a=x;b=copy(x)
    for i in 1:n
        b=a
        a=copy(x)
    end
    return a===b,b===x,a===x
end
function swap_identity(x,n)
    a=x;b=copy(x)
    for i in 1:n
        a,b=b,a
    end
    return a===x,b===x,a===b
end
function nested_identity(x,n)
    a=x
    for i in 1:n
        for j in 1:i
            a=isodd(j) ? x : copy(x)
        end
    end
    return a===x
end
select_array(x,c)=c ? x : copy(x)
function cached_identity(x,n)
    c=isodd(n)
    a=Reactant.Ops.call(structured(select_array),x,c)
    b=Reactant.Ops.call(structured(select_array),x,c)
    return a===x,a===b
end
function cached_identity_host(x,n)
    a=select_array(x,isodd(n));b=select_array(x,isodd(n))
    return a===x,a===b
end
function cached_loop_identity(x,n)
    a=x;b=copy(x)
    for i in 1:n
        b=a
        a=Reactant.Ops.call(structured(select_array),x,isodd(i))
    end
    return a===x,b===x,a===b
end
function cached_loop_identity_host(x,n)
    a=x;b=copy(x)
    for i in 1:n
        b=a
        a=select_array(x,isodd(i))
    end
    return a===x,b===x,a===b
end
function views_identity(x,n)
    a=isodd(n) ? view(x,1:2) : view(x,1:2)
    return a===view(x,1:2),a===x
end
function view_or_parent_identity(x,n)
    a=isodd(n) ? view(x,:) : x
    return a===x,a===view(x,:)
end
function loop_view_identity(x,n)
    a=x
    for i in 1:n
        a=isodd(i) ? view(x,:) : x
    end
    return a===x,a===view(x,:)
end
function identity_loss(x,n)
    a=copy(x)
    for i in 1:n
        a=isodd(i) ? x : copy(x)
    end
    factor=a===x ? 2.0 : 3.0
    return factor*sum(a.^2)
end
gradient(f,x,n)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(n))[1]
concrete(x::Tuple)=map(concrete,x)
concrete(x::Union{Bool,Number})=x
concrete(x::Reactant.ConcreteRNumber{T}) where {T}=T(x)

Base.@noinline identity_helper(a,b)=a===b
function view_identity_across_call(x,n)
    a=isodd(n) ? view(x,:) : x
    return identity_helper(a,view(x,:))
end
function view_identity_across_cached_call(x,n)
    a=isodd(n) ? view(x,:) : x
    return Reactant.Ops.call(structured(identity_helper),a,view(x,:))
end

Base.@noinline array_kind(::SubArray)=1
Base.@noinline array_kind(::AbstractArray)=2
function selected_view_dispatch(x,n)
    a=Base.inferencebarrier(isodd(n) ? view(x,:) : x)
    return array_kind(a)
end

function index_type_choice(x,n)
    a=view(x,:)
    for i in 1:n
        a=isodd(i) ? view(x,:) : view(x,1:length(x))
    end
    return a===view(x,:),a===view(x,1:length(x))
end
function index_value_choice(x,n)
    a=view(x,1:2)
    for i in 1:n
        a=isodd(i) ? view(x,2:3) : view(x,1:2)
    end
    return a===view(x,1:2),a===view(x,2:3)
end
function repeated_equal_view(x,n)
    a=view(x,1:2);b=a
    for i in 1:n
        b=a
        a=view(x,1:2)
    end
    return a===b,a===view(x,1:2)
end
function retained_new_parent(x,n)
    a=view(copy(x),:);b=a
    for i in 1:n
        b=a
        a=view(copy(x),:)
    end
    return a===b,a===view(parent(a),:),b===view(parent(b),:),a===view(x,:)
end
function mixed_new_parent(x,n)
    a=x;b=view(x,:)
    for i in 1:n
        b=a
        a=isodd(i) ? view(copy(x),:) : copy(x)
    end
    return a===b,a===x,a===view(x,:)
end
function incoming_query(x,n)
    a=x;count=0
    for i in 1:n
        was_view=a===view(x,:)
        count+=was_view
        a=was_view ? x : view(x,:)
    end
    return a===view(x,:),count
end
Base.@noinline same_object(a,b)=a===b
function cached_same(a,b)
    if Reactant.ReactantCore.within_compile()
        return Reactant.Ops.call(structured(same_object),a,b)
    end
    return same_object(a,b)
end
function cached_queries(x,n)
    a=x;count=0
    for i in 1:n
        was_view=cached_same(a,view(x,:))
        count+=was_view
        a=was_view ? x : view(x,:)
    end
    return cached_same(a,view(x,:)),count
end
function nested_queries(x,n)
    a=x
    for i in 1:n
        for j in 1:i
            a=a===x ? view(x,:) : x
        end
    end
    return a===x,a===view(x,:)
end
function distinct_parents(x,n)
    y=copy(x);a=view(x,:)
    for i in 1:n
        a=isodd(i) ? view(x,:) : view(y,:)
    end
    return a===view(x,:),a===view(y,:)
end
function transpose_choice(x,n)
    a=transpose(x)
    for i in 1:n
        a=isodd(i) ? transpose(x) : transpose(copy(x))
    end
    return a===transpose(x),a===transpose(parent(a))
end
function loss_closed(x,n)
    a=x
    for i in 1:n
        a=a===x ? view(x,:) : x
        a .*=1.25
    end
    return (a===view(x,:) ? 2 : 3)*sum(a.^2)
end
function loss_cached(x,n)
    a=x
    for i in 1:n
        a=cached_same(a,x) ? view(x,:) : x
        a .*=1.25
    end
    return (cached_same(a,view(x,:)) ? 2 : 3)*sum(a.^2)
end

@testset "Observable identity across regions and calls" begin
    for (f,host) in ((retain_previous,retain_previous),(retain_original,retain_original),
                     (swap_identity,swap_identity),(nested_identity,nested_identity),
                     (cached_identity,cached_identity_host),(cached_loop_identity,cached_loop_identity_host),
                     (views_identity,views_identity),(view_or_parent_identity,view_or_parent_identity),
                     (loop_view_identity,loop_view_identity),
                     (view_identity_across_call,view_identity_across_call),
                     (view_identity_across_cached_call,view_identity_across_call),
                     (selected_view_dispatch,selected_view_dispatch))
        @testset "$f" begin
            x=[0.1,0.3,0.7,1.1]
            compiled=Reactant.compile(structured(f),(R(x),R(3)))
            for n in (0,1,2,3,4,6)
                @test concrete(compiled(R(copy(x)),R(n))) == host(copy(x),n)
            end
        end
    end
    @testset "Identity-dependent gradient: $policy" for policy in (:auto,Reactant.Periodic(2),Reactant.Binomial(3))
        f=structured(identity_loss;checkpointing=policy)
        x=[0.1,0.3,0.7,1.1]
        compiled=Reactant.compile(gradient,(f,R(x),R(3)))
        for n in (0,1,2,3,4,6)
            expected=2 .* x .* (isodd(n) ? 2 : 3)
            @test Array(compiled(f,R(copy(x)),R(n))) ≈ expected rtol=1e-7 atol=1e-9
        end
    end

    # Reuse each executable across zero, one, and several loop iterations.
    # Equal tensor contents do not imply equal Julia object identity.
    @testset "Loop wrapper structure: $f" for f in (
        index_type_choice, index_value_choice, repeated_equal_view,
        retained_new_parent, mixed_new_parent, incoming_query,
        cached_queries, nested_queries, distinct_parents, transpose_choice,
    )
        x=[0.1,0.3,0.7,1.1]
        compiled=Reactant.compile(structured(f),(R(copy(x)),R(3)))
        for n in (0,1,2,3,4,6)
            @test concrete(compiled(R(copy(x)),R(n))) == f(copy(x),n)
        end
    end
    @testset "Loop identity gradient: $f, $policy" for f in (loss_closed,loss_cached), policy in (:auto,Reactant.Periodic(2),Reactant.Binomial(3))
        x=[0.1,0.3,0.7,1.1]
        program=structured(f;checkpointing=policy)
        compiled=Reactant.compile(gradient,(program,R(copy(x)),R(3)))
        for n in (0,1,2,3,4,6)
            epsilon=1e-5
            expected=[(f(x .+ (eachindex(x).==i).*epsilon,n)-f(x .- (eachindex(x).==i).*epsilon,n))/(2epsilon) for i in eachindex(x)]
            @test Array(compiled(program,R(copy(x)),R(n))) ≈ expected rtol=1e-6 atol=1e-7
        end
    end
end
end # module
