# Explicit acceptance gate; intentionally separate from runtests.jl until the
# frontend preserves branch-selected object representations in all these cases.
module SelectedObjectAcceptance
using Reactant, StructuredReactant, Enzyme, LinearAlgebra, Test
Reactant.set_default_backend("cpu")

R(x)=Reactant.to_rarray(x;track_numbers=Number)
Base.@noinline array_kind(::SubArray)=1
Base.@noinline array_kind(::AbstractArray)=2
Base.@noinline tuple_kind(t)=array_kind(t[1])
Base.@noinline named_kind(t)=array_kind(t.value)
struct Holder{T};value::T;end
Base.@noinline holder_kind(::Holder{<:SubArray})=1
Base.@noinline holder_kind(::Holder)=2
function tuple_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    return tuple_kind(Base.inferencebarrier((a,)))
end
function named_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    return named_kind(Base.inferencebarrier((;value=a)))
end
function holder_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    return holder_kind(Base.inferencebarrier(Holder(a)))
end
function closure_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    f=()->array_kind(a)
    return Base.inferencebarrier(f)()
end
Base.@noinline wrapped_kind(::Transpose{<:Any,<:SubArray})=1
Base.@noinline wrapped_kind(::Transpose)=2
function wrapped_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    return wrapped_kind(Base.inferencebarrier(transpose(a)))
end
Base.@noinline type_kind(::Type{<:SubArray})=1
Base.@noinline type_kind(::Type{<:AbstractArray})=2
function selected_type(x,c)
    a=Base.inferencebarrier(c ? view(x,:) : x)
    T=Base.inferencebarrier(typeof(a))
    return type_kind(T)
end
function selected_indices(x,c)
    a=Base.inferencebarrier(c ? view(x,2:3) : view(x,1:2))
    inds=Base.inferencebarrier(parentindices(a))
    return first(inds[1])
end
function mutate_parent(x,c)
    a=Base.inferencebarrier(c ? view(x,2:3) : view(copy(x),1:2))
    p=parent(a)
    p .*=2
    return sum(a)+3sum(x)+(p===x ? 5 : 7)
end
Base.@noinline function mutate_helper(a,x)
    factor=array_kind(a)
    a .*=2
    x .+=1
    return factor*sum(a)+3sum(x)
end
function mutate_dispatch(x,c)
    a=Base.inferencebarrier(c ? view(x,2:3) : copy(x[1:2]))
    return mutate_helper(a,x)
end
Base.@noinline twice(x)=2sum(x)
Base.@noinline thrice(x)=3sum(x)
function selected_callable(x,c)
    f=Base.inferencebarrier(c ? twice : thrice)
    return f(x)
end
gradient(f,x,c)=Enzyme.gradient(Reverse,f,x,Enzyme.Const(c))[1]

@testset "Branch-selected Julia objects" begin
    for f in (tuple_dispatch,named_dispatch,holder_dispatch,closure_dispatch,
              wrapped_dispatch,selected_type,selected_indices,mutate_parent,
              mutate_dispatch,selected_callable)
        @testset "$f" begin
            x=[0.1,0.3,0.7,1.1]
            compiled=Reactant.compile(structured(f),(R(copy(x)),R(false)))
            for c in (false,true)
                input=copy(x)
                expected=f(input,c)
                traced_input=R(copy(x))
                actual=Float64(compiled(traced_input,R(c)))
                @test actual ≈ expected rtol=1e-7 atol=1e-9
                @test Array(traced_input) ≈ input rtol=1e-7 atol=1e-9
            end
        end
    end
    @testset "Mutation gradient: $f" for f in (mutate_parent,mutate_dispatch)
        x=[0.1,0.3,0.7,1.1]
        program=structured(f)
        compiled=Reactant.compile(gradient,(program,R(copy(x)),R(false)))
        for c in (false,true)
            epsilon=1e-5
            expected=[(f(x .+ (eachindex(x).==i).*epsilon,c)-
                       f(x .- (eachindex(x).==i).*epsilon,c))/(2epsilon)
                      for i in eachindex(x)]
            actual=Array(compiled(program,R(copy(x)),R(c)))
            @test actual ≈ expected rtol=1e-6 atol=1e-7
        end
    end
end
end # module
