const Checkpointing = Union{Nothing,Bool,Symbol,ReactantCore.Periodic,ReactantCore.Binomial}
const CURRENT_CHECKPOINTING = Base.ScopedValues.ScopedValue{Checkpointing}(:auto)

function validate_checkpointing(strategy)
    strategy === nothing ||
        strategy isa Bool ||
        strategy === :auto ||
        strategy isa Union{ReactantCore.Periodic,ReactantCore.Binomial} ||
        throw(
            ArgumentError(
                "checkpointing must be :auto, a Bool, Periodic(n), or Binomial(n)"
            ),
        )
    budget = if strategy isa ReactantCore.Periodic
        strategy.n
    elseif strategy isa ReactantCore.Binomial
        strategy.budget
    else
        1
    end
    budget > 0 || throw(ArgumentError("checkpointing budget must be positive"))
    strategy isa ReactantCore.Binomial &&
        budget < 2 &&
        throw(ArgumentError("binomial checkpointing requires at least two checkpoints"))
    return strategy
end

"""
    structured(f; checkpointing=:auto)

Wrap `f` so that Reactant captures its control flow from inferred structured IR
instead of executing it with traced values. `if`, `while`, and `for` over traced
values become `stablehlo.if` and `stablehlo.while` without `@trace`, in `f` and
in every function it calls. Outside compilation the wrapper simply calls `f`.

Use it wherever Reactant accepts a callable:

```julia
step(x) = sum(x) > 0 ? x .* 2 : x ./ 2
compiled = @compile structured(step)(x)
@jit structured(step)(x)
@code_hlo structured(step)(x)
Enzyme.gradient(Reverse, structured(loss), x)   # inside a compiled function
```

Methods that Reactant defines, and Base methods applied to traced values, are
emission leaves and behave exactly as they do today. User functions passed to
such leaves (the function given to `Enzyme.gradient` or mapped over an array)
are wrapped automatically.

`checkpointing=:auto` enables `Reactant.Binomial(4)` for counted loops whose
trip count is traced; statically counted loops keep their existing caching.
Use `false` to disable checkpointing, `true` to select periodic checkpointing
for static counts and binomial for runtime counts, or a `Reactant.Periodic(n)`
or `Reactant.Binomial(n)` strategy. Options apply to nested structured calls;
an explicitly configured nested wrapper overrides the enclosing choice.
The native backend currently requires a canonical counted loop for checkpointing;
this option does not yet enable reverse mode through arbitrary conditions.
"""
structured(f; checkpointing=nothing) = Program(f, validate_checkpointing(checkpointing))

# A tracing option is static metadata, never an extra device argument.
struct Program{F,C} <: Function
    f::F
end
Program(f, checkpointing) = Program{typeof(f),checkpointing}(f)
checkpointing(::Program{F,C}) where {F,C} = C

function structured(p::Program; checkpointing=nothing)
    checkpointing === nothing && return p
    return Program(p.f, validate_checkpointing(checkpointing))
end

function with_checkpointing(f, p::Program)
    option = checkpointing(p)
    strategy = option === nothing ? CURRENT_CHECKPOINTING[] : option
    return Base.ScopedValues.@with CURRENT_CHECKPOINTING => strategy f()
end

# Runtime trip counts need bounded storage in the reverse pass. Use Reactant's
# checkpointing implementation; static loops keep their existing default.
function loop_checkpointing(count=nothing)
    strategy = CURRENT_CHECKPOINTING[]
    if strategy === :auto
        return count isa TracedRNumber ? ReactantCore.Binomial(4) : false
    elseif strategy === true
        return if count isa Integer
            ReactantCore.Periodic(max(1, isqrt(count)))
        else
            ReactantCore.Binomial(4)
        end
    end
    return strategy
end

# The plain call runs `f` outside compilation. Reactant's interpreter types it
# for callbacks, under which `f` may "always throw" (a branch on a traced Bool),
# so the body is hidden behind a barrier and asserted to this frontend's type.
function program_call_generator(
    world::UInt, source, @nospecialize(P::Type), @nospecialize(argtypes::Tuple)
)
    @nospecialize
    rt = try
        Base.invoke_in_world(
            COMPILER_WORLD[],
            inferred_return_type,
            Interpreter(world),
            Tuple{P.parameters[1],argtypes...},
        )
    catch
        Any
    end
    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(Symbol("#self#"), :args), Core.svec()
    )
    body = :(Base.inferencebarrier(getfield(var"#self#", :f))(args...)::$(rt))
    return stub(world, source, body)
end

@eval function (p::Program)(args...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, program_call_generator))
end

function Core.kwcall(kwargs::NamedTuple, p::Program, args...)
    return Base.inferencebarrier(p.f)(args...; kwargs...)
end

function Base.show(io::IO, p::Program)
    print(io, "structured(", p.f)
    checkpointing(p) === nothing || print(io, "; checkpointing=", checkpointing(p))
    return print(io, ")")
end

# Reactant reaches every call through `call_with_reactant`: a `Program` is
# emitted from IR instead of executed. The barrier keeps native inference of
# these methods out of the emitter, where it would type `f` without Reactant's
# overlays (a nested `Enzyme.gradient` then aborts under native Enzyme).
function Reactant.call_with_reactant(p::Program, args...)
    return with_checkpointing(p) do
        return scalar_indexing_preserved() do
            return Base.inferencebarrier(emit_method)(p.f, args, nothing)
        end
    end
end

function Reactant.call_with_reactant(
    ::typeof(Core.kwcall), kwargs::NamedTuple, p::Program, args...
)
    return with_checkpointing(p) do
        return scalar_indexing_preserved() do
            return Base.inferencebarrier(emit_method)(
                Core.kwcall, (kwargs, p.f, args...), nothing
            )
        end
    end
end

# `@allowscalar` restores the scalar-indexing state in a `finally` body, of which
# only the normal path is emitted; a failed emission would leave it set.
function scalar_indexing_preserved(f)
    tls = task_local_storage()
    saved = get(tls, :ScalarIndexing, nothing)
    try
        return f()
    finally
        saved === nothing ? delete!(tls, :ScalarIndexing) : (tls[:ScalarIndexing] = saved)
    end
end

function Reactant.call_with_reactant(::Reactant.EnsureReturnType, p::Program, args...)
    return Reactant.call_with_reactant(p, args...)
end

function Reactant.call_with_reactant(
    ::Reactant.EnsureReturnType,
    kwcall::typeof(Core.kwcall),
    kwargs::NamedTuple,
    p::Program,
    args...,
)
    return Reactant.call_with_reactant(kwcall, kwargs, p, args...)
end
