"""
    structured(f)

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
emission leaves and behave exactly as they do today, so results agree with
Reactant's frontend wherever both apply. User functions passed to such leaves
(the function given to `Enzyme.gradient` or mapped over an array) are wrapped
automatically. Calls whose arguments carry no traced values run as ordinary
Julia.
"""
structured(f) = Program(f)

struct Program{F} <: Function
    f::F
end

structured(p::Program) = p

# The plain call runs `f` outside compilation. Reactant's interpreter also
# infers it when a leaf calls back into a `Program` (a mapped function, the
# body given to `Enzyme.autodiff`) and asks `return_type` of it for element
# types. Under Reactant's semantics `f` may "always throw" (a branch on a traced
# Bool), and Reactant would then treat the call as non-returning, so the body is
# hidden behind an inference barrier and asserted to the type this frontend
# infers for `f`, computed in the generator at the caller's world.
function program_call_generator(
    world::UInt, source, @nospecialize(P::Type), @nospecialize(argtypes::Tuple)
)
    @nospecialize
    rt = try
        CC._return_type(Interpreter(world), Tuple{P.parameters[1],argtypes...})
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

Base.show(io::IO, p::Program) = print(io, "structured(", p.f, ")")

# Reactant calls the function it compiles, and every callback it builds
# regions from, through `call_with_reactant`. These methods are the frontend's
# entry points: a `Program` is emitted from IR rather than executed. Code that
# Reactant rewrote itself prefixes the call with the return type it inferred.
# Julia infers these methods natively, on the program's own types, and would
# follow them into the emitter and type `f(args...)` there on traced values
# without Reactant's overlays: a nested `Enzyme.gradient` then compiles under
# native Enzyme and aborts. The barrier keeps that inference out of the emitter.
function Reactant.call_with_reactant(p::Program, args...)
    return scalar_indexing_preserved() do
        return Base.inferencebarrier(emit_method)(p.f, args, nothing)
    end
end

function Reactant.call_with_reactant(
    ::typeof(Core.kwcall), kwargs::NamedTuple, p::Program, args...
)
    return scalar_indexing_preserved() do
        return Base.inferencebarrier(emit_method)(
            Core.kwcall, (kwargs, p.f, args...), nothing
        )
    end
end

# `@allowscalar` restores the task's scalar-indexing state in a `finally` body,
# of which the emitter keeps only the normal path; an emission that fails
# inside one would leave the state set. Restore it here instead.
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
