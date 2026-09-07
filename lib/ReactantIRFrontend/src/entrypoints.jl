# Reactant's entry points, with the callee wrapped in `structured`.

"""
    @compile [option = value]... f(args...)
    @jit [option = value]... f(args...)
    @code_hlo [option = value]... f(args...)

Reactant's macros of the same names applied to `structured(f)`. Options, keyword
arguments and the broadcast form `f.(args...)` keep Reactant's meaning. They are
not exported, so that `using Reactant` stays unambiguous; import them explicitly
to make the frontend the default:

    using Reactant
    using ReactantIRFrontend: @compile, @jit, @code_hlo
"""
macro compile(args...)
    return forward(Symbol("@compile"), __source__, args)
end
macro jit(args...)
    return forward(Symbol("@jit"), __source__, args)
end
macro code_hlo(args...)
    return forward(Symbol("@code_hlo"), __source__, args)
end

function forward(name::Symbol, source::LineNumberNode, args)
    isempty(args) && error("expected a function call")
    call = Expr(
        :macrocall, GlobalRef(Reactant, name), source, Base.front(args)..., wrap(last(args))
    )
    return esc(call)
end

# `f(args...)` becomes `structured(f)(args...)`; `f.(args...)` keeps its dot.
function wrap(call)
    if Meta.isexpr(call, :call)
        return Expr(:call, Expr(:call, structured, call.args[1]), call.args[2:end]...)
    elseif Meta.isexpr(call, :., 2) && Meta.isexpr(call.args[2], :tuple)
        return Expr(:., Expr(:call, structured, call.args[1]), call.args[2])
    end
    return error("expected a function call, got $(call)")
end

"""
    compile(f, args::Tuple; kwargs...)
    code_hlo(f, args::Tuple; kwargs...)

The functions behind [`@compile`](@ref) and [`@code_hlo`](@ref): compile
`structured(f)` for `args`, or return its MLIR module as text. Keyword arguments
are Reactant's compile options, and `fn_kwargs` the keyword arguments of `f`.
"""
function compile(f, args::Tuple; kwargs...)
    return Reactant.Compiler.compile(structured(f), args; kwargs...)
end

function code_hlo(f, args::Tuple; kwargs...)
    return Reactant.MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        mod = Reactant.Compiler.code_hlo(ctx, structured(f), args; kwargs...)
        try
            Reactant.Compiler.TextualModule(mod)
        finally
            Reactant.MLIR.IR.dispose(mod)
        end
    end
end
