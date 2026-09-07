# The emitter: an interpreter over structured IR.
#
# Values are ordinary Julia values, traced or not. A statement is evaluated by
# resolving its operands and either running it (builtins, host-only calls),
# handing it to Reactant (leaf methods, intrinsics on traced numbers), or
# recursing into the callee's structured IR. Control-flow ops are emitted in
# regions.jl through Reactant's region builders.

# `for i in a:b` with a traced endpoint iterates a `TracedUnitRange`, whose
# `iterate` cannot decide termination on the host. The protocol is implemented
# symbolically instead (regions.jl): the iterator state is this counter,
# `iterate(r, state)` advances it, and the `=== nothing` test Julia lowers the
# loop with becomes a traced comparison against the range's end.
struct Iteration{I,S}
    i::I
    stop::S
end

# A traced range captured by a region travels as its endpoints (Reactant's
# tracer does not handle the range type) and is rebuilt on the other side.
struct RangeCapture{S,T}
    start::S
    stop::T
    length::Int
end

"""
    Frame

Emission state of one method activation: the values bound to arguments, SSA
values, and region block arguments. Regions fork the frame so that values
re-traced by Reactant's builders shadow the originals only inside the region.
"""
mutable struct Frame
    const code::Code
    const arguments::Vector{Any}   # Argument(n) => arguments[n]; arguments[1] is the callee
    const ssa::Vector{Any}         # SSAValue(id) => ssa[id]
    const blockargs::Vector{Any}   # BlockArgument(id) => blockargs[id]
    const parent::Union{Nothing,Frame}
    pc::Int                        # SSA index being emitted, for diagnostics
    exits::Bool                    # in a general loop body: continue/break yield (done, values...)
end

function Frame(code::Code, @nospecialize(f), args::Tuple, parent::Union{Nothing,Frame})
    sci = code.sci
    return Frame(
        code,
        pack_arguments(code.method, f, args),
        Vector{Any}(undef, sci.max_ssa_idx),
        Vector{Any}(undef, sci.max_arg_idx),
        parent,
        0,
        false,
    )
end

function fork(fr::Frame; exits::Bool=fr.exits)
    return Frame(
        fr.code,
        copy(fr.arguments),
        copy(fr.ssa),
        copy(fr.blockargs),
        fr.parent,
        fr.pc,
        exits,
    )
end

# A vararg method receives its trailing arguments as one tuple.
function pack_arguments(method::Method, @nospecialize(f), args::Tuple)
    method.isva || return Any[f, args...]
    fixed = Int(method.nargs) - 2
    return Any[f, args[1:fixed]..., args[(fixed + 1):end]]
end

operand(::Frame, @nospecialize(x)) = x
operand(fr::Frame, x::Core.SSAValue) = fr.ssa[x.id]
operand(fr::Frame, x::Core.Argument) = fr.arguments[x.n]
operand(fr::Frame, x::BlockArgument) = fr.blockargs[x.id]
operand(fr::Frame, x::Core.PiNode) = operand(fr, x.val)
operand(::Frame, x::QuoteNode) = x.value
operand(::Frame, x::GlobalRef) = getglobal(x.mod, x.name)
# An `undef` operand is never read on its path, so any value of its type
# serves; a number is materialized so that a slot undefined in both branches
# of a traced `if` does not need Reactant to fill it in.
function operand(::Frame, u::Undef)
    T = u.type
    T isa DataType && T <: Number && return zero(T)
    return MissingTracedValue()
end

operands(fr::Frame, xs) = Tuple(operand(fr, x) for x in xs)

bind!(fr::Frame, x::Core.SSAValue, @nospecialize(v)) = (fr.ssa[x.id] = v)
bind!(fr::Frame, x::Core.Argument, @nospecialize(v)) = (fr.arguments[x.n] = v)
bind!(fr::Frame, x::BlockArgument, @nospecialize(v)) = (fr.blockargs[x.id] = v)

function bind!(fr::Frame, keys, values)
    length(keys) == length(values) || unsupported(
        fr, "a region receiving $(length(values)) values for $(length(keys)) arguments"
    )
    for (key, value) in zip(keys, values)
        bind!(fr, key, uncapture(value))
    end
    return fr
end

uncapture(@nospecialize(v)) = v
function uncapture(v::RangeCapture)
    T = typeof(v.start)
    return Reactant.TracedUnitRange{T}(v.start, convert(T, v.stop), v.length)
end

unsupported(::Frame, what) = throw(FrontendError(string(what, " is not supported")))

"""
    has_traced(x) -> Bool

Whether `x` is or contains a traced value. Types, modules, and other
metadata are leaves; containers and structs are searched.
"""
function has_traced(@nospecialize(x), seen::Base.IdSet{Any}=Base.IdSet{Any}())
    x isa TracedType && return true
    x isa Union{Type,Module,Symbol,AbstractString,Core.MethodInstance,Method} &&
        return false
    T = typeof(x)
    isprimitivetype(T) && return false
    if x isa Array   # other array types (wrappers, ranges) are searched by field
        isbitstype(eltype(x)) && return false
        return any(i -> isassigned(x, i) && has_traced(x[i], seen), eachindex(x))
    end
    if ismutabletype(T)
        x in seen && return false
        push!(seen, x)
    end
    for i in 1:fieldcount(T)
        isdefined(x, i) && has_traced(getfield(x, i), seen) && return true
    end
    return false
end

function describe(fr::Frame)
    method = fr.code.method
    location = try
        scopes = IRStructurizer.source_location(fr.code.sci, fr.pc)
        isempty(scopes) ? "" : string(" at ", last(scopes).file, ":", last(scopes).line)
    catch
        ""
    end
    return string(method.name, " in ", method.module, location)
end

# How a block ends. Regions must yield; the function body must return.
struct Returned
    value::Any
end
struct Yielded
    values::Tuple
end
struct Conditioned
    condition::Any
    values::Tuple
end

"""
    Continuation

The rest of a block after an `if` that contains a `return`. StableHLO regions
cannot return from the enclosing function, so the statements after such an `if`
are emitted inside each branch that yields, and both branches then produce the
function result. `next` continues the enclosing block when the `if` is nested.
"""
struct Continuation
    block::Block
    position::Int
    next::Union{Nothing,Continuation}
end

function resume(k::Continuation, fr::Frame, yielded::Tuple)
    fr.ssa[k.block.body.ssa_idxes[k.position]] = yielded
    return emit_block(fr, k.block, k.position + 1, k.next)
end

function emit_block(
    fr::Frame, block::Block, from::Int=1, k::Union{Nothing,Continuation}=nothing
)
    body = block.body
    for position in from:length(body.ssa_idxes)
        idx = body.ssa_idxes[position]
        stmt = body.stmts[position]
        fr.pc = idx
        if stmt isa IfOp && (has_return(stmt) || (fr.exits && exits_loop(stmt)))
            return emit_if(fr, stmt, Continuation(block, position, k))
        end
        fr.ssa[idx] = emit_stmt(fr, stmt)
    end
    return emit_terminator(fr, block.terminator, k)
end

function emit_terminator(fr::Frame, t, k::Union{Nothing,Continuation})
    if t isa Core.ReturnNode
        isdefined(t, :val) || unsupported(fr, "code after a call that always throws")
        return Returned(operand(fr, t.val))
    elseif t isa YieldOp
        values = operands(fr, t.values)
        return k === nothing ? Yielded(values) : resume(k, fr, values)
    elseif t isa ContinueOp
        values = operands(fr, t.values)
        fr.exits && return Yielded((false, values...))
        return k === nothing ? Yielded(values) : resume(k, fr, values)
    elseif t isa BreakOp
        fr.exits || unsupported(fr, "`break` outside a general loop")
        return Yielded((true, operands(fr, t.values)...))
    elseif t isa ConditionOp
        return Conditioned(operand(fr, t.condition), operands(fr, t.args))
    end
    return unsupported(fr, "a block without a terminator")
end

has_return(op::ControlFlowOp) = any(has_return, blocks(op))
function has_return(block::Block)
    block.terminator isa Core.ReturnNode && return true
    return any(e -> e.stmt isa ControlFlowOp && has_return(e.stmt), values(block.body))
end

# Does a branch end the enclosing loop's iteration (`continue`/`break`) on some path?
exits_loop(op::IfOp) = any(exits_loop, blocks(op))
function exits_loop(block::Block)
    block.terminator isa Union{BreakOp,ContinueOp} && return true
    return any(e -> e.stmt isa IfOp && exits_loop(e.stmt), values(block.body))
end

emit_stmt(fr::Frame, @nospecialize(stmt)) = operand(fr, stmt)
emit_stmt(::Frame, ::Nothing) = nothing
emit_stmt(fr::Frame, op::IfOp) = emit_if(fr, op, nothing).values
emit_stmt(fr::Frame, op::Union{WhileOp,ForOp,LoopOp}) = emit_loop(fr, op)
emit_stmt(fr::Frame, ::Core.PhiNode) = unsupported(fr, "unstructured control flow")
function emit_stmt(fr::Frame, ::Union{Core.GotoNode,Core.GotoIfNot})
    return unsupported(fr, "unstructured control flow")
end
function emit_stmt(fr::Frame, ::Union{Core.PhiCNode,Core.UpsilonNode,Core.EnterNode})
    return unsupported(fr, "`try`/`catch`")
end
function emit_stmt(fr::Frame, ::Core.ReturnNode)
    return unsupported(fr, "code after a call that always throws")
end

const SILENT_EXPRESSIONS = (
    :meta,
    :inbounds,
    :loopinfo,
    :code_coverage_effect,
    :gc_preserve_begin,
    :gc_preserve_end,
    :aliasscope,
    :popaliasscope,
)

function emit_stmt(fr::Frame, ex::Expr)
    head = ex.head
    if head === :call
        return emit_call(fr, operand(fr, ex.args[1]), operands(fr, @view ex.args[2:end]))
    elseif head === :invoke
        return emit_call(fr, operand(fr, ex.args[2]), operands(fr, @view ex.args[3:end]))
    elseif head === :new
        fields = Any[operand(fr, a) for a in @view ex.args[2:end]]
        return construct(fr, operand(fr, ex.args[1]), fields)
    elseif head === :splatnew
        fields = Any[operand(fr, ex.args[2])...]
        return construct(fr, operand(fr, ex.args[1]), fields)
    elseif head === :static_parameter
        return fr.code.sparams[ex.args[1]::Int]
    elseif head === :boundscheck
        return true
    elseif head === :throw_undef_if_not
        defined = operand(fr, ex.args[2])
        defined === true && return nothing
        defined === false && throw(UndefVarError(ex.args[1]::Symbol))
        unsupported(
            fr, "reading `$(ex.args[1])`, which may be unassigned on the traced path"
        )
    elseif head in SILENT_EXPRESSIONS
        return nothing
    end
    return unsupported(fr, "the `$(head)` expression")
end

# `:new` bypasses constructors because inference already checked the field
# values. A value that became traced after inference (a loop carry or an
# induction variable) no longer matches the type inference chose, so the object
# is rebuilt through the type's constructor on the runtime values instead.
function construct(fr::Frame, @nospecialize(T), fields::Vector{Any})
    T isa DataType || unsupported(fr, "constructing a value of type $(T)")
    if all(i -> fields[i] isa fieldtype(T, i), eachindex(fields))
        return ccall(
            :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields, length(fields)
        )
    end
    any(has_traced, fields) || return ccall(
        :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields, length(fields)
    )
    T <: NamedTuple && return NamedTuple{fieldnames(T)}(Tuple(fields))
    wrapper = T.name.wrapper
    if isempty(methods(wrapper))   # a closure type has no constructor
        R = reparameterize(T, fields)
        return ccall(
            :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), R, fields, length(fields)
        )
    end
    return wrapper(fields...)
end

function reparameterize(T::DataType, fields::Vector{Any})
    body = Base.unwrap_unionall(T.name.wrapper)
    params = Any[T.parameters...]
    for i in 1:fieldcount(body)
        ft = fieldtype(body, i)
        ft isa TypeVar || continue
        j = findfirst(p -> p === ft, body.parameters)
        j === nothing || (params[j] = Core.Typeof(fields[i]))
    end
    return T.name.wrapper{params...}
end

"""
    emit_call(fr, f, args)

Emit a call site. Builtins run in Julia; intrinsics run in Julia unless an
operand is traced; a call whose arguments carry no traced values is ordinary
Julia; everything else is dispatched with `emit_method`.
"""
function emit_call(fr::Frame, @nospecialize(f), args::Tuple)
    if f isa Core.IntrinsicFunction   # intrinsics are builtins too; test them first
        return emit_intrinsic(fr, f, args)
    elseif f isa Core.Builtin
        return emit_builtin(fr, f, args)
    elseif has_traced(f) || any(has_traced, args)
        return emit_method(f, args, fr)
    end
    return f(args...)
end

"""
    emit_method(f, args, parent)

Dispatch `f(args...)` on the runtime types of its arguments. A leaf method is
called through `Reactant.call_with_reactant`, with any user function among its
arguments wrapped in `structured` so that callbacks the leaf invokes (the body
given to `Enzyme.autodiff`, the function mapped over an array) are captured
too; any other method is emitted from its structured IR in a new frame.
"""
function emit_method(@nospecialize(f), args::Tuple, parent::Union{Nothing,Frame})
    f === Base.iterate && iterates_traced_range(args) && return traced_iterate(args...)
    Reactant.should_rewrite_call(Core.Typeof(f)) ||
        return Reactant.call_with_reactant(f, map(structure_callback, args)...)
    sig = Tuple{Core.Typeof(f),map(Core.Typeof, args)...}
    resolution = resolve(sig, Base.get_world_counter())
    resolution === nothing && return f(args...)   # no method: Julia raises the MethodError
    code = resolution.code
    code === nothing &&
        return Reactant.call_with_reactant(f, map(structure_callback, args)...)

    fr = Frame(code, f, args, parent)
    ancestor = parent
    while ancestor !== nothing
        ancestor.code === code && throw(
            FrontendError("recursion on traced values is not supported", [describe(fr)])
        )
        ancestor = ancestor.parent
    end
    outcome = try
        emit_block(fr, code.sci.entry)
    catch err
        err isa FrontendError && push!(err.context, describe(fr))
        rethrow()
    end
    outcome isa Returned || unsupported(fr, "a function body that does not return")
    return outcome.value
end

# Base, Reactant, and Enzyme functions are left as they are: leaves compare
# them by identity (`op === Base.add_sum`).
function structure_callback(@nospecialize(x))
    x isa Function || return x
    x isa Program && return x
    root = Base.moduleroot(parentmodule(typeof(x)))
    (root in LEAF_ROOTS || isstdlib(root)) && return x
    return structured(x)
end

function emit_builtin(fr::Frame, @nospecialize(f), args::Tuple)
    if f === Core._apply_iterate
        flat = Any[]
        for iterable in args[3:end]
            splat!(fr, flat, iterable)
        end
        return emit_call(fr, args[2], Tuple(flat))
    elseif f === Core.ifelse && args[1] isa TracedRNumber{Bool}
        return Reactant.call_with_reactant(Base.ifelse, args...)
    elseif f === Core.getfield && length(args) >= 2 && args[1] isa Iteration
        return iteration_field(fr, args[1], args[2])
    elseif f === Core.:(===) && length(args) == 2 && any(a -> a isa Iteration, args)
        return iteration_done(fr, args[1], args[2])
    elseif f === Core.typeassert && args[1] isa TracedRNumber
        # A traced number stands in for its element type.
        Reactant.unwrapped_eltype(args[1]) <: args[2] && return args[1]
    elseif f === Core.:(===) && length(args) == 2 && any(a -> a isa TracedRNumber, args)
        all(a -> a isa Number, args) && return identical(fr, args[1], args[2])
    elseif f === Core.getfield && length(args) >= 2 && args[2] isa TracedRNumber
        unsupported(fr, "indexing a tuple or struct with a traced integer")
    end
    return f(args...)
end

# `===` on numbers is identity within a type. Structurization synthesizes it for
# integer branch discriminators, which become traced through a traced `if`; a
# traced number stands in for its element type, so this is `==` for integers of
# one type and `false` across types. Floating-point identity is bitwise and has
# no operation here.
function identical(fr::Frame, a::Number, b::Number)
    element(x) = x isa TracedRNumber ? Reactant.unwrapped_eltype(x) : typeof(x)
    element(a) === element(b) || return false
    element(a) <: Integer && return Reactant.call_with_reactant(==, a, b)
    return unsupported(fr, "`===` on traced floating-point values")
end

function splat!(::Frame, flat, x::Union{Tuple,NamedTuple,AbstractArray,Core.SimpleVector})
    return append!(flat, x)
end
splat!(fr::Frame, flat, ::TracedRArray) = unsupported(fr, "splatting a traced array")
function splat!(::Frame, flat, x)
    for v in x
        push!(flat, v)
    end
    return flat
end
