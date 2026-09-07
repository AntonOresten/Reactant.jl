# Control flow through Reactant's region builders.
#
# `Ops.if_condition` and `Ops.while_loop` build their regions by calling back
# into Julia with re-traced copies of the values the region needs. A region
# callback here forks the enclosing frame, binds those copies over the
# originals, and emits the region's block. Everything Reactant does with the
# results (zero-filling missing values, tracking mutated arguments, typing the
# yields) is reused unchanged.

# Values a region reads from its surroundings: operands used inside it that it
# does not define. A continuation adds the rest of the enclosing blocks, which
# are emitted inside the region too.
function captures(op::ControlFlowOp, extra::Vector{Any}, k::Union{Nothing,Continuation})
    defined = Set{Any}()
    used = copy(extra)
    op isa ForOp && push!(defined, op.iv_arg)   # the induction variable is not a block arg
    for region in blocks(op)
        collect_definitions!(defined, region)
        collect_uses!(used, region)
    end
    while k !== nothing
        body = k.block.body
        push!(defined, Core.SSAValue(body.ssa_idxes[k.position]))
        for position in (k.position + 1):length(body.ssa_idxes)
            stmt = body.stmts[position]
            push!(defined, Core.SSAValue(body.ssa_idxes[position]))
            stmt isa ControlFlowOp &&
                foreach(b -> collect_definitions!(defined, b), blocks(stmt))
            collect_uses!(used, stmt)
        end
        collect_uses!(used, k.block.terminator)
        k = k.next
    end
    free = Any[]
    for v in used
        (v isa Core.SSAValue || v isa Core.Argument || v isa BlockArgument) || continue
        v in defined && continue
        v in free && continue
        push!(free, v)
    end
    return free
end

function collect_definitions!(defined, block::Block)
    for b in eachblock(block)
        union!(defined, b.args)
        for (idx, entry) in b.body
            push!(defined, Core.SSAValue(idx))
            entry.stmt isa ForOp && push!(defined, entry.stmt.iv_arg)
        end
    end
    return defined
end

function collect_uses!(used, block::Block)
    for (_, entry) in block.body
        collect_uses!(used, entry.stmt)
    end
    return collect_uses!(used, block.terminator)
end
function collect_uses!(used, ex::Expr)
    return append!(used, @view ex.args[(ex.head === :invoke ? 2 : 1):end])
end
function collect_uses!(used, op::IfOp)
    push!(used, op.condition)
    return foreach(b -> collect_uses!(used, b), blocks(op))
end
function collect_uses!(used, op::ForOp)
    push!(used, op.lower, op.upper, op.step)
    append!(used, op.init_values)
    return collect_uses!(used, op.body)
end
function collect_uses!(used, op::Union{WhileOp,LoopOp})
    append!(used, op.init_values)
    return foreach(b -> collect_uses!(used, b), blocks(op))
end
collect_uses!(used, t::Union{YieldOp,ContinueOp,BreakOp}) = append!(used, t.values)
collect_uses!(used, t::ConditionOp) = (push!(used, t.condition); append!(used, t.args))
collect_uses!(used, t::Core.ReturnNode) = isdefined(t, :val) ? push!(used, t.val) : used
collect_uses!(used, p::Core.PiNode) = push!(used, p.val)
collect_uses!(used, ::Nothing) = used
collect_uses!(used, @nospecialize(x)) = push!(used, x)

# Only traced values go through the builders. Host values are read from the
# forked frame, exactly as Julia would read them from the enclosing scope.
function traced_captures(fr::Frame, keys::Vector{Any})
    traced = Any[]
    values = Any[]
    for key in keys
        value = operand(fr, key)
        has_traced(value) || continue
        value isa Reactant.TracedUnitRange &&
            (value = RangeCapture(value.start, value.stop, value.length))
        push!(traced, key)
        push!(values, value)
    end
    return traced, Tuple(values)
end

#-----------------------------------------------------------------------------
# if
#-----------------------------------------------------------------------------

struct Region
    frame::Frame
    block::Block
    keys::Vector{Any}
    continuation::Union{Nothing,Continuation}
    result_type::Any               # the `if`'s result tuple type, from inference
    other::Base.RefValue{Any}      # what the branch emitted first yielded
end

Base.show(io::IO, ::Region) = print(io, "region")

function Reactant.call_with_reactant(r::Region, values...)
    fr = fork(r.frame)
    bind!(fr, r.keys, values)
    outcome = emit_block(fr, r.block, 1, r.continuation)
    result = if outcome isa Yielded
        outcome.values
    elseif outcome isa Returned
        (outcome.value,)   # the builder expects a tuple
    else
        unsupported(fr, "a loop condition inside a branch")
    end
    if r.other[] === nothing
        r.other[] = result
        return result
    end
    return fill_dead_slots(result, r.other[], r.result_type)
end

# Structurization multiplexes several exit paths through one `if`, forwarding
# `undef` in the slots a path does not define. A slot undefined in one branch
# is filled by Reactant from the other branch's value; a slot undefined in both
# is dead on this path but Reactant cannot type it, so the second branch
# yields another slot of the same static type in its place.
function fill_dead_slots(values::Tuple, other::Tuple, @nospecialize(T))
    T isa DataType && T <: Tuple && length(T.parameters) == length(values) || return values
    missing(k) = values[k] isa MissingTracedValue && other[k] isa MissingTracedValue
    any(missing, eachindex(values)) || return values
    return ntuple(length(values)) do k
        missing(k) || return values[k]
        j = findfirst(eachindex(values)) do j
            return T.parameters[j] === T.parameters[k] &&
                   !(values[j] isa MissingTracedValue)
        end
        return j === nothing ? values[k] : values[j]
    end
end

function emit_if(fr::Frame, op::IfOp, k::Union{Nothing,Continuation})
    condition = operand(fr, op.condition)
    if condition isa Bool
        return emit_block(fr, condition ? op.then_region : op.else_region, 1, k)
    end
    condition isa TracedRNumber{Bool} ||
        unsupported(fr, "branching on a value of type $(typeof(condition))")
    for region in blocks(op)
        throws(region) && unsupported(
            fr,
            "a branch on a traced condition that always throws (an `error` there, " *
            "or a variable Julia could not prove assigned, such as a reassigned capture)",
        )
    end
    keys, values = traced_captures(fr, captures(op, Any[], k))
    result_type =
        k === nothing ? IRStructurizer.def(fr.code.sci, Core.SSAValue(fr.pc))[:type] : Any
    other = Ref{Any}(nothing)
    then_region = Region(fr, op.then_region, keys, k, result_type, other)
    else_region = Region(fr, op.else_region, keys, k, result_type, other)
    result = Ops.if_condition(
        condition, then_region, else_region, values...; track_numbers=Number
    )
    k === nothing && return Yielded(Tuple(result))
    # With a continuation both branches produced the enclosing block's outcome:
    # the function result, or, in a general loop body, (done, values...).
    return fr.exits ? Yielded(Tuple(result)) : Returned(only(result))
end

function throws(block::Block)
    return block.terminator isa Core.ReturnNode && !isdefined(block.terminator, :val)
end

#-----------------------------------------------------------------------------
# while and for
#-----------------------------------------------------------------------------

# Reactant's loop builder communicates through mutation: the body callback
# writes the next iteration's values into the carries it received, and the
# builder writes the loop results into the carries it was given. Carries are
# therefore fresh traced copies, so SSA values before the loop stay intact.
#
# The builder only accepts singleton callbacks (it wraps anything with fields
# in `apply` and re-traces it), so the state a callback needs is dynamically
# scoped over the builder call instead of stored in the callback.
struct Loop
    frame::Frame
    op::Union{WhileOp,ForOp,LoopOp}
    keys::Vector{Any}
end

const CURRENT_LOOP = Base.ScopedValues.ScopedValue{Union{Nothing,Loop}}(nothing)

struct LoopCondition end
struct LoopBody end

# A loop runs here, at emission, for as long as its condition and the values it
# carries are host values, exactly as it would in Julia; its body may still emit
# operations on traced invariants. From the first iteration at which the
# condition or a carried value is traced, the remainder becomes a
# `stablehlo.while`. This also covers a carry without an initial value (Julia
# dropped a dead initializer such as `v = similar(x)` before a loop that always
# assigns `v`): the first host iteration supplies it.
function emit_loop(fr::Frame, op::ForOp)
    has_return(op) && unsupported(fr, "`return` inside a loop")
    carries = operands(fr, op.init_values)
    lower, upper, step = operand(fr, op.lower), operand(fr, op.upper), operand(fr, op.step)
    if lower isa Integer && upper isa Integer && step isa Integer
        step > 0 || unsupported(fr, "a counted loop with a non-positive step")
        while lower < upper && (unassigned(carries) || !any(has_traced, carries))
            bind!(fr, op.iv_arg, lower)
            carries = iteration(fr, op.body, carries)
            lower += step
        end
        lower < upper || return carries
    end
    unassigned(carries) &&
        unsupported(fr, "reading a value after a loop that is only assigned inside it")
    return roll(fr, op, (carry(fr, lower), map(v -> carry(fr, v), carries)...))
end

function emit_loop(fr::Frame, op::WhileOp)
    has_return(op) && unsupported(fr, "`return` inside a loop")
    carries = operands(fr, op.init_values)
    while true
        bind!(fr, op.before.args, carries)
        outcome = emit_block(fr, op.before)
        outcome isa Conditioned || unsupported(fr, "a loop condition that does not yield")
        outcome.condition isa Bool && !any(has_traced, outcome.values) || break
        outcome.condition || return outcome.values
        carries = iteration(fr, op.after, outcome.values)
    end
    unassigned(carries) &&
        unsupported(fr, "reading a value after a loop that is only assigned inside it")
    return roll(fr, op, map(v -> carry(fr, v), carries))
end

unassigned(carries::Tuple) = any(v -> v isa MissingTracedValue, carries)

function iteration(fr::Frame, body::Block, carries::Tuple)
    bind!(fr, body.args, carries)
    outcome = emit_block(fork(fr; exits=false), body)
    outcome isa Yielded || unsupported(fr, "a loop body that does not continue")
    return outcome.values
end

# A general loop exits only through `break`, so its body runs at least once; the
# first iteration also supplies carries without initial values. Iterations stay
# on the host while the exit decision and the carried values are host values;
# from the first traced decision on, the remainder is a `stablehlo.while` whose
# first carry is the `done` flag.
function emit_loop(fr::Frame, op::LoopOp)
    carries = operands(fr, op.init_values)
    local done
    while true
        done, carries... = iteration_of_general_loop(fr, op.body, carries)
        done isa Bool || break
        done && return carries
        any(has_traced, carries) && break
    end
    # The last decision may already be traced: the rolled loop starts from it.
    return roll(fr, op, (carry(fr, done), map(v -> carry(fr, v), carries)...))
end

function iteration_of_general_loop(fr::Frame, body::Block, carries::Tuple)
    bind!(fr, body.args, carries)
    outcome = emit_block(fork(fr; exits=true), body)
    outcome isa Yielded ||
        unsupported(fr, "a general loop body that neither continues nor breaks")
    return outcome.values
end

# Build the `stablehlo.while` for the remaining iterations. For a `for` loop the
# first carry is the induction variable.
function roll(fr::Frame, op::Union{WhileOp,ForOp,LoopOp}, carries::Tuple)
    extra = op isa ForOp ? Any[op.upper, op.step] : Any[]
    keys, invariants = traced_captures(fr, captures(op, extra, nothing))
    Base.ScopedValues.@with CURRENT_LOOP => Loop(fr, op, keys) begin
        Ops.while_loop(
            LoopCondition(), LoopBody(), carries, invariants; track_numbers=Number
        )
    end

    op isa ForOp && return Base.tail(carries)   # drop the induction variable
    op isa LoopOp && return Base.tail(carries)  # drop the done flag
    # A while loop's results are the values its condition region forwards on exit.
    exit = fork(fr)
    bind!(exit, op.before.args, carries)
    return (emit_block(exit, op.before)::Conditioned).values
end

function Reactant.call_with_reactant(::LoopCondition, carries, invariants)
    return emit_loop_region(CURRENT_LOOP[], :condition, carries, invariants)
end
function Reactant.call_with_reactant(::LoopBody, carries, invariants)
    return emit_loop_region(CURRENT_LOOP[], :body, carries, invariants)
end

function emit_loop_region(loop::Loop, role::Symbol, carries, invariants)
    # A loop rolled inside a general loop's body must not inherit that body's
    # `exits`: its own `continue` yields carries, not a `done` flag.
    fr = fork(loop.frame; exits=false)
    bind!(fr, loop.keys, invariants)
    op = loop.op
    if op isa WhileOp
        bind!(fr, op.before.args, carries)
        outcome = emit_block(fr, op.before)
        outcome isa Conditioned || unsupported(fr, "a loop condition that does not yield")
        if role === :condition
            outcome.condition isa TracedRNumber{Bool} ||
                unsupported(fr, "a loop condition of type $(typeof(outcome.condition))")
            return outcome.condition
        end
        bind!(fr, op.after.args, outcome.values)
        next = (emit_block(fr, op.after)::Yielded).values
        update!(fr, carries, next)
    elseif op isa LoopOp
        done, rest = carries[1], Base.tail(carries)
        role === :condition && return traced(!)(done)
        update!(fr, carries, iteration_of_general_loop(fr, op.body, rest))
    else
        iv, rest = carries[1], Base.tail(carries)
        role === :condition && return traced(<)(iv, operand(fr, op.upper))
        bind!(fr, op.iv_arg, iv)
        bind!(fr, op.body.args, rest)
        next = (emit_block(fr, op.body)::Yielded).values
        update!(fr, carries, (traced(+)(iv, operand(fr, op.step)), next...))
    end
    return nothing
end

function carry(::Frame, x::TracedRArray{T,N}) where {T,N}
    return TracedRArray{T,N}((), x.mlir_data, size(x))
end
carry(::Frame, x::TracedRNumber{T}) where {T} = TracedRNumber{T}((), x.mlir_data)
carry(::Frame, x::Number) = Ops.constant(x)
carry(fr::Frame, x::Union{Tuple,NamedTuple}) = map(v -> carry(fr, v), x)
carry(fr::Frame, x::Iteration) = Iteration(carry(fr, x.i), carry(fr, x.stop))
function carry(fr::Frame, x::MissingTracedValue)
    return unsupported(fr, "reading a value after a loop that is only assigned inside it")
end
carry(fr::Frame, x) = unsupported(fr, "a loop-carried value of type $(typeof(x))")

function update!(fr::Frame, carries::Union{Tuple,NamedTuple}, next)
    length(carries) == length(next) ||
        unsupported(fr, "a loop that changes how many values it carries")
    for (c, n) in zip(carries, next)
        update!(fr, c, n)
    end
    return nothing
end
function update!(fr::Frame, c::Iteration, next)
    next isa Iteration || unsupported(fr, "a loop-carried range iterator that changes type")
    update!(fr, c.i, next.i)
    return update!(fr, c.stop, next.stop)
end
function update!(fr::Frame, c::TracedType, @nospecialize(next))
    next isa Number && (next = Ops.constant(next))
    next isa TracedType || unsupported(
        fr, "a loop-carried value that changes from $(typeof(c)) to $(typeof(next))"
    )
    Reactant.MLIR.IR.type(c.mlir_data) == Reactant.MLIR.IR.type(next.mlir_data) ||
        unsupported(fr, "a loop-carried value that changes type or shape")
    Reactant.TracedUtils.set_mlir_data!(c, next.mlir_data)
    return nothing
end

#-----------------------------------------------------------------------------
# iterating a traced range
#-----------------------------------------------------------------------------

function iterates_traced_range(args::Tuple)
    length(args) in (1, 2) || return false
    args[1] isa Reactant.TracedUnitRange || return false
    return length(args) == 1 || args[2] isa Iteration
end

traced_iterate(r::Reactant.TracedUnitRange) = Iteration(r.start, r.stop)
function traced_iterate(::Reactant.TracedUnitRange, it::Iteration)
    return Iteration(traced(+)(it.i, 1), it.stop)
end

function iteration_field(fr::Frame, it::Iteration, field)
    field == 1 && return it.i     # the element
    field == 2 && return it       # the state
    return unsupported(fr, "field $(field) of a range iterator")
end

function iteration_done(fr::Frame, a, b)
    it, other = a isa Iteration ? (a, b) : (b, a)
    other === nothing || return false
    return traced(>)(it.i, it.stop)
end

# Reactant's builders trace the values a region receives; the counter and the
# captured range are traced field by field like any immutable struct.
function Reactant.make_tracer(seen, it::Iteration, @nospecialize(path), mode; kwargs...)
    subpath(k) = mode == Reactant.TracedToTypes ? path : Reactant.append_path(path, k)
    i = Reactant.make_tracer(seen, it.i, subpath(1), mode; kwargs...)
    stop = Reactant.make_tracer(seen, it.stop, subpath(2), mode; kwargs...)
    return Iteration(i, stop)
end
function Reactant.make_tracer(seen, r::RangeCapture, @nospecialize(path), mode; kwargs...)
    subpath(k) = mode == Reactant.TracedToTypes ? path : Reactant.append_path(path, k)
    start = Reactant.make_tracer(seen, r.start, subpath(1), mode; kwargs...)
    stop = Reactant.make_tracer(seen, r.stop, subpath(2), mode; kwargs...)
    return RangeCapture(start, stop, r.length)
end
