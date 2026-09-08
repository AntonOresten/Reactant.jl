# The structured IR of one method specialization, cached per call signature.
struct Code
    sci::StructuredIRCode
    method::Method
    sparams::Core.SimpleVector
    valid_worlds::CC.WorldRange
    # Memoized per control-flow op: does it return, does it leave the loop.
    returns::IdDict{Any,Bool}
    exits::IdDict{Any,Bool}
    blocks::IdDict{Block,PreparedBlock}
end
function Code(sci, method, sparams, valid_worlds)
    returns, exits = IdDict{Any,Bool}(), IdDict{Any,Bool}()
    return Code(
        sci, method, sparams, valid_worlds, returns, exits, prepare(sci, returns, exits)
    )
end

# What a call signature dispatches to. `code === nothing` marks a leaf.
struct Resolution
    code::Union{Nothing,Code}
    valid_worlds::CC.WorldRange
end

# An `IdDict`: its methods do not specialize on the key, a `Dict{Type,...}`'s do,
# once per signature.
const RESOLUTIONS = IdDict{Any,Resolution}()
const RESOLUTIONS_LOCK = ReentrantLock()

function covers(r::Resolution, world::UInt)
    return r.valid_worlds.min_world <= world <= r.valid_worlds.max_world
end

function worlds_intersection(a::CC.WorldRange, b::CC.WorldRange)
    return CC.WorldRange(max(a.min_world, b.min_world), min(a.max_world, b.max_world))
end

# What `sig` dispatches to in `world`: `nothing` without a method, a leaf, or
# the method's structured IR; cached until a method definition invalidates it.
function resolve(@nospecialize(sig::Type), world::UInt)
    # Not `lock(f, l)`: a closure over `sig` has one type per signature and would
    # have Julia compile `lock` once for each.
    lock(RESOLUTIONS_LOCK)
    try
        return resolve_locked(sig, world)
    finally
        unlock(RESOLUTIONS_LOCK)
    end
end

function resolve_locked(@nospecialize(sig::Type), world::UInt)
    cached = get(RESOLUTIONS, sig, nothing)
    cached !== nothing && covers(cached, world) && return cached

    interp = Interpreter(world)
    match, valid_worlds = CC.findsup(sig, CC.method_table(interp))
    match === nothing && return nothing

    resolution = if isleaf(match.method, sig)
        Resolution(nothing, valid_worlds)
    else
        mi = specialization(interp, match, sig)
        ir, inferred_worlds = infer(interp, mi)
        ir === nothing && throw(FrontendError("inference of $(sig) failed"))
        ir = strip_exception_handling!(ir)
        STRUCTURIZER_HOISTS || dedup_getfield!(ir)
        sci = try
            StructuredIRCode(ir)
        catch err
            message = sprint(showerror, err)
            throw(
                FrontendError(
                    "the control flow of $(match.method) could not be structured: " *
                    message,
                ),
            )
        end
        valid_worlds = worlds_intersection(valid_worlds, inferred_worlds)
        Resolution(Code(sci, match.method, mi.sparam_vals, valid_worlds), valid_worlds)
    end
    RESOLUTIONS[sig] = resolution
    return resolution
end

# Julia 1.12 records the valid worlds on the `IRCode`; 1.11 only on the frame,
# so there the optimizer is run by hand, as `typeinf_ircode` does.
@static if VERSION >= v"1.12-"
    function infer(interp::Interpreter, mi::Core.MethodInstance)
        ir, _ = CC.typeinf_ircode(interp, mi, nothing)
        return ir, ir === nothing ? nothing : ir.valid_worlds
    end
else
    function infer(interp::Interpreter, mi::Core.MethodInstance)
        frame = CC.typeinf_frame(interp, mi, false)
        frame === nothing && return nothing, nothing
        opt = CC.OptimizationState(frame, interp)
        ir = CC.run_passes_ipo_safe(opt.src, opt, frame.result, nothing)
        return ir, frame.valid_worlds
    end
end

# Traced Bool arguments are admitted as host Bools, as leaf results are; the
# concrete signature stays the cache key.
function specialization(interp::Interpreter, match::Core.MethodMatch, @nospecialize(sig))
    widened = Tuple{map(widen_traced_bool, sig.parameters)...}
    widened === sig && return CC.specialize_method(match)
    rematch, _ = CC.findsup(widened, CC.method_table(interp))
    rematch !== nothing &&
        rematch.method === match.method &&
        return CC.specialize_method(rematch)
    return CC.specialize_method(match)
end

function widen_traced_bool(@nospecialize(T))
    return T === TracedRNumber{Bool} ? Union{Bool,TracedRNumber{Bool}} : T
end

# Handlers are dropped: nothing throws in the compiled program, and an exception
# while tracing aborts the compile. The normal path keeps its `finally` copy.
function strip_exception_handling!(ir::CC.IRCode)
    found = false
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Core.EnterNode
            found = true
            CC.kill_edge!(ir, CC.block_for_inst(ir.cfg, i), stmt.catch_dest)
            setstmt!(ir, i, nothing)
        elseif stmt isa Core.UpsilonNode ||
            (stmt isa Expr && stmt.head in (:leave, :pop_exception))
            setstmt!(ir, i, nothing)
        end
    end
    found || return ir
    return fold_trivial_phis!(decide_literal_branches!(CC.compact!(ir, true)))
end

# Julia 1.11 lowers `finally` to one body dispatching on a state value; with the
# handler gone the rethrow branch compares two literals, and left undecided it
# would become a throwing exit carried out of an enclosing loop.
function decide_literal_branches!(ir::CC.IRCode)
    decided = false
    for (b, block) in enumerate(ir.cfg.blocks)
        i = last(block.stmts)
        t = ir.stmts[i][:stmt]
        t isa Core.GotoIfNot || continue
        taken = literal_condition(ir, t.cond)
        taken === nothing && continue
        if taken
            setstmt!(ir, i, nothing)
            CC.kill_edge!(ir, b, t.dest)
        else
            setstmt!(ir, i, Core.GotoNode(t.dest))
            CC.kill_edge!(ir, b, b + 1)
        end
        decided = true
    end
    return decided ? CC.compact!(ir, true) : ir
end

function literal_condition(ir::CC.IRCode, @nospecialize(cond))
    cond isa Bool && return cond
    cond isa Core.SSAValue || return nothing
    stmt = ir.stmts[cond.id][:stmt]
    is_call(stmt, ===, 2) || return nothing
    a, b = literal(stmt.args[2]), literal(stmt.args[3])
    (a === nothing || b === nothing) && return nothing
    return a[] === b[]
end

# A literal operand, boxed so that `nothing` itself can be one.
function literal(@nospecialize(x))
    x isa QuoteNode && (x = x.value)
    x isa Union{Core.SSAValue,Core.Argument,GlobalRef,Expr,Core.PhiNode} && return nothing
    return Ref{Any}(x)
end

# Julia 1.11 saves every variable a `try` reads in a slot, which leaves loop phis
# merging a value with itself once the catch edge is gone; such a phi would be a
# loop carry that rolls the loop from its first iteration.
function fold_trivial_phis!(ir::CC.IRCode)
    rename = Dict{Int,Any}()
    function target(@nospecialize(v))
        seen = 0
        while v isa Core.SSAValue && haskey(rename, v.id) && (seen += 1) <= length(rename)
            v = rename[v.id]
        end
        return v
    end
    # A phi turns trivial once the phis it merges with fold: fixed point.
    changed = true
    while changed
        changed = false
        for i in 1:length(ir.stmts)
            haskey(rename, i) && continue
            phi = ir.stmts[i][:stmt]
            phi isa Core.PhiNode || continue
            value = Ref{Any}()
            trivial = all(eachindex(phi.values)) do k
                isassigned(phi.values, k) || return false
                v = target(phi.values[k])
                v === Core.SSAValue(i) && return true
                isassigned(value) || (value[] = v)
                return v === value[]
            end
            trivial && isassigned(value) || continue
            rename[i] = value[]
            changed = true
        end
    end
    isempty(rename) && return ir
    for i in 1:length(ir.stmts)
        setstmt!(ir, i, haskey(rename, i) ? nothing : CC.ssamap(target, ir.stmts[i][:stmt]))
    end
    return ir
end

# On Julia 1.11 `Base.setindex!` has no method for an `Instruction`.
function setstmt!(ir::CC.IRCode, i::Int, @nospecialize(stmt))
    return CC.setindex!(ir.stmts[i], stmt, :stmt)
end

# The inlined `iterate` over a range from a leaf call re-reads `stop` inside the
# loop; IRStructurizer up to 0.6.4 then takes that body-defined value as the
# bound. Later releases hoist such reads (maleadt/IRStructurizer.jl#62).
const STRUCTURIZER_HOISTS = pkgversion(IRStructurizer) > v"0.6.4"

function dedup_getfield!(ir::CC.IRCode)
    domtree = CC.construct_domtree(ir.cfg.blocks)
    seen = Dict{Any,Tuple{Int,Int}}()
    rename = Dict{Int,Int}()
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        stmt isa Expr && stmt.head === :call && length(stmt.args) == 3 || continue
        f = stmt.args[1]
        f isa GlobalRef &&
            (f = isdefined(f.mod, f.name) ? getglobal(f.mod, f.name) : nothing)
        f === Core.getfield || continue
        object, field = stmt.args[2], stmt.args[3]
        object isa Union{Core.SSAValue,Core.Argument} || continue
        field isa QuoteNode && (field = field.value)
        field isa Union{Symbol,Int} || continue
        T = CC.widenconst(CC.argextype(object, ir))
        (T isa DataType && isconcretetype(T) && !ismutabletype(T)) || continue
        (field isa Symbol ? field in fieldnames(T) : 1 <= field <= fieldcount(T)) ||
            continue
        block = CC.block_for_inst(ir.cfg, i)
        earlier = get(seen, (object, field), nothing)
        if earlier !== nothing && CC.dominates(domtree, earlier[2], block)
            rename[i] = earlier[1]
        else
            seen[(object, field)] = (i, block)
        end
    end
    isempty(rename) && return ir
    for i in 1:length(ir.stmts)
        renamed = CC.ssamap(v -> Core.SSAValue(get(rename, v.id, v.id)), ir.stmts[i][:stmt])
        setstmt!(ir, i, renamed)
    end
    return ir
end
