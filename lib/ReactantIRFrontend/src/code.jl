# Structured IR per call signature, cached for the worlds it is valid in.

"""
    Code

The structured IR of one method specialization, with what the emitter needs to
run it: the method (for argument packing and diagnostics), its static
parameters, and the worlds in which the inference result is valid.
"""
struct Code
    sci::StructuredIRCode
    method::Method
    sparams::Core.SimpleVector
    valid_worlds::CC.WorldRange
end

# What a call signature dispatches to. `code === nothing` marks a leaf.
struct Resolution
    code::Union{Nothing,Code}
    valid_worlds::CC.WorldRange
end

const RESOLUTIONS = Dict{Type,Resolution}()
const RESOLUTIONS_LOCK = ReentrantLock()

function covers(r::Resolution, world::UInt)
    return r.valid_worlds.min_world <= world <= r.valid_worlds.max_world
end

function worlds_intersection(a::CC.WorldRange, b::CC.WorldRange)
    return CC.WorldRange(max(a.min_world, b.min_world), min(a.max_world, b.max_world))
end

"""
    resolve(sig, world) -> Union{Nothing, Resolution}

Find the method that `sig` dispatches to in `world`: `nothing` when there is no
method, a leaf resolution when Reactant owns it, and otherwise the method's
structured IR. Results are cached until a method definition invalidates them.
"""
function resolve(@nospecialize(sig::Type), world::UInt)
    return lock(RESOLUTIONS_LOCK) do
        cached = get(RESOLUTIONS, sig, nothing)
        cached !== nothing && covers(cached, world) && return cached

        interp = Interpreter(world)
        match, valid_worlds = CC.findsup(sig, CC.method_table(interp))
        match === nothing && return nothing

        resolution = if isleaf(match.method, sig)
            Resolution(nothing, valid_worlds)
        else
            mi = specialization(interp, match, sig)
            ir, _ = CC.typeinf_ircode(interp, mi, nothing)
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
            valid_worlds = worlds_intersection(valid_worlds, sci.valid_worlds)
            Resolution(Code(sci, match.method, mi.sparam_vals, valid_worlds), valid_worlds)
        end
        RESOLUTIONS[sig] = resolution
        return resolution
    end
end

# Infer with traced Bool arguments admitted as host Bools, for the same reason
# leaf results are (see the interpreter). The concrete signature stays the
# cache key; the widened one only shapes inference.
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

# Exception handlers are not traced: nothing throws in the compiled program, and
# an exception raised while tracing aborts the compile. Only the normal path
# survives, including the copy of a `finally` body on it, which is what
# `@allowscalar` needs to restore the task's scalar-indexing state.
function strip_exception_handling!(ir::CC.IRCode)
    found = false
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Core.EnterNode
            found = true
            CC.kill_edge!(ir, CC.block_for_inst(ir.cfg, i), stmt.catch_dest)
            ir.stmts[i][:stmt] = nothing
        elseif stmt isa Core.UpsilonNode ||
            (stmt isa Expr && stmt.head in (:leave, :pop_exception))
            ir.stmts[i][:stmt] = nothing
        end
    end
    return found ? CC.compact!(ir, true) : ir
end

# Reading a field of an immutable value is pure, so a read dominated by an
# identical earlier read can reuse it. Julia does not do this itself, and the
# inlined `iterate` over a range that came from a leaf call (`eachindex(x)`,
# `axes(x, 1)`) re-reads `stop` inside the loop; IRStructurizer releases up to
# 0.6.4 then take that body-defined value as the loop bound and produce invalid
# IR. Later releases hoist such reads themselves (maleadt/IRStructurizer.jl#62).
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
        ir.stmts[i][:stmt] = CC.ssamap(
            v -> Core.SSAValue(get(rename, v.id, v.id)), ir.stmts[i][:stmt]
        )
    end
    return ir
end
