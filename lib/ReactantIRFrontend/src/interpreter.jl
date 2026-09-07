# Inference on traced types.
#
# The interpreter is a plain `NativeInterpreter` with three differences:
#   * it sees Reactant's overlay method table, so the program is typed exactly as
#     Reactant would execute it;
#   * `within_compile()` is a constant `true`, as in Reactant's own interpreter,
#     so code that still uses `@trace` keeps taking its traced path;
#   * leaf calls (Reactant's own methods, and Base or standard-library methods
#     applied to Reactant types) are neither inferred here nor inlined. A leaf
#     call is typed by Reactant's own interpreter, which infers those bodies
#     anyway when the leaf is emitted.

struct CacheOwner end

struct Interpreter <: CC.AbstractInterpreter
    world::UInt
    inference_params::CC.InferenceParams
    optimization_params::CC.OptimizationParams
    inference_cache::Vector{CC.InferenceResult}
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
end

function Interpreter(world::UInt=Base.get_world_counter())
    table = CC.OverlayMethodTable(world, Reactant.REACTANT_METHOD_TABLE)
    return Interpreter(
        world,
        CC.InferenceParams(),
        CC.OptimizationParams(),
        CC.InferenceResult[],
        CC.CachedMethodTable(table),
    )
end

CC.InferenceParams(interp::Interpreter) = interp.inference_params
CC.OptimizationParams(interp::Interpreter) = interp.optimization_params
CC.get_inference_cache(interp::Interpreter) = interp.inference_cache
CC.get_inference_world(interp::Interpreter) = interp.world
CC.method_table(interp::Interpreter) = interp.method_table
CC.cache_owner(::Interpreter) = CacheOwner()
CC.lock_mi_inference(::Interpreter, ::Core.MethodInstance) = nothing
CC.unlock_mi_inference(::Interpreter, ::Core.MethodInstance) = nothing
CC.may_discard_trees(::Interpreter) = false

"""
    isleaf(method, sig) -> Bool

Whether the call `sig` to `method` is an emission leaf. The emitter never walks
a leaf's IR; it calls it through `Reactant.call_with_reactant`, as Reactant's
frontend does at every call site. Leaves are Reactant's own methods and the
methods in its overlay table, and, when the call involves Reactant types,
methods of Base and the standard library: those run exactly as they do under
Reactant today, and what happens inside them is Reactant's business.
"""
function isleaf(method::Method, @nospecialize(sig))
    if isdefined(method, :external_mt)
        method.external_mt === Reactant.REACTANT_METHOD_TABLE && return true
    end
    root = Base.moduleroot(method.module)
    root in LEAF_ROOTS && return true
    return isstdlib(root) && involves_reactant(sig)
end

const LEAF_ROOTS = Module[Reactant, ReactantCore, Reactant.Enzyme, Reactant.EnzymeCore]

function isstdlib(root::Module)
    (root === Base || root === Core) && return true
    path = pathof(root)
    return path !== nothing && startswith(path, Sys.STDLIB)
end

function involves_reactant(@nospecialize(T))
    T = Base.unwrap_unionall(T)
    if T isa DataType
        Base.moduleroot(T.name.module) === Reactant && return true
        return any(involves_reactant, T.parameters)
    elseif T isa Union
        return involves_reactant(T.a) || involves_reactant(T.b)
    elseif T isa Core.TypeofVararg
        return isdefined(T, :T) && involves_reactant(T.T)
    end
    return false
end

function CC.src_inlining_policy(
    interp::Interpreter,
    @nospecialize(src),
    @nospecialize(info::CC.CallInfo),
    stmt_flag::UInt32,
)
    inlines_leaf(info) && return false
    opaque_to_emitter(src) && return false
    return @invoke CC.src_inlining_policy(
        interp::CC.AbstractInterpreter, src::Any, info::CC.CallInfo, stmt_flag::UInt32
    )
end

# Host helpers such as `task_local_storage` bottom out in foreign calls, which
# the emitter cannot interpret but can run natively as an out-of-line call. Keep
# them, and anything with an exception handler, out of line. Sources must stay
# uncompressed for the policy to see them.
CC.may_compress(::Interpreter) = false

function opaque_to_emitter(@nospecialize(src))
    stmts = if src isa Core.CodeInfo
        src.code
    elseif src isa CC.IRCode
        src.stmts.stmt
    else
        return false
    end
    return any(stmts) do stmt
        return (stmt isa Expr && stmt.head === :foreigncall) || stmt isa Core.EnterNode
    end
end

function inlines_leaf(info::CC.MethodMatchInfo)
    return any(m -> isleaf(m.method, m.spec_types), info.results)
end
inlines_leaf(info::CC.UnionSplitInfo) = any(inlines_leaf, info.split)
inlines_leaf(info::CC.ConstCallInfo) = inlines_leaf(info.call)
inlines_leaf(info::CC.ApplyCallInfo) = inlines_leaf(info.call)
inlines_leaf(info::CC.UnionSplitApplyCallInfo) = any(inlines_leaf, info.infos)
inlines_leaf(info::CC.InvokeCallInfo) = isleaf(info.match.method, info.match.spec_types)
inlines_leaf(@nospecialize(info::CC.CallInfo)) = false

# Type a leaf call with Reactant's interpreter and stop there. Descending into
# MLIR builders and Enzyme from this interpreter is wasted work and, for deep
# call chains, overflows the stack during inference.
function CC.abstract_call_method(
    interp::Interpreter,
    method::Method,
    @nospecialize(sig),
    sparams::Core.SimpleVector,
    hardlimit::Bool,
    si::CC.StmtInfo,
    sv::CC.AbsIntState,
)
    if isleaf(method, sig)
        rt = leaf_return_type(sig, interp.world)
        # Reactant's interpreter sees a callback that branches on a traced Bool
        # as always throwing; here the callback is structured, so the leaf may
        # well return.
        rt === Union{} && (rt = Any)
        # A traced Bool in boolean context branches symbolically instead of
        # throwing. Julia would prove such a branch dead; admitting a host
        # Bool keeps it, and dispatch on the traced value is unaffected.
        if TracedRNumber{Bool} in Base.uniontypes(rt)
            rt = Union{rt,Bool}
        end
        result = CC.MethodCallResult(rt, Any, CC.Effects(), nothing, false, false)
        return CC.Future{CC.MethodCallResult}(result)
    end
    return @invoke CC.abstract_call_method(
        interp::CC.AbstractInterpreter,
        method::Method,
        sig::Any,
        sparams::Core.SimpleVector,
        hardlimit::Bool,
        si::CC.StmtInfo,
        sv::CC.AbsIntState,
    )
end

function leaf_return_type(@nospecialize(sig), world::UInt)
    sig = Base.unwrap_unionall(sig)
    sig isa DataType || return Any
    return CC._return_type(Reactant.ReactantInterpreter(; world), sig)
end

function CC.abstract_call_known(
    interp::Interpreter,
    @nospecialize(f),
    arginfo::CC.ArgInfo,
    si::CC.StmtInfo,
    sv::CC.InferenceState,
    max_methods::Int=CC.get_max_methods(interp, f, sv),
)
    if f === ReactantCore.within_compile && length(arginfo.argtypes) == 1
        meta = CC.CallMeta(
            Core.Const(true), Union{}, CC.EFFECTS_TOTAL, CC.MethodResultPure()
        )
        return CC.Future{CC.CallMeta}(meta)
    end
    return @invoke CC.abstract_call_known(
        interp::CC.AbstractInterpreter,
        f::Any,
        arginfo::CC.ArgInfo,
        si::CC.StmtInfo,
        sv::CC.InferenceState,
        max_methods::Int,
    )
end
