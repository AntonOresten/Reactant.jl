module StructuredReactant

using CompilerCaching: CompilerCaching, CacheView

using Reactant: Reactant, Ops, TracedRArray, TracedRNumber, TracedType
using ReactantCore: ReactantCore, MissingTracedValue
using IRStructurizer:
    IRStructurizer,
    StructuredIRCode,
    Block,
    BlockArgument,
    Undef,
    ControlFlowOp,
    IfOp,
    WhileOp,
    ForOp,
    LoopOp,
    YieldOp,
    ContinueOp,
    BreakOp,
    ConditionOp,
    blocks,
    eachblock

const CC = Core.Compiler

# Keep native compiler infrastructure usable after later package loads. User
# methods are still inferred in the explicit world supplied to Interpreter.
# During precompilation invoke_in_world clamps typemax to the current world.
const COMPILER_WORLD = Ref{UInt}(typemax(UInt))

function __init__()
    COMPILER_WORLD[] = Base.get_world_counter()
    return nothing
end

include("errors.jl")
public FrontendError

include("interpreter.jl")

include("prepare.jl")

include("code.jl")

include("program.jl")
export structured

include("entrypoints.jl")
public @compile, @jit, @code_hlo, compile, code_hlo

include("emit.jl")

include("intrinsics.jl")

include("regions.jl")
public UNROLL_WARNING

include("precompile.jl")

end
