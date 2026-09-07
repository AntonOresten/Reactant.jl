module StructuredReactant

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

include("errors.jl")
public FrontendError

include("interpreter.jl")

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
