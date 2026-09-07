"""
    ReactantIRFrontend

Ordinary Julia control flow for Reactant, without `@trace`: the program is
inferred on traced types, restructured into `if`/`while`/`for` regions with
IRStructurizer, and interpreted; every call to a method Reactant owns is handed
to Reactant unchanged. See [`structured`](@ref) and the package's [`@compile`](@ref).
"""
module ReactantIRFrontend

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

export structured

include("errors.jl")
include("interpreter.jl")
include("code.jl")
include("program.jl")
include("entrypoints.jl")
include("emit.jl")
include("intrinsics.jl")
include("regions.jl")
include("precompile.jl")

end
