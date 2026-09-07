"""
    ReactantIRFrontend

Capture ordinary Julia control flow for Reactant from inferred, structured IR.

Reactant discovers a program by executing it with traced values, so a branch or
loop on a traced value has to be annotated with `@trace`. This package infers the
program on traced types instead, restructures the SSA IR into `if`/`while`/`for`
regions with IRStructurizer, and walks that IR. Every call to a method Reactant
owns is handed to Reactant unchanged; only the control flow between those calls
is emitted here, through Reactant's own region builders.

See [`structured`](@ref), and the package's [`@compile`](@ref) for using it as the
default frontend.
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
