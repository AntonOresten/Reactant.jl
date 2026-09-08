# Resolve operand locations once per method, rather than dispatching on Julia
# IR nodes at every statement. Globals remain reads: a non-const binding can
# change without advancing the method world.
@enum OperandKind::UInt8 LITERAL SSA ARGUMENT BLOCK_ARGUMENT GLOBAL UNDEFINED

struct PreparedOperand
    kind::OperandKind
    index::Int
    value::Any
end

prepare_operand(@nospecialize(x)) = PreparedOperand(LITERAL, 0, x)
prepare_operand(x::Core.SSAValue) = PreparedOperand(SSA, x.id, nothing)
prepare_operand(x::Core.Argument) = PreparedOperand(ARGUMENT, x.n, nothing)
prepare_operand(x::BlockArgument) = PreparedOperand(BLOCK_ARGUMENT, x.id, nothing)
prepare_operand(x::Core.PiNode) = prepare_operand(x.val)
prepare_operand(x::QuoteNode) = PreparedOperand(LITERAL, 0, x.value)
prepare_operand(x::GlobalRef) = PreparedOperand(GLOBAL, 0, x)
prepare_operand(x::Undef) = PreparedOperand(UNDEFINED, 0, x)

@enum StatementKind::UInt8 VALUE CALL NEW OTHER

struct PreparedStatement
    kind::StatementKind
    value::PreparedOperand
    operands::Vector{PreparedOperand}
    # An if needs a continuation when it returns, or exits a general loop.
    returns::Bool
    exits::Bool
end

@enum TerminatorKind::UInt8 RETURN YIELD CONTINUE BREAK CONDITION UNREACHABLE UNKNOWN

struct PreparedTerminator
    kind::TerminatorKind
    value::PreparedOperand
    operands::Vector{PreparedOperand}
end

struct PreparedBlock
    statements::Vector{PreparedStatement}
    terminator::PreparedTerminator
end

const NO_OPERANDS = PreparedOperand[]

function prepare_statement(@nospecialize(stmt), returns, exits)
    if stmt isa Expr
        head = stmt.head
        if head === :call || head === :invoke || head === :new
            first = head === :invoke ? 2 : 1
            args = PreparedOperand[
                prepare_operand(stmt.args[i]) for i in (first + 1):length(stmt.args)
            ]
            kind = head === :new ? NEW : CALL
            return PreparedStatement(
                kind, prepare_operand(stmt.args[first]), args, false, false
            )
        end
    elseif stmt isa IfOp
        return PreparedStatement(
            OTHER,
            prepare_operand(stmt),
            NO_OPERANDS,
            has_return(returns, stmt),
            exits_loop(exits, stmt),
        )
    elseif !(
        stmt isa Union{
            ControlFlowOp,
            Core.PhiNode,
            Core.GotoNode,
            Core.GotoIfNot,
            Core.PhiCNode,
            Core.UpsilonNode,
            Core.EnterNode,
            Core.ReturnNode,
        }
    )
        return PreparedStatement(VALUE, prepare_operand(stmt), NO_OPERANDS, false, false)
    end
    return PreparedStatement(OTHER, prepare_operand(stmt), NO_OPERANDS, false, false)
end

function prepare_terminator(@nospecialize(t))
    if t isa Core.ReturnNode
        isdefined(t, :val) &&
            return PreparedTerminator(RETURN, prepare_operand(t.val), NO_OPERANDS)
        return PreparedTerminator(UNREACHABLE, prepare_operand(nothing), NO_OPERANDS)
    elseif t isa Union{YieldOp,ContinueOp,BreakOp}
        kind = if t isa YieldOp
            YIELD
        elseif t isa ContinueOp
            CONTINUE
        else
            BREAK
        end
        return PreparedTerminator(
            kind, prepare_operand(nothing), map(prepare_operand, t.values)
        )
    elseif t isa ConditionOp
        return PreparedTerminator(
            CONDITION, prepare_operand(t.condition), map(prepare_operand, t.args)
        )
    end
    return PreparedTerminator(UNKNOWN, prepare_operand(nothing), NO_OPERANDS)
end

function prepare(sci::StructuredIRCode, returns, exits)
    prepared = IdDict{Block,PreparedBlock}()
    for block in eachblock(sci.entry)
        prepared[block] = PreparedBlock(
            PreparedStatement[
                prepare_statement(stmt, returns, exits) for stmt in block.body.stmts
            ],
            prepare_terminator(block.terminator),
        )
    end
    return prepared
end
