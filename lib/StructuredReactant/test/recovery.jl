module TraceRecoveryTests
using Reactant, StructuredReactant, Test

const fail = Ref(false)
Base.@noinline function maybe_fail()
    fail[] && throw(ArgumentError("deliberate trace failure"))
    return 0.5
end
function branch(x)
    if sum(x) > 0
        y = x .* 2
        return y .* maybe_fail()
    end
    return x .* 3
end
function loop(x, n)
    for i in 1:n
        x = x .* maybe_fail()
    end
    return x
end
function nested(x, n)
    for i in 1:n
        if sum(x) > 0
            x = x .* maybe_fail()
        else
            x = x .* 3
        end
    end
    return x
end
function scalar(x, n)
    for i in 1:n
        Reactant.@allowscalar x[1] = x[1] * maybe_fail()
    end
    return x
end

const stacks = (:mlir_context_stack, :mlir_block, :mlir_module,
                :entry_block, :reactant_is_raising)
stack_depths() = map(key -> length(get(task_local_storage(), key, ())), stacks)
R(x) = Reactant.to_rarray(x; track_numbers=Number)

@testset "Failure recovery in traced regions" begin
    for f in (branch, loop, nested, scalar)
        input = f === branch ? ([1.0, 2.0],) : ([1.0, 2.0], 3)
        args = map(R, input)
        before = stack_depths()
        saved_scalar = get(task_local_storage(), :ScalarIndexing, nothing)
        saved_loop = StructuredReactant.CURRENT_LOOP[]
        fail[] = true
        try
            @test_throws ArgumentError("deliberate trace failure") Reactant.compile(
                structured(f), args
            )
        finally
            fail[] = false
        end
        @test stack_depths() == before
        @test get(task_local_storage(), :ScalarIndexing, nothing) === saved_scalar
        @test StructuredReactant.CURRENT_LOOP[] === saved_loop
        compiled = Reactant.compile(structured(f), args)
        for sign in (1, -1)
            host_args = f === branch ? (sign .* [1.0, 2.0],) : (sign .* [1.0, 2.0], 3)
            expected = f(deepcopy(host_args)...)
            @test Array(compiled(map(R, host_args)...)) ≈ expected
        end
    end
end
end
