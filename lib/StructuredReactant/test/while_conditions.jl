# Regression corpus for condition cases discussed in Reactant.jl#3265.
# Each executable is reused across inputs that take different runtime paths.
module WhileConditions
function while_and(x, n)
    k = 0
    while n > 0 && sum(x) > 1
        x = x .* 0.5f0
        n -= 1
        k += 1
    end
    return x, n, k
end
function while_or(x, n)
    k = 0
    while n > 0 || sum(x) > 1
        x = x .* 0.5f0
        n -= 1
        k += 1
    end
    return x, n, k
end
function condition_rebind(x, n)
    k = 0
    while (x=x .* 0.5f0; n > 0 && sum(x) > 1)
        n -= 1
        k += 1
    end
    return x, n, k
end
function condition_mutate(x, n)
    k = 0
    while (x .*= 0.5f0; n > 0 && sum(x) > 1)
        n -= 1
        k += 1
    end
    return x, n, k
end
function counted_condition(x, n, visits)
    k = 0
    while (visits .+= 1; n > 0 && sum(x) > 1)
        x .*= 0.5f0
        n -= 1
        k += 1
    end
    return x, n, k, visits
end
function and_mutate(x, n, visits)
    k = 0
    while n > 0 && (visits .+= 1; sum(x) > 1)
        x .*= 0.5f0
        n -= 1
        k += 1
    end
    return x, n, k, visits
end
function or_mutate(x, n, visits)
    k = 0
    while n > 0 || (visits .+= 1; sum(x) > 1)
        x .*= 0.5f0
        n -= 1
        k += 1
    end
    return x, n, k, visits
end
@noinline function check!(x, visits)
    visits .+= 1
    x .*= 0.5f0
    return sum(x) > 1
end
function callee_mutate(x, n, visits)
    k = 0
    while n > 0 && check!(x, visits)
        n -= 1
        k += 1
    end
    return x, n, k, visits
end
function plain_rebind(x)
    k = 0
    while (x=x .* 0.5f0; sum(x) > 1)
        k += 1
    end
    return x, k
end
function plain_mutate(x)
    k = 0
    while (x .*= 0.5f0; sum(x) > 1)
        k += 1
    end
    return x, k
end
function plain_counted(x, visits)
    k = 0
    while (visits .+= 1; sum(x) > 1)
        x .*= 0.5f0
        k += 1
    end
    return x, k, visits
end

# The condition mutates captured fields; the body uses another alias of x.
function plain_alias(x, visits)
    state = (value=x, visits=visits)
    alias = x
    k = 0
    while check!(state.value, state.visits)
        alias .+= 0.125f0
        k += 1
    end
    return x, alias, k, visits
end

# The at-most-one-iteration shape explicitly listed in Reactant.jl#3265.
function early_return(x)
    while x < 5
        x += 1
        return x
    end
    x *= 2
    return x
end
end

@testset "while conditions" begin
    WC = WhileConditions
    pairs = [
        (Float32[a, a], n) for (a, n) in ((8, 4), (0.25, 4), (8, 0), (0.25, 0), (1, 1))
    ]
    functions = (
        WC.while_and,
        WC.while_or,
        WC.condition_rebind,
        WC.condition_mutate,
        WC.counted_condition,
        WC.and_mutate,
        WC.or_mutate,
        WC.callee_mutate,
    )
    for (i, f) in enumerate(functions)
        cases = i <= 4 ? pairs : [(x, n, zeros(Int32, 1)) for (x, n) in pairs]
        compiled = Reactant.compile(structured(f), map(R, deepcopy(first(cases))))
        @testset "$(nameof(f))" begin
            for args in cases
                expected = f(deepcopy(args)...)
                @test matches(compiled(map(R, deepcopy(args))...), expected)
            end
        end
    end

    cases = [(Float32[a, a],) for a in (8, 0.25, 1)]
    compiled = Reactant.compile(structured(WC.plain_rebind), map(R, deepcopy(first(cases))))
    for args in cases
        expected = WC.plain_rebind(deepcopy(args)...)
        @test matches(compiled(map(R, deepcopy(args))...), expected)
    end

    # The initial and final condition each run once, including when the body
    # executes zero times. Rebinding and mutation must agree on this boundary.
    for f in (WC.plain_mutate, WC.plain_counted)
        inputs = f === WC.plain_mutate ? cases : [(x, zeros(Int32, 1)) for (x,) in cases]
        compiled = Reactant.compile(structured(f), map(R, deepcopy(first(inputs))))
        @testset "$(nameof(f))" begin
            for args in inputs
                expected = f(deepcopy(args)...)
                actual = compiled(map(R, deepcopy(args))...)
                @test matches(actual, expected)
            end
        end
    end
    inputs = [(Float32[a, a], zeros(Int32, 1)) for a in (8, 0.25, 1)]
    compiled = Reactant.compile(structured(WC.plain_alias), map(R, deepcopy(first(inputs))))
    for args in inputs
        expected = WC.plain_alias(deepcopy(args)...)
        @test matches(compiled(map(R, deepcopy(args))...), expected)
    end
    for value in (1.0f0, 4.0f0, 5.0f0, 6.0f0)
        @test agrees(WC.early_return, (value,))
    end
end
