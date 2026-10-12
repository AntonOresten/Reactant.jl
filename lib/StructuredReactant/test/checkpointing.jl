module CheckpointCases
using Reactant, StructuredReactant, Enzyme

function iterated(x, n)
    for _ in 1:n
        x = sin.(x)
    end
    return sum(x)
end
function mutating(x, n)
    for _ in 1:n
        x .= sin.(x)
    end
    return sum(x)
end
fixed(x) = iterated(x, 5)
keyword(x; n) = iterated(x, n)
function gradient(f, x, n)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, f, Active, Duplicated(x, dx), Const(n))
    return dx
end
inherited_gradient(x, n) = gradient(iterated, x, n)
failed(x) = error("checkpoint scope restoration")
function overridden(x, n)
    f = structured(iterated; checkpointing=Reactant.Binomial(2))
    return f(x, n)
end
fixed_gradient(f, x) = Enzyme.gradient(Reverse, f, x)[1]

# Independent chain rule for the nonlinear recurrence, including zero steps.
function reference(x, n)
    value, derivative = copy(x), ones(eltype(x), size(x))
    for _ in 1:n
        derivative .*= cos.(value)
        value = sin.(value)
    end
    return derivative
end
end

@testset "checkpointing" begin
    CCases = CheckpointCases
    input = Float32[0.7, 0.3]
    for strategy in (:auto, true, Reactant.Binomial(2), Reactant.Binomial(4))
        for body in (CCases.iterated, CCases.mutating)
            f = structured(body; checkpointing=strategy)
            compiled = Reactant.compile(CCases.gradient, (f, R(input), R(3)))
            for n in (0, 1, 3, 11)
                @test Array(compiled(f, R(copy(input)), R(n))) ≈ CCases.reference(input, n)
            end
        end
    end

    # The outer option reaches callbacks wrapped when Enzyme is entered.
    f = structured(CCases.inherited_gradient; checkpointing=Reactant.Binomial(3))
    compiled = Reactant.compile(f, (R(input), R(3)))
    @test Array(compiled(R(copy(input)), R(7))) ≈ CCases.reference(input, 7)
    ir = unoptimized(f, input, 3)
    @test occursin("enzyme.checkpoint_period = 3", ir)

    f = structured(CCases.overridden; checkpointing=Reactant.Binomial(7))
    ir = unoptimized(f, input, 3)
    @test occursin("enzyme.checkpoint_period = 2", ir)
    @test !occursin("enzyme.checkpoint_period = 7", ir)

    # A keyword entry point must establish the same scope as an ordinary call.
    f = structured(CCases.keyword; checkpointing=Reactant.Binomial(7))
    ir = string(Reactant.@code_hlo optimize=false f(R(input); n=3))
    @test occursin("enzyme.checkpoint_period = 7", ir)

    # Static loops retain ordinary caching by default; explicit options remain
    # available to trade recomputation against memory.
    @test !occursin("enzyme.enable_checkpointing", unoptimized(CCases.fixed, input))
    f = structured(CCases.fixed; checkpointing=Reactant.Periodic(2))
    @test occursin("enzyme.checkpoint_period = 2", unoptimized(f, input))
    f = structured(CCases.iterated; checkpointing=false)
    @test !occursin("enzyme.enable_checkpointing", unoptimized(f, input, 3))

    for strategy in (false, true, Reactant.Periodic(2), Reactant.Binomial(3))
        f = structured(CCases.fixed; checkpointing=strategy)
        compiled = Reactant.compile(CCases.fixed_gradient, (f, R(input)))
        @test Array(compiled(f, R(copy(input)))) ≈ CCases.reference(input, 5)
    end

    # Failure cannot leak a strategy into a later, independent compilation.
    f = structured(CCases.failed; checkpointing=Reactant.Binomial(7))
    @test_throws Exception Reactant.compile(f, (R(input),))
    @test occursin("enzyme.checkpoint_period = 4", unoptimized(CCases.iterated, input, 3))

    for strategy in (:invalid, 3, Reactant.Periodic(0), Reactant.Binomial(1))
        @test_throws ArgumentError structured(CCases.iterated; checkpointing=strategy)
    end
end
