# Standalone acceptance suite for the unreleased general-loop checkpointing
# pipeline and shared Enzyme gradient-cleanup fix. See compatibility.md.
# The default test runner deliberately does not include this file: use a CPU
# environment selecting the patched native library and normalization pipeline.
using Reactant, StructuredReactant, Enzyme, Test
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x; track_numbers=Number)
include("differentiation.jl")
module ExpandedLoops
using Reactant, Enzyme
function early_break(x)
    while true
        sum(x) <= 0.2f0 && break
        x = sin.(x) .* 0.5f0
    end
    return sum(x)
end
function mutating_condition(x)
    while begin
        x .= sin.(x) .* 0.5f0
        sum(x) > 0.2f0
    end
        x .*= 0.8f0
    end
    return sum(x)
end
Base.@noinline function inner(x)
    while sum(x) > 0.4f0
        x = sin.(x) .* 0.5f0
    end
    return x
end
function nested_general(x)
    while sum(x) > 0.2f0
        x = inner(x)
        x = sin.(x) .* 0.5f0
    end
    return sum(x)
end
function branch_body(x)
    while sum(x) > 0.2f0
        if sum(x) > 0.6f0
            x = sin.(x) .* 0.4f0
        else
            x = sin.(x) .* 0.6f0
        end
    end
    return sum(x)
end
function nested_return(x)
    while sum(x) > 0.2f0
        for _ in 1:2
            sum(x) < 0.3f0 && return sum(x)
            x = sin.(x) .* 0.5f0
        end
    end
    return sum(x)
end
reverse(f, x) = Enzyme.gradient(Reverse, f, x)[1]
forward(f, x, dx) = Enzyme.autodiff(Forward, Const(f), Duplicated(x, dx))[1]
end
function reference(f, x)
    # Independent high-precision finite difference, away from branch boundaries.
    x = Float64.(x)
    h = 1e-6
    return map(eachindex(x)) do i
        lo, hi = copy(x), copy(x)
        lo[i] -= h
        hi[i] += h
        (f(hi) - f(lo)) / (2h)
    end
end
inputs = (Float32[0.04, 0.05], Float32[0.21, 0.14], Float32[0.8, 0.5])
direction = Float32[0.3, -0.7]
for body in (ExpandedLoops.early_break, ExpandedLoops.mutating_condition,
             ExpandedLoops.nested_general, ExpandedLoops.branch_body,
             ExpandedLoops.nested_return)
    for strategy in (true, Reactant.Binomial(2), Reactant.Binomial(4))
        println("START ", body, " ", strategy); flush(stdout)
        @testset "$body $strategy" begin
            f = structured(body; checkpointing=strategy)
            primal = Reactant.compile(f, (R(copy(first(inputs))),))
            rev = Reactant.compile(ExpandedLoops.reverse, (f, R(copy(first(inputs)))))
            fwd = Reactant.compile(ExpandedLoops.forward, (f, R(copy(first(inputs))), R(direction)))
            for input in inputs
                want = reference(body, input)
                @test primal(R(copy(input)))[] ≈ body(copy(input))
                @test Array(rev(f, R(copy(input)))) ≈ want rtol=1e-4 atol=1e-5
                @test fwd(f, R(copy(input)), R(copy(direction)))[] ≈ sum(want .* direction) rtol=1e-4 atol=1e-5
            end
        end
    end
end

function vanilla_convergence(x)
    Reactant.@trace checkpointing=Reactant.Binomial(2) while sum(x) > 0.2f0
        x = sin.(x) .* 0.5f0
    end
    return sum(x)
end
inputs = (Float32[0.04, 0.05], Float32[0.2, 0.1], Float32[0.8, 0.5])
direction = Float32[0.3, -0.7]
cases = DifferentiationCases
for body in (cases.convergence, cases.convergence_mutation, cases.nested, cases.early_exit, vanilla_convergence)
    println("START ", body); flush(stdout)
    @testset "$body reverse checkpointing" begin
        f = body === vanilla_convergence ? body : structured(body; checkpointing=Reactant.Binomial(2))
        compiled = Reactant.compile(cases.rev, (f, R(copy(first(inputs))), R(direction)))
        for input in inputs
            @test Array(compiled(f, R(copy(input)), R(copy(direction)))) ≈ cases.gradient_reference(input, body === cases.nested) rtol=1.0f-5 atol=1.0f-6
        end
        if body === cases.convergence || body === vanilla_convergence
            ir = string(Reactant.@code_hlo f(R(copy(last(inputs)))))
            @test count("stablehlo.while", ir) == 1
        end
    end
end
