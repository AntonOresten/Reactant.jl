module DifferentiationCases
using Reactant, Enzyme
function convergence(x)
    while sum(x) > 0.2f0
        x = sin.(x) .* 0.5f0
    end
    return sum(x)
end
function convergence_mutation(x)
    while sum(x) > 0.2f0
        x .= sin.(x) .* 0.5f0
    end
    return sum(x)
end
# Keep the inner counted loop across a call boundary, also exercising nested
# checkpoint policy. Directly nested for IR is covered in runtests.jl and still
# requires post-0.6.4 IRStructurizer for its carried initializers.
Base.@noinline function inner(x)
    for j in 1:2
        x = sin.(x) .* 0.5f0
    end
    return x
end
function nested(x)
    for i in 1:3
        x = inner(x)
    end
    return sum(x)
end
function early_exit(x)
    while true
        sum(x) <= 0.2f0 && return sum(x)
        x = sin.(x) .* 0.5f0
    end
end
function gradient_reference(x, fixed)
    d = ones(Float32, size(x))
    value = copy(x)
    if fixed
        for i in 1:6
            d .*= 0.5f0 .* cos.(value)
            value = 0.5f0 .* sin.(value)
        end
    else
        while sum(value) > 0.2f0
            d .*= 0.5f0 .* cos.(value)
            value = 0.5f0 .* sin.(value)
        end
    end
    return d
end
fwd(f, x, v) = Enzyme.autodiff(Forward, Const(f), Duplicated(x, v))[1]
rev(f, x, v) = Enzyme.gradient(Reverse, f, x)[1]
end

@testset "differentiation across control flow" begin
    cases = DifferentiationCases
    inputs = (Float32[0.04, 0.05], Float32[0.2, 0.1], Float32[0.8, 0.5])
    direction = Float32[0.3, -0.7]
    for body in
        (cases.convergence, cases.convergence_mutation, cases.nested, cases.early_exit)
        f = structured(body; checkpointing=Reactant.Binomial(4))
        compiled = Reactant.compile(cases.fwd, (f, R(first(inputs)), R(direction)))
        for input in inputs
            derivative = cases.gradient_reference(input, body === cases.nested)
            result = compiled(f, R(copy(input)), R(copy(direction)))
            @test result[] ≈ sum(derivative .* direction) rtol=1.0f-5 atol=1.0f-6
        end
    end
    f = structured(cases.nested; checkpointing=Reactant.Binomial(4))
    compiled = Reactant.compile(cases.rev, (f, R(first(inputs)), R(direction)))
    for input in inputs
        @test Array(compiled(f, R(copy(input)), R(copy(direction)))) ≈
            cases.gradient_reference(input, true) rtol=1.0f-5 atol=1.0f-6
    end
    # General-condition reverse-mode acceptance remains recorded separately:
    # the native cache-removal pass rejects its unknown iteration counts.
end
