# Standalone acceptance: host storage iteration must preserve traced loop bodies.
module HostIteratorAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x)

function dictionary_loss(x)
    values = IdDict{Any,Any}()
    values[Ref(:first)] = x
    values[Ref(:second)] = 2 .* x
    result = sum(zero.(x))
    for (_, value) in values
        if sum(value) > 0
            result += sum(value .* value)
        else
            result += sum(3 .* value)
        end
    end
    return result
end

struct UserIterator{T}
    value::T
end
function Base.iterate(values::UserIterator, state::Int=1)
    state > 1 && return nothing
    x = values.value
    value = if sum(x) > 0
        x .* x
    else
        3 .* x
    end
    return value, 2
end
function user_iterator_loss(x)
    result = sum(zero.(x))
    for value in UserIterator{Any}(x)
        result += sum(value)
    end
    return result
end

reverse(f, x) = Enzyme.gradient(Reverse, f, x)[1]
forward(f, x, direction) = Enzyme.autodiff(Forward, Const(f), Duplicated(x, direction))[1]
function finite_gradient(f, x)
    step = 1e-5
    return map(eachindex(x)) do i
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (f(plus) - f(minus)) / (2 * step)
    end
end

@testset "host iterators preserve structured values and gradients" begin
    inputs = (Float64[0.8, 1.3, 2.1], Float64[-0.4, -1.6, -2.2])
    direction = Float64[0.2, -0.3, 0.7]
    for body in (dictionary_loss, user_iterator_loss)
        @testset "$body" begin
            f = structured(body)
            compiled = Reactant.compile(f, (R(first(inputs)),))
            reverse_compiled = Reactant.compile(reverse, (f, R(first(inputs))))
            forward_compiled = Reactant.compile(forward, (f, R(first(inputs)), R(direction)))
            for input in inputs
                expected_gradient = finite_gradient(body, input)
                @test compiled(R(input))[] ≈ body(input) rtol=1e-10 atol=1e-10
                @test Array(reverse_compiled(f, R(input))) ≈ expected_gradient rtol=1e-7 atol=1e-8
                @test forward_compiled(f, R(input), R(direction))[] ≈ sum(expected_gradient .* direction) rtol=1e-7 atol=1e-8
            end
        end
    end
end
end
