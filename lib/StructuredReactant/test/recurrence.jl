# Reduced from the Deriche kernel in the public Polybench comparison corpus.
module RecurrenceCases
function alias_change(x, ::Val{N}) where {N}
    a, b = x, x
    for _ in 1:N
        a .+= one(eltype(a))
        b = b .* 2
    end
    return a, b
end
function forward(x)
    out = zero(x)
    xm1 = ym1 = ym2 = 0.0
    for j in axes(x, 1)
        out[j] = 0.1 * x[j] + 0.2 * xm1 + 0.3 * ym1 + 0.4 * ym2
        xm1 = x[j]
        ym2 = ym1
        ym1 = out[j]
    end
    return out
end

function backward(x)
    out = zero(x)
    xm1 = ym1 = ym2 = 0.0
    for j in length(x):-1:1
        out[j] = 0.1 * x[j] + 0.2 * xm1 + 0.3 * ym1 + 0.4 * ym2
        xm1 = x[j]
        ym2 = ym1
        ym1 = out[j]
    end
    return out
end
function nested_state(x)
    out = zero(x)
    # Start with traced fields: this checks simultaneous nested updates without
    # introducing a separate change of the tuple's field types during inference.
    z = zero(x[1])
    state = (input=z, outputs=(last=z, previous=z))
    for j in axes(x, 1)
        out[j] =
            0.1 * x[j] +
            0.2 * state.input +
            0.3 * state.outputs.last +
            0.4 * state.outputs.previous
        state = (input=x[j], outputs=(last=out[j], previous=state.outputs.last))
    end
    return out
end
end

@testset "loop recurrence" begin
    Reactant.allowscalar() do
        for n in (3, 5)
            input = Float64.(1:n)
            for f in (
                RecurrenceCases.forward,
                RecurrenceCases.backward,
                RecurrenceCases.nested_state,
            )
                # Isolate frontend semantics from the native optimization failure
                # also found in both frontends by the larger Polybench corpus.
                compiled = Reactant.compile(structured(f), (R(input),); optimize=false)
                for x in (input, reverse(input))
                    actual = Array(compiled(R(copy(x))))
                    expected = f(copy(x))
                    @test actual ≈ expected
                end
            end
        end
    end
end

@testset "loop entry aliases" begin
    for n in (0, 1, 8)
        input = Float64[0.01, 0.1, 0.5, 0.9]
        f = RecurrenceCases.alias_change
        compiled = Reactant.compile(structured(f), (R(input), Val(n)))
        for values in (input, reverse(input))
            expected = f(copy(values), Val(n))
            actual = compiled(R(copy(values)), Val(n))
            @test matches(actual, expected)
        end
    end
end
