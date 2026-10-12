module ProductCheckpointAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x; track_numbers=Number)
function product_loss(x, n)
    result = zero(eltype(x))
    for i in 1:n
        result += prod(x .+ (i - 2))
    end
    return result
end
gradient(f, x, n) = Enzyme.gradient(Reverse, f, x, Const(n))[1]
function hvp(f, x, v, n)
    body(x) = gradient(f, x, n)
    return Enzyme.autodiff(Forward, Const(body), Duplicated(x, v))[1]
end
function reference(x, v, n)
    g = zeros(size(x)); h = zeros(size(x))
    for t in 1:n
        y = x .+ (t - 2)
        for i in eachindex(y)
            g[i] += prod((y[j] for j in eachindex(y) if j != i); init=1.0)
            h[i] += sum((v[j] * prod((y[k] for k in eachindex(y) if k != i && k != j); init=1.0)
                         for j in eachindex(y) if j != i); init=0.0)
        end
    end
    return g, h
end
@testset "checkpointed products at zeros" begin
    x = [0., 1., 0., 2.]; v = [0.5, -1., 2., 0.25]
    for policy in (:auto, Reactant.Binomial(3), Reactant.Binomial(8))
        @testset "$(policy)" begin
            body = structured(product_loss; checkpointing=policy)
            f = Reactant.compile(gradient, (body, R(x), R(2)))
            h = Reactant.compile(hvp, (body, R(x), R(v), R(2)))
            for n in 0:6
                expected, expected_hvp = reference(x, v, n)
                @test Array(f(body, R(x), R(n))) ≈ expected
                @test Array(h(body, R(x), R(v), R(n))) ≈ expected_hvp
            end
        end
    end
end
end
