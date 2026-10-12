module ProductDerivativeAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x; track_numbers=Number)
frontend(f) = get(ENV, "FRONTEND", "automatic") == "automatic" ? structured(f) : f

# Independent polynomial oracles remain defined at one or several zero factors.
function product_gradient(x)
    return [prod((x[j] for j in eachindex(x) if j != i); init=1.0) for i in eachindex(x)]
end
function product_hvp(x, v)
    return [sum((v[j] * prod((x[k] for k in eachindex(x) if k != i && k != j); init=1.0)
                 for j in eachindex(x) if j != i); init=0.0) for i in eachindex(x)]
end
gradient_product(x) = Enzyme.gradient(Reverse, prod, x)[1]
hvp_product(x, v) = Enzyme.autodiff(Forward, Const(gradient_product), Duplicated(x, v))[1]
batch_hvp_product(x, v, w) = Enzyme.autodiff(Forward, Const(gradient_product), BatchDuplicated(x, (v, w)))[1]
batch_jvp_product(x, v, w) = Enzyme.autodiff(Forward, Const(prod), BatchDuplicated(x, (v, w)))[1]
partial_loss(x) = sum(prod(x; dims=(1, 3)))
gradient_partial(x) = Enzyme.gradient(Reverse, partial_loss, x)[1]
hvp_partial(x, v) = Enzyme.autodiff(Forward, Const(gradient_partial), Duplicated(x, v))[1]

@testset "product derivatives at zeros" begin
    inputs = ([1., 2., 3., 4.], [0., 2., 3., 4.], [0., 2., 0., 4.], zeros(4))
    v = [0.5, -1., 2., 0.25]
    w = [-2., 0.25, 1.5, 1.]
    @testset "gradient" begin
        f = Reactant.compile(frontend(gradient_product), (R(first(inputs)),))
        for x in inputs
            @test Array(f(R(x))) ≈ product_gradient(x)
        end
    end
    @testset "forward over reverse" begin
        f = Reactant.compile(frontend(hvp_product), (R(first(inputs)), R(v)))
        for x in inputs
            @test Array(f(R(x), R(v))) ≈ product_hvp(x, v)
        end
    end
    @testset "batched forward over reverse" begin
        f = Reactant.compile(frontend(batch_hvp_product), (R(first(inputs)), R(v), R(w)))
        for x in inputs
            result = f(R(x), R(v), R(w))
            @test Array(result[1]) ≈ product_hvp(x, v)
            @test Array(result[2]) ≈ product_hvp(x, w)
        end
    end
    @testset "batched forward" begin
        f = Reactant.compile(frontend(batch_jvp_product), (R(first(inputs)), R(v), R(w)))
        for x in inputs
            result = f(R(x), R(v), R(w))
            @test Float64(result[1]) ≈ sum(product_gradient(x) .* v)
            @test Float64(result[2]) ≈ sum(product_gradient(x) .* w)
        end
    end
    @testset "noncontiguous reduction axes" begin
        x = reshape([0., 2., 3., 0., 5., 6., 7., 8., 0., 10., 11., 12.], 2, 3, 2)
        v = reshape(collect(1.:12.) ./ 10, size(x))
        expected = similar(x)
        expected_hvp = similar(x)
        for j in axes(x, 2)
            expected[:, j, :] = reshape(product_gradient(vec(x[:, j, :])), 2, 2)
            expected_hvp[:, j, :] = reshape(product_hvp(vec(x[:, j, :]), vec(v[:, j, :])), 2, 2)
        end
        f = Reactant.compile(frontend(gradient_partial), (R(x),))
        h = Reactant.compile(frontend(hvp_partial), (R(x), R(v)))
        @test Array(f(R(x))) ≈ expected
        @test Array(h(R(x), R(v))) ≈ expected_hvp
    end
end
end
