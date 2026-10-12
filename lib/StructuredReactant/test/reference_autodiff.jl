module ReferenceAutodiffAcceptance
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x) = Reactant.to_rarray(x; track_numbers=Number)
cubic_loss(x) = sum(x .^ 3)
function mutation_loss(x)
    x .*= 2
    return sum(x .^ 2)
end
function selected_gradient(x, y, flag)
    selected = flag ? x : y
    return Enzyme.gradient(Reverse, cubic_loss, selected)[1]
end
function selected_view_gradient(x, flag)
    selected = flag ? view(x, 1:2) : view(x, 2:3)
    return Enzyme.gradient(Reverse, cubic_loss, selected)[1]
end
function selected_mutation_gradient(x, y, flag)
    selected = flag ? x : y
    derivative = Enzyme.gradient(Reverse, mutation_loss, selected)[1]
    return derivative, x, y
end
function vjp_cubic(x, lambdas)
    result = similar(lambdas)
    for i in 1:size(lambdas, 2)
        lambda = lambdas[:, i]
        loss(x) = sum(x .^ 3 .* lambda)
        result[:, i] .= Enzyme.gradient(Reverse, Const(loss), x)[1]
    end
    return result
end
function nested_derivative(x, direction, lambdas)
    body(x) = vjp_cubic(x, lambdas)
    return Enzyme.autodiff(Forward, Const(body), Duplicated(x, direction))[1]
end
@testset "reference-backed autodiff" begin
    x = Float64[0.7, 1.2, -0.4]
    y = Float64[-1.1, 0.5, 1.8]
    @testset "nested forward over reverse" begin
        direction = Float64[-0.2, 0.3, 0.6]
        lambdas = Float64[1.2 -0.1; 0.3 0.8; -0.5 1.7]
        f = Reactant.compile(structured(nested_derivative), (R(x), R(direction), R(lambdas)))
        @test Array(f(R(x), R(direction), R(lambdas))) ≈ 6 .* x .* direction .* lambdas
    end
    @testset "selected arrays" begin
        f = Reactant.compile(structured(selected_gradient), (R(x), R(y), R(false)))
        for flag in (false, true)
            @test Array(f(R(x), R(y), R(flag))) ≈ 3 .* (flag ? x : y) .^ 2
        end
    end
    @testset "selected views" begin
        f = Reactant.compile(structured(selected_view_gradient), (R(x), R(false)))
        for flag in (false, true)
            selected = flag ? x[1:2] : x[2:3]
            @test Array(f(R(x), R(flag))) ≈ 3 .* selected .^ 2
        end
    end
    @testset "selected mutation" begin
        f = Reactant.compile(structured(selected_mutation_gradient), (R(x), R(y), R(false)))
        for flag in (false, true)
            dx, ax, ay = f(R(x), R(y), R(flag))
            @test Array(dx) ≈ 8 .* (flag ? x : y)
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
        end
    end
end
end
