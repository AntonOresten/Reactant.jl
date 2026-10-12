using Reactant, Enzyme, Test

function independent_constants(x, reload)
    initial = reshape([1.0, 2.0], size(x))
    a = Reactant.Ops.constant(initial)
    if reload
        a .+= x
        b = Reactant.Ops.constant(initial)
    else
        b = Reactant.Ops.constant(initial)
        a .+= x
    end
    return a, b
end

function independent_constants_loss(x, reload)
    a, b = independent_constants(x, reload)
    return sum(a) + 3sum(b)
end

function independent_constants_gradient(x, reload)
    return Enzyme.gradient(Reverse, independent_constants_loss, x, Enzyme.Const(reload))[1]
end

@testset "Independent mutable constant results" begin
    for shape in ((2,), (1, 2), (2, 1)), reload in (false, true)
        @testset "shape=$shape reload=$reload" begin
            initial = reshape([1.0, 2.0], shape)
            input = Reactant.to_rarray(reshape([3.0, 4.0], shape))
            primal = Reactant.compile(independent_constants, (input, reload))
            reverse = Reactant.compile(independent_constants_gradient, (input, reload))
            for values in ([3.0, 4.0], [-1.0, 0.5])
                host = reshape(values, shape)
                a, b = primal(Reactant.to_rarray(host), reload)
                @test Array(a) ≈ initial + host
                @test Array(b) ≈ initial
                @test Array(reverse(Reactant.to_rarray(host), reload)) ≈ ones(shape)
            end
        end
    end
end
