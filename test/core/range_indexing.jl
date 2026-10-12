using Reactant, Enzyme, Test

select_range(r, selector) = r[selector]
function shifted_range_loss(start)
    r = Reactant.TracedUnitRange(start, start + 3.0, 4)
    return sum(r[[1, 3, 4]] .^ 2)
end
shifted_range_gradient(start) = Enzyme.gradient(Reverse, shifted_range_loss, start)[1]

@testset "Traced unit range selectors" begin
    for T in (Int32, Int64), selector in (Base.OneTo(4), 2:3, 1:2:3, [4, 2], reshape([4, 2], 1, 2))
        @testset "$T $(typeof(selector))" begin
            input = Reactant.to_rarray(T(2):T(5); track_numbers=true)
            compiled = Reactant.compile(select_range, (input, selector))
            for start in (-3, 2, 7)
                host = T(start):T(start + 3)
                actual = compiled(Reactant.to_rarray(host; track_numbers=true), selector)
                @test Array(actual) == host[selector]
                @test size(actual) == size(host[selector])
            end
        end
    end
end
@testset "Differentiating selected range values" begin
    R(x) = Reactant.to_rarray(x; track_numbers=true)
    primal = Reactant.compile(shifted_range_loss, (R(1.0),))
    reverse = Reactant.compile(shifted_range_gradient, (R(1.0),))
    for start in (-2.0, 0.5, 3.0)
        values = start .+ [0.0, 2.0, 3.0]
        @test Float64(primal(R(start))) ≈ sum(abs2, values)
        @test Float64(reverse(R(start))) ≈ 2sum(values)
    end
end
