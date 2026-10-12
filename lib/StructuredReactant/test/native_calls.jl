module NativeCalls
using Reactant, Enzyme, Statistics, Random

# A package's keyword method is native, but a user callback passed to it must
# re-enter structured tracing. Neither the function nor its callback is a
# special tensor operation known to the frontend.
native(x; callback=identity, scale=2.0f0) = callback(x) .* scale
Base.Experimental.@overlay Reactant.REACTANT_METHOD_TABLE function native(
    x; callback=identity, scale=2.0f0
)
    return Reactant.call_with_reactant(callback, x) .* scale
end
branch(x) = sum(x) > 0 ? x .* 3 : x ./ 3
plain(x) = native(x; scale=4.0f0)
callback(x) = native(x; callback=branch, scale=4.0f0)
loss(x) = sum(callback(x))
gradient(f, x) = Enzyme.gradient(Reverse, f, x)[1]

# The overlay has a different result representation from the ordinary method.
# Both a nested leaf and a Program used as a callback must infer that overlay.
as_result(x) = (x,)
Base.Experimental.@overlay Reactant.REACTANT_METHOD_TABLE as_result(
    x::Reactant.TracedRArray
) = x .* 2
pass_call(f, x) = f(x)

# Dispatch on a stdlib callback must survive crossing a native boundary even
# when Julia loads that stdlib from a package depot.
native_mean(f, x) = f(x)
Base.Experimental.@overlay Reactant.REACTANT_METHOD_TABLE native_mean(::typeof(mean), x) =
    x .* 2
keep_mean(x) = native_mean(mean, x)
with_bias(bias) = x -> as_result(x) .+ bias

# An explicit native call must select the ordinary method and keep that result
# representation through downstream dispatch, including keyword calls.
escape_result(x; scale=2) = (x, scale)
Base.Experimental.@overlay Reactant.REACTANT_METHOD_TABLE escape_result(x; scale=2) =
    x .* scale
choose_native(x::Tuple) = first(x) .+ 1
choose_native(x::Reactant.TracedRArray) = x .* 99
native_escape(x) = choose_native(Reactant.call_with_native(escape_result, x))
function native_keyword_escape(x)
    return choose_native(Reactant.call_with_native(escape_result, x; scale=3))
end

native_predicate(x) = x > 0.0f0
native_branch(x) = Reactant.call_with_native(native_predicate, x) ? x * 3 : x / 3

function seeded_random!(x)
    rng = MersenneTwister(0)
    Random.rand!(rng, x)
    return x
end

function normalize_shape!(x)
    for j in axes(x, 2), i in axes(x, 1)
        x[i, j] /= sqrt(size(x, 1))
    end
    return x
end
invalid_host_shape(x) = sum(x) + sqrt(-size(x, 1))
end

@testset "native keyword calls and callbacks" begin
    for body in (NativeCalls.plain, NativeCalls.callback)
        input = Float32[1, 2]
        compiled = Reactant.compile(structured(body), (R(input),))
        for x in (input, -input)
            @test Array(compiled(R(x))) ≈ body(x)
        end
    end
end

@testset "differentiate a native keyword callback" begin
    input = Float32[1, 2]
    f = structured(NativeCalls.loss)
    compiled = Reactant.compile(NativeCalls.gradient, (f, R(input)))
    for x in (input, -input)
        expected = fill(sum(x) > 0 ? 12.0f0 : 4.0f0 / 3.0f0, size(x))
        @test Array(compiled(f, R(x))) ≈ expected
    end
end

@testset "overlay result types at native boundaries" begin
    input = Float32[1, 2]
    f = structured(NativeCalls.as_result)
    compiled = Reactant.compile(NativeCalls.pass_call, (f, R(input)))
    biased = Reactant.compile(structured(NativeCalls.with_bias(input)), (R(input),))
    for x in (input, -input)
        @test Array(compiled(f, R(x))) == 2 .* x
        @test Array(biased(R(x))) == 2 .* x .+ input
        @test f(x) == (x,)
    end
end

@testset "stdlib callbacks retain native dispatch" begin
    input = Float32[1, 2]
    compiled = Reactant.compile(structured(NativeCalls.keep_mean), (R(input),))
    for x in (input, -input)
        @test Array(compiled(R(x))) == 2 .* x
    end
end

@testset "explicit native calls retain ordinary dispatch" begin
    input = Float32[1, 2]
    for f in (NativeCalls.native_escape, NativeCalls.native_keyword_escape)
        compiled = Reactant.compile(structured(f), (R(input),))
        for x in (input, -input)
            @test Array(compiled(R(x))) == x .+ 1
        end
    end
end

@testset "native predicates retain traced branches" begin
    compiled = Reactant.compile(
        structured(NativeCalls.native_branch), (Reactant.ConcreteRNumber(1.0f0),)
    )
    for x in (1.0f0, -1.0f0)
        @test Float32(compiled(Reactant.ConcreteRNumber(x))) ≈ (x > 0 ? x * 3 : x / 3)
    end
end

@testset "locally constructed RNG" begin
    # MersenneTwister construction reaches a host generator reduction during
    # inference. Losing the native escape repeatedly wraps that reduction's op.
    input = zeros(Float32, 2, 3)
    vanilla = Reactant.compile(NativeCalls.seeded_random!, (R(input),))
    compiled = Reactant.compile(structured(NativeCalls.seeded_random!), (R(input),))
    expected = Array(vanilla(R(input)))
    for _ in 1:2
        @test Array(compiled(R(input))) == expected
    end
end

@testset "host error paths stay within native helpers" begin
    Reactant.allowscalar() do
        input = reshape(Float64.(1:16), 4, 4)
        target = structured(NativeCalls.normalize_shape!)
        # The host sqrt's domain check must not become a traced loop-exit tag.
        ir = string(Reactant.@code_hlo optimize=false target(R(input)))
        @test occursin("stablehlo.while", ir)
        compiled = Reactant.compile(target, (R(input),))
        for x in (input, input .+ 2)
            @test Array(compiled(R(x))) ≈ x / 2
        end
        for f in (NativeCalls.invalid_host_shape, structured(NativeCalls.invalid_host_shape))
            @test_throws DomainError Reactant.compile(f, (R(input),))
        end
    end
end
