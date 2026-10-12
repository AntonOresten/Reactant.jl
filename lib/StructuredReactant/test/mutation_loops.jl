module MutationLoops
using Reactant, Enzyme
function stamp!(x)
    for i in 1:length(x)
        Reactant.@allowscalar x[i] = x[i]^2 + i
    end
    return x
end
function rows(x)
    y = similar(x)
    for i in 1:size(x, 1)
        y[i, :] = (x[i, :] .+ 1) .^ 2
    end
    return y
end
function host_table(x)
    table = zeros(Float32, length(x))
    for i in 1:length(table)
        table[i] = cos(Float32(i))
    end
    return x .* table
end
function host_ref!(x)
    count = Ref(0)
    for i in 1:length(x)
        count[] += i
        Reactant.@allowscalar x[i] = x[i] * count[]
    end
    return x, count[]
end
# The call keeps the Ref observable even on Julia versions that scalar-replace
# the local Ref in host_ref!. Its mutation must still execute on the host.
Base.@noinline function add_count!(count, i)
    count[] += i
    return count[]
end
function host_ref_calls!(x)
    count = Ref(0)
    for i in 1:length(x)
        c = add_count!(count, i)
        Reactant.@allowscalar x[i] = x[i] * c
    end
    return x, count[]
end
function while_stamp!(x)
    i = 1
    while i <= length(x)
        Reactant.@allowscalar x[i] = x[i]^2 + i
        i += 1
    end
    return x
end
function break_stamp!(x)
    i = 1
    while true
        i > length(x) && break
        Reactant.@allowscalar x[i] = x[i]^2 + i
        i += 1
    end
    return x
end
loss(x) = sum(stamp!(x))
gradient(f, x) = Enzyme.gradient(Reverse, f, x)[1]
end

@testset "mutation-only loops" begin
    for n in (0, 1, 4, 32)
        input = Float32.(1:n) ./ max(1, n)
        for body in
            (MutationLoops.stamp!, MutationLoops.while_stamp!, MutationLoops.break_stamp!)
            ir = unoptimized(body, input)
            if n == 0 || n > 1 || body !== MutationLoops.break_stamp!
                @test occursin("stablehlo.while", ir) == (n > 1)
            end
            @test Array(Reactant.@jit structured(body)(R(copy(input)))) ≈ body(copy(input))
        end
    end
    for n in (1, 4, 1024)
        input = reshape(Float32.(1:(4n)), n, 4) ./ n
        @test occursin("stablehlo.while", unoptimized(MutationLoops.rows, input)) == (n > 1)
        @test Array(Reactant.@jit structured(MutationLoops.rows)(R(input))) ≈
            MutationLoops.rows(input)
    end
    for n in (1, 4, 32)
        input = Float32.(1:n) ./ n
        @test !occursin("stablehlo.while", unoptimized(MutationLoops.host_table, input))
        @test Array(Reactant.@jit structured(MutationLoops.host_table)(R(input))) ≈
            MutationLoops.host_table(input)
    end
    for n in (0, 1, 4)
        input = Float32.(1:n) ./ max(1, n)
        for body in (MutationLoops.host_ref!, MutationLoops.host_ref_calls!)
            if body === MutationLoops.host_ref_calls!
                @test !occursin("stablehlo.while", unoptimized(body, input))
            end
            value, count = Reactant.@jit structured(body)(R(copy(input)))
            expected, expected_count = body(copy(input))
            @test Array(value) ≈ expected
            @test count == expected_count
        end
    end
    for n in (1, 4)
        input = Float32.(1:n) ./ n
        f = structured(MutationLoops.loss)
        compiled = Reactant.compile(MutationLoops.gradient, (f, R(input)))
        @test Array(compiled(f, R(copy(input)))) ≈ 2 .* input
    end
end
