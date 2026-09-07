using Test
using Reactant, ReactantIRFrontend, Enzyme
using Reactant: @allowscalar
using ReactantIRFrontend: FrontendError

Reactant.set_default_backend("cpu")

R(x) = Reactant.to_rarray(x; track_numbers=Number)   # scalars are inputs, not constants

# Bring compiled results back to the host so they compare like Julia values.
host(x::Union{Tuple,NamedTuple}) = map(host, x)
host(x::Reactant.ConcreteRNumber) = x[]
host(x::Reactant.AbstractConcreteArray) = Array(x)
host(x) = x

matches(got, want) = host(got) ≈ want
matches(got::Union{Tuple,NamedTuple}, want) = all(map(matches, got, want))

# Compile `f` once through the frontend and check it against Julia on every
# input; opposite inputs of one case exercise both sides of every branch.
function agrees(f, cases...; kwargs...)
    compiled = Reactant.@compile structured(f)(map(R, first(cases))...; kwargs...)
    return all(cases) do args
        return matches(compiled(map(R, args)...), f(args...; kwargs...))
    end
end

hlo(f, args...) = string(Reactant.@code_hlo structured(f)(map(R, args)...))
function unoptimized(f, args...)
    return string(Reactant.@code_hlo optimize = false structured(f)(map(R, args)...))
end

const x = Float32[1, 2]
const y = Float32[-1, -2]

@testset "ReactantIRFrontend" begin
    @testset "branches" begin
        helper(x) = sum(x) > 0.0f0 ? x + x : x - x
        outer(x) = helper(x)
        @test agrees(outer, (x,), (y,))
        @test occursin("stablehlo.if", hlo(outer, x))

        function bare(x)
            y = x
            if sum(x) > 0.0f0
                y = x + x
            end
            return y
        end
        @test agrees(bare, (x,), (y,))

        function nested(x)
            if sum(x) > 2.0f0
                if sum(x) > 5.0f0
                    x + x
                else
                    x - x
                end
            elseif sum(x) < -2.0f0
                x * 3.0f0
            else
                x * 4.0f0
            end
        end
        @test agrees(nested, [(Float32[a, a],) for a in (-3, -1, 1, 2, 4)]...)

        shortand(x, y) = sum(x) > 0.0f0 && sum(y) > 0.0f0 ? x + y : x - y
        shortor(x, y) = sum(x) > 0.0f0 || sum(y) > 0.0f0 ? x + y : x - y
        pairs = [(Float32[a, a], Float32[b, b]) for a in (-2, 2) for b in (-3, 3)]
        @test agrees(shortand, pairs...)
        @test agrees(shortor, pairs...)

        # Structurization multiplexes the shared `elseif` chain behind an
        # integer discriminator that it tests with `===`.
        function chained(x, y)
            xs, ys = sum(x), sum(y)
            if xs > 0.0f0 && ys > 0.0f0
                z = xs + ys
            elseif xs > 0.0f0
                z = xs - ys
            else
                z = ys - xs
            end
            return z
        end
        @test agrees(chained, pairs...)

        function multiple(x, y)
            a, b = x, y
            if sum(x) > 0.0f0
                a = x + y
                b = x - y
            else
                a = x - y
                b = x + y
            end
            return a + b, a - b
        end
        @test agrees(multiple, (x, y), (y, x))

        named(x) = sum(x) > 0.0f0 ? (; a=x, b=2x) : (; a=2x, b=x)
        @test agrees(named, (x,), (y,))

        function early(x)
            if sum(x) > 0.0f0
                return x .* 2.0f0
            end
            z = x .- 1.0f0
            return z .* 3.0f0
        end
        @test agrees(early, (x,), (y,))

        function nested_early(x)
            if sum(x) > 0.0f0
                if sum(x) > 10.0f0
                    return x .* 10.0f0
                end
                return x .* 2.0f0
            end
            return x .- 1.0f0
        end
        @test agrees(nested_early, (x,), (y,), (Float32[10, 20],))
    end

    @testset "scalar and boolean inputs" begin
        scalar(x) = x > 0.0f0 ? sin(x) + exp(x) : cos(x) - tanh(x)
        @test agrees(scalar, (1.0f0,), (-1.0f0,))

        choose(p, x) = p ? x .+ x : x .- x
        @test agrees(choose, (true, x), (false, x))

        scale(x, n) = n > 2 ? x .* n : x
        @test agrees(scale, (x, 3), (x, 1))

        comparisons(a, b) = (a > b, a < b, a >= b, a <= b, a == b, a != b)
        compiled = Reactant.@compile structured(comparisons)(R(1.0f0), R(1.0f0))
        for (a, b) in ((1.0f0, 1.0f0), (1.0f0, 2.0f0), (2.0f0, 1.0f0))
            @test host(compiled(R(a), R(b))) == comparisons(a, b)
        end
    end

    @testset "loops" begin
        function convergence(x)
            n = 0
            while sum(x) > 1.0f0
                x = x * 0.5f0
                n = n + 1
            end
            return x, n
        end
        @test agrees(
            convergence, (Float32[0.25, 0.25],), (Float32[1, 1],), (Float32[4, 4],)
        )
        @test occursin("stablehlo.while", hlo(convergence, x))

        function counted(x)
            for i in 1:3
                x = x * 0.5f0
            end
            return x
        end
        @test agrees(counted, (x,))

        function indexed(x)
            for i in 1:3
                if i < 3
                    x = x * (i + 1)
                else
                    x = x * i
                end
            end
            return x
        end
        @test agrees(indexed, (x,))

        # The range comes from a leaf call, so `iterate` re-reads its `stop`
        # field inside the loop.
        function over_eachindex(x)
            acc = x
            for i in eachindex(x)
                acc = acc .+ x .* i
            end
            return acc
        end
        @test agrees(over_eachindex, (x,))

        function over_axes(x)
            for i in axes(x, 1)
                x = x .* i
            end
            return x
        end
        @test agrees(over_axes, (x,))

        # Pre-allocation idiom: the initializer is dead, so the carry has no
        # initial value and the first iteration is peeled.
        function preallocated(x)
            v = similar(x)
            for _ in 1:3
                v = x .* 2.0f0
                x = v .+ 1.0f0
            end
            return v
        end
        @test agrees(preallocated, (x,))

        # Host condition and host carries: the loop runs at emission, like Julia.
        function host_loops(x)
            n = 1
            while n < 8
                n *= 2
            end
            s = 0
            for i in 1:3
                s += i
            end
            return x .* (n + s)
        end
        @test agrees(host_loops, (x,))
        @test !occursin("stablehlo.while", hlo(host_loops, x))

        # The accumulator is traced after the first iteration: the rest rolls.
        function host_accumulator(x)
            acc = 0.0f0
            for i in 1:3
                acc += sum(x) * i
            end
            return acc
        end
        @test agrees(host_accumulator, (x,))
        @test occursin("stablehlo.while", unoptimized(host_accumulator, x))

        # General loops: `break`, `continue`, and `return` inside loops, and
        # `for` over a range with a traced endpoint.
        function breaking(x)
            while true
                x = x .* 0.5f0
                sum(x) < 1.0f0 && break
            end
            return x
        end
        @test agrees(breaking, (Float32[8, 8],), (Float32[0.5, 0.5],))
        @test occursin("stablehlo.while", unoptimized(breaking, x))

        function for_break(x)
            for i in 1:10
                x = x .* 0.5f0
                sum(x) < 1.0f0 && break
            end
            return x
        end
        @test agrees(for_break, (Float32[8, 8],), (Float32[0.5, 0.5],))

        function continuing(x)
            for i in 1:4
                i == 2 && continue
                x = x .* i
            end
            return x
        end
        @test agrees(continuing, (x,))

        function returning(x)
            while sum(x) > 1.0f0
                sum(x) < 4.0f0 && return x .+ 100.0f0
                x = x .* 0.5f0
            end
            return x
        end
        @test agrees(
            returning, (Float32[8, 8],), (Float32[0.5, 0.5],), (Float32[1.5, 1.5],)
        )

        function traced_bound(x, n)
            for i in 1:n
                x = x .* 2.0f0
            end
            return x
        end
        @test agrees(traced_bound, (x, 3), (x, 0))

        function traced_start(x, n)
            for i in n:4
                x = x .+ i
            end
            return x
        end
        @test agrees(traced_start, (x, 2), (x, 5))

        function for_in_while(x)
            while sum(x) > 1.0f0
                for i in 1:2
                    x = x .* (0.5f0 * i)
                end
            end
            return x
        end
        @test agrees(for_in_while, (Float32[8, 8],), (Float32[0.5, 0.5],))

        function while_in_for(x)
            for i in 1:2
                while sum(x) > 1.0f0
                    x = x .* 0.5f0
                end
                x = x .* i
            end
            return x
        end
        @test agrees(while_in_for, (Float32[8, 8],))

        function for_in_for(x)
            for i in 1:2
                for j in 1:2
                    x = x .* (i + j)
                end
            end
            return x
        end
        # IRStructurizer up to 0.6.4 fails to structurize a `for` nested directly
        # in a `for` ("SSA values used but not defined"), for plain host code too;
        # fixed upstream (maleadt/IRStructurizer.jl#61) in the release after it.
        if ReactantIRFrontend.STRUCTURIZER_HOISTS
            @test agrees(for_in_for, (x,))
        else
            @test_skip agrees(for_in_for, (x,))
        end

        # A loop rolled inside a general loop's body once inherited that body's
        # exit flag and yielded one value too many. `stamp!` has no value escaping
        # its inner loop, so the released structurizer handles it as well.
        function stamp!(B, n)
            for i in 1:n
                for j in 1:3
                    Reactant.allowscalar(() -> B[i, j] = B[i, j] + i)
                end
            end
            return B
        end
        @test agrees(stamp!, (zeros(Float32, 4, 4), 2))
        function double_until(x)
            while true
                for j in 1:3
                    x = x .* 2
                end
                sum(x) > 100 && break
            end
            return x
        end
        if ReactantIRFrontend.STRUCTURIZER_HOISTS
            @test agrees(double_until, (x,))
        else
            @test_skip agrees(double_until, (x,))
        end

        function loop_with_branch(x)
            while sum(x) > 1.0f0
                if sum(x) > 8.0f0
                    x = x * 0.25f0
                else
                    x = x * 0.5f0
                end
            end
            return x
        end
        @test agrees(loop_with_branch, (Float32[16, 16],), (Float32[2, 2],))

        function scalar_loop(x)
            y = 0.0f0
            while x > 1.0f0
                y = y + x
                x = x * 0.5f0
            end
            return x, y
        end
        @test agrees(scalar_loop, (0.5f0,), (2.0f0,), (8.0f0,))

        # Reactant's own while_convergence test, without `@trace`.
        function regression(x, y)
            diff = x .- y
            while sum(diff) >= 10.0f0
                x = x .- diff ./ 2.0f0
                diff = x .- y
            end
            return diff
        end
        target = Float32[0, -2, -3]
        @test agrees(
            regression,
            (Float32[1, 10, 20], target),
            (target .+ 32, target),
            (target, target),
        )
    end

    @testset "calls" begin
        Base.@noinline noinline_helper(x) = sum(x) > 0.0f0 ? x + x : x - x
        through_noinline(x) = noinline_helper(x)
        @test agrees(through_noinline, (x,), (y,))

        dynamic_helper(x, k::Int) = sum(x) > 0.0f0 ? x .* k : x
        dynamic_helper(x, k::Float32) = sum(x) > 0.0f0 ? x ./ k : x
        dynamic_int(x) = dynamic_helper(x, Base.inferencebarrier(3))
        dynamic_float(x) = dynamic_helper(x, Base.inferencebarrier(2.0f0))
        @test agrees(dynamic_int, (x,), (y,))
        @test agrees(dynamic_float, (x,), (y,))

        varargs(x, ys...) = sum(x) > 0.0f0 ? x + ys[1] : x - ys[end]
        @test agrees(varargs, (x, y, x), (y, y, x))

        keywords(x; scale=2.0f0) = sum(x) > 0.0f0 ? x .* scale : x ./ scale
        @test agrees(keywords, (x,), (y,); scale=3.0f0)

        gain = 2.0f0
        scalar_closure = z -> sum(z) > gain ? z * gain : z / gain
        @test agrees(scalar_closure, (x,), (y,))

        function capturing_counter(x)
            for i in 1:3
                scale = y -> y .* i
                x = scale(x)
            end
            return x
        end
        @test agrees(capturing_counter, (x,))

        weights = R(Float32[10, 20])
        array_closure = z -> sum(z) > 0.0f0 ? z .* weights : z .- weights
        compiled = Reactant.@compile structured(array_closure)(R(x))
        @test host(compiled(R(x))) ≈ x .* Float32[10, 20]
        @test host(compiled(R(y))) ≈ y .- Float32[10, 20]
    end

    @testset "leaves are Reactant's" begin
        # Straight-line code emits exactly what Reactant's frontend emits.
        fused(x) = sum(sin.(x) .+ abs.(x) ./ 2.0f0)
        theirs = split(string(Reactant.@code_hlo fused(R(x))), '\n'; limit=2)[2]
        ours = split(hlo(fused, x), '\n'; limit=2)[2]
        @test ours == theirs
        @test agrees(fused, (x,))

        matmul(a, b) = a * b + a
        m = Float32[1 2; 3 4]
        @test agrees(matmul, (m, m))

        # Dispatch on traced values is unchanged, including where Reactant has no method.
        classify(::Bool) = 10.0f0
        classify(::Integer) = 20.0f0
        bool_sum(x) = classify(sum(x .> 0.0f0))
        @test_throws MethodError Reactant.@jit bool_sum(R(x))
        @test_throws MethodError Reactant.@jit structured(bool_sum)(R(x))

        identity_branch(a, b) = a === b ? 10.0f0 : 20.0f0
        @test agrees(identity_branch, (1, 1), (1, 2))
        @test_throws FrontendError Reactant.@compile structured(identity_branch)(
            R(0.0f0), R(-0.0f0)
        )
    end

    @testset "integration" begin
        step(x) = sum(x) > 0.0f0 ? x .* 2.0f0 : x ./ 2.0f0
        @test host(Reactant.@jit structured(step)(R(x))) ≈ step(x)
        @test structured(step)(x) == step(x)   # outside compilation it is just `step`

        function annotated(x)
            @trace if sum(x) > 0.0f0
                z = x + x
            else
                z = x - x
            end
            return z
        end
        @test agrees(annotated, (x,), (y,))

        function mutating(x)
            if sum(x) > 0.0f0
                x .= x .* 100.0f0
            end
            return x .+ 1.0f0
        end
        @test agrees(mutating, (copy(x),), (copy(y),))

        # Elementwise control flow: Reactant's broadcast calls the program per element.
        signed_step(s) = s > 0.0f0 ? s + 1.0f0 : s - 1.0f0
        elementwise(x) = structured(signed_step).(x)
        @test host(Reactant.@jit elementwise(R(Float32[1, -2, 3]))) ≈ Float32[2, -3, 4]

        # User callbacks handed to leaves are captured without a wrapper.
        mapped(x) = map(s -> s > 0.0f0 ? s + 1.0f0 : s - 1.0f0, x)
        @test agrees(mapped, (Float32[1, -2, 3],))
        counted_positive(x) = sum(s -> s > 0.0f0 ? 1.0f0 : 0.0f0, x)
        @test agrees(counted_positive, (Float32[1, -2, 3],))

        loss(x) = sum(x) > 0.0f0 ? sum(x .* x) : sum(3.0f0 .* x .* x)
        gradient(x) = Enzyme.gradient(Reverse, loss, x)[1]   # `loss` is wrapped at the leaf
        compiled = Reactant.@compile structured(gradient)(R(x))
        @test host(compiled(R(x))) ≈ 2x
        @test host(compiled(R(y))) ≈ 6y

        function loop_loss(x)
            for i in 1:3
                x = x * 0.5f0
            end
            return sum(x)
        end
        loop_gradient(x) = Enzyme.gradient(Reverse, structured(loop_loss), x)[1]
        @test host(Reactant.@jit loop_gradient(R(x))) ≈ fill(0.125f0, 2)
    end

    @testset "entry points" begin
        function bump(x, y; scale=1)
            return sum(x) > 0 ? x .* y .* scale : x .- y
        end
        x, y = Float32[1, -2, 3], Float32[2, 2, 2]
        compiled = ReactantIRFrontend.@compile bump(R(x), R(y))
        @test matches(compiled(R(x), R(y)), bump(x, y))
        @test matches(compiled(R(-x), R(y)), bump(-x, y))
        @test matches((ReactantIRFrontend.@jit sync = true bump(R(x), R(y))), bump(x, y))
        @test matches(
            ReactantIRFrontend.@jit(bump(R(x), R(y); scale=2)), bump(x, y; scale=2)
        )
        ir = string(ReactantIRFrontend.@code_hlo optimize = false bump(R(x), R(y)))
        @test occursin("stablehlo.if", ir)
        half(v) = v > 0 ? v / 2 : v
        @test matches(ReactantIRFrontend.@jit(half.(R(x))), half.(x))
        @test matches(
            ReactantIRFrontend.compile(bump, (R(x), R(y)))(R(x), R(y)), bump(x, y)
        )
        @test occursin(
            "stablehlo.if",
            string(ReactantIRFrontend.code_hlo(bump, (R(x), R(y)); optimize=false)),
        )
        # An explicit import shadows Reactant's exported macros without ambiguity.
        m = Module()
        Core.eval(m, :(using Reactant; using ReactantIRFrontend: @jit))
        Core.eval(m, :(pick(x) = sum(x) > 0 ? x : -x))
        @test matches(Core.eval(m, :(@jit pick($(R(-x))))), -(-x))
    end

    # Cases kept from the review of the prototype in `experimental/IRFrontend`.
    @testset "regressions" begin
        x = Float32[1, -2, 3]

        # Regions must agree on what they produce: shapes, and array versus
        # scalar. Reactant's region builders reject the first two at compile
        # time; the loop check is the frontend's.
        branch_shapes(x, y) = sum(x) > 0.0f0 ? x : y
        @test_throws Exception Reactant.@compile structured(branch_shapes)(
            R(x), R(Float32[1, 2])
        )
        branch_kind(x) = sum(x) > 0.0f0 ? x : sum(x)
        @test_throws Exception Reactant.@compile structured(branch_kind)(R(x))
        function changing_shape(x, b)
            while sum(x) > 1.0f0
                x = x * b
            end
            return x
        end
        err = try
            Reactant.@compile structured(changing_shape)(
                R(ones(Float32, 2, 3)), R(ones(Float32, 3, 2))
            )
        catch e
            e
        end
        @test err isa FrontendError && occursin("shape", err.message)

        # Unsigned comparisons on a counter that is traced once its loop rolls: a
        # signed one is rejected rather than silently compared as signed, whether
        # it was a constant or an input; an unsigned one is compared as unsigned.
        # (`i` must not be assigned elsewhere in this block, or Julia would box it.)
        function unsigned_host(x)
            i = -2
            while i < 0
                x = Core.Intrinsics.ult_int(i, 0) ? x .* 0.25f0 : x .* 0.5f0
                i += 1
            end
            return x
        end
        function unsigned_traced(x, i)
            while i < 0
                x = Core.Intrinsics.ule_int(i, 0) ? x .* 0.25f0 : x .* 0.5f0
                i += 1
            end
            return x
        end
        for (f, args) in ((unsigned_host, (R(x),)), (unsigned_traced, (R(x), R(-2))))
            err = try
                Reactant.@compile structured(f)(args...)
            catch e
                e
            end
            @test err isa FrontendError && occursin("unsigned", err.message)
        end
        function unsigned_counter(x)
            i = UInt32(0)
            while i < UInt32(3)
                x = Core.Intrinsics.ult_int(i, UInt32(2)) ? x .* 2.0f0 : x .* 3.0f0
                i += UInt32(1)
            end
            return x
        end
        @test agrees(unsigned_counter, (x,))

        # Inference sees Reactant's overlay table: an overlaid method is the leaf.
        overlaid(x) = x .+ x
        Base.Experimental.@overlay Reactant.REACTANT_METHOD_TABLE overlaid(
            x::Reactant.TracedRArray
        ) = x .* 2.0f0
        ir = unoptimized(overlaid, x)
        @test occursin("stablehlo.multiply", ir) && !occursin("stablehlo.add", ir)

        # A short-circuit right-hand side is emitted inside the region its guard
        # opens, not ahead of it: in `main`, the division (a batched scalar helper
        # in unoptimized IR) sits deeper than the `if` and before the first line
        # back at the `if`'s depth.
        guarded(x) = (s=sum(x); s > 0.0f0 && sum(x ./ s) > 0.5f0) ? x : -x
        @test agrees(guarded, (x,), (-x,), (Float32[0.1, 0.1, 0.1],))
        lines = split(unoptimized(guarded, x), '\n')
        indent(l) = length(l) - length(lstrip(l))
        main = findfirst(l -> occursin("func.func @main", l), lines)
        if_line = findnext(l -> occursin("stablehlo.if", l), lines, main)
        divides(l) = occursin("stablehlo.divide", l) || occursin("/_broadcast_scalar", l)
        div_line = findnext(divides, lines, main)
        @test if_line !== nothing && div_line !== nothing && if_line < div_line
        depth = indent(lines[if_line])
        region_end = findnext(
            l -> !isempty(strip(l)) && indent(l) <= depth, lines, if_line + 1
        )
        @test indent(lines[div_line]) > depth && div_line < region_end
    end

    @testset "redefinition" begin
        @eval redefined(x) = x .+ 1.0f0
        @eval uses_redefined(x) = sum(x) > 0.0f0 ? redefined(x) : x
        before = Reactant.@compile structured(uses_redefined)(R(x))
        @eval redefined(x) = x .+ 2.0f0
        after = Base.invokelatest(() -> Reactant.@compile structured(uses_redefined)(R(x)))
        @test host(before(R(x))) ≈ x .+ 1.0f0
        @test host(after(R(x))) ≈ x .+ 2.0f0
    end

    @testset "diagnostics" begin
        function assigned_inside(x)
            local z
            while sum(x) > 1.0f0
                z = x .* 0.5f0
                x = z
            end
            return z
        end
        @test_throws FrontendError Reactant.@compile structured(assigned_inside)(R(x))

        recursive(x) = sum(x) > 0.0f0 ? recursive(x .* 0.5f0) : x
        @test_throws FrontendError Reactant.@compile structured(recursive)(R(x))

        checked(x) = sum(x) > 0.0f0 ? x : error("negative")
        @test_throws FrontendError Reactant.@compile structured(checked)(R(x))

        function scalar_block(x)
            if sum(x) > 0.0f0
                @allowscalar x[1] = 1.0f0
            end
            return x
        end
        err = try
            Reactant.@compile structured(scalar_block)(R(x))
        catch e
            e
        end
        @test err isa FrontendError && occursin("try", err.message)

        function traced_index(x)
            t = (1.0f0, 2.0f0)
            for i in 1:2
                x = x .* t[i]
            end
            return x
        end
        @test_throws Exception Reactant.@compile structured(traced_index)(R(x))

        err = try
            Reactant.@compile structured(checked)(R(x))
        catch e
            e
        end
        @test occursin("checked", sprint(showerror, err))
    end
end
