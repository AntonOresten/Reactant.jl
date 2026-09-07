# Migration experiment: how much of one of Reactant's own test files passes
# when `@trace` is an identity macro and every compilation goes through
# `structured`? Not part of the test suite. Run in its own process, naming the
# file under `test/core` (`control_flow` by default, `autodiff` is the other
# one that exercises control flow):
#
#     julia --project=lib/ReactantIRFrontend/test lib/ReactantIRFrontend/test/migration.jl [autodiff]
#
# The upstream file and its assertions are included unchanged. Failures are
# classified as unsupported (a `FrontendError`), mismatch (an assertion that
# compiled but disagreed), or other.
using Reactant, ReactantIRFrontend, Test
using ReactantIRFrontend: FrontendError

const compile_calls = Ref(0)

# Functions given to Reactant are routed through the frontend, including
# callbacks such as the function differentiated by `Enzyme.gradient`. Base and
# Reactant functions passed as arguments are left alone.
function route(f)
    f isa Base.Broadcast.BroadcastFunction &&
        return Base.Broadcast.BroadcastFunction(route(f.f))
    f isa Function || return f
    root = Base.moduleroot(parentmodule(typeof(f)))
    root in (Base, Core, Reactant) && return f
    return structured(f)
end

# The overrides are more specific than Reactant's methods (`args::Tuple`), so
# `invoke` on the general signature reaches the originals in the current world.
for entry in (:compile, :code_hlo, :code_mhlo, :code_xla)
    @eval function Reactant.Compiler.$entry(ctx, f, args::Tuple; kwargs...)
        compile_calls[] += 1
        result = Base.invoke(
            Reactant.Compiler.$entry,
            Tuple{Any,Any,Any},
            ctx,
            route(f),
            map(route, args);
            kwargs...,
        )
        # A compiled function's signature holds the routed function-valued
        # arguments, so calls made with the bare functions are routed the same way.
        return $(entry === :compile) ? (a...) -> result(map(route, a)...) : result
    end
end

@eval Reactant.ReactantCore macro trace(args...)
    return esc(last(args))
end

file = isempty(ARGS) ? "control_flow" : only(ARGS)
source = normpath(joinpath(@__DIR__, "..", "..", "..", "test", "core", file * ".jl"))
suite = Test.DefaultTestSet("Reactant $(file).jl through structured"; verbose=false)
Test.push_testset(suite)
try
    Base.include(Module(:Upstream), source)
catch err
    println("The file itself failed to load: ", sprint(showerror, err))
finally
    Test.pop_testset()
end
suite.time_end = time()

function classify(result)
    result isa Test.Fail && return "mismatch"
    message = sprint(show, result)
    m = match(r"ReactantIRFrontend: ([^\n]*)", message)
    m !== nothing && return "unsupported: " * m.captures[1]
    m = match(
        r"(?:Test threw exception|Got exception outside of a @test)\n\s*([^\n]*)", message
    )
    return "error: " * (m === nothing ? "?" : m.captures[1])
end

function collect!(counts, examples, ts, path)
    for r in ts.results
        if r isa Test.AbstractTestSet
            collect!(counts, examples, r, path * " / " * r.description)
        elseif r isa Test.Pass
            counts["pass"] = get(counts, "pass", 0) + 1
        elseif r isa Test.Broken
            counts["broken"] = get(counts, "broken", 0) + 1
        else
            kind = classify(r)
            counts[kind] = get(counts, kind, 0) + 1
            push!(get!(Vector{String}, examples, kind), path)
        end
    end
end

counts = Dict{String,Int}()
examples = Dict{String,Vector{String}}()
collect!(counts, examples, suite, "")
counts["pass"] = Test.get_test_counts(suite).cumulative_passes   # passes are not stored

println("\nCompilation requests routed through `structured`: ", compile_calls[])
println("Results:")
for (kind, n) in sort!(collect(counts); by=last, rev=true)
    println(lpad(n, 5), "  ", kind)
    kind in ("pass", "broken") && continue
    for testset in unique(examples[kind])[1:min(end, 4)]
        println("       e.g. ", strip(testset, [' ', '/']))
    end
end
