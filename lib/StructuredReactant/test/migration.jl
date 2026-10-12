# Run an upstream Reactant test file unchanged in an isolated process.
# MIGRATION_MODE=vanilla: original frontend and annotations.
# MIGRATION_MODE=structured: structured frontend, preserving annotations.
# MIGRATION_MODE=automatic (default): also remove bare control-flow @trace.
# MIGRATION_REPORT optionally writes a machine-readable TOML result.
#     julia --project=... lib/StructuredReactant/test/migration.jl core/autodiff
using Reactant, Test, TOML, SHA

const MODE = get(ENV, "MIGRATION_MODE", "automatic")
MODE in ("vanilla", "structured", "automatic") || error("unknown migration mode: $MODE")
if MODE != "vanilla"
    using StructuredReactant
end
Reactant.set_default_backend("cpu")

const PACKAGE_PATHS = Dict("Reactant" => realpath(pkgdir(Reactant)))
if MODE != "vanilla"
    PACKAGE_PATHS["StructuredReactant"] = realpath(pkgdir(StructuredReactant))
end
for (name, path) in PACKAGE_PATHS
    expected = get(ENV, "MIGRATION_EXPECT_" * uppercase(name), "")
    isempty(expected) || path == realpath(expected) ||
        error("loaded $name from $path, expected $expected")
end

const compile_calls = Ref(0)

# Use the frontend callback boundary, including stdlibs and package extensions.
function route(f)
    f isa Base.Broadcast.BroadcastFunction &&
        return Base.Broadcast.BroadcastFunction(route(f.f))
    return StructuredReactant.structure_callback(f)
end

# The overrides are more specific than Reactant's methods (`args::Tuple`), so
# `invoke` on the general signature reaches the originals in the current world.
for entry in (MODE == "vanilla" ? () : (:compile, :code_hlo, :code_mhlo, :code_xla))
    @eval function Reactant.Compiler.$entry(ctx, f, args::Tuple; kwargs...)
        compile_calls[] += 1
        if get(ENV, "MIGRATION_PROGRESS", "false") == "true"
            println(
                "compile ", compile_calls[], ": ", Test.get_testset().description, " / ", f
            )
            flush(stdout)
        end
        routed_args = map(route, args)
        result = Base.invoke(
            Reactant.Compiler.$entry,
            Tuple{Any,Any,Any},
            ctx,
            route(f),
            routed_args;
            kwargs...,
        )
        # A compiled function's signature holds the routed function-valued
        # arguments, so calls made with the bare functions are routed the same way.
        return if $(entry === :compile) && routed_args !== args
            (a...) -> result(map(route, a)...)
        else
            result
        end
    end
end

# A bare `@trace` on control flow leaves its body to the frontend. Preserve the
# macro's statement result (`nothing`), including for a terminal `if`: a plain
# Julia `if` would instead return its selected branch's value. Options and call
# outlining forms are hints the frontend does not replace, and are kept.
const TRACE_WORLD = Base.get_world_counter()   # the original macro lives here
if MODE == "automatic"
    @eval Reactant.ReactantCore macro trace(args...)
        if length(args) == 1 && Meta.isexpr(only(args), (:if, :for, :while))
            return quote
                $(esc(only(args)))
                nothing
            end
        end
        return Base.invoke_in_world(
            $TRACE_WORLD, var"@trace", __source__, __module__, args...
        )
    end
end

file = isempty(ARGS) ? "core/control_flow" : only(ARGS)
occursin('/', file) || (file = "core/" * file)
endswith(file, ".jl") && (file = chop(file; tail=3))
test_root = normpath(joinpath(@__DIR__, "..", "..", "..", "test"))
source = normpath(joinpath(test_root, file * ".jl"))
startswith(source, test_root * "/") || error("test file must be inside test/")
isfile(source) || error("test file does not exist: $source")

# Julia 1.13 keeps the active testset in a scoped value instead of a stack.
function within(f, ts::Test.AbstractTestSet)
    @static if isdefined(Test, :push_testset)
        Test.push_testset(ts)
        try
            return f()
        finally
            Test.pop_testset()
        end
    else
        return Base.ScopedValues.with(
            f, Test.CURRENT_TESTSET => ts, Test.TESTSET_DEPTH => 1
        )
    end
end

suite = Test.DefaultTestSet("Reactant $(file).jl ($MODE)"; verbose=false)
load_error = Ref("")
within(suite) do
    try
        selected = get(ENV, "MIGRATION_TESTSET", "")
        function select_testset(expr)
            if !isempty(selected) &&
                Meta.isexpr(expr, :macrocall) &&
                expr.args[1] === Symbol("@testset") &&
                length(expr.args) >= 3 &&
                expr.args[3] isa String &&
                !occursin(selected, expr.args[3])
                return nothing
            end
            return expr
        end
        upstream = Module(:Upstream)
        Core.eval(upstream, :(using Test, Reactant)) # Match the upstream worker setup.
        Base.include(select_testset, upstream, source)
    catch err
        load_error[] = sprint(showerror, err, catch_backtrace())
        println("The file itself failed to load: ", load_error[])
    end
end

function classify(result)
    result isa Test.Fail && return "mismatch"
    message = sprint(show, result)
    m = match(r"StructuredReactant: ([^\n]*)", message)
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

if !isempty(load_error[])
    counts["load_error"] = 1
end
report = Dict(
    "mode" => MODE,
    "package_paths" => PACKAGE_PATHS,
    "source" => source,
    "source_sha256" => bytes2hex(sha256(read(source))),
    "julia" => string(VERSION),
    "compile_calls" => compile_calls[],
    "testset_filter" => get(ENV, "MIGRATION_TESTSET", ""),
    "counts" => counts,
    "examples" => examples,
    "load_error" => load_error[],
)
if haskey(ENV, "MIGRATION_REPORT")
    open(ENV["MIGRATION_REPORT"], "w") do io
        return TOML.print(io, report; sorted=true)
    end
end
exit(any(kind != "pass" && kind != "broken" && n > 0 for (kind, n) in counts) ? 1 : 0)
