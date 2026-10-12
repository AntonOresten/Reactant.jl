# Standalone comparison against the pinned corpus from Reactant.jl#1828.
# See comparison.md for provenance, adaptations, and commands.
using Reactant, LinearAlgebra, SHA, TOML

length(ARGS) == 2 || error("Usage: polybench.jl SOURCE RESULTS.toml")
source_path, output = abspath.(ARGS)
source = read(source_path, String)
bytes2hex(sha256(source)) ==
"25b3999e9e9ab8d593f34204e4f597d04c53f23d7936f138a356579ea14d1485" ||
    error("The corpus differs from the pinned source; review it first")
# Keep the functions verbatim; replace the original harness with fresh references.
kernels = first(split(source, "\n@testset \"kernel_correlation\""; limit=2)) * "\nend\n"
include_string(@__MODULE__, kernels, source_path)
# These three assertions deliberately require traced values. Only the host copy
# drops them; the frontend compiles the original function bodies.
host_kernels = replace(
    kernels,
    "module Polybench" => "module PolybenchHost",
    "::Reactant.TracedRArray{Float64,1}" => "",
    "::Reactant.TracedRArray{Float64,2}" => "",
)
include_string(@__MODULE__, host_kernels, source_path * ".host")
mode = get(ENV, "FRONTEND", "structured")
mode in ("structured", "vanilla") || error("Unknown frontend: $mode")
if mode == "structured"
    using StructuredReactant
end
Reactant.set_default_backend("cpu")
Reactant.allowscalar(true)

function inputs(name, n, phase)
    m() = [Float64((i == j ? 3 : 0) + sin(i + 2j + phase)/10) for i in 1:n, j in 1:n]
    v() = [Float64(i + phase/2)/n for i in 1:n]
    z() = zeros(n)
    if name === :kernel_correlation
        # Exercise both the small-variance and ordinary-variance branches.
        return (phase == 0 ? m() : ones(n, n),)
    elseif name in (:kernel_covariance, :kernel_lu, :kernel_floyd_warshall)
        return (m(),)
    elseif name === :kernel_cholesky
        A=m()
        return (A*A' + Matrix{Float64}(I, n, n),)
    elseif name in (:kernel_gemm, :kernel_symm, :kernel_syr2k)
        return (1.5, 0.25, m(), m())
    elseif name === :kernel_gemmver
        return (1.5, 0.25, v(), v(), v(), v(), m(), v(), v(), v())
    elseif name === :kernel_gesummv
        return (1.5, 0.25, m(), m(), v())
    elseif name === :kernel_syrk
        return (1.5, 0.25, m())
    elseif name === :kernel_trmm
        return (1.5, m(), m())
    elseif name === :kernel_2mm
        return (1.5, 0.25, m(), m(), m(), m())
    elseif name === :kernel_3mm
        return (m(), m(), m(), m())
    elseif name === :kernel_atax
        return (m(), v())
    elseif name === :kernel_bicg
        return (m(), v(), v())
    elseif name === :kernel_doitgen
        return (reshape(Float64.(1:(n ^ 3)), n, n, n)/n^3, m())
    elseif name === :kernel_mvt
        return (v(), v(), v(), v(), m())
    elseif name === :kernel_durbin
        return (v() ./ 10, z())
    elseif name === :kernel_gramschidt
        return (m(), zeros(n, n), zeros(n, n))
    elseif name === :kernel_ludcmp
        return (m(), v(), z(), z())
    elseif name === :kernel_trisolv
        return (m(), z(), v())
    elseif name === :kernel_adi
        return (2, m(), m(), m(), m())
    elseif name === :kernel_fdtd_2d
        return (m(), m(), m(), v())
    elseif name === :kernel_heat_3d
        return (
            2,
            reshape(Float64.(1:(n ^ 3)) .+ phase, n, n, n)/n^3,
            fill(1.0 + phase, n, n, n),
        )
    elseif name === :kernel_jacobi_1d
        return (2, v(), z())
    elseif name === :kernel_jacobi_2d
        return (2, m(), zeros(n, n))
    elseif name === :kernel_seidel_2d
        return (2, m())
    elseif name === :kernel_deriche
        return (0.1, m())
    elseif name === :kernel_nussinov
        return (Float64.(mod.((1:n) .+ phase, 4)), zeros(n, n))
    end
    return error("Missing input fixture for $name")
end
R(x::AbstractArray) = Reactant.to_rarray(x)
R(x) = x
host(x::Reactant.AbstractConcreteArray) = Array(x)
host(x::Tuple) = map(host, x)
host(x) = x
same(a::Tuple, b::Tuple) = length(a)==length(b) && all(map(same, a, b))
same(a, b) = isapprox(a, b; rtol=1e-7, atol=1e-9)
finite(x::Tuple) = all(finite, x)
finite(x::AbstractArray) = all(isfinite, x)
finite(x) = isfinite(x)
# Observe mutation as well as returned results; do not reuse donated inputs.
observe(f) = (args...) -> (f(args...), args)
results = Dict{String,Any}[]
for name in sort(filter(n->startswith(string(n), "kernel_"), names(Polybench; all=true)))
    string(name) in split(get(ENV, "SKIP", ""), ',') && continue
    haskey(ENV, "CASE") && string(name)!=ENV["CASE"] && continue
    println("START ", mode, " ", name)
    flush(stdout)
    f = getfield(Polybench, name)
    reference = observe(getfield(PolybenchHost, name))
    wrapped = observe(f)
    row = Dict{String,Any}(
        "case"=>string(name), "frontend"=>mode, "runs"=>Dict{String,Any}[]
    )
    try
        for n in (4, 5)
            args=inputs(name, n, 0)
            row["stage"]="host"
            expected=reference(deepcopy(args)...)
            finite(expected) || error("Nonfinite host reference")
            row["stage"]="compile"
            target = mode=="structured" ? StructuredReactant.structured(wrapped) : wrapped
            if get(ENV, "OPTIMIZE_FALSE", "0")=="1"
                compileargs=map(R, deepcopy(args))
                ir=string(Reactant.@code_hlo optimize=false target(compileargs...))
                write(joinpath(dirname(output), string(name)*"-"*string(n)*".mlir"), ir)
                row["while_count_" * string(n)]=count("stablehlo.while", ir)
                row["if_count_" * string(n)]=count("stablehlo.if", ir)
                exec=Reactant.compile(target, map(R, deepcopy(args)); optimize=false)
            else
                exec=Reactant.compile(target, map(R, deepcopy(args)))
            end
            row["stage"]="execute"
            for phase in (0, 1)
                fresh=inputs(name, n, phase)
                want=reference(deepcopy(fresh)...)
                finite(want) || error("Nonfinite host reference for phase $phase")
                got=host(exec(map(R, deepcopy(fresh))...))
                pass=same(got, want)
                run=Dict{String,Any}("n"=>n, "phase"=>phase, "pass"=>pass)
                if !pass
                    run["actual"]=repr(got)
                    run["expected"]=repr(want)
                end
                push!(row["runs"], run)
            end
        end
        row["status"]=all(r["pass"] for r in row["runs"]) ? "pass" : "mismatch"
    catch err
        row["status"]=row["stage"]=="host" ? "invalid_reference" : "error"
        row["error"]=sprint(showerror, err, catch_backtrace())
    end
    println(
        "RESULT ",
        row["case"],
        " ",
        row["status"],
        " ",
        (x->first(split(x, '\n')))(get(row, "error", "")),
    )
    flush(stdout)
    push!(results, row)
    open(output, "w") do io
        return TOML.print(
            io,
            Dict(
                "frontend"=>mode,
                "julia"=>string(VERSION),
                "structurizer"=>(
                    if mode=="structured"
                        string(pkgversion(StructuredReactant.IRStructurizer))
                    else
                        "unused"
                    end
                ),
                "results"=>results,
            ),
        )
    end
end

isempty(results) && error("No kernels selected; check CASE and SKIP")
exit(all(r["status"] == "pass" for r in results) ? 0 : 1)
