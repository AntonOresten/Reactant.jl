# The first `resolve` of a session JIT-compiles the compiler and IRStructurizer
# code the interpreter reaches (over ten seconds); put it in the package image.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    branch(x, y) = sum(x) > 0 ? x .* y : x .- y
    function halve(x)
        n = 0
        while sum(x) > 1
            x = x ./ 2
            n += 1
        end
        return x, n
    end
    function count_up(x, n)
        for i in 1:n
            x = x .+ i
        end
        return x
    end
    chain(x) = sum((x .* 2) .- 1)
    A = TracedRArray{Float32,1}
    @compile_workload begin
        world = Base.get_world_counter()
        resolve(Tuple{typeof(branch),A,A}, world)
        resolve(Tuple{typeof(halve),A}, world)
        resolve(Tuple{typeof(count_up),A,Int}, world)
        resolve(Tuple{typeof(count_up),A,TracedRNumber{Int}}, world)
        resolve(Tuple{typeof(chain),A}, world)
        if Reactant.Reactant_jll.is_available()
            let x = Reactant.to_rarray(Float32[1, 2]), n = Reactant.ConcreteRNumber(3)
                Reactant.@code_hlo optimize = false structured(branch)(x, x)
                Reactant.@code_hlo optimize = false structured(halve)(x)
                Reactant.@code_hlo optimize = false structured(count_up)(x, 3)
                Reactant.@code_hlo optimize = false structured(count_up)(x, n)
                Reactant.@code_hlo optimize = false structured(chain)(x)
            end
        end
    end
    # Dispatch lookups contain this process's world numbers. CodeInstances and
    # their prepared results are serialized with Julia's dependency tracking.
    empty!(RESOLUTIONS)
end
