# Standalone acceptance for path-dependent array sharing. This deliberately fails
# until reference identity is carried through branches and loops; it is not part
# of runtests.jl yet. Both numerical results and input mutation must match Julia.
module ConditionalAliasAcceptance

using Reactant, StructuredReactant, Enzyme, Test

Reactant.set_default_backend("cpu")
if haskey(ENV, "EXPECTED_REACTANT_ROOT")
    @assert realpath(pkgdir(Reactant)) == realpath(ENV["EXPECTED_REACTANT_ROOT"])
end
R(x) = Reactant.to_rarray(x; track_numbers=Number)

function direct(x,c)
    y=c ? copy(x) : x
    x .*= 10
    return sum(y)
end
function tuple_result(x,c)
    y=c ? (copy(x),) : (x,)
    x .*= 10
    return sum(y[1])
end
Base.@noinline choose(x,c)=c ? copy(x) : x
function across_call(x,c)
    y=choose(x,c)
    x .*= 10
    return sum(y)
end
function view_result(x,c)
    y=c ? copy(x) : view(x,:)
    x .*= 10
    return sum(y)
end
function inside_loop(x,c)
    y=copy(x)
    for i in 1:3
        y=c ? copy(x) : x
        x .*= 2
    end
    return sum(y)
end

function mutate_selected_copy(x,c)
    y=c ? copy(x) : x
    y .*= 2
    return sum(x) + 3sum(y)
end
function overlapping_views(x,c)
    y=c ? view(x,1:1) : view(x,2:2)
    y .*= 2
    return sum(x)
end
function fresh_siblings(x,c)
    y,z = if c
        a=copy(x)
        (a,a)
    else
        (copy(x),copy(x))
    end
    y .*= 2
    return sum(z)
end
function nested_write(x,c)
    y=c ? copy(x) : x
    if sum(x)>0
        y .*= 2
    end
    return sum(x) + 3sum(y)
end
function chained_choices(x,c)
    y=c ? copy(x) : x
    z=c ? x : y
    z .*= 3
    return sum(x) + 3sum(y)
end
function view_of_selected(x,c)
    y=c ? copy(x) : x
    v=view(y,2:2)
    v .*= 2
    return sum(x) + 3sum(y)
end
function runtime_loop_alias(x,c)
    y=x
    for i in 1:ifelse(c,1,0)
        y=copy(x)
    end
    x .*= 2
    return sum(y)
end


function nested_alias_result(x,c)
    y=if c
        x
    else
        sum(x)>0 ? x : copy(x)
    end
    x .*= 2
    return sum(y)
end
function read_inside_nested_branch(x,c)
    y=c ? copy(x) : x
    if sum(x)>0
        y .*= 2
        return sum(x) + 3sum(y)
    end
    return sum(x) + 3sum(y)
end
function alternating_loop_alias(x,c)
    y=copy(x)
    for i in 1:ifelse(c,3,4)
        y=isodd(i) ? x : copy(x)
        x .*= 2
    end
    x .*= 2
    return sum(y)
end
function fresh_view_siblings(x,c)
    y,z=if c
        a=copy(x)
        (view(a,1:2),view(a,2:3))
    else
        (copy(x[1:2]),copy(x[2:3]))
    end
    y .*= 2
    return sum(z)
end


function for_break(x, c)
    for i in 1:10
        x = x .* 0.5
        sum(x) < 1 && break
    end
    return sum(x)
end

function chosen_identity(x, c)
    y = c ? x : copy(x)
    return y === x
end

function loop_identity(x, n)
    y = copy(x)
    for i in 1:n
        y = isodd(i) ? x : copy(x)
    end
    return y === x
end

gradient(f, x, c) = Enzyme.gradient(Reverse, f, x, Enzyme.Const(c))[1]

function finite_difference(f, x, c)
    epsilon = 1e-5
    return [
        (f(x .+ (eachindex(x) .== i) .* epsilon, c) -
         f(x .- (eachindex(x) .== i) .* epsilon, c)) / (2epsilon)
        for i in eachindex(x)
    ]
end

const cases = (
    direct, tuple_result, across_call, view_result, inside_loop,
    mutate_selected_copy, overlapping_views, fresh_siblings, nested_write,
    chained_choices, view_of_selected, runtime_loop_alias, nested_alias_result,
    read_inside_nested_branch, alternating_loop_alias, fresh_view_siblings, for_break,
)

@testset "Conditional array sharing acceptance" begin
    for f in cases
        @testset "$f" begin
            program = structured(f)
            for mode in (:primal, :reverse)
                @testset "$mode" begin
                    args = (R([1.0, 2.0, 3.0, 4.0]), R(false))
                    compiled = nothing
                    error_text = nothing
                    try
                        compiled = mode === :primal ?
                            Reactant.compile(program, args) :
                            Reactant.compile(gradient, (program, args...))
                    catch err
                        # Test's raw display of an exception may otherwise dereference
                        # an MLIR handle whose context has already been destroyed.
                        error_text = sprint(showerror, err)
                    end
                    @test error_text === nothing
                    error_text === nothing || continue
                    for sign in (1.0, -1.0), condition in (true, false)
                        x = sign .* [1.0, 2.0, 3.0, 4.0]
                        xr = R(copy(x))
                        if mode === :primal
                            host = copy(x)
                            expected = f(host, condition)
                            actual = compiled(xr, R(condition))[]
                            @test isapprox(actual, expected; rtol=1e-7, atol=1e-9)
                            @test isapprox(Array(xr), host; rtol=1e-7, atol=1e-9)
                        else
                            expected = finite_difference(f, x, condition)
                            actual = Array(compiled(program, xr, R(condition)))
                            @test isapprox(actual, expected; rtol=1e-6, atol=1e-7)
                        end
                    end
                end
            end
        end
    end
end

# Correct mutation alone does not establish observable Julia identity.
@testset "Observable array identity" begin
    for (f, values) in ((chosen_identity, (false, true)), (loop_identity, (0, 1, 2, 3)))
        @testset "$f" begin
            x = [1.0, 2.0]
            compiled = nothing
            error_type = nothing
            try
                compiled = Reactant.compile(structured(f), (R(x), R(first(values))))
            catch err
                error_type = string(typeof(err))
            end
            @test error_type === nothing
            error_type === nothing || continue
            for value in values
                @test Bool(compiled(R(x), R(value))) == f(copy(x), value)
            end
        end
    end
end

end # module
