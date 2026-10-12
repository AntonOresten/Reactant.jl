module ReferenceShadowMatrix
using Test, Reactant, StructuredReactant, Enzyme
Reactant.set_default_backend("cpu")
R(x)=Reactant.to_rarray(x;track_numbers=Number)
cubic(x)=sum(x .^ 3)
function mutation_loss(x)
    x .*= 2
    return sum(x .^ 2)
end
function overlapping_forward(x,flag,d)
    selected=flag ? view(x,1:2) : view(x,2:3)
    return Enzyme.autodiff(Forward,Const(cubic),Duplicated(selected,flag ? view(d,1:2) : view(d,2:3)))[1]
end
function overlapping_reverse(x,flag,d)
    selected=flag ? view(x,1:2) : view(x,2:3)
    Enzyme.autodiff(Reverse,Const(mutation_loss),Active,Duplicated(selected,flag ? view(d,1:2) : view(d,2:3)))
    return d,x
end
function selected_batch(x,y,flag,d1,d2)
    selected=flag ? x : y
    return Enzyme.autodiff(Forward,Const(cubic),BatchDuplicated(selected,(d1,d2)))[1]
end
function selected_stacked(x,y,flag,d)
    selected=flag ? x : y
    return Enzyme.autodiff(Forward,Const(cubic),Reactant.StackedBatchDuplicated(selected,d))[1]
end
function selected_batch_reverse(x,y,flag,d1,d2)
    selected=flag ? x : y
    Enzyme.autodiff(Reverse,Const(mutation_loss),Active,BatchDuplicated(selected,(d1,d2)))
    return d1,d2,x,y
end
function selected_stacked_reverse(x,y,flag,d)
    selected=flag ? x : y
    Enzyme.autodiff(Reverse,Const(mutation_loss),Active,Reactant.StackedBatchDuplicated(selected,d))
    return d,x,y
end
@testset "reference shadow matrix" begin
    x=Float64[0.7,1.2,-0.4];y=Float64[-1.1,0.5,1.8]
    d1=Float64[-0.2,0.3,0.6];d2=Float64[0.7,-0.5,0.2]
    @testset "overlapping selected views: forward" begin
        f=Reactant.compile(structured(overlapping_forward),(R(x),R(false),R(d1)))
        for flag in (false,true)
            a=flag ? x[1:2] : x[2:3]
            @test Float64(f(R(x),R(flag),R(d1))) ≈ sum(3 .* a.^2 .* (flag ? d1[1:2] : d1[2:3]))
        end
    end
    @testset "overlapping selected views: reverse mutation" begin
        f=Reactant.compile(structured(overlapping_reverse),(R(x),R(false),R(d1)))
        for flag in (false,true)
            indices=flag ? (1:2) : (2:3);a=x[indices];expected=copy(x);expected[indices].*=2
            derivative,actual=f(R(x),R(flag),R(d1))
            expected_d=copy(d1);expected_d[indices]=8 .* a .+ 2 .* d1[indices]
            @test Array(derivative) ≈ expected_d
            @test Array(actual) ≈ expected
        end
    end
    @testset "batched selected directions" begin
        f=Reactant.compile(structured(selected_batch),(R(x),R(y),R(false),R(d1),R(d2)))
        for flag in (false,true)
            a=flag ? x : y;derivatives=f(R(x),R(y),R(flag),R(d1),R(d2))
            @test Float64(derivatives[1]) ≈ sum(3 .* a.^2 .* d1)
            @test Float64(derivatives[2]) ≈ sum(3 .* a.^2 .* d2)
        end
    end
    @testset "stacked selected directions" begin
        d=hcat(d1,d2);f=Reactant.compile(structured(selected_stacked),(R(x),R(y),R(false),R(d)))
        for flag in (false,true)
            a=flag ? x : y;derivatives=f(R(x),R(y),R(flag),R(d))
            @test Float64(derivatives[1]) ≈ sum(3 .* a.^2 .* d1)
            @test Float64(derivatives[2]) ≈ sum(3 .* a.^2 .* d2)
        end
    end
    @testset "stacked reverse mutation seeds" begin
        d=hcat(d1,d2)
        f=Reactant.compile(structured(selected_stacked_reverse),(R(x),R(y),R(false),R(d)))
        for flag in (false,true)
            a=flag ? x : y;derivatives,ax,ay=f(R(x),R(y),R(flag),R(d))
            @test Array(derivatives) ≈ hcat(8 .* a .+ 2 .* d1,8 .* a .+ 2 .* d2)
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
        end
    end
    @testset "batched reverse mutation seeds" begin
        f=Reactant.compile(structured(selected_batch_reverse),(R(x),R(y),R(false),R(d1),R(d2)))
        for flag in (false,true)
            a=flag ? x : y;dx1,dx2,ax,ay=f(R(x),R(y),R(flag),R(d1),R(d2))
            @test Array(dx1) ≈ 8 .* a .+ 2 .* d1
            @test Array(dx2) ≈ 8 .* a .+ 2 .* d2
            @test Array(ax) ≈ (flag ? 2 .* x : x)
            @test Array(ay) ≈ (flag ? y : 2 .* y)
        end
    end
end
end
