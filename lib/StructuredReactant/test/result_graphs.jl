# Standalone CPU gate for mutable object graphs returned by an executable.
module ResultGraphAcceptance
using Test, Reactant, StructuredReactant
Reactant.set_default_backend("cpu")
frontend(f)=get(ENV,"FRONTEND","structured")=="vanilla" ? f : structured(f)
R(x)=Reactant.to_rarray(x;track_numbers=Number)
cached(f,args...)=Reactant.ReactantCore.within_compile() ? Reactant.Ops.call(frontend(f),args...) : f(args...)
function dict_first(x)
    k=Ref{Any}();d=Dict(k=>x);k[]=d;return d,k
end
function key_first(x)
    k=Ref{Any}();d=Dict(k=>x);k[]=d;return k,d
end
cached_dict_first(x)=cached(dict_first,x)
cached_key_first(x)=cached(key_first,x)
function self_ref(x)
    r=Ref{Any}();r[]=r;return r,x
end
mutable struct Partial
    a::Any
    b::Any
    c::Any
    Partial()=new()
end
function partial_fields(x)
    r=Partial();r.b=x;return r,r
end
mutable struct Fixed
    const value::Any
    link::Any
    Fixed(x)=new(x)
end
function const_cycle(x)
    r=Fixed(x);r.link=r;return r,r
end

@testset "Executable result graphs: $(get(ENV,"FRONTEND","structured"))" begin
    for f in (dict_first,key_first,cached_dict_first,cached_key_first,self_ref,partial_fields,const_cycle)
        @testset "$f" begin
        x=[.1,.3,.7,1.1];executable=Reactant.compile(frontend(f),(R(copy(x)),))
        a,b=executable(R(copy(x)));c,d=executable(R(-2x));flags=Dict{String,Bool}()
        if f in (dict_first,key_first,cached_dict_first,cached_key_first)
            if f in (key_first,cached_key_first);a,b=b,a;c,d=d,c;end
            flags["key_alias_1"]=first(keys(a))===b
            flags["key_alias_2"]=first(keys(c))===d
            flags["cycle_1"]=b[]===a
            flags["cycle_2"]=d[]===c
            flags["fresh_dict"]=a!==c
            flags["fresh_key"]=b!==d
            flags["value_1"]=Array(first(values(a)))==x
            flags["value_2"]=Array(first(values(c)))==-2x
        elseif f===self_ref
            flags["self_1"]=a[]===a;flags["self_2"]=c[]===c;flags["fresh"]=a!==c
            flags["value_1"]=Array(b)==x;flags["value_2"]=Array(d)==-2x
        elseif f===partial_fields
            flags["aliases"]=a===b && c===d;flags["fresh"]=a!==c
            flags["defined_1"]=!isdefined(a,:a)&&isdefined(a,:b)&&!isdefined(a,:c)
            flags["defined_2"]=!isdefined(c,:a)&&isdefined(c,:b)&&!isdefined(c,:c)
            flags["value_1"]=isdefined(a,:b)&&Array(a.b)==x
            flags["value_2"]=isdefined(c,:b)&&Array(c.b)==-2x
        else
            flags["aliases"]=a===b && c===d;flags["fresh"]=a!==c
            flags["cycle_1"]=a.link===a;flags["cycle_2"]=c.link===c
            flags["value_1"]=Array(a.value)==x;flags["value_2"]=Array(c.value)==-2x
        end
            for (label, flag) in flags
                @testset "$label" begin
                    @test flag
                end
            end
        end
    end
end
end # module
