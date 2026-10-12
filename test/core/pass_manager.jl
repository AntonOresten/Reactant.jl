using Reactant, Test

@testset "Pass manager IR printing" begin
    IR = Reactant.MLIR.IR
    ctx = IR.Context(Reactant.registry[])
    IR.activate(ctx)
    try
        mod = parse(IR.Module, "module { func.func @identity(%x: i32) -> i32 { return %x : i32 } }")
        pm = IR.PassManager()
        try
            IR.add_pipeline!(IR.OpPassManager(pm), "canonicalize")
            mktemp() do _, io
                redirect_stderr(io) do
                    @test IR.enable_ir_printing!(pm; before_all=true, after_all=true) === pm
                    IR.run!(pm, IR.Operation(mod))
                end
                seekstart(io)
                output = read(io, String)
                @test occursin("IR Dump Before", output)
                @test occursin("IR Dump After", output)
                @test occursin("@identity", output)
            end
        finally
            IR.dispose(pm)
            IR.dispose(mod)
        end
    finally
        IR.deactivate(ctx)
        IR.dispose(ctx)
    end
end
