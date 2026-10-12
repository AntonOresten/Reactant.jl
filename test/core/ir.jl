using Reactant: MLIR

@testset "inject" begin
    MLIR.IR.@with_context Reactant.ReactantContext() begin
        mod = MLIR.IR.Module()

        MLIR.IR.@with_module mod begin
            MLIR.IR.inject!(
                "MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr"
            )

            MLIR.IR.inject!(
                "MPI_Comm_rank", "llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32"
            )

            MLIR.IR.inject!(
                "wrapper_function",
                """
                llvm.func @wrapper_function(%rank_ptr : !llvm.ptr) -> () {
                    %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
                    %errcode = llvm.call @MPI_Comm_rank(%comm, %rank_ptr) : (!llvm.ptr, !llvm.ptr) -> (i32)
                    llvm.return
                }
                """,
            )
        end

        mod_op = MLIR.IR.Operation(mod)
        @test MLIR.API.mlirOperationVerify(mod_op)

        MLIR.IR.dispose(mod)
    end
end

@testset "enable_ir_printing!" begin
    MLIR.IR.@with_context Reactant.ReactantContext() begin
        mod = parse(
            MLIR.IR.Module, "func.func @identity(%x: i32) -> i32 { return %x : i32 }"
        )
        pm = MLIR.IR.PassManager()
        MLIR.IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "canonicalize")
        output = mktemp() do _, io
            redirect_stderr(io) do
                @test MLIR.IR.enable_ir_printing!(pm; before_all=true) === pm
                MLIR.IR.run!(pm, MLIR.IR.Operation(mod))
            end
            seekstart(io)
            return read(io, String)
        end
        @test occursin("IR Dump Before", output)
        @test occursin("@identity", output)
        MLIR.IR.dispose(pm)
        MLIR.IR.dispose(mod)
    end
end
