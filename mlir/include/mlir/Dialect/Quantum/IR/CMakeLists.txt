# Generate quantum dialect TableGen files
set(LLVM_TARGET_DEFINITIONS QuantumOps.td)
mlir_tablegen(QuantumOps.h.inc -gen-op-decls)
mlir_tablegen(QuantumOps.cpp.inc -gen-op-defs)
mlir_tablegen(QuantumOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QuantumOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRQuantumOpsIncGen)
