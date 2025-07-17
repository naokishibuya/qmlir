# Generate quantum passes TableGen files
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quantum)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Quantum)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Quantum)
add_public_tablegen_target(MLIRQuantumPassesIncGen)

# Generate documentation
add_mlir_doc(Passes QuantumPasses ./ -gen-pass-doc)
