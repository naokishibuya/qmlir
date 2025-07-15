# MLIR Quantum Dialect
set(LLVM_TARGET_DEFINITIONS QuantumOps.td)
add_mlir_dialect(QuantumOps quantum)
add_mlir_doc(QuantumOps QuantumOps Dialects/ -gen-dialect-doc)
