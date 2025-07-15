add_mlir_dialect_library(MLIRQuantum
  QuantumDialect.cpp
  QuantumOps.cpp
  Passes/QuantumCancelXPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Quantum

  DEPENDS
  MLIRQuantumOpsIncGen
  MLIRQuantumPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTransforms
)
