# Quantum dialect library
add_mlir_dialect_library(MLIRQuantumDialect
  QuantumDialect.cpp
  QuantumOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/backend/include/mlir/Dialect/Quantum
  ${PROJECT_BINARY_DIR}/backend/include/mlir/Dialect/Quantum

  DEPENDS
  MLIRQuantumOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
)

# Quantum passes library
add_mlir_dialect_library(MLIRQuantumPasses
  Passes/QuantumCancelSelfInversePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/backend/include/mlir/Dialect/Quantum

  DEPENDS
  MLIRQuantumPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRQuantumDialect
  MLIRTransforms
)
